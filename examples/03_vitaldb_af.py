# -*- coding:utf-8 -*-
"""03 — Real data: atrial fibrillation detection on VitalDB.

    pip install vitaldb scikit-learn
    python examples/03_vitaldb_af.py [--model carmen-base] [--n-cases 100] [--random-init]

Downloads open data only:
  - rhythm labels from the VitalDB Arrhythmia Database (PhysioNet, CC BY 4.0) — beat-level
    rhythm annotations reviewed by anesthesiologists;
  - the matching intraoperative ECG lead II from VitalDB Open (``vitaldb.load_case``).

Each 30 s window whose beats are all one rhythm becomes a sample: AF (``AFIB/AFL``) = 1,
normal sinus rhythm (``N``) = 0. CARMEN stays frozen; a logistic regression on its pooled
features is trained on half the patients and tested on the other half (split by
``subjectid``, so no patient appears on both sides). VitalDB comes from the same
hospital as part of the pretraining corpus, and this demo does not remove patients that
may also be in it — the paper's evaluation does.

This is a small demo, not the paper protocol (which uses all 482 cases, 4 rhythm
classes and patient-level 5-fold CV); the AUROC depends on --n-cases and --seed.
"""
import argparse
import csv
import http.client
import io
import random
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor

import _common  # noqa: F401 — puts the repo root on sys.path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from carmen import CARMEN, DownstreamModelWrapper, make_batch

ANNOT_URL = "https://physionet.org/files/vitaldb-arrhythmia/1.0.0/"
LABELS = {"N": 0, "AFIB/AFL": 1}
SR_RAW, SR = 500, 100
WIN_SEC = 30


def fetch_csv(url: str, tries: int = 5) -> list[dict]:
    for i in range(tries):  # PhysioNet reads time out now and then
        try:
            with urllib.request.urlopen(url, timeout=60) as r:
                return list(csv.DictReader(io.StringIO(r.read().decode("utf-8"))))
        except (OSError, http.client.HTTPException):
            if i == tries - 1:
                raise
            time.sleep(5 * (i + 1))


def label_windows(beats: list[dict], max_windows: int) -> list[tuple[float, int]]:
    """Non-overlapping 30 s windows whose beats share one label (N or AF), no bad quality."""
    t = np.array([float(b["time_second"]) for b in beats])
    out = []
    start = t[0]
    while start + WIN_SEC <= t[-1] and len(out) < max_windows:
        win = beats[np.searchsorted(t, start) : np.searchsorted(t, start + WIN_SEC)]
        labels = {b["rhythm_label"].strip() for b in win}
        if (
            len(win) >= 10
            and len(labels) == 1
            and next(iter(labels)) in LABELS
            and not any(b["bad_signal_quality"].strip().lower() == "true" for b in win)
        ):
            out.append((start, LABELS[next(iter(labels))]))
        start += WIN_SEC
    return out


def load_case(case: dict, max_windows: int) -> list[tuple[str, np.ndarray, int]]:
    """One case → [(subjectid, 30 s ECG @ 100 Hz in mV, label), ...]."""
    import vitaldb

    beats = fetch_csv(f"{ANNOT_URL}Annotation_Files/Annotation_file_{case['case_id']}.csv")
    windows = label_windows(beats, max_windows)
    if not windows:
        return []
    ecg = vitaldb.load_case(int(case["case_id"]), ["SNUADC/ECG_II"], 1 / SR_RAW)[:, 0]
    # Same band as pretraining: 0.5–40 Hz, then 500 → 100 Hz.
    sos = butter(4, [0.5, 40.0], btype="band", fs=SR_RAW, output="sos")
    out = []
    for start, y in windows:
        s0 = int(start * SR_RAW)
        seg = ecg[s0 : s0 + WIN_SEC * SR_RAW]
        if len(seg) < WIN_SEC * SR_RAW or np.isnan(seg).any():
            continue
        x = resample_poly(sosfiltfilt(sos, seg), 1, SR_RAW // SR).astype(np.float32)
        out.append((case["subjectid"], x, y))
    return out


def load_case_retry(case: dict, max_windows: int, tries: int = 3) -> list:
    """``load_case`` with retries; a case that keeps failing is skipped, not fatal."""
    for i in range(tries):
        try:
            return load_case(case, max_windows)
        except Exception as e:  # network timeouts from PhysioNet / VitalDB
            err = e
            time.sleep(5 * (i + 1))
    print(f"  skipped case {case['case_id']}: {err!r}")
    return []


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="carmen-base", help="model name or path to .pt")
    ap.add_argument("--n-cases", type=int, default=100, help="half with AF, half without")
    ap.add_argument("--max-windows", type=int, default=20, help="per case")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--random-init", action="store_true",
                    help="same architecture, untrained weights (control for pretraining)")
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) Pick cases: half that contain AF, half that do not.
    meta = fetch_csv(ANNOT_URL + "metadata.csv")
    has_af = [c for c in meta if "AFIB/AFL" in c["rhythm_classes"]]
    no_af = [c for c in meta if "AFIB/AFL" not in c["rhythm_classes"]]
    k = args.n_cases // 2
    cases = rng.sample(has_af, min(k, len(has_af))) + rng.sample(no_af, min(k, len(no_af)))
    print(f"downloading {len(cases)} cases (labels: PhysioNet, ECG: VitalDB Open) ...")
    with ThreadPoolExecutor(args.workers) as pool:
        data = [w for ws in pool.map(lambda c: load_case_retry(c, args.max_windows), cases) for w in ws]
    subjects = np.array([s for s, _, _ in data])
    y = np.array([lab for _, _, lab in data])
    print(f"{len(data)} windows from {len(set(subjects))} patients — AF {y.sum()}, NSR {(y == 0).sum()}")

    # 2) Frozen CARMEN features, one 30 s window at a time.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DownstreamModelWrapper(args.model, device=device)
    if args.random_init:
        torch.manual_seed(args.seed)
        model = CARMEN.from_config(wrapper.config)
        model.rope_pi = wrapper.model.rope_pi
        wrapper.model = model.to(device).eval()
    feats = []
    with torch.no_grad():
        for i, (_, x, _) in enumerate(data):
            batch = make_batch([("ecg", torch.from_numpy(x))], patch_size=wrapper.patch_size)
            feats.append(wrapper.extract_features(batch).float().cpu().numpy()[0])
            if (i + 1) % 200 == 0:
                print(f"  features {i + 1}/{len(data)}")
    X = np.stack(feats)

    # 3) Patient-level split, logistic regression on the frozen features.
    pats = sorted(set(subjects))
    rng.shuffle(pats)
    test_pats = set(pats[: len(pats) // 2])
    test = np.array([s in test_pats for s in subjects])
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=2000, class_weight="balanced"))
    clf.fit(X[~test], y[~test])
    auroc = roc_auc_score(y[test], clf.predict_proba(X[test])[:, 1])
    print(
        f"train {(~test).sum()} windows / test {test.sum()} windows "
        f"({len(test_pats)} held-out patients, AF {y[test].sum()})"
    )
    print(f"AF vs NSR — test AUROC {auroc:.3f}" + (" (random init)" if args.random_init else ""))


if __name__ == "__main__":
    main()
