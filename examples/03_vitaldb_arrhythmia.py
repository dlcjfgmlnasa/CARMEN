# -*- coding:utf-8 -*-
"""03 — Real data: 4-class rhythm classification on VitalDB.

    pip install vitaldb scikit-learn
    python examples/03_vitaldb_arrhythmia.py [--model carmen-base] [--cases-per-class 40]

Downloads open data only:
  - rhythm labels from the VitalDB Arrhythmia Database (PhysioNet, CC BY 4.0) — beat-level
    rhythm annotations reviewed by anesthesiologists;
  - the matching intraoperative ECG lead II from VitalDB Open (``vitaldb.load_case``).

Each 30 s window whose beats all share one rhythm becomes a sample, with the paper's four
classes: NSR, AF (AFIB/AFL), VA (VT, patterned ventricular ectopy) and BradyCond (AVB,
SND). Supraventricular ectopy, noise and unclassifiable beats are dropped. CARMEN stays
frozen; a logistic regression on its pooled features is trained on half the patients and
tested on the other half (split by ``subjectid``, so no patient is on both sides).

VitalDB comes from the same hospital as part of the pretraining corpus, and this demo does
not remove patients that may also be in it — the paper's evaluation does. It is a small
demo, not the paper protocol (all 482 cases, patient-level 5-fold CV); numbers move with
--cases-per-class and --seed.
"""
import argparse
import csv
import http.client
import io
import random
import time
import urllib.request
from collections import Counter
from concurrent.futures import ThreadPoolExecutor

import _common  # noqa: F401 — puts the repo root on sys.path

import numpy as np
import torch
from scipy.signal import butter, resample_poly, sosfiltfilt
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from carmen import DownstreamModelWrapper, make_batch

ANNOT_URL = "https://physionet.org/files/vitaldb-arrhythmia/1.0.0/"
CLASSES = ["NSR", "AF", "VA", "BradyCond"]
LABELS = {
    "N": 0,
    "AFIB/AFL": 1,
    "VT": 2,
    "Patterned Ventricular Ectopy": 2,
    "AVB": 3,
    "SND": 3,
}  # any other rhythm label drops the window
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


def label_windows(beats: list[dict], max_per_class: int) -> list[tuple[float, int]]:
    """Non-overlapping 30 s windows whose beats all map to one class, no bad quality.

    Up to ``max_per_class`` windows per class, so a long normal stretch does not crowd
    out a short run of a rare rhythm in the same case.
    """
    t = np.array([float(b["time_second"]) for b in beats])
    out, count = [], Counter()
    start = t[0]
    while start + WIN_SEC <= t[-1]:
        win = beats[np.searchsorted(t, start) : np.searchsorted(t, start + WIN_SEC)]
        ys = {LABELS.get(b["rhythm_label"].strip()) for b in win}
        if (
            len(win) >= 10  # low enough to keep bradycardia
            and len(ys) == 1
            and None not in ys
            and not any(b["bad_signal_quality"].strip().lower() == "true" for b in win)
        ):
            y = ys.pop()
            if count[y] < max_per_class:
                out.append((start, y))
                count[y] += 1
        start += WIN_SEC
    return out


def load_case(case: dict, max_per_class: int) -> list[tuple[str, np.ndarray, int]]:
    """One case → [(subjectid, 30 s ECG @ 100 Hz in mV, label), ...]."""
    import vitaldb

    beats = fetch_csv(f"{ANNOT_URL}Annotation_Files/Annotation_file_{case['case_id']}.csv")
    windows = label_windows(beats, max_per_class)
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


def load_case_retry(case: dict, max_per_class: int, tries: int = 3) -> list:
    """``load_case`` with retries; a case that keeps failing is skipped, not fatal."""
    for i in range(tries):
        try:
            return load_case(case, max_per_class)
        except Exception as e:  # network timeouts from PhysioNet / VitalDB
            err = e
            time.sleep(5 * (i + 1))
    print(f"  skipped case {case['case_id']}: {err!r}")
    return []


def features(wrapper: DownstreamModelWrapper, data: list) -> np.ndarray:
    """Frozen, mean-pooled CARMEN features, one 30 s window at a time."""
    out = []
    with torch.no_grad():
        for _, x, _ in data:
            batch = make_batch([("ecg", torch.from_numpy(x))], patch_size=wrapper.patch_size)
            out.append(wrapper.extract_features(batch).float().cpu().numpy()[0])
    return np.stack(out)


def evaluate(X: np.ndarray, y: np.ndarray, test: np.ndarray) -> tuple[float, list[float]]:
    """Macro and per-class one-vs-rest AUROC of a logistic regression."""
    clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=3000, class_weight="balanced"))
    clf.fit(X[~test], y[~test])
    p = clf.predict_proba(X[test])
    per = [roc_auc_score(y[test] == k, p[:, k]) for k in range(len(CLASSES))]
    return float(np.mean(per)), per


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="carmen-base", help="model name or path to .pt")
    ap.add_argument("--cases-per-class", type=int, default=40,
                    help="cases sampled per rhythm class (cases can repeat across classes)")
    ap.add_argument("--max-windows", type=int, default=20, help="per class per case")
    ap.add_argument("--workers", type=int, default=4, help="parallel downloads")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    rng = random.Random(args.seed)

    # 1) Pick cases so every class is represented: for each class, cases that contain it.
    meta = fetch_csv(ANNOT_URL + "metadata.csv")
    picked = {}
    for k in range(len(CLASSES)):
        pool = [c for c in meta
                if any(LABELS.get(r.strip()) == k for r in c["rhythm_classes"].split(","))]
        for c in rng.sample(pool, min(args.cases_per_class, len(pool))):
            picked[c["case_id"]] = c
    cases = list(picked.values())
    print(f"downloading {len(cases)} cases (labels: PhysioNet, ECG: VitalDB Open) ...")
    with ThreadPoolExecutor(args.workers) as pool:
        data = [w for ws in pool.map(lambda c: load_case_retry(c, args.max_windows), cases) for w in ws]
    subjects = np.array([s for s, _, _ in data])
    y = np.array([lab for _, _, lab in data])
    print(f"{len(data)} windows from {len(set(subjects))} patients — "
          + ", ".join(f"{n} {(y == k).sum()}" for k, n in enumerate(CLASSES)))

    # 2) Patient-level split; redraw until every class is on both sides.
    pats = sorted(set(subjects))
    for _ in range(100):
        rng.shuffle(pats)
        test = np.isin(subjects, pats[: len(pats) // 2])
        if all(((y == k) & test).any() and ((y == k) & ~test).any() for k in range(len(CLASSES))):
            break
    print(f"train {(~test).sum()} windows / test {test.sum()} windows "
          f"({len(pats) // 2} held-out patients; test "
          + ", ".join(f"{n} {((y == k) & test).sum()}" for k, n in enumerate(CLASSES)) + ")")

    # 3) Frozen CARMEN features → logistic regression.
    device = "cuda" if torch.cuda.is_available() else "cpu"
    wrapper = DownstreamModelWrapper(args.model, device=device)
    macro, per = evaluate(features(wrapper, data), y, test)
    print(f"\n{'test AUROC':<22}{'macro':>7}" + "".join(f"{n:>11}" for n in CLASSES))
    print(f"{args.model:<22}{macro:>7.3f}" + "".join(f"{a:>11.3f}" for a in per))


if __name__ == "__main__":
    main()
