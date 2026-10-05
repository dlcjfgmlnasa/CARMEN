<div align="center">

# CARMEN

### A Cardiorespiratory Foundation Model for Continuous Physiological Waveforms

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Modalities](https://img.shields.io/badge/modalities-10-teal.svg)](#-overview)
[![Params](https://img.shields.io/badge/params-~174M-8A2BE2.svg)](#-overview)

[**Model Weights**](https://github.com/dlcjfgmlnasa/CARMEN/releases) ·
[**Quickstart Notebook**](examples/quickstart.ipynb) ·
[**Examples**](examples) ·
[**Inference API**](#-inference-api)

</div>

## 📰 News

- **2026.10** — CARMEN has been accepted to the **AI4Health Workshop at NeurIPS 2026**. 🎉

## 📖 Overview

**CARMEN** is a foundation model for the continuous waveforms recorded at the bedside
and in the operating room. A **single Transformer encoder (~174M parameters)** is
pretrained across **10 signal modalities** and serves as a frozen **feature
extractor** for downstream clinical tasks — detection, prediction, outcome,
estimation and phenotyping — with a light head on top.

Two design choices carry most of the weight. Every modality is tokenized the same
way — raw patches, one shared encoder — so a single model covers all ten instead of
one model per signal. And because per-window normalization would otherwise throw
away the absolute level of a pressure waveform, the `(loc, scale)` stripped out by the
scaler — together with each patch's own mean and standard deviation — is fed back into
**every** layer as AdaLN modulation (`LSCNorm`), keeping clinically meaningful
magnitudes available to the encoder. PPG amplitude is set by the device's gain rather
than by physiology, so its absolute `(loc, scale)` is gated out of the conditioning.

The modalities are ECG, ABP, PPG, CVP, CO2, AWP, ICP, RESP_Impedance, RESP_Flow and
PAP (`signal_type` 0–9; see `carmen.SIGNAL_TYPE_NAMES`).

> [!IMPORTANT]
> CARMEN is pretrained at **100 Hz** with a patch size of **25 samples (0.25 s/token)**.
> Resample your signals to 100 Hz before use.

> [!NOTE]
> This repository is **inference-only** — the pretraining loop is not included.

## 🚀 Quick Start

### 📦 Installation

```bash
git clone https://github.com/dlcjfgmlnasa/CARMEN.git && cd CARMEN
pip install -e .              # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10, PyTorch ≥ 2.2, einops ≥ 0.7.

### 🧠 Model weights

> **Pretrained weights will be published here upon publication of the full paper.**
> Until then this repository ships the model implementation and inference API only —
> the release asset referenced below is not yet available. You can still build the
> model from a config and run a forward pass without weights
> (see [`examples/00_smoke_test.py`](examples/00_smoke_test.py)).

Weights are distributed as a **GitHub Release asset** and are *not* committed to git.
Download a checkpoint into `checkpoints/` — see [`checkpoints/README.md`](checkpoints/README.md):

```bash
curl -L -o checkpoints/carmen.pt \
  https://github.com/dlcjfgmlnasa/CARMEN/releases/download/v1.0/carmen.pt
```

Each checkpoint embeds its own `ModelConfig`, so the architecture is reconstructed
automatically — you never specify it by hand.

### ✨ Extracting features

```python
import torch
from carmen import DownstreamModelWrapper, make_batch

# 1. Load the pretrained encoder (frozen, eval mode)
wrapper = DownstreamModelWrapper("checkpoints/carmen.pt", device="cpu")

# 2. Pack raw 1-D signals (100 Hz) into a batch — one patient, multiple modalities
t = torch.linspace(0, 30, 3000)
batch = make_batch(
    [("ecg", torch.sin(2*torch.pi*1.2*t)),
     ("ppg", torch.sin(2*torch.pi*1.2*t - 0.6)),
     ("abp", 80 + 30*torch.sin(2*torch.pi*1.2*t - 0.3))],
    patch_size=wrapper.patch_size,
)

# 3. One feature vector per patient
features = wrapper.extract_features(batch)   # (B, d_model)
```

No checkpoint yet? `python examples/00_smoke_test.py` verifies the install against a
randomly initialized model.

## 🧩 Inference API

`CARMEN.from_pretrained(path)` returns the bare encoder; `DownstreamModelWrapper` adds
loading, freezing, pooling and LoRA on top. `batch` below is what `make_batch` returns.

```python
from carmen import CARMEN, DownstreamModelWrapper

model   = CARMEN.from_pretrained("checkpoints/carmen.pt")                  # bare encoder
wrapper = DownstreamModelWrapper("checkpoints/carmen.pt", device="cuda")   # + freeze / pool / LoRA

# ── Representations ──────────────────────────────────────────────────────
feats = wrapper.extract_features(batch)                 # (B, d_model)      pooled, frozen
feats = wrapper.extract_features(batch, pool="none")    # (B, N, d_model)   per patch
enc   = model.extract_features(batch)                   # dict of raw encoder outputs

# ── Adaptation & scoring ─────────────────────────────────────────────────
wrapper.inject_lora(rank=8)                             # LoRA on q_proj / v_proj
score = wrapper.get_reconstruction_loss(batch, mask)    # scalar MSE, anomaly scoring

# ── Finer tokens from the same weights ───────────────────────────────────
fine = DownstreamModelWrapper("checkpoints/carmen.pt", patch_stride=5)
                                                        # overlapping patches; RoPE positions
                                                        # are rescaled to physical spacing
```

<details>
<summary><b>Feeding your own data</b></summary>

<br/>

`make_batch` covers the common case. For full control, build one `BiosignalSample` per
channel and collate them with `PackCollate`:

```python
from carmen import BiosignalSample, PackCollate, CHANNEL_NAME_TO_SIGNAL_TYPE

sample = BiosignalSample(
    values=ecg,                 # 1-D tensor @ 100 Hz
    length=ecg.numel(),
    channel_idx=0, recording_idx=0, n_channels=1, win_start=0,
    sampling_rate=100.0,
    signal_type=CHANNEL_NAME_TO_SIGNAL_TYPE["ECG II"],   # -> 0
    session_id="patient-001",   # samples sharing a session are packed together
    start_sample=0,
)
batch = PackCollate(max_length=8192, patch_size=25)([sample])
```

`collate_mode="any_variate"` (default) groups a patient's modalities into one row so
the encoder can attend across them; `collate_mode="ci"` treats each signal as an
independent row.

</details>

## 🔬 Examples

📓 &nbsp;**[`quickstart.ipynb`](examples/quickstart.ipynb)** — end to end: build → features → pretrained weights

- **[`00_smoke_test.py`](examples/00_smoke_test.py)** — build from config and run a forward, no weights needed
- **[`01_extract_features.py`](examples/01_extract_features.py)** — load a checkpoint, extract pooled features
- **[`02_downstream_probe.py`](examples/02_downstream_probe.py)** — linear probe and LoRA on frozen features

```bash
python examples/00_smoke_test.py
python examples/01_extract_features.py checkpoints/carmen.pt
```

## 🗂️ Repository Layout

```
carmen/
├── model.py         CARMEN encoder + inference API
├── config.py        ModelConfig (embedded in every checkpoint)
├── checkpoint.py    checkpoint save / load
├── wrapper.py       DownstreamModelWrapper (load / freeze / LoRA), LinearProbe
├── batch.py         make_batch / to_device — raw signals -> PackedBatch
├── loss.py          MaskedPatchLoss (reconstruction scoring)
├── data/            PackCollate (bin-packing), BiosignalSample, signal-type maps
└── modules/         attention (GQA), GLU FFN, RMSNorm / LSCNorm, patch embedding,
                     packed scalers, RoPE / attention bias
examples/            runnable examples + quickstart notebook
checkpoints/         put downloaded weights here (gitignored)
```

## ⚠️ Caveats

- **`any_variate` batching is non-deterministic.** `PackCollate` trims a patient's
  variates to one common length so they pair up; with unequal-length inputs that length
  is drawn at random. Seed `random.seed()` if you need reproducible batches, or use
  `collate_mode="ci"`.

## 🙏 Acknowledgements

This research was supported by a grant of the Korea Health Technology R&D Project through
the Korea Health Industry Development Institute (KHIDI), funded by the Ministry of Health &
Welfare, Republic of Korea (grant number : RS-2024-00439677 , NTIS number:2460003917)

## 📜 Citation

CARMEN has been accepted to the AI4Health Workshop at NeurIPS 2026; the full paper is in
preparation. Until a citable version is out, please cite this repository:

```bibtex
@software{carmen2026,
  title  = {CARMEN: A Cardiorespiratory Foundation Model for Continuous Physiological Waveforms},
  author = {The CARMEN Authors},
  year   = {2026},
  url    = {https://github.com/dlcjfgmlnasa/CARMEN}
}
```

## ⚖️ License

Released under the [Apache License 2.0](LICENSE).
