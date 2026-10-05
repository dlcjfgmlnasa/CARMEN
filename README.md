# CARMEN

### A Cardiorespiratory Foundation Model for Continuous Physiological Waveforms

[![License](https://img.shields.io/badge/License-Apache_2.0-blue.svg)](LICENSE)
[![Python](https://img.shields.io/badge/Python-3.10+-3776AB.svg?logo=python&logoColor=white)](https://www.python.org/)
[![PyTorch](https://img.shields.io/badge/PyTorch-2.2+-EE4C2C.svg?logo=pytorch&logoColor=white)](https://pytorch.org/)
[![Modalities](https://img.shields.io/badge/modalities-10-teal.svg)](#-overview)
[![Params](https://img.shields.io/badge/params-~174M-8A2BE2.svg)](#-overview)

[**Quickstart Notebook**](examples/quickstart.ipynb) · [**Examples**](examples) · [**Inference API**](#-inference-api) · [**Model Weights**](#-model-weights) (coming soon)

## 📰 News

- **2026.10** — CARMEN has been accepted to the **AI4Health Workshop at NeurIPS 2026**. 🎉

## 📖 Overview

**CARMEN** is a foundation model for the continuous waveforms recorded at the bedside
and in the operating room. A **single Transformer encoder (~174M parameters)** is
pretrained on **10 signal modalities** and used as a frozen **feature extractor** for
downstream clinical tasks (detection, prediction, outcome, estimation and
phenotyping), with only a lightweight head trained on top.

Two design choices do most of the work:

- **One tokenizer, one encoder.** Every modality is tokenized the same way, as raw
  patches fed to a shared encoder, so a single model covers all ten signals instead of
  one model per signal.
- **Absolute level is preserved (`LSCNorm`).** Per-window normalization would
  otherwise discard the absolute level of a pressure waveform. The `(loc, scale)`
  removed by the scaler, together with each patch's own mean and standard deviation,
  is fed back into **every** layer as AdaLN modulation, so clinically meaningful
  magnitudes stay available to the encoder. PPG is the exception: its amplitude is set
  by device gain rather than physiology, so PPG is excluded from this conditioning
  altogether.

> [!NOTE]
> This repository is **inference-only** — the pretraining loop is not included.

> [!WARNING]
> **Research use only.** CARMEN is not a medical device and has not been cleared or
> approved for clinical use. Do not use its outputs for diagnosis or to guide patient
> care.

## 📐 Input Requirements

CARMEN conditions on the **absolute level** of each signal, so inputs must be in the
units used during pretraining. A signal in the wrong unit will run without error but
produce degraded features.

| `signal_type` | Modality | `make_batch` key | Expected unit |
| :-: | --- | --- | --- |
| 0 | ECG | `"ecg"` | mV |
| 1 | ABP | `"abp"` | mmHg |
| 2 | PPG | `"ppg"` | arbitrary (device-dependent; absolute level is not used) |
| 3 | CVP | `"cvp"` | mmHg |
| 4 | CO2 | `"co2"` | mmHg (convert vol% × 7.13) |
| 5 | AWP | `"awp"` | cmH₂O (convert hPa × 1.0197) |
| 6 | ICP | `"icp"` | mmHg |
| 7 | RESP_Impedance | `"resp_impedance"` | arbitrary (device-dependent impedance) |
| 8 | RESP_Flow | `"resp_flow"` | L/min |
| 9 | PAP | `"pap"` | mmHg |

The authoritative mapping is `carmen.SIGNAL_TYPE_NAMES`.

> [!IMPORTANT]
> CARMEN is pretrained at **100 Hz** with a patch size of **25 samples (0.25 s/token)**.
> Resample your signals to 100 Hz before use.

## 🚀 Quick Start

### 📦 Installation

```bash
git clone https://github.com/dlcjfgmlnasa/CARMEN.git && cd CARMEN
pip install -e .              # or: pip install -r requirements.txt
```

Requires Python ≥ 3.10, PyTorch ≥ 2.2, einops ≥ 0.7.

### ✅ Verify the install (no weights needed)

```bash
python examples/00_smoke_test.py
```

This builds the model from a config with random weights and runs a forward pass. It is
the only example that works before the pretrained weights are released.

### 🧠 Model weights

> [!NOTE]
> **Pretrained weights are not yet available.** They will be published as a GitHub
> Release asset upon publication of the full paper. Until then this repository ships
> the model implementation and inference API only, and every example below that loads
> `checkpoints/carmen.pt` requires the release.

Once released, download the checkpoint into `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md)); weights are *not* committed to git:

```bash
# available after release
curl -L -o checkpoints/carmen.pt \
  https://github.com/dlcjfgmlnasa/CARMEN/releases/download/v2.0.0/carmen.pt
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

> [!WARNING]
> **Pass equal-length signals.** With the default `collate_mode="any_variate"`,
> `PackCollate` trims a patient's signals to one common length so they pair up. When
> the inputs differ in length, that length is drawn at random, and a signal shorter
> than the drawn length is **dropped** from the batch (as is any signal shorter than
> 5 patches = 1.25 s), so features can change between runs. `make_batch` warns when it
> drops a signal. For reproducible results, pass equal-length signals, call
> `random.seed(0)` before building the batch, or use `collate_mode="ci"`.

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

# ── Adaptation ───────────────────────────────────────────────────────────
wrapper.inject_lora(rank=8)                             # LoRA on q_proj / v_proj

# ── Finer tokens from the same weights ───────────────────────────────────
fine = DownstreamModelWrapper("checkpoints/carmen.pt", patch_stride=5)
                                                        # overlapping patches; RoPE positions
                                                        # are rescaled to physical spacing
```

**Feeding your own data**

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

## 🔬 Examples

📓 **[`quickstart.ipynb`](examples/quickstart.ipynb)** — end to end: build → features → pretrained weights

- **[`00_smoke_test.py`](examples/00_smoke_test.py)** — build from config and run a forward pass, no weights needed
- **[`01_extract_features.py`](examples/01_extract_features.py)** — load a checkpoint, extract pooled features *(requires weights)*
- **[`02_downstream_probe.py`](examples/02_downstream_probe.py)** — linear probe and LoRA on frozen features *(requires weights)*

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
├── data/            PackCollate (bin-packing), BiosignalSample, signal-type maps
└── modules/         attention (GQA), GLU FFN, RMSNorm / LSCNorm, patch embedding,
                     packed scalers, RoPE / attention bias
examples/            runnable examples + quickstart notebook
checkpoints/         put downloaded weights here (gitignored)
```

## 🙏 Acknowledgements

This research was supported by a grant of the Korea Health Technology R&D Project
through the Korea Health Industry Development Institute (KHIDI), funded by the Ministry
of Health & Welfare, Republic of Korea (grant number: RS-2024-00439677; NTIS number:
2460003917).

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
