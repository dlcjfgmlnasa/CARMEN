# CARMEN

**CARMEN** is a cardiorespiratory foundation model for continuous physiological
waveforms. A single Transformer encoder (~30M parameters) is pretrained across
**9 signal modalities** and transfers to feature extraction, cross-modal waveform
generation, and forecasting.

| id | modality                      | id | modality                         |
|----|-------------------------------|----|----------------------------------|
| 0  | ECG (electrocardiogram)       | 5  | AWP (airway pressure)            |
| 1  | ABP (arterial blood pressure) | 6  | ICP (intracranial pressure)      |
| 2  | PPG (photoplethysmography)    | 7  | RESP_Impedance (chest impedance) |
| 3  | CVP (central venous pressure) | 8  | RESP_Flow (ventilator flow)      |
| 4  | CO2 (capnography)             |    |                                  |

CARMEN is pretrained at **100 Hz** with a patch size of **200 samples (2 s/token)** —
resample your signals to 100 Hz before use.

This repository is **inference-only**: the pretraining loop is not included.

## Install

```bash
pip install -e .            # or: pip install -r requirements.txt
```

Then download a checkpoint into `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md)). Weights are **not** committed to git.

## Quickstart

```python
import torch
from carmen import DownstreamModelWrapper, make_batch

# 1. Load the pretrained encoder (the checkpoint embeds its own config)
wrapper = DownstreamModelWrapper("checkpoints/carmen.pt", device="cpu")

# 2. Pack raw 1-D signals (100 Hz) into a batch — one patient, multiple modalities
t = torch.linspace(0, 30, 3000)
batch = make_batch(
    [("ecg", torch.sin(2*torch.pi*1.2*t)),
     ("ppg", torch.sin(2*torch.pi*1.2*t - 0.6)),
     ("abp", 80 + 30*torch.sin(2*torch.pi*1.2*t - 0.3))],
    patch_size=wrapper.patch_size,
)

# 3. Extract a feature vector per patient
features = wrapper.extract_features(batch)   # (B, d_model)
```

No checkpoint yet? Run `python examples/00_smoke_test.py` to verify the install with a
randomly initialized model.

## Inference API

`CARMEN.from_pretrained(path)` gives you the bare encoder; `DownstreamModelWrapper`
adds loading + freezing + pooling + LoRA on top. Key methods:

| method                                        | purpose                                             |
|-----------------------------------------------|-----------------------------------------------------|
| `model.extract_features(batch)`               | encoder embeddings for downstream heads             |
| `model.generate_cross_modal(batch, target)`   | synthesize one modality from others (zero-shot)     |
| `model.forecast(batch)`                       | block next-patch prediction map `(B, N, K, P)`      |
| `model.generate(batch, n_steps)`              | autoregressive waveform roll-out                    |
| `wrapper.extract_features(batch, pool=...)`   | frozen features (+ optional gap-masking / pooling)  |
| `wrapper.inject_lora(rank=8)`                 | parameter-efficient fine-tuning of the encoder      |

`forward(batch, task=...)` takes `task="masked"` (bidirectional attention →
`reconstructed`, `cross_pred_per_type`) or `task="next_pred"` (causal attention →
`next_pred`).

## Examples

| file                                      | what it shows                                    |
|-------------------------------------------|--------------------------------------------------|
| `examples/quickstart.ipynb`               | end-to-end notebook (build → features → generate)|
| `examples/00_smoke_test.py`               | build from config, run a forward (no weights)    |
| `examples/01_extract_features.py`         | load a checkpoint, extract features              |
| `examples/02_downstream_probe.py`         | linear probe / LoRA on frozen features           |
| `examples/03_cross_modal_generation.py`   | ECG + PPG → ABP                                   |
| `examples/04_forecasting.py`              | forecast + autoregressive generation             |

```bash
python examples/00_smoke_test.py
python examples/01_extract_features.py checkpoints/carmen.pt
```

## Repository layout

```
carmen/
  model.py         CARMEN encoder + inference API
  config.py        ModelConfig (embedded in every checkpoint)
  checkpoint.py    checkpoint save / load
  wrapper.py       DownstreamModelWrapper (load / freeze / LoRA), LinearProbe
  batch.py         make_batch / to_device — raw signals -> PackedBatch
  loss.py          MaskedPatchLoss (reconstruction scoring)
  data/            PackCollate (bin-packing), BiosignalSample, signal-type maps
  modules/         building blocks — attention (GQA), GLU FFN, RMSNorm/LSCNorm,
                   patch embedding, packed scalers, RoPE / attention bias
examples/          runnable examples + quickstart notebook
checkpoints/       put downloaded weights here (gitignored)
```

## Notes

- The checkpoint stores `model_state_dict`, `config`, and `epoch`. The embedded
  `config` reconstructs the architecture automatically — you never specify it by hand.
- Feeding data: `make_batch` covers the common case. For full control, build a
  `BiosignalSample` per channel and collate with `PackCollate`. Channel names map to
  signal types via `carmen.CHANNEL_NAME_TO_SIGNAL_TYPE`.
- In `collate_mode="any_variate"`, `PackCollate` trims a patient's variates to one
  common length so they pair up across modalities; with unequal-length inputs that
  length is chosen randomly, so seed `random.seed()` if you need reproducible batches.
- Reliable cross-modal source/target pairs are listed in
  `carmen.CROSS_PRED_ALLOWED_PAIRS`.

## License

Released under the [Apache License 2.0](LICENSE). Portions of the model building
blocks are adapted from [uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
(Apache 2.0); see [`NOTICE`](NOTICE).
