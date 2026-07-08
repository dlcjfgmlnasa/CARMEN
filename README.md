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

## Install

```bash
pip install -r requirements.txt   # torch >= 2.2, einops >= 0.7
```

Then download a checkpoint and place it in `checkpoints/` (see
[`checkpoints/README.md`](checkpoints/README.md)). Weights are **not** committed to git.

## Quickstart

```python
import torch
from wrapper import DownstreamModelWrapper
from examples._common import make_batch

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

Build the model directly with `from model import CARMEN` (or via
`DownstreamModelWrapper` for loading + freezing + LoRA). Key methods:

| method                                        | purpose                                             |
|-----------------------------------------------|-----------------------------------------------------|
| `model.extract_features(batch)`               | encoder embeddings for downstream heads             |
| `model.generate_cross_modal(batch, target)`   | synthesize one modality from others (zero-shot)     |
| `model.forecast(batch)`                       | block next-patch prediction map `(B, N, K, P)`      |
| `model.generate(batch, n_steps)`              | autoregressive waveform roll-out                    |
| `wrapper.extract_features(batch, pool=...)`   | frozen features (+ optional gap-masking / pooling)  |
| `wrapper.inject_lora(rank=8)`                 | parameter-efficient fine-tuning of the encoder      |

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
model/        CARMEN model (biosignal_model.py), ModelConfig, checkpoint I/O
module/       building blocks — attention (GQA), GLU FFN, RMSNorm/LSCNorm,
              patch embedding, packed scalers, RoPE / attention bias
data/         PackCollate (bin-packing), BiosignalSample, signal-type maps
loss/         create_patch_mask + MaskedPatchLoss (reconstruction scoring)
wrapper.py    DownstreamModelWrapper (load / freeze / LoRA), LinearProbe
examples/     runnable examples + quickstart notebook
checkpoints/  put downloaded weights here (gitignored)
```

## Notes

- The checkpoint stores `model_state_dict`, `config`, and `epoch`. The embedded
  `config` reconstructs the architecture automatically — you never specify it by hand.
- Feeding data: build a `BiosignalSample` per channel and collate with `PackCollate`,
  or use `examples/_common.make_batch`. Channel names map to signal types via
  `data.spatial_map.CHANNEL_NAME_TO_SIGNAL_TYPE`.

## License

Released under the [Apache License 2.0](LICENSE). Portions of the model building
blocks are adapted from [uni2ts](https://github.com/SalesforceAIResearch/uni2ts)
(Apache 2.0); see [`NOTICE`](NOTICE).
