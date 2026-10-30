# Checkpoints

CARMEN is released in three sizes. Weights are hosted on the Hugging Face Hub at
[dlcjfgmlnasa28/CARMEN](https://huggingface.co/dlcjfgmlnasa28/CARMEN) — not committed to git, so
clones stay small. `CARMEN.from_pretrained("carmen-base")` downloads and caches them for
you; to keep a local copy, place the files here.

| Model | Params | File | Size (fp32) |
| --- | --: | --- | --: |
| CARMEN-Small | 52M | `carmen-small.pt` | ~210 MB |
| CARMEN-Base | 174M | `carmen-base.pt` | ~700 MB |
| CARMEN-Large | 410M | `carmen-large.pt` | ~1.6 GB |

Download the checkpoint(s) you need and place them here:

```
checkpoints/
├── carmen-small.pt
├── carmen-base.pt       # <- the examples default to Base
└── carmen-large.pt
```

If you already have a checkpoint file, just place it here:
`CARMEN.from_pretrained("carmen-base")` then loads the local file and never contacts the Hub.

## Download

```bash
pip install huggingface_hub
for size in small base large; do
  huggingface-cli download dlcjfgmlnasa28/CARMEN carmen-$size.pt --local-dir checkpoints
done
```

## Publishing weights (maintainers)

A training checkpoint also carries the optimizer state, which roughly triples the file.
Keep only what inference needs before uploading:

```python
import torch
s = torch.load("ckpt_from_training.pt", map_location="cpu", weights_only=False)
torch.save({k: s[k] for k in ("model_state_dict", "config", "epoch")}, "checkpoints/carmen-base.pt")
```

Then upload it to the Hub:

```bash
huggingface-cli upload dlcjfgmlnasa28/CARMEN checkpoints/carmen-base.pt carmen-base.pt
```

## What is inside a checkpoint

Each checkpoint is a `torch.save` dict with at least:

| key                | meaning                                                         |
| ------------------ | -------------------------------------------------------------- |
| `model_state_dict` | model weights                                                  |
| `config`           | the `ModelConfig` used to build the model (so it self-restores) |
| `epoch`            | training epoch the checkpoint was taken at                     |

Because `config` is embedded, you never have to specify the architecture by hand —
`CARMEN.from_pretrained` / `DownstreamModelWrapper` reconstruct it automatically, for
every size.
