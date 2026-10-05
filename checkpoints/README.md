# Checkpoints

CARMEN is released in three sizes. Weights are distributed as **GitHub Release assets**
— not committed to git, so clones stay small. Release assets allow files up to 2 GB, so
every fp32 checkpoint fits.

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

## Download

Get the weights from the Releases page:
<https://github.com/dlcjfgmlnasa/CARMEN/releases>

```bash
# once a release is published (example tag v2.0.0):
for size in small base large; do
  curl -L -o checkpoints/carmen-$size.pt \
    https://github.com/dlcjfgmlnasa/CARMEN/releases/download/v2.0.0/carmen-$size.pt
done
```

## Publishing weights (maintainers)

A training checkpoint also carries the optimizer state, which roughly triples the file
(Base ~2 GB, Large ~5 GB — over the 2 GB Release asset limit). Keep only what inference
needs before uploading:

```python
import torch
s = torch.load("ckpt_from_training.pt", map_location="cpu", weights_only=False)
torch.save({k: s[k] for k in ("model_state_dict", "config", "epoch")}, "checkpoints/carmen-base.pt")
```

Then upload all three as assets of one release:

```bash
gh release create v2.0.0 checkpoints/carmen-small.pt checkpoints/carmen-base.pt \
  checkpoints/carmen-large.pt --title "CARMEN v2.0.0" --notes "Pretrained CARMEN weights"
# or: create a new release in the GitHub UI and drag-and-drop the .pt files
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
