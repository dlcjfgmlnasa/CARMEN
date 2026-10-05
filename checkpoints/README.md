# Checkpoints

CARMEN is a **~174M-parameter** model, so a checkpoint is roughly **700 MB (fp32)** or
**350 MB (fp16)**. Weights are distributed as a **GitHub Release asset** — not committed
to git, so clones stay small. (Release assets allow files up to 2 GB, so even the fp32
checkpoint fits with room to spare.)

Download the checkpoint and place it here:

```
checkpoints/
└── carmen.pt        # <- put the downloaded checkpoint here
```

## Download

Get the weights from the Releases page:
<https://github.com/dlcjfgmlnasa/CARMEN/releases>

```bash
# once a release is published (example tag v2.0.0):
curl -L -o checkpoints/carmen.pt \
  https://github.com/dlcjfgmlnasa/CARMEN/releases/download/v2.0.0/carmen.pt
```

## Publishing weights (maintainers)

A training checkpoint also carries the optimizer state, which roughly triples the file
(~2 GB, at the Release asset limit). Keep only what inference needs before uploading:

```python
import torch
s = torch.load("ckpt_from_training.pt", map_location="cpu", weights_only=False)
torch.save({k: s[k] for k in ("model_state_dict", "config", "epoch")}, "checkpoints/carmen.pt")
```

Then upload it as a Release asset (fp32, ~700 MB):

```bash
gh release create v2.0.0 checkpoints/carmen.pt \
  --title "CARMEN v2.0.0" --notes "Pretrained CARMEN weights"
# or: create a new release in the GitHub UI and drag-and-drop the .pt file
```

## What is inside a checkpoint

Each checkpoint is a `torch.save` dict with at least:

| key                | meaning                                                         |
| ------------------ | -------------------------------------------------------------- |
| `model_state_dict` | model weights                                                  |
| `config`           | the `ModelConfig` used to build the model (so it self-restores) |
| `epoch`            | training epoch the checkpoint was taken at                     |

Because `config` is embedded, you never have to specify the architecture by hand —
`CARMEN.from_pretrained` / `DownstreamModelWrapper` reconstruct it automatically.
