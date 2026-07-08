# Checkpoints

CARMEN is a **~30M-parameter** model, so a checkpoint is roughly **120 MB (fp32)** or
**60 MB (fp16)**. Weights are distributed as a **GitHub Release asset** — not committed
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
# once a release is published (example tag v1.0):
curl -L -o checkpoints/carmen.pt \
  https://github.com/dlcjfgmlnasa/CARMEN/releases/download/v1.0/carmen.pt
```

## Publishing weights (maintainers)

Upload the checkpoint as a Release asset — fp32 (~120 MB) is fine:

```bash
gh release create v1.0 checkpoints/carmen.pt \
  --title "CARMEN v1.0" --notes "Pretrained CARMEN weights"
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
`DownstreamModelWrapper` / `load_checkpoint` reconstruct it automatically.
