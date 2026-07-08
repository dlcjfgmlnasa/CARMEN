# Checkpoints

Pretrained CARMEN weights are **not** stored in this git repository (a foundation-model
checkpoint is large and GitHub rejects files over 100 MB). Download the weights and place
the `.pt` file in this directory.

```
checkpoints/
└── carmen.pt        # <- put the downloaded checkpoint here
```

## Download

> **TODO:** add the download link (e.g. a GitHub Release asset, Hugging Face Hub, or an
> institutional file server).

```bash
# example (fill in the real URL):
# curl -L -o checkpoints/carmen.pt "<DOWNLOAD_URL>"
```

## What is inside a checkpoint

Each checkpoint is a `torch.save` dict with at least:

| key                | meaning                                                        |
| ------------------ | -------------------------------------------------------------- |
| `model_state_dict` | model weights                                                  |
| `config`           | the `ModelConfig` used to build the model (so it self-restores) |
| `epoch`            | training epoch the checkpoint was taken at                     |

Because `config` is embedded, you never have to specify the architecture by hand —
`DownstreamModelWrapper` / `load_checkpoint` reconstruct it automatically.
