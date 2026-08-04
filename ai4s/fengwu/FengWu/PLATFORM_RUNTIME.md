# Inference runtime notes

The public inference entry point is `../scripts/infer.sh` from the model root.
It selects the accelerator, controls FlagGems and supports foreground or
background execution. See:

- [User Guide](../README.md)
- [Platform environments and FlagGems policy](../docs/platforms.md)
- [Data and checkpoint preparation](../docs/data-and-checkpoints.md)

`networks/applepen.sh` is an optional generic Docker host wrapper. Set
`FENGWU_CONTAINER` and, when bind-mount inspection cannot map this repository,
`FENGWU_CONTAINER_WORKDIR` explicitly.

