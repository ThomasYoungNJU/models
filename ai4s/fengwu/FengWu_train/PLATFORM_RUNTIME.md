# Training runtime notes

The public training entry point is `../scripts/train.sh` from the model root.
It selects the accelerator, controls FlagGems and timing, and supports foreground
or background execution. See:

- [User Guide](../README.md)
- [Platform environments and FlagGems policy](../docs/platforms.md)
- [Six-platform validation](../docs/validation.md)

`applepen.sh` is an optional generic Docker host wrapper. Set
`FENGWU_CONTAINER` and, when bind-mount inspection cannot map this repository,
`FENGWU_CONTAINER_WORKDIR` explicitly.

