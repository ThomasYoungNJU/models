# FengWu training implementation

This directory contains the training implementation used by the unified FengWu
package. Start with the model-level [User Guide](../README.md) and use the public
entry point from the model root:

```bash
bash scripts/train.sh --platform nvidia --device 0
```

`run_train.sh` remains the low-level entry point for an already prepared
environment. It does not install a vendor framework or prepare ERA5 data.

The baseline configuration is `config/fengwu_single_sample.yaml`. Dataset
preparation is documented in
[docs/data-and-checkpoints.md](../docs/data-and-checkpoints.md), and accelerator
details are documented in [docs/platforms.md](../docs/platforms.md).

