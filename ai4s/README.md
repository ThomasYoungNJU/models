# AI4S Models

A collection of **AI for Science (AI4S)** models. This directory hosts multiple, independent model repositories. Each model lives in its own subfolder and is self-contained, so that both **training** and **inference** work out of the box.

Every model must run training **and** inference on all six supported accelerator
families — NVIDIA, Huawei Ascend (华为昇腾), MetaX (沐曦), Hygon DCU (海光),
T-Head PPU (平头哥), and Moore Threads (摩尔线程). See the
[Multi-Platform User Guide](docs/multi-platform-guide.md) for detailed,
step-by-step setup and run instructions for each.

## Available models

| Model | Task | Platforms |
|---|---|---|
| [FengWu](fengwu/README.md) | Global medium-range weather training and 40-step inference | NVIDIA H100, Huawei Ascend 910C, Hygon BW1000, MetaX C550, Moore Threads MTT S5000, T-Head PPU-ZW810E |

## Layout

```
ai4s/
├── README.md            # this file
├── <model-a>/           # one self-contained model
│   ├── README.md        # model-specific docs
│   ├── requirements.txt # (or environment.yml / pyproject.toml)
│   ├── configs/         # training & inference configs
│   ├── data/            # data or data-download scripts
│   ├── src/             # model, dataset, and utility code
│   ├── scripts/         # train.sh / infer.sh entry points
│   ├── checkpoints/     # saved weights (usually git-ignored)
│   └── ...
├── <model-b>/
└── ...
```

## Conventions

Each model subfolder should be independently runnable and follow these guidelines:

1. **Self-contained** — keep code, configs, and dependency declarations inside the model's own folder. Avoid cross-model imports.
2. **Documented** — include a `README.md` covering the task, data, how to train, and how to run inference.
3. **Reproducible** — pin dependencies (`requirements.txt`, `environment.yml`, or `pyproject.toml`) and document the Python/CUDA versions used.
4. **Training-ready** — provide a clear training entry point and config, e.g. `scripts/train.sh` or `python -m src.train --config configs/train.yaml`.
5. **Inference-ready** — provide a clear inference entry point, e.g. `scripts/infer.sh` or `python -m src.infer --checkpoint <path>`.
6. **Weights & data** — keep large checkpoints and datasets out of git (use `.gitignore`); document where to download or how to regenerate them.
7. **Multi-platform** — the same codebase must run on all six accelerators. Take a
   `--device` argument (never hard-code `cuda`), document the actual vendor backend
   used for validation, and confirm every item in the acceptance checklist of
   the [Multi-Platform User Guide](docs/multi-platform-guide.md).

## Adding a New Model

```bash
# from within the ai4s/ directory
mkdir my-model
cd my-model
# add your code, configs, and a README.md following the conventions above
```

Then verify that a fresh checkout can both train and run inference before committing.

## License

See the [LICENSE](../LICENSE) file at the repository root.
