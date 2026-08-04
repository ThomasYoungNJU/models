# FengWu 六平台统一训练与推理代码

本目录将 FengWu 训练和 40 步自回归推理整合为同一套代码，已在 NVIDIA
H100、华为昇腾 910C、海光 BW1000、沐曦 C550、摩尔线程 MTT S5000 和
平头哥 PPU-ZW810E 上进行实际部署验证。

完整英文 User Guide 见 [README.md](README.md)。以下为常用命令速查。

## 环境

先安装与本机驱动匹配的厂商 PyTorch，再安装通用依赖：

```bash
python3 -m pip install -r requirements.txt
```

不要用 PyPI 通用 PyTorch 覆盖厂商版本。华为需要 `torch_npu`，摩尔线程需要
`torch_musa`；各平台还需要与 PyTorch 匹配的 torchvision。

## 训练

```bash
bash scripts/train.sh --platform nvidia --device 0
bash scripts/train.sh --platform ascend --device npu:0
bash scripts/train.sh --platform hcu --device 0
bash scripts/train.sh --platform metax --device 0
bash scripts/train.sh --platform mthreads --device musa:0
bash scripts/train.sh --platform thead --device 0
```

后台运行：

```bash
bash scripts/train.sh --platform nvidia --device 0 --background
```

## 推理

准备 `FengWu/model` 下的四个必要文件后执行：

```bash
bash scripts/infer.sh \
  --platform nvidia --device 0 \
  --model-dir ./FengWu/model \
  --output-dir ./FengWu/outputs \
  --steps 40
```

40 步推理应输出 40 个 NetCDF 文件。数据与权重要求见
[docs/data-and-checkpoints.md](docs/data-and-checkpoints.md)，平台环境与 FlagGems
排除策略见 [docs/platforms.md](docs/platforms.md)，实机结果见
[docs/validation.md](docs/validation.md)。

