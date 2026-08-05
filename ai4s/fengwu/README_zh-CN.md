# FengWu 六平台统一训练与推理代码

本目录将 FengWu 训练和 40 步自回归推理整合为同一套代码，已在 NVIDIA
H100、华为昇腾 910C、海光 BW1000、沐曦 C550、摩尔线程 MTT S5000 和
平头哥 PPU-ZW810E 上进行实际部署验证。

## 关于风乌

"风乌"（FengWu）是全球中期天气预报AI大模型，由上海人工智能实验室联合中国科学技术大学、上海交通大学、南京信息工程大学、中国科学院大气物理研究所等机构于2023年4月正式发布。该模型基于多模态和多任务深度学习方法构建，首次实现了在高分辨率上对核心大气变量进行超过10天的有效预报。

在性能上，风乌的10天预报误差相比DeepMind的GraphCast降低**10.87%**。在预报时效上，其有效预报时长达到**10.75天**，优于传统物理模型ECMWF HRES的8.5天上限。2024年3月，进一步升级的**"风乌GHR"**将分辨率提升至0.09°×0.09°经纬度（约9公里×9公里），较此前精细度提高7倍以上，有效预报时限进一步延长至**11.25天**，再次刷新世界纪录。

风乌模型凭卓越的预报精度与极低的算力成本，为农林牧渔、新能源、航空航海等行业气象服务提供了重要支撑，标志着AI气象预报从技术验证走向业务化应用。

完整英文 User Guide 见 [README.md](README.md)。以下为常用命令速查。

在任一支持平台上测试 FengWu 前，均应先从项目 H100 数据源机器下载所需数据，
再传输至目标平台：

- 机器名称：`p-jn-sz-cw-h1-su1-gpu02-402-12a-02u-208-118`；
- 内网地址：`10.6.208.118`；
- 机器 ID：`58f05dc7-4bb0-4e9f-8e7f-d8d57e2e0188`；
- 数据源目录：`/public/FengwuData`。

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

