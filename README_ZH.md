# HiDe-KST

[English](README.md) | 简体中文

本仓库是论文 **HiDe-KST: Hierarchical Decomposition with Low-Rank Koopman for Spatio-Temporal Forecasting** 的官方实现。

HiDe-KST 是一种面向交通、空气质量和水质数据的分解驱动时空预测模型。该方法结合多尺度时间 patch 金字塔、全局—局部空间分解、紧凑 Koopman 演化与残差式 backcast/forecast 学习，在保持较强预测能力的同时降低计算量与 GPU 显存消耗。

## 方法特点

- **层次化时间分解：**使用逐渐增大的 patch 尺度建模不同时间分辨率下的动态。
- **全局—局部空间分解：**图分支提取共享空间结构，残差分支保留节点特异性变化。
- **低秩 Koopman 演化：**在学习到的测量空间中使用紧凑线性算子建模分解后的动态。
- **残差式预测：**每个 block 的 backcast 去除已解释的动态，各 block 的 forecast 累加形成最终预测。
- **统一实验流程：**同一套配置与训练接口支持主模型、消融、敏感性实验和论文 baseline。

## 仓库结构

```text
HierarchicalKST/
├── experiments/
│   ├── forecast.py              # 训练与评估入口
│   ├── datasets.py              # 统一数据读取
│   ├── configs/
│   │   ├── hkst/                # HiDe-KST 正文实验
│   │   ├── ablation/            # 消融实验
│   │   ├── sensitivity/         # 敏感性实验
│   │   └── ...                  # baseline 配置
├── models/
│   ├── hkst/                    # 本文模型
│   └── ...                      # baseline 实现或适配器
├── utils/                       # 指标、损失、checkpoint 和图处理工具
├── datasets/                    # 本地数据；真实内容被 Git 忽略
├── experiment_storage/          # 实验输出；被 Git 忽略
├── Makefile
├── requirements.txt
├── README.md
└── README_ZH.md
```

## 环境配置

实验使用的服务器环境为：

- Python 3.9
- PyTorch 2.5.1
- CUDA 11.8
- NVIDIA V100 32 GB

创建独立 Conda 环境：

```bash
conda create -n hidekst python=3.9 -y
conda activate hidekst

# 首先安装与本机 CUDA 对应的 PyTorch。
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## 数据准备

所有数据应放置在 `datasets/` 下。整理后的数据压缩包和网盘下载链接将在后续补充；当前公开版本不包含原始数据。数据链接发布后，请下载并校验文件，然后直接解压到仓库根目录：

```bash
sha256sum hidekst_datasets.zip
unzip hidekst_datasets.zip -d .
```

解压后的目录应为：

```text
datasets/
├── PEMS03/
│   ├── data.dat
│   ├── desc.json
│   └── adj_mx.pkl
├── PEMS04/
│   ├── data.dat
│   ├── desc.json
│   └── adj_mx.pkl
├── PEMS07/
│   ├── data.dat
│   ├── desc.json
│   └── adj_mx.pkl
├── PEMS08/
│   ├── data.dat
│   ├── desc.json
│   └── adj_mx.pkl
├── air_yrd/
│   ├── data.nc
│   └── adj_mx.pkl
└── water_mn/
    ├── data.csv
    └── adj_mx.pkl
```

支持的数据格式如下：

| 数据类型 | 格式 | 必要文件或字段 |
|---|---|---|
| PEMS 交通数据 | memory-mapped `.dat` | `data.dat`、`desc.json` 和 `adj_mx.pkl` |
| 空气质量数据 | NetCDF `.nc` | `data.nc`、`adj_mx.pkl` 和 `ForecastDataset.variable_name`（`O3` 或 `PM2.5`） |
| 水质数据 | `.csv` | 第一列为时间戳，其余列为节点观测值，同时需要 `adj_mx.pkl` |

Gin 配置中的数据路径均相对于 `datasets/`：

```gin
build.data_dir = 'PEMS08/data.dat'
instance.gso_path = 'PEMS08/adj_mx.pkl'
```

不要在 README 中填写网盘账号密码或私人访问令牌。如果数据许可证不允许重新分发，应把网盘链接改为官方数据链接和预处理说明。

## 快速开始

所有命令都应在仓库根目录执行。

### 1. 生成实验实例

一个模板配置会展开为所有随机种子和预测长度的具体实验目录：

```bash
python -m experiments.forecast \
  --config_path=experiments/configs/hkst/hkst_PEMS08.gin \
  build_experiment
```

默认正文配置生成 12 个实例：三个随机种子（`42`、`123`、`456`）乘以四个预测长度（`24`、`48`、`72`、`96`）。

### 2. 运行单个实例

```bash
python -m experiments.forecast \
  --config_path=experiment_storage/PEMS08/hkst/output_len=24/seed=42/config.gin \
  run
```

建议先运行一个实例，确认环境、数据路径和 GPU 显存均正常，再启动完整实验。

### 3. 查看结果

```bash
python -c "import numpy as np; p='experiment_storage/PEMS08/hkst/output_len=24/seed=42/metrics.npy'; print(np.load(p, allow_pickle=True).item()['test'])"
```

`test` 字段包含 MAE、RMSE、MAPE、SMAPE 和 NSE。论文中的 MAE 与 RMSE 由三个固定随机种子的平均值和样本标准差得到。

## 复现正文实验

正文 HiDe-KST 配置包括：

```text
experiments/configs/hkst/hkst_PEMS03.gin
experiments/configs/hkst/hkst_PEMS04.gin
experiments/configs/hkst/hkst_PEMS07.gin
experiments/configs/hkst/hkst_PEMS08.gin
experiments/configs/hkst/hkst_yrd.gin
experiments/configs/hkst/hkst_mn.gin
```

一次生成全部正文实验：

```bash
make build-all path=experiments/configs/hkst
```

每个实例目录都会保存冻结后的 `config.gin` 和对应的 `command`。正式运行时应使用实例目录中的冻结配置，因为它记录了具体的随机种子、预测长度和完整参数。

论文实验首先使用 batch size 32。若 baseline 出现 OOM，则依次降低为 16 和 8；如果 batch size 8 仍然失败，则记录为 OOM。HiDe-KST 正文实验使用公开 Gin 文件中指定的 batch size。

## 常用配置

```gin
build.variables_dict = {
    'seed': [42, 123, 456],
    'instance.output_len': [24, 48, 72, 96]
}

instance.input_len = 48
get_data.batch_size = 32
train.epochs = 50
Checkpoint.patience = 7
```

- 切换数据集时，应同时修改 `build.data_dir` 和 `instance.gso_path`。
- NetCDF 空气质量数据需要设置 `ForecastDataset.variable_name`。
- 诊断实验应使用新的 `build.subdir`，避免覆盖已有结果。
- 后续分析需要模型权重时，设置 `instance.save_pth = True`。
- 后续分析需要预测值和真实值时，设置 `instance.save_vals = True`。

默认 HiDe-KST 使用三个 block，patch sizes 为 `[16, 24, 32]`，strides 为 `[8, 12, 16]`，时间 rank 为 128，空间 rank 根据数据集规模设置。

## 实验输出

实验结果保存在 `experiment_storage/`：

```text
experiment_storage/<dataset>/<model>/output_len=<H>/seed=<seed>/
├── config.gin            # 冻结后的完整配置
├── command               # 复现命令
├── metrics.npy           # 验证集、测试集指标及资源统计
├── model.pth             # 可选模型权重
├── preds.npz             # 可选预测结果
├── trues.npz             # 可选真实值
├── instance.log          # 训练日志
└── _SUCCESS              # 成功完成标记
```

## Baseline、消融和敏感性实验

仓库包含 DLinear、Koopa、PatchMLP、PatchTST、iTransformer、STWave、STDN、STAEformer 和 PDFormer 的实现或适配器。消融与敏感性配置分别位于：

```text
experiments/configs/ablation/
experiments/configs/sensitivity/
```

STAEformer 和 PDFormer 的适配代码已在源文件中注明上游实现：

- STAEformer：<https://github.com/XDZhelheim/STAEformer>
- PDFormer：<https://github.com/BUAABIGSCity/PDFormer>

公开前必须检查所有改编 baseline 的许可证，保留必要的版权与许可证声明，并说明主要修改。HiDe-KST 自身选择的许可证不能覆盖第三方代码原有的许可证义务。

## 常见问题

**出现 `ModuleNotFoundError: No module named 'gin'`**

请先激活正确的 Conda 环境，并安装 `requirements.txt` 中的 `gin-config`。

**找不到数据文件**

确认命令在仓库根目录执行，并核对 `datasets/` 下的数据目录结构。

**NetCDF 提示缺少 `variable_name`**

在 Gin 配置中将 `ForecastDataset.variable_name` 设置为需要预测的变量。

**运行命令后立即退出**

对应实例可能已经存在 `_SUCCESS` 或 `_RUNNING`。修改状态文件前必须确认没有训练进程仍在使用该实例。干净重跑时更推荐设置新的输出子目录。

**出现 CUDA OOM**

降低实际 batch size。如果使用梯度累积，应同时记录实际 batch 和有效 batch；二者在显存统计中不能互相替代。

## 引用

如果本代码对您的研究有帮助，请引用对应论文。论文正式发表后将在这里补充最终 BibTeX：

```text
HiDe-KST: Hierarchical Decomposition with Low-Rank Koopman for
Spatio-Temporal Forecasting
```

## 许可证

公开前必须选择并添加 `LICENSE` 文件。在没有许可证的情况下，即使仓库公开，也不等于自动授权他人复制、修改或分发代码。

对于 HiDe-KST 原创代码，可以根据学校或团队政策考虑 MIT 或 Apache-2.0；但最终选择必须同时满足仓库中第三方 baseline 代码的许可证要求。

## 致谢

感谢所有 baseline 方法作者和公开数据集维护者。复现实验或基于本项目开展后续研究时，请同时引用对应模型论文和数据来源。
