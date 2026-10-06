# HiDe-KST

English | [简体中文](README_ZH.md)

Official implementation of **HiDe-KST: Hierarchical Decomposition with Low-Rank Koopman for Spatio-Temporal Forecasting**.

HiDe-KST is a decomposition-driven forecasting model for traffic, air-quality, and water-quality data. It combines a multi-scale temporal patch pyramid, global--local spatial decomposition, compact Koopman evolution, and residual backcast/forecast learning. Low-rank spatial and temporal operators are used to retain competitive forecasting accuracy while reducing computation and GPU-memory usage.

## Highlights

- **Hierarchical temporal decomposition:** progressively coarser patch scales model temporal dynamics at multiple resolutions.
- **Global--local spatial decomposition:** a graph-based branch captures shared spatial structure, while a residual branch preserves node-specific variation.
- **Low-rank Koopman evolution:** decomposed components evolve through compact linear operators in learned measurement spaces.
- **Residual forecasting:** block-wise backcasts remove explained dynamics and block forecasts are accumulated into the final prediction.
- **Unified benchmark pipeline:** one configuration system supports the proposed model, ablations, sensitivity studies, and competitive baselines.

## Repository layout

```text
HierarchicalKST/
├── experiments/
│   ├── forecast.py              # training and evaluation entry point
│   ├── datasets.py              # unified dataset reader
│   ├── configs/
│   │   ├── hkst/                # main HiDe-KST experiments
│   │   ├── ablation/            # ablation experiments
│   │   ├── sensitivity/         # sensitivity experiments
│   │   └── ...                  # baseline configurations
├── models/
│   ├── hkst/                    # proposed model
│   └── ...                      # baseline implementations/adapters
├── utils/                       # metrics, losses, checkpoints, and graph tools
├── datasets/                    # local datasets; contents are ignored by Git
├── experiment_storage/          # generated runs; ignored by Git
├── Makefile
├── requirements.txt
└── README.md
```

## Environment

The experiments were tested with the following server environment:

- Python 3.9
- PyTorch 2.5.1
- CUDA 11.8
- NVIDIA V100 32 GB

Create an isolated environment:

```bash
conda create -n hidekst python=3.9 -y
conda activate hidekst

# Install the PyTorch build appropriate for your CUDA environment first.
pip install torch==2.5.1 --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

## Data preparation

The repository expects all datasets under `datasets/`. The seven processed datasets can be downloaded from [Tianyi Cloud](https://cloud.189.cn/t/ANJ36nzIVvYr) using the access code `ap6b`. After downloading the archive, extract it directly into the repository root:

```bash
sha256sum hidekst_datasets.zip
unzip hidekst_datasets.zip -d .
```

Expected layout:

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

Supported input formats:

| Domain | Format | Required files/fields |
|---|---|---|
| PEMS traffic | memory-mapped `.dat` | `data.dat`, `desc.json`, and `adj_mx.pkl` |
| Air quality | NetCDF `.nc` | `data.nc`, `adj_mx.pkl`, and `ForecastDataset.variable_name` (`O3` or `PM2.5`) |
| Water quality | `.csv` | timestamp in the first column, node observations in the remaining columns, and `adj_mx.pkl` |

Paths in Gin files are relative to `datasets/`. For example:

```gin
build.data_dir = 'PEMS08/data.dat'
instance.gso_path = 'PEMS08/adj_mx.pkl'
```

## Quick start

Run all commands from the repository root.

### 1. Build experiment instances

A template configuration expands into one directory for every random-seed and prediction-horizon combination:

```bash
python -m experiments.forecast \
  --config_path=experiments/configs/hkst/hkst_PEMS08.gin \
  build_experiment
```

The default main configuration creates 12 runs: three random seeds (`42`, `123`, and `456`) times four horizons (`24`, `48`, `72`, and `96`).

### 2. Run one instance

```bash
python -m experiments.forecast \
  --config_path=experiment_storage/PEMS08/hkst/output_len=24/seed=42/config.gin \
  run
```

Use a single instance first to verify the environment, dataset paths, and GPU memory before launching the complete benchmark.

### 3. Inspect the result

```bash
python -c "import numpy as np; p='experiment_storage/PEMS08/hkst/output_len=24/seed=42/metrics.npy'; print(np.load(p, allow_pickle=True).item()['test'])"
```

The `test` dictionary contains MAE, RMSE, MAPE, SMAPE, and NSE. The paper reports MAE and RMSE as the mean and sample standard deviation over the three fixed seeds.

## Reproducing the main experiments

Main HiDe-KST configurations are provided for the four PEMS datasets, YRD air-quality data, and water-quality data:

```text
experiments/configs/hkst/hkst_PEMS03.gin
experiments/configs/hkst/hkst_PEMS04.gin
experiments/configs/hkst/hkst_PEMS07.gin
experiments/configs/hkst/hkst_PEMS08.gin
experiments/configs/hkst/hkst_yrd.gin
experiments/configs/hkst/hkst_mn.gin
```

Build all main configurations with:

```bash
make build-all path=experiments/configs/hkst
```

Each generated instance contains a frozen `config.gin` and a `command` file. Run the frozen configuration rather than repeatedly running the template, because it records the exact seed, horizon, and resolved settings.

The paper protocol starts with batch size 32. If a baseline runs out of GPU memory, the batch size is reduced to 16 and then 8. A configuration that still fails at batch size 8 is reported as OOM. HiDe-KST main experiments use the batch size specified in their released Gin files.

## Configuration guide

The most frequently changed settings are:

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

- Change `build.data_dir` and `instance.gso_path` together when switching datasets.
- Set `ForecastDataset.variable_name` for NetCDF variables such as `O3` and `PM2.5`.
- Use a new `build.subdir` for diagnostic runs so that existing results are not overwritten.
- Set `instance.save_pth = True` when a later analysis requires a checkpoint.
- Set `instance.save_vals = True` when predictions and ground truth are required for downstream analysis.

The default HiDe-KST architecture uses three blocks with patch sizes `[16, 24, 32]`, strides `[8, 12, 16]`, temporal rank 128, and dataset-dependent spatial rank.

## Output format

Generated files are stored under `experiment_storage/`:

```text
experiment_storage/<dataset>/<model>/output_len=<H>/seed=<seed>/
├── config.gin            # frozen configuration
├── command               # reproduction command
├── metrics.npy           # validation/test metrics and resource usage
├── model.pth             # optional checkpoint
├── preds.npz             # optional predictions
├── trues.npz             # optional targets
├── instance.log          # training log
└── _SUCCESS              # successful completion marker
```

## Baselines and studies

The repository contains configurations and implementations/adapters for DLinear, Koopa, PatchMLP, PatchTST, iTransformer, STWave, STDN, STAEformer, and PDFormer. Ablation and sensitivity configurations are under `experiments/configs/ablation/` and `experiments/configs/sensitivity/`.

The STAEformer and PDFormer adapters explicitly identify their upstream implementations in the source files:

- STAEformer: <https://github.com/XDZhelheim/STAEformer>
- PDFormer: <https://github.com/BUAABIGSCity/PDFormer>

Before publishing this repository, review the licenses of **all** adapted baseline implementations, retain their copyright/license notices, and document substantial changes. The license selected for HiDe-KST cannot override the license obligations of third-party code.

## Troubleshooting

**`ModuleNotFoundError: No module named 'gin'`**

Activate the intended environment and install `gin-config` from `requirements.txt`.

**A dataset file cannot be found**

Run the command from the repository root and verify the layout under `datasets/`.

**NetCDF loading asks for `variable_name`**

Set `ForecastDataset.variable_name` to the required variable in the Gin configuration.

**A run exits immediately**

The instance may already contain `_SUCCESS` or `_RUNNING`. Confirm that no process is using the instance before changing status files. Prefer a new output subdirectory for a clean rerun.

**CUDA out of memory**

Reduce the physical batch size. If gradient accumulation is used, record both the physical and effective batch sizes because they are not interchangeable for memory reporting.

## Citation

If this code is useful in your research, please cite the paper. The final BibTeX entry will be added after publication:

```text
HiDe-KST: Hierarchical Decomposition with Low-Rank Koopman for
Spatio-Temporal Forecasting
```

## License

A `LICENSE` file must be selected and added before public release. Until then, public visibility alone does not grant permission to reuse the code. A permissive license such as MIT or Apache-2.0 may be suitable for the original HiDe-KST code, subject to institutional policy and the licenses of included third-party components.

## Acknowledgements

We thank the authors of the baseline methods and the maintainers of the public datasets used in this study. Please cite the corresponding papers and dataset sources when reproducing or extending the experiments.
