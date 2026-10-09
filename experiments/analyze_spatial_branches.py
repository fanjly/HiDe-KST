"""Quantify the global and node-specific branches of trained HiDe-KST models.

No model is retrained.  The script reports four quantities:
  C: cross-node coherence of each latent forecast branch;
  E: relative squared-energy share of the two branches;
  L: normalized node-localization (0=uniform, 1=one-node concentration);
  Delta MAE: relative MAE increase when that branch is suppressed.
"""

import argparse
import csv
import glob
import pickle
import random
import sys
import types
from pathlib import Path

import gin
import numpy as np
import torch

from experiments.datasets import ForecastDataset
from torch.utils.data import DataLoader
from utils.graph_features import calc_gso
from utils.ops import default_device, to_tensor


ROOT = Path(__file__).resolve().parents[1]
EPS = 1e-12

# Import HiDe-KST without executing models/__init__.py, whose eager baseline
# imports are irrelevant to this analysis and may require optional packages.
if 'models' not in sys.modules:
    models_package = types.ModuleType('models')
    models_package.__path__ = [str(ROOT / 'models')]
    sys.modules['models'] = models_package
from models.hkst.model import hkst


# Register only the experiment-level bindings needed to reconstruct a saved
# run. Other training-only entries in config.gin are deliberately skipped.
@gin.configurable('instance')
def _saved_instance(input_len=None, output_len=None, seed=None,
                    model_type=None, save_vals=None, save_pth=None,
                    gso_path=None, gso_type=None):
    return None


@gin.configurable('get_data')
def _saved_get_data(batch_size=32, input_len=None, output_len=None,
                    existing=None, num_workers=0, pin_memory=True):
    return None


def query(name, default=None):
    try:
        return gin.query_parameter(name)
    except ValueError:
        return default


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def resolve_dataset_path(path):
    path = Path(path)
    return path if path.is_absolute() else ROOT / 'datasets' / path


def make_loader():
    input_len = int(query('instance.input_len'))
    output_len = int(query('instance.output_len'))
    batch_size = int(query('get_data.batch_size', 32))
    train_set = ForecastDataset('train', input_len, output_len)
    test_set = ForecastDataset.from_existing(train_set, 'test')
    loader = DataLoader(test_set, batch_size=batch_size, shuffle=False,
                        drop_last=False, num_workers=0, pin_memory=True)
    return train_set, loader


def make_model(train_set, loader, device):
    gso_path = query('instance.gso_path')
    gso_type = query('instance.gso_type', 'sym_renorm_adj')
    gso = None
    if gso_path is not None:
        with resolve_dataset_path(gso_path).open('rb') as handle:
            adjacency = np.asarray(pickle.load(handle), dtype=np.float32)
        gso = torch.from_numpy(calc_gso(adjacency, gso_type).toarray()).to(device)
    return hkst(
        num_nodes=train_set.data.shape[1],
        input_len=int(query('instance.input_len')),
        output_len=int(query('instance.output_len')),
        adj_matrix=gso,
    ).to(device)


class BranchAccumulator:
    def __init__(self, num_nodes):
        self.num_nodes = num_nodes
        self.count = 0
        self.node_sum = np.zeros(num_nodes, dtype=np.float64)
        self.node_sq_sum = np.zeros(num_nodes, dtype=np.float64)
        self.total_sum = 0.0
        self.total_sq_sum = 0.0
        self.energy = 0.0
        self.localization_sum = 0.0
        self.localization_count = 0

    def update(self, value):
        # value: [B, N, P, D].  P and D are treated as repeated observations.
        value = value.double()
        observations = value.permute(0, 2, 3, 1).reshape(-1, self.num_nodes)
        self.count += observations.shape[0]
        self.node_sum += observations.sum(0).cpu().numpy()
        self.node_sq_sum += observations.square().sum(0).cpu().numpy()
        totals = observations.sum(1)
        self.total_sum += totals.sum().item()
        self.total_sq_sum += totals.square().sum().item()
        self.energy += value.square().sum().item()

        node_energy = value.square().sum(dim=(2, 3))
        proportions = node_energy / node_energy.sum(1, keepdim=True).clamp_min(EPS)
        ipr = proportions.square().sum(1)
        localization = (self.num_nodes * ipr - 1.0) / max(self.num_nodes - 1, 1)
        self.localization_sum += localization.sum().item()
        self.localization_count += localization.numel()

    def coherence(self):
        diagonal_ss = (
            self.node_sq_sum - np.square(self.node_sum) / max(self.count, 1)
        ).sum()
        total_ss = self.total_sq_sum - self.total_sum ** 2 / max(self.count, 1)
        off_diagonal_ss = total_ss - diagonal_ss
        return off_diagonal_ss / ((self.num_nodes - 1) * diagonal_ss + EPS)

    def localization(self):
        return self.localization_sum / max(self.localization_count, 1)


def inverse_arrays(loader, forecast, target):
    forecast = forecast.detach().cpu().numpy()
    target = target.detach().cpu().numpy()
    if loader.dataset.scale:
        forecast = loader.dataset.inverse_transform(forecast)
        target = loader.dataset.inverse_transform(target, round=True)
    return forecast, target


@torch.no_grad()
def evaluate(model, loader, capture=False, ablate=None):
    model.eval()
    model.set_spatial_branch_analysis(capture=capture, ablate=ablate)
    global_stats = residual_stats = None
    absolute_error = 0.0
    value_count = 0

    for data in loader:
        x, y, x_stamp, y_stamp = map(to_tensor, data)
        forecast = model(x, x_stamp, y_stamp)
        forecast_np, target_np = inverse_arrays(loader, forecast, y)
        absolute_error += np.abs(forecast_np - target_np).sum(dtype=np.float64)
        value_count += target_np.size

        if capture:
            branch_outputs = model.get_spatial_branch_outputs()
            if global_stats is None:
                nodes = branch_outputs[0][0].shape[1]
                global_stats = BranchAccumulator(nodes)
                residual_stats = BranchAccumulator(nodes)
            for global_branch, residual_branch in branch_outputs:
                global_stats.update(global_branch)
                residual_stats.update(residual_branch)

    return absolute_error / value_count, global_stats, residual_stats


def dataset_name():
    data_dir = str(query('ForecastDataset.data_dir'))
    variable = query('ForecastDataset.variable_name')
    if data_dir.startswith('air_yrd') and variable:
        return f'YRD-{variable}'
    return Path(data_dir).parts[0]


def analyze_checkpoint(checkpoint_dir):
    checkpoint_dir = Path(checkpoint_dir).resolve()
    gin.clear_config()
    gin.parse_config_file(str(checkpoint_dir / 'config.gin'), skip_unknown=True)
    seed = int(query('instance.seed', 0))
    set_seed(seed)
    device = default_device()
    train_set, loader = make_loader()
    model = make_model(train_set, loader, device)
    state = torch.load(checkpoint_dir / 'model.pth', map_location=device,
                       weights_only=True)
    model.load_state_dict(state)

    full_mae, global_stats, residual_stats = evaluate(
        model, loader, capture=True, ablate=None
    )
    without_global_mae, _, _ = evaluate(
        model, loader, capture=False, ablate='global'
    )
    without_residual_mae, _, _ = evaluate(
        model, loader, capture=False, ablate='residual'
    )
    total_energy = global_stats.energy + residual_stats.energy
    return {
        'dataset': dataset_name(),
        'seed': seed,
        'checkpoint': str(checkpoint_dir),
        'C_g': global_stats.coherence(),
        'C_r': residual_stats.coherence(),
        'E_g': global_stats.energy / (total_energy + EPS),
        'E_r': residual_stats.energy / (total_energy + EPS),
        'L_g': global_stats.localization(),
        'L_r': residual_stats.localization(),
        'mae_full': full_mae,
        'mae_without_global': without_global_mae,
        'mae_without_residual': without_residual_mae,
        'delta_mae_g_pct': (without_global_mae - full_mae) / full_mae * 100.0,
        'delta_mae_r_pct': (without_residual_mae - full_mae) / full_mae * 100.0,
    }


def expand_checkpoints(patterns):
    paths = []
    for pattern in patterns:
        matches = glob.glob(pattern)
        paths.extend(matches if matches else [pattern])
    unique = []
    for path in paths:
        resolved = str(Path(path).resolve())
        if resolved not in unique:
            unique.append(resolved)
    return unique


def summarize(rows):
    grouped = {}
    for row in rows:
        grouped.setdefault(row['dataset'], []).append(row)
    fields = ['C_g', 'C_r', 'E_g', 'E_r', 'L_g', 'L_r',
              'delta_mae_g_pct', 'delta_mae_r_pct']
    print('\nDataset        N       C_g/C_r       E_g/E_r       L_g/L_r    '
          'DeltaMAE_g/DeltaMAE_r (%)')
    print('-' * 99)
    for dataset, members in sorted(grouped.items()):
        means = {field: np.mean([row[field] for row in members]) for field in fields}
        print(f"{dataset:<13} {len(members):>2}  "
              f"{means['C_g']:.4f}/{means['C_r']:.4f}  "
              f"{means['E_g']:.4f}/{means['E_r']:.4f}  "
              f"{means['L_g']:.4f}/{means['L_r']:.4f}  "
              f"{means['delta_mae_g_pct']:+.2f}/{means['delta_mae_r_pct']:+.2f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('checkpoints', nargs='+', help=(
        'checkpoint directories or glob patterns containing config.gin and model.pth'
    ))
    parser.add_argument('--output', default=(
        'experiment_storage/analysis/spatial_branch_metrics.csv'
    ))
    args = parser.parse_args()

    rows = []
    for checkpoint in expand_checkpoints(args.checkpoints):
        print(f'[ANALYZE] {checkpoint}', flush=True)
        rows.append(analyze_checkpoint(checkpoint))
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    summarize(rows)
    print(f'\nSaved: {output}')


if __name__ == '__main__':
    main()
