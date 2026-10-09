"""Compare completed SKOLR/TimeMixer runs with the paper HiDe-KST values."""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path('experiment_storage/modern_baselines')
HORIZONS = (24, 48, 72, 96)
MODELS = ('skolr', 'timemixer')
DATASETS = (
    'PEMS03', 'PEMS04', 'PEMS07', 'PEMS08',
    'air_yrdO3', 'air_yrdPM', 'water_mn',
)
LABELS = {
    'PEMS03': 'PEMS03',
    'PEMS04': 'PEMS04',
    'PEMS07': 'PEMS07',
    'PEMS08': 'PEMS08',
    'air_yrdO3': 'YRD-O3',
    'air_yrdPM': 'YRD-PM2.5',
    'water_mn': 'Water',
}

# Values printed in the final main-results table, intentionally used instead
# of re-reading experimental HiDe-KST files.
HIDE_PAPER = {
    'PEMS03': {24: (17.71, 28.59), 48: (22.05, 36.89),
               72: (25.69, 44.85), 96: (27.70, 47.63)},
    'PEMS04': {24: (21.70, 34.96), 48: (24.92, 40.43),
               72: (27.38, 44.71), 96: (30.11, 48.59)},
    'PEMS07': {24: (24.64, 40.19), 48: (29.39, 48.53),
               72: (33.34, 55.12), 96: (36.82, 61.47)},
    'PEMS08': {24: (17.03, 28.24), 48: (20.09, 33.12),
               72: (22.66, 36.76), 96: (24.36, 39.54)},
    'air_yrdO3': {24: (20.31, 27.62), 48: (23.46, 31.37),
                  72: (24.84, 33.03), 96: (25.72, 34.09)},
    'air_yrdPM': {24: (10.72, 16.15), 48: (12.74, 18.95),
                  72: (13.81, 20.12), 96: (14.19, 20.56)},
    'water_mn': {24: (0.428, 0.759), 48: (0.550, 0.908),
                 72: (0.637, 1.006), 96: (0.713, 1.093)},
}


def collect():
    values = defaultdict(lambda: {'mae': [], 'rmse': []})
    for metric_file in ROOT.glob('**/metrics.npy'):
        parts = metric_file.relative_to(ROOT).parts
        if len(parts) < 5:
            continue
        dataset, model = parts[0], parts[1]
        if dataset not in DATASETS or model not in MODELS:
            continue
        horizon_match = re.search(r'output_len=(\d+)', parts[-3])
        seed_match = re.search(r'seed=(\d+)', parts[-2])
        if not horizon_match or not seed_match:
            continue
        payload = np.load(metric_file, allow_pickle=True).item()['test']
        key = (model, dataset, int(horizon_match.group(1)))
        values[key]['mae'].append(float(payload['mae']))
        values[key]['rmse'].append(float(payload['rmse']))
    return values


def mean_std(numbers):
    array = np.asarray(numbers, dtype=float)
    std = array.std(ddof=1) if len(array) > 1 else 0.0
    return float(array.mean()), float(std)


def print_progress(values):
    print('=' * 78)
    print('Completed seed metrics (target: 3 seeds per horizon)')
    print('=' * 78)
    print(f"{'Model':<12}{'Dataset':<14}{'Seeds':>7}{'Groups complete':>18}")
    for model in MODELS:
        for dataset in DATASETS:
            counts = [len(values[(model, dataset, h)]['mae']) for h in HORIZONS]
            print(
                f"{model:<12}{LABELS[dataset]:<14}{sum(counts):>3}/12"
                f"{sum(count == 3 for count in counts):>11}/4"
            )


def print_comparison(values):
    print('\n' + '=' * 144)
    print('Completed results vs. paper HiDe-KST (Delta = baseline - HiDe; positive means HiDe-KST is better)')
    print('=' * 144)
    header = (
        f"{'Model':<11}{'Dataset':<12}{'H':>4}{'N':>3}  "
        f"{'Baseline MAE':>19}{'HiDe MAE':>11}{'Delta':>10}  "
        f"{'Baseline RMSE':>20}{'HiDe RMSE':>12}{'Delta':>10}"
    )
    print(header)
    print('-' * 144)
    wins = defaultdict(lambda: [0, 0, 0])
    for model in MODELS:
        for dataset in DATASETS:
            decimals = 3 if dataset == 'water_mn' else 2
            for horizon in HORIZONS:
                result = values.get((model, dataset, horizon))
                if not result or not result['mae']:
                    continue
                mae, mae_std = mean_std(result['mae'])
                rmse, rmse_std = mean_std(result['rmse'])
                hide_mae, hide_rmse = HIDE_PAPER[dataset][horizon]
                mae_delta, rmse_delta = mae - hide_mae, rmse - hide_rmse
                wins[model][0] += int(mae_delta > 0)
                wins[model][1] += int(rmse_delta > 0)
                wins[model][2] += 1
                mae_text = f'{mae:.{decimals}f}+/-{mae_std:.{decimals}f}'
                rmse_text = f'{rmse:.{decimals}f}+/-{rmse_std:.{decimals}f}'
                print(
                    f"{model:<11}{LABELS[dataset]:<12}{horizon:>4}"
                    f"{len(result['mae']):>3}  {mae_text:>19}"
                    f"{hide_mae:>11.{decimals}f}{mae_delta:>+10.{decimals}f}  "
                    f"{rmse_text:>20}{hide_rmse:>12.{decimals}f}"
                    f"{rmse_delta:>+10.{decimals}f}"
                )
    print('\nSummary over currently available dataset-horizon groups:')
    for model in MODELS:
        mae_wins, rmse_wins, total = wins[model]
        print(
            f'{model}: HiDe-KST lower MAE in {mae_wins}/{total}; '
            f'lower RMSE in {rmse_wins}/{total}'
        )


def main():
    values = collect()
    print_progress(values)
    print_comparison(values)


if __name__ == '__main__':
    main()
