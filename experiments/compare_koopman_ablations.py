"""Compare completed Koopman ablations with the paper HiDe-KST runs."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from experiments.checkResult import FULL_RESULT_ROOTS, collect_seed_metrics


ABLATIONS = {
    "wokoopman": (
        "w/o Koopman",
        Path("experiment_storage/ablation_wokoopman"),
    ),
    "wobranchk": (
        "w/o Branch-specific K",
        Path("experiment_storage/ablation_wobranchk"),
    ),
}

DATASET_LABELS = {
    "PEMS03": "PEMS03",
    "PEMS04": "PEMS04",
    "PEMS07": "PEMS07",
    "PEMS08": "PEMS08",
    "air_yrdO3": "YRD-O3",
    "air_yrdPM": "YRD-PM2.5",
    "water_mn": "Water",
}


def mean_std(values: list[float]) -> str:
    decimals = 3 if values and max(abs(value) for value in values) < 2 else 2
    mean = np.mean(values)
    if len(values) == 1:
        return f"{mean:.{decimals}f}"
    std = np.std(values, ddof=1)
    return f"{mean:.{decimals}f}+/-{std:.{decimals}f}"


def compare(ablation_name: str) -> None:
    label, ablation_root = ABLATIONS[ablation_name]
    completed_rows = 0
    partial_rows = 0

    print(f"\n{label} vs. HiDe-KST (positive delta means the ablation is worse)")
    print(
        "Dataset    H   N   HiDe-KST MAE     Ablation MAE     "
        "Delta MAE(%)   HiDe-KST RMSE    Ablation RMSE    Delta RMSE(%)"
    )
    print("-" * 126)

    for dataset, full_root in FULL_RESULT_ROOTS.items():
        require_koopman = "revision" in full_root.parts
        full = collect_seed_metrics(full_root, require_koopman=require_koopman)
        ablation = collect_seed_metrics(
            ablation_root / dataset / "hkst",
            required_params=("patience=7",),
        )

        for horizon in (24, 48, 72, 96):
            common_seeds = sorted(
                seed
                for (candidate_horizon, seed) in ablation
                if candidate_horizon == horizon and (horizon, seed) in full
            )
            if not common_seeds:
                continue

            full_mae = [full[(horizon, seed)]["mae"] for seed in common_seeds]
            abl_mae = [
                ablation[(horizon, seed)]["mae"] for seed in common_seeds
            ]
            full_rmse = [full[(horizon, seed)]["rmse"] for seed in common_seeds]
            abl_rmse = [
                ablation[(horizon, seed)]["rmse"] for seed in common_seeds
            ]
            mae_delta = (np.mean(abl_mae) / np.mean(full_mae) - 1.0) * 100
            rmse_delta = (np.mean(abl_rmse) / np.mean(full_rmse) - 1.0) * 100

            print(
                f"{DATASET_LABELS[dataset]:<10} {horizon:>3} {len(common_seeds):>3} "
                f"{mean_std(full_mae):>16} {mean_std(abl_mae):>16} "
                f"{mae_delta:>12.2f}   {mean_std(full_rmse):>16} "
                f"{mean_std(abl_rmse):>16} {rmse_delta:>13.2f}"
            )
            if len(common_seeds) == 3:
                completed_rows += 1
            else:
                partial_rows += 1

    print(
        f"\nComplete three-seed settings: {completed_rows}/28; "
        f"partial settings shown: {partial_rows}."
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--ablation",
        choices=("wokoopman", "wobranchk", "all"),
        default="all",
    )
    args = parser.parse_args()
    names = ABLATIONS if args.ablation == "all" else (args.ablation,)
    for name in names:
        compare(name)


if __name__ == "__main__":
    main()
