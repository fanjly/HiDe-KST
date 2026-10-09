"""Summarize three-seed residual-mask-ratio sensitivity results."""

from __future__ import annotations

import csv
import re
from collections import defaultdict
from pathlib import Path

import numpy as np


ROOT = Path("experiment_storage/sensitivity_rho")
PARAM_PATTERN = re.compile(r"(?:^|,)recon_mask_ratio=([^,]+)")


def main():
    ROOT.mkdir(parents=True, exist_ok=True)
    values = defaultdict(lambda: defaultdict(list))
    for path in sorted(ROOT.glob("**/metrics.npy")):
        if "patience=7" not in path.parents[1].name:
            continue
        match = PARAM_PATTERN.search(path.parents[1].name)
        if match is None:
            continue
        dataset = path.parents[3].name
        rho = float(match.group(1))
        metrics = np.load(path, allow_pickle=True).item()["test"]
        values[(dataset, rho)]["mae"].append(float(metrics["mae"]))
        values[(dataset, rho)]["rmse"].append(float(metrics["rmse"]))

    rows = []
    print("Dataset       rho   N       MAE mean+/-std      RMSE mean+/-std")
    print("-" * 72)
    for (dataset, rho), metrics in sorted(values.items()):
        mae = np.asarray(metrics["mae"])
        rmse = np.asarray(metrics["rmse"])
        mae_std = mae.std(ddof=1) if len(mae) > 1 else 0.0
        rmse_std = rmse.std(ddof=1) if len(rmse) > 1 else 0.0
        print(
            f"{dataset:12s} {rho:3.1f}  {len(mae):1d}  "
            f"{mae.mean():9.4f}+/-{mae_std:.4f}  "
            f"{rmse.mean():9.4f}+/-{rmse_std:.4f}"
        )
        rows.append(
            [dataset, 96, rho, len(mae), mae.mean(), mae_std, rmse.mean(), rmse_std]
        )

    output = ROOT / "rho_sensitivity_summary.csv"
    with output.open("w", newline="", encoding="utf-8-sig") as stream:
        writer = csv.writer(stream)
        writer.writerow(
            ["dataset", "horizon", "rho", "N", "mae_mean", "mae_std", "rmse_mean", "rmse_std"]
        )
        writer.writerows(rows)
    print(f"\nSaved: {output}")


if __name__ == "__main__":
    main()
