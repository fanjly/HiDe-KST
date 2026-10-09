"""Summarize H=96 ModernTCN efficiency metrics used by Fig. 4."""

from pathlib import Path

import numpy as np


ROOT = Path("experiment_storage/moderntcn_stid")
DATASETS = {
    "PEMS08": "PEMS08",
    "air_yrdPM": "YRD-PM2.5",
}


def mean_std(values):
    array = np.asarray(values, dtype=float)
    std = array.std(ddof=1) if len(array) > 1 else 0.0
    return float(array.mean()), float(std)


def main():
    for folder, label in DATASETS.items():
        paths = sorted(
            path
            for path in (ROOT / folder / "moderntcn").glob(
                "output_len=96,*/seed=*/metrics.npy"
            )
            if "patience=7" in path.parents[1].name
        )
        print(f"\n{label}: {len(paths)} seed files")
        values = {"mae": [], "time": [], "memory": []}
        for path in paths:
            payload = np.load(path, allow_pickle=True).item()
            usage = payload.get("usage", {})
            mae = float(payload["test"]["mae"])
            epoch_time = float(usage["avg_epoch_time"])
            memory = usage.get("max_alloc")
            if memory is None:
                raise KeyError(f"Missing usage.max_alloc in {path}")
            memory = float(memory)
            seed = path.parent.name.split("=", 1)[-1]
            values["mae"].append(mae)
            values["time"].append(epoch_time)
            values["memory"].append(memory)
            print(
                f"  seed={seed:<3} MAE={mae:.4f}  "
                f"time/epoch={epoch_time:.2f}s  peak={memory:.3f}GB"
            )

        if not paths:
            print("  No completed H=96 metrics found.")
            continue
        print("  Three-seed summary:")
        for key, unit in (("mae", ""), ("time", "s"), ("memory", "GB")):
            mean, std = mean_std(values[key])
            print(f"    {key:<6} {mean:.4f} +/- {std:.4f}{unit}")


if __name__ == "__main__":
    main()
