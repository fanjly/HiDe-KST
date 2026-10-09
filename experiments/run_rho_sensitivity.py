"""Build and run the residual-mask-ratio sensitivity experiment."""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
from pathlib import Path


ROOT = Path("experiment_storage/sensitivity_rho")
CONFIG_ROOT = Path("experiments/configs/sensitivity/sensitivity_rho")
BATCH_PATTERN = re.compile(
    r"^get_data\.batch_size\s*=\s*\d+\s*$", re.MULTILINE
)


def configs():
    return sorted(
        (
            path
            for path in ROOT.glob("**/config.gin")
            if "patience=7" in path.parents[1].name
        ),
        key=lambda path: path.as_posix(),
    )


def build():
    for config in sorted(CONFIG_ROOT.glob("*.gin")):
        print(f"[BUILD] {config}", flush=True)
        subprocess.run(
            [
                sys.executable,
                "-m",
                "experiments.forecast",
                f"--config_path={config}",
                "build_experiment",
            ],
            check=True,
        )
    status()


def state_counts():
    counts = {
        "success": 0,
        "metrics": 0,
        "running": 0,
        "claimed": 0,
        "oom": 0,
        "error": 0,
        "pending": 0,
    }
    for config in configs():
        run_dir = config.parent
        counts["success"] += (run_dir / "_SUCCESS").is_file()
        counts["metrics"] += (run_dir / "metrics.npy").is_file()
        counts["running"] += (run_dir / "_RUNNING").is_file()
        counts["claimed"] += (run_dir / "_CLAIM").is_file()
        counts["oom"] += (run_dir / "_OOM").is_file()
        counts["error"] += (run_dir / "_ERROR").is_file()
        terminal = any(
            (run_dir / name).is_file()
            for name in (
                "_SUCCESS", "metrics.npy", "_RUNNING", "_CLAIM", "_OOM", "_ERROR"
            )
        )
        counts["pending"] += not terminal
    return counts


def status():
    counts = state_counts()
    print("=" * 72)
    print("Residual-mask-ratio sensitivity progress")
    print("=" * 72)
    print(
        f"Config={len(configs())}  Success={counts['success']}  "
        f"Metrics={counts['metrics']}  Running={counts['running']}  "
        f"Claimed={counts['claimed']}  OOM={counts['oom']}  "
        f"Error={counts['error']}  Pending={counts['pending']}"
    )


def claim(config):
    run_dir = config.parent
    if any(
        (run_dir / name).is_file()
        for name in ("_SUCCESS", "metrics.npy", "_RUNNING", "_CLAIM", "_OOM", "_ERROR")
    ):
        return False
    path = run_dir / "_CLAIM"
    try:
        descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(f"host={socket.gethostname()} pid={os.getpid()}\n")
    return True


def set_batch(config, batch_size):
    content = config.read_text(encoding="utf-8")
    content, replacements = BATCH_PATTERN.subn(
        f"get_data.batch_size = {batch_size}", content, count=1
    )
    if replacements != 1:
        raise RuntimeError(f"Missing get_data.batch_size in {config}")
    config.write_text(content, encoding="utf-8")


def run_one(config):
    run_dir = config.parent
    for batch_size in (32, 16, 8):
        set_batch(config, batch_size)
        log_path = run_dir / "instance.log"
        print(f"[RUN batch={batch_size}] {run_dir}", flush=True)
        with log_path.open("a", encoding="utf-8") as log:
            result = subprocess.run(
                [
                    sys.executable,
                    "-m",
                    "experiments.forecast",
                    f"--config_path={config}",
                    "run",
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0 and (run_dir / "metrics.npy").is_file():
            (run_dir / "_BATCH_SIZE").write_text(str(batch_size), encoding="utf-8")
            print(f"[SUCCESS batch={batch_size}] {run_dir}", flush=True)
            return
        tail = log_path.read_text(encoding="utf-8", errors="replace")[-20000:].lower()
        if "out of memory" not in tail:
            (run_dir / "_ERROR").write_text(
                f"non-OOM failure at batch={batch_size}; inspect instance.log\n",
                encoding="utf-8",
            )
            print(f"[ERROR] {run_dir}", flush=True)
            return
        print(f"[OOM batch={batch_size}] {run_dir}", flush=True)
    (run_dir / "_OOM").write_text(
        "CUDA OOM at batch sizes 32, 16, and 8\n", encoding="utf-8"
    )


def run_pending():
    processed = 0
    while True:
        config = next((item for item in configs() if claim(item)), None)
        if config is None:
            break
        claim_path = config.parent / "_CLAIM"
        try:
            run_one(config)
            processed += 1
        finally:
            claim_path.unlink(missing_ok=True)
    print(f"[QUEUE EMPTY] processed={processed}", flush=True)
    status()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("command", choices=("build", "run-pending", "status"))
    args = parser.parse_args()
    if args.command == "build":
        build()
    elif args.command == "run-pending":
        run_pending()
    else:
        status()


if __name__ == "__main__":
    main()
