"""Build and run the w/o Koopman and w/o branch-specific K ablations."""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


ABLATIONS = {
    "wokoopman": {
        "root": Path("experiment_storage/ablation_wokoopman"),
        "configs": Path("experiments/configs/ablation/wokoopman"),
    },
    "wobranchk": {
        "root": Path("experiment_storage/ablation_wobranchk"),
        "configs": Path("experiments/configs/ablation/wobranchk"),
    },
}
BATCH_PATTERN = re.compile(
    r"^get_data\.batch_size\s*=\s*\d+\s*$", re.MULTILINE
)
FINAL_MARKERS = ("_SUCCESS", "_RUNNING", "_OOM", "_ERROR", "metrics.npy")


def selected_names(ablation: str | None):
    return [ablation] if ablation else list(ABLATIONS)


def instance_configs(ablation: str | None = None):
    configs = []
    for name in selected_names(ablation):
        configs.extend(
            path
            for path in ABLATIONS[name]["root"].glob("**/config.gin")
            if "patience=7" in path.parents[1].name
        )
    return sorted(configs, key=lambda path: path.as_posix())


def build_all(ablation: str | None = None):
    for name in selected_names(ablation):
        config_root = ABLATIONS[name]["configs"]
        for config in sorted(config_root.glob("*.gin")):
            print(f"[BUILD {name}] {config}", flush=True)
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


def status(
    ablation: str | None = None,
    worker_index: int | None = None,
    num_workers: int | None = None,
):
    configs = instance_configs(ablation)
    if worker_index is not None:
        if num_workers is None or worker_index < 1 or worker_index > num_workers:
            raise ValueError(
                "worker-index must be in [1, num-workers] for partition status"
            )
        configs = configs[worker_index - 1::num_workers]
    counts = {
        name: sum((path.parent / marker).is_file() for path in configs)
        for name, marker in (
            ("Success", "_SUCCESS"),
            ("Metrics", "metrics.npy"),
            ("Running", "_RUNNING"),
            ("Claimed", "_CLAIM"),
            ("OOM", "_OOM"),
            ("Error", "_ERROR"),
        )
    }
    pending = sum(
        not any(
            (path.parent / marker).is_file()
            for marker in FINAL_MARKERS + ("_CLAIM",)
        )
        for path in configs
    )
    title = ablation or "w/o Koopman + w/o Branch-specific K"
    if worker_index is not None:
        title += f" worker {worker_index}/{num_workers}"
    print("=" * 72)
    print(f"{title} ablation progress")
    print("=" * 72)
    print(
        f"Config={len(configs)}  Success={counts['Success']}  "
        f"Metrics={counts['Metrics']}  Running={counts['Running']}  "
        f"Claimed={counts['Claimed']}  OOM={counts['OOM']}  "
        f"Error={counts['Error']}  Pending={pending}"
    )


def set_batch_size(config: Path, batch_size: int):
    content = config.read_text(encoding="utf-8")
    updated, count = BATCH_PATTERN.subn(
        f"get_data.batch_size = {batch_size}", content, count=1
    )
    if count != 1:
        raise RuntimeError(f"Cannot locate get_data.batch_size in {config}")
    config.write_text(updated, encoding="utf-8")


def try_claim(config: Path):
    run_dir = config.parent
    if any(
        (run_dir / marker).is_file()
        for marker in FINAL_MARKERS + ("_CLAIM",)
    ):
        return False
    claim = run_dir / "_CLAIM"
    try:
        descriptor = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, "w", encoding="utf-8") as stream:
        stream.write(
            f"host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n"
        )
    return True


def run_one(config: Path):
    run_dir = config.parent
    if (run_dir / "metrics.npy").is_file():
        (run_dir / "_SUCCESS").touch(exist_ok=True)
        return

    for batch_size in (32, 16, 8):
        set_batch_size(config, batch_size)
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
            (run_dir / "_BATCH_SIZE").write_text(
                str(batch_size), encoding="utf-8"
            )
            (run_dir / "_SUCCESS").touch(exist_ok=True)
            print(f"[SUCCESS batch={batch_size}] {run_dir}", flush=True)
            return

        tail = log_path.read_text(
            encoding="utf-8", errors="replace"
        )[-30000:].lower()
        is_oom = "out of memory" in tail or "cuda error: out of memory" in tail
        if not is_oom:
            (run_dir / "_ERROR").write_text(
                f"non-OOM failure at batch={batch_size}; inspect instance.log\n",
                encoding="utf-8",
            )
            print(f"[ERROR] {run_dir}; inspect {log_path}", flush=True)
            return
        print(f"[OOM batch={batch_size}] {run_dir}", flush=True)

    (run_dir / "_OOM").write_text(
        "CUDA OOM at batch sizes 32, 16, and 8\n", encoding="utf-8"
    )
    print(f"[OOM FINAL] {run_dir}", flush=True)


def process(configs):
    for config in configs:
        if not try_claim(config):
            continue
        claim = config.parent / "_CLAIM"
        try:
            run_one(config)
        finally:
            claim.unlink(missing_ok=True)


def run_worker(worker_index: int, num_workers: int, ablation: str | None = None):
    if worker_index < 1 or worker_index > num_workers:
        raise ValueError("worker-index must be in [1, num-workers]")
    configs = instance_configs(ablation)[worker_index - 1::num_workers]
    print(
        f"[WORKER {worker_index}/{num_workers}] assigned {len(configs)} configs",
        flush=True,
    )
    process(configs)
    status(ablation, worker_index, num_workers)


def run_pending(ablation: str | None = None):
    process(instance_configs(ablation))
    status(ablation)


def add_ablation_argument(parser):
    parser.add_argument("--ablation", choices=tuple(ABLATIONS))


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest="command", required=True)
    build_parser = commands.add_parser("build")
    add_ablation_argument(build_parser)
    status_parser = commands.add_parser("status")
    add_ablation_argument(status_parser)
    status_parser.add_argument("--worker-index", type=int)
    status_parser.add_argument("--num-workers", type=int)
    pending_parser = commands.add_parser("run-pending")
    add_ablation_argument(pending_parser)
    run_parser = commands.add_parser("run")
    run_parser.add_argument("--worker-index", type=int, required=True)
    run_parser.add_argument("--num-workers", type=int, default=1)
    add_ablation_argument(run_parser)
    args = parser.parse_args()

    if args.command == "build":
        build_all(args.ablation)
        status(args.ablation)
    elif args.command == "status":
        status(
            args.ablation,
            args.worker_index,
            args.num_workers,
        )
    elif args.command == "run-pending":
        run_pending(args.ablation)
    else:
        run_worker(args.worker_index, args.num_workers, args.ablation)


if __name__ == "__main__":
    main()
