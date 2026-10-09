"""Build and run the ModernTCN/STID baseline queue on four servers."""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('experiment_storage/moderntcn_stid')
CONFIG_ROOT = Path('experiments/configs/moderntcn_stid')
BATCH_PATTERN = re.compile(r'^get_data\.batch_size\s*=\s*\d+\s*$', re.MULTILINE)
FINAL_MARKERS = ('_SUCCESS', '_RUNNING', '_OOM', '_ERROR', 'metrics.npy')


def instance_configs(model: str | None = None):
    configs = sorted(
        (
            path
            for path in ROOT.glob('**/config.gin')
            if 'patience=7' in path.parents[1].name
        ),
        key=lambda path: path.as_posix(),
    )
    if model is not None:
        configs = [path for path in configs if path.parents[2].name == model]
    return configs


def build_all():
    for config in sorted(CONFIG_ROOT.glob('*.gin')):
        print(f'[BUILD] {config}', flush=True)
        subprocess.run(
            [
                sys.executable,
                '-m',
                'experiments.forecast',
                f'--config_path={config}',
                'build_experiment',
            ],
            check=True,
        )


def status(model: str | None = None):
    configs = instance_configs(model)
    counts = {
        name: sum((path.parent / marker).is_file() for path in configs)
        for name, marker in (
            ('Success', '_SUCCESS'),
            ('Metrics', 'metrics.npy'),
            ('Running', '_RUNNING'),
            ('Claimed', '_CLAIM'),
            ('OOM', '_OOM'),
            ('Error', '_ERROR'),
        )
    }
    pending = sum(
        not any((path.parent / marker).is_file() for marker in FINAL_MARKERS + ('_CLAIM',))
        for path in configs
    )
    print('=' * 72)
    title = f'{model} baseline progress' if model else 'ModernTCN + STID baseline progress'
    print(title)
    print('=' * 72)
    print(
        f"Config={len(configs)}  Success={counts['Success']}  "
        f"Metrics={counts['Metrics']}  Running={counts['Running']}  "
        f"Claimed={counts['Claimed']}  OOM={counts['OOM']}  "
        f"Error={counts['Error']}  Pending={pending}"
    )


def set_batch_size(config: Path, batch_size: int):
    content = config.read_text(encoding='utf-8')
    updated, count = BATCH_PATTERN.subn(
        f'get_data.batch_size = {batch_size}', content, count=1
    )
    if count != 1:
        raise RuntimeError(f'Cannot locate get_data.batch_size in {config}')
    config.write_text(updated, encoding='utf-8')


def try_claim(config: Path):
    run_dir = config.parent
    if any((run_dir / marker).is_file() for marker in FINAL_MARKERS + ('_CLAIM',)):
        return False
    claim = run_dir / '_CLAIM'
    try:
        descriptor = os.open(str(claim), os.O_CREAT | os.O_EXCL | os.O_WRONLY)
    except FileExistsError:
        return False
    with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
        stream.write(
            f'host={socket.gethostname()} pid={os.getpid()} time={time.time()}\n'
        )
    return True


def run_one(config: Path):
    run_dir = config.parent
    if (run_dir / 'metrics.npy').is_file():
        (run_dir / '_SUCCESS').touch(exist_ok=True)
        return
    if any((run_dir / marker).is_file() for marker in FINAL_MARKERS):
        return

    for batch_size in (32, 16, 8):
        set_batch_size(config, batch_size)
        log_path = run_dir / 'instance.log'
        print(f'[RUN batch={batch_size}] {run_dir}', flush=True)
        with log_path.open('a', encoding='utf-8') as log:
            result = subprocess.run(
                [
                    sys.executable,
                    '-m',
                    'experiments.forecast',
                    f'--config_path={config}',
                    'run',
                ],
                stdout=log,
                stderr=subprocess.STDOUT,
            )
        if result.returncode == 0 and (run_dir / 'metrics.npy').is_file():
            (run_dir / '_BATCH_SIZE').write_text(str(batch_size), encoding='utf-8')
            (run_dir / '_SUCCESS').touch(exist_ok=True)
            print(f'[SUCCESS batch={batch_size}] {run_dir}', flush=True)
            return

        tail = log_path.read_text(encoding='utf-8', errors='replace')[-30000:].lower()
        is_oom = 'out of memory' in tail or 'cuda error: out of memory' in tail
        if not is_oom:
            (run_dir / '_ERROR').write_text(
                f'non-OOM failure at batch={batch_size}; inspect instance.log\n',
                encoding='utf-8',
            )
            print(f'[ERROR] {run_dir}; inspect {log_path}', flush=True)
            return
        print(f'[OOM batch={batch_size}] {run_dir}', flush=True)

    (run_dir / '_OOM').write_text(
        'CUDA OOM at batch sizes 32, 16, and 8\n', encoding='utf-8'
    )
    print(f'[OOM FINAL] {run_dir}', flush=True)


def process(configs):
    for config in configs:
        if not try_claim(config):
            continue
        claim = config.parent / '_CLAIM'
        try:
            run_one(config)
        finally:
            claim.unlink(missing_ok=True)


def run_worker(worker_index: int, num_workers: int, model: str | None = None):
    if worker_index < 1 or worker_index > num_workers:
        raise ValueError('worker-index must be in [1, num-workers]')
    configs = instance_configs(model)[worker_index - 1::num_workers]
    print(
        f'[WORKER {worker_index}/{num_workers}] assigned {len(configs)} configs',
        flush=True,
    )
    process(configs)
    status(model)


def run_pending(model: str | None = None):
    completed = 0
    while True:
        claimed = next((c for c in instance_configs(model) if try_claim(c)), None)
        if claimed is None:
            break
        claim = claimed.parent / '_CLAIM'
        try:
            run_one(claimed)
            completed += 1
        finally:
            claim.unlink(missing_ok=True)
    print(f'[DRAIN COMPLETE] processed {completed} configs', flush=True)
    status(model)


def main():
    parser = argparse.ArgumentParser()
    commands = parser.add_subparsers(dest='command', required=True)
    commands.add_parser('build')
    status_parser = commands.add_parser('status')
    status_parser.add_argument('--model', choices=('moderntcn', 'stid'))
    pending_parser = commands.add_parser('run-pending')
    pending_parser.add_argument('--model', choices=('moderntcn', 'stid'))
    run_parser = commands.add_parser('run')
    run_parser.add_argument('--worker-index', type=int, required=True)
    run_parser.add_argument('--num-workers', type=int, default=4)
    run_parser.add_argument('--model', choices=('moderntcn', 'stid'))
    args = parser.parse_args()

    if args.command == 'build':
        build_all()
        status()
    elif args.command == 'status':
        status(args.model)
    elif args.command == 'run-pending':
        run_pending(args.model)
    else:
        run_worker(args.worker_index, args.num_workers, args.model)


if __name__ == '__main__':
    main()
