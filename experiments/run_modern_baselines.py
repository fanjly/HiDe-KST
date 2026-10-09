"""Build, run, and inspect the SKOLR/TimeMixer baseline queue.

The ``run`` command keeps the original deterministic worker sharding, while
``run-pending`` lets multiple hosts safely drain a shared queue through atomic
claim files.  Training first uses batch size 32; CUDA OOM retries the same run
with 16 and then 8.  A failure at 8 is recorded as ``_OOM`` and is not retried
automatically.
"""

from __future__ import annotations

import argparse
import os
import re
import socket
import subprocess
import sys
import time
from pathlib import Path


ROOT = Path('experiment_storage/modern_baselines')
CONFIG_ROOT = Path('experiments/configs/modern_baselines')
BATCH_PATTERN = re.compile(r'^get_data\.batch_size\s*=\s*\d+\s*$', re.MULTILINE)


def instance_configs():
    return sorted(ROOT.glob('**/config.gin'), key=lambda path: path.as_posix())


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


def status():
    configs = instance_configs()
    success = sum((path.parent / '_SUCCESS').is_file() for path in configs)
    running = sum((path.parent / '_RUNNING').is_file() for path in configs)
    claimed = sum((path.parent / '_CLAIM').is_file() for path in configs)
    oom = sum((path.parent / '_OOM').is_file() for path in configs)
    error = sum((path.parent / '_ERROR').is_file() for path in configs)
    metrics = sum((path.parent / 'metrics.npy').is_file() for path in configs)
    pending = sum(
        not any((path.parent / marker).is_file() for marker in (
            '_SUCCESS', '_RUNNING', '_CLAIM', '_OOM', '_ERROR', 'metrics.npy'
        ))
        for path in configs
    )
    print('=' * 72)
    print('SKOLR + TimeMixer baseline progress')
    print('=' * 72)
    print(
        f'Config={len(configs)}  Success={success}  Metrics={metrics}  '
        f'Running={running}  Claimed={claimed}  OOM={oom}  Error={error}  '
        f'Pending={pending}'
    )


def set_batch_size(config: Path, batch_size: int):
    content = config.read_text(encoding='utf-8')
    updated, count = BATCH_PATTERN.subn(
        f'get_data.batch_size = {batch_size}', content, count=1
    )
    if count != 1:
        raise RuntimeError(f'Cannot locate get_data.batch_size in {config}')
    config.write_text(updated, encoding='utf-8')


def run_one(config: Path):
    run_dir = config.parent
    if (run_dir / 'metrics.npy').is_file():
        (run_dir / '_SUCCESS').touch(exist_ok=True)
        return
    if any((run_dir / marker).is_file() for marker in (
        '_SUCCESS', '_RUNNING', '_OOM', '_ERROR'
    )):
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
            print(f'[SUCCESS batch={batch_size}] {run_dir}', flush=True)
            return
        tail = log_path.read_text(encoding='utf-8', errors='replace')[-20000:].lower()
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


def run_worker(worker_index: int, num_workers: int):
    if worker_index < 1 or worker_index > num_workers:
        raise ValueError('worker-index must be in [1, num-workers]')
    configs = instance_configs()[worker_index - 1::num_workers]
    print(
        f'[WORKER {worker_index}/{num_workers}] assigned {len(configs)} configs',
        flush=True,
    )
    for config in configs:
        if not try_claim(config):
            continue
        claim = config.parent / '_CLAIM'
        try:
            run_one(config)
        finally:
            claim.unlink(missing_ok=True)
    status()


def try_claim(config: Path):
    run_dir = config.parent
    if any((run_dir / marker).is_file() for marker in (
        '_SUCCESS', '_RUNNING', '_CLAIM', '_OOM', '_ERROR', 'metrics.npy'
    )):
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


def run_pending():
    completed = 0
    while True:
        claimed_config = None
        for config in instance_configs():
            if try_claim(config):
                claimed_config = config
                break
        if claimed_config is None:
            break
        claim = claimed_config.parent / '_CLAIM'
        try:
            run_one(claimed_config)
            completed += 1
        finally:
            claim.unlink(missing_ok=True)
    print(f'[DRAIN COMPLETE] processed {completed} claimed configs', flush=True)
    status()


def main():
    parser = argparse.ArgumentParser()
    subparsers = parser.add_subparsers(dest='command', required=True)
    subparsers.add_parser('build')
    subparsers.add_parser('status')
    subparsers.add_parser('run-pending')
    run_parser = subparsers.add_parser('run')
    run_parser.add_argument('--worker-index', type=int, required=True)
    run_parser.add_argument('--num-workers', type=int, default=4)
    args = parser.parse_args()

    if args.command == 'build':
        build_all()
        status()
    elif args.command == 'status':
        status()
    elif args.command == 'run-pending':
        run_pending()
    else:
        run_worker(args.worker_index, args.num_workers)


if __name__ == '__main__':
    main()
