"""Build and run the non-overlapping HiDe-KST dynamic-graph v1 jobs."""

import argparse
import subprocess
import sys
from pathlib import Path


EXPERIMENT_ROOT = Path('experiment_storage/dynamic_graph_v1')
CONFIG_ROOT = Path('experiments/configs/hkst')
CONFIG_NAMES = (
    'hkst_PEMS03.gin',
    'hkst_PEMS04.gin',
    'hkst_PEMS07.gin',
    'hkst_PEMS08.gin',
    'hkst_yrd.gin',
    'hkst_mn.gin',
)
SERVER_IDS = (1, 2, 3, 4)


def run_command(config_path: Path, action: str) -> None:
    command = [sys.executable, '-m', 'experiments.forecast',
               f'--config_path={config_path}', action]
    print(f"[{action.upper()}] {config_path}", flush=True)
    subprocess.run(command, check=True)


def build_configs() -> None:
    for name in CONFIG_NAMES:
        run_command(CONFIG_ROOT / name, 'build_experiment')


def assigned_configs(server: int):
    configs = sorted(EXPERIMENT_ROOT.glob('**/config.gin'))
    return configs[server - 1::len(SERVER_IDS)]


def pending_configs(server: int):
    for config_path in assigned_configs(server):
        run_root = config_path.parent
        if (run_root / '_SUCCESS').exists() or (run_root / '_RUNNING').exists():
            continue
        yield config_path


def print_status(server: int) -> None:
    configs = assigned_configs(server)
    success = sum((path.parent / '_SUCCESS').exists() for path in configs)
    running = sum((path.parent / '_RUNNING').exists() for path in configs)
    failed = sum((path.parent / '_FAILED').exists() for path in configs)
    pending = len(configs) - success - running
    print(f'SERVER{server} total={len(configs)} success={success} '
          f'running={running} failed={failed} pending={pending}')


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument('--server', type=int, choices=SERVER_IDS, required=True)
    parser.add_argument('--build', action='store_true', help='build all configs before running')
    parser.add_argument('--build-only', action='store_true', help='build all configs and exit')
    parser.add_argument('--status', action='store_true', help='only print progress')
    args = parser.parse_args()

    if args.build or args.build_only:
        build_configs()
    if args.build_only:
        print_status(args.server)
        return
    if args.status:
        print_status(args.server)
        return

    jobs = list(pending_configs(args.server))
    print(f'[SERVER{args.server}] pending jobs: {len(jobs)}', flush=True)
    for index, config_path in enumerate(jobs, start=1):
        print(f'[SERVER{args.server}] job {index}/{len(jobs)}', flush=True)
        run_command(config_path, 'run')
    print_status(args.server)


if __name__ == '__main__':
    main()
