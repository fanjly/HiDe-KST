import glob
import os
import re
import pandas as pd
import numpy as np
from pathlib import Path
from collections import defaultdict

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def format_mean_std(values):
    if len(values) == 0:
        return 'N/A'
    mean_val = np.mean(values)
    if len(values) > 1:
        std_val = np.std(values, ddof=1)
        return f'{mean_val:.2f}(±{std_val:.2f})'
    else:
        return f'{mean_val:.2f}'


FULL_RESULT_ROOTS = {
    'PEMS03': Path('experiment_storage/comp/PEMS03/hkst'),
    'PEMS04': Path('experiment_storage/revision/PEMS04/hkst'),
    'PEMS07': Path('experiment_storage/comp/PEMS07/hkst'),
    'PEMS08': Path('experiment_storage/revision/PEMS08/hkst'),
    'air_yrdO3': Path('experiment_storage/comp/air_yrdO3/hkst'),
    'air_yrdPM': Path('experiment_storage/comp/air_yrdPM/hkst'),
    'water_mn': Path('experiment_storage/revision/water_mn/hkst'),
}


def collect_seed_metrics(root, require_koopman=False, required_params=()):
    results = {}
    for metric_file in root.glob('**/metrics.npy'):
        relative_parts = metric_file.relative_to(root).parts
        if len(relative_parts) < 3:
            continue

        params_dir, seed_dir = relative_parts[-3], relative_parts[-2]
        if require_koopman and 'evolution_mode=koopman' not in params_dir:
            continue
        if any(value not in params_dir for value in required_params):
            continue

        output_match = re.search(r'output_len=(\d+)', params_dir)
        seed_match = re.search(r'seed=(\d+)', seed_dir)
        if output_match is None or seed_match is None:
            continue

        metrics = np.load(metric_file, allow_pickle=True).item()['test']
        results[(int(output_match.group(1)), int(seed_match.group(1)))] = {
            'mae': float(metrics['mae']),
            'rmse': float(metrics['rmse']),
            'mape': float(metrics['mape']) * 100,
        }
    return results


def make_paper_ablation_table(detail):
    dataset_labels = {
        'PEMS03': 'PEMS03',
        'PEMS04': 'PEMS04',
        'PEMS07': 'PEMS07',
        'PEMS08': 'PEMS08',
        'air_yrdO3': 'YRD-O3',
        'air_yrdPM': 'YRD-PM2.5',
        'water_mn': 'Water',
    }
    columns = pd.MultiIndex.from_product(
        [dataset_labels.values(), ('MAE', 'RMSE')],
        names=['Dataset', 'Metric'])
    index = pd.MultiIndex.from_product(
        [('w/o Cross-scale Gating', 'HiDe-KST'), (24, 48, 72, 96)],
        names=['Method', 'Horizon'])
    paper_table = pd.DataFrame(index=index, columns=columns, dtype=float)

    for _, row in detail.iterrows():
        dataset = dataset_labels[row['Dataset']]
        horizon = int(row['Horizon'])
        paper_table.loc[('w/o Cross-scale Gating', horizon),
                        (dataset, 'MAE')] = row['WoGating_MAE']
        paper_table.loc[('w/o Cross-scale Gating', horizon),
                        (dataset, 'RMSE')] = row['WoGating_RMSE']
        paper_table.loc[('HiDe-KST', horizon),
                        (dataset, 'MAE')] = row['Full_MAE']
        paper_table.loc[('HiDe-KST', horizon),
                        (dataset, 'RMSE')] = row['Full_RMSE']

    return paper_table


def make_wogating_latex_rows(detail):
    dataset_order = list(FULL_RESULT_ROOTS)
    lines = []
    for row_index, horizon in enumerate((24, 48, 72, 96)):
        values = []
        for dataset in dataset_order:
            selected = detail[
                (detail['Dataset'] == dataset)
                & (detail['Horizon'] == horizon)]
            if selected.empty:
                values.extend(['--', '--'])
                continue
            result = selected.iloc[0]
            decimals = 3 if dataset == 'water_mn' else 2
            values.extend([
                f"{result['WoGating_MAE']:.{decimals}f}",
                f"{result['WoGating_RMSE']:.{decimals}f}",
            ])

        if row_index == 0:
            prefix = (r'\multirow{4}{*}{\shortstack{w/o Cross-scale\\'
                      r'Gating}} ' + '\n' + f'& {horizon}')
        else:
            prefix = f'& {horizon}'
        lines.append(prefix + ' & ' + ' & '.join(values) + r' \\')
    return '\n'.join(lines) + '\n'


def summarize_wogating_vs_full():
    downloaded_root = Path('controlled_ablation/wogating')
    storage_root = Path(
        'experiment_storage/controlled_ablation/wogating')
    ablation_root = (downloaded_root if downloaded_root.exists()
                     else storage_root)
    output_file = ablation_root / 'wogating_vs_full.xlsx'
    horizons = (24, 48, 72, 96)
    rows = []
    missing = []

    for dataset, full_root in FULL_RESULT_ROOTS.items():
        full = collect_seed_metrics(full_root, require_koopman=(
            'revision' in full_root.parts))
        wogating = collect_seed_metrics(
            ablation_root / dataset / 'hkst')

        for horizon in horizons:
            full_seeds = {seed for h, seed in full if h == horizon}
            ablation_seeds = {
                seed for h, seed in wogating if h == horizon}
            common_seeds = sorted(full_seeds & ablation_seeds)

            if len(common_seeds) != 3:
                missing.append({
                    'Dataset': dataset,
                    'Horizon': horizon,
                    'Full seeds': sorted(full_seeds),
                    'WoGating seeds': sorted(ablation_seeds),
                    'Common seeds': common_seeds,
                })
                continue

            row = {
                'Dataset': dataset,
                'Horizon': horizon,
                'N': len(common_seeds),
                'Seeds': ','.join(map(str, common_seeds)),
                'Full source': str(full_root),
            }

            for metric in ('mae', 'rmse', 'mape'):
                full_values = np.array([
                    full[(horizon, seed)][metric]
                    for seed in common_seeds])
                ablation_values = np.array([
                    wogating[(horizon, seed)][metric]
                    for seed in common_seeds])
                full_mean = full_values.mean()
                ablation_mean = ablation_values.mean()
                delta = ablation_mean - full_mean
                label = metric.upper()

                row[f'Full_{label}'] = full_mean
                row[f'Full_{label}_STD'] = full_values.std(ddof=1)
                row[f'WoGating_{label}'] = ablation_mean
                row[f'WoGating_{label}_STD'] = ablation_values.std(ddof=1)
                row[f'{label}_Delta'] = delta
                row[f'{label}_Improve(%)'] = (
                    delta / ablation_mean * 100)

            rows.append(row)

    detail = pd.DataFrame(rows)
    summary_rows = []
    for dataset in FULL_RESULT_ROOTS:
        selected = detail[detail['Dataset'] == dataset]
        if selected.empty:
            continue
        summary_rows.append({
            'Dataset': dataset,
            'Horizons': len(selected),
            'Full_Avg_MAE': selected['Full_MAE'].mean(),
            'WoGating_Avg_MAE': selected['WoGating_MAE'].mean(),
            'MAE_Delta': selected['WoGating_MAE'].mean()
                         - selected['Full_MAE'].mean(),
            'MAE_Improve(%)': selected['MAE_Improve(%)'].mean(),
            'Full_Wins': int((selected['MAE_Delta'] > 0).sum()),
            'WoGating_Wins': int((selected['MAE_Delta'] < 0).sum()),
            'Ties': int(np.isclose(selected['MAE_Delta'], 0).sum()),
        })
    summary = pd.DataFrame(summary_rows)
    paper_table = make_paper_ablation_table(detail)
    latex_rows = make_wogating_latex_rows(detail)

    overall = pd.DataFrame([{
        'Compared cases': len(detail),
        'Expected cases': len(FULL_RESULT_ROOTS) * len(horizons),
        'Full_Avg_MAE': detail['Full_MAE'].mean(),
        'WoGating_Avg_MAE': detail['WoGating_MAE'].mean(),
        'MAE_Delta': detail['WoGating_MAE'].mean()
                     - detail['Full_MAE'].mean(),
        'MAE_Improve(%)': detail['MAE_Improve(%)'].mean(),
        'Full_Wins': int((detail['MAE_Delta'] > 0).sum()),
        'WoGating_Wins': int((detail['MAE_Delta'] < 0).sum()),
        'Ties': int(np.isclose(detail['MAE_Delta'], 0).sum()),
    }])
    missing_frame = pd.DataFrame(missing)

    output_file.parent.mkdir(parents=True, exist_ok=True)
    with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
        detail.to_excel(writer, sheet_name='Detailed', index=False)
        summary.to_excel(writer, sheet_name='Dataset_Summary', index=False)
        overall.to_excel(writer, sheet_name='Overall', index=False)
        missing_frame.to_excel(writer, sheet_name='Missing', index=False)
        paper_table.to_excel(writer, sheet_name='Paper_Table')

    latex_file = ablation_root / 'wogating_table_rows.tex'
    latex_file.write_text(latex_rows, encoding='utf-8')

    print('\n完整 HiDe-KST 与 w/o Cross-scale Gating 对比')
    print('Delta = w/o Gating - Full；正数表示使用 Gating 更好\n')
    print(detail.to_string(index=False))
    print('\n按数据集汇总')
    print(summary.to_string(index=False))
    print('\n总体汇总')
    print(overall.to_string(index=False))
    print('\n论文 Table II 格式（新增消融与完整模型）')
    print(paper_table.to_string())
    print('\n可插入 Table II 的 w/o Cross-scale Gating 行')
    print(latex_rows)
    if missing:
        print('\n未完整配对的实验：')
        print(missing_frame.to_string(index=False))
    else:
        print('\n完整性检查通过：28组均包含3个随机种子。')
    print(f'\n结果已保存到：{output_file}')
    print(f'LaTeX 行已保存到：{latex_file}')


if __name__ == '__main__':
    exp_name = 'comp'
    if len(sys.argv) == 2:
        exp_name = sys.argv[1]
    else:
        exp_name = 'comp'

    if exp_name == 'wogating_vs_full':
        summarize_wogating_vs_full()
        sys.exit(0)

    metric_files = glob.glob(os.path.join('experiment_storage', exp_name, '**', 'metrics.npy'), recursive=True)
    if not metric_files and '_' in exp_name:
        metric_files = glob.glob(os.path.join('experiment_storage', exp_name.split('_')[0], exp_name, '**', 'metrics.npy'), recursive=True)
    print(f"找到 {len(metric_files)} 个metrics.npy文件")

    results = defaultdict(
        lambda: defaultdict(
            lambda: defaultdict(
                lambda: defaultdict(list))))

    models = set()

    for f in metric_files:
        path_parts = list(Path(f).parts)

        # 解析路径: ['experiment_storage', 'comp', 'PEMS04', 'koopa', 'output_len=36', 'seed=42', 'metrics.npy']
        if exp_name in ('comp', 'old', 'mine', 'DEPRECATEDcomp'):
            dataset = path_parts[2]
            model = path_parts[3]
            output_len = int(path_parts[4].split('=')[1])
            seed = path_parts[5]

        # 解析路径: ['experiment_storage', 'revision', 'PEMS04', 'hkst',
        #            'output_len=24,evolution_mode=koopman', 'seed=42', 'metrics.npy']
        # 或:       ['experiment_storage', 'ablation_wokoopman', 'PEMS04', 'hkst',
        #            'output_len=24,evolution_mode=none', 'seed=42', 'metrics.npy']
        # 与 IJCAI 阶段 comp/mine 的差别仅在于 param_str 含逗号分隔的多值
        # (output_len + evolution_mode)。把 evolution_mode 拼到 model 名里
        # 方便在同一张汇总表里横向对比 hkst_koopman vs hkst_none。
        elif exp_name in ('revision', 'ablation_wokoopman'):
            dataset = path_parts[2]
            model = path_parts[3]
            params_str = path_parts[4]
            params = {}
            for item in params_str.split(','):
                if '=' in item:
                    k, v = item.split('=')
                    params[k] = v
            output_len = int(params.get('output_len', 0))
            if 'evolution_mode' in params:
                model = f"{model}_{params['evolution_mode']}"
            seed = path_parts[5]

        # 解析路径: ['experiment_storage', 'search', 'PEMS08', 'hkst',
        #           'output_len=48,use_graph=True,spat_decomp=False,
        #            spat_res=False,temp_res_ln=False,use_node_modem=True,
        #            use_timestamp=True', 'seed=42', 'metrics.npy']
        elif exp_name in ('search', 'adjustment', 'ablation'):
            dataset = path_parts[2]
            params = {}
            model = ''
            for item in path_parts[4].split(','):
                if '=' in item:
                    key, val = item.split('=')
                    params[key] = val
                    if val == 'True':
                        model += '1'
                    elif val == 'False':
                        model += '0'
                    elif key != 'output_len':
                        model += item
            if 'output_len' in params:
                output_len = int(params['output_len'])
            else:
                output_len = 0
                print(f"Warning: output_len not found in {path_parts[4]}")

            seed = path_parts[5]

        # 解析路径: ['experiment_storage', 'sensitivity', 'sensitivity_t', 'PEMS07',
        #           'hkst', 'output_len=96,temp_rank=256', 'seed=42', 'metrics.npy']
        elif exp_name in ('sensitivity_b', 'sensitivity_p'):
            dataset = path_parts[2]
            model = path_parts[3]
            params_str = path_parts[4]
            params = {}
            for item in params_str.split(','):
                if '=' in item:
                    k, v = item.split('=')
                    params[k] = v
            output_len = int(params.get('output_len', 0))
            if 'patch_sizes' in params:
                model = params['patch_sizes']
            seed = path_parts[5]

        elif '_' in exp_name:
            dataset = path_parts[3]
            model = eval(path_parts[5].split('=')[-1])
            output_len = int(path_parts[5].split(',')[0].split('=')[1])
            seed = path_parts[6]

        # 解析路径: ['experiment_storage', 'other', 'PEMS07', 'hkst',
        #           'output_len=96,attr=0.?', 'seed=42', 'metrics.npy']
        else:
            dataset = path_parts[2]
            model = path_parts[4].split('=')[-1]
            output_len = int(path_parts[4].split(',')[0].split('=')[1])
            seed = path_parts[5]

        models.add(model)

        # 读取metrics
        metrics = np.load(f, allow_pickle=True).item()
        test_metrics = metrics['test']

        # 存储MAE, RMSE, MAPE, NSE
        results[dataset][model][output_len]['mae'].append(test_metrics['mae'])
        results[dataset][model][output_len]['rmse'].append(test_metrics['rmse'])
        results[dataset][model][output_len]['mape'].append(test_metrics['mape'] * 100)
        results[dataset][model][output_len]['nse'].append(test_metrics['nse'])

        if 'usage' in metrics:
            results[dataset][model][output_len]['time'].append(metrics['usage']['avg_epoch_time'])
            results[dataset][model][output_len]['size'].append(metrics['usage']['model_size'] / (1024 * 1024))

    datasets = sorted(results.keys())
    models = sorted(list(models))

    print('datasets: ', datasets)
    print('models: ', models)

    # 创建多层索引的DataFrame
    index_data = []
    table_data = []

    for dataset in datasets:
        for model in models:
            for output_len in sorted(results[dataset][model].keys()):
                index_data.append((dataset, model, output_len))
                mae_str = format_mean_std(results[dataset][model][output_len]['mae'])
                rmse_str = format_mean_std(results[dataset][model][output_len]['rmse'])
                mape_str = format_mean_std(results[dataset][model][output_len]['mape'])
                nse_str = format_mean_std(results[dataset][model][output_len]['nse'])

                time_str = format_mean_std(results[dataset][model][output_len]['time'])
                size_str = format_mean_std(results[dataset][model][output_len]['size'])

                table_data.append([mae_str, rmse_str, mape_str, nse_str, time_str, size_str])

    # 创建多层索引
    multi_index = pd.MultiIndex.from_tuples(index_data, names=['Dataset', 'Model', 'T'])

    # 创建DataFrame
    df_advanced = pd.DataFrame(table_data,
                               index=multi_index,
                               columns=['MAE', 'RMSE', 'MAPE', 'NSE', 'Time(s)', 'Size(MB)'])

    print('\n表格格式 (数据集-模型-输出长度 vs 指标):')
    print(df_advanced.to_string())

    # df_advanced = df_advanced.unstack('T')
    # df_advanced = df_advanced.sort_index(axis=1, level=0)
    # print('\n宽表格式 (数据集-模型 vs 输出长度-指标):')
    # print(df_advanced.to_string())

    # 保存Excel
    # output_file = os.path.join('experiment_storage', f'experiment_results_{exp_name}.xlsx')
    # with pd.ExcelWriter(output_file, engine='openpyxl') as writer:
    #     df_advanced.to_excel(writer, sheet_name='Advanced_Format')

    # output_file = os.path.join('experiment_storage', f'experiment_results_{exp_name}.csv')
    # df_advanced.to_csv(output_file, index=False)

    # print(f'\n结果已保存到: {output_file}')
