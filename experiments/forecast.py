import os
# os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
import math
import time
import logging
from os.path import join
from typing import Callable, Optional, Union, Dict, Tuple
from tqdm import tqdm

import gin
from fire import Fire
import random
import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import optim
from torch import nn
import pickle

import sys
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from experiments.base import Experiment
from experiments.datasets import ForecastDataset
from models import get_model
from utils.checkpoint import Checkpoint
from utils.ops import default_device, to_tensor, get_memory_stats
from utils.graph_features import calc_gso
from utils.losses import get_loss_fn
from utils.metrics import calc_metrics


class ForecastExperiment(Experiment):
    def fix_seed(self, seed):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False

    @gin.configurable()
    def instance(self,
                 model_type: str,
                 output_len: int,
                 input_len: int = None,
                 seed: int = None,
                 save_vals: Optional[bool] = False,
                 save_pth: Optional[bool] = False,
                 gso_path: Optional[str] = None,
                 gso_type: Optional[str] = 'sym_renorm_adj'):
        self.seed = seed
        if self.seed is not None:
            self.fix_seed(self.seed)

        if input_len is None:  # LSTF setting
            input_len = output_len * 2

        # get data
        logging.info(f"{model_type} start loading dataset")
        train_set, train_loader = get_data(flag='train', input_len=input_len, output_len=output_len)
        val_set, val_loader = get_data(flag='val', existing=train_set)
        test_set, test_loader = get_data(flag='test', existing=train_set)
        logging.info(f"{model_type} finish loading dataset")

        device = default_device()

        # tackle gso
        def _resolve_dataset_path(path: str) -> str:
            if os.path.isabs(path):
                return path
            base = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            return os.path.join(base, 'datasets', path)

        def _load_adj_pkl(path: str) -> np.ndarray:
            with open(path, 'rb') as f:
                obj = pickle.load(f)
            adj = np.array(obj, dtype=np.float32)
            return adj

        gso_tensor = None
        if gso_path is not None:
            gso_path = _resolve_dataset_path(gso_path)
            adj_np = _load_adj_pkl(gso_path)
            gso = calc_gso(adj_np, gso_type)
            gso_tensor = torch.from_numpy(gso.toarray()).to(device)

        # get model
        model = get_model(model_type, input_len=input_len, output_len=output_len, train_loader=train_loader,
                          num_nodes=train_set.data.shape[1], gso=gso_tensor, gso_path=gso_path, device=device).to(device)
        checkpoint = Checkpoint(self.root, save_pth)

        # parameter count (for complexity / training-summary analysis; cheap, always recorded)
        param_count = int(sum(p.numel() for p in model.parameters()))
        trainable_param_count = int(sum(p.numel() for p in model.parameters() if p.requires_grad))

        # train forecasting task
        torch.cuda.empty_cache()
        if device.type == 'cuda':
            torch.cuda.reset_peak_memory_stats(device)
        model, train_stats = train(model, checkpoint, train_loader, val_loader, test_loader)
        if device.type == 'cuda':
            _, _, max_alloc = get_memory_stats(device)
        else:
            max_alloc = 0.0

        # testing
        val_metrics = validate(model, loader=val_loader)
        test_metrics = validate(model, loader=test_loader, save_path=self.root if save_vals else None)

        model_size = os.path.getsize(checkpoint.model_path) if os.path.exists(checkpoint.model_path) else 0
        usage = {
            'avg_epoch_time': train_stats['avg_epoch_time'],
            'model_size': model_size,
            'max_alloc': max_alloc,
            # New fields added during TKDE revision for P1-4 / P3-6:
            'epoch_times': train_stats['epoch_times'],
            'convergence_epoch': train_stats['convergence_epoch'],
            'total_epochs': train_stats['total_epochs'],
            'param_count': param_count,
            'trainable_param_count': trainable_param_count,
        }
        np.save(join(self.root, 'metrics.npy'), {'val': val_metrics,
                                                 'test': test_metrics,
                                                 'usage': usage})

        val_metrics = {f'ValMetric/{k}': v for k, v in val_metrics.items()}
        test_metrics = {f'TestMetric/{k}': v for k, v in test_metrics.items()}
        checkpoint.close({**val_metrics, **test_metrics})


@gin.configurable()
def get_optimizer(model: nn.Module,
                  lr: Optional[float] = 1e-3,
                  lambda_lr: Optional[float] = 1.,
                  weight_decay: Optional[float] = 0.) -> optim.Optimizer:
    group1 = []  # lambda
    group2 = []  # no decay
    group3 = []  # decay
    no_decay_list = ('bias', 'norm',)
    for param_name, param in model.named_parameters():
        if '_lambda' in param_name:
            group1.append(param)
        elif any([mod in param_name for mod in no_decay_list]):
            group2.append(param)
        else:
            group3.append(param)
    optimizer = optim.Adam([
        {'params': group1, 'weight_decay': 0, 'lr': lambda_lr, 'scheduler': 'cosine_annealing'},
        {'params': group2, 'weight_decay': 0, 'scheduler': 'cosine_annealing_with_linear_warmup'},
        {'params': group3, 'scheduler': 'cosine_annealing_with_linear_warmup'},
    ], lr=lr, weight_decay=weight_decay)
    return optimizer


@gin.configurable()
def get_scheduler(optimizer: optim.Optimizer,
                  T_max: int,
                  warmup_epochs: int,
                  eta_min: Optional[float] = 0.) -> optim.lr_scheduler.LambdaLR:
    scheduler_fns = []
    for param_group in optimizer.param_groups:
        scheduler = param_group['scheduler']
        if scheduler == 'none':
            fn = lambda T_cur: 1
        elif scheduler == 'cosine_annealing':
            lr = eta_max = param_group['lr']
            fn = lambda T_cur: (eta_min + 0.5 * (eta_max - eta_min) * (
                1.0 + math.cos((T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
        elif scheduler == 'cosine_annealing_with_linear_warmup':
            lr = eta_max = param_group['lr']
            fn = lambda T_cur: T_cur / warmup_epochs if T_cur < warmup_epochs else (eta_min + 0.5 * (
                eta_max - eta_min) * (1.0 + math.cos(
                    (T_cur - warmup_epochs) / (T_max - warmup_epochs) * math.pi))) / lr
        else:
            raise ValueError(f'No such scheduler, {scheduler}')
        scheduler_fns.append(fn)
    scheduler = optim.lr_scheduler.LambdaLR(optimizer, lr_lambda=scheduler_fns)
    return scheduler


@gin.configurable()
def get_data(flag: str,
             batch_size: int,
             input_len: int = None,
             output_len: int = None,
             existing: ForecastDataset = None,
             num_workers: int = 0,
             pin_memory: bool = True) -> Tuple[ForecastDataset, DataLoader]:
    if flag in ("val", "test"):
        shuffle = False
        drop_last = False
    elif flag == "train":
        shuffle = True
        drop_last = True
    else:
        raise ValueError(f"no such flag {flag}")

    if existing is None and output_len is not None:
        dataset = ForecastDataset(flag, input_len, output_len)
    elif existing is not None:
        dataset = ForecastDataset.from_existing(existing, flag)
    else:
        raise ValueError("either existing or output_len should be provided")

    data_loader = DataLoader(dataset,
                             batch_size=batch_size,
                             shuffle=shuffle,
                             drop_last=drop_last,
                             num_workers=num_workers,
                             pin_memory=pin_memory)
    return dataset, data_loader


@gin.configurable()
def train(model: nn.Module,
          checkpoint: Checkpoint,
          train_loader: DataLoader,
          val_loader: DataLoader,
          test_loader: DataLoader,
          loss_name: str,
          epochs: int,
          gradient_accumulation_steps: int = 1) -> Tuple[nn.Module, Dict[str, Union[float, int, list]]]:
    if gradient_accumulation_steps < 1:
        raise ValueError('gradient_accumulation_steps must be at least 1')
    optimizer = get_optimizer(model)
    scheduler = get_scheduler(optimizer=optimizer, T_max=epochs)
    training_loss_fn = get_loss_fn(loss_name)

    logging.info("start training")
    epoch_times = []
    for epoch in range(epochs):
        start_time = time.time()
        train_loss_sum = torch.zeros((), device=default_device())
        train_batches = 0
        model.train()
        optimizer.zero_grad()
        for it, data in enumerate(train_loader):
            x, y, x_stamp, y_stamp = map(to_tensor, data)
            forecast = model(x, x_stamp, y_stamp)
            losses = [training_loss_fn(forecast, y)]
            if hasattr(model, 'stwave_y_hat_l'):
                y_np = y.detach().cpu().numpy()
                y_inv = train_loader.dataset.inverse_transform(y_np)
                y_inv_expanded = np.expand_dims(y_inv, axis=-1)
                yl_np, _ = model.disentangle(y_inv_expanded)
                yl_np = train_loader.dataset.transform(yl_np[..., 0])
                yl = torch.from_numpy(yl_np).to(y.device).unsqueeze(dim=-1)
                losses.append(training_loss_fn(yl, model.stwave_y_hat_l))
            if hasattr(model, 'get_ortho_loss') and (model.te_ortho_loss or model.sp_ortho_loss):
                losses.append(model.get_ortho_loss())
            if hasattr(model, 'get_residual_signal') and model.recon_mask_ratio > 0:
                res_signal = model.get_residual_signal()
                losses.append(training_loss_fn(res_signal, torch.zeros_like(res_signal)))
            if hasattr(model, 'get_dynamic_graph_loss') and model.use_dynamic_graph_basis:
                losses.append(model.get_dynamic_graph_loss())
            loss = torch.sum(torch.stack(losses))
            (loss / gradient_accumulation_steps).backward()
            should_step = ((it + 1) % gradient_accumulation_steps == 0
                           or it + 1 == len(train_loader))
            if should_step:
                optimizer.step()
                optimizer.zero_grad()

            train_loss_sum = train_loss_sum + loss.detach()
            train_batches += 1
            if (it + 1) % 100 == 0:
                logging.info(f"epochs: {epoch + 1}, iters: {it + 1} | training loss: {loss.item():.2f}")
        scheduler.step()
        epoch_times.append(time.time() - start_time)

        train_loss = (train_loss_sum / max(train_batches, 1)).item()
        val_loss = evaluate_loss(model, loader=val_loader, loss_fn=training_loss_fn)
        test_loss = evaluate_loss(model, loader=test_loader, loss_fn=training_loss_fn)

        scalars = {'Loss/Train': train_loss,
                   'Loss/Val': val_loss,
                   'Loss/Test': test_loss}
        for i, param_group in enumerate(optimizer.param_groups):
            scalars[f'LR/group_{i}'] = param_group['lr']
        checkpoint(epoch + 1, model, scalars=scalars)

        torch.cuda.empty_cache()

        if checkpoint.early_stop:
            logging.info("Early stopping")
            break

    if epochs > 0 and os.path.exists(checkpoint.model_path):
        model.load_state_dict(torch.load(checkpoint.model_path, weights_only=True))

    train_stats = {
        'avg_epoch_time': float(np.mean(epoch_times)) if epoch_times else 0.0,
        'epoch_times': [float(t) for t in epoch_times],
        'convergence_epoch': int(checkpoint.best_epoch),
        'total_epochs': int(len(epoch_times)),
    }
    return model, train_stats


@torch.no_grad()
def evaluate_loss(model: nn.Module,
                  loader: DataLoader,
                  loss_fn: Optional[Callable] = None) -> float:
    """ just evaluate the loss of the model on the given loader
    """
    model.eval()
    total_loss = []
    for it, data in enumerate(loader):
        x, y, x_stamp, y_stamp = map(to_tensor, data)
        forecast = model(x, x_stamp, y_stamp)
        loss = loss_fn(forecast, y, reduction='none')
        total_loss.append(loss)
    total_loss = torch.cat(total_loss, dim=0).cpu()
    return np.average(total_loss)


@torch.no_grad()
def validate(model: nn.Module,
             loader: DataLoader,
             save_path: Optional[str] = None) -> Union[Dict[str, float], float]:
    """ report detailed metrics of the model on the given loader
    """
    model.eval()
    preds = []
    trues = []
    inps = []
    for it, data in enumerate(loader):
        x, y, x_stamp, y_stamp = map(to_tensor, data)
        forecast = model(x, x_stamp, y_stamp)

        if loader.dataset.scale:
            forecast_np = forecast.detach().cpu().numpy()
            x_np = x.detach().cpu().numpy()
            y_np = y.detach().cpu().numpy()
            forecast_inv = loader.dataset.inverse_transform(forecast_np)
            x_inv = loader.dataset.inverse_transform(x_np, round=True)
            y_inv = loader.dataset.inverse_transform(y_np, round=True)
            forecast = torch.from_numpy(forecast_inv).to(forecast.device)
            x = torch.from_numpy(x_inv).to(x.device)
            y = torch.from_numpy(y_inv).to(y.device)

        preds.append(forecast)
        trues.append(y)
        if save_path is not None:
            inps.append(x)

    preds = torch.cat(preds, dim=0).detach().cpu().numpy()
    trues = torch.cat(trues, dim=0).detach().cpu().numpy()
    if save_path is not None:
        inps = torch.cat(inps, dim=0).detach().cpu().numpy()
        np.savez(join(save_path, 'inps.npz'), data=inps)
        np.savez(join(save_path, 'preds.npz'), data=preds)
        np.savez(join(save_path, 'trues.npz'), data=trues)
    metrics = calc_metrics(preds, trues)
    return metrics


if __name__ == '__main__':
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s - %(funcName)s - %(message)s",
        force=True
    )

    # 如果没有传入参数，批量执行command文件
    if len(sys.argv) == 1:
        import glob
        import contextlib

        # 查找所有command文件
        command_files = glob.glob(os.path.join('experiment_storage', 'mine', '**', 'command'), recursive=True)
        print(f"找到 {len(command_files)} 个command文件")

        for i, cmd_file in enumerate(tqdm(command_files)):
            print(f"\n[{i+1}/{len(command_files)}] 执行: {cmd_file}")

            # 读取命令内容
            with open(cmd_file, 'r') as f:
                cmd_content = f.read().strip()

            # 解析命令参数
            parts = cmd_content.split(' --config_path=')
            if len(parts) == 2:
                # config_path = parts[1].split(' run')[0]
                config_path = os.path.join(os.path.dirname(cmd_file), 'config.gin')

                # 获取日志文件路径（和command文件同目录下的instance.log）
                log_file = os.path.join(os.path.dirname(cmd_file), 'instance.log')

                # 重置logging配置，避免文件句柄冲突
                for handler in logging.root.handlers[:]:
                    logging.root.removeHandler(handler)

                # 重定向输出到日志文件
                with open(log_file, 'a', encoding='utf-8') as f:
                    with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                        logging.basicConfig(
                            level=logging.INFO,
                            format="%(asctime)s - %(funcName)s - %(message)s",
                            force=True
                        )
                        # 设置参数并执行
                        sys.argv = ['forecast.py', 'run', f'--config_path={config_path}']
                        try:
                            Fire(ForecastExperiment)
                            print("完成")
                        except Exception as e:
                            print(f"失败: {e}")
            else:
                print(f"命令格式错误: {cmd_content}")
    else:
        Fire(ForecastExperiment)
