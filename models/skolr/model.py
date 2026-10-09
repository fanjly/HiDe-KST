"""SKOLR adapter for the forecasting interface used by this project.

This is an in-framework implementation of the architecture described in
Zhang et al., "SKOLR: Structured Koopman Operator Linear RNN for Time-Series
Forecasting" (ICML 2025).  It follows the paper's channel-independent
normalization, learnable soft frequency decomposition, non-overlapping patch
tokenization, branch-specific MLP measurement functions, and full-matrix
linear RNN evolution.  The public SKOLR repository does not currently expose
the ``KoopBlock`` module referenced by its traffic script, so this adapter is
based on the paper equations rather than presented as an unmodified upstream
source file.
"""

import math

import gin
import torch
from torch import nn


def _mlp(in_dim, hidden_dim, out_dim, hidden_layers, dropout):
    layers = []
    current = in_dim
    for _ in range(hidden_layers):
        layers.extend([
            nn.Linear(current, hidden_dim),
            nn.ReLU(),
            nn.Dropout(dropout),
        ])
        current = hidden_dim
    layers.append(nn.Linear(current, out_dim))
    return nn.Sequential(*layers)


class KoopmanBranch(nn.Module):
    def __init__(self, patch_len, dynamic_dim, hidden_dim, hidden_layers,
                 dropout):
        super().__init__()
        self.encoder = _mlp(
            patch_len, hidden_dim, dynamic_dim, hidden_layers, dropout
        )
        self.decoder = _mlp(
            dynamic_dim, hidden_dim, patch_len, hidden_layers, dropout
        )
        self.transition = nn.Parameter(torch.empty(dynamic_dim, dynamic_dim))
        nn.init.orthogonal_(self.transition)
        with torch.no_grad():
            self.transition.mul_(0.9)

    def forward(self, patches, future_patches):
        # patches: [batch * channels, input_patches, patch_len]
        encoded = self.encoder(patches)
        state = torch.zeros_like(encoded[:, 0])
        for step in range(encoded.shape[1]):
            state = torch.nn.functional.linear(state, self.transition) + encoded[:, step]

        predictions = []
        for _ in range(future_patches):
            state = torch.nn.functional.linear(state, self.transition)
            predictions.append(self.decoder(state))
        return torch.stack(predictions, dim=1)


class SKOLR(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, num_branches=2,
                 dynamic_dim=256, hidden_dim=512, hidden_layers=2,
                 patch_len=None, dropout=0.05, eps=1e-5):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.patch_len = patch_len or max(1, input_len // 6)
        if input_len % self.patch_len != 0:
            raise ValueError('SKOLR requires input_len divisible by patch_len')
        self.eps = eps
        freq_bins = input_len // 2 + 1
        self.frequency_logits = nn.Parameter(torch.zeros(num_branches, freq_bins))
        self.branches = nn.ModuleList([
            KoopmanBranch(
                self.patch_len, dynamic_dim, hidden_dim, hidden_layers, dropout
            )
            for _ in range(num_branches)
        ])
        self._initialize_frequency_gates()

    def _initialize_frequency_gates(self):
        # Complementary initial bands break branch symmetry while remaining soft.
        bins = self.frequency_logits.shape[1]
        centers = torch.linspace(0, bins - 1, self.frequency_logits.shape[0])
        positions = torch.arange(bins).float()
        width = max(1.0, bins / self.frequency_logits.shape[0])
        logits = torch.stack([
            2.0 - (positions - center).abs() * (4.0 / width)
            for center in centers
        ])
        with torch.no_grad():
            self.frequency_logits.copy_(logits)

    def forward(self, x, x_time=None, y_time=None):
        if x.ndim != 3:
            raise ValueError('SKOLR expects input shaped [batch, time, nodes]')
        mean = x.mean(dim=1, keepdim=True).detach()
        std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + self.eps).detach()
        normalized = (x - mean) / std

        spectrum = torch.fft.rfft(normalized, dim=1)
        batch_size, _, num_nodes = x.shape
        future_patches = math.ceil(self.output_len / self.patch_len)
        branch_predictions = []
        for branch_index, branch in enumerate(self.branches):
            gate = torch.sigmoid(self.frequency_logits[branch_index]).view(1, -1, 1)
            component = torch.fft.irfft(
                spectrum * gate, n=self.input_len, dim=1
            )
            patches = component.transpose(1, 2).reshape(
                batch_size * num_nodes,
                self.input_len // self.patch_len,
                self.patch_len,
            )
            prediction = branch(patches, future_patches)
            prediction = prediction.reshape(batch_size, num_nodes, -1)
            branch_predictions.append(prediction[:, :, :self.output_len])

        forecast = torch.stack(branch_predictions, dim=0).sum(dim=0)
        forecast = forecast.transpose(1, 2)
        return forecast * std + mean


@gin.configurable()
def skolr(num_nodes, input_len, output_len, num_branches=2,
          dynamic_dim=256, hidden_dim=512, hidden_layers=2,
          patch_len=None, dropout=0.05):
    return SKOLR(
        num_nodes=num_nodes,
        input_len=input_len,
        output_len=output_len,
        num_branches=num_branches,
        dynamic_dim=dynamic_dim,
        hidden_dim=hidden_dim,
        hidden_layers=hidden_layers,
        patch_len=patch_len,
        dropout=dropout,
    )
