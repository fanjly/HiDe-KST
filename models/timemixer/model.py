"""TimeMixer adapter for this project's forecasting protocol.

The implementation follows the official ICLR 2024 TimeMixer architecture:
multiscale average pooling, past decomposable mixing with bottom-up seasonal
and top-down trend paths, and additive multi-predictor mixing.  It is kept
self-contained so it does not add the upstream repository's legacy package
dependencies.

Reference implementation: https://github.com/kwuking/TimeMixer
License of the reference implementation: Apache-2.0.
"""

import gin
import torch
from torch import nn
from torch.nn import functional as F


class MovingAverageDecomposition(nn.Module):
    def __init__(self, kernel_size):
        super().__init__()
        self.kernel_size = kernel_size

    def forward(self, x):
        left = (self.kernel_size - 1) // 2
        right = self.kernel_size - 1 - left
        padded = F.pad(x.transpose(1, 2), (left, right), mode='replicate')
        trend = F.avg_pool1d(padded, self.kernel_size, stride=1).transpose(1, 2)
        return x - trend, trend


class MultiScaleSeasonMixing(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(lengths[i], lengths[i + 1]),
                nn.GELU(),
                nn.Linear(lengths[i + 1], lengths[i + 1]),
            )
            for i in range(len(lengths) - 1)
        ])

    def forward(self, values):
        # Values are [B, D, T], ordered from fine to coarse scale.
        high = values[0]
        outputs = [high.transpose(1, 2)]
        for index, layer in enumerate(self.layers):
            low = values[index + 1] + layer(high)
            outputs.append(low.transpose(1, 2))
            high = low
        return outputs


class MultiScaleTrendMixing(nn.Module):
    def __init__(self, lengths):
        super().__init__()
        reversed_lengths = list(reversed(lengths))
        self.layers = nn.ModuleList([
            nn.Sequential(
                nn.Linear(reversed_lengths[i], reversed_lengths[i + 1]),
                nn.GELU(),
                nn.Linear(reversed_lengths[i + 1], reversed_lengths[i + 1]),
            )
            for i in range(len(reversed_lengths) - 1)
        ])

    def forward(self, values):
        reversed_values = list(reversed(values))
        low = reversed_values[0]
        outputs = [low.transpose(1, 2)]
        for index, layer in enumerate(self.layers):
            high = reversed_values[index + 1] + layer(low)
            outputs.append(high.transpose(1, 2))
            low = high
        return list(reversed(outputs))


class PastDecomposableMixing(nn.Module):
    def __init__(self, lengths, d_model, d_ff, moving_avg, dropout):
        super().__init__()
        self.decomposition = MovingAverageDecomposition(moving_avg)
        self.season_mixing = MultiScaleSeasonMixing(lengths)
        self.trend_mixing = MultiScaleTrendMixing(lengths)
        self.cross_mlp = nn.Sequential(
            nn.Linear(d_model, d_ff),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_ff, d_model),
        )

    def forward(self, x_list):
        season, trend = zip(*(self.decomposition(x) for x in x_list))
        season = [self.cross_mlp(x) for x in season]
        trend = [self.cross_mlp(x) for x in trend]
        season = self.season_mixing([x.transpose(1, 2) for x in season])
        trend = self.trend_mixing([x.transpose(1, 2) for x in trend])
        return [s + t for s, t in zip(season, trend)]


class TimeMixer(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, d_model=128,
                 d_ff=256, e_layers=5, down_sampling_layers=1,
                 down_sampling_window=2, moving_avg=25, dropout=0.1,
                 use_norm=False):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.down_sampling_layers = down_sampling_layers
        self.down_sampling_window = down_sampling_window
        self.use_norm = use_norm
        self.lengths = [
            input_len // (down_sampling_window ** level)
            for level in range(down_sampling_layers + 1)
        ]
        if self.lengths[-1] < 1:
            raise ValueError('Too many TimeMixer down-sampling layers')

        self.embedding = nn.Linear(num_nodes, d_model)
        self.blocks = nn.ModuleList([
            PastDecomposableMixing(
                self.lengths, d_model, d_ff, moving_avg, dropout
            )
            for _ in range(e_layers)
        ])
        self.predict_layers = nn.ModuleList([
            nn.Linear(length, output_len) for length in self.lengths
        ])
        self.projection = nn.Linear(d_model, num_nodes)
        self.residual_layers = nn.ModuleList([
            nn.Linear(length, length) for length in self.lengths
        ])
        self.regression_layers = nn.ModuleList([
            nn.Linear(length, output_len) for length in self.lengths
        ])

    def _make_scales(self, x):
        scales = [x]
        current = x.transpose(1, 2)
        for _ in range(self.down_sampling_layers):
            current = F.avg_pool1d(
                current, self.down_sampling_window, self.down_sampling_window
            )
            scales.append(current.transpose(1, 2))
        return scales

    def forward(self, x, x_time=None, y_time=None):
        if x.ndim != 3:
            raise ValueError('TimeMixer expects input shaped [batch, time, nodes]')
        if self.use_norm:
            mean = x.mean(dim=1, keepdim=True).detach()
            std = torch.sqrt(x.var(dim=1, keepdim=True, unbiased=False) + 1e-5).detach()
            x = (x - mean) / std
        else:
            mean = std = None

        raw_scales = self._make_scales(x)
        hidden = [self.embedding(scale) for scale in raw_scales]
        for block in self.blocks:
            hidden = block(hidden)

        predictions = []
        for index, (encoded, raw) in enumerate(zip(hidden, raw_scales)):
            mixed = self.predict_layers[index](encoded.transpose(1, 2)).transpose(1, 2)
            decoded = self.projection(mixed)
            residual = self.residual_layers[index](raw.transpose(1, 2))
            residual = self.regression_layers[index](residual).transpose(1, 2)
            predictions.append(decoded + residual)
        forecast = torch.stack(predictions, dim=0).sum(dim=0)
        if self.use_norm:
            forecast = forecast * std + mean
        return forecast


@gin.configurable()
def timemixer(num_nodes, input_len, output_len, d_model=128, d_ff=256,
              e_layers=5, down_sampling_layers=1,
              down_sampling_window=2, moving_avg=25, dropout=0.1,
              use_norm=False):
    return TimeMixer(
        num_nodes=num_nodes,
        input_len=input_len,
        output_len=output_len,
        d_model=d_model,
        d_ff=d_ff,
        e_layers=e_layers,
        down_sampling_layers=down_sampling_layers,
        down_sampling_window=down_sampling_window,
        moving_avg=moving_avg,
        dropout=dropout,
        use_norm=use_norm,
    )
