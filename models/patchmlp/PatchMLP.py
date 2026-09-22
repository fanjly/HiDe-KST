import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from models.patchmlp.layers.Embed import Emb

import gin


@gin.configurable()
def patchmlp(num_nodes, input_len, output_len):
    class PatchMLPConfig:
        def __init__(self, input_len, output_len, num_nodes):
            self.seq_len = input_len
            self.pred_len = output_len
            self.output_attention = False
            self.use_norm = True

            self.dropout = 0.1
            self.d_model = 1024
            self.enc_in = num_nodes
            self.e_layers = 1

    config = PatchMLPConfig(input_len, output_len, num_nodes)
    return Model(config)


class moving_avg(nn.Module):
    """
    Moving average block to highlight the trend of time series
    """

    def __init__(self, kernel_size, stride):
        super(moving_avg, self).__init__()
        self.kernel_size = kernel_size
        self.avg = nn.AvgPool1d(kernel_size=kernel_size, stride=stride, padding=0)

    def forward(self, x):
        # padding on the both ends of time series
        front = x[:, :, 0:1].repeat(1, 1, (self.kernel_size - 1) // 2)
        end = x[:, :, -1:].repeat(1, 1, (self.kernel_size - 1) // 2)
        x = torch.cat([front, x, end], dim=-1)

        x = self.avg(x)
        return x


class series_decomp(nn.Module):
    """
    Series decomposition block
    """

    def __init__(self, kernel_size):
        super(series_decomp, self).__init__()
        self.moving_avg = moving_avg(kernel_size, stride=1)

    def forward(self, x):
        moving_mean = self.moving_avg(x)
        res = x - moving_mean
        return res, moving_mean


class Model(nn.Module):

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm

        self.decompsition = series_decomp(13)
        # Embedding
        self.emb = Emb(configs.seq_len, configs.d_model)
        self.seasonal_layers = nn.ModuleList(
            [Encoder(configs.d_model, configs.enc_in) for i in range(configs.e_layers)]
        )
        self.trend_layers = nn.ModuleList(
            [Encoder(configs.d_model, configs.enc_in) for i in range(configs.e_layers)]
        )

        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc):
        # x_enc: [B, T, N]

        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev
        x = x_enc.permute(0, 2, 1)
        x = self.emb(x)

        seasonal_init, trend_init = self.decompsition(x)

        for mod in self.seasonal_layers:
            seasonal_init = mod(seasonal_init)
        for mod in self.trend_layers:
            trend_init = mod(trend_init)

        x = seasonal_init + trend_init
        dec_out = self.projector(x)
        dec_out = dec_out.permute(0, 2, 1)
        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out

    def forward(self, x, x_stamp=None, y_stamp=None, mask=None):
        # Adapting to HKST signature: x, x_stamp, y_stamp
        dec_out = self.forecast(x)
        return dec_out[:, -self.pred_len:, :]  # [B, L, D]


class Encoder(nn.Module):

    def __init__(self, d_model, enc_in):
        super().__init__()
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)

        self.ff1 = nn.Sequential(
            nn.Linear(d_model, d_model), nn.GELU(), nn.Dropout(0.1)
        )

        self.ff2 = nn.Sequential(nn.Linear(enc_in, enc_in), nn.GELU(), nn.Dropout(0.1))

    def forward(self, x):

        y_0 = self.ff1(x)
        y_0 = y_0 + x
        y_0 = self.norm1(y_0)
        y_1 = y_0.permute(0, 2, 1)
        y_1 = self.ff2(y_1)
        y_1 = y_1.permute(0, 2, 1)
        y_2 = y_1 * y_0 + x
        y_2 = self.norm1(y_2)

        return y_2
