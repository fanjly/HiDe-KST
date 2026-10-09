"""STID adapted to the forecasting interface used in this repository."""

import gin
import torch
from torch import nn


class ResidualMLP(nn.Module):
    def __init__(self, channels: int, dropout: float):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(channels, channels, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Dropout(dropout),
            nn.Conv2d(channels, channels, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x + self.net(x)


class STID(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        num_time_features: int,
        embed_dim: int = 32,
        node_dim: int = 32,
        temp_dim_tid: int = 32,
        temp_dim_diw: int = 32,
        num_layers: int = 3,
        dropout: float = 0.15,
        time_of_day_size: int = 288,
        calendar_features: bool = False,
    ):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.num_time_features = num_time_features
        self.time_of_day_size = time_of_day_size
        self.calendar_features = calendar_features

        input_dim = input_len * (1 + num_time_features)
        hidden_dim = embed_dim + node_dim + temp_dim_tid + temp_dim_diw
        self.time_series_emb_layer = nn.Conv2d(input_dim, embed_dim, 1)
        self.node_emb = nn.Parameter(torch.empty(num_nodes, node_dim))
        self.time_in_day_emb = nn.Parameter(
            torch.empty(time_of_day_size, temp_dim_tid)
        )
        self.day_in_week_emb = nn.Parameter(torch.empty(7, temp_dim_diw))
        self.encoder = nn.Sequential(
            *(ResidualMLP(hidden_dim, dropout) for _ in range(num_layers))
        )
        self.regression_layer = nn.Conv2d(hidden_dim, output_len, 1)
        nn.init.xavier_uniform_(self.node_emb)
        nn.init.xavier_uniform_(self.time_in_day_emb)
        nn.init.xavier_uniform_(self.day_in_week_emb)

    def _time_features(self, x_time: torch.Tensor):
        if self.calendar_features:
            month = x_time[..., 0] / 12.0
            day = x_time[..., 1] / 31.0
            weekday = x_time[..., 2] / 7.0
            hour = x_time[..., 3] / 24.0
            normalized = torch.stack((month, day, weekday, hour), dim=-1)
            tid = x_time[:, -1, 3].long().clamp(0, self.time_of_day_size - 1)
            diw = x_time[:, -1, 2].long().clamp(0, 6)
        else:
            normalized = x_time[..., :self.num_time_features]
            tid = torch.round(
                x_time[:, -1, 0] * self.time_of_day_size
            ).long().remainder(self.time_of_day_size)
            diw = torch.round(x_time[:, -1, 1] * 7).long().remainder(7)
        return normalized, tid, diw

    def forward(
        self,
        x_enc: torch.Tensor,
        x_time: torch.Tensor,
        y_time: torch.Tensor,
    ) -> torch.Tensor:
        del y_time
        batch_size = x_enc.size(0)
        time_features, tid, diw = self._time_features(x_time)
        time_features = time_features.unsqueeze(2).expand(
            -1, -1, self.num_nodes, -1
        )
        history = torch.cat((x_enc.unsqueeze(-1), time_features), dim=-1)
        history = history.permute(0, 2, 1, 3).reshape(
            batch_size, self.num_nodes, -1
        )
        history = history.transpose(1, 2).unsqueeze(-1)
        history_emb = self.time_series_emb_layer(history)

        node_emb = self.node_emb.t().unsqueeze(0).unsqueeze(-1).expand(
            batch_size, -1, -1, -1
        )
        tid_emb = self.time_in_day_emb[tid].unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.num_nodes, -1
        )
        diw_emb = self.day_in_week_emb[diw].unsqueeze(-1).unsqueeze(-1).expand(
            -1, -1, self.num_nodes, -1
        )
        hidden = torch.cat((history_emb, node_emb, tid_emb, diw_emb), dim=1)
        return self.regression_layer(self.encoder(hidden)).squeeze(-1)


@gin.configurable()
def stid(
    num_nodes: int,
    input_len: int,
    output_len: int,
    num_time_features: int = 2,
    embed_dim: int = 32,
    node_dim: int = 32,
    temp_dim_tid: int = 32,
    temp_dim_diw: int = 32,
    num_layers: int = 3,
    dropout: float = 0.15,
    time_of_day_size: int = 288,
    calendar_features: bool = False,
) -> STID:
    return STID(
        num_nodes,
        input_len,
        output_len,
        num_time_features,
        embed_dim,
        node_dim,
        temp_dim_tid,
        temp_dim_diw,
        num_layers,
        dropout,
        time_of_day_size,
        calendar_features,
    )
