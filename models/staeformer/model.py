"""STAEformer adapted to the forecasting interface used by this project.

The attention and embedding blocks follow the authors' official implementation:
https://github.com/XDZhelheim/STAEformer
"""

import gin
import torch
from torch import nn


class AttentionLayer(nn.Module):
    def __init__(self, model_dim, num_heads=8):
        super().__init__()
        if model_dim % num_heads != 0:
            raise ValueError('model_dim must be divisible by num_heads')
        self.num_heads = num_heads
        self.head_dim = model_dim // num_heads
        self.q_proj = nn.Linear(model_dim, model_dim)
        self.k_proj = nn.Linear(model_dim, model_dim)
        self.v_proj = nn.Linear(model_dim, model_dim)
        self.out_proj = nn.Linear(model_dim, model_dim)

    def forward(self, x):
        batch_size = x.shape[0]
        q = torch.cat(torch.split(self.q_proj(x), self.head_dim, dim=-1), dim=0)
        k = torch.cat(torch.split(self.k_proj(x), self.head_dim, dim=-1), dim=0)
        v = torch.cat(torch.split(self.v_proj(x), self.head_dim, dim=-1), dim=0)
        scores = (q @ k.transpose(-1, -2)) / self.head_dim ** 0.5
        out = scores.softmax(dim=-1) @ v
        out = torch.cat(torch.split(out, batch_size, dim=0), dim=-1)
        return self.out_proj(out)


class SelfAttentionLayer(nn.Module):
    def __init__(self, model_dim, feed_forward_dim, num_heads, dropout):
        super().__init__()
        self.attn = AttentionLayer(model_dim, num_heads)
        self.feed_forward = nn.Sequential(
            nn.Linear(model_dim, feed_forward_dim),
            nn.ReLU(inplace=True),
            nn.Linear(feed_forward_dim, model_dim),
        )
        self.norm1 = nn.LayerNorm(model_dim)
        self.norm2 = nn.LayerNorm(model_dim)
        self.dropout1 = nn.Dropout(dropout)
        self.dropout2 = nn.Dropout(dropout)

    def forward(self, x, dim):
        x = x.transpose(dim, -2)
        x = self.norm1(x + self.dropout1(self.attn(x)))
        x = self.norm2(x + self.dropout2(self.feed_forward(x)))
        return x.transpose(dim, -2)


def _time_indices(x_time, steps_per_day, time_feature_style):
    if time_feature_style == 'pems':
        tod = torch.round(x_time[..., 0] * steps_per_day).long() % steps_per_day
        dow = torch.round(x_time[..., 1] * 7).long() % 7
    elif time_feature_style == 'calendar':
        # Calendar features are [month, day, weekday, hour].
        tod = torch.round(x_time[..., 3] * steps_per_day / 24).long() % steps_per_day
        dow = x_time[..., 2].long() % 7
    else:
        raise ValueError("time_feature_style must be 'pems' or 'calendar'")
    return tod, dow


class STAEformer(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, steps_per_day=288,
                 time_feature_style='pems', input_embedding_dim=24,
                 tod_embedding_dim=24, dow_embedding_dim=24,
                 adaptive_embedding_dim=80, feed_forward_dim=256,
                 num_heads=4, num_layers=3, dropout=0.1):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.steps_per_day = steps_per_day
        self.time_feature_style = time_feature_style
        self.input_proj = nn.Linear(3, input_embedding_dim)
        self.tod_embedding = nn.Embedding(steps_per_day, tod_embedding_dim)
        self.dow_embedding = nn.Embedding(7, dow_embedding_dim)
        self.adaptive_embedding = nn.Parameter(
            torch.empty(input_len, num_nodes, adaptive_embedding_dim)
        )
        nn.init.xavier_uniform_(self.adaptive_embedding)

        model_dim = (input_embedding_dim + tod_embedding_dim
                     + dow_embedding_dim + adaptive_embedding_dim)
        self.temporal_layers = nn.ModuleList([
            SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.spatial_layers = nn.ModuleList([
            SelfAttentionLayer(model_dim, feed_forward_dim, num_heads, dropout)
            for _ in range(num_layers)
        ])
        self.output_proj = nn.Linear(input_len * model_dim, output_len)

    def forward(self, x, x_time=None, y_time=None):
        if x_time is None:
            raise ValueError('STAEformer requires historical time features')
        batch_size = x.shape[0]
        tod, dow = _time_indices(
            x_time, self.steps_per_day, self.time_feature_style
        )
        tod_value = (tod.float() / self.steps_per_day).unsqueeze(2).expand(
            -1, -1, self.num_nodes
        )
        dow_value = (dow.float() / 7).unsqueeze(2).expand(
            -1, -1, self.num_nodes
        )
        value = self.input_proj(torch.stack([x, tod_value, dow_value], dim=-1))
        adaptive = self.adaptive_embedding.unsqueeze(0).expand(batch_size, -1, -1, -1)
        hidden = torch.cat([
            value,
            self.tod_embedding(tod).unsqueeze(2).expand(-1, -1, self.num_nodes, -1),
            self.dow_embedding(dow).unsqueeze(2).expand(-1, -1, self.num_nodes, -1),
            adaptive,
        ], dim=-1)
        for layer in self.temporal_layers:
            hidden = layer(hidden, dim=1)
        for layer in self.spatial_layers:
            hidden = layer(hidden, dim=2)
        hidden = hidden.transpose(1, 2).reshape(batch_size, self.num_nodes, -1)
        return self.output_proj(hidden).transpose(1, 2)


@gin.configurable()
def staeformer(num_nodes, input_len, output_len, steps_per_day=288,
               time_feature_style='pems', input_embedding_dim=24,
               tod_embedding_dim=24, dow_embedding_dim=24,
               adaptive_embedding_dim=80, feed_forward_dim=256,
               num_heads=4, num_layers=3, dropout=0.1):
    return STAEformer(
        num_nodes=num_nodes,
        input_len=input_len,
        output_len=output_len,
        steps_per_day=steps_per_day,
        time_feature_style=time_feature_style,
        input_embedding_dim=input_embedding_dim,
        tod_embedding_dim=tod_embedding_dim,
        dow_embedding_dim=dow_embedding_dim,
        adaptive_embedding_dim=adaptive_embedding_dim,
        feed_forward_dim=feed_forward_dim,
        num_heads=num_heads,
        num_layers=num_layers,
        dropout=dropout,
    )
