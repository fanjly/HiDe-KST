"""Self-contained PDFormer adapter for this project's forecasting protocol.

The three-way attention, pattern matching and prediction head follow the
official implementation: https://github.com/BUAABIGSCity/PDFormer
"""

import math
import os
import gin
import numpy as np
import torch
from torch import nn
from torch.nn import functional as F


def drop_path(x, drop_prob=0.0, training=False):
    if drop_prob == 0.0 or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    return x.div(keep_prob) * random_tensor


class DropPath(nn.Module):
    def __init__(self, drop_prob=0.0):
        super().__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class TokenEmbedding(nn.Module):
    def __init__(self, input_dim, embed_dim):
        super().__init__()
        self.proj = nn.Linear(input_dim, embed_dim)

    def forward(self, x):
        return self.proj(x)


class PositionalEncoding(nn.Module):
    def __init__(self, embed_dim, max_len):
        super().__init__()
        pe = torch.zeros(max_len, embed_dim)
        position = torch.arange(max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, embed_dim, 2).float()
                    * -(math.log(10000.0) / embed_dim)).exp()
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        self.register_buffer('pe', pe.unsqueeze(0), persistent=False)

    def forward(self, x):
        return self.pe[:, :x.shape[1]].unsqueeze(2)


def _minute_and_weekday(x_time, time_feature_style):
    if time_feature_style == 'pems':
        minute = torch.round(x_time[..., 0] * 1440).long() % 1440
        weekday = torch.round(x_time[..., 1] * 7).long() % 7
    elif time_feature_style == 'calendar':
        minute = torch.round(x_time[..., 3] * 60).long() % 1440
        weekday = x_time[..., 2].long() % 7
    else:
        raise ValueError("time_feature_style must be 'pems' or 'calendar'")
    return minute, weekday


class DataEmbedding(nn.Module):
    def __init__(self, embed_dim, lape_dim, input_len, dropout):
        super().__init__()
        self.value_embedding = TokenEmbedding(1, embed_dim)
        self.position_encoding = PositionalEncoding(embed_dim, input_len)
        self.daytime_embedding = nn.Embedding(1440, embed_dim)
        self.weekday_embedding = nn.Embedding(7, embed_dim)
        self.spatial_embedding = nn.Linear(lape_dim, embed_dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, minute, weekday, laplacian_pe):
        hidden = self.value_embedding(x.unsqueeze(-1))
        hidden = hidden + self.position_encoding(hidden)
        hidden = hidden + self.daytime_embedding(minute).unsqueeze(2)
        hidden = hidden + self.weekday_embedding(weekday).unsqueeze(2)
        hidden = hidden + self.spatial_embedding(laplacian_pe).unsqueeze(0).unsqueeze(0)
        return self.dropout(hidden)


class STSelfAttention(nn.Module):
    def __init__(self, dim, s_attn_size, geo_num_heads=4, sem_num_heads=2,
                 temporal_num_heads=2, qkv_bias=True, attn_drop=0.0,
                 proj_drop=0.0):
        super().__init__()
        total_heads = geo_num_heads + sem_num_heads + temporal_num_heads
        if dim % total_heads != 0:
            raise ValueError('embed_dim must be divisible by the total number of heads')
        self.geo_num_heads = geo_num_heads
        self.sem_num_heads = sem_num_heads
        self.temporal_num_heads = temporal_num_heads
        self.head_dim = dim // total_heads
        self.scale = self.head_dim ** -0.5
        self.geo_dim = self.head_dim * geo_num_heads
        self.sem_dim = self.head_dim * sem_num_heads
        self.temporal_dim = self.head_dim * temporal_num_heads

        self.pattern_q = nn.Linear(dim, self.geo_dim)
        self.pattern_k = nn.Linear(dim, self.geo_dim)
        self.pattern_v = nn.Linear(dim, self.geo_dim)
        self.geo_q = nn.Conv2d(dim, self.geo_dim, 1, bias=qkv_bias)
        self.geo_k = nn.Conv2d(dim, self.geo_dim, 1, bias=qkv_bias)
        self.geo_v = nn.Conv2d(dim, self.geo_dim, 1, bias=qkv_bias)
        self.sem_q = nn.Conv2d(dim, self.sem_dim, 1, bias=qkv_bias)
        self.sem_k = nn.Conv2d(dim, self.sem_dim, 1, bias=qkv_bias)
        self.sem_v = nn.Conv2d(dim, self.sem_dim, 1, bias=qkv_bias)
        self.temporal_q = nn.Conv2d(dim, self.temporal_dim, 1, bias=qkv_bias)
        self.temporal_k = nn.Conv2d(dim, self.temporal_dim, 1, bias=qkv_bias)
        self.temporal_v = nn.Conv2d(dim, self.temporal_dim, 1, bias=qkv_bias)
        self.geo_attn_drop = nn.Dropout(attn_drop)
        self.sem_attn_drop = nn.Dropout(attn_drop)
        self.temporal_attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    @staticmethod
    def _conv_features(layer, x):
        return layer(x.permute(0, 3, 1, 2)).permute(0, 2, 3, 1)

    def forward(self, x, x_patterns, pattern_keys, geo_mask, sem_mask):
        batch_size, steps, nodes, dim = x.shape

        tq = self._conv_features(self.temporal_q, x).reshape(
            batch_size, steps, nodes, self.temporal_num_heads, self.head_dim
        ).permute(0, 2, 3, 1, 4)
        tk = self._conv_features(self.temporal_k, x).reshape(
            batch_size, steps, nodes, self.temporal_num_heads, self.head_dim
        ).permute(0, 2, 3, 1, 4)
        tv = self._conv_features(self.temporal_v, x).reshape(
            batch_size, steps, nodes, self.temporal_num_heads, self.head_dim
        ).permute(0, 2, 3, 1, 4)
        temporal_attn = self.temporal_attn_drop(
            ((tq @ tk.transpose(-2, -1)) * self.scale).softmax(dim=-1)
        )
        temporal_x = (temporal_attn @ tv).transpose(2, 3).reshape(
            batch_size, nodes, steps, self.temporal_dim
        ).transpose(1, 2)

        gq = self._conv_features(self.geo_q, x)
        gk = self._conv_features(self.geo_k, x)
        pattern_q = self.pattern_q(x_patterns)
        pattern_k = self.pattern_k(pattern_keys)
        pattern_v = self.pattern_v(pattern_keys)
        pattern_attn = ((pattern_q @ pattern_k.transpose(-2, -1))
                        * self.scale).softmax(dim=-1)
        gk = gk + pattern_attn @ pattern_v
        gv = self._conv_features(self.geo_v, x)
        gq = gq.reshape(batch_size, steps, nodes, self.geo_num_heads,
                        self.head_dim).permute(0, 1, 3, 2, 4)
        gk = gk.reshape(batch_size, steps, nodes, self.geo_num_heads,
                        self.head_dim).permute(0, 1, 3, 2, 4)
        gv = gv.reshape(batch_size, steps, nodes, self.geo_num_heads,
                        self.head_dim).permute(0, 1, 3, 2, 4)
        geo_attn = (gq @ gk.transpose(-2, -1)) * self.scale
        geo_attn = geo_attn.masked_fill(geo_mask, float('-inf'))
        geo_attn = self.geo_attn_drop(geo_attn.softmax(dim=-1))
        geo_x = (geo_attn @ gv).transpose(2, 3).reshape(
            batch_size, steps, nodes, self.geo_dim
        )

        sq = self._conv_features(self.sem_q, x).reshape(
            batch_size, steps, nodes, self.sem_num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        sk = self._conv_features(self.sem_k, x).reshape(
            batch_size, steps, nodes, self.sem_num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        sv = self._conv_features(self.sem_v, x).reshape(
            batch_size, steps, nodes, self.sem_num_heads, self.head_dim
        ).permute(0, 1, 3, 2, 4)
        sem_attn = (sq @ sk.transpose(-2, -1)) * self.scale
        sem_attn = sem_attn.masked_fill(sem_mask, float('-inf'))
        sem_attn = self.sem_attn_drop(sem_attn.softmax(dim=-1))
        sem_x = (sem_attn @ sv).transpose(2, 3).reshape(
            batch_size, steps, nodes, self.sem_dim
        )
        return self.proj_drop(self.proj(torch.cat([temporal_x, geo_x, sem_x], dim=-1)))


class MLP(nn.Module):
    def __init__(self, dim, ratio, dropout):
        super().__init__()
        hidden_dim = int(dim * ratio)
        self.net = nn.Sequential(
            nn.Linear(dim, hidden_dim), nn.GELU(), nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim), nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.net(x)


class STEncoderBlock(nn.Module):
    def __init__(self, dim, s_attn_size, geo_num_heads, sem_num_heads,
                 temporal_num_heads, mlp_ratio, dropout, attn_drop,
                 drop_path_prob):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = STSelfAttention(
            dim, s_attn_size, geo_num_heads, sem_num_heads,
            temporal_num_heads, True, attn_drop, dropout
        )
        self.drop_path = DropPath(drop_path_prob) if drop_path_prob > 0 else nn.Identity()
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(dim, mlp_ratio, dropout)

    def forward(self, x, x_patterns, pattern_keys, geo_mask, sem_mask):
        x = self.norm1(x + self.drop_path(
            self.attn(x, x_patterns, pattern_keys, geo_mask, sem_mask)
        ))
        return self.norm2(x + self.drop_path(self.mlp(x)))


def _resolve_asset_path(path):
    if os.path.isabs(path):
        return path
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    return os.path.join(project_root, 'datasets', path)


class PDFormer(nn.Module):
    def __init__(self, num_nodes, input_len, output_len, asset_path,
                 time_feature_style='pems', embed_dim=64, skip_dim=256,
                 lape_dim=8, geo_num_heads=4, sem_num_heads=2,
                 temporal_num_heads=2, mlp_ratio=4, dropout=0.0,
                 attn_drop=0.0, drop_path_rate=0.3, s_attn_size=3,
                 enc_depth=6, far_mask_delta=5, dtw_delta=5,
                 random_lape_flip=True):
        super().__init__()
        assets = np.load(_resolve_asset_path(asset_path))
        required = {'dtw_matrix', 'pattern_keys', 'hop_matrix', 'laplacian_pe'}
        missing = required.difference(assets.files)
        if missing:
            raise ValueError(f'PDFormer asset file is missing: {sorted(missing)}')
        if assets['dtw_matrix'].shape != (num_nodes, num_nodes):
            raise ValueError('PDFormer assets do not match num_nodes')
        pattern_keys = torch.from_numpy(assets['pattern_keys']).float()
        if pattern_keys.ndim == 2:
            pattern_keys = pattern_keys.unsqueeze(-1)
        if pattern_keys.shape[1] != s_attn_size:
            raise ValueError('pattern key length does not match s_attn_size')

        hop_matrix = torch.from_numpy(assets['hop_matrix'])
        geo_mask = hop_matrix.T >= far_mask_delta
        dtw_matrix = torch.from_numpy(assets['dtw_matrix'])
        sem_mask = torch.ones(num_nodes, num_nodes, dtype=torch.bool)
        nearest = dtw_matrix.argsort(dim=1)[:, :dtw_delta]
        sem_mask.scatter_(1, nearest, False)
        self.register_buffer('geo_mask', geo_mask.bool(), persistent=True)
        self.register_buffer('sem_mask', sem_mask, persistent=True)
        self.register_buffer('pattern_keys', pattern_keys, persistent=True)
        self.register_buffer(
            'laplacian_pe', torch.from_numpy(assets['laplacian_pe'][:, :lape_dim]).float(),
            persistent=True
        )

        self.num_nodes = num_nodes
        self.input_len = input_len
        self.output_len = output_len
        self.time_feature_style = time_feature_style
        self.s_attn_size = s_attn_size
        self.random_lape_flip = random_lape_flip
        self.pattern_embedding = TokenEmbedding(s_attn_size, embed_dim)
        self.data_embedding = DataEmbedding(embed_dim, lape_dim, input_len, dropout)
        drop_probs = torch.linspace(0, drop_path_rate, enc_depth).tolist()
        self.encoder_blocks = nn.ModuleList([
            STEncoderBlock(
                embed_dim, s_attn_size, geo_num_heads, sem_num_heads,
                temporal_num_heads, mlp_ratio, dropout, attn_drop,
                drop_probs[i]
            ) for i in range(enc_depth)
        ])
        self.skip_convs = nn.ModuleList([
            nn.Conv2d(embed_dim, skip_dim, 1) for _ in range(enc_depth)
        ])
        self.end_conv1 = nn.Conv2d(input_len, output_len, 1)
        self.end_conv2 = nn.Conv2d(skip_dim, 1, 1)

    def _patterns(self, x):
        patterns = []
        for i in range(self.s_attn_size):
            part = F.pad(
                x[:, :x.shape[1] + i + 1 - self.s_attn_size],
                (0, 0, self.s_attn_size - 1 - i, 0)
            ).unsqueeze(-1)
            patterns.append(part)
        return torch.cat(patterns, dim=-1)

    def forward(self, x, x_time=None, y_time=None):
        if x_time is None:
            raise ValueError('PDFormer requires historical time features')
        minute, weekday = _minute_and_weekday(x_time, self.time_feature_style)
        x_patterns = self.pattern_embedding(self._patterns(x))
        pattern_keys = self.pattern_embedding(self.pattern_keys[..., 0])
        laplacian_pe = self.laplacian_pe
        if self.training and self.random_lape_flip:
            signs = torch.where(
                torch.rand(laplacian_pe.shape[1], device=x.device) >= 0.5,
                1.0, -1.0
            )
            laplacian_pe = laplacian_pe * signs
        hidden = self.data_embedding(x, minute, weekday, laplacian_pe)
        skip = 0
        for block, skip_conv in zip(self.encoder_blocks, self.skip_convs):
            hidden = block(
                hidden, x_patterns, pattern_keys, self.geo_mask, self.sem_mask
            )
            skip = skip + skip_conv(hidden.permute(0, 3, 2, 1))
        skip = self.end_conv1(F.relu(skip.permute(0, 3, 2, 1)))
        skip = self.end_conv2(F.relu(skip.permute(0, 3, 2, 1)))
        return skip.permute(0, 3, 2, 1).squeeze(-1)


@gin.configurable()
def pdformer(num_nodes, input_len, output_len, asset_path,
             time_feature_style='pems', embed_dim=64, skip_dim=256,
             lape_dim=8, geo_num_heads=4, sem_num_heads=2,
             temporal_num_heads=2, mlp_ratio=4, dropout=0.0,
             attn_drop=0.0, drop_path_rate=0.3, s_attn_size=3,
             enc_depth=6, far_mask_delta=5, dtw_delta=5,
             random_lape_flip=True):
    return PDFormer(
        num_nodes, input_len, output_len, asset_path,
        time_feature_style, embed_dim, skip_dim, lape_dim,
        geo_num_heads, sem_num_heads, temporal_num_heads, mlp_ratio,
        dropout, attn_drop, drop_path_rate, s_attn_size, enc_depth,
        far_mask_delta, dtw_delta, random_lape_flip
    )
