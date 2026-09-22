import torch
import torch.nn as nn
import torch.nn.functional as F
from models.itransformer.layers.Transformer_EncDec import Encoder, EncoderLayer
from models.itransformer.layers.SelfAttention_Family import (
    FullAttention,
    AttentionLayer,
)
from models.itransformer.layers.Embed import DataEmbedding_inverted
import numpy as np
import gin


@gin.configurable()
def itransformer(num_nodes, input_len, output_len):
    class iTransformerConfig:
        def __init__(self, seq_len, pred_len, output_attention, use_norm, d_model, embed, freq, dropout,
                     class_strategy, factor, n_heads, d_ff, activation, e_layers):
            self.seq_len = seq_len
            self.pred_len = pred_len
            self.output_attention = output_attention
            self.use_norm = use_norm
            self.d_model = d_model
            self.embed = embed
            self.freq = freq
            self.dropout = dropout
            self.class_strategy = class_strategy
            self.factor = factor
            self.n_heads = n_heads
            self.d_ff = d_ff
            self.activation = activation
            self.e_layers = e_layers

    output_attention = False
    use_norm = True
    d_model = 512
    embed = 'timeF'
    freq = 'h'
    dropout = 0.1
    class_strategy = 'projection'
    factor = 1
    n_heads = 8
    d_ff = 2048
    activation = 'gelu'
    e_layers = 2

    config = iTransformerConfig(
        seq_len=input_len,
        pred_len=output_len,
        output_attention=output_attention,
        use_norm=use_norm,
        d_model=d_model,
        embed=embed,
        freq=freq,
        dropout=dropout,
        class_strategy=class_strategy,
        factor=factor,
        n_heads=n_heads,
        d_ff=d_ff,
        activation=activation,
        e_layers=e_layers
    )
    return Model(config)


class Model(nn.Module):
    """
    Paper link: https://arxiv.org/abs/2310.06625
    """

    def __init__(self, configs):
        super(Model, self).__init__()
        self.seq_len = configs.seq_len
        self.pred_len = configs.pred_len
        self.output_attention = configs.output_attention
        self.use_norm = configs.use_norm
        # Embedding
        self.enc_embedding = DataEmbedding_inverted(
            configs.seq_len,
            configs.d_model,
            configs.embed,
            configs.freq,
            configs.dropout,
        )
        self.class_strategy = configs.class_strategy
        # Encoder-only architecture
        self.encoder = Encoder(
            [
                EncoderLayer(
                    AttentionLayer(
                        FullAttention(
                            False,
                            configs.factor,
                            attention_dropout=configs.dropout,
                            output_attention=configs.output_attention,
                        ),
                        configs.d_model,
                        configs.n_heads,
                    ),
                    configs.d_model,
                    configs.d_ff,
                    dropout=configs.dropout,
                    activation=configs.activation,
                )
                for l in range(configs.e_layers)
            ],
            norm_layer=torch.nn.LayerNorm(configs.d_model),
        )
        self.projector = nn.Linear(configs.d_model, configs.pred_len, bias=True)

    def forecast(self, x_enc, x_mark_enc, x_dec, x_mark_dec):
        if self.use_norm:
            # Normalization from Non-stationary Transformer
            means = x_enc.mean(1, keepdim=True).detach()
            x_enc = x_enc - means
            stdev = torch.sqrt(
                torch.var(x_enc, dim=1, keepdim=True, unbiased=False) + 1e-5
            )
            x_enc /= stdev

        _, _, N = x_enc.shape  # B L N
        # B: batch_size;    E: d_model;
        # L: seq_len;       S: pred_len;
        # N: number of variate (tokens), can also includes covariates

        # Embedding
        # B L N -> B N E                (B L N -> B L E in the vanilla Transformer)
        enc_out = self.enc_embedding(
            x_enc, x_mark_enc
        )  # covariates (e.g timestamp) can be also embedded as tokens

        # B N E -> B N E                (B L E -> B L E in the vanilla Transformer)
        # the dimensions of embedded time series has been inverted, and then processed by native attn, layernorm and ffn modules
        enc_out, attns = self.encoder(enc_out, attn_mask=None)

        # B N E -> B N S -> B S N
        dec_out = self.projector(enc_out).permute(0, 2, 1)[
            :, :, :N
        ]  # filter the covariates

        if self.use_norm:
            # De-Normalization from Non-stationary Transformer
            dec_out = dec_out * (
                stdev[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )
            dec_out = dec_out + (
                means[:, 0, :].unsqueeze(1).repeat(1, self.pred_len, 1)
            )

        return dec_out, attns

    def forward(self, x_enc, x_mark_enc=None, x_dec=None, x_mark_dec=None, mask=None):
        x_mark_enc, x_dec, x_mark_dec = None, None, None
        dec_out, attns = self.forecast(x_enc, x_mark_enc, x_dec, x_mark_dec)

        if self.output_attention:
            return dec_out[:, -self.pred_len:, :], attns
        else:
            return dec_out[:, -self.pred_len:, :]  # [B, L, D]
