import math
import torch
import torch.nn as nn


class RevIN(nn.Module):
    # https://github.com/ts-kim/RevIN
    def __init__(self, num_features: int, eps=1e-5, affine=True):
        """
        :param num_features: the number of features or channels
        :param eps: a value added for numerical stability
        :param affine: if True, RevIN has learnable affine parameters
        """
        super(RevIN, self).__init__()
        self.num_features = num_features
        self.eps = eps
        self.affine = affine
        if self.affine:
            self._init_params()

    def forward(self, x, mode: str):
        if mode == 'norm':
            self._get_statistics(x)
            x = self._normalize(x)
        elif mode == 'denorm':
            x = self._denormalize(x)
        else: raise NotImplementedError
        return x

    def _init_params(self):
        # initialize RevIN params: (C,)
        self.affine_weight = nn.Parameter(torch.ones(self.num_features))
        self.affine_bias = nn.Parameter(torch.zeros(self.num_features))

    def _get_statistics(self, x):
        dim2reduce = tuple(range(1, x.ndim - 1))
        self.mean = torch.mean(x, dim=dim2reduce, keepdim=True).detach()
        self.stdev = torch.sqrt(torch.var(x, dim=dim2reduce, keepdim=True, unbiased=False) + self.eps).detach()

    def _normalize(self, x):
        x = x - self.mean
        x = x / self.stdev
        if self.affine:
            x = x * self.affine_weight
            x = x + self.affine_bias
        return x

    def _denormalize(self, x):
        if self.affine:
            x = x - self.affine_bias
            x = x / (self.affine_weight + self.eps * self.eps)
        x = x * self.stdev
        x = x + self.mean
        return x


class MLP(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers, res_ln=False, dropout=0.05):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.res_ln = res_ln
        self.dropout = dropout

        self.activation = nn.GELU()

        layers = [
            nn.Linear(self.input_dim, self.hidden_dim),
            self.activation,
            nn.Dropout(self.dropout),
        ]
        for i in range(self.hidden_layers - 2):
            layers += [
                nn.Linear(self.hidden_dim, self.hidden_dim),
                self.activation,
                nn.Dropout(self.dropout),
            ]
        layers += [nn.Linear(self.hidden_dim, self.output_dim)]
        self.layers = nn.Sequential(*layers)

        if self.res_ln:
            self.ln = nn.LayerNorm(self.output_dim)
            if self.input_dim != self.output_dim:
                self.res_proj = nn.Linear(self.input_dim, self.output_dim)

    def forward(self, x):
        # x: [Batch, Nodes, Time, Features]
        out = self.layers(x)
        if self.res_ln:
            if self.input_dim != self.output_dim:
                res = self.res_proj(x)
            else:
                res = x
            out = res + out
            out = self.ln(out)
        return out


class NodeModem(nn.Module):
    def __init__(self, num_nodes):
        super().__init__()
        self.num_nodes = num_nodes
        self.amp = nn.Parameter(torch.ones(num_nodes, 1, 1))
        self.bias = nn.Parameter(torch.zeros(num_nodes, 1, 1))

    def forward(self, x):
        # x: [Batch, Nodes, Time, Features]
        return self.amp * x + self.bias


class GraphConv(nn.Module):
    def __init__(self, input_dim, output_dim, bias=True):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.weight = nn.Parameter(torch.FloatTensor(input_dim, output_dim))
        if bias:
            self.bias = nn.Parameter(torch.FloatTensor(output_dim))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, x, gso):
        # x: [Batch, Nodes, Time, Features]
        first_mul = torch.einsum('nm,bmtc->bntc', gso, x)
        second_mul = torch.einsum('bnti,ij->bntj', first_mul, self.weight)

        if self.bias is not None:
            graph_conv = torch.add(second_mul, self.bias)
        else:
            graph_conv = second_mul

        return graph_conv


class GCN(nn.Module):
    def __init__(self, input_dim, output_dim, hidden_dim, hidden_layers, dropout=0.05):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.dropout = dropout

        self.activation = nn.GELU()
        self.dropout_layer = nn.Dropout(self.dropout)

        self.convs = nn.ModuleList()
        self.convs.append(GraphConv(self.input_dim, self.hidden_dim))
        for i in range(self.hidden_layers - 2):
            self.convs.append(GraphConv(self.hidden_dim, self.hidden_dim))
        self.convs.append(GraphConv(self.hidden_dim, self.output_dim))

    def forward(self, x, gso):
        # x: [Batch, Nodes, Time, Features]
        for conv in self.convs[:-1]:
            x = conv(x, gso)
            x = self.activation(x)
            x = self.dropout_layer(x)
        x = self.convs[-1](x, gso)

        return x
