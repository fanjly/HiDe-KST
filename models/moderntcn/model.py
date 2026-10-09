"""ModernTCN adapted to short-horizon multivariate forecasting."""

import gin
import torch
import torch.nn.functional as F
from torch import nn


class RevIN(nn.Module):
    def __init__(self, num_nodes: int, affine: bool = False):
        super().__init__()
        self.affine = affine
        if affine:
            self.weight = nn.Parameter(torch.ones(1, 1, num_nodes))
            self.bias = nn.Parameter(torch.zeros(1, 1, num_nodes))

    def normalize(self, x: torch.Tensor) -> torch.Tensor:
        self.mean = x.mean(dim=1, keepdim=True).detach()
        self.std = torch.sqrt(
            x.var(dim=1, keepdim=True, unbiased=False) + 1e-5
        ).detach()
        x = (x - self.mean) / self.std
        if self.affine:
            x = x * self.weight + self.bias
        return x

    def denormalize(self, x: torch.Tensor) -> torch.Tensor:
        if self.affine:
            x = (x - self.bias) / (self.weight + 1e-10)
        return x * self.std + self.mean


class LargeKernelConv(nn.Module):
    def __init__(self, channels: int, large_kernel: int, small_kernel: int):
        super().__init__()
        self.large = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                large_kernel,
                padding=large_kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )
        self.small = nn.Sequential(
            nn.Conv1d(
                channels,
                channels,
                small_kernel,
                padding=small_kernel // 2,
                groups=channels,
                bias=False,
            ),
            nn.BatchNorm1d(channels),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.large(x) + self.small(x)


class ModernTCNBlock(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        dim: int,
        large_kernel: int,
        small_kernel: int,
        ffn_ratio: int,
        dropout: float,
    ):
        super().__init__()
        hidden_dim = dim * ffn_ratio
        channels = num_nodes * dim
        self.num_nodes = num_nodes
        self.dim = dim
        self.temporal = LargeKernelConv(channels, large_kernel, small_kernel)
        self.norm = nn.BatchNorm1d(dim)
        self.variable_ffn = nn.Sequential(
            nn.Conv1d(channels, num_nodes * hidden_dim, 1, groups=num_nodes),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(num_nodes * hidden_dim, channels, 1, groups=num_nodes),
            nn.Dropout(dropout),
        )
        self.cross_variable_ffn = nn.Sequential(
            nn.Conv1d(channels, dim * num_nodes * ffn_ratio, 1, groups=dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Conv1d(dim * num_nodes * ffn_ratio, channels, 1, groups=dim),
            nn.Dropout(dropout),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch_size, num_nodes, dim, length = x.shape
        residual = x
        y = self.temporal(x.reshape(batch_size, num_nodes * dim, length))
        y = y.reshape(batch_size * num_nodes, dim, length)
        y = self.norm(y).reshape(batch_size, num_nodes * dim, length)
        y = self.variable_ffn(y).reshape(batch_size, num_nodes, dim, length)
        y = y.permute(0, 2, 1, 3).reshape(batch_size, dim * num_nodes, length)
        y = self.cross_variable_ffn(y)
        y = y.reshape(batch_size, dim, num_nodes, length).permute(0, 2, 1, 3)
        return residual + y


class ModernTCN(nn.Module):
    def __init__(
        self,
        num_nodes: int,
        input_len: int,
        output_len: int,
        patch_size: int = 8,
        patch_stride: int = 4,
        num_blocks=(1, 1, 1, 1),
        large_kernels=(31, 29, 27, 13),
        small_kernels=(5, 5, 5, 5),
        dims=(64, 64, 64, 64),
        ffn_ratio: int = 2,
        downsample_ratio: int = 2,
        dropout: float = 0.1,
        use_multi_scale: bool = True,
        revin: bool = True,
        affine: bool = False,
    ):
        super().__init__()
        if not (len(num_blocks) == len(large_kernels) == len(small_kernels) == len(dims)):
            raise ValueError('Stage parameter lists must have the same length')
        self.num_nodes = num_nodes
        self.revin = RevIN(num_nodes, affine) if revin else None
        self.use_multi_scale = use_multi_scale
        self.stem = nn.Conv1d(
            1, dims[0], patch_size, stride=patch_stride, bias=False
        )
        self.stem_norm = nn.BatchNorm1d(dims[0])
        self.stem_padding = patch_size - patch_stride
        self.stages = nn.ModuleList()
        self.downsamples = nn.ModuleList()
        for stage_index, dim in enumerate(dims):
            self.stages.append(
                nn.Sequential(
                    *(
                        ModernTCNBlock(
                            num_nodes,
                            dim,
                            large_kernels[stage_index],
                            small_kernels[stage_index],
                            ffn_ratio,
                            dropout,
                        )
                        for _ in range(num_blocks[stage_index])
                    )
                )
            )
            if stage_index < len(dims) - 1:
                self.downsamples.append(
                    nn.Conv1d(
                        dims[stage_index],
                        dims[stage_index + 1],
                        downsample_ratio,
                        stride=downsample_ratio,
                        bias=False,
                    )
                )

        self.lateral = nn.ModuleList(
            nn.Conv1d(dim, dims[-1], 1) for dim in dims
        )
        self.smooth = nn.Conv1d(dims[-1], dims[-1], 3, padding=1)
        patch_count = input_len // patch_stride
        self.head = nn.Linear(dims[-1] * patch_count, output_len)

    def forward(
        self,
        x_enc: torch.Tensor,
        x_time: torch.Tensor,
        y_time: torch.Tensor,
    ) -> torch.Tensor:
        del x_time, y_time
        x = self.revin.normalize(x_enc) if self.revin is not None else x_enc
        batch_size, _, num_nodes = x.shape
        x = x.permute(0, 2, 1).reshape(batch_size * num_nodes, 1, -1)
        if self.stem_padding:
            x = torch.cat(
                (x, x[:, :, -1:].repeat(1, 1, self.stem_padding)), dim=-1
            )
        x = self.stem_norm(self.stem(x))
        x = x.reshape(batch_size, num_nodes, x.size(1), x.size(2))

        outputs = []
        for stage_index, stage in enumerate(self.stages):
            x = stage(x)
            outputs.append(x)
            if stage_index < len(self.downsamples):
                length = x.size(-1)
                x = self.downsamples[stage_index](
                    x.reshape(batch_size * num_nodes, x.size(2), length)
                )
                x = x.reshape(batch_size, num_nodes, x.size(1), x.size(2))

        if self.use_multi_scale:
            target_length = outputs[0].size(-1)
            fused = 0
            for output, lateral in zip(outputs, self.lateral):
                length = output.size(-1)
                output = lateral(
                    output.reshape(batch_size * num_nodes, output.size(2), length)
                )
                if output.size(-1) != target_length:
                    output = F.interpolate(
                        output, size=target_length, mode='linear', align_corners=False
                    )
                fused = fused + output
            x = self.smooth(fused)
        else:
            length = outputs[-1].size(-1)
            x = self.lateral[-1](
                outputs[-1].reshape(
                    batch_size * num_nodes, outputs[-1].size(2), length
                )
            )
            x = F.interpolate(
                x, size=outputs[0].size(-1), mode='linear', align_corners=False
            )

        prediction = self.head(x.flatten(1)).reshape(
            batch_size, num_nodes, -1
        ).permute(0, 2, 1)
        if self.revin is not None:
            prediction = self.revin.denormalize(prediction.permute(0, 1, 2))
        return prediction


@gin.configurable()
def moderntcn(
    num_nodes: int,
    input_len: int,
    output_len: int,
    patch_size: int = 8,
    patch_stride: int = 4,
    num_blocks=(1, 1, 1, 1),
    large_kernels=(31, 29, 27, 13),
    small_kernels=(5, 5, 5, 5),
    dims=(64, 64, 64, 64),
    ffn_ratio: int = 2,
    downsample_ratio: int = 2,
    dropout: float = 0.1,
    use_multi_scale: bool = True,
    revin: bool = True,
    affine: bool = False,
) -> ModernTCN:
    return ModernTCN(
        num_nodes,
        input_len,
        output_len,
        patch_size,
        patch_stride,
        num_blocks,
        large_kernels,
        small_kernels,
        dims,
        ffn_ratio,
        downsample_ratio,
        dropout,
        use_multi_scale,
        revin,
        affine,
    )
