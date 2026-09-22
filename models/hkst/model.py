import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hkst.layers import KoopmanLayer, AdaptiveSpatialFactorization
from models.hkst.modules import MLP, RevIN
import gin


@gin.configurable()
def hkst(num_nodes, input_len, output_len, adj_matrix,
         num_feat, hidden_dim, spat_layers, temp_layers,
         spat_rank, temp_rank, spat_decomp, use_node_modem,
         use_graph, patch_sizes, strides, recon_mask_ratio=1.0,
         te_ortho_loss=True, sp_ortho_loss=True, use_timestamp=True,
         evolution_mode: str = 'koopman', use_low_rank: bool = True,
         use_cross_scale_gating: bool = True):
    return HKST(
        input_len=input_len,
        output_len=output_len,
        num_nodes=num_nodes,
        num_feat=num_feat,
        hidden_dim=hidden_dim,
        spat_layers=spat_layers,
        temp_layers=temp_layers,
        spat_rank=spat_rank,
        temp_rank=temp_rank,
        use_timestamp=use_timestamp,
        spat_decomp=spat_decomp,
        use_node_modem=use_node_modem,
        use_graph=use_graph,
        patch_sizes=patch_sizes,
        strides=strides,
        dropout=0.1,
        recon_mask_ratio=recon_mask_ratio,
        te_ortho_loss=te_ortho_loss,
        sp_ortho_loss=sp_ortho_loss,
        adj_matrix=adj_matrix,
        evolution_mode=evolution_mode,
        use_low_rank=use_low_rank,
        use_cross_scale_gating=use_cross_scale_gating,
    )


class HKSTBlock(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, num_feat, patch_size, stride, hidden_dim,
                 spat_layers, temp_layers, spat_rank, temp_rank, spat_decomp, spat_res, temp_res_ln,
                 use_node_modem, prev_patch_num, use_graph, adj_matrix, dropout, te_ortho_loss, sp_ortho_loss,
                 time_dim=0, evolution_mode: str = 'koopman', use_low_rank: bool = True,
                 use_cross_scale_gating: bool = True):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.num_feat = num_feat
        self.patch_size = patch_size
        self.stride = stride
        self.hidden_dim = hidden_dim
        self.spat_layers = spat_layers
        self.temp_layers = temp_layers
        self.spat_rank = spat_rank
        self.temp_rank = temp_rank
        self.spat_decomp = spat_decomp
        self.spat_res = spat_res
        self.temp_res_ln = temp_res_ln
        self.use_node_modem = use_node_modem
        self.dropout = dropout
        self.use_graph = use_graph
        self.te_ortho_loss = te_ortho_loss
        self.sp_ortho_loss = sp_ortho_loss
        self.time_dim = time_dim
        self.evolution_mode = evolution_mode
        self.use_low_rank = use_low_rank
        self.use_cross_scale_gating = use_cross_scale_gating

        self.patch_num_in = (self.input_len - self.patch_size) // self.stride + 1
        target_len = (self.patch_num_in - 1) * self.stride + self.patch_size
        if target_len < self.input_len:
            self.patch_num_in += 1
            target_len = (self.patch_num_in - 1) * self.stride + self.patch_size
        assert self.patch_num_in >= 1, 'patch_num_in must be at least 1'
        self.pad_len = target_len - self.input_len
        self.patch_num_out = max(1, (self.output_len - self.patch_size) // self.stride + 1)
        target_out_len = (self.patch_num_out - 1) * self.stride + self.patch_size
        if target_out_len < self.output_len:
            self.patch_num_out += 1

        self.input_proj = MLP(self.num_feat * self.patch_size, self.hidden_dim,
                              self.hidden_dim, 2, self.temp_res_ln, self.dropout)
        if prev_patch_num is not None and self.use_cross_scale_gating:
            self.context_proj = nn.Linear(prev_patch_num, self.patch_num_in * 2)
        else:
            self.context_proj = None

        if self.use_graph:
            self.spatial_aggregation = AdaptiveSpatialFactorization(
                num_nodes=self.num_nodes,
                rank=self.spat_rank,
                input_dim=self.hidden_dim,
                output_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim,
                hidden_layers=self.spat_layers,
                res=self.spat_res,
                dropout=self.dropout,
                adj_matrix=adj_matrix
            )
        else:
            self.spatial_aggregation = None

        self.koopman_evolution = KoopmanLayer(
            num_nodes=self.num_nodes,
            input_dim=self.hidden_dim,
            low_rank_dim=self.temp_rank,
            output_dim=self.hidden_dim,
            hidden_dim=self.hidden_dim * 2,
            hidden_layers=self.temp_layers,
            res_ln=self.temp_res_ln,
            use_node_modem=False if self.spat_decomp else self.use_node_modem,
            dropout=self.dropout,
            patch_num_out=self.patch_num_out,
            time_dim=self.time_dim,
            evolution_mode=self.evolution_mode,
            use_low_rank=self.use_low_rank,
        )
        if self.spat_decomp:
            self.koopman_evolution_res = KoopmanLayer(
                num_nodes=self.num_nodes,
                input_dim=self.hidden_dim,
                low_rank_dim=self.temp_rank,
                output_dim=self.hidden_dim,
                hidden_dim=self.hidden_dim * 2,
                hidden_layers=self.temp_layers,
                res_ln=self.temp_res_ln,
                use_node_modem=self.use_node_modem,
                dropout=self.dropout,
                patch_num_out=self.patch_num_out,
                time_dim=self.time_dim,
                evolution_mode=self.evolution_mode,
                use_low_rank=self.use_low_rank,
            )

        self.backcast_head = MLP(self.patch_num_in * self.hidden_dim, self.input_len,
                                 self.hidden_dim, 2, self.temp_res_ln, self.dropout)
        self.forecast_head = MLP(self.patch_num_out * self.hidden_dim, self.output_len,
                                 self.hidden_dim, 2, self.temp_res_ln, self.dropout)

    def _patching(self, x):
        if self.pad_len > 0:
            x_pad = F.pad(x, (0, 0, 0, self.pad_len), 'replicate')
        else:
            x_pad = x
        x_patched = x_pad.unfold(dimension=2, size=self.patch_size, step=self.stride)
        return x_patched

    def _context_matching(self, ctx):
        ctx = ctx.permute(0, 1, 3, 2)  # [B, N, hidden_dim, P_prev]
        ctx = self.context_proj(ctx)   # [B, N, hidden_dim, P_in]
        ctx = ctx.permute(0, 1, 3, 2)  # [B, N, 2 * P_in, hidden_dim]
        return ctx

    def _patch_y_time(self, y_time):
        # y_time: [B, output_len, C]
        B, T, C = y_time.shape
        target_out_len = (self.patch_num_out - 1) * self.stride + self.patch_size
        pad_len = target_out_len - self.output_len
        if pad_len > 0:
            pad = y_time[:, -1:, :].expand(-1, pad_len, -1)
            y_time_pad = torch.cat([y_time, pad], dim=1)
        else:
            y_time_pad = y_time

        # [B, T_pad, C] -> [B, P_out, L, C]
        y_patched = y_time_pad.unfold(dimension=1, size=self.patch_size, step=self.stride)
        return y_patched.reshape(B, self.patch_num_out, -1)  # [B, P, C*L]

    def forward(self, x, context=None, y_time=None):
        # x: [Batch, Nodes, Time, Channels]
        B, N, T, C = x.shape

        x_patched = self._patching(x)  # [B, N, P, C, L]
        x_flat = x_patched.reshape(B, N, self.patch_num_in, -1)  # [B, N, P, C*L]

        z = self.input_proj(x_flat)  # [B, N, P_in, hidden_dim]

        if self.use_cross_scale_gating and self.context_proj is not None and context is not None:
            # ctx: [B, N, P_prev, hidden_dim] -> [B, N, P_in, hidden_dim]
            ctx = self._context_matching(context)
            z = z + torch.mul(F.tanh(ctx[:, :, :self.patch_num_in]), F.sigmoid(ctx[:, :, self.patch_num_in:]))  # gate

        y_time_patched = None
        if self.time_dim > 0 and y_time is not None:
            y_time_patched = self._patch_y_time(y_time)

        if self.use_graph:
            z_aggregated = self.spatial_aggregation(z)
            z_back, z_fore = self.koopman_evolution(z_aggregated, y_time=y_time_patched)
            if self.spat_decomp:
                z_res = z - z_aggregated
                z_back_res, z_fore_res = self.koopman_evolution_res(z_res, y_time=y_time_patched)
                z_back = z_back + z_back_res
                z_fore = z_fore + z_fore_res
        else:
            z_back, z_fore = self.koopman_evolution(z, y_time=y_time_patched)

        # Backcast
        # z_back: [B, N, P_in, hidden_dim]
        z_back_flat = z_back.flatten(start_dim=2)   # [B, N, P_in * hidden_dim]
        backcast = self.backcast_head(z_back_flat)  # [B, N, input_len]

        # Forecast
        # z_fore: [B, N, P_out, hidden_dim]
        z_fore_flat = z_fore.flatten(start_dim=2)   # [B, N, P_out * hidden_dim]
        forecast = self.forecast_head(z_fore_flat)  # [B, N, output_len]

        return backcast, forecast, z_back

    def get_ortho_loss(self):
        ortho_loss = 0
        if self.te_ortho_loss:
            ortho_loss = ortho_loss + self.koopman_evolution.get_ortho_loss()
        if self.use_graph:
            if self.sp_ortho_loss:
                ortho_loss = ortho_loss + self.spatial_aggregation.get_ortho_loss()
            if self.spat_decomp and self.te_ortho_loss:
                ortho_loss = ortho_loss + self.koopman_evolution_res.get_ortho_loss()
        return ortho_loss


class HKST(nn.Module):
    def __init__(self, input_len, output_len, num_nodes, num_feat, hidden_dim,
                 spat_layers, temp_layers, spat_rank, temp_rank, spat_decomp,
                 use_node_modem, use_graph, patch_sizes, strides, dropout, recon_mask_ratio,
                 spat_res=False, temp_res_ln=True, use_timestamp=True, adj_matrix=None,
                 te_ortho_loss=True, sp_ortho_loss=True, show_stack=False,
                 evolution_mode: str = 'koopman', use_low_rank: bool = True,
                 use_cross_scale_gating: bool = True):
        super().__init__()
        self.input_len = input_len
        self.output_len = output_len
        self.num_nodes = num_nodes
        self.num_feat = num_feat if use_timestamp else 1
        self.hidden_dim = hidden_dim
        self.spat_layers = spat_layers
        self.temp_layers = temp_layers
        self.spat_rank = spat_rank
        self.temp_rank = temp_rank
        self.spat_decomp = spat_decomp and use_graph
        self.use_node_modem = use_node_modem
        self.use_graph = use_graph
        self.dropout = dropout
        self.recon_mask_ratio = recon_mask_ratio
        self.spat_res = spat_res and use_graph
        self.temp_res_ln = temp_res_ln
        self.use_timestamp = use_timestamp
        self.te_ortho_loss = te_ortho_loss
        self.sp_ortho_loss = sp_ortho_loss
        self.evolution_mode = evolution_mode
        self.use_low_rank = use_low_rank
        self.use_cross_scale_gating = use_cross_scale_gating

        self.patch_sizes = patch_sizes
        self.strides = strides
        assert len(patch_sizes) == len(strides), "patch_sizes and strides must have the same length"
        n_layers = len(patch_sizes)

        self.revin = RevIN(num_features=num_nodes)

        self.x_time_proj = None
        self.y_time_proj = None
        self.merge_proj = None
        if self.use_timestamp and self.num_feat > 1:
            time_dim = self.num_feat - 1
            self.x_time_proj = MLP(time_dim, self.num_nodes, self.hidden_dim,
                                   3, False, self.dropout)
            self.y_time_proj = MLP(time_dim, self.num_nodes, self.hidden_dim,
                                   3, False, self.dropout)
            self.merge_proj = MLP(self.input_len, 2, self.hidden_dim,
                                  2, False, self.dropout)

        if adj_matrix is None:
            adj_matrix = torch.eye(num_nodes)
        if not isinstance(adj_matrix, torch.Tensor):
            adj_matrix = torch.tensor(adj_matrix, dtype=torch.float32)

        self.blocks = nn.ModuleList()
        prev_patch_num = None
        for i in range(n_layers):
            block = HKSTBlock(
                input_len=self.input_len,
                output_len=self.output_len,
                num_nodes=self.num_nodes,
                num_feat=self.num_feat,
                patch_size=self.patch_sizes[i],
                stride=self.strides[i],
                hidden_dim=self.hidden_dim,
                spat_layers=self.spat_layers,
                temp_layers=self.temp_layers,
                spat_rank=self.spat_rank,
                temp_rank=self.temp_rank,
                spat_decomp=self.spat_decomp,
                spat_res=self.spat_res,
                temp_res_ln=self.temp_res_ln,
                use_node_modem=self.use_node_modem,
                prev_patch_num=prev_patch_num,
                use_graph=self.use_graph,
                adj_matrix=adj_matrix,
                dropout=self.dropout,
                te_ortho_loss=self.te_ortho_loss,
                sp_ortho_loss=self.sp_ortho_loss,
                time_dim=(self.num_feat - 1) * self.patch_sizes[i],
                evolution_mode=self.evolution_mode,
                use_low_rank=self.use_low_rank,
                use_cross_scale_gating=self.use_cross_scale_gating,
            )
            self.blocks.append(block)
            prev_patch_num = block.patch_num_in

        self.show_stack = show_stack
        self.stack_outputs = []

    def forward(self, x_enc, x_time, y_time):
        # x_enc: [Batch, Time, Nodes], x_time: [Batch, Time, Ct]
        B, T, N = x_enc.shape

        x_enc = self.revin(x_enc, 'norm')
        x_enc_true = x_enc

        if self.use_timestamp:
            val = x_enc.unsqueeze(-1)  # [B, T, N, 1]
            time_emb = x_time.unsqueeze(2).expand(-1, -1, N, -1)  # [B, T, N, Ct]
            x_enc = torch.cat([val, time_emb], dim=-1)  # [B, T, N, C]
        else:
            x_enc = x_enc.unsqueeze(-1)  # [B, T, N, 1]
            x_time, y_time = None, None

        residual = x_enc.permute(0, 2, 1, 3)  # [B, N, T, C]
        forecast_accum = torch.zeros((B, N, self.output_len), device=x_enc.device)
        context = None

        if self.show_stack:
            self.stack_outputs = []

        for block in self.blocks:
            backcast, forecast, context = block(residual, context, y_time=y_time)

            signal = residual[..., 0] - backcast
            if self.use_timestamp:
                residual = torch.cat([signal.unsqueeze(-1), residual[..., 1:]], dim=-1)
            else:
                residual = signal.unsqueeze(-1)

            forecast_accum = forecast_accum + forecast

            if self.show_stack:
                b_out = backcast.detach().permute(0, 2, 1)
                f_out = forecast.detach().permute(0, 2, 1)
                self.stack_outputs.append(torch.cat([b_out, f_out], dim=1))

        out = forecast_accum.permute(0, 2, 1)  # [B, T, N]
        if self.y_time_proj is not None and y_time is not None:
            x_te = self.x_time_proj(x_time)
            merge_w = self.merge_proj((x_enc_true - x_te).permute(0, 2, 1)).unsqueeze(1)  # [B, 1, N, 2]
            y_te = self.y_time_proj(y_time)
            y_time_bias = torch.stack([out, y_te], dim=-1)  # [B, T, N, 2]
            y_time_bias = torch.sum(y_time_bias * merge_w, dim=-1)  # [B, T, N]
            out = out + y_time_bias
            if self.show_stack:
                f_bias = y_time_bias.detach()  # [B, T, N]
                b_bias = torch.zeros((B, self.input_len, N), device=f_bias.device)
                stack_bias = torch.cat([b_bias, f_bias], dim=1)
                for i in range(len(self.stack_outputs)):
                    self.stack_outputs[i] = self.stack_outputs[i] + stack_bias / len(self.blocks)

        out = self.revin(out, 'denorm')
        if self.show_stack and self.stack_outputs:
            n_stack = len(self.stack_outputs)
            for i in range(n_stack):
                if self.revin.affine:
                    self.stack_outputs[i] = self.stack_outputs[i] - self.revin.affine_bias / n_stack
                    self.stack_outputs[i] = self.stack_outputs[i] / (self.revin.affine_weight + self.revin.eps * self.revin.eps)
                self.stack_outputs[i] = self.stack_outputs[i] * self.revin.stdev + self.revin.mean / n_stack

        self.res_signal = residual[..., 0]
        if self.recon_mask_ratio < 1.0:
            mask = torch.rand(self.res_signal.shape, device=self.res_signal.device) < self.recon_mask_ratio
            self.res_signal = self.res_signal * mask

        return out

    def get_ortho_loss(self):
        total_loss = 0
        for block in self.blocks:
            total_loss += block.get_ortho_loss()
        return total_loss

    def get_residual_signal(self):
        return self.res_signal

    def get_spatial_matrices(self):
        if self.use_graph:
            return [b.spatial_aggregation.get_low_rank_gso() for b in self.blocks]
        else:
            return None

    def get_koopman_matrices(self):
        # Returns per-block (U, K, V) tuples. When evolution_mode='none', K is None.
        if self.use_graph and self.spat_decomp:
            return [[(b.koopman_evolution.get_matrix()),
                     (b.koopman_evolution_res.get_matrix())] for b in self.blocks]
        else:
            return [(b.koopman_evolution.get_matrix()) for b in self.blocks]

    def get_evolution_mode(self):
        return self.evolution_mode

    def get_stack_outputs(self):
        return self.stack_outputs
