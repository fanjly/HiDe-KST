import torch
import torch.nn as nn
import torch.nn.functional as F
from models.hkst.modules import MLP, NodeModem, GCN


class KoopmanLayer(nn.Module):
    def __init__(self, num_nodes, input_dim, low_rank_dim, output_dim,
                 hidden_dim, hidden_layers, res_ln, use_node_modem, dropout, patch_num_out, time_dim=0,
                 evolution_mode: str = 'koopman', use_low_rank: bool = True):
        super().__init__()
        self.num_nodes = num_nodes
        self.input_dim = input_dim
        self.low_rank_dim = low_rank_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.res_ln = res_ln
        self.use_node_modem = use_node_modem
        self.dropout = dropout
        self.patch_num_out = patch_num_out
        self.time_dim = time_dim
        assert evolution_mode in ('koopman', 'none'), \
            f"evolution_mode must be 'koopman' or 'none', got {evolution_mode}"
        self.evolution_mode = evolution_mode
        self.use_low_rank = use_low_rank
        self.state_dim = self.low_rank_dim if self.use_low_rank else self.hidden_dim

        # Encoders and Decoders to/from the Koopman invariant subspace
        self.node_modulator = NodeModem(self.num_nodes) if self.use_node_modem else None
        self.encoder = MLP(self.input_dim, self.hidden_dim, self.hidden_dim,
                           self.hidden_layers, self.res_ln, self.dropout)
        self.decoder = MLP(self.hidden_dim, self.output_dim, self.hidden_dim,
                           self.hidden_layers, self.res_ln, self.dropout)
        self.node_demodulator = NodeModem(self.num_nodes) if self.use_node_modem else None

        # Time projection for forcing term
        if self.time_dim > 0:
            self.time_proj = MLP(self.time_dim, self.state_dim, self.hidden_dim,
                                 self.hidden_layers, self.res_ln, self.dropout)
            self.time_gate = nn.Sequential(
                nn.Linear(self.state_dim * 2, self.state_dim),
                nn.Sigmoid()
            )

        # map to/from low rank space
        if self.use_low_rank:
            self.U = nn.Linear(self.hidden_dim, self.low_rank_dim, bias=False)
            self.V = nn.Linear(self.low_rank_dim, self.hidden_dim, bias=False)
            nn.init.orthogonal_(self.U.weight)  # semi-orthogonal
            with torch.no_grad():
                self.V.weight.copy_(self.U.weight.T)
        else:
            # Full-rank control: evolve directly in the H-dimensional space.
            self.U = None
            self.V = None

        # The Koopman operator K
        if self.evolution_mode == 'koopman':
            self.K = nn.Parameter(torch.randn(self.state_dim, self.state_dim))
            nn.init.orthogonal_(self.K)
        else:
            # w/o Koopman ablation (Plan B): no linear evolution operator.
            # latent token is held constant across patches; temporal dynamics
            # must be carried by the forcing term (time projection) and the
            # downstream forecast head. This isolates the Koopman contribution.
            self.K = None

    def _get_powered_K(self):
        K = self.K
        K_powers = [K @ K]
        for _ in range(self.patch_num_out - 2):
            K_powers.append(K_powers[-1] @ K)
        K_powers = torch.stack(K_powers, dim=0)
        return K_powers   # [P - 1, r, r]

    def _to_lantent(self, x):
        # x: [Batch, Nodes, Time, Features]
        if self.use_node_modem:
            x = self.node_modulator(x)
        z = self.encoder(x)  # project to Koopman invariant subspace
        if self.use_low_rank:
            z = self.U(z)        # project to low rank space
        return z  # [Batch, Nodes, Time, r]

    def _to_observation(self, x):
        # x: [Batch, Nodes, Time, r]
        z = x
        if self.use_low_rank:
            z = self.V(x)        # project from low rank space
        z = self.decoder(z)  # project to observation space
        if self.use_node_modem:
            z = self.node_demodulator(z)
        return z  # [Batch, Nodes, Time, Features]

    def forward(self, x, y_time=None):
        # x: [Batch, Nodes, Time, Features]
        # y_time: [Batch, P_out, TimeDim] (Optional)
        z = self._to_lantent(x)

        if self.evolution_mode == 'koopman':
            # Backcast: shift the latent sequence forward by one step using K
            z_evolved = torch.matmul(z, self.K)
            z_back = self._to_observation(z_evolved[:, :, :-1, :])
            z_back = torch.cat((x[:, :, :1, :], z_back), dim=2)

            # Forecast: roll the last latent forward by powers of K
            if self.patch_num_out >= 2:
                K_powers = self._get_powered_K()  # [P - 1, r, r]
                # [B, N, r] @ [P - 1, r, r] -> [B, N, P - 1, r]
                z_lat_future = torch.einsum("bnr,prk->bnpk", z[:, :, -1, :], K_powers)
                z_lat_future = torch.cat((z_evolved[:, :, -1:, :], z_lat_future), dim=2)  # [B, N, P, r]
            else:
                z_lat_future = z_evolved[:, :, -1:, :]
        else:
            # 'none' mode (Plan B w/o Koopman): identity evolution.
            # Backcast: each latent token is mapped back to observation as-is.
            z_back = self._to_observation(z)
            # Forecast: replicate the last input latent for all P_out future patches.
            z_lat_future = z[:, :, -1:, :].expand(-1, -1, self.patch_num_out, -1).contiguous()

        if self.time_dim > 0 and y_time is not None:
            t_fore = self.time_proj(y_time).unsqueeze(1)  # [B, 1, P, r]
            t_fore_expanded = t_fore.expand_as(z_lat_future)
            fusion_input = torch.cat([z_lat_future, t_fore_expanded], dim=-1)
            gate = self.time_gate(fusion_input)
            z_lat_future = z_lat_future + gate * t_fore_expanded

        z_fore = self._to_observation(z_lat_future)

        return z_back, z_fore

    def get_ortho_loss(self):
        # constrain to be inverse by loss function tr(F @ F^T), F=U^T @ V - I
        # self.U.weight: [r, d]
        # self.V.weight: [d, r]
        if not self.use_low_rank:
            return 0.0
        ortho_loss = self.U.weight @ self.V.weight - torch.eye(self.low_rank_dim, device=self.U.weight.device)
        return torch.mean(ortho_loss ** 2)

    def get_matrix(self):
        # When evolution_mode='none', K is undefined; return None so downstream
        # analysis scripts can skip eigenvalue computation safely.
        K_data = self.K.data if self.K is not None else None
        u = self.U.weight.data if self.use_low_rank else None
        v = self.V.weight.data if self.use_low_rank else None
        return u, K_data, v


class AdaptiveSpatialFactorization(nn.Module):
    def __init__(self, num_nodes, rank, input_dim, output_dim, hidden_dim, hidden_layers,
                 res, dropout, adj_matrix=None):
        super().__init__()
        self.num_nodes = num_nodes
        self.rank = rank
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.hidden_dim = hidden_dim
        self.hidden_layers = hidden_layers
        self.res = res
        self.dropout = dropout

        self.P = nn.Parameter(torch.randn(num_nodes, self.rank))
        nn.init.orthogonal_(self.P)
        # maintain symmetry using orthogonal transformation

        self.gcn = GCN(self.input_dim, self.output_dim, self.hidden_dim,
                       self.hidden_layers, self.dropout)
        self.ln = nn.LayerNorm(self.output_dim)
        self.register_buffer("adj", adj_matrix)

    def forward(self, x):
        # x: [Batch, Nodes, Time, Features]
        A_low = self.P.t() @ (self.adj @ self.P)
        A_low = F.relu(A_low)

        z = torch.einsum("bnfc,nr->brfc", x, self.P)
        z = self.gcn(z, A_low)
        z = torch.einsum("brfc,rn->bnfc", z, self.P.t())

        if self.res:
            z = z + x
            z = self.ln(z)
            z = F.gelu(z)

        return z

    def get_ortho_loss(self):
        # constrain to be orthogonal by loss function tr(F @ F^T), F=P^T @ P - I
        # self.P: [N, r]
        ortho_loss = self.P.t() @ self.P - torch.eye(self.rank, device=self.P.device)
        return torch.mean(ortho_loss ** 2)

    def get_low_rank_gso(self):
        return F.relu(self.P.t() @ (self.adj @ self.P))
