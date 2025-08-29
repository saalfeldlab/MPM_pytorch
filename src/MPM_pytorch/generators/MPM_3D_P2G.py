import torch
import torch_geometric as pyg


class MPM_3D_P2G(pyg.nn.MessagePassing):
    def __init__(self, aggr_type='add', device='cpu'):
        super(MPM_3D_P2G, self).__init__(aggr=aggr_type)
        self.device = device

    def forward(self, data):
        x, edge_index, fx_per_edge, affine_per_edge, dpos_per_edge = (data.x, data.edge_index,
                                                                      data.fx_per_edge, data.affine_per_edge,
                                                                      data.dpos_per_edge)

        mass = x[:, 0:1]
        d_pos = x[:, 1:4]  # 3D positions instead of 2D

        pred = self.propagate(edge_index, mass=mass, d_pos=d_pos, fx=fx_per_edge, affine=affine_per_edge,
                              dpos_per_edge=dpos_per_edge)
        return pred

    def message(self, mass_j, d_pos_j, fx, affine, dpos_per_edge):
        # Each edge corresponds to one 3x3x3 offset (27 neighbors)
        n_edges = mass_j.size(0)
        offset_idx = torch.arange(n_edges, device=self.device) % 27

        # Convert flat index to 3D: 0->(0,0,0), 1->(0,0,1), 2->(0,0,2), 3->(0,1,0), etc.
        i_idx = offset_idx // 9  # z-level: [0,0,0,...,0,1,1,1,...,1,2,2,2,...,2] (9 zeros, 9 ones, 9 twos)
        temp = offset_idx % 9
        j_idx = temp // 3  # y-row: [0,0,0,1,1,1,2,2,2, 0,0,0,1,1,1,2,2,2, ...]
        k_idx = temp % 3  # x-col: [0,1,2,0,1,2,0,1,2, 0,1,2,0,1,2,0,1,2, ...]

        # Quadratic B-spline weights for 3D
        w_0 = 0.5 * (1.5 - fx) ** 2
        w_1 = 0.75 - (fx - 1) ** 2
        w_2 = 0.5 * (fx - 0.5) ** 2
        w = torch.stack([w_0, w_1, w_2], dim=1)  # [n_edges, 3, 3] for 3D

        # Select weights for each edge based on its 3D offset
        edge_indices = torch.arange(n_edges, device=self.device)
        weights = (w[edge_indices, i_idx, 0] *
                   w[edge_indices, j_idx, 1] *
                   w[edge_indices, k_idx, 2])  # Triple product for 3D

        out_m = mass_j.squeeze(-1) * weights

        # 3D affine contribution: affine is now [n_edges, 3, 3], dpos_per_edge is [n_edges, 3]
        out_v = weights.unsqueeze(-1) * (mass_j * d_pos_j + torch.bmm(affine, dpos_per_edge.unsqueeze(-1)).squeeze(-1))

        # Return [mass, vel_x, vel_y, vel_z] instead of [mass, vel_x, vel_y]
        return torch.cat([out_m.unsqueeze(-1), out_v], dim=-1)