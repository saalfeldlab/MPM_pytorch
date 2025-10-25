import torch
import torch_geometric as pyg


class MPM_P2G(pyg.nn.MessagePassing):
    def __init__(self, aggr_type='add', device='cpu'):
        super(MPM_P2G, self).__init__(aggr=aggr_type)
        self.device = device   
    
    def forward(self, data):
        x, edge_index, weights_per_edge, affine_per_edge, dpos_per_edge = (
            data.x, data.edge_index, data.weights_per_edge, 
            data.affine_per_edge, data.dpos_per_edge
        )
        mass = x[:, 0:1]
        d_pos = x[:, 1:3]
        pred = self.propagate(edge_index, mass=mass, d_pos=d_pos, 
                            weights=weights_per_edge, affine=affine_per_edge, 
                            dpos_per_edge=dpos_per_edge)
        return pred

    def message(self, mass_j, d_pos_j, weights, affine, dpos_per_edge):
        """
        weights: pre-computed kernel weights [n_edges]
        """
        # No need to compute weights - they're passed in!
        out_m = mass_j.squeeze(-1) * weights
        out_v = weights.unsqueeze(-1) * (mass_j * d_pos_j + 
                                        torch.bmm(affine, dpos_per_edge.unsqueeze(-1)).squeeze(-1))
        return torch.cat([out_m.unsqueeze(-1), out_v], dim=-1)
