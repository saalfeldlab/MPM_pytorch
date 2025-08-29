import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from MPM_pytorch.models.MLP import MLP
from MPM_pytorch.utils import to_numpy, reparameterize
from MPM_pytorch.models.Siren_Network import *


class Affine_Particle(pyg.nn.MessagePassing):
    """Interaction Network as proposed in this paper:
    https://proceedings.neurips.cc/paper/2016/hash/3147da8ab4a0437c15ef51a5cc7f2dc4-Abstract.html"""

    """
    Model learning the acceleration of particles as a function of their relative distance and relative velocities.
    The interaction function is defined by a MLP self.lin_edge
    The particle embedding is defined by a table self.a

    Inputs
    ----------
    data : a torch_geometric.data object

    Returns
    -------
    pred : float
        the acceleration of the particles (dimension 2)
    """

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Affine_Particle, self).__init__(aggr='mean')  # "Add" aggregation.

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.input_size = model_config.input_size
        self.output_size = model_config.output_size
        self.hidden_dim = model_config.hidden_dim
        self.n_layers = model_config.n_layers

        self.update_type = model_config.update_type
        self.n_layers_update = model_config.n_layers_update
        self.input_size_update = model_config.input_size_update
        self.hidden_dim_update = model_config.hidden_dim_update
        self.output_size_update = model_config.output_size_update

        self.model = model_config.particle_model_name
        self.n_dataset = train_config.n_runs
        self.dimension = dimension

        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim
        self.n_frames = simulation_config.n_frames
        self.prediction = model_config.prediction
        self.bc_dpos = bc_dpos
        self.max_radius = simulation_config.max_radius
        self.rotation_augmentation = train_config.rotation_augmentation
        self.reflection_augmentation = train_config.reflection_augmentation

        self.delta_t = simulation_config.delta_t
        self.noise_level = train_config.noise_level


        self.lin_edge = MLP(input_size=self.input_size, output_size=self.output_size, nlayers=self.n_layers,
                                hidden_size=self.hidden_dim, device=self.device)

        if self.update_type == 'mlp':
            self.lin_phi = MLP(input_size=self.input_size_update, output_size=self.output_size_update,
                               nlayers=self.n_layers_update,
                               hidden_size=self.hidden_dim_update, device=self.device)

        self.a = nn.Parameter(
            torch.tensor(np.ones((int(self.n_particles) , self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))


    def forward(self, data=[], training=True):
        x, edge_index = data.x, data.edge_index
        self.training = training

        pos = x[:, 1:self.dimension + 1]
        d_pos = x[:, self.dimension + 1:1 + 2 * self.dimension]

        particle_id = x[:, 0:1].long()
        embedding = self.a[particle_id, :].squeeze()

        if (self.noise_level>0) and self.training:
            # Add noise to the position and velocity vectors
            noise = torch.randn_like(pos) * self.noise_level
            pos += noise
            d_pos += noise / self.delta_t

        # Rotation Augmentation (Only during training)
        if self.rotation_augmentation and self.training:
            self.phi = torch.randn(1, dtype=torch.float32, requires_grad=False, device=self.device) * 2 * np.pi
            cos_phi = torch.cos(self.phi)
            sin_phi = torch.sin(self.phi)

            # Rotation matrix R
            R = torch.stack([
                torch.stack([cos_phi, sin_phi]),
                torch.stack([-sin_phi, cos_phi])
            ], dim=0).squeeze()
            self.rotation_matrix = R
            # Inverse rotation (transpose of R)
            R_T = R.T

            # Rotate velocity vectors
            d_pos[:, :2] = d_pos[:, :2] @ R

        # Message passing
        out = self.propagate(edge_index, pos=pos, d_pos=d_pos, embedding=embedding)

        # Optional update step (e.g., MLP)
        if self.update_type == 'mlp':
            out = self.lin_phi(out)

        # Un-rotate predicted Jacobians if present
        if self.rotation_augmentation and self.training:
            C_rotated = out[:, :4].reshape(-1, 2, 2)
            C_original = R_T @ C_rotated @ R
            out[:, :4] = C_original.reshape(-1, 4)

        return out

    def message(self, edge_index_i, edge_index_j, pos_i, pos_j, d_pos_i, d_pos_j, embedding_i):

        # distance normalized by the max radius
        r = torch.sqrt(torch.sum(self.bc_dpos(pos_j - pos_i) ** 2, dim=1)) / self.max_radius
        delta_pos = self.bc_dpos(pos_j - pos_i) / self.max_radius
        if self.rotation_augmentation & self.training:
            delta_pos[:, :2] = delta_pos[:, :2] @ self.rotation_matrix

        match self.model:
            case 'PDE_MPM':
                in_features = torch.cat((delta_pos, d_pos_i, d_pos_j), dim=-1)
            case 'PDE_MPM_A':
                in_features = torch.cat((delta_pos, d_pos_i, d_pos_j, embedding_i), dim=-1)

        out = self.lin_edge(in_features)

        return out

    def update(self, aggr_out):

        return aggr_out  # self.lin_node(aggr_out)

