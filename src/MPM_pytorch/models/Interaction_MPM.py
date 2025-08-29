import torch_geometric as pyg
import torch_geometric.utils as pyg_utils
from MPM_pytorch.models.MLP import MLP
from MPM_pytorch.utils import to_numpy, reparameterize
from MPM_pytorch.models.Siren_Network import *
from MPM_pytorch.generators.MPM_P2G import MPM_P2G
from MPM_pytorch.utils import *
import torch_geometric.data as data
import torch_geometric.data as data

from MPM_pytorch.models.Affine_Particle import Affine_Particle


class Interaction_MPM(nn.Module):

    def __init__(self, config, device, aggr_type=None, bc_dpos=None, dimension=2):

        super(Interaction_MPM, self).__init__()

        simulation_config = config.simulation
        model_config = config.graph_model
        train_config = config.training

        self.device = device

        self.model = model_config.particle_model_name
        self.n_dataset = train_config.n_runs
        self.dimension = dimension
        self.n_particles = simulation_config.n_particles
        self.embedding_dim = model_config.embedding_dim

        self.input_size_nnr = model_config.input_size_nnr
        self.n_layers_nnr = model_config.n_layers_nnr
        self.hidden_dim_nnr = model_config.hidden_dim_nnr
        self.output_size_nnr = model_config.output_size_nnr
        self.outermost_linear_nnr = model_config.outermost_linear_nnr
        self.omega = model_config.omega

        self.n_particle_types = simulation_config.n_particle_types
        self.n_particles = simulation_config.n_particles
        self.n_grid = simulation_config.n_grid

        self.delta_t = simulation_config.delta_t
        self.n_frames = simulation_config.n_frames
        self.dx, self.inv_dx = 1 / self.n_grid, float(self.n_grid)
        self.grid_i, self.grid_j = torch.meshgrid(
            torch.arange(self.n_grid, device=device, dtype=torch.float32),
            torch.arange(self.n_grid, device=device, dtype=torch.float32),
            indexing='ij'
        )  # Shape: [n_grid, n_grid]
        self.grid_coords = self.dx * torch.stack([
            self.grid_i,  # x coordinates
            self.grid_j  # y coordinates
        ], dim=-1).reshape(-1, 2)  # Shape: [1024, 2]

        self.p_vol, self.p_rho = (self.dx * 0.5) ** 2, 1
        E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
        self.mu_0, self.lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

        self.offsets = torch.tensor([[i, j] for i in range(3) for j in range(3)],
                                    device=device, dtype=torch.float32)  # [9, 2]
        self.particle_offsets = self.offsets.unsqueeze(0).expand(self.n_particles, -1, -1)
        self.expansion_factor = simulation_config.MPM_expansion_factor
        self.gravity = simulation_config.MPM_gravity

        self.model_MPM_P2G = MPM_P2G(aggr_type='add', device=device)

        siren_params = model_config.multi_siren_params

        # Extract parameters for each Siren
        F_siren_params = siren_params[0]  # [in_features, out_features, hidden_features, hidden_layers, first_omega_0, hidden_omega_0, outermost_linear]
        Jp_siren_params = siren_params[1]
        C_normal_siren_params = siren_params[2]
        C_PDE_MPM_A_siren_params = siren_params[3]

        # Create Siren networks using config parameters
        self.siren_F = Siren(
            in_features=F_siren_params[0],
            out_features=F_siren_params[1],
            hidden_features=F_siren_params[2],
            hidden_layers=F_siren_params[3],
            first_omega_0=F_siren_params[4],
            hidden_omega_0=F_siren_params[5],
            outermost_linear=F_siren_params[6]
        ).to(device)

        self.siren_Jp = Siren(
            in_features=Jp_siren_params[0],
            out_features=Jp_siren_params[1],
            hidden_features=Jp_siren_params[2],
            hidden_layers=Jp_siren_params[3],
            first_omega_0=Jp_siren_params[4],
            hidden_omega_0=Jp_siren_params[5],
            outermost_linear=Jp_siren_params[6]
        ).to(device)

        if self.model == 'PDE_MPM_A':
            self.siren_C = Siren(
                in_features=C_PDE_MPM_A_siren_params[0],
                out_features=C_PDE_MPM_A_siren_params[1],
                hidden_features=C_PDE_MPM_A_siren_params[2],
                hidden_layers=C_PDE_MPM_A_siren_params[3],
                first_omega_0=C_PDE_MPM_A_siren_params[4],
                hidden_omega_0=C_PDE_MPM_A_siren_params[5],
                outermost_linear=C_PDE_MPM_A_siren_params[6]
            ).to(device)
        else:
            self.siren_C = Siren(
                in_features=C_normal_siren_params[0],
                out_features=C_normal_siren_params[1],
                hidden_features=C_normal_siren_params[2],
                hidden_layers=C_normal_siren_params[3],
                first_omega_0=C_normal_siren_params[4],
                hidden_omega_0=C_normal_siren_params[5],
                outermost_linear=C_normal_siren_params[6]
            ).to(device)


        mlp_params = model_config.multi_mlp_params

        # Extract parameters for each MLP
        mu_lambda_params = mlp_params[0]  # [input_size, output_size, n_layers, hidden_size, initialisation]
        sig_params = mlp_params[1]
        F_params = mlp_params[2]
        stress_params = mlp_params[3]

        # Create MLPs using config parameters
        self.MLP_mu_lambda = MLP(
            input_size=mu_lambda_params[0],
            output_size=mu_lambda_params[1],
            nlayers=mu_lambda_params[2],
            hidden_size=mu_lambda_params[3],
            device=device,
            initialisation=mu_lambda_params[4] if len(mu_lambda_params) > 4 else "normal"
        )

        self.MLP_sig = MLP(
            input_size=sig_params[0],
            output_size=sig_params[1],
            nlayers=sig_params[2],
            hidden_size=sig_params[3],
            device=device,
            initialisation=sig_params[4] if len(sig_params) > 4 else "ones"
        )

        self.MLP_F = MLP(
            input_size=F_params[0],
            output_size=F_params[1],
            nlayers=F_params[2],
            hidden_size=F_params[3],
            device=device,
            initialisation=F_params[4] if len(F_params) > 4 else "normal"
        )

        self.MLP_stress = MLP(
            input_size=stress_params[0],
            output_size=stress_params[1],
            nlayers=stress_params[2],
            hidden_size=stress_params[3],
            device=device,
            initialisation=stress_params[4] if len(stress_params) > 4 else "normal"
        )

        self.a = nn.Parameter(
            torch.tensor(np.ones((self.n_dataset, int(self.n_particles), self.embedding_dim)),
                         device=self.device,
                         requires_grad=True, dtype=torch.float32))

        self.GNN_C = Affine_Particle(aggr_type='mean', config=config, device=device, bc_dpos=bc_dpos,
                                     dimension=dimension)

    def forward(self, data=[], data_id=[], k=[], trainer=[], training=True):

        x, edge_index = data.x, data.edge_index
        self.data_id = data_id.long()

        N = x[:, 0:1]
        pos = x[:, 1:3]  # pos is the absolute position
        d_pos = x[:, 3:5]  # d_pos is the velocity
        C = x[:, 5:9].reshape(-1, 2, 2)  # C is the affine deformation gradient
        F = x[:, 9:13].reshape(-1, 2, 2)  # F is the deformation gradient
        Jp = x[:, 13:14]  # Jp is the Jacobian of the deformation gradient
        T = x[:, 14:15].long()  # T is the type of particle
        M = x[:, 15:16]  # M is the mass of the particle
        S = x[:, 16:20].reshape(-1, 2, 2)
        frame = k / self.n_frames

        embedding = self.a[self.data_id.detach(), N.long(), :].squeeze()

        if 'next' in trainer:
            C, F, Jp, S = self.MPM_engine(self.model_MPM_P2G, pos, d_pos, C, F,
                                          T, Jp, M, self.n_particles, self.n_grid,
                                          self.delta_t, self.dx, self.inv_dx, embedding,
                                          self.offsets, self.particle_offsets, self.grid_coords,
                                          self.expansion_factor, self.gravity, self.device)
        else:
            if 'C' in trainer:
                if self.model == 'PDE_MPM_A':
                    features = torch.cat((pos, d_pos, embedding, frame), dim=1).detach()
                else:
                    features = torch.cat((pos, d_pos, frame), dim=1).detach()
                C = self.siren_C(features)
            if 'F' in trainer:
                features = torch.cat((pos, frame), dim=1).detach()
                F = self.siren_F(features)
            if 'Jp' in trainer:
                features = torch.cat((pos, frame), dim=1).detach()
                Jp = self.siren_Jp(features)

        return C, F, Jp, S

    def MPM_engine(self, model_MPM_P2G, X, V, C, F, T, Jp, M, n_particles,
                   n_grid, dt, dx, inv_dx, embedding,
                   offsets, particle_offsets, grid_coords, expansion_factor,
                   gravity, device):
        """
        MPM substep implementation
        """

        p_mass = M.squeeze(-1)
        identity = torch.eye(2, device=device).unsqueeze(0).expand(F.shape[0], -1, -1)

        # Update deformation gradient: F = (I + dt * C) * F_old
        F = (identity + dt * C) @ F

        h = torch.exp(10 * (1.0 - Jp.squeeze()))

        mu_lambda = self.MLP_mu_lambda(torch.cat((embedding, h[:, None]), dim=1))
        mu, lambda_ = mu_lambda[:, 0:1], mu_lambda[:, 1:2]

        U, sig, Vh = torch.linalg.svd(F, driver='gesvdj')
        det_U = torch.det(U)
        det_Vh = torch.det(Vh)

        U, sig, Vh = self.correct_SVD(U, sig, Vh, det_U, det_Vh)

        original_sig = sig.clone()

        sig = self.MLP_sig(torch.cat((embedding, sig), dim=1)) ** 2

        # sig_plastic_ratio = self.MLP_sig_plastic_ratio(
        #     torch.cat((embedding, sig, det_U[:,None], det_Vh[:,None], mu, lambda_), dim=1))
        #
        # sig, plastic_ratio = sig_plastic_ratio[:, 0:2]**2, sig_plastic_ratio[:, 2:3]**2

        plastic_ratio = torch.prod(original_sig / sig, dim=1, keepdim=True)
        Jp = Jp * plastic_ratio
        J = torch.prod(sig, dim=1)
        sig_diag = torch.diag_embed(sig)

        F = self.MLP_F(
            torch.cat((embedding, J[:, None], sig_diag.reshape(-1, 4), U.reshape(-1, 4), Vh.reshape(-1, 4)), dim=1))

        F = F.reshape(-1, 2, 2)
        R = U @ Vh
        F_minus_R = F - R

        # S = self.MLP_stress(torch.cat((mu, lambda_, J[:,None], F.reshape(-1, 4), R.reshape(-1, 4)), dim=1))

        S = (2 * mu.unsqueeze(-1) * F_minus_R @ F.transpose(-2, -1) + identity * (
                    lambda_.squeeze() * J * (J - 1)).unsqueeze(-1).unsqueeze(-1))
        S = (-dt * self.p_vol * 4 * self.inv_dx * self.inv_dx) * S

        F = F.reshape(-1, 4)
        S = S.reshape(-1, 4)

        return C, F, Jp, S

    def correct_SVD(self, U, sig, Vh, det_U, det_Vh):
        """
        Correct the singular values to ensure they are positive.
        """
        neg_det_U = det_U < 0  # [n_particles] bool tensor
        neg_det_Vh = det_Vh < 0
        # Reshape masks for broadcasting
        neg_det_U_mask = neg_det_U.unsqueeze(-1).unsqueeze(-1)  # [n_particles,1,1]
        neg_det_sig_U_mask = neg_det_U.unsqueeze(-1)  # [n_particles,1]
        neg_det_Vh_mask = neg_det_Vh.unsqueeze(-1).unsqueeze(-1)  # [n_particles,1,1]
        neg_det_sig_Vh_mask = neg_det_Vh.unsqueeze(-1)  # [n_particles,1]
        # Flip signs on last columns/rows accordingly, out-of-place
        U = torch.where(
            neg_det_U_mask.expand_as(U),
            torch.cat([U[:, :, :-1], -U[:, :, -1:].clone()], dim=2),
            U
        )
        sig = torch.where(
            neg_det_sig_U_mask.expand_as(sig),
            torch.cat([sig[:, :-1], -sig[:, -1:].clone()], dim=1),
            sig
        )
        Vh = torch.where(
            neg_det_Vh_mask.expand_as(Vh),
            torch.cat([Vh[:, :-1, :], -Vh[:, -1:, :].clone()], dim=1),
            Vh
        )
        sig = torch.where(
            neg_det_sig_Vh_mask.expand_as(sig),
            torch.cat([sig[:, :-1], -sig[:, -1:].clone()], dim=1),
            sig
        )
        # Clamp singular values
        min_val = 1e-6
        sig = torch.where(
            sig < min_val,
            min_val + 0.01 * (sig - min_val),  # small slope below min_val
            sig
        )

        return U, sig, Vh