from turtle import pos
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
from MPM_pytorch.generators.MPM_step import MPM_step


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
        self.friction = simulation_config.MPM_friction
        self.surface_tension = simulation_config.MPM_surface_tension
        self.tension_scaling = simulation_config.MPM_tension_scaling

        self.F_amplitude = simulation_config.MPM_F_amplitude
        self.C_amplitude = simulation_config.MPM_C_amplitude
        self.Jp_amplitude = simulation_config.MPM_Jp_amplitude

        siren_params = model_config.multi_siren_params

        self.identity = torch.eye(2, device=self.device).unsqueeze(0)
        # self.identity = self.identity.repeat(train_config.batch_size, 1, 1)

        # Extract parameters for each Siren
        F_siren_params = siren_params[0]  # [in_features, out_features, hidden_features, hidden_layers, first_omega_0, hidden_omega_0, outermost_linear]
        Jp_siren_params = siren_params[1]
        C_normal_siren_params = siren_params[2]

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
        
        self.MPM_P2G = MPM_P2G(aggr_type='add', device=device)

    def forward(self, x=[], data_id=[], k=[], trainer=[], batch_size=1):


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

        embedding = self.a[data_id.detach().long(), N.long(), :].squeeze()

        # Deformation Gradient F
        if 'F' in trainer:
            features = torch.cat((pos, frame), dim=1).detach()
            if self.F_amplitude > 0:
                # Perturbation around identity
                F = self.identity + self.F_amplitude * torch.tanh(self.siren_F(features).reshape(-1, 2, 2))
            elif self.F_amplitude == 0:
                # Direct prediction (no constraints)
                F = self.siren_F(features).reshape(-1, 2, 2)
            elif self.F_amplitude == -1:
                # SVD-constrained prediction (material-agnostic, prevents collapse)
                F_raw = self.siren_F(features).reshape(-1, 2, 2)
                U, S_f, Vh = torch.linalg.svd(F_raw)
                # Universal range covering all material types
                min_sig = 0.1   # Prevents collapse, allows compression
                max_sig = 3.0   # Allows jelly expansion, snow stays in [0.975, 1.0045]
                # Soft clamping for smooth gradients
                S_clamped = min_sig + (max_sig - min_sig) * torch.sigmoid(S_f)
                F = torch.bmm(U, torch.bmm(torch.diag_embed(S_clamped), Vh))

        # Affine Velocity Gradient C
        if 'C' in trainer:
            features = torch.cat((pos, frame), dim=1).detach()
            if self.C_amplitude > 0:
                # Perturbation around zero (small velocity gradients)
                C = self.C_amplitude * torch.tanh(self.siren_C(features).reshape(-1, 2, 2))
            elif self.C_amplitude == 0:
                # Direct prediction (no constraints)
                C = self.siren_C(features).reshape(-1, 2, 2)
            elif self.C_amplitude == -1:
                # Bounded prediction (typical velocity gradients)
                C_raw = self.siren_C(features).reshape(-1, 2, 2)
                # Velocity gradients typically in reasonable range
                max_C = 10.0  # Adjust based on dt and typical velocities
                C = max_C * torch.tanh(C_raw)  # C ∈ [-10, 10]

        # Plastic Deformation Jp
        if 'Jp' in trainer:
            features = torch.cat((pos, frame), dim=1).detach()
            if self.Jp_amplitude > 0:
                # Perturbation around 1.0 (no plastic deformation)
                Jp = 1.0 + self.Jp_amplitude * torch.tanh(self.siren_Jp(features).reshape(-1, 1))
            elif self.Jp_amplitude == 0:
                # Direct prediction (dangerous - can go negative!)
                Jp_raw = self.siren_Jp(features).reshape(-1, 1)
                Jp = torch.clamp(Jp_raw, min=1e-4, max=10.0)  # Safety clamp
            elif self.Jp_amplitude == -1:
                # Safe exponential parameterization (always positive)
                Jp_raw = self.siren_Jp(features).reshape(-1, 1)
                # Jp should be positive and typically in [0.5, 2.0]
                # Use exp to ensure positivity, centered around 1.0
                Jp = torch.exp(torch.clamp(Jp_raw, min=-1.0, max=1.0))
                # Jp ∈ [exp(-1), exp(1)] = [0.368, 2.718]

        X, V, C, F, Jp, T, M, S, GM, GV = MPM_step(self.MPM_P2G, pos, d_pos, C, F, Jp, T,
                                M, self.n_particles, self.n_grid,
                                self.delta_t, self.dx, self.inv_dx, self.mu_0, self.lambda_0,
                                self.p_vol, self.offsets, self.particle_offsets, 
                                self.expansion_factor, self.gravity, self.friction, 0, 
                                self.surface_tension, self.tension_scaling, False, False,
                                self.device)

        return X, C, F, Jp, S
