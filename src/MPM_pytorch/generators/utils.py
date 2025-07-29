
from MPM_pytorch.generators import *
from MPM_pytorch.utils import *
from time import sleep
from scipy.spatial import Delaunay
from tifffile import imread, imwrite as imsave
from torch_geometric.utils import get_mesh_laplacian
from tqdm import trange
from torch_geometric.utils import dense_to_sparse
from scipy import stats
import seaborn as sns



def choose_model(config=[], W=[], device=[]):
    particle_model_name = config.graph_model.particle_model_name
    model_signal_name = config.graph_model.signal_model_name
    aggr_type = config.graph_model.aggr_type
    n_particles = config.simulation.n_particles
    delta_t = config.simulation.delta_t
    n_particle_types = config.simulation.n_particle_types
    short_term_plasticity_mode = config.simulation.short_term_plasticity_mode

    bc_pos, bc_dpos = choose_boundary_values(config.simulation.boundary)

    dimension = config.simulation.dimension
    max_radius = config.simulation.max_radius

    params = config.simulation.params
    p = torch.tensor(params, dtype=torch.float32, device=device).squeeze()

    # create GNN depending in type specified in config file
    match particle_model_name:
        case 'PDE_A' | 'PDE_ParticleField_A' | 'PDE_Cell_A' :
            if config.simulation.non_discrete_level>0:
                p = torch.ones(n_particle_types, 4, device=device) + torch.rand(n_particle_types, 4, device=device)
                pp=[]
                n_particle_types = len(params)
                for n in range(n_particle_types):
                    p[n] = torch.tensor(params[n])
                for n in range(n_particle_types):
                    if n==0:
                        pp=p[n].repeat(n_particles//n_particle_types,1)
                    else:
                        pp=torch.cat((pp,p[n].repeat(n_particles//n_particle_types,1)),0)
                p=pp.clone().detach()
                p=p+torch.randn(n_particles,4,device=device) * config.simulation.non_discrete_level
            sigma = config.simulation.sigma
            p = p if n_particle_types == 1 else torch.squeeze(p)
            func_p = config.simulation.func_params
            embedding_step = config.simulation.n_frames // 100
            model = PDE_A(aggr_type=aggr_type, p=p, func_p = func_p, sigma=sigma, bc_dpos=bc_dpos, dimension=dimension, embedding_step=embedding_step)
        case 'PDE_B' | 'PDE_ParticleField_B' | 'PDE_Cell_B' | 'PDE_Cell_B_area':  # comprised between 10 and 50
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_B_mass':
            final_cell_mass = torch.tensor(config.simulation.final_cell_mass, device=device)
            model = PDE_B_mass(aggr_type=aggr_type, p=p, final_mass = final_cell_mass, bc_dpos=bc_dpos)
        case 'PDE_B_bis':
            model = PDE_B_bis(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos)
        case 'PDE_G':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_G(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_GS':
            if params[0] == [-1]:
                p = np.linspace(0.5, 5, n_particle_types)
                p = torch.tensor(p, device=device)
            model = PDE_GS(aggr_type=aggr_type, p=p, clamp=config.training.clamp,
                          pred_limit=config.training.pred_limit, bc_dpos=bc_dpos)
        case 'PDE_E':
            model = PDE_E(aggr_type=aggr_type, p=p,
                          clamp=config.training.clamp, pred_limit=config.training.pred_limit,
                          prediction=config.graph_model.prediction, bc_dpos=bc_dpos)

        case 'PDE_F' |'PDE_F_A' | 'PDE_F_B' :
            model = PDE_F(aggr_type=aggr_type, p=torch.tensor(params, dtype=torch.float32, device=device), bc_dpos=bc_dpos,
                          dimension=dimension, delta_t=delta_t, max_radius=max_radius, field_type=config.graph_model.field_type)
        case 'PDE_K':
            p = params
            edges = np.random.choice(p[0], size=(n_particles, n_particles), p=p[1])
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)
            connection_matrix = torch.tensor(edges, dtype=torch.float32, device=device)
            model = PDE_K(aggr_type=aggr_type, connection_matrix=connection_matrix, bc_dpos=bc_dpos)

        case 'PDE_O':
            model = PDE_O(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos, beta=config.simulation.beta)
        case 'Maze':
            model = PDE_B(aggr_type=aggr_type, p=p, bc_dpos=bc_dpos)
        case _:
            model = PDE_Z(device=device)


    match config.simulation.phi:
        case 'tanh':
            phi=torch.tanh
        case 'relu':
            phi=torch.relu
        case 'sigmoid':
            phi=torch.sigmoid
        case _:
            phi=torch.sigmoid


    match model_signal_name:
        case 'PDE_N2':
            model = PDE_N2(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N3':
            model = PDE_N3(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N4':
            model = PDE_N4(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N5':
            model = PDE_N5(aggr_type=aggr_type, p=p, W=W, phi=phi)
        case 'PDE_N6':
            model = PDE_N6(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)
        case 'PDE_N7':
            model = PDE_N7(aggr_type=aggr_type, p=p, W=W, phi=phi, short_term_plasticity_mode = short_term_plasticity_mode)


    return model, bc_pos, bc_dpos


def initialize_random_values(n, device):
    return torch.ones(n, 1, device=device) + torch.rand(n, 1, device=device)

def init_MPM_shapes(
        geometry='cubes',  # 'cubes', 'discs', 'stars', 'letters'
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_particle_types=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        device='cpu'
):
    torch.manual_seed(seed)

    p_vol = (dx * 0.5) ** 2

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    v = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    C = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    F = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1, -1)
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    GM = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)
    GP = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Determine grid layout and spacing
    if n_shapes == 3:
        shape_row = group_indices
        shape_col = torch.zeros_like(group_indices)
        size, spacing, start_x, start_y = 0.1, 0.32, 0.3, 0.15
    elif n_shapes == 9:
        shape_row = group_indices // 3
        shape_col = group_indices % 3
        size, spacing, start_x, start_y = 0.075, 0.25, 0.2, 0.375
    elif n_shapes == 16:
        shape_row = group_indices // 4
        shape_col = group_indices % 4
        size, spacing, start_x, start_y = 0.04, 0.2, 0.2, 0.2
    elif n_shapes == 25:
        shape_row = group_indices // 5
        shape_col = group_indices % 5
        size, spacing, start_x, start_y = 0.07, 0.16, 0.15, 0.2
    elif n_shapes == 36:
        shape_row = group_indices // 6
        shape_col = group_indices % 6
        size, spacing, start_x, start_y = 0.035, 0.13, 0.1, 0.14
    else:
        # General case: try to make a square grid
        grid_size = int(n_shapes ** 0.5)
        shape_row = group_indices // grid_size
        shape_col = group_indices % grid_size
        size, spacing = 0.4 / grid_size, 0.8 / grid_size
        start_x, start_y = 0.1 + size, 0.1 + size

    center_x = start_x + spacing * shape_col.float()
    center_y = start_y + spacing * shape_row.float()

    # Random rotation angles for each shape (except discs)
    shape_rotations = torch.rand(n_shapes, device=device) * 2 * torch.pi

    # Define letter shapes as line segments (relative to center, normalized to [-1,1])
    def get_letter_segments(letter):
        if letter == 'A':
            return [
                [[-0.5, -1], [0, 1]],  # Left diagonal
                [[0.5, -1], [0, 1]],  # Right diagonal
                [[-0.25, 0], [0.25, 0]]  # Crossbar
            ]
        elif letter == 'E':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Vertical line
                [[-0.5, 1], [0.5, 1]],  # Top horizontal
                [[-0.5, 0], [0.2, 0]],  # Middle horizontal
                [[-0.5, -1], [0.5, -1]]  # Bottom horizontal
            ]
        elif letter == 'F':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Vertical line
                [[-0.5, 1], [0.5, 1]],  # Top horizontal
                [[-0.5, 0], [0.2, 0]]  # Middle horizontal
            ]
        elif letter == 'H':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Left vertical
                [[0.5, -1], [0.5, 1]],  # Right vertical
                [[-0.5, 0], [0.5, 0]]  # Crossbar
            ]
        elif letter == 'I':
            return [
                [[-0.3, 1], [0.3, 1]],  # Top horizontal
                [[0, 1], [0, -1]],  # Vertical line
                [[-0.3, -1], [0.3, -1]]  # Bottom horizontal
            ]
        elif letter == 'L':
            return [
                [[-0.5, -1], [-0.5, 1]],  # Vertical line
                [[-0.5, -1], [0.5, -1]]  # Bottom horizontal
            ]
        elif letter == 'T':
            return [
                [[-0.5, 1], [0.5, 1]],  # Top horizontal
                [[0, 1], [0, -1]]  # Vertical line
            ]
        else:  # Default to 'O' (circle-like)
            # Approximate circle with 8 line segments
            angles = torch.linspace(0, 2 * torch.pi, 9)
            segments = []
            for i in range(8):
                x1, y1 = 0.5 * torch.cos(angles[i]), 0.5 * torch.sin(angles[i])
                x2, y2 = 0.5 * torch.cos(angles[i + 1]), 0.5 * torch.sin(angles[i + 1])
                segments.append([[x1.item(), y1.item()], [x2.item(), y2.item()]])
            return segments

    # Generate particles based on geometry
    if geometry == 'cubes':
        # Generate cube particles relative to center
        rel_x = torch.rand(n_particles, device=device) * (size * 2) - size
        rel_y = torch.rand(n_particles, device=device) * (size * 2) - size

        # Apply rotation to each cube
        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = start_idx + group_size

            angle = shape_rotations[shape_idx]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)

            # Rotate relative positions
            rotated_x = rel_x[start_idx:end_idx] * cos_a - rel_y[start_idx:end_idx] * sin_a
            rotated_y = rel_x[start_idx:end_idx] * sin_a + rel_y[start_idx:end_idx] * cos_a

            # Add center position
            x[start_idx:end_idx, 0] = center_x[start_idx] + rotated_x
            x[start_idx:end_idx, 1] = center_y[start_idx] + rotated_y

    elif geometry == 'discs':
        # Use the better circular generation
        outer_radius = size

        particles_per_shape = group_size
        valid_particles = []

        for shape_idx in range(n_shapes):
            shape_particles = []

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]

            # Generate particles in circular pattern
            for _ in range(particles_per_shape):
                r_test = torch.rand(1, device=device).sqrt() * outer_radius
                theta_test = torch.rand(1, device=device) * 2 * torch.pi

                px = shape_center_x + r_test * torch.cos(theta_test)
                py = shape_center_y + r_test * torch.sin(theta_test)
                shape_particles.append([px.item(), py.item()])

            valid_particles.extend(shape_particles)

        disc_positions = torch.tensor(valid_particles[:n_particles], device=device)
        x[:, 0] = disc_positions[:, 0]
        x[:, 1] = disc_positions[:, 1]

    elif geometry == 'stars':
        outer_radius = size
        inner_radius = outer_radius * 0.4
        n_points = 5

        particles_per_shape = group_size
        valid_particles = []

        for shape_idx in range(n_shapes):
            shape_particles = []

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]

            # Apply rotation to star
            rotation_angle = shape_rotations[shape_idx]

            # Create 5-pointed star vertices with rotation
            star_angles = torch.linspace(0, 2 * torch.pi, n_points * 2 + 1, device=device)[:-1] + rotation_angle
            star_radii = torch.zeros_like(star_angles)
            star_radii[::2] = outer_radius  # Outer points
            star_radii[1::2] = inner_radius  # Inner points

            # Create star vertices for this shape
            star_x = shape_center_x + star_radii * torch.cos(star_angles)
            star_y = shape_center_y + star_radii * torch.sin(star_angles)

            # Fill star by generating particles in triangular sectors
            particles_per_triangle = particles_per_shape // n_points

            for i in range(n_points):
                # Each star triangle: center -> outer point -> inner point -> next outer point
                p1_x, p1_y = shape_center_x, shape_center_y  # Center
                p2_x, p2_y = star_x[i * 2], star_y[i * 2]  # Outer point
                p3_x, p3_y = star_x[(i * 2 + 1) % (n_points * 2)], star_y[(i * 2 + 1) % (n_points * 2)]  # Inner point
                p4_x, p4_y = star_x[(i * 2 + 2) % (n_points * 2)], star_y[
                    (i * 2 + 2) % (n_points * 2)]  # Next outer point

                # Fill triangle (center, outer, inner)
                for _ in range(particles_per_triangle // 2):
                    r1, r2 = torch.rand(2, device=device)
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2

                    px = p1_x + r1 * (p2_x - p1_x) + r2 * (p3_x - p1_x)
                    py = p1_y + r1 * (p2_y - p1_y) + r2 * (p3_y - p1_y)
                    shape_particles.append([px.item(), py.item()])

                # Fill triangle (center, inner, next outer)
                for _ in range(particles_per_triangle // 2):
                    r1, r2 = torch.rand(2, device=device)
                    if r1 + r2 > 1:
                        r1, r2 = 1 - r1, 1 - r2

                    px = p1_x + r1 * (p3_x - p1_x) + r2 * (p4_x - p1_x)
                    py = p1_y + r1 * (p3_y - p1_y) + r2 * (p4_y - p1_y)
                    shape_particles.append([px.item(), py.item()])

            # Fill any remaining particles
            while len(shape_particles) < particles_per_shape:
                r_fill = torch.rand(1, device=device).sqrt() * inner_radius * 0.5
                theta_fill = torch.rand(1, device=device) * 2 * torch.pi
                px = shape_center_x + r_fill * torch.cos(theta_fill)
                py = shape_center_y + r_fill * torch.sin(theta_fill)
                shape_particles.append([px.item(), py.item()])

            valid_particles.extend(shape_particles[:particles_per_shape])

        star_positions = torch.tensor(valid_particles[:n_particles], device=device)
        x[:, 0] = star_positions[:, 0]
        x[:, 1] = star_positions[:, 1]

    elif geometry == 'letters':
        # Available letters
        letters = ['A', 'E', 'F', 'H', 'I', 'L', 'T', 'O']

        particles_per_shape = group_size
        valid_particles = []

        for shape_idx in range(n_shapes):
            shape_particles = []

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]

            # Choose random letter for this shape
            letter = letters[torch.randint(0, len(letters), (1,)).item()]
            segments = get_letter_segments(letter)

            # Apply rotation
            rotation_angle = shape_rotations[shape_idx]
            cos_a, sin_a = torch.cos(rotation_angle), torch.sin(rotation_angle)

            # Sample particles along letter segments
            particles_per_segment = particles_per_shape // len(segments)

            for segment in segments:
                x1, y1 = segment[0]
                x2, y2 = segment[1]

                for _ in range(particles_per_segment):
                    # Sample point along line segment
                    t = torch.rand(1, device=device)
                    rel_x = x1 + t * (x2 - x1)
                    rel_y = y1 + t * (y2 - y1)

                    # Scale by size
                    rel_x *= size
                    rel_y *= size

                    # Apply rotation
                    rotated_x = rel_x * cos_a - rel_y * sin_a
                    rotated_y = rel_x * sin_a + rel_y * cos_a

                    # Add center position
                    px = shape_center_x + rotated_x
                    py = shape_center_y + rotated_y
                    shape_particles.append([px.item(), py.item()])

            # Fill any remaining particles near center
            while len(shape_particles) < particles_per_shape:
                r_fill = torch.rand(1, device=device) * size * 0.1
                theta_fill = torch.rand(1, device=device) * 2 * torch.pi
                px = shape_center_x + r_fill * torch.cos(theta_fill)
                py = shape_center_y + r_fill * torch.sin(theta_fill)
                shape_particles.append([px.item(), py.item()])

            valid_particles.extend(shape_particles[:particles_per_shape])

        letter_positions = torch.tensor(valid_particles[:n_particles], device=device)
        x[:, 0] = letter_positions[:, 0]
        x[:, 1] = letter_positions[:, 1]

    # Random materials for each shape
    if n_particle_types> 1:
        shape_materials = torch.randperm(n_shapes, device=device) % n_particle_types
        T = shape_materials[group_indices].unsqueeze(1).int()

    # Calculate mass based on material type
    # Material 0: water (density = 1.0)
    # Material 1: jelly (density = 0.5, twice lighter than water)
    # Material 2: snow (density = 0.25, four times lighter than water)
    material_densities = torch.tensor(rho_list, device=device)
    particle_densities = material_densities[T.squeeze()]
    M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device) * particle_densities.unsqueeze(1)

    # Random velocity per shape
    shape_velocities = (torch.rand(n_shapes, 2, device=device) - 0.5) * 4.0
    v = shape_velocities[group_indices]

    # Object ID for each particle
    ID = group_indices.unsqueeze(1).int()
    id_permutation = torch.randperm(n_shapes, device=device)
    ID = id_permutation[ID.squeeze()].unsqueeze(1)


    return N, x, v, C, F, T, Jp, M, S, ID


def init_MPM_3D_shapes(
        geometry='cubes',  # 'cubes' or 'spheres' only
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_particle_types=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        device='cpu'
):
    torch.manual_seed(seed)

    # 3D volume instead of 2D area
    p_vol = (dx * 0.5) ** 3

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 3), dtype=torch.float32, device=device)  # 3D positions
    v = torch.zeros((n_particles, 3), dtype=torch.float32, device=device)  # 3D velocities (will be set to random)
    C = torch.zeros((n_particles, 3, 3), dtype=torch.float32, device=device)  # 3x3 affine matrix
    F = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1,
                                                                             -1)  # 3x3 deformation gradient
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 3, 3), dtype=torch.float32, device=device)  # 3x3 stress tensor

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Generate random rotations for each shape (3D Euler angles)
    shape_rotations_x = torch.rand(n_shapes, device=device) * 2 * torch.pi  # Roll
    shape_rotations_y = torch.rand(n_shapes, device=device) * 2 * torch.pi  # Pitch
    shape_rotations_z = torch.rand(n_shapes, device=device) * 2 * torch.pi  # Yaw

    # Determine 3D grid layout and spacing
    if n_shapes == 27:
        # General case: try to make a cubic grid
        grid_size = int(round(n_shapes ** (1 / 3)))
        if grid_size ** 3 < n_shapes:
            grid_size += 1

        shape_depth = group_indices // (grid_size * grid_size)
        temp = group_indices % (grid_size * grid_size)
        shape_row = temp // grid_size
        shape_col = temp % grid_size

        size = 0.3 / (grid_size + 1)
        spacing_x = spacing_y = spacing_z = 0.6 / grid_size
        start_x = start_y = start_z = 0.2

    # Calculate center positions for each shape
    center_x = start_x + shape_col.float() * spacing_x
    center_y = start_y + shape_row.float() * spacing_y
    center_z = start_z + shape_depth.float() * spacing_z

    # Create 3D rotation matrices for each shape
    def create_rotation_matrix(roll, pitch, yaw):
        """Create 3D rotation matrix from Euler angles"""
        cos_r, sin_r = torch.cos(roll), torch.sin(roll)
        cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
        cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

        # Rotation matrices for each axis
        R_x = torch.stack([
            torch.stack([torch.ones_like(roll), torch.zeros_like(roll), torch.zeros_like(roll)]),
            torch.stack([torch.zeros_like(roll), cos_r, -sin_r]),
            torch.stack([torch.zeros_like(roll), sin_r, cos_r])
        ])

        R_y = torch.stack([
            torch.stack([cos_p, torch.zeros_like(pitch), sin_p]),
            torch.stack([torch.zeros_like(pitch), torch.ones_like(pitch), torch.zeros_like(pitch)]),
            torch.stack([-sin_p, torch.zeros_like(pitch), cos_p])
        ])

        R_z = torch.stack([
            torch.stack([cos_y, -sin_y, torch.zeros_like(yaw)]),
            torch.stack([sin_y, cos_y, torch.zeros_like(yaw)]),
            torch.stack([torch.zeros_like(yaw), torch.zeros_like(yaw), torch.ones_like(yaw)])
        ])

        # Combined rotation: R = R_z @ R_y @ R_x
        R = torch.matmul(torch.matmul(R_z.permute(2, 0, 1), R_y.permute(2, 0, 1)), R_x.permute(2, 0, 1))
        return R

    rotation_matrices = create_rotation_matrix(shape_rotations_x, shape_rotations_y, shape_rotations_z)

    # Generate particles within each shape
    if geometry == 'cubes':
        # Generate particles in cubic volumes with rotation
        particles_per_dim = int(round((group_size) ** (1 / 3)))
        if particles_per_dim ** 3 < group_size:
            particles_per_dim += 1

        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = min(start_idx + group_size, n_particles)
            actual_particles = end_idx - start_idx

            # Generate relative positions in cube
            rel_positions = torch.zeros((actual_particles, 3), device=device)
            for i in range(actual_particles):
                # 3D indexing within cube
                z_idx = i // (particles_per_dim * particles_per_dim)
                temp = i % (particles_per_dim * particles_per_dim)
                y_idx = temp // particles_per_dim
                x_idx = temp % particles_per_dim

                # Normalize to [-0.5, 0.5] range then scale
                local_x = (x_idx / max(particles_per_dim - 1, 1) - 0.5) * size
                local_y = (y_idx / max(particles_per_dim - 1, 1) - 0.5) * size
                local_z = (z_idx / max(particles_per_dim - 1, 1) - 0.5) * size

                rel_positions[i] = torch.tensor([local_x, local_y, local_z], device=device)

            # Apply 3D rotation
            rotated_positions = torch.matmul(rel_positions, rotation_matrices[shape_idx].T)

            # Add center position
            x[start_idx:end_idx, 0] = center_x[start_idx] + rotated_positions[:, 0]
            x[start_idx:end_idx, 1] = center_y[start_idx] + rotated_positions[:, 1]
            x[start_idx:end_idx, 2] = center_z[start_idx] + rotated_positions[:, 2]

    elif geometry == 'spheres':
        # Generate particles in spherical volumes
        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = min(start_idx + group_size, n_particles)
            actual_particles = end_idx - start_idx

            shape_center = torch.tensor([center_x[start_idx], center_y[start_idx], center_z[start_idx]], device=device)

            for i in range(actual_particles):
                # Generate random point in unit sphere using rejection sampling
                while True:
                    rand_pos = torch.rand(3, device=device) * 2 - 1  # [-1, 1]³
                    if torch.sum(rand_pos ** 2) <= 1.0:
                        break

                # Scale by size and translate to center
                x[start_idx + i] = shape_center + rand_pos * size * 0.75

    else:  # Default to cubes
        # Same as cubes case
        particles_per_dim = int(round((group_size) ** (1 / 3)))
        if particles_per_dim ** 3 < group_size:
            particles_per_dim += 1

        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = min(start_idx + group_size, n_particles)
            actual_particles = end_idx - start_idx

            rel_positions = torch.zeros((actual_particles, 3), device=device)
            for i in range(actual_particles):
                z_idx = i // (particles_per_dim * particles_per_dim)
                temp = i % (particles_per_dim * particles_per_dim)
                y_idx = temp // particles_per_dim
                x_idx = temp % particles_per_dim

                local_x = (x_idx / max(particles_per_dim - 1, 1) - 0.5) * size
                local_y = (y_idx / max(particles_per_dim - 1, 1) - 0.5) * size
                local_z = (z_idx / max(particles_per_dim - 1, 1) - 0.5) * size

                rel_positions[i] = torch.tensor([local_x, local_y, local_z], device=device)

            # Apply rotation
            rotated_positions = torch.matmul(rel_positions, rotation_matrices[shape_idx].T)

            x[start_idx:end_idx, 0] = center_x[start_idx] + rotated_positions[:, 0]
            x[start_idx:end_idx, 1] = center_y[start_idx] + rotated_positions[:, 1]
            x[start_idx:end_idx, 2] = center_z[start_idx] + rotated_positions[:, 2]

    # Random materials for each shape
    if n_particle_types > 1:
        shape_materials = torch.randperm(n_shapes, device=device) % n_particle_types
        T = shape_materials[group_indices].unsqueeze(1).int()
    else:
        T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)

    # Calculate mass based on material type and density
    material_densities = torch.tensor(rho_list, device=device)
    if len(rho_list) > 0:
        particle_densities = material_densities[T.squeeze().clamp(0, len(rho_list) - 1)]
        M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device) * particle_densities.unsqueeze(1)
    else:
        M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device)

    # Random velocity per shape (3D)
    shape_velocities = (torch.rand(n_shapes, 3, device=device) - 0.5) * 4.0  # Random 3D velocities
    v = shape_velocities[group_indices]

    # Object ID for each particle with random permutation
    ID = group_indices.unsqueeze(1).int()
    if n_shapes > 1:
        id_permutation = torch.randperm(n_shapes, device=device)
        ID = id_permutation[ID.squeeze()].unsqueeze(1)

    return N, x, v, C, F, T, Jp, M, S, ID


def init_MPM_3D_cells(
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        nucleus_ratio=0.6,  # nucleus radius / total radius
        device='cpu'
):
    torch.manual_seed(seed)

    # 3D volume instead of 2D area
    p_vol = (dx * 0.5) ** 3

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 3), dtype=torch.float32, device=device)  # 3D positions
    v = torch.zeros((n_particles, 3), dtype=torch.float32, device=device)  # 3D velocities
    C = torch.zeros((n_particles, 3, 3), dtype=torch.float32, device=device)  # 3x3 affine matrix
    F = torch.eye(3, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1,
                                                                             -1)  # 3x3 deformation gradient
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 3, 3), dtype=torch.float32, device=device)  # 3x3 stress tensor

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Determine 3D grid layout and spacing
    if n_shapes == 27:
        # 3x3x3 grid
        grid_size = 3
        shape_depth = group_indices // (grid_size * grid_size)
        temp = group_indices % (grid_size * grid_size)
        shape_row = temp // grid_size
        shape_col = temp % grid_size
        size, spacing, start_x, start_y, start_z = 0.08, 0.25, 0.2, 0.2, 0.2
    else:
        # General case: try to make a cubic grid
        grid_size = int(round(n_shapes ** (1 / 3)))
        if grid_size ** 3 < n_shapes:
            grid_size += 1

        shape_depth = group_indices // (grid_size * grid_size)
        temp = group_indices % (grid_size * grid_size)
        shape_row = temp // grid_size
        shape_col = temp % grid_size

        size = 0.3 / (grid_size + 1)
        spacing = 0.6 / grid_size
        start_x = start_y = start_z = 0.2

    center_x = start_x + spacing * shape_col.float()
    center_y = start_y + spacing * shape_row.float()
    center_z = start_z + spacing * shape_depth.float()

    # Generate 3D cell particles (spheres with nucleus and membrane)
    outer_radius = size
    nucleus_radius = outer_radius * nucleus_ratio

    particles_per_shape = group_size
    valid_particles = []
    particle_materials = []

    # Calculate particles distribution: nucleus volume vs membrane volume
    nucleus_volume = (4 / 3) * torch.pi * nucleus_radius ** 3
    membrane_volume = (4 / 3) * torch.pi * (outer_radius ** 3 - nucleus_radius ** 3)
    total_volume = nucleus_volume + membrane_volume

    particles_nucleus = int(particles_per_shape * nucleus_volume / total_volume)
    particles_membrane = particles_per_shape - particles_nucleus

    for shape_idx in range(n_shapes):
        shape_particles = []
        shape_materials = []

        shape_center_x = center_x[shape_idx * group_size]
        shape_center_y = center_y[shape_idx * group_size]
        shape_center_z = center_z[shape_idx * group_size]

        # Generate nucleus particles (material 0 - liquid)
        for _ in range(particles_nucleus):
            # Generate random point in sphere using rejection sampling
            while True:
                rand_pos = torch.rand(3, device=device) * 2 - 1  # [-1, 1]³
                if torch.sum(rand_pos ** 2) <= 1.0:
                    break

            # Scale by nucleus radius and translate to center
            px = shape_center_x + rand_pos[0] * nucleus_radius
            py = shape_center_y + rand_pos[1] * nucleus_radius
            pz = shape_center_z + rand_pos[2] * nucleus_radius

            shape_particles.append([px.item(), py.item(), pz.item()])
            shape_materials.append(0)  # Material 0 for nucleus

        # Generate membrane particles (material 1 - jelly)
        for _ in range(particles_membrane):
            # Generate random point on unit sphere
            while True:
                rand_pos = torch.rand(3, device=device) * 2 - 1  # [-1, 1]³
                r_sq = torch.sum(rand_pos ** 2)
                if r_sq <= 1.0 and r_sq > 0:
                    break

            # Normalize to unit sphere
            rand_pos = rand_pos / torch.sqrt(r_sq)

            # Generate radius with proper volume weighting for spherical shell
            u = torch.rand(1, device=device)
            r_cubed = u * (outer_radius ** 3 - nucleus_radius ** 3) + nucleus_radius ** 3
            r = r_cubed ** (1 / 3)

            # Scale and translate to center
            px = shape_center_x + rand_pos[0] * r
            py = shape_center_y + rand_pos[1] * r
            pz = shape_center_z + rand_pos[2] * r

            shape_particles.append([px.item(), py.item(), pz.item()])
            shape_materials.append(1)  # Material 1 for membrane

        valid_particles.extend(shape_particles)
        particle_materials.extend(shape_materials)

    cell_positions = torch.tensor(valid_particles[:n_particles], device=device)
    x[:, 0] = cell_positions[:, 0]
    x[:, 1] = cell_positions[:, 1]
    x[:, 2] = cell_positions[:, 2]

    # Set materials based on nucleus/membrane assignment
    T = torch.tensor(particle_materials[:n_particles], device=device).unsqueeze(1).int()

    # Calculate mass based on material type
    material_densities = torch.tensor(rho_list, device=device)
    if len(rho_list) > 0:
        particle_densities = material_densities[T.squeeze().clamp(0, len(rho_list) - 1)]
        M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device) * particle_densities.unsqueeze(1)
    else:
        M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device)

    # Random velocity per shape (3D)
    shape_velocities = (torch.rand(n_shapes, 3, device=device) - 0.5) * 4.0
    v = shape_velocities[group_indices]

    # Object ID for each particle
    ID = group_indices.unsqueeze(1).int()
    if n_shapes > 1:
        id_permutation = torch.randperm(n_shapes, device=device)
        ID = id_permutation[ID.squeeze()].unsqueeze(1)

    return N, x, v, C, F, T, Jp, M, S, ID

def init_MPM_cells(
        n_shapes=3,
        seed=42,
        n_particles=[],
        n_grid=[],
        dx=[],
        rho_list=[],
        nucleus_ratio=0.6,  # nucleus radius / total radius
        device='cpu'
):
    torch.manual_seed(seed)

    p_vol = (dx * 0.5) ** 2

    N = torch.arange(n_particles, dtype=torch.float32, device=device)[:, None]
    x = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    v = torch.zeros((n_particles, 2), dtype=torch.float32, device=device)
    C = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    F = torch.eye(2, dtype=torch.float32, device=device).unsqueeze(0).expand(n_particles, -1, -1)
    T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)
    Jp = torch.ones((n_particles, 1), dtype=torch.float32, device=device)
    S = torch.zeros((n_particles, 2, 2), dtype=torch.float32, device=device)
    GM = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)
    GP = torch.zeros((n_grid, n_grid), dtype=torch.float32, device=device)

    group_size = n_particles // n_shapes
    group_indices = torch.arange(n_particles, device=device) // group_size

    # Determine grid layout and spacing
    if n_shapes == 3:
        shape_row = group_indices
        shape_col = torch.zeros_like(group_indices)
        size, spacing, start_x, start_y = 0.1, 0.32, 0.3, 0.15
    elif n_shapes == 9:
        shape_row = group_indices // 3
        shape_col = group_indices % 3
        size, spacing, start_x, start_y = 0.075, 0.25, 0.2, 0.375
    elif n_shapes == 16:
        shape_row = group_indices // 4
        shape_col = group_indices % 4
        size, spacing, start_x, start_y = 0.04, 0.2, 0.2, 0.2
    elif n_shapes == 25:
        shape_row = group_indices // 5
        shape_col = group_indices % 5
        size, spacing, start_x, start_y = 0.035, 0.16, 0.15, 0.2
    elif n_shapes == 36:
        shape_row = group_indices // 6
        shape_col = group_indices % 6
        size, spacing, start_x, start_y = 0.035, 0.13, 0.1, 0.14
    else:
        # General case: try to make a square grid
        grid_size = int(n_shapes ** 0.5)
        shape_row = group_indices // grid_size
        shape_col = group_indices % grid_size
        size, spacing = 0.4 / grid_size, 0.8 / grid_size
        start_x, start_y = 0.1 + size, 0.1 + size

    center_x = start_x + spacing * shape_col.float()
    center_y = start_y + spacing * shape_row.float()

    # Generate cell particles (discs with nucleus and membrane)
    outer_radius = size
    nucleus_radius = outer_radius * nucleus_ratio

    particles_per_shape = group_size
    valid_particles = []
    particle_materials = []

    # Calculate particles distribution: nucleus area vs membrane area
    nucleus_area = torch.pi * nucleus_radius ** 2
    membrane_area = torch.pi * (outer_radius ** 2 - nucleus_radius ** 2)
    total_area = nucleus_area + membrane_area

    particles_nucleus = int(particles_per_shape * nucleus_area / total_area)
    particles_membrane = particles_per_shape - particles_nucleus

    for shape_idx in range(n_shapes):
        shape_particles = []
        shape_materials = []

        shape_center_x = center_x[shape_idx * group_size]
        shape_center_y = center_y[shape_idx * group_size]

        # Generate nucleus particles (material 0 - liquid)
        for _ in range(particles_nucleus):
            r_test = torch.rand(1, device=device).sqrt() * nucleus_radius
            theta_test = torch.rand(1, device=device) * 2 * torch.pi

            px = shape_center_x + r_test * torch.cos(theta_test)
            py = shape_center_y + r_test * torch.sin(theta_test)
            shape_particles.append([px.item(), py.item()])
            shape_materials.append(0)  # Material 0 for nucleus

        # Generate membrane particles (material 1 - jelly)
        for _ in range(particles_membrane):
            # Sample in annular region between nucleus_radius and outer_radius
            r_min_sq = nucleus_radius ** 2
            r_max_sq = outer_radius ** 2
            r_test = torch.sqrt(torch.rand(1, device=device) * (r_max_sq - r_min_sq) + r_min_sq)
            theta_test = torch.rand(1, device=device) * 2 * torch.pi

            px = shape_center_x + r_test * torch.cos(theta_test)
            py = shape_center_y + r_test * torch.sin(theta_test)
            shape_particles.append([px.item(), py.item()])
            shape_materials.append(1)  # Material 1 for membrane

        valid_particles.extend(shape_particles)
        particle_materials.extend(shape_materials)

    cell_positions = torch.tensor(valid_particles[:n_particles], device=device)
    x[:, 0] = cell_positions[:, 0]
    x[:, 1] = cell_positions[:, 1]

    # Set materials based on nucleus/membrane assignment
    T = torch.tensor(particle_materials[:n_particles], device=device).unsqueeze(1).int()

    # Calculate mass based on material type
    # Material 0: liquid (nucleus)
    # Material 1: jelly (membrane)
    material_densities = torch.tensor(rho_list, device=device)
    particle_densities = material_densities[T.squeeze()]
    M = torch.full((n_particles, 1), p_vol, dtype=torch.float32, device=device) * particle_densities.unsqueeze(1)

    # Random velocity per shape
    shape_velocities = (torch.rand(n_shapes, 2, device=device) - 0.5) * 4.0
    v = shape_velocities[group_indices]

    # Object ID for each particle
    ID = group_indices.unsqueeze(1).int()
    id_permutation = torch.randperm(n_shapes, device=device)
    ID = id_permutation[ID.squeeze()].unsqueeze(1)

    return N, x, v, C, F, T, Jp, M, S, ID


def get_index(n_particles, n_particle_types):
    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange((n_particles // n_particle_types) * n, (n_particles // n_particle_types) * (n + 1)))
    return index_particles


