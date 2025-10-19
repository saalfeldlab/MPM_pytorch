
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
from plyfile import PlyData, PlyElement


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

    if geometry == 'gummy_bear':

        valid_particles = []
        particles_per_shape = group_size
        positions = generate_gummy_bear_particles(particles_per_shape, torch.zeros(2, device=device), 0.15, device=device)
        c_positions = positions - positions.mean(dim=0, keepdim=True)
        for shape_idx in range(n_shapes):

            shape_center_x = center_x[shape_idx * group_size]
            shape_center_y = center_y[shape_idx * group_size]


            object_positions = c_positions.clone()
            angle = shape_rotations[shape_idx]
            cos_a, sin_a = torch.cos(angle), torch.sin(angle)
            object_positions[:,0] = c_positions[:,0] * cos_a - c_positions[:,1]  * sin_a
            object_positions[:,1] = c_positions[:,0] * sin_a + c_positions[:,1]  * cos_a


            shape_center_x = shape_center_x.unsqueeze(0)  # shape [1]
            shape_center_y = shape_center_y.unsqueeze(0)  # shape [1]
            object_positions[:, 0] += shape_center_x
            object_positions[:, 1] += shape_center_y

            valid_particles.append(object_positions)

        all_positions = torch.cat(valid_particles, dim=0)
        x[:, 0] = all_positions[:, 0]
        x[:, 1] = all_positions[:, 1]

        plt.figure(figsize=(5,5))
        plt.scatter(x[:,0].cpu(), x[:,1].cpu(), s=1)
        plt.xlim(0,1)
        plt.ylim(0,1)
        plt.savefig('gummy_bear_particles.png', dpi=300)
        plt.close()


    elif geometry == 'cubes':
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
    if geometry == 'gummy_bear':
        T = torch.ones((n_particles, 1), dtype=torch.int32, device=device)


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


def generate_gummy_bear_particles(n_particles, center, size, device='cpu'):
    """
    Generate particles approximating a 2D gummy bear shape with legs touching the body.
    """
    cx, cy = center
    positions = []

    while len(positions) < n_particles:
        px = (torch.rand(1, device=device) - 0.5) * size * 2
        py = (torch.rand(1, device=device) - 0.5) * size * 2
        x_rel, y_rel = px.item(), py.item()

        # Head
        head_radius = 0.2 * size
        head_center = (0, 0.35 * size)
        head_dist = (x_rel - head_center[0]) ** 2 + (y_rel - head_center[1]) ** 2

        # Body
        body_width = 0.4 * size
        body_height = 0.5 * size
        body_xmin, body_xmax = -body_width / 2, body_width / 2
        body_ymin, body_ymax = -0.15 * size, 0.25 * size

        # Arms
        arm_radius = 0.1 * size
        left_arm_center = (-0.25 * size, 0.1 * size)
        right_arm_center = (0.25 * size, 0.1 * size)

        # Legs – moved up so they touch the body
        leg_radius = 0.12 * size
        leg_offset = body_ymin + leg_radius  # top of leg touches bottom of body
        left_leg_center = (-0.15 * size, leg_offset - leg_radius)  # adjust so circle top touches body
        right_leg_center = (0.15 * size, leg_offset - leg_radius)

        # Check if point is inside any part
        inside_head = head_dist <= head_radius ** 2
        inside_body = (body_xmin <= x_rel <= body_xmax) and (body_ymin <= y_rel <= body_ymax)
        inside_left_arm = (x_rel - left_arm_center[0]) ** 2 + (y_rel - left_arm_center[1]) ** 2 <= arm_radius ** 2
        inside_right_arm = (x_rel - right_arm_center[0]) ** 2 + (y_rel - right_arm_center[1]) ** 2 <= arm_radius ** 2
        inside_left_leg = (x_rel - left_leg_center[0]) ** 2 + (y_rel - left_leg_center[1]) ** 2 <= leg_radius ** 2
        inside_right_leg = (x_rel - right_leg_center[0]) ** 2 + (y_rel - right_leg_center[1]) ** 2 <= leg_radius ** 2

        if inside_head or inside_body or inside_left_arm or inside_right_arm or inside_left_leg or inside_right_leg:
            positions.append([cx + x_rel, cy + y_rel])

    return torch.tensor(positions, dtype=torch.float32, device=device)


def random_rotation_matrix(device='cpu'):
    # Random Euler angles
    roll = torch.rand(1, device=device) * 2 * torch.pi
    pitch = torch.rand(1, device=device) * 2 * torch.pi
    yaw = torch.rand(1, device=device) * 2 * torch.pi

    cos_r, sin_r = torch.cos(roll), torch.sin(roll)
    cos_p, sin_p = torch.cos(pitch), torch.sin(pitch)
    cos_y, sin_y = torch.cos(yaw), torch.sin(yaw)

    # Rotation matrices around each axis
    R_x = torch.tensor([
        [1, 0, 0],
        [0, cos_r, -sin_r],
        [0, sin_r, cos_r]
    ], device=device).squeeze()

    R_y = torch.tensor([
        [cos_p, 0, sin_p],
        [0, 1, 0],
        [-sin_p, 0, cos_p]
    ], device=device).squeeze()

    R_z = torch.tensor([
        [cos_y, -sin_y, 0],
        [sin_y, cos_y, 0],
        [0, 0, 1]
    ], device=device).squeeze()

    # Combined rotation matrix: R = R_z * R_y * R_x
    R = R_z @ R_y @ R_x
    return R


def stratified_sphere_points(n_points, radius=1.0, device='cpu'):
    # Estimate number of shells (radial layers)
    n_shells = int(torch.ceil(torch.tensor(n_points ** (1/3))).item())
    points = []

    total_points = 0
    for i in range(n_shells):
        r_lower = i / n_shells
        r_upper = (i + 1) / n_shells
        r_mean = (r_lower + r_upper) / 2

        # Fraction of points proportional to shell volume
        shell_volume = r_upper**3 - r_lower**3
        n_shell_points = int(shell_volume * n_points)

        if n_shell_points == 0:
            continue

        # Stratified indices within shell
        indices = torch.arange(n_shell_points, dtype=torch.float32, device=device) + 0.5

        # Spherical coordinates for points uniformly distributed on shell surface
        phi = torch.acos(1 - 2 * indices / n_shell_points)  # polar angle [0, pi]
        theta = 2 * torch.pi * indices * ((1 + 5 ** 0.5) / 2)  # golden angle for good azimuthal spacing

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        shell_points = torch.stack([x, y, z], dim=1) * (r_mean * radius)
        points.append(shell_points)

        total_points += n_shell_points

    # If not enough points generated due to rounding, fill with random points inside the sphere
    if total_points < n_points:
        remaining = n_points - total_points

        u = torch.rand(remaining, device=device)
        r = radius * u.pow(1/3)  # Correct radius distribution for uniform volume density

        phi = torch.acos(1 - 2 * torch.rand(remaining, device=device))
        theta = 2 * torch.pi * torch.rand(remaining, device=device)

        x = torch.sin(phi) * torch.cos(theta)
        y = torch.sin(phi) * torch.sin(theta)
        z = torch.cos(phi)

        random_points = torch.stack([x, y, z], dim=1) * r.unsqueeze(1)
        points.append(random_points)

    all_points = torch.cat(points, dim=0)
    return all_points[:n_points]


def get_equidistant_3D_points(n_points=1024):
    """
    Generate equidistant points within a unit sphere using improved 3D distribution.

    Args:
        n_points: Number of points to generate

    Returns:
        x, y, z: Arrays of coordinates for points within unit sphere
    """
    indices = np.arange(0, n_points, dtype=float) + 0.5

    # Radial distribution for uniform density in sphere volume
    # Use cube root for 3D volume distribution
    r = np.cbrt(indices / n_points)

    # Use Fibonacci spiral for uniform surface distribution
    # Golden angle in radians
    golden_angle = np.pi * (3.0 - np.sqrt(5.0))
    theta = golden_angle * indices

    # For uniform distribution on sphere surface (not clustered at poles)
    # y should be uniform in [-1, 1], not cos(phi)
    y = 1 - 2 * indices / n_points

    # Calculate radius in xy-plane
    radius_xy = np.sqrt(1 - y * y)

    # Convert to Cartesian coordinates
    x = radius_xy * np.cos(theta) * r
    y = y * r
    z = radius_xy * np.sin(theta) * r

    return x, y, z


def init_MPM_3D_shapes(
        geometry='cubes',  # 'cubes', 'spheres', or 'stars'
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
    if n_shapes == 8:
        # 2x2x2 cube
        shape_depth = group_indices // 4
        temp = group_indices % 4
        shape_row = temp // 2
        shape_col = temp % 2
        size = 0.15
        spacing_x = spacing_y = spacing_z = 0.4
        start_x = start_y = start_z = 0.2
    elif n_shapes == 27:
        # 3x3x3 cube
        shape_depth = group_indices // 9
        temp = group_indices % 9
        shape_row = temp // 3
        shape_col = temp % 3
        size = 0.08
        spacing_x = spacing_y = spacing_z = 0.25
        start_x = start_y = start_z = 0.15
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


    if geometry == 'gummy_bear':

        thickness = size * 0.2 * 2.5 # Small Z thickness
        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = min(start_idx + group_size, n_particles)
            actual_particles = end_idx - start_idx
            positions_2d = generate_gummy_bear_particles(
                actual_particles,
                center=(0.0, 0.0),
                size=size*2.5,
                device=device
            )
            z_offsets = (torch.rand(actual_particles, 1, device=device) - 0.5) * thickness
            positions_3d = torch.cat([positions_2d[:, :2], z_offsets], dim=1)

            rotated_positions = torch.matmul(positions_3d, rotation_matrices[shape_idx].T)

            x[start_idx:end_idx, 0] = center_x[start_idx] + rotated_positions[:, 0]
            x[start_idx:end_idx, 1] = center_y[start_idx] + rotated_positions[:, 1]
            x[start_idx:end_idx, 2] = center_z[start_idx] + rotated_positions[:, 2]

    elif geometry == 'cubes':
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
        # Generate particles in spherical volumes using equidistant distribution
        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = min(start_idx + group_size, n_particles)
            actual_particles = end_idx - start_idx

            # Get equidistant points in unit sphere
            sphere_x, sphere_y, sphere_z = get_equidistant_3D_points(actual_particles)

            # Convert to torch tensors and move to device
            sphere_points = torch.stack([
                torch.from_numpy(sphere_x).float(),
                torch.from_numpy(sphere_y).float(),
                torch.from_numpy(sphere_z).float()
            ], dim=1).to(device)

            # Apply 3D rotation
            rotated_positions = torch.matmul(sphere_points, rotation_matrices[shape_idx].T)

            # Scale by size and translate to shape center
            shape_center = torch.tensor([center_x[start_idx], center_y[start_idx], center_z[start_idx]], device=device)
            x[start_idx:end_idx] = shape_center + rotated_positions * size * 0.75

    elif geometry == 'stars':
        # Generate 3D stars
        outer_radius = size
        inner_radius = outer_radius * 0.4
        n_points = 5  # 5-pointed stars

        for shape_idx in range(n_shapes):
            start_idx = shape_idx * group_size
            end_idx = min(start_idx + group_size, n_particles)
            actual_particles = end_idx - start_idx

            # Adaptive layers based on particle count for this shape
            n_layers = min(10, max(3, actual_particles // 50))

            shape_center = torch.tensor([center_x[start_idx], center_y[start_idx], center_z[start_idx]], device=device)

            # Ensure we have enough particles per layer
            particles_per_layer = max(1, actual_particles // n_layers)
            star_particles = []

            for layer_idx in range(n_layers):
                # Z position for this layer - make stars flatter by reducing z-extent
                z_progress = layer_idx / max(n_layers - 1, 1)  # 0 to 1
                local_z = (z_progress - 0.5) * size * 0.3  # Reduced from size to size * 0.3 for flatter stars

                # Vary radius to create 3D star shape (double cone/spindle)
                # Maximum radius at center, tapering to points at ends
                layer_radius_scale = 1.0 - 2 * abs(z_progress - 0.5)  # Diamond profile
                layer_radius_scale = max(layer_radius_scale, 0.1)  # Minimum scale

                layer_outer_radius = outer_radius * layer_radius_scale
                layer_inner_radius = inner_radius * layer_radius_scale

                # Create 5-pointed star vertices for this layer
                star_angles = torch.linspace(0, 2 * torch.pi, n_points * 2 + 1, device=device)[:-1]
                star_radii = torch.zeros_like(star_angles)
                star_radii[::2] = layer_outer_radius  # Outer points
                star_radii[1::2] = layer_inner_radius  # Inner points

                # Create star vertices for this layer
                star_x = star_radii * torch.cos(star_angles)
                star_y = star_radii * torch.sin(star_angles)

                # Calculate particles for this layer
                if layer_idx == n_layers - 1:
                    # Last layer gets remaining particles
                    layer_particles = actual_particles - len(star_particles)
                else:
                    layer_particles = particles_per_layer

                # Ensure we have at least some particles per triangle
                particles_per_triangle = max(1,
                                             layer_particles // (n_points * 2))  # Split between two triangles per point

                particles_added_this_layer = 0

                for i in range(n_points):
                    if particles_added_this_layer >= layer_particles:
                        break

                    # Each star triangle: center -> outer point -> inner point -> next outer point
                    p1_x, p1_y, p1_z = 0.0, 0.0, local_z  # Center of this layer
                    p2_x, p2_y, p2_z = star_x[i * 2].item(), star_y[i * 2].item(), local_z  # Outer point
                    p3_x, p3_y, p3_z = star_x[(i * 2 + 1) % (n_points * 2)].item(), star_y[
                        (i * 2 + 1) % (n_points * 2)].item(), local_z  # Inner point
                    p4_x, p4_y, p4_z = star_x[(i * 2 + 2) % (n_points * 2)].item(), star_y[
                        (i * 2 + 2) % (n_points * 2)].item(), local_z  # Next outer point

                    # Fill triangle (center, outer, inner)
                    triangle_particles = min(particles_per_triangle, layer_particles - particles_added_this_layer)
                    for _ in range(triangle_particles):
                        r1, r2 = torch.rand(2, device=device)
                        if r1 + r2 > 1:
                            r1, r2 = 1 - r1, 1 - r2

                        px = p1_x + r1.item() * (p2_x - p1_x) + r2.item() * (p3_x - p1_x)
                        py = p1_y + r1.item() * (p2_y - p1_y) + r2.item() * (p3_y - p1_y)
                        pz = p1_z

                        star_particles.append([float(px), float(py), float(pz)])
                        particles_added_this_layer += 1

                        if particles_added_this_layer >= layer_particles:
                            break

                    if particles_added_this_layer >= layer_particles:
                        break

                    # Fill triangle (center, inner, next outer)
                    triangle_particles = min(particles_per_triangle, layer_particles - particles_added_this_layer)
                    for _ in range(triangle_particles):
                        r1, r2 = torch.rand(2, device=device)
                        if r1 + r2 > 1:
                            r1, r2 = 1 - r1, 1 - r2

                        px = p1_x + r1.item() * (p3_x - p1_x) + r2.item() * (p4_x - p1_x)
                        py = p1_y + r1.item() * (p3_y - p1_y) + r2.item() * (p4_y - p1_y)
                        pz = p1_z

                        star_particles.append([float(px), float(py), float(pz)])
                        particles_added_this_layer += 1

                        if particles_added_this_layer >= layer_particles:
                            break

            # Fill any remaining particles with random points in inner region
            while len(star_particles) < actual_particles:
                # Random layer
                layer_idx = torch.randint(0, n_layers, (1,)).item()
                z_progress = layer_idx / max(n_layers - 1, 1)
                local_z = (z_progress - 0.5) * size * 0.3  # Flatter profile
                layer_radius_scale = max(1.0 - 2 * abs(z_progress - 0.5), 0.1)

                r_fill = torch.rand(1, device=device).sqrt().item() * inner_radius * layer_radius_scale * 0.5
                theta_fill = torch.rand(1, device=device).item() * 2 * torch.pi
                px = r_fill * np.cos(theta_fill)
                py = r_fill * np.sin(theta_fill)
                pz = local_z

                star_particles.append([float(px), float(py), float(pz)])

            # Convert to tensor and apply 3D rotation
            if len(star_particles) > 0:
                star_positions = torch.tensor(star_particles[:actual_particles], device=device)
                rotated_positions = torch.matmul(star_positions, rotation_matrices[shape_idx].T)

                # Translate to shape center
                x[start_idx:end_idx] = shape_center + rotated_positions
            else:
                # Fallback: create a simple star shape
                print(f"Warning: Star generation failed for shape {shape_idx}, using fallback")
                # Create a simple cross pattern as fallback
                for i in range(actual_particles):
                    angle = (i / actual_particles) * 2 * torch.pi
                    radius = outer_radius * (0.5 + 0.5 * torch.cos(5 * angle))
                    px = radius * torch.cos(angle)
                    py = radius * torch.sin(angle)
                    pz = (torch.rand(1) - 0.5) * size * 0.3  # Flatter profile

                    rel_pos = torch.tensor([px, py, pz], device=device)
                    rotated_pos = torch.matmul(rel_pos, rotation_matrices[shape_idx].T)
                    x[start_idx + i] = shape_center + rotated_pos

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
    T = torch.zeros((n_particles, 1), dtype=torch.int32, device=device)
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

def generate_compressed_video_mp4(output_dir, run=0, framerate=10, output_name=".mp4", config_indices=None, crf=23):
    """
    Generate a compressed video using ffmpeg's libx264 codec in MP4 format.
    Automatically handles odd dimensions by scaling to even dimensions.
    Parameters:
        output_dir (str): Path to directory containing Fig/Fig_*.png.
        run (int): Run index to use in filename pattern.
        framerate (int): Desired video framerate.
        output_name (str): Name of output .mp4 file.
        config_indices: Configuration indices for filename.
        crf (int): Constant Rate Factor for quality (0-51, lower = better quality, 23 is default).
    """
    import os
    import subprocess
    fig_dir = os.path.join(output_dir, "Fig")
    input_pattern = os.path.join(fig_dir, f"Fig_{run}_%06d.png")
    output_path = os.path.join(output_dir, f"input_{config_indices}{output_name}")
    
    ffmpeg_cmd = [
        "ffmpeg",
        "-y",
        "-loglevel", "error",
        "-framerate", str(framerate),
        "-i", input_pattern,
        "-vf", "scale='trunc(iw/2)*2:trunc(ih/2)*2'",
        "-c:v", "libx264",
        "-crf", str(crf),
        "-preset", "medium",
        "-pix_fmt", "yuv420p",
        output_path,
    ]
    
    subprocess.run(ffmpeg_cmd, check=True)
    print(f"compressed video (libx264) saved to: {output_path}")



def save_gaussian_splat_ply(output_path, positions, colors, scales, rotations, opacities):
    """
    Save Gaussian Splatting data in PLY format.
    
    Parameters:
        output_path: Path to save PLY file
        positions: (N, 3) xyz coordinates
        colors: (N, 3) RGB values [0-1]
        scales: (N, 3) scale per axis
        rotations: (N, 4) quaternions (w, x, y, z)
        opacities: (N, 1) opacity values [0-1]
    """
    import numpy as np
    from plyfile import PlyData, PlyElement
    
    num_points = positions.shape[0]
    
    # Convert colors from [0-1] to [0-255]
    colors_255 = (np.clip(colors, 0, 1) * 255).astype(np.uint8)
    
    # Create structured array for PLY format
    # Standard 3D Gaussian Splatting format
    dtype = [
        ('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
        ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),  # Normals (unused, set to 0)
        ('f_dc_0', 'f4'), ('f_dc_1', 'f4'), ('f_dc_2', 'f4'),  # RGB colors
        ('opacity', 'f4'),
        ('scale_0', 'f4'), ('scale_1', 'f4'), ('scale_2', 'f4'),
        ('rot_0', 'f4'), ('rot_1', 'f4'), ('rot_2', 'f4'), ('rot_3', 'f4'),  # Quaternion
    ]
    
    elements = np.empty(num_points, dtype=dtype)
    
    # Positions
    elements['x'] = positions[:, 0]
    elements['y'] = positions[:, 1]
    elements['z'] = positions[:, 2]
    
    # Normals (not used in Gaussian Splatting, set to 0)
    elements['nx'] = 0
    elements['ny'] = 0
    elements['nz'] = 0
    
    # Colors (convert to spherical harmonics DC component)
    # For DC component: (color - 0.5) / 0.28209479177387814
    SH_C0 = 0.28209479177387814
    elements['f_dc_0'] = (colors[:, 0] - 0.5) / SH_C0
    elements['f_dc_1'] = (colors[:, 1] - 0.5) / SH_C0
    elements['f_dc_2'] = (colors[:, 2] - 0.5) / SH_C0
    
    # Opacity (convert to logit space for better optimization)
    # logit(x) = log(x / (1 - x))
    opacities_clamped = np.clip(opacities.flatten(), 0.001, 0.999)
    elements['opacity'] = np.log(opacities_clamped / (1 - opacities_clamped))
    
    # Scales (convert to log space)
    scales_clamped = np.clip(scales, 1e-6, None)
    elements['scale_0'] = np.log(scales_clamped[:, 0])
    elements['scale_1'] = np.log(scales_clamped[:, 1])
    elements['scale_2'] = np.log(scales_clamped[:, 2])
    
    # Rotations (quaternion: w, x, y, z)
    elements['rot_0'] = rotations[:, 0]  # w
    elements['rot_1'] = rotations[:, 1]  # x
    elements['rot_2'] = rotations[:, 2]  # y
    elements['rot_3'] = rotations[:, 3]  # z
    
    # Create PLY element and save
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(output_path)





def export_for_gaussian_splatting(pos, particle_colors, frame_idx, output_dir, dataset_name, particle_volumes=None):
    """
    Export MPM particle data in Gaussian Splatting PLY format.
    
    Parameters:
        pos: particle positions (N, 3) - numpy array
        particle_colors: particle colors (N, 3) RGB values [0-1] - numpy array
        frame_idx: current frame number
        output_dir: base directory to save the splat files
        dataset_name: name of the dataset for folder organization
        particle_volumes: optional (N,) array of particle volumes for adaptive scaling
    """
    import numpy as np
    import os
    
    num_particles = pos.shape[0]
    
    # Create dataset-specific directory for Gaussian Splats
    splat_dir = os.path.join(output_dir, dataset_name, "GaussianSplats")
    os.makedirs(splat_dir, exist_ok=True)
    
    # Scale: use particle volumes if provided, otherwise uniform
    if particle_volumes is not None:
        # Scale based on volume: scale ~ volume^(1/3)
        base_scale = np.cbrt(particle_volumes)
        scales = np.stack([base_scale, base_scale, base_scale], axis=1) * 0.5
    else:
        # Default uniform scale
        scales = np.ones((num_particles, 3)) * 0.01
    
    # Rotations: identity quaternion (w=1, x=0, y=0, z=0)
    rotations = np.zeros((num_particles, 4))
    rotations[:, 0] = 1.0  # w component
    
    # Opacities: semi-transparent
    opacities = np.ones((num_particles, 1)) * 0.8
    
    # Save as PLY file
    output_path = os.path.join(splat_dir, f"splat_{frame_idx:06d}.ply")
    save_gaussian_splat_ply(output_path, pos, particle_colors, scales, rotations, opacities)
    
    return output_path