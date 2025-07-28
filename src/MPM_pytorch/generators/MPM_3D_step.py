import torch
from MPM_pytorch.utils import *
import torch_geometric.data as data


def MPM_3D_step(
        model_MPM,
        X,
        V,
        C,
        F,
        T,
        Jp,
        M,
        n_particles,
        n_grid,
        dt,
        dx,
        inv_dx,
        mu_0,
        lambda_0,
        p_vol,
        offsets,
        particle_offsets,
        expansion_factor,
        gravity,
        friction,
        frame,
        device,
        verbose=False
):
    """
    Fast vectorized 3D MPM substep implementation
    """

    # Material masks
    liquid_mask = (T.squeeze() == 0)
    jelly_mask = (T.squeeze() == 1)
    snow_mask = (T.squeeze() == 2)

    # Create 3D identity matrices for all particles
    identity = torch.eye(3, device=device).unsqueeze(0).expand(n_particles, -1, -1)

    # Calculate F ############################################################################################

    # Update deformation gradient: F = (I + dt * C) * F_old
    F = (identity + dt * C) @ F

    # Hardening coefficient
    h = torch.exp(10 * (1.0 - Jp.squeeze()))
    h = torch.where(jelly_mask, torch.tensor(1.0, device=device), h)

    # Lamé parameters
    mu = mu_0 * h
    la = lambda_0 * h
    mu = torch.where(liquid_mask, torch.tensor(0.0, device=device), mu)

    # SVD decomposition for 3x3 matrices
    U, sig, Vh = torch.linalg.svd(F, driver='gesvdj')

    # SVD sign correction
    det_U = torch.det(U)
    det_Vh = torch.det(Vh)
    neg_det_U = det_U < 0
    if neg_det_U.any():
        U[neg_det_U, :, -1] *= -1
        sig[neg_det_U, -1] *= -1
    neg_det_Vh = det_Vh < 0
    if neg_det_Vh.any():
        Vh[neg_det_Vh, -1, :] *= -1
        sig[neg_det_Vh, -1] *= -1

    # Clamp singular values
    sig = torch.clamp(sig, min=1e-6)
    original_sig = sig.clone()

    # Apply plasticity constraints for snow
    new_sig = torch.where(snow_mask.unsqueeze(1), torch.clamp(sig, min=1 - 2.5e-2, max=1 + 4.5e-3), sig)

    # Update plastic deformation
    plastic_ratio = torch.prod(original_sig / new_sig, dim=1, keepdim=True)
    Jp = Jp * plastic_ratio
    sig = new_sig
    J = torch.prod(sig, dim=1)

    if frame > 1000:
        expansion_factor = 1.0

    J = J / expansion_factor  # Adjust J for expansion factor
    # Reconstruct deformation gradient
    sig_diag = torch.diag_embed(sig) / expansion_factor

    # For liquid: F = sqrt(J) * I
    F_liquid = identity * torch.sqrt(J).unsqueeze(-1).unsqueeze(-1)
    # For solid materials: F = U @ sig @ Vh
    F_solid = U @ sig_diag @ Vh

    # Apply reconstruction based on material type
    F = torch.where(liquid_mask.unsqueeze(-1).unsqueeze(-1), F_liquid, F)
    F = torch.where((jelly_mask | snow_mask).unsqueeze(-1).unsqueeze(-1), F_solid, F)

    # Calculate stress ############################################################################################
    R = U @ Vh
    F_minus_R = F - R
    stress = (2 * mu.unsqueeze(-1).unsqueeze(-1) * F_minus_R @ F.transpose(-2, -1) +
              identity * (la * J * (J - 1)).unsqueeze(-1).unsqueeze(-1))
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
    p_mass = M.squeeze(-1)
    affine = stress + p_mass.unsqueeze(-1).unsqueeze(-1) * C

    # P2G loop ###################################################################################################

    # Clear 3D grid
    grid_v = torch.zeros((n_grid, n_grid, n_grid, 3), device=device, dtype=torch.float32)
    grid_m = torch.zeros((n_grid, n_grid, n_grid), device=device, dtype=torch.float32)

    # Calculate base grid positions and fractional offsets
    base = (X * inv_dx - 0.5).int()  # [n_particles, 3]
    fx = X * inv_dx - base.float()  # [n_particles, 3]

    # Quadratic B-spline kernel weights for 3D
    w_0 = 0.5 * (1.5 - fx) ** 2
    w_1 = 0.75 - (fx - 1) ** 2
    w_2 = 0.5 * (fx - 0.5) ** 2
    # Stack weights [n_particles, 3, 3]
    w = torch.stack([w_0, w_1, w_2], dim=1)

    # P2G transfer (using pre-computed 3D offsets)
    # Expand for all particles: [n_particles, 27, 3]
    particle_base = base.unsqueeze(1).expand(-1, 27, -1)  # [n_particles, 27, 3]
    particle_fx = fx.unsqueeze(1).expand(-1, 27, -1)  # [n_particles, 27, 3]

    # Calculate grid positions for all particle-offset combinations
    grid_positions = particle_base + offsets.long()  # [n_particles, 27, 3]

    # Calculate weights for all 27 combinations
    # offsets is [27, 3] with entries like [0,0,0], [0,0,1], [0,0,2], [0,1,0], etc.
    weights = (w[:, offsets[:, 0].long(), 0] *
               w[:, offsets[:, 1].long(), 1] *
               w[:, offsets[:, 2].long(), 2])  # [n_particles, 27]

    # Calculate dpos for all combinations
    dpos = (particle_offsets - particle_fx) * dx  # [n_particles, 27, 3]

    # Bounds checking for 3D
    valid_mask = ((grid_positions[:, :, 0] >= 0) & (grid_positions[:, :, 0] < n_grid) &
                  (grid_positions[:, :, 1] >= 0) & (grid_positions[:, :, 1] < n_grid) &
                  (grid_positions[:, :, 2] >= 0) & (grid_positions[:, :, 2] < n_grid))

    # Flatten everything for scatter operations
    valid_indices = torch.where(valid_mask)
    particle_idx = valid_indices[0]  # Which particle
    offset_idx = valid_indices[1]  # Which offset (0-26)

    # Get valid data
    valid_grid_pos = grid_positions[valid_indices]  # [num_valid, 3]
    valid_weights = weights[valid_indices]  # [num_valid]
    valid_dpos = dpos[valid_indices]  # [num_valid, 3]

    # Calculate contributions
    affine_contrib = torch.bmm(affine[particle_idx],
                               valid_dpos.unsqueeze(-1)).squeeze(-1)  # [num_valid, 3]
    momentum_contrib = valid_weights.unsqueeze(-1) * (
            p_mass[particle_idx].unsqueeze(-1) * V[particle_idx] + affine_contrib)
    mass_contrib = valid_weights * p_mass[particle_idx]

    # Convert 3D grid positions to 1D indices for scatter
    grid_1d_idx = (valid_grid_pos[:, 0] * n_grid * n_grid +
                   valid_grid_pos[:, 1] * n_grid +
                   valid_grid_pos[:, 2])

    # Scatter add to flattened grid
    grid_v_flat = grid_v.view(-1, 3)
    grid_m_flat = grid_m.view(-1)
    grid_v_flat.scatter_add_(0, grid_1d_idx.unsqueeze(-1).expand(-1, 3), momentum_contrib)
    grid_m_flat.scatter_add_(0, grid_1d_idx, mass_contrib)

    # VECTORIZED: Convert momentum to velocity and apply boundary conditions ################################################

    # Create mask for valid grid points (non-zero mass)
    valid_mass_mask = grid_m > 0

    # Convert momentum to velocity (vectorized)
    grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                         grid_v / grid_m.unsqueeze(-1),
                         grid_v)

    # Apply gravity (vectorized) - in Y direction
    gravity_force = torch.tensor([0.0, dt * gravity, 0.0], device=device)
    grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                         grid_v + gravity_force,
                         grid_v)

    # VECTORIZED Boundary conditions for 3D (6 faces)
    # Create coordinate grids for boundary checking
    i_coords = torch.arange(n_grid, device=device).view(n_grid, 1, 1).expand(n_grid, n_grid, n_grid)
    j_coords = torch.arange(n_grid, device=device).view(1, n_grid, 1).expand(n_grid, n_grid, n_grid)
    k_coords = torch.arange(n_grid, device=device).view(1, 1, n_grid).expand(n_grid, n_grid, n_grid)

    # Apply friction to all boundary faces
    boundary_thickness = 3

    # X boundaries
    left_boundary_mask = (i_coords < boundary_thickness) & (grid_v[:, :, :, 0] < 0) & valid_mass_mask
    right_boundary_mask = (i_coords > n_grid - boundary_thickness) & (grid_v[:, :, :, 0] > 0) & valid_mass_mask
    grid_v[:, :, :, 0] = torch.where(left_boundary_mask, grid_v[:, :, :, 0] * friction, grid_v[:, :, :, 0])
    grid_v[:, :, :, 0] = torch.where(right_boundary_mask, grid_v[:, :, :, 0] * friction, grid_v[:, :, :, 0])

    # Y boundaries
    bottom_boundary_mask = (j_coords < boundary_thickness) & (grid_v[:, :, :, 1] < 0) & valid_mass_mask
    top_boundary_mask = (j_coords > n_grid - boundary_thickness) & (grid_v[:, :, :, 1] > 0) & valid_mass_mask
    grid_v[:, :, :, 1] = torch.where(bottom_boundary_mask, grid_v[:, :, :, 1] * friction, grid_v[:, :, :, 1])
    grid_v[:, :, :, 1] = torch.where(top_boundary_mask, grid_v[:, :, :, 1] * friction, grid_v[:, :, :, 1])

    # Z boundaries
    front_boundary_mask = (k_coords < boundary_thickness) & (grid_v[:, :, :, 2] < 0) & valid_mass_mask
    back_boundary_mask = (k_coords > n_grid - boundary_thickness) & (grid_v[:, :, :, 2] > 0) & valid_mass_mask
    grid_v[:, :, :, 2] = torch.where(front_boundary_mask, grid_v[:, :, :, 2] * friction, grid_v[:, :, :, 2])
    grid_v[:, :, :, 2] = torch.where(back_boundary_mask, grid_v[:, :, :, 2] * friction, grid_v[:, :, :, 2])

    # G2P transfer - VECTORIZED VERSION
    new_V = torch.zeros_like(V)
    new_C = torch.zeros_like(C)

    # G2P loop ###################################################################################################
    # Process all 27 neighbors simultaneously (using pre-computed offsets)

    # Expand offset for all particles and compute dpos for all neighbors (using pre-computed fx)
    dpos_all = offsets.unsqueeze(0) - fx.unsqueeze(1)  # [n_particles, 27, 3]

    # Grid positions for all neighbors (using pre-computed base)
    grid_pos_all = base.unsqueeze(1) + offsets.long().unsqueeze(0)  # [n_particles, 27, 3]

    # Weights for all neighbors: w[:, i, 0] * w[:, j, 1] * w[:, k, 2] for all (i,j,k) combinations
    i_indices = offsets[:, 0].long()  # [27] - i values
    j_indices = offsets[:, 1].long()  # [27] - j values
    k_indices = offsets[:, 2].long()  # [27] - k values
    weights_all = (w[:, i_indices, 0] *
                   w[:, j_indices, 1] *
                   w[:, k_indices, 2])  # [n_particles, 27]

    # Bounds checking for all neighbors
    valid_mask_all = ((grid_pos_all[:, :, 0] >= 0) & (grid_pos_all[:, :, 0] < n_grid) &
                      (grid_pos_all[:, :, 1] >= 0) & (grid_pos_all[:, :, 1] < n_grid) &
                      (grid_pos_all[:, :, 2] >= 0) & (grid_pos_all[:, :, 2] < n_grid))  # [n_particles, 27]

    # Get grid velocities for all neighbors with bounds checking
    g_v_all = torch.zeros((n_particles, 27, 3), device=device)

    # Flatten for efficient indexing
    flat_valid = valid_mask_all.flatten()  # [n_particles * 27]
    flat_grid_pos = grid_pos_all.reshape(-1, 3)  # [n_particles * 27, 3]

    if flat_valid.any():
        valid_positions = flat_grid_pos[flat_valid]
        flat_g_v = torch.zeros((n_particles * 27, 3), device=device)
        flat_g_v[flat_valid] = grid_v[valid_positions[:, 0], valid_positions[:, 1], valid_positions[:, 2]]
        g_v_all = flat_g_v.reshape(n_particles, 27, 3)

    # Accumulate velocity contributions from all neighbors
    velocity_contribs = weights_all.unsqueeze(-1) * g_v_all  # [n_particles, 27, 3]
    new_V += velocity_contribs.sum(dim=1)  # Sum over the 27 neighbors

    # CORRECTED APIC update - vectorized outer product for all neighbors
    # Reshape for batch matrix multiplication: [n_particles * 27, 3, 1] x [n_particles * 27, 1, 3]
    g_v_flat = g_v_all.reshape(-1, 3, 1)  # [n_particles * 27, 3, 1]
    dpos_flat = dpos_all.reshape(-1, 1, 3)  # [n_particles * 27, 1, 3]
    outer_products = torch.bmm(g_v_flat, dpos_flat).reshape(n_particles, 27, 3, 3)  # [n_particles, 27, 3, 3]

    # Weight the outer products and sum over neighbors
    weighted_outer_products = weights_all.unsqueeze(-1).unsqueeze(-1) * outer_products  # [n_particles, 27, 3, 3]
    new_C += 4 * inv_dx * weighted_outer_products.sum(dim=1)  # Sum over the 27 neighbors

    # Update particle state
    V.copy_(new_V)
    C.copy_(new_C)

    # Particle advection
    X = X + dt * V

    # Boundary conditions for particles (keep them in [0,1] bounds)
    X = torch.clamp(X, 0.01, 0.99)

    # Update particle velocities where they hit boundaries
    boundary_hit = (X <= 0.01) | (X >= 0.99)
    V = torch.where(boundary_hit, V * friction, V)


    # Grid momentum norm before velocity conversion (GP)
    grid_momentum_norm = torch.norm(grid_v, dim=3)

    return X, V, C, F, T, Jp, M, stress, grid_m, grid_momentum_norm