def MPM_step_no_GNN(model_MPM, X, V, C, F, T, Jp, M, n_particles, n_grid, dt, dx, inv_dx, mu_0, lambda_0, p_vol, offsets, particle_offsets, grid_coords, device,
                verbose=False):
    """
    MPM substep implementation
    """

    # Material masks
    liquid_mask = (T.squeeze() == 0)
    jelly_mask = (T.squeeze() == 1)
    snow_mask = (T.squeeze() == 2)
    # Create identity matrices for all particles
    identity = torch.eye(2, device=device).unsqueeze(0).expand(n_particles, -1, -1)

    # Calculate F ############################################################################################

    # Update deformation gradient: F = (I + dt * C) * F_old
    F = (identity + dt * C) @ F
    # Hardening coefficient
    h = torch.exp(10 * (1.0 - Jp.squeeze()))
    h = torch.where(jelly_mask, torch.tensor(0.3, device=device), h)
    # Lamé parameters
    mu = mu_0 * h
    la = lambda_0 * h
    mu = torch.where(liquid_mask, torch.tensor(0.0, device=device), mu)
    # SVD decomposition
    U, sig, Vh = torch.linalg.svd(F, driver='gesvdj')
    # SVD sign correction without in-place ops
    det_U = torch.det(U)
    det_Vh = torch.det(Vh)
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
    original_sig = sig.clone()
    # Apply plasticity constraints for snow
    new_sig = torch.where(snow_mask.unsqueeze(1),
                          torch.clamp(sig, min=1 - 2.5e-2, max=1 + 4.5e-3),
                          sig)
    # Update plastic deformation
    plastic_ratio = torch.prod(original_sig / new_sig, dim=1, keepdim=True)
    Jp = Jp * plastic_ratio
    sig = new_sig
    J = torch.prod(sig, dim=1)
    sig_diag = torch.diag_embed(sig)
    # For liquid: F = sqrt(J) * I
    F_liquid = identity * torch.sqrt(J).unsqueeze(-1).unsqueeze(-1)
    # For solid materials: F = U @ sig_diag @ Vh
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

    # Calculate distances between grid points and particles
    base = (X * inv_dx - 0.5).int()
    fx = X * inv_dx - base.float()
    w_0 = 0.5 * (1.5 - fx) ** 2
    w_1 = 0.75 - (fx - 1) ** 2
    w_2 = 0.5 * (fx - 0.5) ** 2
    w = torch.stack([w_0, w_1, w_2], dim=1)
    grid_positions = base.unsqueeze(1) + offsets.long()  # [n_particles, 9, 2]
    particle_indices = torch.arange(n_particles, device=device).unsqueeze(1).expand(-1, 9).flatten()
    grid_indices = grid_positions.flatten().reshape(-1, 2)  # Flatten to [n_particles*9, 2]
    grid_indices_1d = grid_indices[:, 0] * n_grid + grid_indices[:, 1]
    edge_index = torch.stack([particle_indices, grid_indices_1d], dim=0)
    edge_index [0,:] += n_grid**2  # offset particle indices
    x_ = torch.cat((torch.zeros((n_grid,1),dtype=torch.float32,device=device),p_mass[:,None]))
    dataset = data.Data(x=x_, edge_index=edge_index, w=w)
    grid_m = model_MPM(dataset)

    # Clear grid
    grid_v = torch.zeros((n_grid, n_grid, 2), device=device, dtype=torch.float32)
    grid_m = torch.zeros((n_grid, n_grid), device=device, dtype=torch.float32)
    # Calculate base grid positions and fractional offsets
    base = (X * inv_dx - 0.5).int()
    fx = X * inv_dx - base.float()

    # Quadratic B-spline kernel weights
    w_0 = 0.5 * (1.5 - fx) ** 2
    w_1 = 0.75 - (fx - 1) ** 2
    w_2 = 0.5 * (fx - 0.5) ** 2
    # Stack weights [n_particles, 3, 2]
    w = torch.stack([w_0, w_1, w_2], dim=1)

    # P2G transfer (using pre-computed offsets)
    # Expand for all particles: [n_particles, 9, 2]
    particle_base = base.unsqueeze(1).expand(-1, 9, -1)  # [n_particles, 9, 2]
    particle_fx = fx.unsqueeze(1).expand(-1, 9, -1)  # [n_particles, 9, 2]
    # Calculate grid positions for all particle-offset combinations
    grid_positions = particle_base + offsets.long()  # [n_particles, 9, 2]
    # Calculate weights for all combinations
    weights = w[:, offsets[:, 0].long(), 0] * w[:, offsets[:, 1].long(), 1]  # [n_particles, 9]
    # Calculate dpos for all combinations
    dpos = (particle_offsets - particle_fx) * dx  # [n_particles, 9, 2]
    # Bounds checking
    valid_mask = ((grid_positions[:, :, 0] >= 0) & (grid_positions[:, :, 0] < n_grid) &
                  (grid_positions[:, :, 1] >= 0) & (grid_positions[:, :, 1] < n_grid))
    # Flatten everything for scatter operations
    valid_indices = torch.where(valid_mask)
    particle_idx = valid_indices[0]  # Which particle
    offset_idx = valid_indices[1]  # Which offset (0-8)


    # Get valid data
    valid_grid_pos = grid_positions[valid_indices]  # [num_valid, 2]
    valid_weights = weights[valid_indices]  # [num_valid]
    valid_dpos = dpos[valid_indices]  # [num_valid, 2]
    # Calculate contributions
    affine_contrib = torch.bmm(affine[particle_idx],
                               valid_dpos.unsqueeze(-1)).squeeze(-1)  # [num_valid, 2]
    momentum_contrib = valid_weights.unsqueeze(-1) * (
            p_mass[particle_idx].unsqueeze(-1) * V[particle_idx] + affine_contrib)
    mass_contrib = valid_weights * p_mass[particle_idx]
    # Convert 2D grid positions to 1D indices for scatter
    grid_1d_idx = valid_grid_pos[:, 0] * n_grid + valid_grid_pos[:, 1]
    # Scatter add to flattened grid
    grid_v_flat = grid_v.view(-1, 2)
    grid_m_flat = grid_m.view(-1)
    grid_v_flat.scatter_add_(0, grid_1d_idx.unsqueeze(-1).expand(-1, 2), momentum_contrib)
    grid_m_flat.scatter_add_(0, grid_1d_idx, mass_contrib)

    # VECTORIZED: Convert momentum to velocity and apply boundary conditions ################################################

    # Create mask for valid grid points (non-zero mass)
    valid_mass_mask = grid_m > 0

    # Convert momentum to velocity (vectorized)
    grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                         grid_v / grid_m.unsqueeze(-1),
                         grid_v)

    # Apply gravity (vectorized)
    gravity_force = torch.tensor([0.0, dt * (-50)], device=device)
    grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                         grid_v + gravity_force,
                         grid_v)

    # VECTORIZED Boundary conditions
    # Create coordinate grids for boundary checking
    i_coords = torch.arange(n_grid, device=device).unsqueeze(1).expand(n_grid, n_grid)  # [n_grid, n_grid]
    j_coords = torch.arange(n_grid, device=device).unsqueeze(0).expand(n_grid, n_grid)  # [n_grid, n_grid]

    # Left boundary: i < 3 and v_x < 0 → set v_x = 0
    left_boundary_mask = (i_coords < 3) & (grid_v[:, :, 0] < 0) & valid_mass_mask
    grid_v[:, :, 0] = torch.where(left_boundary_mask, 0.0, grid_v[:, :, 0])

    # Right boundary: i > n_grid - 3 and v_x > 0 → set v_x = 0
    right_boundary_mask = (i_coords > n_grid - 3) & (grid_v[:, :, 0] > 0) & valid_mass_mask
    grid_v[:, :, 0] = torch.where(right_boundary_mask, 0.0, grid_v[:, :, 0])

    # Bottom boundary: j < 3 and v_y < 0 → set v_y = 0
    bottom_boundary_mask = (j_coords < 3) & (grid_v[:, :, 1] < 0) & valid_mass_mask
    grid_v[:, :, 1] = torch.where(bottom_boundary_mask, 0.0, grid_v[:, :, 1])

    # Top boundary: j > n_grid - 3 and v_y > 0 → set v_y = 0
    top_boundary_mask = (j_coords > n_grid - 3) & (grid_v[:, :, 1] > 0) & valid_mass_mask
    grid_v[:, :, 1] = torch.where(top_boundary_mask, 0.0, grid_v[:, :, 1])

    # G2P transfer - CORRECTED VERSION
    new_V = torch.zeros_like(V)
    new_C = torch.zeros_like(C)

    # G2P loop ###################################################################################################
    # Process all 9 neighbors simultaneously (using pre-computed offsets)

    # Expand offset for all particles and compute dpos for all neighbors (using pre-computed fx)
    dpos_all = offsets.unsqueeze(0) - fx.unsqueeze(1)  # [n_particles, 9, 2]

    # Grid positions for all neighbors (using pre-computed base)
    grid_pos_all = base.unsqueeze(1) + offsets.long().unsqueeze(0)  # [n_particles, 9, 2]

    # Weights for all neighbors: w[:, i, 0] * w[:, j, 1] for all (i,j) combinations (using pre-computed w)
    i_indices = offsets[:, 0].long()  # [9] - i values: [0,0,0,1,1,1,2,2,2]
    j_indices = offsets[:, 1].long()  # [9] - j values: [0,1,2,0,1,2,0,1,2]
    weights_all = w[:, i_indices, 0] * w[:, j_indices, 1]  # [n_particles, 9]

    # Bounds checking for all neighbors
    valid_mask_all = ((grid_pos_all[:, :, 0] >= 0) & (grid_pos_all[:, :, 0] < n_grid) &
                      (grid_pos_all[:, :, 1] >= 0) & (grid_pos_all[:, :, 1] < n_grid))  # [n_particles, 9]

    # Get grid velocities for all neighbors with bounds checking
    g_v_all = torch.zeros((n_particles, 9, 2), device=device)

    # Flatten for efficient indexing
    flat_valid = valid_mask_all.flatten()  # [n_particles * 9]
    flat_grid_pos = grid_pos_all.reshape(-1, 2)  # [n_particles * 9, 2]

    if flat_valid.any():
        valid_positions = flat_grid_pos[flat_valid]
        flat_g_v = torch.zeros((n_particles * 9, 2), device=device)
        flat_g_v[flat_valid] = grid_v[valid_positions[:, 0], valid_positions[:, 1]]
        g_v_all = flat_g_v.reshape(n_particles, 9, 2)

    # Accumulate velocity contributions from all neighbors
    velocity_contribs = weights_all.unsqueeze(-1) * g_v_all  # [n_particles, 9, 2]
    new_V += velocity_contribs.sum(dim=1)  # Sum over the 9 neighbors

    # CORRECTED APIC update - vectorized outer product for all neighbors
    # Reshape for batch matrix multiplication: [n_particles * 9, 2, 1] x [n_particles * 9, 1, 2]
    g_v_flat = g_v_all.reshape(-1, 2, 1)  # [n_particles * 9, 2, 1]
    dpos_flat = dpos_all.reshape(-1, 1, 2)  # [n_particles * 9, 1, 2]
    outer_products = torch.bmm(g_v_flat, dpos_flat).reshape(n_particles, 9, 2, 2)  # [n_particles, 9, 2, 2]

    # Weight the outer products and sum over neighbors
    weighted_outer_products = weights_all.unsqueeze(-1).unsqueeze(-1) * outer_products  # [n_particles, 9, 2, 2]
    new_C += 4 * inv_dx * weighted_outer_products.sum(dim=1)  # Sum over the 9 neighbors

    # Update particle state
    V.copy_(new_V)
    C.copy_(new_C)

    # Particle advection
    X = X + dt * V

    # Convert PyTorch tensors to numpy
    grid_m_np = grid_m.cpu().numpy()
    grid_v_np = grid_v.cpu().numpy()
    grid_v_norm = np.sqrt(grid_v_np[:, :, 0] ** 2 + grid_v_np[:, :, 1] ** 2)

    # Calculate stress for return (S)
    stress_norm_return = torch.norm(stress.view(n_particles, -1), dim=1, keepdim=True)
    # Grid momentum norm before velocity conversion (GP)
    grid_momentum_norm = torch.norm(grid_v, dim=2)

    return X, V, C, F, T, Jp, M, stress_norm_return, grid_m, grid_momentum_norm