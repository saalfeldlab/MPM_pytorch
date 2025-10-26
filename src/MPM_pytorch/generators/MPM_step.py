import torch
from MPM_pytorch.utils import *
import torch_geometric.data as data


def MPM_step(
        model_MPM,
        X,
        V,
        C,
        F,
        Jp,
        T,
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
        surface_tension_coeff,  # N/m (water at room temperature)
        tension_scaling,
        enable_surface_tension,
        debug_surface,
        device
):
    """
    MPM substep implementation
    """

    liquid_mask = (T.squeeze() == 0).detach()
    jelly_mask = (T.squeeze() == 1).detach()
    snow_mask = (T.squeeze() == 2).detach()

    # initialize mass
    p_mass = M.squeeze(-1)

    identity = torch.eye(2, device=device).unsqueeze(0)  # [1, 2, 2] - Identity matrix for all particles

    # update deformation gradient F ################################################################################
    F = (identity + dt * C) @ F  # F^{n+1} = (I + dt*C^n) @ F^n - Update F using velocity gradient C

    h = torch.exp(10 * (1.0 - Jp.squeeze()))  # Jp close to 1, Jp< 1 => h>1 (hardening), Jp>1 => h<1 (softening)
    h = torch.where(jelly_mask, torch.tensor(0.3, device=device), h)  # jelly uses constant h=0.3 (70% softer)

    # Lamé parameters
    mu = mu_0 * h  # shear modulus (shape resistance) 
    la = lambda_0 * h  # first Lamé parameter (volume resistance) 
    mu = torch.where(liquid_mask, torch.tensor(0.0, device=device), mu)  # liquids cannot resist shear deformation, no shape memory

    # SVD decomposition
    F_reg = F + 1e-6 * identity 

     # small regularization to avoid numerical issues

    U, sig, Vh = torch.linalg.svd(F_reg, driver='gesvdj')
    # U, sig, Vh = safe_svd(F_reg)  # F = U @ diag(sig) @ Vh - polar decomposition

    # U and Vh should be rotation matrices with det=+1, but numerical issues may lead to reflections (det=-1)
    det_U = torch.det(U)
    det_Vh = torch.det(Vh)
    neg_det_U = det_U < 0
    neg_det_Vh = det_Vh < 0

    # sign corrections - handle each case properly without double-modifying sig
    if neg_det_U.any():
        # flip last column of U and last singular value
        U_corrected = U.clone()
        U_corrected[neg_det_U, :, -1] *= -1
        U = U_corrected
        
        sig_corrected = sig.clone()
        sig_corrected[neg_det_U, -1] *= -1
        sig = sig_corrected

    if neg_det_Vh.any():
        # flip last row of Vh and last singular value
        Vh_corrected = Vh.clone()
        Vh_corrected[neg_det_Vh, -1, :] *= -1
        Vh = Vh_corrected
        
        sig_corrected = sig.clone()
        sig_corrected[neg_det_Vh, -1] *= -1
        sig = sig_corrected

    # soft clamp of sig to preserve gradient
    min_val = 1e-6
    sig = torch.where(sig < min_val, min_val + 0.01 * (sig - min_val), sig)
    original_sig = sig.clone()
    sig = torch.where(snow_mask.unsqueeze(1), torch.clamp(sig, min=1 - 2.5e-2, max=1 + 4.5e-3), sig)

    # update plastic deformation
    plastic_ratio = torch.prod(original_sig / sig, dim=1, keepdim=True)  # volume ratio of plastic deformation
    Jp = Jp * plastic_ratio  # accumulate plastic volume change

    J = torch.prod(sig, dim=1)  # product of singular values
    J = torch.clamp(J, min=1e-4)  # prevent volume collapsing to 0

    if frame > 1000:  # remove expansion after frame 1000
        expansion_factor = 1.0  
    J = J / expansion_factor  # scale down volume
    sig_diag = torch.diag_embed(sig) / expansion_factor  # scale down singular values proportionally

    F_liquid = identity * torch.sqrt(J).unsqueeze(-1).unsqueeze(-1)  # for liquid: F = sqrt(J) * I, isotropic deformation (no shear, only volume)
    F_solid = U @ sig_diag @ Vh  # for solid materials: F = U @ sig_diag @ Vh, after plasticity projection
    F = torch.where(liquid_mask.unsqueeze(-1).unsqueeze(-1), F_liquid, F)  
    F = torch.where((jelly_mask | snow_mask).unsqueeze(-1).unsqueeze(-1), F_solid, F)  





    # update stress ############################################################################################
    # first Piola-Kirchhoff stress: P = 2μ(F-R)F^T + λJ(J-1)I [Pa] (fixed corotated hyperelastic model)
    R = U @ Vh  # Rotation matrix from polar decomposition
    F_minus_R = F - R  # F minus rotation (captures deformation without rigid rotation)
    stress = (2 * mu.unsqueeze(-1).unsqueeze(-1) * F_minus_R @ F.transpose(-2, -1) +
            identity * (la * J * (J - 1)).unsqueeze(-1).unsqueeze(-1))  # Cauchy/PK1 stress [Pa]
    stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress  # Scale for MPM P2G transfer [force]

    # add surface tension stress for liquids (heuristic approach) ##############################################
    if enable_surface_tension and liquid_mask.any():
        debug_this_frame = debug_surface and (frame % 100 == 0 or frame == 0)
        
        if debug_this_frame:
            print(f"\n=== Frame {frame} - Surface Tension Debug ===")
            print(f"Applying surface tension with coeff={surface_tension_coeff}")
        
        surface_stress = compute_surface_tension_stress(X, T, surface_tension_coeff, dx, device, debug_this_frame)
        
        surface_stress_scaled = surface_stress * (-dt * p_vol * 4 * inv_dx * inv_dx) * tension_scaling

        liquid_stress_scaled = surface_stress_scaled[liquid_mask]

        stress_norms = torch.norm(liquid_stress_scaled.reshape(liquid_stress_scaled.shape[0], -1), dim=1)
        percentile = torch.quantile(stress_norms, 0.8)
        needs_clamping = stress_norms > percentile
        if needs_clamping.any():
            scale = (percentile / stress_norms[needs_clamping]).unsqueeze(-1).unsqueeze(-1)
            liquid_stress_scaled[needs_clamping] = liquid_stress_scaled[needs_clamping] * scale
                
        # only add surface stress to liquid particles
        stress = torch.where(liquid_mask.unsqueeze(-1).unsqueeze(-1),
                           stress +  surface_stress_scaled,
                           stress)
        
        if debug_this_frame:
            regular_stress_norm = torch.norm(stress[liquid_mask].reshape(-1, 4), dim=1).mean()
            surface_stress_norm = torch.norm(surface_stress_scaled[liquid_mask].reshape(-1, 4), dim=1).mean()
            print(f"regular stress magnitude: {regular_stress_norm:.6f}")
            print(f"surface stress magnitude: {surface_stress_norm:.6f}")
            print(f"ratio surface/regular: {surface_stress_norm/regular_stress_norm:.3f}")

    affine = stress + p_mass.unsqueeze(-1).unsqueeze(-1) * C
    # stress=force + momentum gradient=force






    # P2G loop ###################################################################################################

    base = (X * inv_dx - 0.5).int()

    fx = X * inv_dx - base.float()
    fx_per_edge = fx.unsqueeze(1).expand(-1, 9, -1).flatten(end_dim=1)  # [n_particles*9, 2]
    
    # compute weights for both P2G and G2P
    w_0 = 0.5 * (1.5 - fx) ** 2
    w_1 = 0.75 - (fx - 1) ** 2
    w_2 = 0.5 * (fx - 0.5) ** 2
    w = torch.stack([w_0, w_1, w_2], dim=1)  # [n_particles, 3, 2]

    # prepare weights per edge for GNN
    i_indices = offsets[:, 0].long().detach()  # [9]
    j_indices = offsets[:, 1].long().detach()  # [9]
    weights_all = w[:, i_indices, 0] * w[:, j_indices, 1]  # [n_particles, 9]
    weights_per_edge = weights_all.flatten()  # [n_particles*9]

    grid_positions = base.unsqueeze(1) + offsets.unsqueeze(0).detach()  # [n_particles, 9, 2]
    particle_indices = torch.arange(n_particles, device=device).unsqueeze(1).expand(-1, 9).flatten()
    grid_indices = grid_positions.flatten().reshape(-1, 2)  # Flatten to [n_particles*9, 2]
    grid_indices_1d = grid_indices[:, 0] * n_grid + grid_indices[:, 1]
    edge_index = torch.stack([particle_indices, grid_indices_1d], dim=0).long()
    edge_index[0, :] += n_grid ** 2  # offset particle indices


    # Compute dpos for each edge (needed for affine contribution)
    particle_fx_expanded = fx.unsqueeze(1).expand(-1, 9, -1)  # [n_particles, 9, 2]
    dpos = (particle_offsets - particle_fx_expanded) * dx  # [n_particles, 9, 2]
    dpos_per_edge = dpos.flatten(end_dim=1)  # [n_particles*9, 2]

    # Affine matrices per edge (replicate for each particle's 9 edges)
    affine_per_edge = affine.unsqueeze(1).expand(-1, 9, -1, -1).flatten(end_dim=1)  # [n_particles*9, 2, 2]

    # Extended node features: [mass, vel_x, vel_y] for particles, [0, 0, 0] for grid
    grid_features = torch.zeros((n_grid ** 2, 3), dtype=torch.float32, device=device)  # [mass, vel_x, vel_y]
    particle_features = torch.cat([p_mass[:, None], V], dim=1)  # [n_particles, 3]
    x_ = torch.cat([grid_features, particle_features], dim=0)  # [n_grid**2 + n_particles, 3]

    # GNN inference
    dataset = data.Data(x=x_, edge_index=edge_index, weights_per_edge=weights_per_edge,
                        affine_per_edge=affine_per_edge, dpos_per_edge=dpos_per_edge)
    
    grid_output = model_MPM(dataset)[0:n_grid ** 2]  # [n_grid**2, 3]
    grid_m = grid_output[:, 0].view(n_grid, n_grid)  # mass component
    grid_v = grid_output[:, 1:3].view(n_grid, n_grid, 2)  # velocity components # Reshape to [n_grid, n_grid]





    # Create mask for valid grid points (non-zero mass)
    valid_mass_mask = grid_m > 0
    # Convert momentum to velocity (vectorized)
    eps = 1e-10
    grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                        grid_v / (grid_m.unsqueeze(-1) + eps),
                        grid_v)

    # Apply gravity (vectorized)
    gravity_force = torch.tensor([0.0, dt * (gravity)], device=device)
    grid_v = torch.where(valid_mass_mask.unsqueeze(-1),
                         grid_v + gravity_force,
                         grid_v)

    # Create coordinate grids
    i_coords = torch.arange(n_grid, device=device).unsqueeze(1).expand(n_grid, n_grid)
    j_coords = torch.arange(n_grid, device=device).unsqueeze(0).expand(n_grid, n_grid)
    valid_mass_mask = torch.ones_like(grid_v[:, :, 0], dtype=torch.bool)

    # Boundary masks
    left_mask = (i_coords < 3) & valid_mass_mask
    right_mask = (i_coords > n_grid - 3) & valid_mass_mask
    bottom_mask = (j_coords < 3) & valid_mass_mask
    top_mask = (j_coords > n_grid - 3) & valid_mass_mask

    # Apply normal boundary conditions (prevent penetration)
    grid_v[:, :, 0] = torch.where(left_mask & (grid_v[:, :, 0] < 0), 0.0, grid_v[:, :, 0])
    grid_v[:, :, 0] = torch.where(right_mask & (grid_v[:, :, 0] > 0), 0.0, grid_v[:, :, 0])
    grid_v[:, :, 1] = torch.where(bottom_mask & (grid_v[:, :, 1] < 0), 0.0, grid_v[:, :, 1])
    grid_v[:, :, 1] = torch.where(top_mask & (grid_v[:, :, 1] > 0), 0.0, grid_v[:, :, 1])

    # Apply friction to tangential components
    friction_factor = 1.0 - friction

    # Horizontal boundaries affect v_y (tangential)
    horizontal_boundary_mask = left_mask | right_mask
    grid_v[:, :, 1] = torch.where(horizontal_boundary_mask,
                                  grid_v[:, :, 1] * friction_factor,
                                  grid_v[:, :, 1])

    # Vertical boundaries affect v_x (tangential)
    vertical_boundary_mask = bottom_mask | top_mask
    grid_v[:, :, 0] = torch.where(vertical_boundary_mask,
                                  grid_v[:, :, 0] * friction_factor,
                                  grid_v[:, :, 0])

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
    new_V = new_V + velocity_contribs.sum(dim=1)  # Sum over the 9 neighbors

    # CORRECTED APIC update - vectorized outer product for all neighbors
    # Reshape for batch matrix multiplication: [n_particles * 9, 2, 1] x [n_particles * 9, 1, 2]
    g_v_flat = g_v_all.reshape(-1, 2, 1)  # [n_particles * 9, 2, 1]
    dpos_flat = dpos_all.reshape(-1, 1, 2)  # [n_particles * 9, 1, 2]
    outer_products = torch.bmm(g_v_flat, dpos_flat).reshape(n_particles, 9, 2, 2)  # [n_particles, 9, 2, 2]

    # Weight the outer products and sum over neighbors
    weighted_outer_products = weights_all.unsqueeze(-1).unsqueeze(-1) * outer_products  # [n_particles, 9, 2, 2]
    new_C = new_C + 4 * inv_dx * weighted_outer_products.sum(dim=1)  # Sum over the 9 neighbors

    # Update particle state
    V = new_V
    C = new_C

    # Particle advection
    X = X + dt * new_V

    # margin = 2 * dx  # Keep particles away from boundaries
    # X = torch.clamp(X, margin, 1.0 - margin)



    return X, V, C, F, Jp, T, M, stress, grid_m, grid_v



def compute_surface_tension_stress(X, T, surface_tension_coeff, dx, device, debug=False):
    """
    Simplified surface tension computation that creates a stress tensor
    pulling particles toward their local center of mass.
    """
    n_particles = X.shape[0]
    surface_stress = torch.zeros((n_particles, 2, 2), device=device)
    
    # Only apply to liquid particles
    liquid_mask = (T.squeeze() == 0)
    if not liquid_mask.any() or surface_tension_coeff <= 0:
        if debug:
            print(f"No liquid particles or zero coefficient")
        return surface_stress
    
    liquid_positions = X[liquid_mask]
    liquid_indices = torch.where(liquid_mask)[0]
    n_liquid = liquid_positions.shape[0]
    
    if debug:
        print(f"Processing {n_liquid} liquid particles")
    
    # Simple approach: each liquid particle experiences attraction to nearby particles
    # This mimics surface tension by creating cohesive forces
    neighbor_radius = 3.0 * dx
    
    # Compute all pairwise distances at once for efficiency
    if n_liquid < 5000:  # For reasonable sized systems
        # Compute distance matrix
        dist_matrix = torch.cdist(liquid_positions, liquid_positions, p=2)
        
        # Count neighbors for each particle (excluding self)
        neighbor_counts = (dist_matrix < neighbor_radius).sum(dim=1) - 1
        
        # Statistics for determining surface particles
        if debug:
            print(f"Neighbor counts: min={neighbor_counts.min()}, max={neighbor_counts.max()}, mean={neighbor_counts.float().mean():.1f}")
        
        # Dynamic threshold: particles with fewer neighbors than average are on surface
        mean_neighbors = neighbor_counts.float().mean()
        std_neighbors = neighbor_counts.float().std()
        
        # Surface particles have significantly fewer neighbors (e.g., < mean - 0.5*std)
        surface_threshold = mean_neighbors - 0.5 * std_neighbors
        surface_threshold = max(surface_threshold, mean_neighbors * 0.7)  # At least 70% of mean
        
        if debug:
            print(f"Surface threshold: {surface_threshold:.1f}")
        
        # Identify surface particles
        is_surface = neighbor_counts < surface_threshold
        
        # Also mark boundary particles as surface
        x_min, x_max = liquid_positions[:, 0].min(), liquid_positions[:, 0].max()
        y_min, y_max = liquid_positions[:, 1].min(), liquid_positions[:, 1].max()
        boundary_tolerance = neighbor_radius * 0.3
        
        is_boundary = ((liquid_positions[:, 0] < x_min + boundary_tolerance) |
                      (liquid_positions[:, 0] > x_max - boundary_tolerance) |
                      (liquid_positions[:, 1] < y_min + boundary_tolerance) |
                      (liquid_positions[:, 1] > y_max - boundary_tolerance))
        
        is_surface = is_surface | is_boundary
        
        n_surface = is_surface.sum()
        if debug:
            print(f"Found {n_surface} surface particles ({100*n_surface/n_liquid:.1f}%)")
        
        # Apply surface tension stress to surface particles
        for i in range(n_liquid):
            if is_surface[i]:
                idx = liquid_indices[i]
                particle_pos = liquid_positions[i]
                
                # Find neighbors
                neighbor_mask = (dist_matrix[i] < neighbor_radius) & (dist_matrix[i] > 1e-8)
                
                if neighbor_mask.sum() > 0:
                    # Compute direction to local center of mass
                    neighbors = liquid_positions[neighbor_mask]
                    center_of_mass = neighbors.mean(dim=0)
                    direction = center_of_mass - particle_pos
                    
                    # Normalize direction
                    dist_to_center = torch.norm(direction)
                    if dist_to_center > 1e-8:
                        direction = direction / dist_to_center
                        
                        # Create stress that pulls toward center
                        # The magnitude is proportional to how much of a surface particle it is
                        surface_strength = 1.0 - neighbor_counts[i] / mean_neighbors
                        surface_strength = torch.clamp(surface_strength, 0.0, 1.0)
                        
                        magnitude = surface_tension_coeff * surface_strength
                        
                        # Stress tensor: creates contraction in the direction toward center
                        surface_stress[idx] = -magnitude * (direction.unsqueeze(-1) @ direction.unsqueeze(0))
    
    else:  # Fallback for large systems
        # For large systems, compute adaptive threshold based on sample
        for i, idx in enumerate(liquid_indices):
            particle_pos = X[idx]
            
            # Find neighbors
            distances = torch.norm(liquid_positions - particle_pos.unsqueeze(0), dim=1)
            neighbor_mask = (distances < neighbor_radius) & (distances > 1e-8)
            n_neighbors = neighbor_mask.sum()
            
            # Sample first 100 particles to estimate mean neighbors
            if i == 0:
                neighbor_samples = []
            if i < min(100, len(liquid_indices)):
                neighbor_samples.append(n_neighbors.item())
            if i == min(100, len(liquid_indices)) - 1:
                mean_neighbors = sum(neighbor_samples) / len(neighbor_samples)
                surface_threshold = mean_neighbors * 0.7  # 70% of mean
                if debug:
                    print(f"Adaptive threshold: {surface_threshold:.1f} (mean: {mean_neighbors:.1f})")
            
            # Use adaptive threshold
            if i >= min(100, len(liquid_indices)):
                if n_neighbors < surface_threshold:
                    if neighbor_mask.sum() > 0:
                        neighbors = liquid_positions[neighbor_mask]
                        center_of_mass = neighbors.mean(dim=0)
                        direction = center_of_mass - particle_pos
                        
                        dist_to_center = torch.norm(direction)
                        if dist_to_center > 1e-8:
                            direction = direction / dist_to_center
                            # Scale magnitude based on relative neighbor count
                            magnitude = surface_tension_coeff * (1.0 - n_neighbors / mean_neighbors)
                            surface_stress[idx] = -magnitude * (direction.unsqueeze(-1) @ direction.unsqueeze(0))
    
    if debug:
        stress_norms = torch.norm(surface_stress.reshape(n_particles, -1), dim=1)
        nonzero = stress_norms > 1e-10
        if nonzero.any():
            print(f"Surface stress applied to {nonzero.sum().item()} particles")
            print(f"Stress magnitudes: min={stress_norms[nonzero].min():.6f}, max={stress_norms[nonzero].max():.6f}")
        else:
            print(f"WARNING: No surface stress applied!")
    
    return surface_stress

class SafeSVD(torch.autograd.Function):
    """
    Custom SVD with numerically stable backward pass.
    Addresses the instability in torch.linalg.svd backward when:
    1. Singular values are very close together
    2. Singular values are very small
    3. Deformation gradients are extreme
    """
    
    @staticmethod
    def forward(ctx, F):
        """
        Forward: Standard SVD
        F: [batch, 2, 2] or [batch, 3, 3]
        Returns: U, sig, Vh
        """
        U, sig, Vh = torch.linalg.svd(F, driver='gesvdj')
        
        # Save for backward with regularization
        ctx.save_for_backward(U, sig, Vh)
        return U, sig, Vh
    
    @staticmethod
    def backward(ctx, grad_U, grad_sig, grad_Vh):
        """
        Backward: Stable gradient computation
        Based on: https://arxiv.org/abs/1509.07838
        With additional stabilization for extreme values
        """
        U, sig, Vh = ctx.saved_tensors
        
        # Reconstruct F = U @ diag(sig) @ Vh
        batch_size = U.shape[0]
        dim = U.shape[1]
        
        # Regularization parameters
        epsilon = 1e-6  # For division safety
        sig_separation = 1e-4  # Minimum separation between singular values
        
        # Ensure singular values are separated for stable gradients
        # This prevents 1/(sig_i - sig_j) from exploding
        sig_diff = sig.unsqueeze(-1) - sig.unsqueeze(-2)  # [batch, dim, dim]
        sig_diff_safe = torch.where(
            torch.abs(sig_diff) < sig_separation,
            torch.sign(sig_diff) * sig_separation + epsilon,
            sig_diff + epsilon * torch.sign(sig_diff)
        )
        
        # Compute gradient w.r.t. F using chain rule
        # dL/dF = U @ (dL/dSigma + F_term) @ Vh
        
        # 1. Direct gradient through singular values
        grad_sigma_diag = torch.diag_embed(grad_sig)
        
        # 2. Off-diagonal terms from U and Vh gradients
        # This is where instability occurs in standard implementation
        if grad_U is not None or grad_Vh is not None:
            # Compute K matrix with stable division
            K = 1.0 / sig_diff_safe
            K = K * (1 - torch.eye(dim, device=U.device).unsqueeze(0))
            
            F_term = torch.zeros_like(grad_sigma_diag)
            
            if grad_U is not None:
                # Symmetric term from U gradient
                U_term = U.transpose(-2, -1) @ grad_U
                F_term = F_term + (U_term - U_term.transpose(-2, -1)) * K * torch.diag_embed(sig).unsqueeze(-1)
            
            if grad_Vh is not None:
                # Symmetric term from Vh gradient  
                Vh_term = grad_Vh @ Vh.transpose(-2, -1)
                F_term = F_term + torch.diag_embed(sig).unsqueeze(-2) * K * (Vh_term - Vh_term.transpose(-2, -1))
            
            grad_F = U @ (grad_sigma_diag + F_term) @ Vh
        else:
            grad_F = U @ grad_sigma_diag @ Vh
        
        # Gradient clipping for safety
        grad_norm = torch.norm(grad_F.reshape(batch_size, -1), dim=1, keepdim=True).unsqueeze(-1)
        max_grad_norm = 100.0
        grad_F = torch.where(
            grad_norm > max_grad_norm,
            grad_F * (max_grad_norm / (grad_norm + epsilon)),
            grad_F
        )
        
        return grad_F


def safe_svd(F):
    """
    Safe SVD wrapper that can be used as drop-in replacement for torch.linalg.svd
    
    Args:
        F: [batch, d, d] deformation gradient tensor
        
    Returns:
        U, sig, Vh: SVD components with stable gradients
    """
    return SafeSVD.apply(F)


def test_safe_svd():
    """Test that safe SVD produces correct forward pass and stable gradients"""
    torch.manual_seed(42)
    
    # Test with extreme deformations
    F = torch.randn(10, 2, 2, requires_grad=True) * 10  # Large random deformations
    
    # Standard SVD
    try:
        U1, sig1, Vh1 = torch.linalg.svd(F, driver='gesvdj')
        loss1 = (U1.sum() + sig1.sum() + Vh1.sum())
        loss1.backward()
        print("Standard SVD: Forward OK, Backward OK")
        print(f"  Gradient stats: min={F.grad.min():.3f}, max={F.grad.max():.3f}, has NaN={F.grad.isnan().any()}")
    except Exception as e:
        print(f"Standard SVD failed: {e}")
    
    # Safe SVD
    F.grad = None
    U2, sig2, Vh2 = safe_svd(F)
    loss2 = (U2.sum() + sig2.sum() + Vh2.sum())
    loss2.backward()
    print("Safe SVD: Forward OK, Backward OK")
    print(f"  Gradient stats: min={F.grad.min():.3f}, max={F.grad.max():.3f}, has NaN={F.grad.isnan().any()}")
    
    # Check forward pass matches
    U1_check, sig1_check, Vh1_check = torch.linalg.svd(F.detach(), driver='gesvdj')
    print(f"\nForward pass accuracy:")
    print(f"  U error: {(U2 - U1_check).abs().max():.2e}")
    print(f"  sig error: {(sig2 - sig1_check).abs().max():.2e}")
    print(f"  Vh error: {(Vh2 - Vh1_check).abs().max():.2e}")