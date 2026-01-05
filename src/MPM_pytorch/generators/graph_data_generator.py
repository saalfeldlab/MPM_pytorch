import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
import torch
from MPM_pytorch.generators.utils import *
from MPM_pytorch.models.utils import *


from run_MPM import *
from MPM_pytorch.utils import set_size
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from scipy import stats
from matplotlib.patches import Polygon
from matplotlib.collections import PatchCollection
import tifffile
from mpl_toolkits.mplot3d.art3d import Poly3DCollection

# from fa2_modified import ForceAtlas2
# import h5py as h5
# import zarr
# import xarray as xr
import pandas as pd
import tables
import json
import torch_geometric.utils as pyg_utils
from scipy.ndimage import zoom
import re
import imageio
from MPM_pytorch.generators.utils import *

import random

from functools import partial
from PIL import Image

def data_generate(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    bSave=True,
):

    dataset_name = config.dataset

    print(f"\033[92mdataset_name: {dataset_name}\033[0m")

    if (os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.npy")) | (
        os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.pt")
    ):
        print("watch out: data already generated")
        # return

        if '3D' in config.dataset:
            data_generate_MPM_3D(
                config,
                visualize=visualize,
                run_vizualized=run_vizualized,
                style=style,
                erase=erase,
                step=step,
                alpha=0.2,
                ratio=ratio,
                scenario=scenario,
                device=device,
                bSave=bSave,
            )
        else:   #default 2D
            data_generate_MPM_2D(
                config,
                visualize=visualize,
                run_vizualized=run_vizualized,
                style=style,
                erase=erase,
                step=step,
                alpha=0.2,
                ratio=ratio,
                scenario=scenario,
                device=device,
                bSave=bSave,
            )

    plt.style.use("default")


def data_generate_MPM_2D(
        config,
        visualize=True,
        run_vizualized=0,
        style='color',
        erase=False,
        step=5,
        alpha=0.2,
        ratio=1,
        scenario='none',
        best_model=[],
        device=None,
        bSave=True
):

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'generating 2D data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = 2
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_grid = simulation_config.n_grid
    MPM_n_objects = simulation_config.MPM_n_objects
    MPM_object_type = simulation_config.MPM_object_type
    gravity = simulation_config.MPM_gravity
    friction = simulation_config.MPM_friction

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    dx, inv_dx = 1 / n_grid, float(n_grid)

    p_vol = (dx * 0.5) ** 2
    rho_list = simulation_config.MPM_rho_list
    young_coeff = simulation_config.MPM_young_coeff

    E, nu = 0.1e4 / young_coeff, 0.2  # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters
    offsets = torch.tensor([[i, j] for i in range(3) for j in range(3)],
                           device=device, dtype=torch.float32)  # [9, 2]
    particle_offsets = offsets.unsqueeze(0).expand(n_particles, -1, -1)
    expansion_factor = simulation_config.MPM_expansion_factor

    model_MPM = MPM_P2G(aggr_type='add', device=device)

    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f'./graphs_data/{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (
                    f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)
    os.makedirs(f'./graphs_data/{dataset_name}/Grid/', exist_ok=True)
    files = glob.glob(f'./graphs_data/{dataset_name}/Grid/*')
    for f in files:
        os.remove(f)

    for run in range(config.training.n_runs):
        x_list = []

        if 'cells' in config.dataset:
            # initialize 2D MPM shapes as cells
            N, X, V, C, F, T, Jp, M, S, ID = init_MPM_cells(
                n_shapes=MPM_n_objects,
                seed=simulation_config.seed,
                n_particles=n_particles,
                n_grid=n_grid,
                dx=dx,
                rho_list=rho_list,
                nucleus_ratio=0.6,
                device=device
            )
        else:
            # initialize 2D MPM shapes
            N, X, V, C, F, T, Jp, M, S, ID = init_MPM_shapes(
                geometry=MPM_object_type,
                n_shapes=MPM_n_objects,
                seed=simulation_config.seed,
                n_particles=n_particles,
                n_particle_types=n_particle_types,
                n_grid=n_grid,
                dx=dx,
                rho_list=rho_list,
                device=device
            )

        # main simulation loop
        idx = 0
        for it in trange(simulation_config.start_frame, n_frames, ncols=150):
            x = torch.cat((N.clone().detach(), X.clone().detach(), V.clone().detach(),
                           C.reshape(n_particles, 4).clone().detach(),
                           F.reshape(n_particles, 4).clone().detach(),
                           Jp.clone().detach(), T.clone().detach(), M.clone().detach(),
                           S.reshape(n_particles, 4).clone().detach(),ID.clone().detach()), 1)
            if (it >= 0):
                x_list.append(to_numpy(x))

            X, V, C, F, Jp, T, M, S, GM, GV = MPM_step(model_MPM, X, V, C, F, Jp, T, M, n_particles, n_grid,
                                                       delta_t, dx, inv_dx, mu_0, lambda_0, p_vol, offsets, particle_offsets,
                                                       expansion_factor, gravity, friction, it, device)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                if 'black' in style:
                    plt.style.use('dark_background')
                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                fig, ax = fig_init(formatx="%.1f", formaty="%.1f")

                # Determine color mode based on style
                if 'F' in style:
                    # Color by deformation gradient magnitude
                    f_norm = torch.norm(F.view(n_particles, -1), dim=1).cpu().numpy()
                    scatter = plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=f_norm, s=10, cmap='coolwarm', vmin=0.9, vmax=1.6)
                    plt.colorbar(scatter, fraction=0.046, pad=0.04)
                elif 'M' in style:
                    # Color by material type
                    for n in range(3):
                        pos = torch.argwhere(T == n)[:,0]
                        if len(pos) > 0:
                            plt.scatter(to_numpy(x[pos, 1]), to_numpy(x[pos, 2]), s=10, color=cmap.color(n))
                else:
                    # Default: color by particle ID or material type
                    for n in range(3):
                        pos = torch.argwhere(T == n)[:,0]
                        if len(pos) > 0:
                            plt.scatter(to_numpy(x[pos, 1]), to_numpy(x[pos, 2]), s=10, color=cmap.color(n))

                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                num = f"{idx:06}"
                plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=80)
                plt.close()

                if 'grid' in style:
                    plt.figure(figsize=(15, 10))
                    # 1. V particle level
                    plt.subplot(2, 3, 1)
                    plt.title('objects')
                    for n in range(3):
                        pos = torch.argwhere(T == n)[:,0]
                        plt.scatter(to_numpy(x[pos, 1]), to_numpy(x[pos, 2]), s=1, color=cmap.color(n))
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    plt.subplot(2, 3, 4)
                    plt.title('Jp (volume deformation)')
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=Jp.cpu(), s=1, cmap='viridis', vmin=0.75, vmax=1.25)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 2. C particle level
                    plt.subplot(2, 3, 2)
                    c_norm = torch.norm(C.view(n_particles, -1), dim=1).cpu().numpy()
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=c_norm, s=1, cmap='viridis', vmin=0, vmax=80)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('C (Jacobian of velocity)')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 3. F particle level
                    plt.subplot(2, 3, 3)
                    f_norm = torch.norm(F.view(n_particles, -1), dim=1).cpu().numpy()
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=f_norm, s=1, cmap='coolwarm', vmin=1.44-0.1, vmax=1.44+0.1)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    # print(
                    #     f"F min: {np.min(f_norm):.6f}, max: {np.max(f_norm):.6f}, mean: {np.mean(f_norm):.6f}, std: {np.std(f_norm):.6f}")
                    plt.title('F (deformation)')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 4. Stress particle level
                    plt.subplot(2, 3, 5)
                    stress_norm = torch.norm(S.view(n_particles, -1), dim=1)
                    stress_norm = stress_norm[:,None]
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=stress_norm[:, 0].cpu(), s=1, cmap='hot', vmin=0, vmax=6E-3)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('stress')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # # 5. M grid level - scatter plot (every 2nd point)
                    # plt.subplot(2, 3, 6)
                    # grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, n_grid), torch.linspace(0, 1, n_grid),
                    #                                 indexing='ij')
                    # # Take every 2nd row and column
                    # grid_x_sub = grid_x[::2, ::2]
                    # grid_y_sub = grid_y[::2, ::2]
                    # gm_sub = GM[::2, ::2].cpu()
                    # grid_x_flat = grid_x_sub.flatten()
                    # grid_y_flat = grid_y_sub.flatten()
                    # gm_flat = gm_sub.cpu().flatten()
                    # plt.scatter(grid_x_flat, grid_y_flat, c=gm_flat, s=4, cmap='viridis', vmin=0, vmax=1E-4)
                    # plt.colorbar(fraction=0.046, pad=0.04)
                    # plt.title('grid mass')
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 1])
                    # plt.gca().set_aspect('equal')

                    # 6. Momentum grid level - scatter plot (every 2nd point)
                    plt.subplot(2, 3, 6)
                    GP = torch.norm(GV, dim=2)
                    gp_sub = GP[::2, ::2]
                    gp_flat = gp_sub.cpu().flatten()

                    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, n_grid), torch.linspace(0, 1, n_grid),
                                                    indexing='ij')
                    grid_x_sub = grid_x[::2, ::2]
                    grid_y_sub = grid_y[::2, ::2]
                    grid_x_flat = grid_x_sub.flatten()
                    grid_y_flat = grid_y_sub.flatten()

                    plt.scatter(to_numpy(grid_x_flat), to_numpy(grid_y_flat), c=to_numpy(gp_flat), s=4, cmap='viridis', vmin=0, vmax=6)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('grid momentum')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    plt.tight_layout()
                    num = f"{idx:06}"
                    plt.savefig(f"graphs_data/{dataset_name}/Grid/Fig_{run}_{num}.png", dpi=100)
                    plt.close()

                idx += 1

        # save results
        if bSave:
            dataset_name = config.dataset
            x_array = np.array(x_list)
            np.save(f'graphs_data/{dataset_name}/generated_data_{run}.npy', x_array)
            print(f'data saved at: graphs_data/{dataset_name}/generated_data_{run}.npy')

        if visualize & (run == run_vizualized):
            config_indices = 'fig'
            src = f"./graphs_data/{dataset_name}/Fig/Fig_0_000000.png"
            dst = f"./graphs_data/{dataset_name}/input_{config_indices}.png"
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}/Fig", run=run,
                                        config_indices=config_indices, framerate=50)
            files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
            for f in files:
                os.remove(f)

            if 'grid' in style:
                config_indices = 'grid'
                src = f"./graphs_data/{dataset_name}/Grid/Fig_0_000000.png"
                dst = f"./graphs_data/{dataset_name}/input_{config_indices}.png"
                with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                    fdst.write(fsrc.read())
                generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}/Grid", run=run,
                                            config_indices=config_indices, framerate=50)
                files = glob.glob(f'./graphs_data/{dataset_name}/Grid/*')
                for f in files:
                    os.remove(f)


def data_generate_MPM_3D(
        config,
        visualize=True,
        run_vizualized=0,
        style='color',
        erase=False,
        step=5,
        alpha=0.2,
        ratio=1,
        scenario='none',
        best_model=[],
        device=None,
        bSave=True
):
    """
    3D MPM data generation function
    """

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'generating 3D data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = 3  # Force 3D
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_grid = simulation_config.n_grid
    group_size = n_particles // n_particle_types
    MPM_n_objects = simulation_config.MPM_n_objects
    MPM_object_type = simulation_config.MPM_object_type
    gravity = simulation_config.MPM_gravity
    friction = simulation_config.MPM_friction

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    dx, inv_dx = 1 / n_grid, float(n_grid)

    # 3D volume instead of 2D area
    p_vol = (dx * 0.5) ** 3
    rho_list = simulation_config.MPM_rho_list
    young_coeff = simulation_config.MPM_young_coeff

    E, nu = 0.1e4 / young_coeff, 0.2  # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = E / (2 * (1 + nu)), E * nu / ((1 + nu) * (1 - 2 * nu))  # Lame parameters

    # 3D offsets for 27 neighbors (3x3x3)
    offsets = torch.tensor([[i, j, k] for i in range(3) for j in range(3) for k in range(3)],
                           device=device, dtype=torch.float32)  # [27, 3]
    particle_offsets = offsets.unsqueeze(0).expand(n_particles, -1, -1)
    expansion_factor = simulation_config.MPM_expansion_factor

    # Initialize 3D MPM model
    model_MPM = MPM_3D_P2G(aggr_type='add', device=device)

    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f'./graphs_data/{dataset_name}/'
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (f[-3:] != 'Fig') & (f[-14:] != 'generated_data') & (f != 'p.pt') & (f != 'cycle_length.pt') & (
                    f != 'model_config.json') & (f != 'generation_code.py'):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f'./graphs_data/{dataset_name}/Fig/', exist_ok=True)
    files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    for f in files:
        os.remove(f)
    os.makedirs(f'./graphs_data/{dataset_name}/Grid/', exist_ok=True)
    files = glob.glob(f'./graphs_data/{dataset_name}/Grid/*')
    for f in files:
        os.remove(f)
    os.makedirs(f'./graphs_data/{dataset_name}/Splat/', exist_ok=True)
    files = glob.glob(f'./graphs_data/{dataset_name}/Splat/*')
    for f in files:
        os.remove(f)

    for run in range(config.training.n_runs):
        x_list = []

        # Initialize 3D MPM shapes
        N, X, V, C, F, T, Jp, M, S, ID = init_MPM_3D_shapes(
            geometry=MPM_object_type,
            n_shapes=MPM_n_objects,
            seed=simulation_config.seed,
            n_particles=n_particles,
            n_particle_types=n_particle_types,
            n_grid=n_grid,
            dx=dx,
            rho_list=rho_list,
            device=device
        )

        # # Initialize 3D MPM shapes
        # N, X, V, C, F, T, Jp, M, S, ID = init_MPM_3D_cells(
        #     n_shapes=MPM_n_objects,
        #     seed=simulation_config.seed,
        #     n_particles=n_particles,
        #     n_grid=n_grid,
        #     dx=dx,
        #     rho_list=rho_list,
        #     nucleus_ratio=0.6,
        #     device=device
        # )

        # Main simulation loop
        idx = 0
        for it in trange(simulation_config.start_frame, n_frames, ncols=150):
            # Concatenate state tensors - 3D matrices flattened to 9 components
            x = torch.cat((N.clone().detach(), X.clone().detach(), V.clone().detach(),
                           C.reshape(n_particles, 9).clone().detach(),  # 3x3 matrix flattened
                           F.reshape(n_particles, 9).clone().detach(),  # 3x3 matrix flattened
                           T.clone().detach(), Jp.clone().detach(), M.clone().detach(),
                           S.reshape(n_particles, 9).clone().detach(),  # 3x3 matrix flattened
                           ID.clone().detach()), 1)

            if (it >= 0):
                x_list.append(to_numpy(x))

            # 3D MPM step
            X, V, C, F, T, Jp, M, S, GM, GV = MPM_3D_step(
                model_MPM, X, V, C, F, T, Jp, M, n_particles, n_grid,
                delta_t, dx, inv_dx, mu_0, lambda_0, p_vol, offsets, particle_offsets,
                expansion_factor, gravity, friction, it, device
            )

            # 3D visualization (if enabled)
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                if 'black' in style:
                    plt.style.use('dark_background')

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                # determine color mode based on style
                if 'F' in style:
                    color_mode = 'F'
                    F_param = F
                elif 'M' in style:
                    color_mode = 'material'
                    F_param = None
                else:
                    color_mode = 'id'
                    F_param = None

                plot_3d_pointcloud(
                    X=X,
                    ID=ID,
                    T=T,
                    frame_idx=idx,
                    output_dir="./graphs_data",
                    dataset_name=dataset_name,
                    run=run,
                    color_mode=color_mode,
                    F=F_param
                )

                export_for_gaussian_splatting(
                    X=X,
                    ID=ID,
                    T=T,
                    frame_idx=idx,
                    output_dir="./graphs_data",
                    dataset_name=dataset_name,
                    particle_volumes=None,
                    color_mode=color_mode,
                    F=F_param,
                    output_format='ply',  # Options: 'ply', 'splat', or 'both',
                    splat_scale=0.005,
                    opacity=0.1,
                    debug=False
                )

                idx += 1





        if bSave:

            x_array = np.array(x_list)
            np.save(f'graphs_data/{dataset_name}/generated_data_{run}.npy', x_array)
            print(f'data saved at: graphs_data/{dataset_name}/generated_data_{run}.npy')


        if visualize & (run == run_vizualized):
            config_indices = dataset_name.split('fly_N9_')[1] if 'fly_N9_' in dataset_name else 'no_id'
            src = f"./graphs_data/{dataset_name}/Fig/Fig_0_000000.png"
            dst = f"./graphs_data/{dataset_name}/input_{config_indices}.png"
            with open(src, "rb") as fsrc, open(dst, "wb") as fdst:
                fdst.write(fsrc.read())
            generate_compressed_video_mp4(output_dir=f"./graphs_data/{dataset_name}", run=run,
                                        config_indices=config_indices, framerate=50)
            files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
            for f in files:
                os.remove(f)




    # Save configuration
    with open(f"graphs_data/{dataset_name}/model_config.json", "w") as json_file:
        json.dump(config, json_file, default=lambda o: dict(o) if hasattr(o, '__dict__') else str(o))

    print(f'3D MPM data generation completed for {config.training.n_runs} runs')

