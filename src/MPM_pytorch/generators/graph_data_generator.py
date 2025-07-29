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
import taichi as ti
import random
import jax
import jax.numpy as jp
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
    has_particle_field = (
        "PDE_ParticleField" in config.graph_model.particle_model_name
    ) | ("PDE_F" in config.graph_model.particle_model_name)
    has_signal = "PDE_N" in config.graph_model.signal_model_name
    has_mesh = config.graph_model.mesh_model_name != ""
    has_cell_division = config.simulation.has_cell_division
    has_WBI = "WBI" in config.dataset
    has_fly = "fly" in config.dataset
    has_city = ("mouse_city" in config.dataset) | ("rat_city" in config.dataset)
    has_MPM = "MPM" in config.graph_model.particle_model_name
    dataset_name = config.dataset

    print("")
    print(f"dataset_name: {dataset_name}")

    if (os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.npy")) | (
        os.path.isfile(f"./graphs_data/{dataset_name}/x_list_0.pt")
    ):
        print("watch out: data already generated")
        # return

    if config.data_folder_name != "none":
        generate_from_data(config=config, device=device, visualize=visualize)
    elif has_city:
        data_generate_rat_city(
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
    elif has_particle_field:
        data_generate_particle_field(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            alpha=0.2,
            ratio=ratio,
            scenario="none",
            device=device,
            bSave=bSave,
        )
    elif has_mesh:
        data_generate_mesh(
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
    elif has_cell_division:
        data_generate_cell(
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
    elif has_WBI:
        data_generate_WBI(
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
    elif has_fly:
        data_generate_fly_voltage(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
        )
    elif has_signal:
        data_generate_synaptic(
            config,
            visualize=visualize,
            run_vizualized=run_vizualized,
            style=style,
            erase=erase,
            step=step,
            device=device,
            bSave=bSave,
        )
    elif has_MPM:
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
        else:
            data_generate_MPM(
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
    else:
        data_generate_particle(
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



def taichi_MPM():
    torch.manual_seed(42)
    np.random.seed(42)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(42)

    ti.init(arch=ti.gpu)

    # Try to run on GPU
    quality = 1  # Use a larger value for higher-res simulations
    n_particles, n_grid = 9000 * quality**2, 128 * quality
    dx, inv_dx = 1 / n_grid, float(n_grid)
    dt = 1e-4 / quality
    p_vol, p_rho = (dx * 0.5) ** 2, 1
    p_mass = p_vol * p_rho
    E, nu = 0.1e4, 0.2  # Young's modulus and Poisson's ratio
    mu_0, lambda_0 = (
        E / (2 * (1 + nu)),
        E * nu / ((1 + nu) * (1 - 2 * nu)),
    )  # Lame parameters

    x = ti.Vector.field(2, dtype=float, shape=n_particles)  # position
    v = ti.Vector.field(2, dtype=float, shape=n_particles)  # velocity
    C = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # affine velocity field
    F = ti.Matrix.field(2, 2, dtype=float, shape=n_particles)  # deformation gradient
    material = ti.field(dtype=int, shape=n_particles)  # material id
    Jp = ti.field(dtype=float, shape=n_particles)  # plastic deformation
    grid_v = ti.Vector.field(
        2, dtype=float, shape=(n_grid, n_grid)
    )  # grid node momentum/velocity
    grid_m = ti.field(dtype=float, shape=(n_grid, n_grid))  # grid node mass

    @ti.kernel
    def substep():
        for i, j in grid_m:
            grid_v[i, j] = [0, 0]
            grid_m[i, j] = 0
        for p in x:  # Particle state update and scatter to grid (P2G)
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            # Quadratic kernels  [http://mpm.graphics   Eqn. 123, with x=fx, fx-1,fx-2]
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1) ** 2, 0.5 * (fx - 0.5) ** 2]
            # F[p]: deformation gradient update
            F[p] = (ti.Matrix.identity(float, 2) + dt * C[p]) @ F[p]
            h = ti.exp(10 * (1.0 - Jp[p]))
            if material[p] == 1:  # jelly, make it softer
                h = 0.3
            mu, la = mu_0 * h, lambda_0 * h
            if material[p] == 0:  # liquid
                mu = 0.0

            U, sig, V = ti.svd(F[p])

            # Avoid zero eigenvalues because of numerical errors
            for d in ti.static(range(2)):
                sig[d, d] = ti.max(sig[d, d], 1e-6)
            J = 1.0
            for d in ti.static(range(2)):
                new_sig = sig[d, d]
                if material[p] == 2:  # Snow
                    new_sig = ti.min(
                        ti.max(sig[d, d], 1 - 2.5e-2), 1 + 4.5e-3
                    )  # Plasticity
                Jp[p] *= sig[d, d] / new_sig
                sig[d, d] = new_sig
                J *= new_sig
            if material[p] == 0:
                # Reset deformation gradient to avoid numerical instability
                F[p] = ti.Matrix.identity(float, 2) * ti.sqrt(J)
            elif material[p] == 2:
                # Reconstruct elastic deformation gradient after plasticity
                F[p] = U @ sig @ V.transpose()
            stress = 2 * mu * (F[p] - U @ V.transpose()) @ F[
                p
            ].transpose() + ti.Matrix.identity(float, 2) * la * J * (J - 1)
            stress = (-dt * p_vol * 4 * inv_dx * inv_dx) * stress
            affine = stress + p_mass * C[p]
            # Loop over 3x3 grid node neighborhood
            for i, j in ti.static(ti.ndrange(3, 3)):
                offset = ti.Vector([i, j])
                dpos = (offset.cast(float) - fx) * dx
                weight = w[i][0] * w[j][1]
                grid_v[base + offset] += weight * (p_mass * v[p] + affine @ dpos)
                grid_m[base + offset] += weight * p_mass

        for i, j in grid_m:
            if grid_m[i, j] > 0:  # No need for epsilon here
                grid_v[i, j] = (1 / grid_m[i, j]) * grid_v[i, j]  # Momentum to velocity
                grid_v[i, j][1] -= dt * 50  # gravity
                if i < 3 and grid_v[i, j][0] < 0:
                    grid_v[i, j][0] = 0  # Boundary conditions
                if i > n_grid - 3 and grid_v[i, j][0] > 0:
                    grid_v[i, j][0] = 0
                if j < 3 and grid_v[i, j][1] < 0:
                    grid_v[i, j][1] = 0
                if j > n_grid - 3 and grid_v[i, j][1] > 0:
                    grid_v[i, j][1] = 0
        for p in x:  # grid to particle (G2P)
            base = (x[p] * inv_dx - 0.5).cast(int)
            fx = x[p] * inv_dx - base.cast(float)
            w = [0.5 * (1.5 - fx) ** 2, 0.75 - (fx - 1.0) ** 2, 0.5 * (fx - 0.5) ** 2]
            new_v = ti.Vector.zero(float, 2)
            new_C = ti.Matrix.zero(float, 2, 2)
            for i, j in ti.static(ti.ndrange(3, 3)):
                # loop over 3x3 grid node neighborhood
                dpos = ti.Vector([i, j]).cast(float) - fx
                g_v = grid_v[base + ti.Vector([i, j])]
                weight = w[i][0] * w[j][1]
                new_v += weight * g_v
                new_C += 4 * inv_dx * weight * g_v.outer_product(dpos)
            v[p], C[p] = new_v, new_C
            x[p] += dt * v[p]  # advection

    group_size = n_particles // 3

    @ti.kernel
    def initialize():
        for i in range(n_particles):
            x[i] = [
                ti.random() * 0.2 + 0.3 + 0.10 * (i // group_size),
                ti.random() * 0.2 + 0.05 + 0.32 * (i // group_size),
            ]
            material[i] = 1  # i // group_size  # 0: fluid 1: jelly 2: snow
            v[i] = ti.Matrix([0, 0])
            F[i] = ti.Matrix([[1, 0], [0, 1]])
            Jp[i] = 1

    initialize()

    # for n in range(2000):
    #     substep()

    # Separate particle visualization
    x_np = x.to_numpy()
    material_np = material.to_numpy()

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

    colors = ["blue", "red", "green"]
    material_names = ["Liquid", "Jelly", "Snow"]

    # Full domain view
    for mat_type in range(3):
        mask = material_np == mat_type
        if np.any(mask):
            ax1.scatter(
                x_np[mask, 0],
                x_np[mask, 1],
                s=3,
                color=colors[mat_type],
                label=material_names[mat_type],
                alpha=0.7,
            )
    ax1.set_xlim([0, 1])
    ax1.set_ylim([0, 1])
    ax1.set_title("Final Particle Positions")
    ax1.set_xlabel("x")
    ax1.set_ylabel("y")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Zoomed view
    for mat_type in range(3):
        mask = material_np == mat_type
        if np.any(mask):
            ax2.scatter(
                x_np[mask, 0],
                x_np[mask, 1],
                s=8,
                color=colors[mat_type],
                label=material_names[mat_type],
                alpha=0.7,
            )
    ax2.set_xlim([0.2, 0.8])
    ax2.set_ylim([0.2, 0.8])
    ax2.set_title("Particle Positions (Zoomed)")
    ax2.set_xlabel("x")
    ax2.set_ylabel("y")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    plt.tight_layout()
    plt.savefig("particles_taichi.png", dpi=150, bbox_inches="tight")
    plt.close()

    gui = ti.GUI("Taichi MLS-MPM-99", res=512, background_color=0x112F41)
    while not gui.get_event(ti.GUI.ESCAPE, ti.GUI.EXIT):
        for s in range(int(2e-3 // dt)):
            substep()
        gui.circles(
            x.to_numpy(),
            radius=1.5,
            palette=[0x068587, 0xED553B, 0xEEEEF0],
            palette_indices=material,
        )
        # Change to gui.show(f'{frame:06d}.png') to write images to disk
        gui.show()


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

    for run in range(config.training.n_runs):
        x_list = []

        # Initialize 3D MPM shapes
        N, X, V, C, F, T, Jp, M, S, ID = init_MPM_3D_shapes(
            geometry=MPM_object_type,
            n_shapes=MPM_n_objects,
            seed=42,
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
        #     seed=42,
        #     n_particles=n_particles,
        #     n_grid=n_grid,
        #     dx=dx,
        #     rho_list=rho_list,
        #     nucleus_ratio=0.6,
        #     device=device
        # )

        # Main simulation loop
        for it in trange(simulation_config.start_frame, n_frames):
            # Concatenate state tensors - 3D matrices flattened to 9 components
            x = torch.cat((N.clone().detach(), X.clone().detach(), V.clone().detach(),
                           C.reshape(n_particles, 9).clone().detach(),  # 3x3 matrix flattened
                           F.reshape(n_particles, 9).clone().detach(),  # 3x3 matrix flattened
                           T.clone().detach(), Jp.clone().detach(), M.clone().detach(),
                           S.reshape(n_particles, 9).clone().detach(),  # 3x3 matrix flattened
                           ID.clone().detach()), 1)

            if (it >= 0):
                x_list.append(x.clone().detach())

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

                if 'color' in style:
                    # 1. 3D Angled View
                    from mpl_toolkits.mplot3d import Axes3D

                    import pyvista as pv

                    def plot_3d_shaded_pointcloud(X, ID, T, output_path):
                        plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
                        plotter.set_background("lightgray")

                        MPM_n_objects = 3

                        for n in range(min(3, MPM_n_objects)):
                            pos = torch.argwhere(T == n)[:, 0]
                            if len(pos) > 0:
                                # pts = to_numpy(X[pos])
                                pts = to_numpy(X[pos])[:, [0, 2, 1]]
                                ids = to_numpy(ID[pos].squeeze())
                                cloud = pv.PolyData(pts)
                                cloud["ID"] = ids
                                plotter.add_points(
                                    cloud,
                                    scalars="ID",
                                    cmap="nipy_spectral",
                                    point_size=5,
                                    render_points_as_spheres=True,
                                    opacity=0.9,
                                    show_scalar_bar=False  # ✅ correct
                                )

                        # Add bounding box (wireframe cube)
                        cube = pv.Cube(center=(0.5, 0.5, 0.5), x_length=1.0, y_length=1.0, z_length=1.0)
                        frame = cube.extract_all_edges()
                        plotter.add_mesh(frame, color='white', line_width=1.0, opacity=0.5)

                        # Add axes with bounds
                        # plotter.show_bounds(grid='back', location='outer', all_edges=True,
                        #                     color='white', line_width=1.5,
                        #                     xlabel='X', ylabel='Y', zlabel='Z')

                        plotter.view_vector((1.1, 0.9, 0.45))

                        plotter.enable_eye_dome_lighting()

                        plotter.camera.zoom(1.1)

                        plotter.screenshot(output_path)
                        plotter.close()

                    def plot_3d_material_pointcloud_separated(X, ID, T, output_path):
                        """Alternative version that plots each material type separately for better control"""
                        plotter = pv.Plotter(off_screen=True, window_size=(1800, 1200))
                        plotter.set_background("lightgray")

                        # Get unique material types
                        unique_materials = torch.unique(T).cpu().numpy()

                        # Define colors for different materials
                        material_colors = ['blue', 'red', 'green', 'yellow', 'purple', 'orange', 'cyan', 'magenta']
                        material_names = ['Liquid', 'Jelly', 'Snow', 'Material_3', 'Material_4', 'Material_5',
                                          'Material_6', 'Material_7']

                        for i, mat in enumerate(unique_materials):
                            pos = torch.argwhere(T.squeeze() == mat)[:, 0]
                            if len(pos) > 0:
                                pts = to_numpy(X[pos])[:, [0, 2, 1]]  # Swap y and z coordinates

                                cloud = pv.PolyData(pts)
                                color = material_colors[i % len(material_colors)]
                                mat_name = material_names[i % len(material_names)]

                                plotter.add_points(
                                    cloud,
                                    color=color,
                                    point_size=15,
                                    render_points_as_spheres=True,
                                    opacity=0.05,
                                    label=f'{mat_name} (Type {mat})'
                                )

                        # Add bounding box (wireframe cube)
                        cube = pv.Cube(center=(0.5, 0.5, 0.5), x_length=1.0, y_length=1.0, z_length=1.0)
                        frame = cube.extract_all_edges()
                        plotter.add_mesh(frame, color='white', line_width=1.0, opacity=0.5)

                        plotter.view_vector((1.1, 0.9, 0.45))
                        plotter.enable_eye_dome_lighting()
                        plotter.camera.zoom(1.1)

                        plotter.screenshot(output_path)
                        plotter.close()

                    output_path_3d = f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it:06}_3D.png"
                    plot_3d_shaded_pointcloud(X, ID, T, output_path_3d)

                    #
                    # fig = plt.figure(figsize=(18, 12))
                    # ax = fig.add_subplot(2, 3, 1, projection='3d')
                    # ax.set_title('3D Angled View')
                    # for n in range(min(3, MPM_n_objects)):
                    #     pos = torch.argwhere(T == n)[:,0]
                    #     if len(pos) > 0:
                    #         ax.scatter(to_numpy(X[pos, 0]), to_numpy(X[pos, 2]), to_numpy(X[pos, 1]),
                    #                    cmap='nipy_spectral', edgecolors = 'none',
                    #                    s=20, c=to_numpy(ID.squeeze()), alpha=0.7)
                    #
                    #         # plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), c=to_numpy(ID.squeeze()), s=2, alpha=0.3,
                    #         #             cmap='nipy_spectral', edgecolors = 'none')
                    #
                    # ax.set_xlim([0, 1])
                    # ax.set_ylim([0, 1])
                    # ax.set_zlim([0, 1])
                    # ax.set_xlabel('X')
                    # ax.set_ylabel('Y')
                    # ax.set_zlabel('Z')
                    # # Set viewing angle (elevation, azimuth)
                    # ax.view_init(elev=20, azim=45)
                    # plt.tight_layout()
                    # num = f"{it:06}"
                    # plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.png", dpi=150, bbox_inches='tight')
                    # plt.close()

        if bSave:
            torch.save(x_list, f'graphs_data/{dataset_name}/generated_data_{run}.pt')
            print(f'data saved at: graphs_data/{dataset_name}/generated_data_{run}.pt')

    # Save configuration
    with open(f"graphs_data/{dataset_name}/model_config.json", "w") as json_file:
        json.dump(config, json_file, default=lambda o: dict(o) if hasattr(o, '__dict__') else str(o))

    print(f'3D MPM data generation completed for {config.training.n_runs} runs')


def data_generate_MPM(
        config,
        visualize=True,
        run_vizualized=0,
        style='color',
        erase=False,
        step=5,
        alpha=0.2,
        ratio=1,
        scenario='none',
        device=None,
        bSave=True
):
    #taichi_MPM_deubg()

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(f'generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}')

    dimension = simulation_config.dimension
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

        if 'cells' in dataset_name:
            N, X, V, C, F, T, Jp, M, S, ID = init_MPM_cells(n_shapes=MPM_n_objects, seed=42, n_particles=n_particles,
                                                            n_grid=n_grid, dx=dx, rho_list=rho_list, nucleus_ratio=0.6,
                                                            device=device)
        else:
            N, X, V, C, F, T, Jp, M, S, ID = init_MPM_shapes(geometry=MPM_object_type, n_shapes=MPM_n_objects, seed=42, n_particles=n_particles,
                                                   n_particle_types=n_particle_types, n_grid=n_grid, dx=dx, rho_list=rho_list, device=device)
# Main simulation loop
        for it in trange(simulation_config.start_frame, n_frames):
            x = torch.cat((N.clone().detach(), X.clone().detach(), V.clone().detach(),
                           C.reshape(n_particles, 4).clone().detach(),
                           F.reshape(n_particles, 4).clone().detach(),
                           T.clone().detach(), Jp.clone().detach(), M.clone().detach(),
                           S.reshape(n_particles, 4).clone().detach(),ID.clone().detach()), 1)
            if (it >= 0):
                x_list.append(x.clone().detach())

            X, V, C, F, T, Jp, M, S, GM, GV = MPM_step(model_MPM, X, V, C, F, T, Jp, M, n_particles, n_grid,
                                                       delta_t, dx, inv_dx, mu_0, lambda_0, p_vol, offsets, particle_offsets,
                                                       expansion_factor, gravity, friction, it, device)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):

                if 'black' in style:
                    plt.style.use('dark_background')

                if 'latex' in style:
                    plt.rcParams['text.usetex'] = True
                    rc('font', **{'family': 'serif', 'serif': ['Palatino']})

                if 'color' in style:

                    fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    for n in range(3):
                        pos = torch.argwhere(T == n)[:,0]
                        plt.scatter(to_numpy(x[pos, 1]), to_numpy(x[pos, 2]), s=1, color=cmap.color(n))
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    num = f"{it:06}"
                    plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80)
                    plt.close()

                    plt.figure(figsize=(15, 10))

                    # 1. V particle level
                    plt.subplot(2, 3, 1)
                    # v_norm = torch.norm(V, dim=1).cpu().numpy()
                    # plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=v_norm, s=1, cmap='viridis', vmin=0, vmax=6)
                    # plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('objects')

                    for n in range(3):
                        pos = torch.argwhere(T == n)[:,0]
                        plt.scatter(to_numpy(x[pos, 1]), to_numpy(x[pos, 2]), s=1, color=cmap.color(n))

                    # Overlay transparent color based on object ID
                    # plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), c='w', s=2, edgecolors = 'none')
                    # plt.scatter(to_numpy(x[:, 1]), to_numpy(x[:, 2]), c=to_numpy(ID.squeeze()), s=2, alpha=0.3,
                    #             cmap='nipy_spectral', edgecolors = 'none')

                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.tight_layout()
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 2. C particle level
                    plt.subplot(2, 3, 2)
                    c_norm = torch.norm(C.view(n_particles, -1), dim=1).cpu().numpy()
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=c_norm, s=1, cmap='viridis', vmin=0, vmax=80)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('C (affine velocity)')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 3. F particle level
                    plt.subplot(2, 3, 3)
                    f_norm = torch.norm(F.view(n_particles, -1), dim=1).cpu().numpy()
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=f_norm, s=1, cmap='coolwarm', vmin=1.44-0.2, vmax=1.44+0.2)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    # print(
                    #     f"F min: {np.min(f_norm):.6f}, max: {np.max(f_norm):.6f}, mean: {np.mean(f_norm):.6f}, std: {np.std(f_norm):.6f}")
                    plt.title('F (deformation)')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 4. Stress particle level
                    plt.subplot(2, 3, 4)
                    stress_norm = torch.norm(S.view(n_particles, -1), dim=1)
                    stress_norm = stress_norm[:,None]
                    plt.scatter(X[:, 0].cpu(), X[:, 1].cpu(), c=stress_norm[:, 0].cpu(), s=1, cmap='hot', vmin=0, vmax=6E-3)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('stress')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 5. M grid level - scatter plot (every 2nd point)
                    plt.subplot(2, 3, 5)
                    grid_x, grid_y = torch.meshgrid(torch.linspace(0, 1, n_grid), torch.linspace(0, 1, n_grid),
                                                    indexing='ij')
                    # Take every 2nd row and column
                    grid_x_sub = grid_x[::2, ::2]
                    grid_y_sub = grid_y[::2, ::2]
                    gm_sub = GM[::2, ::2].cpu()
                    grid_x_flat = grid_x_sub.flatten()
                    grid_y_flat = grid_y_sub.flatten()
                    gm_flat = gm_sub.cpu().flatten()
                    plt.scatter(grid_x_flat, grid_y_flat, c=gm_flat, s=4, cmap='viridis', vmin=0, vmax=1E-4)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('grid mass')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    # 6. Momentum grid level - scatter plot (every 2nd point)
                    plt.subplot(2, 3, 6)
                    GP = torch.norm(GV, dim=2)
                    gp_sub = GP[::2, ::2]
                    gp_flat = gp_sub.cpu().flatten()
                    plt.scatter(grid_x_flat, grid_y_flat, c=gp_flat, s=4, cmap='viridis', vmin=0, vmax=6)
                    plt.colorbar(fraction=0.046, pad=0.04)
                    plt.title('grid momentum')
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.gca().set_aspect('equal')

                    plt.tight_layout()
                    num = f"{it:06}"
                    plt.savefig(f"graphs_data/{dataset_name}/Grid/Fig_{run}_{num}.tif", dpi=80)
                    plt.close()

        # Save results
        if bSave:
            dataset_name = config.dataset
            x_list = np.array(to_numpy(torch.stack(x_list)))
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)

