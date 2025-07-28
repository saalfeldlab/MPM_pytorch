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


def generate_from_data(config, device, visualize=True, step=None, cmap=None):
    data_folder_name = config.data_folder_name
    image_data = config.image_data

    if data_folder_name == "graphs_data/solar_system":
        load_solar_system(config, device, visualize, step)
    elif "RGB" in config.graph_model.particle_model_name:
        load_RGB_grid_data(config, device, visualize, step)
    elif "LG-ODE" in data_folder_name:
        load_LG_ODE(config, device, visualize, step)
    elif "WaterDropSmall" in data_folder_name:
        load_WaterDropSmall(config, device, visualize, step, cmap)
    elif "WaterRamps" in data_folder_name:
        load_Goole_data(config, device, visualize, step, cmap)
    elif "MultiMaterial" in data_folder_name:
        load_Goole_data(config, device, visualize, step, cmap)
    elif "Kato" in data_folder_name:
        load_worm_Kato_data(config, device, visualize, step)
    elif "wormvae" in data_folder_name:
        load_wormvae_data(config, device, visualize, step)
    elif "NeuroPAL" in data_folder_name:
        load_neuropal_data(config, device, visualize, step)
    elif "U2OS" in data_folder_name:
        load_2Dfluo_data_on_mesh(config, device, visualize, step)
    elif "cardio" in data_folder_name:
        load_2Dgrid_data(config, device, visualize, step)
    elif image_data.file_type != "none":
        if image_data.file_type == "3D fluo Cellpose":
            load_3Dfluo_data_with_Cellpose(config, device, visualize)
        if image_data.file_type == "2D fluo Cellpose":
            load_2Dfluo_data_with_Cellpose(config, device, visualize)
    else:
        raise ValueError(f"Unknown data folder name {data_folder_name}")


def data_generate_particle(
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
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    connection_matrix_list = []

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)

    if config.data_folder_name != "none":
        print(f"generating from data ...")
        generate_from_data(
            config=config, device=device, visualize=visualize, step=step, cmap=cmap
        )
        return

    # create GNN
    model, bc_pos, bc_dpos = choose_model(config=config, device=device)

    particle_dropout_mask = np.arange(n_particles)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if simulation_config.angular_Bernouilli != [-1]:
        b = simulation_config.angular_Bernouilli
        generative_m = np.array([stats.norm(b[0], b[2]), stats.norm(b[1], b[2])])

    for run in range(config.training.n_runs):
        check_and_clear_memory(
            device=device,
            iteration_number=0,
            every_n_iterations=250,
            memory_percentage_threshold=0.6,
        )

        if "PDE_K" in model_config.particle_model_name:
            p = config.simulation.params
            edges = np.random.choice(p[0], size=(n_particles, n_particles), p=p[1])
            edges = np.tril(edges) + np.tril(edges, -1).T
            np.fill_diagonal(edges, 0)
            connection_matrix = torch.tensor(edges, dtype=torch.float32, device=device)
            model.connection_matrix = connection_matrix.detach().clone()
            connection_matrix_list.append(connection_matrix)

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        edge_p_p_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_particles(
            config=config, scenario=scenario, ratio=ratio, device=device
        )

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):
            # calculate type change
            if simulation_config.state_type == "sequence":
                sample = torch.rand((len(T1), 1), device=device)
                sample = (
                    sample < (1 / config.simulation.state_params[0])
                ) * torch.randint(0, n_particle_types, (len(T1), 1), device=device)
                T1 = (T1 + sample) % n_particle_types

            x = torch.concatenate(
                (
                    N1.clone().detach(),
                    X1.clone().detach(),
                    V1.clone().detach(),
                    T1.clone().detach(),
                    H1.clone().detach(),
                    A1.clone().detach(),
                ),
                1,
            )

            index_particles = get_index_particles(
                x, n_particle_types, dimension
            )  # can be different from frame to frame

            # compute connectivity rule

            distance = torch.sum(
                bc_dpos(x[:, None, 1 : dimension + 1] - x[None, :, 1 : dimension + 1])
                ** 2,
                dim=2,
            )
            adj_t = (
                (distance < max_radius**2) & (distance > min_radius**2)
            ).float() * 1
            edge_index = adj_t.nonzero().t().contiguous()
            edge_p_p_list.append(to_numpy(edge_index))

            dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index, field=[])

            # model prediction
            with torch.no_grad():
                y = model(dataset)

            if simulation_config.angular_sigma > 0:
                phi = (
                    torch.randn(n_particles, device=device)
                    * simulation_config.angular_sigma
                    / 360
                    * np.pi
                    * 2
                )
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)
                new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
                new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()
            if simulation_config.angular_Bernouilli != [-1]:
                z_i = stats.bernoulli(b[3]).rvs(n_particles)
                phi = np.array([g.rvs() for g in generative_m[z_i]]) / 360 * np.pi * 2
                phi = torch.tensor(phi, device=device, dtype=torch.float32)
                cos_phi = torch.cos(phi)
                sin_phi = torch.sin(phi)
                new_vx = cos_phi * y[:, 0] - sin_phi * y[:, 1]
                new_vy = sin_phi * y[:, 0] + cos_phi * y[:, 1]
                y = torch.cat((new_vx[:, None], new_vy[:, None]), 1).clone().detach()

            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    y_list.append(y[particle_dropout_mask].clone().detach())
                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())

            # Particle update

            if model_config.prediction == "2nd_derivative":
                V1 += y * delta_t
            else:
                V1 = y
            X1 = bc_pos(X1 + V1 * delta_t)
            A1 = A1 + 1

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                if "black" in style:
                    plt.style.use("dark_background")

                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                if "bw" in style:
                    fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    s_p = 100
                    for n in range(n_particle_types):
                        plt.scatter(
                            to_numpy(x[index_particles[n], 1]),
                            to_numpy(x[index_particles[n], 2]),
                            s=s_p,
                            color="k",
                        )
                    if training_config.particle_dropout > 0:
                        plt.scatter(
                            x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                            s=25,
                            color="k",
                            alpha=0.75,
                        )
                        plt.plot(
                            x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                            "+",
                            color="w",
                        )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if "PDE_G" in model_config.particle_model_name:
                        plt.xlim([-2, 2])
                        plt.ylim([-2, 2])
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7
                    )
                    plt.close()

                if "color" in style:
                    if model_config.particle_model_name == "PDE_O":
                        fig = plt.figure(figsize=(12, 12))
                        plt.scatter(
                            H1[:, 0].detach().cpu().numpy(),
                            H1[:, 1].detach().cpu().numpy(),
                            s=100,
                            c=np.sin(to_numpy(H1[:, 2])),
                            vmin=-1,
                            vmax=1,
                            cmap="viridis",
                        )
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Lut_Fig_{run}_{it}.jpg",
                            dpi=170.7,
                        )
                        plt.close()

                        fig = plt.figure(figsize=(12, 12))
                        # plt.scatter(H1[:, 0].detach().cpu().numpy(), H1[:, 1].detach().cpu().numpy(), s=5, c='b')
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=10,
                            c="lawngreen",
                            alpha=0.75,
                        )
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Rot_{run}_Fig{it}.jpg",
                            dpi=170.7,
                        )
                        plt.close()

                    elif "PDE_N" in model_config.signal_model_name:
                        matplotlib.rcParams["savefig.pad_inches"] = 0
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                        ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                        ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                        plt.scatter(
                            to_numpy(X1[:, 1]),
                            to_numpy(X1[:, 0]),
                            s=200,
                            c=to_numpy(H1[:, 0]) * 3,
                            cmap="viridis",
                        )  # vmin=0, vmax=3)
                        plt.colorbar()
                        plt.xlim([-1.2, 1.2])
                        plt.ylim([-1.2, 1.2])
                        # plt.text(0, 1.1, f'frame {it}', ha='left', va='top', transform=ax.transAxes, fontsize=24)
                        # cbar = plt.colorbar(shrink=0.5)
                        # cbar.ax.tick_params(labelsize=32)
                        if "latex" in style:
                            plt.xlabel(r"$x$", fontsize=78)
                            plt.ylabel(r"$y$", fontsize=78)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                        elif "frame" in style:
                            plt.xlabel("x", fontsize=48)
                            plt.ylabel("y", fontsize=48)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                            ax.tick_params(axis="both", which="major", pad=15)
                            plt.text(
                                0,
                                1.1,
                                f"frame {it}",
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                                fontsize=48,
                            )
                        else:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()
                        num = f"{it:06}"
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=70,
                        )
                        plt.close()

                    elif (model_config.particle_model_name == "PDE_A") & (
                        dimension == 3
                    ):
                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(111, projection="3d")
                        for n in range(n_particle_types):
                            ax.scatter(
                                to_numpy(x[index_particles[n], 2]),
                                to_numpy(x[index_particles[n], 1]),
                                to_numpy(x[index_particles[n], 3]),
                                s=50,
                                color=cmap.color(n),
                            )
                        ax.set_xlim([0, 1])
                        ax.set_ylim([0, 1])
                        ax.set_zlim([0, 1])
                        pl.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg",
                            dpi=170.7,
                        )
                        plt.close()

                    else:
                        # matplotlib.use("Qt5Agg")

                        fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                        s_p = 25

                        # if 'PDE_K' in model_config.particle_model_name:
                        #     s_p = 5

                        for n in range(n_particle_types):
                            plt.scatter(
                                to_numpy(x[index_particles[n], 2]),
                                to_numpy(x[index_particles[n], 1]),
                                s=s_p,
                                color=cmap.color(n),
                            )
                        if training_config.particle_dropout > 0:
                            plt.scatter(
                                x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                                x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                s=25,
                                color="k",
                                alpha=0.75,
                            )
                            plt.plot(
                                x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                                x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                                "+",
                                color="w",
                            )

                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                        if "PDE_G" in model_config.particle_model_name:
                            plt.xlim([-2, 2])
                            plt.ylim([-2, 2])
                        if "latex" in style:
                            plt.xlabel(r"$x$", fontsize=78)
                            plt.ylabel(r"$y$", fontsize=78)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                        if "frame" in style:
                            plt.xlabel("x", fontsize=48)
                            plt.ylabel("y", fontsize=48)
                            plt.xticks(fontsize=48.0)
                            plt.yticks(fontsize=48.0)
                            ax.tick_params(axis="both", which="major", pad=15)
                            plt.text(
                                0,
                                1.1,
                                f"frame {it}",
                                ha="left",
                                va="top",
                                transform=ax.transAxes,
                                fontsize=48,
                            )
                        if "no_ticks" in style:
                            plt.xticks([])
                            plt.yticks([])
                        plt.tight_layout()

                        num = f"{it:06}"
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=80,
                        )  # 170.7)
                        plt.close()

        if bSave:
            x_list = np.array(to_numpy(torch.stack(x_list)))
            y_list = np.array(to_numpy(torch.stack(y_list)))
            # torch.save(x_list, f'graphs_data/{dataset_name}/x_list_{run}.pt')
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )
            # torch.save(y_list, f'graphs_data/{dataset_name}/y_list_{run}.pt')
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
            np.savez(f"graphs_data/{dataset_name}/edge_p_p_list_{run}", *edge_p_p_list)

            torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")

    if "PDE_K" in model_config.particle_model_name:
        torch.save(
            connection_matrix_list,
            f"graphs_data/{dataset_name}/connection_matrix_list.pt",
        )

    # for handler in logger.handlers[:]:
    #     handler.close()
    #     logger.removeHandler(handler)


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

        # N, X, V, C, F, T, Jp, M, S, ID = init_MPM_shapes(geometry=MPM_object_type, n_shapes=MPM_n_objects, seed=42, n_particles=n_particles,
        #                                              n_particle_types=n_particle_types, n_grid=n_grid, dx=dx, rho_list=rho_list, device=device)
        N, X, V, C, F, T, Jp, M, S, ID = init_MPM_cells(n_shapes=MPM_n_objects, seed=42, n_particles=n_particles,
                                                     n_grid=n_grid, dx=dx, rho_list=rho_list, nucleus_ratio=0.6, device=device)
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

                    # fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    # for n in range(3):
                    #     pos = torch.argwhere(T == n)[:,0]
                    #     plt.scatter(to_numpy(x[pos, 1]), to_numpy(x[pos, 2]), s=1, color=cmap.color(n))
                    # plt.xlim([0, 1])
                    # plt.ylim([0, 1])
                    # plt.tight_layout()
                    # num = f"{it:06}"
                    # plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80)
                    # plt.close()

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




def data_generate_particle_field(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=1,
    alpha=0.2,
    ratio=1,
    scenario="none",
    device=None,
    bSave=True,
):
    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(config.training.seed)

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles = simulation_config.n_particles
    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))
    delta_t = simulation_config.delta_t
    has_adjacency_matrix = simulation_config.connectivity_file != ""
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    bounce = simulation_config.bounce
    bounce_coeff = simulation_config.bounce_coeff
    speedlim = config.plotting.speedlim

    # Create log directory
    log_dir = f"./graphs_data/{dataset_name}/"
    log_file = f"{log_dir}/generator.log"
    os.makedirs(log_dir, exist_ok=True)
    logging.basicConfig(
        filename=log_file, format="%(asctime)s %(message)s", filemode="w"
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)
    copyfile(os.path.realpath(__file__), os.path.join(folder, "generation_code.py"))

    if "calculus" in model_config.field_type:
        model, bc_pos, bc_dpos = choose_model(config=config, device=device)
    else:
        model_p_p, bc_pos, bc_dpos = choose_model(config=config, device=device)
        model_f_p = model_p_p
        # model_f_f = choose_mesh_model(config, device=device)

    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(
            np.arange(
                (n_particles // n_particle_types) * n,
                (n_particles // n_particle_types) * (n + 1),
            )
        )
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_particles))
        cut = int(n_particles * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []
    else:
        particle_dropout_mask = np.arange(n_particles)
    if has_adjacency_matrix:
        mat = scipy.io.loadmat(simulation_config.connectivity_file)
        adjacency = torch.tensor(mat["A"], device=device)
        adj_t = adjacency > 0
        edge_index = adj_t.nonzero().t().contiguous()
        edge_attr_adjacency = adjacency[adj_t]

    for run in range(config.training.n_runs):
        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        x_mesh_list = []
        y_mesh_list = []
        edge_p_p_list = []
        edge_f_p_list = []

        # initialize particle and mesh states
        X1, V1, T1, H1, A1, N1 = init_particles(
            config=config, scenario=scenario, ratio=ratio, device=device
        )
        X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(
            config, device=device
        )

        # matplotlib.use("Qt5Agg")
        # fig = plt.figure(figsize=(12, 12))
        # im = torch.reshape(H1_mesh[:,0:1],(100,100))
        # plt.imshow(to_numpy(im))
        # plt.colorbar()

        torch.save(mesh_data, f"graphs_data/{dataset_name}/mesh_data_{run}.pt")
        mask_mesh = mesh_data["mask"].squeeze()

        check_and_clear_memory(
            device=device,
            iteration_number=0,
            every_n_iterations=250,
            memory_percentage_threshold=0.6,
        )
        time.sleep(1)

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):
            if ("siren" in model_config.field_type) & (it >= 0):
                im = imread(
                    f"graphs_data/{simulation_config.node_value_map}"
                )  # / 255 * 5000
                im = im[it].squeeze()
                im = np.rot90(im, 3)
                im = np.reshape(im, (n_nodes_per_axis * n_nodes_per_axis))
                H1_mesh[:, 0:1] = torch.tensor(
                    im[:, None], dtype=torch.float32, device=device
                )

            x = torch.concatenate(
                (
                    N1.clone().detach(),
                    X1.clone().detach(),
                    V1.clone().detach(),
                    T1.clone().detach(),
                    H1.clone().detach(),
                    A1.clone().detach(),
                ),
                1,
            )

            x_mesh = torch.concatenate(
                (
                    N1_mesh.clone().detach(),
                    X1_mesh.clone().detach(),
                    V1_mesh.clone().detach(),
                    T1_mesh.clone().detach(),
                    H1_mesh.clone().detach(),
                    A1_mesh.clone().detach(),
                ),
                1,
            )

            x_particle_field = torch.concatenate((x_mesh, x), dim=0)

            # model prediction
            if "calculus" in model_config.field_type:
                distance = torch.sum(
                    bc_dpos(
                        x[:, None, 1 : dimension + 1] - x[None, :, 1 : dimension + 1]
                    )
                    ** 2,
                    dim=2,
                )
                adj_t = ((distance < max_radius**2) & (distance >= 0)).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                # pos = torch.argwhere(edge_index[0, :] > 2000)
                # edge_index = edge_index[:, pos.squeeze()]
                dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                y = model(dataset_p_p)
                y = y[:, 0:dimension] - 1e-4 * x[:, 3:5] / delta_t
                # y = y + torch.randn(y.shape, device=device) * 0.001
                y0 = y.clone().detach()
                y1 = y.clone().detach()

                if bounce:
                    V1[0 : n_particles // n_particle_types] = 0
                    y[0 : n_particles // n_particle_types] = 0

                density = model.density

            else:
                distance = torch.sum(
                    bc_dpos(
                        x[:, None, 1 : dimension + 1] - x[None, :, 1 : dimension + 1]
                    )
                    ** 2,
                    dim=2,
                )
                adj_t = (
                    (distance < max_radius**2) & (distance > min_radius**2)
                ).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                dataset_p_p = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if not (has_particle_dropout):
                    edge_p_p_list.append(edge_index)

                distance = torch.sum(
                    bc_dpos(
                        x_particle_field[:, None, 1 : dimension + 1]
                        - x_particle_field[None, :, 1 : dimension + 1]
                    )
                    ** 2,
                    dim=2,
                )
                adj_t = (
                    (distance < (max_radius / 2) ** 2) & (distance > min_radius**2)
                ).float() * 1
                edge_index = adj_t.nonzero().t().contiguous()
                pos = torch.argwhere(
                    (edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes)
                )
                pos = to_numpy(pos[:, 0])
                edge_index = edge_index[:, pos]
                dataset_f_p = data.Data(
                    x=x_particle_field,
                    pos=x_particle_field[:, 1:3],
                    edge_index=edge_index,
                )
                if not (has_particle_dropout):
                    edge_f_p_list.append(edge_index)

                with torch.no_grad():
                    y0 = model_p_p(dataset_p_p, has_field=False)
                    y1 = model_f_p(dataset_f_p, has_field=True)[n_nodes:]
                    y = y0 + y1

            # append list
            if (it >= 0) & bSave:
                if has_particle_dropout:
                    x_ = x[inv_particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_removed_list.append(x[inv_particle_dropout_mask].clone().detach())
                    x_ = x[particle_dropout_mask].clone().detach()
                    x_[:, 0] = torch.arange(len(x_), device=device)
                    x_list.append(x_)
                    y_list.append(y[particle_dropout_mask].clone().detach())

                    distance = torch.sum(
                        bc_dpos(
                            x_[:, None, 1 : dimension + 1]
                            - x_[None, :, 1 : dimension + 1]
                        )
                        ** 2,
                        dim=2,
                    )
                    adj_t = (
                        (distance < max_radius**2) & (distance > min_radius**2)
                    ).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    edge_p_p_list.append(edge_index)

                    x_particle_field = torch.concatenate((x_mesh, x_), dim=0)

                    distance = torch.sum(
                        bc_dpos(
                            x_particle_field[:, None, 1 : dimension + 1]
                            - x_particle_field[None, :, 1 : dimension + 1]
                        )
                        ** 2,
                        dim=2,
                    )
                    adj_t = (
                        (distance < (max_radius / 2) ** 2) & (distance > min_radius**2)
                    ).float() * 1
                    edge_index = adj_t.nonzero().t().contiguous()
                    pos = torch.argwhere(
                        (edge_index[1, :] >= n_nodes) & (edge_index[0, :] < n_nodes)
                    )
                    pos = to_numpy(pos[:, 0])
                    edge_index = edge_index[:, pos]
                    edge_f_p_list.append(edge_index)
                else:
                    x_list.append(x.clone().detach())
                    y_list.append(y.clone().detach())
                    if torch.isnan(x).any() | torch.isnan(y).any():
                        print("nan")

            # Particle update
            with torch.no_grad():
                if model_config.prediction == "2nd_derivative":
                    V1 += y * delta_t
                else:
                    V1 = y
                if bounce:
                    # V1 = V1 * 0.999
                    X1 = X1 + V1 * delta_t
                    gap = 0.005
                    bouncing_pos = torch.argwhere(
                        (X1[:, 0] <= 0.1 + gap) | (X1[:, 0] >= 0.9 - gap)
                    ).squeeze()
                    if bouncing_pos.numel() > 0:
                        V1[bouncing_pos, 0] = -0.7 * bounce_coeff * V1[bouncing_pos, 0]
                        X1[bouncing_pos, 0] += V1[bouncing_pos, 0] * delta_t * 10
                    bouncing_pos = torch.argwhere(
                        (X1[:, 1] <= 0.1 + gap) | (X1[:, 1] >= 0.9 - gap)
                    ).squeeze()
                    if bouncing_pos.numel() > 0:
                        V1[bouncing_pos, 1] = -0.7 * bounce_coeff * V1[bouncing_pos, 1]
                        X1[bouncing_pos, 1] += V1[bouncing_pos, 1] * delta_t * 10
                else:
                    X1 = bc_pos(X1 + V1 * delta_t)
                A1 = A1 + 1

            # Mesh update

            if "calculus" not in model_config.field_type:
                x_mesh_list.append(x_mesh.clone().detach())
                pred = x_mesh[:, 6:7]
                y_mesh_list.append(pred)

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if "black" in style:
                    plt.style.use("dark_background")

                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                if "field" in style:
                    # distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                    # adj_t = ((distance < max_radius ** 2) & (distance >= 0)).float() * 1
                    # edge_index = adj_t.nonzero().t().contiguous()
                    # pos = torch.argwhere(edge_index[1,:]==3393)
                    # pos = edge_index[0,pos.squeeze()]

                    # density_field = to_numpy(density_field)
                    #
                    # # matplotlib.use("Qt5Agg")
                    # fig = plt.figure(figsize=(8, 8))
                    # plt.xticks([])
                    # plt.yticks([])
                    # im = np.reshape(density_field, (100, 100))
                    # # im = np.flipud(im)
                    # im_resized = zoom(im, 10)
                    # plt.imshow(im_resized, vmin=0, vmax=16, cmap='bwr')
                    # # plt.scatter(to_numpy(x_mesh[:, 1] * 1000), to_numpy(x_mesh[:, 2] * 1000), c=density_field, s=40, vmin=2, vmax=6, cmap='bwr')
                    # # plt.text(20, 950, f'{np.mean(density_field):0.3}+/-{np.std(density_field):0.3}', c='k', fontsize=18)
                    # plt.scatter(to_numpy(x[:, 1]*1000), to_numpy(x[:, 2]*1000), s=1, c='k')
                    # # plt.scatter(to_numpy(x[pos, 1] * 1000), to_numpy(x[pos, 2] * 1000), s=10, c='b')
                    # plt.axis('off')
                    # plt.xlim([0,1000])
                    # plt.ylim([-40,1000])
                    # plt.tight_layout()
                    # # plt.show()
                    # plt.savefig(f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=80)
                    # plt.close()

                    fig = plt.figure(figsize=(8, 8))
                    plt.xticks([])
                    plt.yticks([])
                    plt.axis("off")
                    # velocity_field = to_numpy(velocity_field)
                    # im = np.reshape(velocity_field, (100, 100))
                    # im = np.flipud(im)
                    # im_resized = zoom(im, 10)
                    # plt.imshow(im_resized, cmap='viridis', vmin=speedlim[0], vmax=speedlim[1])

                    # for n in range(n_particle_types):
                    #         plt.scatter(to_numpy(x[index_particles[n], 1]*1000), to_numpy(x[index_particles[n], 2]*1000),
                    #                     s=10, color=cmap.color(n), edgecolors='None', alpha=0.9)

                    # plt.scatter(1000 * to_numpy(x[:, 1]), 1000 * to_numpy(x[:, 2]), c=model.correction.detach().cpu().numpy(), s=2, cmap='viridis', vmin =0 , vmax=0.4)

                    plt.xlim([0, 1000])
                    plt.ylim([-40, 1000])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=160
                    )
                    # plt.show()
                    plt.close()

                    # Q = ax.quiver(to_numpy(x[:, 1]), to_numpy(x[:, 2]), to_numpy(y[:, 0]), to_numpy(y[:, 1]), color='r')
                    # Q = ax.quiver(to_numpy(x[:, 1]), to_numpy(x[:, 2]), to_numpy(x[:, 3]), to_numpy(x[:, 4]), color='w')
                    # ax = fig.add_subplot(2,4,3)
                    # plt.scatter(to_numpy(model.delta_pos[:, 0]), to_numpy(model.delta_pos[:, 1]), s=0.1,
                    #             c=to_numpy(model.kernel_operators[:, 0:1]))
                    # plt.title('kernel')
                    # ax = fig.add_subplot(2,4,4)
                    # std_list.append(torch.std((density_field),dim=0))
                    # plt.plot(to_numpy(torch.stack(std_list)), c='w')
                    # plt.xlim([0,200])
                    # plt.ylim([0,1])

                if "graph" in style:
                    fig = plt.figure(figsize=(10, 10))

                    if model_config.mesh_model_name == "RD_Mesh":
                        H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                        plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                    elif (model_config.mesh_model_name == "Wave_Mesh") | (
                        model_config.mesh_model_name == "DiffMesh"
                    ):
                        pts = x_mesh[:, 1:3].detach().cpu().numpy()
                        tri = Delaunay(pts)
                        colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                        if model_config.mesh_model_name == "WaveMesh":
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                edgecolors="k",
                                vmin=-2500,
                                vmax=2500,
                            )
                        else:
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                edgecolors="k",
                                vmin=0,
                                vmax=5000,
                            )
                        plt.xlim([0, 1])
                        plt.ylim([0, 1])
                    elif model_config.particle_model_name == "PDE_G":
                        for n in range(n_particle_types):
                            plt.scatter(
                                x[index_particles[n], 1].detach().cpu().numpy(),
                                x[index_particles[n], 2].detach().cpu().numpy(),
                                s=40,
                                color=cmap.color(n),
                            )
                    elif model_config.particle_model_name == "PDE_E":
                        for n in range(n_particle_types):
                            g = 40
                            if simulation_config.params[n][0] <= 0:
                                plt.scatter(
                                    x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(),
                                    s=g,
                                    c=cmap.color(n),
                                )
                            else:
                                plt.scatter(
                                    x[index_particles[n], 1].detach().cpu().numpy(),
                                    x[index_particles[n], 2].detach().cpu().numpy(),
                                    s=g,
                                    c=cmap.color(n),
                                )
                    else:
                        for n in range(n_particle_types):
                            plt.scatter(
                                x[index_particles[n], 1].detach().cpu().numpy(),
                                x[index_particles[n], 2].detach().cpu().numpy(),
                                s=25,
                                color=cmap.color(n),
                                alpha=0.5,
                            )
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_g_color_{it}.tif", dpi=300
                    )
                    plt.close()

                if "bw" in style:
                    plt.rcParams["text.usetex"] = False
                    plt.rc("font", family="sans-serif")
                    plt.rc("text", usetex=False)
                    matplotlib.rcParams["savefig.pad_inches"] = 0

                    plt.style.use("dark_background")

                    fig, ax = fig_init(formatx="%.1f", formaty="%.1f")
                    # plt.xlabel(r'$x$', fontsize=48)
                    # plt.ylabel(r'$y$', fontsize=48)
                    for n in range(n_particle_types):
                        plt.scatter(
                            to_numpy(x[index_particles[n], 2]),
                            to_numpy(x[index_particles[n], 1]),
                            color=cmap.color(n),
                            s=20,
                        )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7
                    )
                    plt.close()

                    if model_config.prediction == "2nd_derivative":
                        V0_ = y0 * delta_t
                        V1_ = y1 * delta_t
                    else:
                        V0_ = y0
                        V1_ = y1
                    fig = plt.figure(figsize=(12, 12))
                    type_list = to_numpy(get_type_list(x, dimension))
                    plt.scatter(
                        to_numpy(x_mesh[0:n_nodes, 2]),
                        to_numpy(x_mesh[0:n_nodes, 1]),
                        c=to_numpy(x_mesh[0:n_nodes, 6]),
                        cmap="grey",
                        s=5,
                    )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    for n in range(n_particles):
                        plt.arrow(
                            x=to_numpy(x[n, 2]),
                            y=to_numpy(x[n, 1]),
                            dx=to_numpy(V1_[n, 1]) * 4.25,
                            dy=to_numpy(V1_[n, 0]) * 4.25,
                            color=cmap.color(type_list[n].astype(int)),
                            head_width=0.004,
                            length_includes_head=True,
                        )
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Arrow_{run}_{it}.jpg",
                        dpi=170.7,
                    )

                if "color" in style:
                    # matplotlib.use("Qt5Agg")
                    matplotlib.rcParams["savefig.pad_inches"] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    # ax.xaxis.get_major_formatter()._usetex = False
                    # ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.tick_params(axis="both", which="major", pad=15)
                    # if (has_mesh | (simulation_config.boundary == 'periodic')):
                    #     ax = plt.axes([0, 0, 1, 1], frameon=False)
                    # else:
                    #     ax = plt.axes([-2, -2, 2, 2], frameon=False)
                    # ax.get_xaxis().set_visible(False)
                    # ax.get_yaxis().set_visible(False)
                    # plt.autoscale(tight=True)
                    s_p = 20
                    for n in range(n_particle_types):
                        plt.scatter(
                            to_numpy(x[index_particles[n], 2]),
                            to_numpy(x[index_particles[n], 1]),
                            s=s_p,
                            color=cmap.color(n),
                        )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    # plt.xlim([-2,2])
                    # plt.ylim([-2,2])
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel("x", fontsize=78)
                        plt.ylabel("y", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7
                    )
                    plt.close()

                    matplotlib.rcParams["savefig.pad_inches"] = 0

                    if model_config.prediction == "2nd_derivative":
                        V0_ = y0 * delta_t
                        V1_ = y1 * delta_t
                    else:
                        V0_ = y0
                        V1_ = y1
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    type_list = to_numpy(get_type_list(x, dimension))
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.tick_params(axis="both", which="major", pad=15)
                    plt.scatter(
                        to_numpy(x_mesh[0:n_nodes, 2]),
                        to_numpy(x_mesh[0:n_nodes, 1]),
                        c=to_numpy(x_mesh[0:n_nodes, 6]),
                        cmap="grey",
                        s=5,
                    )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    for n in range(n_particles):
                        plt.arrow(
                            x=to_numpy(x[n, 2]),
                            y=to_numpy(x[n, 1]),
                            dx=to_numpy(V1_[n, 1]) * 4.25,
                            dy=to_numpy(V1_[n, 0]) * 4.25,
                            color=cmap.color(type_list[n].astype(int)),
                            head_width=0.004,
                            length_includes_head=True,
                        )
                    # plt.xlim([-2,2])
                    # plt.ylim([-2,2])
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel("x", fontsize=78)
                        plt.ylabel("y", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Arrow_{run}_{it}.jpg",
                        dpi=170.7,
                    )
                    plt.close()

        if bSave:
            x_list_ = np.array(to_numpy(torch.stack(x_list)))
            y_list_ = np.array(to_numpy(torch.stack(y_list)))
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list_)
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list_)
            del x_list_, y_list_

            torch.save(x_mesh_list, f"graphs_data/{dataset_name}/x_mesh_list_{run}.pt")
            torch.save(y_mesh_list, f"graphs_data/{dataset_name}/y_mesh_list_{run}.pt")
            torch.save(
                edge_p_p_list, f"graphs_data/{dataset_name}/edge_p_p_list{run}.pt"
            )
            torch.save(
                edge_f_p_list, f"graphs_data/{dataset_name}/edge_f_p_list{run}.pt"
            )

            # torch.save(model_p_p.p, f'graphs_data/{dataset_name}/model_p.pt')


def data_generate_cell(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    device=None,
    bSave=True,
):
    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(config.training.seed)

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model
    image_data = config.image_data

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dimension = simulation_config.dimension
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particle_types = simulation_config.n_particle_types
    n_particles_max = simulation_config.n_particles_max
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset
    marker_size = config.plotting.marker_size
    has_inert_model = simulation_config.cell_inert_model_coeff > 0
    has_cell_death = simulation_config.has_cell_death
    has_cell_division = True

    max_radius_list = []
    edges_len_list = []
    folder = f"./graphs_data/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-3:] != "Fig")
                & (f[-2:] != "GT")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
        files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
        for f in files:
            os.remove(f)

    logging.basicConfig(
        filename=f"./graphs_data/{dataset_name}/generator.log",
        format="%(asctime)s %(message)s",
        filemode="w",
    )
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    kill_cell_leaving = simulation_config.kill_cell_leaving

    for run in range(config.training.n_runs):
        torch.cuda.empty_cache()

        model, bc_pos, bc_dpos = choose_model(config=config, device=device)

        n_particles = simulation_config.n_particles

        x_list = []
        y_list = []
        area_list = []
        d_pos = []
        x_len_list = []
        edge_p_p_list = []
        vertices_pos_list = []
        vertices_per_cell_list = []
        current_loss = []

        """
        INITIALIZE PER CELL TYPE VALUES1000
        cycle_length
        final_cell_mass
        cell_death_rate

        INITIALIZE PER CELL VALUES
        0 N1 cell index dim=1
        1,2 X1 positions dim=2
        3,4 V1 velocities dim=2
        5 T1 cell type dim=1
        6,7 H1 cell status dim=2  H1[:,0] = cell alive flag, alive : 0 , death : 0 , H1[:,1] = cell division flag, dividing : 1
        8 A1 cell age dim=1
        9 S1 cell stage dim=1  0 = G1 , 1 = S, 2 = G2, 3 = M
        10 M1 cell_mass dim=1 (per node)
        11 R1 cell growth rate dim=1
        12 CL1 cell cycle length dim=1
        13 DR1 cell death rate dim=1
        14 AR1 area of the cell
        15 P1 cell perimeter
        16 ASR1 aspect ratio
        17 OR1 orientation
        """

        if run == 0:
            cycle_length, final_cell_mass, cell_death_rate, cell_area = init_cell_range(
                config, device=device
            )

        N1, X1, V1, T1, H1, A1, S1, M1, R1, CL1, DR1, AR1, P1 = init_cells(
            config,
            cycle_length,
            final_cell_mass,
            cell_death_rate,
            cell_area,
            bc_pos,
            bc_dpos,
            dimension,
            device=device,
        )

        coeff = 0
        num_cells = []
        for i in range(n_particle_types):
            pos = torch.argwhere(T1.squeeze() == i).shape[0]
            num_cells.append(pos)
            coeff += num_cells[i] * cell_area[i]
        target_areas_per_type = torch.tensor(
            [cell_area[i] / coeff for i in range(n_particle_types)], device=device
        )
        target_areas = (
            target_areas_per_type[to_numpy(T1).astype(int)].squeeze().clone().detach()
        )

        T1_list = T1.clone().detach()

        man_track = torch.cat((N1 + 1, torch.zeros((len(N1), 3), device=device)), 1)
        man_track[:, 2] = -1

        logger.info("cell cycle length")
        logger.info(to_numpy(cycle_length))
        logger.info("cell death rate")
        logger.info(to_numpy(cell_death_rate))
        logger.info("cell final mass")
        logger.info(to_numpy(final_cell_mass))
        logger.info("interaction parameters")
        logger.info(to_numpy(model.p))

        index_particles = []
        for n in range(n_particle_types):
            pos = torch.argwhere(T1.squeeze() == n)
            pos = to_numpy(pos[:, 0].squeeze()).astype(int)
            index_particles.append(pos)
        n_particles_alive = len(X1)
        n_particles_dead = 0

        has_cell_division = True
        has_cell_death = True

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):
            # calculate cell death and cell division

            if has_cell_death:
                sample = torch.rand(len(X1), device=device)
                if kill_cell_leaving:
                    pos = torch.argwhere(
                        ((AR1.squeeze() < 2e-4) & (A1.squeeze() > 25))
                        | (sample.squeeze() < DR1.squeeze() / 5e4)
                        | (X1[:, 0] < 0)
                        | (X1[:, 0] > 1)
                        | (X1[:, 1] < 0)
                        | (X1[:, 1] > 1)
                    )
                else:
                    pos = torch.argwhere(
                        ((AR1.squeeze() < 2e-4) & (A1.squeeze() > 25))
                        | (sample.squeeze() < DR1.squeeze() / 5e4)
                    )
                if len(pos) > 0:
                    H1[pos, 0] = 0
                    man_track[to_numpy(N1[pos]).astype(int), 2] = it - 1
                n_particles_alive = torch.sum(H1[:, 0])
                n_particles_dead = n_particles - n_particles_alive

            if (it > 0) & (has_cell_division):
                # cell division
                pos = torch.argwhere(
                    (A1.squeeze() >= CL1.squeeze())
                    & (H1[:, 0].squeeze() == 1)
                    & (S1[:, 0].squeeze() == 3)
                    & (n_particles_alive < n_particles_max)
                ).flatten()
                if len(pos) > 0:
                    n_add_nodes = len(pos) * 2
                    pos = to_numpy(pos).astype(int)

                    N1_ = n_particles + torch.arange(n_add_nodes, device=device)
                    N1 = torch.cat((N1, N1_[:, None]), dim=0)

                    # man_track = tracklet ID, start time, end time, parent tracklet
                    man_track_ = torch.cat(
                        (
                            N1_[:, None] + 1,
                            torch.zeros((n_add_nodes, 3), device=device),
                        ),
                        1,
                    )  # cell ID
                    man_track_[:, 1] = it  # start time
                    man_track_[:, 2] = -1  # end time
                    man_track_[0 : n_add_nodes // 2, 3:4] = N1[pos] + 1  # parent cell
                    man_track_[n_add_nodes // 2 : n_add_nodes, 3:4] = (
                        N1[pos] + 1
                    )  # parent cell
                    man_track = torch.cat((man_track, man_track_), 0)
                    man_track[to_numpy(N1[pos]).astype(int), 2] = it - 1  # end time

                    n_particles = n_particles + n_add_nodes

                    angle = torch.atan(V1[pos, 1] / (V1[pos, 0] + 1e-10))
                    separation = [torch.cos(angle) * 0.005, torch.sin(angle) * 0.005]
                    separation = torch.stack(separation)
                    separation = separation.t()

                    X1 = torch.cat(
                        (X1, X1[pos, :] + separation, X1[pos, :] - separation), dim=0
                    )

                    nd = torch.ones(len(pos), device=device) + 0.05 * torch.randn(
                        len(pos), device=device
                    )
                    var = torch.ones(len(pos), device=device) + 0.20 * torch.randn(
                        len(pos), device=device
                    )

                    V1 = torch.cat(
                        (V1, V1[pos, :], -V1[pos, :]), dim=0
                    )  # the new cell is moving away from its mother
                    T1 = torch.cat((T1, T1[pos, :], T1[pos, :]), dim=0)

                    T1_list = torch.cat((T1_list, T1[pos, :], T1[pos, :]), dim=0)

                    H1[pos, 0] = 0  # mother cell is removed, considered dead
                    H1[pos, 1] = 1  # cell division flag
                    H1 = torch.concatenate(
                        (H1, torch.ones((n_add_nodes, 2), device=device)), 0
                    )
                    H1[-n_add_nodes:, 1] = (
                        0  # cell division flag = 0 for new daughter cells
                    )
                    A1 = torch.cat((A1, torch.ones((n_add_nodes, 1), device=device)), 0)
                    S1 = torch.cat((S1, torch.ones((n_add_nodes, 1), device=device)), 0)
                    M1 = torch.cat(
                        (
                            M1,
                            final_cell_mass[to_numpy(T1[pos, 0]), None] / 2,
                            final_cell_mass[to_numpy(T1[pos, 0]), None] / 2,
                        ),
                        dim=0,
                    )
                    CL1 = torch.cat(
                        (
                            CL1,
                            cycle_length[to_numpy(T1[pos, 0]), None] * var[:, None],
                            cycle_length[to_numpy(T1[pos, 0]), None] * var[:, None],
                        ),
                        dim=0,
                    )
                    DR1 = torch.cat(
                        (
                            DR1,
                            cell_death_rate[to_numpy(T1[pos, 0]), None] * nd[:, None],
                            cell_death_rate[to_numpy(T1[pos, 0]), None] * nd[:, None],
                        ),
                        dim=0,
                    )
                    AR1 = torch.cat((AR1, AR1[pos, :], AR1[pos, :]), dim=0)
                    P1 = torch.cat((P1, P1[pos, :], P1[pos, :]), dim=0)
                    R1 = M1 / (2 * CL1)

                    target_areas = torch.cat(
                        (target_areas, target_areas[pos], target_areas[pos]), dim=0
                    )

                    n_particles_alive = torch.sum(H1[:, 0])

                    if n_particles_alive >= simulation_config.n_particles_max:
                        has_cell_division = False
                        has_cell_death = False

            alive = torch.argwhere(H1[:, 0] == 1).squeeze()

            N1 = N1[alive]
            X1 = X1[alive]
            V1 = V1[alive]
            T1 = T1[alive]
            H1 = H1[alive]
            A1 = A1[alive]
            S1 = S1[alive]
            M1 = M1[alive]
            CL1 = CL1[alive]
            R1 = R1[alive]
            DR1 = DR1[alive]
            AR1 = AR1[alive]
            P1 = P1[alive]
            target_areas = target_areas[alive]

            index_particles = []
            for n in range(n_particle_types):
                pos = torch.argwhere(T1.squeeze() == n)
                pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                index_particles.append(pos)

            # calculate cell type change
            if simulation_config.state_type == "sequence":
                sample = torch.rand((len(T1), 1), device=device)
                sample = (
                    sample < (1 / config.simulation.state_params[0])
                ) * torch.randint(0, n_particle_types, (len(T1), 1), device=device)
                T1 = (T1 + sample) % n_particle_types

            A1 = A1 + delta_t  # update age

            if n_particles_alive < n_particles_max:
                S1 = update_cell_cycle_stage(A1, cycle_length, T1, device)
                M1 += R1 * delta_t

            if it == simulation_config.start_frame:
                ID1 = torch.arange(len(N1), device=device)[:, None]
            else:
                ID1 = torch.arange(
                    int(ID1[-1] + 1), int(ID1[-1] + len(N1) + 1), device=device
                )[:, None]

            x = torch.concatenate(
                (
                    N1.clone().detach(),
                    X1.clone().detach(),
                    V1.clone().detach(),
                    T1.clone().detach(),
                    H1.clone().detach(),
                    A1.clone().detach(),
                    S1.clone().detach(),
                    M1.clone().detach(),
                    R1.clone().detach(),
                    DR1.clone().detach(),
                    AR1.clone().detach(),
                    P1.clone().detach(),
                    ID1.clone().detach(),
                ),
                1,
            )

            # calculate connectivity
            with torch.no_grad():
                edge_index = torch.sum(
                    bc_dpos(
                        x[:, None, 1 : dimension + 1] - x[None, :, 1 : dimension + 1]
                    )
                    ** 2,
                    dim=2,
                )
                edge_index = (
                    (edge_index < max_radius**2) & (edge_index > min_radius**2)
                ).float() * 1
                edge_index = edge_index.nonzero().t().contiguous()
                edge_p_p_list.append(to_numpy(edge_index))
                alive = (H1[:, 0] == 1).float() * 1.0
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                if edge_index.shape[1] > simulation_config.max_edges:
                    max_radius = max_radius / 1.025
                else:
                    max_radius = max_radius * 1.0025
                max_radius = np.clip(
                    max_radius,
                    simulation_config.min_radius,
                    simulation_config.max_radius,
                )
                max_radius_list.append(max_radius)
                edges_len_list.append(edge_index.shape[1])
                x_len_list.append(x.shape[0])

            # model prediction
            with torch.no_grad():
                y = model(dataset, has_field=True)
                y = (
                    y
                    * alive[:, None].repeat(1, 2)
                    * simulation_config.cell_active_model_coeff
                )

            first_X1 = X1.clone().detach()

            if has_inert_model:
                X1_ = X1.clone().detach()
                X1_.requires_grad = True

                optimizer = torch.optim.Adam([X1_], lr=1e-3)
                optimizer.zero_grad()
                vor, vertices_pos, vertices_per_cell, all_points = get_vertices(
                    points=X1_, device=device
                )
                cc, tri = get_Delaunay(all_points, device)
                distance = torch.sum(
                    (vertices_pos[:, None, :].clone().detach() - cc[None, :, :]) ** 2,
                    dim=2,
                )
                result = distance.min(dim=1)
                index = result.indices
                cc = cc[index]

                voronoi_area = get_voronoi_areas(cc, vertices_per_cell, device)
                perimeter = get_voronoi_perimeters(cc, vertices_per_cell, device)
                AR1 = voronoi_area[:, None].clone().detach()
                P1 = perimeter[:, None].clone().detach()

                loss = simulation_config.coeff_area * (
                    target_areas - voronoi_area
                ).norm(2)
                loss += simulation_config.coeff_perimeter * torch.sum(perimeter**2)

                loss.backward()
                optimizer.step()

                # print(f'loss {loss.item()}')
                # fig = plt.figure(figsize=(12, 12))
                # ax = fig.add_subplot(1, 1, 1)
                # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)
                # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
                #                 point_size=0)
                # plt.scatter(to_numpy(cc[:, 0]), to_numpy(cc[:, 1]), s=1, color='r')
                #
                # fig = plt.figure(figsize=(12, 12))
                # ax = fig.add_subplot(1, 1, 1)
                # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1, device=device)
                # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='black', line_width=1, line_alpha=0.5,
                #                 point_size=0)
                # vor, vertices_pos, vertices_per_cell, all_points = get_vertices(points=X1_, device=device)
                # voronoi_plot_2d(vor, ax=ax, show_vertices=False, line_colors='red', line_width=1, line_alpha=0.5,
                #                 point_size=0)

                current_loss.append(loss.item())

                X1 = bc_pos(X1_.clone().detach())

            if model_config.prediction == "2nd_derivative":
                y_voronoi = (
                    (bc_dpos(X1 - first_X1) / delta_t - V1)
                    / delta_t
                    * simulation_config.cell_inert_model_coeff
                )
            else:
                y_voronoi = (
                    bc_dpos(X1 - first_X1)
                    / delta_t
                    * simulation_config.cell_inert_model_coeff
                )

            # append list
            if it >= 0:
                x_list.append(x)
                y_list.append(y + y_voronoi)

            # cell update
            if model_config.prediction == "2nd_derivative":
                V1 += (y + y_voronoi) * delta_t
            else:
                V1 = y + y_voronoi

            if kill_cell_leaving:
                X1 = first_X1 + V1 * delta_t
            else:
                X1 = bc_pos(X1 + V1 * delta_t)

            if has_inert_model:
                vor, vertices_pos, vertices_per_cell, all_points = get_vertices(
                    points=X1, device=device
                )
                vertices_pos_list.append(to_numpy(vertices_pos))

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                plt.style.use("dark_background")
                # matplotlib.use("Qt5Agg")

                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                if "bw" in style:
                    matplotlib.rcParams["savefig.pad_inches"] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    for n in range(n_particle_types):
                        plt.scatter(
                            to_numpy(x[index_particles[n], 1]),
                            to_numpy(x[index_particles[n], 2]),
                            s=marker_size,
                            color="k",
                        )
                    if training_config.particle_dropout > 0:
                        plt.scatter(
                            x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                            s=25,
                            color="k",
                            alpha=0.75,
                        )
                        plt.plot(
                            x[inv_particle_dropout_mask, 1].detach().cpu().numpy(),
                            x[inv_particle_dropout_mask, 2].detach().cpu().numpy(),
                            "+",
                            color="w",
                        )
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    else:
                        plt.xticks([])
                        plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{it}.jpg", dpi=170.7
                    )
                    plt.close()

                if "color" in style:
                    # matplotlib.use("Qt5Agg")
                    matplotlib.rcParams["savefig.pad_inches"] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.xaxis.get_major_formatter()._usetex = False
                    ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    index_particles = []
                    for n in range(n_particle_types):
                        pos = torch.argwhere(
                            (T1.squeeze() == n) & (H1[:, 0].squeeze() == 1)
                        )
                        pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                        index_particles.append(pos)
                        if "inv" in style:
                            plt.scatter(
                                to_numpy(X1[index_particles[n], 0]),
                                1 - to_numpy(X1[index_particles[n], 1]),
                                s=400,
                                color=cmap.color(n),
                            )
                        else:
                            plt.scatter(
                                to_numpy(x[index_particles[n], 1]),
                                to_numpy(x[index_particles[n], 2]),
                                s=40,
                                color=cmap.color(n),
                            )
                    dead_cell = np.argwhere(to_numpy(H1[:, 0]) == 0)
                    if len(dead_cell) > 0:
                        if "inv" in style:
                            plt.scatter(
                                to_numpy(X1[dead_cell[:, 0].squeeze(), 0]),
                                1 - to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                                s=2,
                                color="k",
                                alpha=0.5,
                            )
                        else:
                            plt.scatter(
                                to_numpy(X1[dead_cell[:, 0].squeeze(), 0]),
                                to_numpy(X1[dead_cell[:, 0].squeeze(), 1]),
                                s=2,
                                color="k",
                                alpha=0.5,
                            )
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel("x", fontsize=13)
                        plt.ylabel("y", fontsize=16)
                        plt.xticks(fontsize=16.0)
                        plt.yticks(fontsize=16.0)
                        ax.tick_params(axis="both", which="major", pad=15)
                        plt.text(
                            0,
                            1.05,
                            f"frame {it}, {int(n_particles_alive)} alive particles ({int(n_particles_dead)} dead), {edge_index.shape[1]} edges  ",
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                            fontsize=16,
                        )

                    if "cell_id" in style:
                        for i, txt in enumerate(to_numpy(N1.squeeze())):
                            if "inv" in style:
                                plt.text(
                                    to_numpy(X1[i, 0]),
                                    1 - to_numpy(X1[i, 1]),
                                    1 + int(to_numpy(N1[i])),
                                    fontsize=8,
                                )
                            else:
                                plt.text(
                                    to_numpy(X1[i, 0]),
                                    to_numpy(X1[i, 1]),
                                    1 + int(to_numpy(N1[i])),
                                    fontsize=8,
                                )  # (txt, (to_numpy(X1[i, 0]), to_numpy(X1[i, 1]), 0), fontsize=8)
                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    num = f"{it:06}"
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=85.35
                    )
                    plt.close()

                # fig = plt.figure(figsize=(12, 12))
                # ax = fig.add_subplot(2, 2, 1)
                # plt.plot(current_loss)
                # plt.xlabel('N')
                # plt.ylabel('Current_loss')
                # ax = fig.add_subplot(2, 2, 2)
                # plt.plot(x_len_list, edges_len_list)
                # plt.xlabel('Number of particles')
                # plt.ylabel('Number of edges')
                # ax = fig.add_subplot(2, 2, 3)
                # plt.plot(x_len_list)
                # plt.xlabel('Number of particles')
                # plt.xlabel('Frame')
                # ax = fig.add_subplot(2, 2, 4)
                # for n in range(n_particle_types):
                #     pos = torch.argwhere((T1.squeeze() == n) & (H1[:, 0].squeeze() == 1))
                #     # if pos.shape[0] > 1:
                #         # sns.kdeplot(to_numpy(AR1[pos].squeeze()), fill=True, color=cmap.color(n), alpha=0.5)
                #         # plt.hist(to_numpy(AR1[pos].squeeze()), bins=100, alpha=0.5)
                # plt.tight_layout()
                # plt.savefig(f"graphs_data/{dataset_name}/gen_{run}.jpg", dpi=80)
                # plt.close()

                if "voronoi" in style:
                    if dimension == 2:
                        vor, vertices_pos, vertices_per_cell, all_points = get_vertices(
                            points=X1, device=device
                        )

                        fig = plt.figure(figsize=(12, 12))
                        ax = fig.add_subplot(1, 1, 1)
                        plt.xticks([])
                        plt.yticks([])

                        voronoi_plot_2d(
                            vor,
                            ax=ax,
                            show_vertices=False,
                            line_colors="black",
                            line_width=1,
                            line_alpha=0.5,
                            point_size=0,
                        )

                        for n in range(n_particle_types):
                            pos = torch.argwhere(
                                (T1.squeeze() == n) & (H1[:, 0].squeeze() == 1)
                            )
                            if pos.shape[0] > 1:
                                pos = to_numpy(pos[:, 0].squeeze()).astype(int)
                                patches = []
                                for i in pos:
                                    cell = vertices_per_cell[i]
                                    vertices = to_numpy(vertices_pos[cell, :])
                                    patches.append(Polygon(vertices, closed=True))
                                if (
                                    (n == 0)
                                    & (has_cell_death)
                                    & (n_particle_types == 3)
                                ):
                                    pc = PatchCollection(
                                        patches, alpha=0.75, facecolors="k"
                                    )
                                else:
                                    pc = PatchCollection(
                                        patches, alpha=0.75, facecolors=cmap.color(n)
                                    )
                                ax.add_collection(pc)
                            elif pos.shape[0] == 1:
                                try:
                                    cell = vertices_per_cell[pos]
                                    vertices = to_numpy(vertices_pos[cell, :])
                                    patches = Polygon(vertices, closed=True)
                                    pc = PatchCollection(
                                        patches, alpha=0.4, facecolors=cmap.color(n)
                                    )
                                    ax.add_collection(pc)
                                except:
                                    pass

                        if "center" in style:
                            plt.scatter(
                                to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=1, c="k"
                            )
                            plt.scatter(
                                to_numpy(first_X1[:, 0]),
                                to_numpy(first_X1[:, 1]),
                                s=1,
                                c="r",
                            )

                        if "vertices" in style:
                            plt.scatter(
                                to_numpy(vertices_pos[:, 0]),
                                to_numpy(vertices_pos[:, 1]),
                                s=5,
                                color="k",
                            )

                        plt.xlim([-0.05, 1.05])
                        plt.ylim([-0.05, 1.05])

                        if "cell_id" in style:
                            for i, txt in enumerate(to_numpy(N1.squeeze())):
                                if "inv" in style:
                                    plt.text(
                                        to_numpy(X1[i, 0]),
                                        1 - to_numpy(X1[i, 1]),
                                        1 + int(to_numpy(N1[i])),
                                        fontsize=8,
                                    )
                                else:
                                    plt.text(
                                        to_numpy(X1[i, 0]),
                                        to_numpy(X1[i, 1]),
                                        1 + int(to_numpy(N1[i])),
                                        fontsize=8,
                                    )  # (txt, (to_numpy(X1[i, 0]), to_numpy(X1[i, 1]), 0), fontsize=8)

                        plt.tight_layout()
                        num = f"{it:06}"
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Vor_{run}_{num}.tif",
                            dpi=85.35,
                        )
                        plt.close()

                    elif dimension == 3:
                        n_particles = len(X1)
                        print(
                            f"frame {it}, {n_particles} particles, {edge_index.shape[1]} edges"
                        )
                        vor, vertices_pos, vertices_per_cell, all_points = get_vertices(
                            points=X1, device=device
                        )

                        cells = [[] for i in range(n_particles)]
                        for (l, r), vertices in vor.ridge_dict.items():
                            if l < n_particles:
                                cells[l].append(vor.vertices[vertices])
                            elif r < n_particles:
                                cells[r].append(vor.vertices[vertices])

                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection="3d")
                        for n, poly in enumerate(cells):
                            polygon = Poly3DCollection(
                                poly,
                                alpha=0.5,
                                facecolors=cmap.color(to_numpy(T1[n]).astype(int)),
                                linewidths=0.1,
                                edgecolors="black",
                            )
                            ax.add_collection3d(polygon)
                        plt.tight_layout()

                        num = f"{it:06}"
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=85.35,
                        )
                        plt.close()

        # check consistency between man_track and x_list[0]
        # for n in range(man_track.shape[0]):
        #     track_id = man_track[n, 0]
        #     start=-1
        #     end=-1
        #     for i in range(len(x_list)):
        #         if torch.argwhere(x_list[i][:, 0] == track_id-1).shape[0] > 0:
        #             if start ==-1:
        #                 start = i
        #             end = i
        #     if (int(start)!=int(man_track[n, 1])) | ((int(end)!=int(man_track[n, 2])) & (int(end)!=n_frames)):
        #         print(f'pb *cell_id {n}  track_id-1 {int(track_id-1)}    x_list {int(start)} {int(end)}  man_track {int(man_track[n, 1])} {int(man_track[n, 2])}')

        if bSave:
            torch.save(x_list, f"graphs_data/{dataset_name}/x_list_{run}.pt")
            torch.save(y_list, f"graphs_data/{dataset_name}/y_list_{run}.pt")
            torch.save(T1_list, f"graphs_data/{dataset_name}/type_list_{run}.pt")
            np.savez(f"graphs_data/{dataset_name}/edge_p_p_list_{run}", *edge_p_p_list)
            if has_inert_model:
                np.savez(
                    f"graphs_data/{dataset_name}/vertices_pos_list_{run}",
                    *vertices_pos_list,
                )
            torch.save(cycle_length, f"graphs_data/{dataset_name}/cycle_length.pt")
            torch.save(CL1, f"graphs_data/{dataset_name}/cycle_length_distrib.pt")
            torch.save(
                cell_death_rate, f"graphs_data/{dataset_name}/cell_death_rate.pt"
            )
            torch.save(DR1, f"graphs_data/{dataset_name}/cell_death_rate_distrib.pt")
            torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")

            if run == 0:
                man_track = to_numpy(man_track)
                pos = np.argwhere(man_track[:, 2] == -1)
                if len(pos) > 0:
                    man_track[pos, 2] = n_frames
                man_track = np.int16(man_track)
                np.savetxt(
                    f"graphs_data/{dataset_name}/man_track.txt",
                    man_track,
                    fmt="%d",
                    delimiter=" ",
                    newline="\n",
                )

    for handler in logger.handlers[:]:
        handler.close()
        logger.removeHandler(handler)


def data_generate_fly_voltage(
    config,
    visualize=True,
    run_vizualized=0,
    style="color",
    erase=False,
    step=5,
    device=None,
    bSave=True,
):
    if "black" in style:
        plt.style.use("dark_background")

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dataset_name = config.dataset
    n_neurons = simulation_config.n_neurons
    n_neuron_types = simulation_config.n_neuron_types
    n_input_neurons = simulation_config.n_input_neurons
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    ensemble_id = simulation_config.ensemble_id
    model_id = simulation_config.model_id
    measurement_noise_level = training_config.measurement_noise_level
    noise_model_level = training_config.noise_model_level
    run = 0
    n_extra_null_edges = simulation_config.n_extra_null_edges

    os.makedirs("./graphs_data/fly", exist_ok=True)
    folder = f"./graphs_data/{dataset_name}/"
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    # files = glob.glob(f'./graphs_data/{dataset_name}/Fig/*')
    # for f in files:
    #     os.remove(f)

    from datamate import Namespace
    from flyvis.datasets.sintel import AugmentedSintel
    from flyvis import NetworkView, Network
    from flyvis.utils.config_utils import get_default_config, CONFIG_PATH
    from flyvis.utils.hex_utils import get_num_hexals
    from MPM_pytorch.generators.PDE_N9 import (
        PDE_N9,
        get_photoreceptor_positions_from_net,
        group_by_direction_and_function,
    )  # plot_stimulus_hex, plot_stimulus_hex_flyvis_coords, plot_flyvis_stimulus_sequence

    plt.style.use("dark_background")

    extent = 8

    # Initialize input stimulus data
    config = Namespace(
        n_frames=19,
        flip_axes=[0, 1],
        n_rotations=[0, 1, 2, 3, 4, 5],
        temporal_split=True,
        dt=delta_t,
        interpolate=True,
        boxfilter=dict(extent=extent, kernel_size=13),
        vertical_splits=3,
        center_crop_fraction=0.7,
    )

    stimulus_dataset = AugmentedSintel(**config)
    # Initialize a model with a connectome/eye of less extent to save memory
    # Fine with this connectome version, because inputs don't span more than 8 hexals
    config = get_default_config(
        overrides=[], path=f"{CONFIG_PATH}/network/network.yaml"
    )
    config.connectome.extent = extent
    net = Network(**config)
    # Now load pretrained weights
    nnv = NetworkView(f"flow/{ensemble_id}/{model_id}")
    trained_net = nnv.init_network(checkpoint=0)
    net.load_state_dict(trained_net.state_dict())
    torch.set_grad_enabled(False)

    params = net._param_api()
    p = {
        "tau_i": params.nodes.time_const,
        "V_i_rest": params.nodes.bias,
        "w": params.edges.syn_strength * params.edges.syn_count * params.edges.sign,
    }
    edge_index = torch.stack(
        [
            torch.tensor(net.connectome.edges.source_index[:]),
            torch.tensor(net.connectome.edges.target_index[:]),
        ],
        dim=0,
    )
    edge_index = edge_index.to(device)

    if n_extra_null_edges > 0:
        print(f"adding {n_extra_null_edges} extra null edges...")

        # convert existing edges to set for fast lookup
        existing_edges = set(
            zip(edge_index[0].cpu().numpy(), edge_index[1].cpu().numpy())
        )

        # generate random non-existing edges on-the-fly
        import random

        extra_edges = []
        max_attempts = n_extra_null_edges * 10  # avoid infinite loop
        attempts = 0

        while len(extra_edges) < n_extra_null_edges and attempts < max_attempts:
            # randomly sample source and target
            i = random.randint(0, n_neurons - 1)
            j = random.randint(0, n_neurons - 1)

            # check if valid (not self-connection, not existing)
            if i != j and (i, j) not in existing_edges:
                extra_edges.append((i, j))
                existing_edges.add((i, j))  # avoid duplicates

            attempts += 1

        if len(extra_edges) < n_extra_null_edges:
            print(
                f"warning: could only generate {len(extra_edges)} new edges after {attempts} attempts"
            )
            n_extra_null_edges = len(extra_edges)

        if n_extra_null_edges > 0:
            # convert to tensor and add to edge_index
            extra_edge_tensor = torch.tensor(
                extra_edges, dtype=edge_index.dtype, device=device
            ).T
            edge_index = torch.cat([edge_index, extra_edge_tensor], dim=1)

            # add corresponding zero weights
            extra_weights = torch.zeros(
                n_extra_null_edges, dtype=p["w"].dtype, device=device
            )
            p["w"] = torch.cat([p["w"], extra_weights])

            print(f"total edges after adding nulls: {edge_index.shape[1]}")
            print(
                f"original edges: {edge_index.shape[1] - n_extra_null_edges}, extra null edges: {n_extra_null_edges}"
            )

    torch.save(edge_index, f"graphs_data/{dataset_name}/edge_index.pt")

    connectivity = torch.zeros(
        (n_neurons, n_neurons), dtype=torch.float32, device=device
    )
    connectivity[edge_index[1], edge_index[0]] = p["w"]
    mask = (connectivity != 0).float()
    torch.save(mask, f"./graphs_data/{dataset_name}/mask.pt")
    torch.save(connectivity, f"./graphs_data/{dataset_name}/connectivity.pt")
    torch.save(p["w"], f"./graphs_data/{dataset_name}/weights.pt")

    # plt.figure(figsize=(10, 10))
    # ax = sns.heatmap(to_numpy(connectivity), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046},
    #                  vmin=-0.2, vmax=0.2)
    # cbar = ax.collections[0].colorbar
    # cbar.ax.tick_params(labelsize=32)
    # plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
    # plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=48)
    # plt.xticks(rotation=0)
    # plt.tight_layout()
    # plt.savefig(f'graphs_data/{dataset_name}/connectivity.png', dpi=300)
    # plt.close()

    pde = PDE_N9(p=p, f=torch.nn.functional.relu, device=device)

    x_coords, y_coords, u_coords, v_coords = get_photoreceptor_positions_from_net(net)

    # Create neuron type mapping
    node_types = np.array(net.connectome.nodes["type"])
    node_types_str = [
        t.decode("utf-8") if isinstance(t, bytes) else str(t) for t in node_types
    ]
    grouped_types = np.array(
        [group_by_direction_and_function(t) for t in node_types_str]
    )
    group_names = [
        "R1-R6",
        "R7-R8",
        "L1-L5",
        "Lamina_Inter",
        "Mi_Early",
        "Mi_Mid",
        "Mi_Late",
        "Tm_Early",
        "Tm5_Family",
        "Tm_Mid",
        "Tm_Late",
        "TmY",
        "T4a_Up",
        "T4b_Right",
        "T4c_Down",
        "T4d_Left",
        "T5_OFF",
        "Tangential",
        "Wide_Field",
        "Other",
    ]
    group_mapping = {i: name for i, name in enumerate(group_names)}
    with open(f"./graphs_data/{dataset_name}/neuron_group_mapping.json", "w") as f:
        json.dump(group_mapping, f, indent=2)

    unique_types, node_types_int = np.unique(node_types, return_inverse=True)

    print(f"number of unique types: {len(unique_types)}")  # Should be 64
    print(f"node_types_int shape: {node_types_int.shape}")
    print(f"node_types_int range: {node_types_int.min()} to {node_types_int.max()}")

    X1 = torch.tensor(
        np.stack((x_coords, y_coords), axis=1), dtype=torch.float32, device=device
    )

    # initialize random positions
    xc, yc = get_equidistant_points(n_points=n_neurons - x_coords.shape[0])
    pos = (
        torch.tensor(np.stack((xc, yc), axis=1), dtype=torch.float32, device=device) / 2
    )
    X1 = torch.cat((X1, pos[torch.randperm(pos.size(0))]), dim=0)

    state = net.steady_state(t_pre=2.0, dt=delta_t, batch_size=1)
    initial_state = state.nodes.activity.squeeze()
    n_neurons = len(initial_state)
    n_edges = len(edge_index[0])
    x = torch.zeros(n_neurons, 7)
    x[:, 1:3] = X1
    x[:, 0] = torch.arange(n_neurons, dtype=torch.float32)
    x[:, 3] = initial_state
    # frame = torch.randn(1, 1, 1, get_num_hexals(config.connectome.extent))
    # print(frame.shape)
    # (n_frames, 1, n_receptors)
    sequences = stimulus_dataset[0]["lum"]
    frame = sequences[0][None, None]
    net.stimulus.add_input(frame)
    x[:, 4] = net.stimulus().squeeze()
    x[:, 5] = torch.tensor(grouped_types, dtype=torch.float32, device=device)
    x[:, 6] = torch.tensor(node_types_int, dtype=torch.float32, device=device)

    dataset = pyg.data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

    y_list = []
    x_list = []
    it = 0
    with torch.no_grad():
        for data in tqdm(stimulus_dataset):
            x[:, 3] = initial_state
            sequences = data["lum"]
            for frame_id in range(sequences.shape[0]):
                frame = sequences[frame_id][None, None]
                net.stimulus.add_input(frame)  # (1, 1, n_input_neurons)
                x[:, 4] = net.stimulus().squeeze()
                dataset = pyg.data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)
                y = pde(dataset, has_field=False)
                y_list.append(to_numpy(y.clone().detach()))
                x_list.append(to_numpy(x.clone().detach()))
                if noise_model_level > 0:
                    x[:, 3:4] = (
                        x[:, 3:4]
                        + delta_t * y
                        + torch.randn(
                            (n_neurons, 1), dtype=torch.float32, device=device
                        )
                        * noise_model_level
                    )
                else:
                    x[:, 3:4] = x[:, 3:4] + delta_t * y
                if (
                    visualize
                    & (run == run_vizualized)
                    & (it % step == 0)
                    & (it <= 500 * step)
                ):
                    if "latex" in style:
                        plt.rcParams["text.usetex"] = True
                        rc("font", **{"family": "serif", "serif": ["Palatino"]})

                    matplotlib.rcParams["savefig.pad_inches"] = 0
                    num = f"{it:06}"

                    plt.figure(figsize=(10, 10))
                    plt.axis("off")
                    plt.scatter(
                        to_numpy(X1[n_input_neurons:, 0]),
                        to_numpy(X1[n_input_neurons:, 1]),
                        s=8,
                        c=to_numpy(x[n_input_neurons:, 3]),
                        cmap="viridis",
                        vmin=-2,
                        vmax=2,
                    )
                    # cbar = plt.colorbar()
                    # cbar.ax.yaxis.set_tick_params(labelsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    ax = plt.subplot(771, frame_on=True)
                    plt.axis("off")
                    plt.scatter(
                        to_numpy(X1[:n_input_neurons, 0]),
                        to_numpy(X1[:n_input_neurons, 1]),
                        s=32,
                        c=to_numpy(x[:n_input_neurons, 4]),
                        cmap="viridis",
                        vmin=0,
                        vmax=1.5,
                    )
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80
                    )
                    plt.close()
                it = it + 1

        print(f"generated {len(x_list)} frames")

    x_list = np.array(x_list)
    y_list = np.array(y_list)

    if measurement_noise_level > 0:
        np.save(f"graphs_data/{dataset_name}/raw_x_list_{run}.npy", x_list)
        np.save(f"graphs_data/{dataset_name}/raw_y_list_{run}.npy", y_list)
        for k in range(x_list.shape[0]):
            x_list[k, :, 3] = x_list[k, :, 3] + np.random.normal(
                0, measurement_noise_level, x_list.shape[1]
            )
        for k in range(1, x_list.shape[0] - 1):
            y_list[k] = (x_list[k + 1, :, 3:4] - x_list[k, :, 3:4]) / delta_t
        np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
        np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
        print("data + noise saved ...")
    else:
        np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
        np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
        print("data saved ...")

    if False:
        activity = torch.tensor(x_list[:, :, 3:4], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(
            to_numpy(activity), center=0, cmap="viridis", cbar_kws={"fraction": 0.046}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        ax.invert_yaxis()
        plt.ylabel("neurons", fontsize=64)
        plt.xlabel("time", fontsize=64)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/kinograph.png", dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        if n_particles > 2:
            n = np.random.permutation(n_particles)
            NN = 25
        else:
            n = np.arange(n_particles)
            NN = 2
        for i in range(NN):
            plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        plt.xlabel("time", fontsize=64)
        plt.ylabel("$x_{i}$", fontsize=64)
        # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.xticks(fontsize=48)
        plt.yticks(fontsize=48)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/activity.png", dpi=300)
        plt.xlim([0, 1000])
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/activity_1000.png", dpi=300)
        plt.close()


def data_generate_synaptic(
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
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    n_neuron_types = simulation_config.n_neuron_types
    n_neurons = simulation_config.n_neurons

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    dataset_name = config.dataset
    excitation = simulation_config.excitation
    noise_model_level = training_config.noise_model_level
    measurement_noise_level = training_config.measurement_noise_level
    cmap = CustomColorMap(config=config)

    field_type = model_config.field_type
    if field_type != "":
        n_nodes = simulation_config.n_nodes
        n_nodes_per_axis = int(np.sqrt(n_nodes))

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(training_config.seed)

    if config.data_folder_name != "none":
        print(f"generating from data ...")
        generate_from_data(
            config=config, device=device, visualize=visualize, folder=folder, step=step
        )
        return

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (not ("X1.pt" in f))
                & (not ("Signal" in f))
                & (not ("Viz" in f))
                & (not ("Exc" in f))
                & (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Viz/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Viz/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Exc/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Exc/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Signal/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Signal/*")
    for f in files:
        os.remove(f)

    particle_dropout_mask = np.arange(n_neurons)
    if has_particle_dropout:
        draw = np.random.permutation(np.arange(n_neurons))
        cut = int(n_neurons * (1 - training_config.particle_dropout))
        particle_dropout_mask = draw[0:cut]
        inv_particle_dropout_mask = draw[cut:]
        x_removed_list = []

    if (
        ("modulation" in model_config.field_type)
        | ("visual" in model_config.field_type)
    ) & ("PDE_N6" not in model_config.signal_model_name):
        im = imread(f"graphs_data/{simulation_config.node_value_map}")

    if "permutation" in field_type:
        permutation_indices = torch.randperm(n_neurons)
        inverse_permutation_indices = torch.argsort(permutation_indices)
        torch.save(
            permutation_indices, f"./graphs_data/{dataset_name}/permutation_indices.pt"
        )
        torch.save(
            inverse_permutation_indices,
            f"./graphs_data/{dataset_name}/inverse_permutation_indices.pt",
        )
    if "excitation_single" in field_type:
        parts = field_type.split("_")
        period = int(parts[-2])
        amplitude = float(parts[-1])

    if "black" in style:
        plt.style.use("dark_background")

    for run in range(config.training.n_runs):
        X = torch.zeros((n_neurons, n_frames + 1), device=device)

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1, V1, T1, H1, A1, N1 = init_neurons(
            config=config, scenario=scenario, ratio=ratio, device=device
        )

        A1 = torch.ones((n_neurons, 1), dtype=torch.float32, device=device)
        U1 = torch.rand_like(H1, device=device)
        U1[:, 1] = 0

        if simulation_config.shuffle_particle_types:
            if run == 0:
                index = torch.randperm(n_neurons)
                T1 = T1[index]
                first_T1 = T1.clone().detach()
            else:
                T1 = first_T1.clone().detach()

        if run == 0:
            edge_index, connectivity, mask = init_connectivity(
                simulation_config.connectivity_file,
                simulation_config.connectivity_distribution,
                simulation_config.connectivity_filling_factor,
                T1,
                n_neurons,
                n_neuron_types,
                dataset_name,
                device,
            )

            model, bc_pos, bc_dpos = choose_model(
                config=config, W=connectivity, device=device
            )

            torch.save(edge_index, f"./graphs_data/{dataset_name}/edge_index.pt")
            torch.save(mask, f"./graphs_data/{dataset_name}/mask.pt")
            torch.save(connectivity, f"./graphs_data/{dataset_name}/connectivity.pt")

        if run == run_vizualized:
            if "black" in style:
                plt.style.use("dark_background")
            plt.figure(figsize=(10, 10))
            for n in range(n_neuron_types):
                pos = torch.argwhere(T1.squeeze() == n)
                plt.scatter(
                    to_numpy(X1[pos, 0]),
                    to_numpy(X1[pos, 1]),
                    s=100,
                    color=cmap.color(n),
                )
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/type_distribution.tif", dpi=130)
            plt.close()

        if "modulation" in field_type:
            if run == 0:
                X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = (
                    init_mesh(config, device=device)
                )
                X1 = X1_mesh

        elif "visual" in field_type:
            if run == 0:
                X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = (
                    init_mesh(config, device=device)
                )
                x, y = get_equidistant_points(n_points=1024)
                X1 = (
                    torch.tensor(
                        np.stack((x, y), axis=1), dtype=torch.float32, device=device
                    )
                    / 2
                )
                X1[:, 1] = X1[:, 1] + 1.5
                X1[:, 0] = X1[:, 0] + 0.5
                X1 = torch.cat((X1_mesh, X1[0 : n_neurons - n_nodes]), 0)

        x = torch.concatenate(
            (
                N1.clone().detach(),
                X1.clone().detach(),
                V1.clone().detach(),
                T1.clone().detach(),
                H1.clone().detach(),
                A1.clone().detach(),
            ),
            1,
        )
        check_and_clear_memory(
            device=device,
            iteration_number=0,
            every_n_iterations=1,
            memory_percentage_threshold=0.6,
        )

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames + 1):
            # calculate type change
            with torch.no_grad():
                if simulation_config.state_type == "sequence":
                    sample = torch.rand((len(T1), 1), device=device)
                    sample = (
                        sample < (1 / config.simulation.state_params[0])
                    ) * torch.randint(0, n_neuron_types, (len(T1), 1), device=device)
                    T1 = (T1 + sample) % n_neuron_types
                if ("modulation" in field_type) & (it >= 0):
                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    if "permutation" in model_config.field_type:
                        im_ = im_[permutation_indices]
                    A1[:, 0:1] = torch.tensor(
                        im_[:, None], dtype=torch.float32, device=device
                    )
                if ("visual" in field_type) & (it >= 0):
                    im_ = im[int(it / n_frames * 256)].squeeze()
                    im_ = np.rot90(im_, 3)
                    im_ = np.reshape(im_, (n_nodes_per_axis * n_nodes_per_axis))
                    A1[:n_nodes, 0:1] = torch.tensor(
                        im_[:, None], dtype=torch.float32, device=device
                    )
                    A1[n_nodes:n_neurons, 0:1] = 1

                    # plt.scatter(to_numpy(X1_mesh[:, 1]), to_numpy(X1_mesh[:, 0]), s=40, c=to_numpy(A1), cmap='grey', vmin=0,vmax=1)
                if "excitation_single" in field_type:
                    parts = field_type.split("_")
                    period = int(parts[-2])
                    amplitude = float(parts[-1])
                    if (it - 100) % period == 0:
                        H1[0, 0] = H1[0, 0] + torch.tensor(
                            amplitude, dtype=torch.float32, device=device
                        )

                x = torch.concatenate(
                    (
                        N1.clone().detach(),
                        X1.clone().detach(),
                        V1.clone().detach(),
                        T1.clone().detach(),
                        H1.clone().detach(),
                        A1.clone().detach(),
                        U1.clone().detach(),
                    ),
                    1,
                )
                X[:, it] = H1[:, 0].clone().detach()
                dataset = data.Data(x=x, pos=x[:, 1:3], edge_index=edge_index)

                # model prediction
                if ("modulation" in field_type) & (it >= 0):
                    y = model(dataset, has_field=True)
                elif ("visual" in field_type) & (it >= 0):
                    y = model(dataset, has_field=True)
                elif "PDE_N3" in model_config.signal_model_name:
                    y = model(dataset, has_field=False, alpha=it / n_frames)
                elif "PDE_N6" in model_config.signal_model_name:
                    (
                        y,
                        p,
                    ) = model(dataset, has_field=False)
                elif "PDE_N7" in model_config.signal_model_name:
                    (
                        y,
                        p,
                    ) = model(dataset, has_field=False)
                else:
                    y = model(dataset, has_field=False)

            # append list
            if (it >= 0) & bSave:
                x_list.append(to_numpy(x))
                y_list.append(to_numpy(y))

            # Particle update
            if (config.graph_model.signal_model_name == "PDE_N6") | (
                config.graph_model.signal_model_name == "PDE_N7"
            ):
                H1[:, 1] = y.squeeze()
                H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t
                if noise_model_level > 0:
                    H1[:, 0] = (
                        H1[:, 0]
                        + torch.randn(n_neurons, device=device) * noise_model_level
                    )
                H1[:, 3] = p.squeeze()
                H1[:, 2] = torch.relu(H1[:, 2] + H1[:, 3] * delta_t)

            else:
                H1[:, 1] = y.squeeze()
                H1[:, 0] = H1[:, 0] + H1[:, 1] * delta_t
                if noise_model_level > 0:
                    H1[:, 0] = (
                        H1[:, 0]
                        + torch.randn(n_neurons, device=device) * noise_model_level
                    )

            # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            # output plots
            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                matplotlib.rcParams["savefig.pad_inches"] = 0
                num = f"{it:06}"

                if "visual" in field_type:
                    fig = plt.figure(figsize=(8, 8))
                    plt.axis("off")
                    plt.subplot(211)
                    plt.axis("off")
                    plt.title("neuromodulation $b_i$", fontsize=24)
                    plt.scatter(
                        to_numpy(X1[0:1024, 1]) * 0.95,
                        to_numpy(X1[0:1024, 0]) * 0.95,
                        s=15,
                        c=to_numpy(A1[0:1024, 0]),
                        cmap="viridis",
                        vmin=0,
                        vmax=2,
                    )
                    plt.scatter(
                        to_numpy(X1[1024:, 1]) * 0.95 + 0.2,
                        to_numpy(X1[1024:, 0]) * 0.95,
                        s=15,
                        c=to_numpy(A1[1024:, 0]),
                        cmap="viridis",
                        vmin=0,
                        vmax=2,
                    )
                    # cbar = plt.colorbar()
                    # cbar.ax.yaxis.set_tick_params(labelsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    plt.subplot(212)
                    plt.axis("off")
                    plt.title("$x_i$", fontsize=24)
                    plt.scatter(
                        to_numpy(X1[0:1024, 1]),
                        to_numpy(X1[0:1024, 0]),
                        s=15,
                        c=to_numpy(H1[0:1024, 0]),
                        cmap="viridis",
                        vmin=-10,
                        vmax=10,
                    )
                    plt.scatter(
                        to_numpy(X1[1024:, 1]) + 0.2,
                        to_numpy(X1[1024:, 0]),
                        s=15,
                        c=to_numpy(H1[1024:, 0]),
                        cmap="viridis",
                        vmin=-10,
                        vmax=10,
                    )
                    # cbar = plt.colorbar()
                    # cbar.ax.yaxis.set_tick_params(labelsize=8)
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=170
                    )
                    plt.close()

                elif "modulation" in field_type:
                    fig = plt.figure(figsize=(12, 12))
                    plt.subplot(221)
                    plt.scatter(
                        to_numpy(X1[:, 1]),
                        to_numpy(X1[:, 0]),
                        s=100,
                        c=to_numpy(A1[:, 0]),
                        cmap="viridis",
                        vmin=0,
                        vmax=2,
                    )
                    plt.subplot(222)
                    plt.scatter(
                        to_numpy(X1[:, 1]),
                        to_numpy(X1[:, 0]),
                        s=100,
                        c=to_numpy(H1[:, 0]),
                        cmap="viridis",
                        vmin=-5,
                        vmax=5,
                    )
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=170
                    )
                    plt.close()

                else:
                    if ("PDE_N6" in model_config.signal_model_name) | (
                        "PDE_N7" in model_config.signal_model_name
                    ):
                        plt.figure(figsize=(12, 5.6))
                        plt.axis("off")
                        plt.axis("off")
                        plt.subplot(121)
                        plt.title("activity $x_i$", fontsize=24)
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=200,
                            c=to_numpy(x[:, 6]),
                            cmap="viridis",
                            vmin=-5,
                            vmax=5,
                            edgecolors="k",
                            alpha=1,
                        )
                        cbar = plt.colorbar()
                        cbar.ax.yaxis.set_tick_params(labelsize=12)
                        plt.xticks([])
                        plt.yticks([])
                        plt.subplot(122)
                        plt.title("short term plasticity $y_i$", fontsize=24)
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=200,
                            c=to_numpy(x[:, 8]),
                            cmap="grey",
                            vmin=0,
                            vmax=1,
                            edgecolors="k",
                            alpha=1,
                        )
                        cbar = plt.colorbar()
                        cbar.ax.yaxis.set_tick_params(labelsize=12)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=170,
                        )
                        plt.close()

                    else:
                        plt.figure(figsize=(10, 10))
                        # plt.scatter(to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=10, c=to_numpy(x[:, 6]),
                        #             cmap='viridis', vmin=-10, vmax=10, edgecolors='k', alpha=1)
                        plt.axis("off")
                        plt.scatter(
                            to_numpy(X1[:, 0]),
                            to_numpy(X1[:, 1]),
                            s=100,
                            c=to_numpy(x[:, 6]),
                            cmap="viridis",
                            vmin=-40,
                            vmax=40,
                        )
                        # cbar = plt.colorbar()
                        # cbar.ax.yaxis.set_tick_params(labelsize=8)
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=170,
                        )
                        plt.close()

                        im_ = imread(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif"
                        )
                        plt.figure(figsize=(10, 10))
                        plt.imshow(im_)
                        plt.xticks([])
                        plt.yticks([])
                        plt.subplot(3, 3, 1)
                        plt.imshow(im_[800:1000, 800:1000, :])
                        plt.xticks([])
                        plt.yticks([])
                        plt.tight_layout()
                        plt.savefig(
                            f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif",
                            dpi=80,
                        )
                        plt.close()

        if bSave:
            x_list = np.array(x_list)
            y_list = np.array(y_list)
            # torch.save(x_list, f'graphs_data/{dataset_name}/x_list_{run}.pt')
            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )
            # torch.save(y_list, f'graphs_data/{dataset_name}/y_list_{run}.pt')
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)
            torch.save(model.p, f"graphs_data/{dataset_name}/model_p.pt")

        if measurement_noise_level > 0:
            np.save(f"graphs_data/{dataset_name}/raw_x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/raw_y_list_{run}.npy", y_list)

            for k in range(x_list.shape[0]):
                x_list[k, :, 6] = x_list[k, :, 6] + np.random.normal(
                    0, measurement_noise_level, x_list.shape[1]
                )
            for k in range(1, x_list.shape[0] - 1):
                y_list[k] = (x_list[k + 1, :, 6:7] - x_list[k, :, 6:7]) / delta_t

            np.save(f"graphs_data/{dataset_name}/x_list_{run}.npy", x_list)
            np.save(f"graphs_data/{dataset_name}/y_list_{run}.npy", y_list)

        activity = torch.tensor(x_list[:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.figure(figsize=(15, 10))
        ax = sns.heatmap(
            to_numpy(activity), center=0, cmap="viridis", cbar_kws={"fraction": 0.046}
        )
        cbar = ax.collections[0].colorbar
        cbar.ax.tick_params(labelsize=32)
        ax.invert_yaxis()
        plt.ylabel("neurons", fontsize=64)
        plt.xlabel("time", fontsize=64)
        plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.yticks([0, 999], [1, 1000], fontsize=48)
        plt.xticks(rotation=0)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/kinograph.png", dpi=300)
        plt.close()

        plt.figure(figsize=(15, 10))
        if n_neurons > 2:
            n = np.random.permutation(n_neurons)
            NN = 25
        else:
            n = np.arange(n_neurons)
            NN = 2
        for i in range(NN):
            plt.plot(to_numpy(activity[n[i].astype(int), :]), linewidth=2)
        plt.xlabel("time", fontsize=64)
        plt.ylabel("$x_{i}$", fontsize=64)
        # plt.xticks([10000, 99000], [10000, 100000], fontsize=48)
        plt.xticks(fontsize=48)
        plt.yticks(fontsize=48)
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/activity.png", dpi=300)
        plt.xlim([0, 1000])
        plt.tight_layout()
        plt.savefig(f"graphs_data/{dataset_name}/activity_1000.png", dpi=300)
        plt.close()

        # torch.cuda.memory_allocated(device)
        # gc.collect()
        # torch.cuda.empty_cache()
        # print(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        # print(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")


def data_generate_rat_city(
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
    # sudo mkdir /nearline/
    # sudo mount -o rw,hard,bg,nolock,nfsvers=4.1,sec=krb5 nearline4.hhmi.org:/nearline/ /nearline/

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    data_folder_name = config.data_folder_name

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dimension = simulation_config.dimension
    min_radius = simulation_config.min_radius
    max_radius = simulation_config.max_radius

    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    time_step = training_config.time_step
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    dataset_name = config.dataset
    run = 0

    pic_folder = config.plotting.pic_folder
    pic_format = config.plotting.pic_format
    pic_size = config.plotting.pic_size

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(training_config.seed)

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)

    print(f"Loading data ...")

    if "txt" in data_folder_name:
        dataframe = pd.read_csv(data_folder_name, sep=" ")
        position_columns = [
            "id",
            "frame_id",
            "x",
            "y",
            "previous_id",
        ]  # replace with your position columns
        values = dataframe[position_columns].to_numpy()
    elif "csv" in data_folder_name:
        dataframe = pd.read_csv(data_folder_name, sep=" ", header=None)
        values = dataframe.to_numpy()
        values[:, [2, 3]] = values[:, [3, 2]]
        values[:, 1] = values[:, 1] - 1

    if training_config.tracking_gt_file != "":
        dataframe = pd.read_csv(training_config.tracking_gt_file, sep=" ", header=None)
        motile = dataframe.to_numpy()
        motile[:, 1] = motile[:, 1] - 1
        motile[:, 3] = motile[:, 3] - 1
        motile = torch.tensor(motile, dtype=torch.float32, device=device)

    if os.path.exists(pic_folder):
        files = glob.glob(f"{pic_folder}/*.jpg")
        sorted_files = sorted(files)

    x_list = []
    edge_f_p_list = []
    edge_p_p_list = []
    t_x = []

    for it in trange(n_frames):
        if it % 1000 == 0:
            print(f"frame {it} ...")

        pos = np.argwhere(values[:, 1] == it)
        if len(pos) == 0:
            print(f"pb with frame{it}")
        else:
            pos = pos.squeeze()

            ID1 = torch.tensor(values[pos, 0:1], dtype=torch.float32, device=device) - 1
            TRUE_ID = (
                torch.tensor(values[pos, 4:5], dtype=torch.float32, device=device) - 1
            )
            N1 = torch.arange(len(pos), dtype=torch.float32, device=device)[:, None]
            X1 = torch.tensor(values[pos, 2:4], dtype=torch.float32, device=device)
            X1[:, 1] = 1 - X1[:, 1]
            V1 = torch.zeros_like(X1)
            T1 = torch.zeros_like(ID1)
            H1 = torch.zeros_like(X1)

            if it > 0:
                TRUE_ID1 = torch.zeros_like(ID1)
                if training_config.tracking_gt_file != "":
                    # ground truth tracking file inputs
                    pos = torch.argwhere(motile[:, 1] == it - 1)
                    if len(pos) > 0:
                        pos = pos.squeeze()
                        motile_pos = motile[pos, :]
                        for id, m in enumerate(ID1):
                            pos = torch.argwhere(motile_pos[:, 2] == m)
                            if len(pos) > 0:
                                pos = pos.squeeze()
                                TRUE_ID1[id] = motile_pos[pos, 0]
                else:
                    # get id from prev_id column
                    ID1_ = x_list[-1][:, 0]
                    TRUE_ID1_ = x_list[-1][:, -1]
                    for id, m in enumerate(TRUE_ID):
                        pos = torch.argwhere(ID1_ == m)
                        if len(pos) > 0:
                            pos = pos.squeeze()
                            TRUE_ID1[id] = TRUE_ID1_[pos]
            else:
                TRUE_ID1 = ID1

            x = torch.concatenate(
                (
                    ID1.clone().detach(),
                    X1.clone().detach(),
                    V1.clone().detach(),
                    T1.clone().detach(),
                    H1.clone().detach(),
                    TRUE_ID1.clone().detach(),
                ),
                1,
            )

            if (torch.unique(TRUE_ID1) != torch.arange(len(ID1), device=device)).any():
                raise ValueError("ID1 and TRUE_ID1 are not consistent.")

            # compute connectivity rules
            edge_index = torch.sum(
                (x[:, None, 1 : dimension + 1] - x[None, :, 1 : dimension + 1]) ** 2,
                dim=2,
            )
            edge_index = (
                (edge_index < max_radius**2) & (edge_index > min_radius**2)
            ).float() * 1
            edge_index = edge_index.nonzero().t().contiguous()
            if "rat_city" in dataset_name:
                edge_mask = torch.zeros((edge_index.shape[1]), device=device)
                for k in range(edge_index.shape[1]):
                    x1, y1 = x[to_numpy(edge_index[0, k]), 1:3]
                    x2, y2 = x[to_numpy(edge_index[1, k]), 1:3]
                    # Calculate the slope (m) and intercept (b) of the line
                    if x1 == x2:
                        edge_mask[k] = 1
                    else:
                        m = (y2 - y1) / (x2 - x1)
                        b = y1 - m * x1
                        y_intersection = m * 1.05 + b  # x_vertical = 1
                        if (
                            (y_intersection > 0.7)
                            | ((x1 < 1) & (x2 < 1))
                            | ((x1 > 1) & (x2 > 1))
                        ):
                            edge_mask[k] = 1
                pos = torch.argwhere(edge_mask == 1)
                if pos.numel() == 0:
                    raise ValueError("No edges.")
                edge_index = edge_index[:, pos.squeeze()]
            edge_p_p_list.append(to_numpy(edge_index))

            x_list.append(x)

        # output plots
        if visualize & (run == 0) & (it % step == 0) & (it < 512):
            if "latex" in style:
                plt.rcParams["text.usetex"] = True
                rc("font", **{"family": "serif", "serif": ["Palatino"]})

            if "color" in style:
                matplotlib.rcParams["savefig.pad_inches"] = 0

                # pos = torch.argwhere(edge_index[0, :] == 40000)
                # pos = to_numpy(pos.squeeze())
                # pos = edge_index[1, pos]
                # pos=to_numpy(pos)

                if "rat_city" in dataset_name:
                    plt.style.use("dark_background")

                    fig, ax = plt.subplots(figsize=(16, 8))
                    if os.path.exists(pic_folder):
                        im = imageio.imread(sorted_files[it])
                        im = np.flipud(im)
                        plt.imshow(im)
                        plt.axis("off")
                    pos = x[:, 1:3].clone().detach()
                    pos[:, 0] = 1110 * pos[:, 0]
                    pos[:, 1] = 1000 * pos[:, 1]
                    dataset = data.Data(x=x, pos=pos, edge_index=edge_index)
                    vis = to_networkx(
                        dataset, remove_self_loops=True, to_undirected=True
                    )
                    nx.draw_networkx(
                        vis,
                        pos=to_numpy(pos),
                        node_size=0,
                        linewidths=0,
                        with_labels=False,
                        ax=ax,
                        edge_color="g",
                        width=1,
                    )
                    for n in range(len(pos)):
                        plt.text(
                            to_numpy(pos[n, 0]),
                            to_numpy(pos[n, 1]),
                            f"{to_numpy(x[n, -1]):0.0f}",
                            c="w",
                            fontsize=18,
                        )
                    plt.tight_layout()
                    plt.xlim([0, 2300])
                    plt.ylim([0, 1000])

                else:
                    plt.style.use("dark_background")

                    fig, ax = plt.subplots(figsize=(8, 4))
                    plt.scatter(
                        to_numpy(X1[:, 0]), to_numpy(X1[:, 1]), s=100, c="w", alpha=0.5
                    )
                    pos = x[:, 1:3]
                    dataset = data.Data(x=x, pos=pos, edge_index=edge_index)
                    vis = to_networkx(
                        dataset, remove_self_loops=True, to_undirected=True
                    )
                    nx.draw_networkx(
                        vis,
                        pos=to_numpy(pos),
                        node_size=0,
                        linewidths=0,
                        with_labels=False,
                        ax=ax,
                        edge_color="g",
                        width=1,
                    )
                    # pos_connect = to_numpy(edge_index[1,:]).astype(int)
                    # plt.scatter(to_numpy(X1_mesh[pos_connect, 0]), to_numpy(X1_mesh[pos_connect, 1]), s=100, c='r',alpha=0.1)
                    plt.xticks([])
                    plt.yticks([])

                    plt.xlim([0, 1])
                    plt.ylim([0, 1])
                    plt.tight_layout()

                num = f"{it * time_step:06}"
                plt.savefig(
                    f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80
                )
                plt.close()

    torch.save(x_list, f"graphs_data/{dataset_name}/x_list_{run}.pt")
    np.savez(f"graphs_data/{dataset_name}/edge_p_p_list_{run}", *edge_p_p_list)


def data_generate_WBI(
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
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(42)

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    dimension = simulation_config.dimension
    min_radius = simulation_config.min_radius
    max_radius = simulation_config.max_radius

    n_particles = simulation_config.n_particles
    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    has_particle_dropout = training_config.particle_dropout > 0
    dataset_name = config.dataset

    torch.random.fork_rng(devices=device)
    torch.random.manual_seed(training_config.seed)

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-3:] != "Fig")
                & (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Viz/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Viz/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Exc/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Exc/*")
    for f in files:
        os.remove(f)
    os.makedirs(f"./graphs_data/{dataset_name}/Signal/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Signal/*")
    for f in files:
        os.remove(f)

    print(f"Loading data ...")
    filename = simulation_config.fluo_path
    dff = pd.read_hdf(filename, key="data")

    if "subdata" in simulation_config.fluo_path:
        X1 = pd.read_hdf(filename, key="coords").values
    if (
        "df_xtn_denoised_labelled_musclebrainreplaced_norm"
        in simulation_config.fluo_path
    ):
        X1 = h5.File("/groups/saalfeld/home/allierc/signaling/WBI/crops.h5", "r")[
            "coords"
        ][:]

    T1 = dff.columns.get_level_values("cluster_id").values
    T1 = T1[:, None]
    dff = dff.ffill().bfill().values

    X1 = X1.T
    X1 = torch.tensor(X1, dtype=torch.float32, device=device)
    torch.save(X1, f"./graphs_data/{dataset_name}/X1.pt")
    T1 = torch.tensor(T1, dtype=torch.float32, device=device)
    torch.save(T1, f"./graphs_data/{dataset_name}/T1.pt")
    print("Data loaded ...")

    if os.path.isfile(f"./graphs_data/{dataset_name}/edge_index.pt"):
        print("Load local connectivity ...")
        edge_index = torch.load(
            f"./graphs_data/{dataset_name}/edge_index.pt", map_location=device
        )
        print("Local connectivity loaded ...")
    else:
        print("Calculate local connectivity ...")

        if config.simulation.connectivity_type == "distance":
            pos = to_numpy(X1)
            distance = np.sum((pos[:, None, :] - pos[None, :, :]) ** 2, axis=2)
            distance = ((distance < max_radius**2) & (distance > min_radius**2)) * 1.0
            edge_index = np.array(distance.nonzero())
            edge_index = torch.tensor(edge_index, dtype=torch.int64, device=device)
            torch.save(edge_index, f"./graphs_data/{dataset_name}/edge_index.pt")

        print("Local connectivity calculated ...")

    # create GNN

    for run in range(config.training.n_runs):
        X = torch.zeros((n_particles, n_frames + 1), device=device)

        x_list = []
        y_list = []

        # initialize particle and graph states
        X1_, V1, T1_, H1, A1, N1 = init_particles(
            config=config, scenario=scenario, ratio=ratio, device=device
        )

        x = torch.concatenate(
            (
                N1.clone().detach(),
                X1.clone().detach(),
                V1.clone().detach(),
                T1.clone().detach(),
                H1.clone().detach(),
                A1.clone().detach(),
            ),
            1,
        )

        time.sleep(0.5)
        for it in trange(simulation_config.start_frame, n_frames - 1):
            dff_ = dff[it, :]
            H1[:, 0:1] = torch.tensor(dff_[:, None], dtype=torch.float32, device=device)
            H1[:, 1] = 0

            dff_ = dff[it + 1, :]
            y = torch.tensor(dff_[:, None], dtype=torch.float32, device=device)

            x = torch.concatenate(
                (
                    N1.clone().detach(),
                    X1.clone().detach(),
                    V1.clone().detach(),
                    T1.clone().detach(),
                    H1.clone().detach(),
                    A1.clone().detach(),
                ),
                1,
            )

            X[:, it] = H1[:, 0].clone().detach()

            # append list
            if (it >= 0) & bSave:
                x_list.append(x.clone().detach())
                y_list.append(y.clone().detach())

            # output plots
            if visualize & (run == 0) & (it % step == 0) & (it >= 0):
                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                if "color" in style:
                    matplotlib.rcParams["savefig.pad_inches"] = 0

                    # pos = torch.argwhere(edge_index[0, :] == 40000)
                    # pos = to_numpy(pos.squeeze())
                    # pos = edge_index[1, pos]
                    # pos=to_numpy(pos)

                    fig = plt.figure(figsize=(16, 8))
                    plt.scatter(
                        to_numpy(X1[:, 1]),
                        to_numpy(X1[:, 2]),
                        s=20,
                        c=to_numpy(H1[:, 0]),
                        cmap="viridis",
                        vmin=-2.5,
                        vmax=2.5,
                    )
                    plt.colorbar()
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()

                    num = f"{it:06}"
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=70
                    )
                    plt.close()

        if bSave:
            torch.save(x_list, f"graphs_data/{dataset_name}/x_list_{run}.pt")
            if has_particle_dropout:
                torch.save(
                    x_removed_list,
                    f"graphs_data/{dataset_name}/x_removed_list_{run}.pt",
                )
                np.save(
                    f"graphs_data/{dataset_name}/particle_dropout_mask.npy",
                    particle_dropout_mask,
                )
                np.save(
                    f"graphs_data/{dataset_name}/inv_particle_dropout_mask.npy",
                    inv_particle_dropout_mask,
                )
            torch.save(y_list, f"graphs_data/{dataset_name}/y_list_{run}.pt")

        if run == 0:
            fig = plt.figure(figsize=(16, 8))
            plt.scatter(
                to_numpy(X1[:, 1]),
                to_numpy(X1[:, 2]),
                s=10,
                c=to_numpy(T1[:, 0]),
                cmap="tab20",
                vmin=0,
                vmax=255,
            )
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/type.png", dpi=300)
            plt.close()

            plt.figure(figsize=(10, 3))
            plt.subplot(121)
            ax = sns.heatmap(to_numpy(X), center=0, cbar_kws={"fraction": 0.046})
            ax.invert_yaxis()
            plt.title("Firing rate", fontsize=12)
            plt.ylabel("Units", fontsize=12)
            plt.xlabel("Time", fontsize=12)
            plt.xticks([])
            plt.yticks([0, 999], [1, 1000], fontsize=12)

            plt.subplot(122)
            plt.title("Firing rate samples", fontsize=12)
            for i in range(50):
                plt.plot(to_numpy(X[i, :]), linewidth=1)
            plt.xlabel("Time", fontsize=12)
            plt.ylabel("Normalized activity", fontsize=12)
            plt.xticks([])
            plt.yticks(fontsize=12)
            plt.tight_layout()
            plt.savefig(f"graphs_data/{dataset_name}/activity.png", dpi=300)
            plt.close()

    # for handler in logger.handlers[:]:
    #     handler.close()
    #     logger.removeHandler(handler)


def data_generate_mesh(
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
    simulation_config = config.simulation
    training_config = config.training
    model_config = config.graph_model

    print(
        f"generating data ... {model_config.particle_model_name} {model_config.mesh_model_name}"
    )

    delta_t = simulation_config.delta_t
    n_frames = simulation_config.n_frames
    cmap = CustomColorMap(config=config)
    dataset_name = config.dataset

    folder = f"./graphs_data/{dataset_name}/"
    if erase:
        files = glob.glob(f"{folder}/*")
        for f in files:
            if (
                (f[-14:] != "generated_data")
                & (f != "p.pt")
                & (f != "cycle_length.pt")
                & (f != "model_config.json")
                & (f != "generation_code.py")
            ):
                os.remove(f)
    os.makedirs(folder, exist_ok=True)
    os.makedirs(f"./graphs_data/{dataset_name}/Fig/", exist_ok=True)
    files = glob.glob(f"./graphs_data/{dataset_name}/Fig/*")
    for f in files:
        os.remove(f)

    for run in range(config.training.n_runs):
        X1_mesh, V1_mesh, T1_mesh, H1_mesh, A1_mesh, N1_mesh, mesh_data = init_mesh(
            config, device=device
        )

        mesh_model = choose_mesh_model(config=config, X1_mesh=X1_mesh, device=device)

        torch.save(mesh_data, f"graphs_data/{dataset_name}/mesh_data_{run}.pt")
        mask_mesh = mesh_data["mask"].squeeze()

        time.sleep(0.5)
        x_mesh_list = []
        y_mesh_list = []
        for it in trange(simulation_config.start_frame, n_frames + 1):
            x_mesh = torch.concatenate(
                (
                    N1_mesh.clone().detach(),
                    X1_mesh.clone().detach(),
                    V1_mesh.clone().detach(),
                    T1_mesh.clone().detach(),
                    H1_mesh.clone().detach(),
                ),
                1,
            )
            x_mesh_list.append(x_mesh.clone().detach())

            dataset_mesh = data.Data(
                x=x_mesh,
                edge_index=mesh_data["edge_index"],
                edge_attr=mesh_data["edge_weight"],
                device=device,
            )

            match config.graph_model.mesh_model_name:
                case "DiffMesh":
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                        H1[mask_mesh, 1:2] = pred[mask_mesh]
                    H1_mesh[mask_mesh, 0:1] += pred[mask_mesh, 0:1] * delta_t
                    new_pred = torch.zeros_like(pred)
                    new_pred[mask_mesh] = pred[mask_mesh]
                    pred = new_pred
                case "WaveSmoothParticle":
                    pred = mesh_model(dataset_mesh)
                    H1_mesh[mask_mesh, 1:2] += pred[mask_mesh, :] * delta_t
                    H1_mesh[mask_mesh, 0:1] += H1_mesh[mask_mesh, 1:2] * delta_t
                case "WaveMesh":
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                    H1_mesh[mask_mesh, 1:2] += pred[mask_mesh, :] * delta_t
                    H1_mesh[mask_mesh, 0:1] += H1_mesh[mask_mesh, 1:2] * delta_t
                    # x_ = to_numpy(x_mesh)
                    # plt.scatter(x_[:, 1], x_[:, 2], c=to_numpy(H1_mesh[:, 0]))
                case (
                    "RD_Gray_Scott_Mesh"
                    | "RD_FitzHugh_Nagumo_Mesh"
                    | "RD_Mesh"
                    | "RD_Mesh_bis"
                ):
                    with torch.no_grad():
                        pred = mesh_model(dataset_mesh)
                        H1_mesh[mesh_data["mask"].squeeze(), :] += (
                            pred[mesh_data["mask"].squeeze(), :] * delta_t
                        )
                        H1_mesh[mask_mesh.squeeze(), 6:9] = torch.clamp(
                            H1_mesh[mask_mesh.squeeze(), 6:9], 0, 1
                        )
                        H1 = H1_mesh.clone().detach()
                case "PDE_O_Mesh":
                    pred = []

            y_mesh_list.append(pred)

            if visualize & (run == run_vizualized) & (it % step == 0) & (it >= 0):
                # plt.style.use('dark_background')
                # matplotlib.use("Qt5Agg")

                if "latex" in style:
                    plt.rcParams["text.usetex"] = True
                    rc("font", **{"family": "serif", "serif": ["Palatino"]})

                if "graph" in style:
                    fig = plt.figure(figsize=(12, 12))
                    match model_config.mesh_model_name:
                        case "RD_Mesh":
                            H1_IM = torch.reshape(x_mesh[:, 6:9], (100, 100, 3))
                            plt.imshow(to_numpy(H1_IM), vmin=0, vmax=1)
                        case "Wave_Mesh" | "DiffMesh":
                            pts = x_mesh[:, 1:3].detach().cpu().numpy()
                            tri = Delaunay(pts)
                            colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                edgecolors="k",
                                vmin=-2500,
                                vmax=2500,
                            )
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                    plt.xticks([])
                    plt.yticks([])
                    plt.tight_layout()
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_g_color_{it}.tif", dpi=300
                    )
                    plt.close()

                if "color" in style:
                    # matplotlib.use("Qt5Agg")

                    matplotlib.rcParams["savefig.pad_inches"] = 0
                    fig = plt.figure(figsize=(12, 12))
                    ax = fig.add_subplot(1, 1, 1)
                    ax.tick_params(axis="both", which="major", pad=15)
                    ax.xaxis.get_major_formatter()._usetex = False
                    ax.yaxis.get_major_formatter()._usetex = False
                    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                    ax.xaxis.set_major_formatter(FormatStrFormatter("%.1f"))
                    ax.yaxis.set_major_formatter(FormatStrFormatter("%.1f"))

                    pts = x_mesh[:, 1:3].detach().cpu().numpy()
                    tri = Delaunay(pts)
                    colors = torch.sum(x_mesh[tri.simplices, 6], dim=1) / 3.0
                    match model_config.mesh_model_name:
                        case "DiffMesh":
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                vmin=0,
                                vmax=1000,
                            )
                        case "WaveMesh":
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                vmin=-1000,
                                vmax=1000,
                            )
                            fmt = lambda x, pos: "{:.1f}".format((x) / 100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                        case "WaveSmoothParticle":
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                vmin=-1000,
                                vmax=1000,
                            )
                            fmt = lambda x, pos: "{:.1f}".format((x) / 100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            plt.xlim([0, 1])
                            plt.ylim([0, 1])
                        case "RD_Gray_Scott_Mesh":
                            fig = plt.figure(figsize=(12, 6))
                            ax = fig.add_subplot(1, 2, 1)
                            colors = torch.sum(x[tri.simplices, 6], dim=1) / 3.0
                            plt.tripcolor(
                                pts[:, 0],
                                pts[:, 1],
                                tri.simplices.copy(),
                                facecolors=colors.detach().cpu().numpy(),
                                vmin=0,
                                vmax=1,
                            )
                            plt.xticks([])
                            plt.yticks([])
                            plt.axis("off")
                        case "RD_Mesh" | "RD_Mesh_bis":
                            H1_IM = torch.reshape(H1, (100, 100, 3))
                            plt.imshow(H1_IM.detach().cpu().numpy(), vmin=0, vmax=1)
                            fmt = lambda x, pos: "{:.1f}".format((x) / 100, pos)
                            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            ax.xaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
                            # plt.xticks([])
                            # plt.yticks([])
                            # plt.axis('off')`
                    if "latex" in style:
                        plt.xlabel(r"$x$", fontsize=78)
                        plt.ylabel(r"$y$", fontsize=78)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                    elif "frame" in style:
                        plt.xlabel("x", fontsize=48)
                        plt.ylabel("y", fontsize=48)
                        plt.xticks(fontsize=48.0)
                        plt.yticks(fontsize=48.0)
                        ax.tick_params(axis="both", which="major", pad=15)
                        plt.text(
                            0,
                            1.1,
                            f"frame {it}",
                            ha="left",
                            va="top",
                            transform=ax.transAxes,
                            fontsize=48,
                        )
                    else:
                        plt.xticks([])
                        plt.yticks([])

                    plt.tight_layout()

                    num = f"{it:06}"
                    plt.savefig(
                        f"graphs_data/{dataset_name}/Fig/Fig_{run}_{num}.tif", dpi=80
                    )
                    plt.close()

        if bSave:
            torch.save(x_mesh_list, f"graphs_data/{dataset_name}/x_mesh_list_{run}.pt")
            torch.save(y_mesh_list, f"graphs_data/{dataset_name}/y_mesh_list_{run}.pt")


def try_func(max_radius, device):
    r = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
    sigma = 0.005

    fig, ax = fig_init()
    p = torch.tensor([1.6233, 1.0413, 1.6012, 1.5615], device=device)
    out = r * (
        p[0] * torch.exp(-(r ** (2 * p[1])) / (2 * sigma**2))
        - p[2] * torch.exp(-(r ** (2 * p[3])) / (2 * sigma**2))
    )
    plt.plot(to_numpy(r), to_numpy(out), linewidth=2)
    p = torch.tensor([1.7667, 1.8308, 1.0855, 1.9055], device=device)
    out = r * (
        p[0] * torch.exp(-(r ** (2 * p[1])) / (2 * sigma**2))
        - p[2] * torch.exp(-(r ** (2 * p[3])) / (2 * sigma**2))
    )
    plt.plot(to_numpy(r), to_numpy(out), linewidth=2)
    # p = torch.tensor([1.7226, 1.7850, 1.0584, 1.8579], device=device)
    # out = r * (p[0] * torch.exp(-r ** (2 * p[1]) / (2 * sigma ** 2)) - p[2] * torch.exp(-r ** (2 * p[3]) / (2 * sigma ** 2)))
    # plt.plot(to_numpy(r), to_numpy(out), linewidth=2)
    p = torch.tensor([0.03, 0.03, 100, 1.0], device=device)
    out = p[0] * torch.tanh((r - p[1]) * p[2])
    plt.plot(to_numpy(r), to_numpy(out), linewidth=2)
    p = torch.tensor([0.03, 0.05, 100, 1.0], device=device)
    out = p[0] * torch.tanh((r - p[1]) * p[2])
    plt.plot(to_numpy(r), to_numpy(out), linewidth=2)
    plt.tight_layout()
