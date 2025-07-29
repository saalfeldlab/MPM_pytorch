import os
import time

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random

from run_MPM import *
from MPM_pytorch.models.utils import *
from MPM_pytorch.utils import *
from MPM_pytorch.models.Siren_Network import *
from geomloss import SamplesLoss
from sklearn.neighbors import NearestNeighbors
from scipy.optimize import curve_fit

from torch_geometric.utils import dense_to_sparse
import torch.optim as optim
import seaborn as sns
from sklearn.decomposition import PCA, TruncatedSVD
from sklearn.manifold import TSNE

from scipy.spatial import KDTree
from sklearn import neighbors, metrics
from scipy.ndimage import median_filter
from tifffile import imwrite, imread
from matplotlib.colors import LinearSegmentedColormap


def data_train(config=None, erase=False, best_model=None, device=None):
    # plt.rcParams['text.usetex'] = True
    # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    # matplotlib.rcParams['savefig.pad_inches'] = 0

    seed = config.training.seed

    torch.manual_seed(seed)
    np.random.seed(seed)
    random.seed(seed)

    # torch.autograd.set_detect_anomaly(True)

    has_mesh = (config.graph_model.mesh_model_name != '')
    has_signal = (config.graph_model.signal_model_name != '')
    has_particle_field = ('PDE_ParticleField' in config.graph_model.particle_model_name)
    has_material = 'PDE_MPM' in config.graph_model.particle_model_name
    has_cell_division = config.simulation.has_cell_division
    do_tracking = config.training.do_tracking
    has_state = (config.simulation.state_type != 'discrete')
    has_WBI = 'WBI' in config.dataset
    has_mouse_city = ('mouse_city' in config.dataset) | ('rat_city' in config.dataset)
    sub_sampling = config.simulation.sub_sampling
    rotation_augmentation = config.training.rotation_augmentation

    if rotation_augmentation & (sub_sampling > 1):
        assert (False), 'rotation_augmentation does not work with sub_sampling > 1'

    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    if 'Agents' in config.graph_model.particle_model_name:
        data_train_agents(config, erase, best_model, device)
    elif has_mouse_city:
        data_train_rat_city(config, erase, best_model, device)
    elif has_WBI:
        data_train_WBI(config, erase, best_model, device)
    elif has_particle_field:
        data_train_particle_field(config, erase, best_model, device)
    elif has_mesh:
        data_train_mesh(config, erase, best_model, device)
    elif 'fly' in config.dataset:
        data_train_flyvis(config, erase, best_model, device)
    elif has_signal:
        data_train_synaptic2(config, erase, best_model, device)
    elif do_tracking & has_cell_division:
        data_train_cell(config, erase, best_model, device)
    elif has_cell_division:
        data_train_cell(config, erase, best_model, device)
    elif has_state:
        data_train_particle(config, erase, best_model, device)
    elif 'PDE_GS' in config.graph_model.particle_model_name:
        data_solar_system(config, erase, best_model, device)
    elif has_material:
        data_train_material(config, erase, best_model, device)
    else:
        data_train_particle(config, erase, best_model, device)


def data_train_material(config, erase, best_model, device):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting

    trainer = train_config.MPM_trainer

    print(f'training data ... {model_config.particle_model_name} {model_config.mesh_model_name} loss on {trainer}')

    dimension = simulation_config.dimension
    n_epochs = train_config.n_epochs
    max_radius = simulation_config.max_radius
    min_radius = simulation_config.min_radius
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    n_grid = simulation_config.n_grid
    group_size = n_particles // n_particle_types

    delta_t = simulation_config.delta_t
    time_window = train_config.time_window
    time_step = train_config.time_step
    field_type = model_config.field_type

    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    data_augmentation_loop = train_config.data_augmentation_loop
    recursive_loop = train_config.recursive_loop
    target_batch_size = train_config.batch_size
    replace_with_cluster = 'replace' in train_config.sparsity
    sparsity_freq = train_config.sparsity_freq
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_ratio = train_config.batch_ratio
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    embedding_cluster = EmbeddingCluster(config)
    n_runs = train_config.n_runs

    log_dir, logger = create_log_dir(config, erase)
    print(f'graph files N: {n_runs}')
    logger.info(f'graph files N: {n_runs}')
    time.sleep(0.5)
    print('load data ...')
    x_list = []
    y_list = []
    edge_saved = False

    run_lengths = list()
    time.sleep(0.5)
    n_particles_max = 0

    for run in trange(n_runs):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        # y = np.load(f'graphs_data/{dataset_name}/y_list_{run}.npy')
        if np.isnan(x).any():
            print('Pb isnan')
        if x[0].shape[0] > n_particles_max:
            n_particles_max = x[0].shape[0]
        x_list.append(x)
        # y_list.append(y)
        run_lengths.append(len(x))
    x = torch.tensor(x_list[0][0], dtype=torch.float32, device=device)
    # y = torch.tensor(y_list[0][0], dtype=torch.float32, device=device)

    vnorm = torch.tensor(1, device=device)
    ynorm = torch.tensor(1, device=device)
    torch.save(vnorm, os.path.join(log_dir, 'vnorm.pt'))
    torch.save(ynorm, os.path.join(log_dir, 'ynorm.pt'))
    time.sleep(0.5)
    print(f'N particles: {n_particles}')
    print(f'N grid: {n_grid}')
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    x = []
    y = []

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm
    if (best_model != None) & (best_model != ''):
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
        state_dict = torch.load(net, map_location=device)
        model.load_state_dict(state_dict['model_state_dict'])
        start_epoch = int(best_model.split('_')[0])
        print(f'best_model: {best_model}  start_epoch: {start_epoch}')
        logger.info(f'best_model: {best_model}  start_epoch: {start_epoch}')
    else:
        start_epoch = 0
        net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs.pt"

    lr = train_config.learning_rate_start
    lr_embedding = train_config.learning_rate_embedding_start
    optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)

    logger.info(f"total Trainable Params: {n_total_params}")
    logger.info(f'learning rates: {lr}, {lr_embedding}')
    model.train()

    print(f'network: {net}')
    print(f'initial batch_size: {batch_size}')
    logger.info(f'network: {net}')
    logger.info(f'N epochs: {n_epochs}')
    logger.info(f'initial batch_size: {batch_size}')

    print("start training particles ...")
    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(1)

    trainer = 'C'

    # torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, n_epochs + 1):

        logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        batch_size = int(get_batch_size(epoch))
        logger.info(f'batch_size: {batch_size}')

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio)
        else:
            Niter = n_frames * data_augmentation_loop // batch_size
        if epoch == 0:
            plot_frequency = int(Niter // 20)
            print(f'{Niter} iterations per epoch')
            logger.info(f'{Niter} iterations per epoch')
            print(f'plot every {plot_frequency} iterations')

        time.sleep(1)
        total_loss = 0

        for N in trange(Niter):

            # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 500,
            #                        memory_percentage_threshold=0.6)

            dataset_batch = []
            loss = 0
            for batch in range(batch_size):

                run = 0
                k = time_window + np.random.randint(run_lengths[run] - 1 - time_window - time_step - recursive_loop)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                x_next = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                if trainer == 'F':
                    y = x_next[:, 5 + dimension * 2: 9 + dimension * 2].clone().detach() # F
                elif trainer == 'S':
                    y = x_next[:, 12 + dimension * 2: 16 + dimension * 2].clone().detach() # S
                elif trainer == 'C':
                    y = x_next[:, 1 + dimension * 2: 5 + dimension * 2].clone().detach()

                dataset = data.Data(x=x, edge_index=[], num_nodes=x.shape[0])
                dataset_batch.append(dataset)

                if batch == 0:
                    data_id = torch.ones((n_particles,1), dtype=torch.float32, device=device) * run
                    x_batch = x
                    y_batch = y
                    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                else:
                    data_id = torch.cat((data_id, torch.ones((n_particles,1), dtype=torch.float32, device=device) * run), dim=0)
                    x_batch = torch.cat((x_batch, x), dim=0)
                    y_batch = torch.cat((y_batch, y), dim=0)
                    k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k), dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch in batch_loader:
                pred = model(batch, data_id=data_id, k=k_batch, trainer=trainer)

            loss = F.mse_loss(pred, y_batch)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if ((epoch < 30) & (N % plot_frequency == 0)) | (N == 0):

                k_list = [250, 340, 680, 930]
                error = list([])
                with torch.no_grad():
                    for k in k_list:
                        x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                        if trainer == 'F':
                            y = x[:, 5 + dimension * 2: 9 + dimension * 2].clone().detach() # F
                        elif trainer == 'S':
                            y = x[:, 12 + dimension * 2: 16 + dimension * 2].clone().detach()  # S
                        elif trainer == 'C':
                            y = x[:, 1 + dimension * 2: 5 + dimension * 2].clone().detach()
                        data_id = torch.ones((n_particles, 1), dtype=torch.float32, device=device) * run
                        k_list = torch.ones((n_particles, 1), dtype=torch.int, device=device) * k
                        dataset = data.Data(x=x, edge_index=[], num_nodes=x.shape[0])
                        pred = model(dataset, data_id=data_id, k=k_list, trainer=trainer)
                        error.append(F.mse_loss(pred, y).item())

                plt.style.use('dark_background')

                fig = plt.figure(figsize=(18, 8))
                plt.subplot(1, 2, 1)
                if trainer == 'F':
                    f_norm = torch.norm(y.view(n_particles, -1), dim=1).cpu().numpy()
                    plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=f_norm, s=1, cmap='coolwarm', vmin=1.44 - 0.2,
                                vmax=1.44 + 0.2)
                    plt.colorbar(fraction=0.046, pad=0.04)
                elif trainer == 'S':
                    stress_norm = torch.norm(y.view(n_particles, -1), dim=1)
                    stress_norm = stress_norm[:, None]
                    plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=stress_norm[:, 0].cpu(), s=1, cmap='hot', vmin=0, vmax=6E-3)
                    plt.colorbar(fraction=0.046, pad=0.04)
                elif trainer == 'C':
                    c_norm = torch.norm(y.view(n_particles, -1), dim=1)
                    c_norm = c_norm[:, None]
                    plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=c_norm[:, 0].cpu(), s=1, cmap='viridis', vmin=0, vmax=80)
                    plt.colorbar(fraction=0.046, pad=0.04)

                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.subplot(1, 2, 2)
                if trainer == 'F':
                    f_norm = torch.norm(pred.view(n_particles, -1), dim=1).cpu().numpy()
                    plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=f_norm, s=1, cmap='coolwarm', vmin=1.44 - 0.2,
                                vmax=1.44 + 0.2)
                    plt.colorbar(fraction=0.046, pad=0.04)
                elif trainer == 'S':
                    stress_norm = torch.norm(pred.view(n_particles, -1), dim=1)
                    stress_norm = stress_norm[:, None]
                    plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=stress_norm[:, 0].cpu(), s=1, cmap='hot', vmin=0, vmax=6E-3)
                    plt.colorbar(fraction=0.046, pad=0.04)
                elif trainer == 'C':
                    c_norm = torch.norm(pred.view(n_particles, -1), dim=1)
                    c_norm = c_norm[:, None]
                    plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=c_norm[:, 0].cpu(), s=1, cmap='viridis', vmin=0, vmax=80)
                    plt.colorbar(fraction=0.046, pad=0.04)

                plt.text(0.05, 0.95,
                         f'epoch: {epoch} iteration: {N} error: {np.mean(1000 * error) / len(k_list):.6f}', )
                plt.xlim([0, 1])
                plt.ylim([0, 1])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field/{epoch}_{N}.tif", dpi=87)
                plt.close()

                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                           os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))

                check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50,
                                       memory_percentage_threshold=0.6)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_particles))
        logger.info("Epoch {}. Loss: {:.10f}".format(epoch, total_loss / n_particles))
        list_loss.append(total_loss / n_particles)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

        fig = plt.figure(figsize=(22, 5))
        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='w')
        plt.xlim([0, n_epochs])
        plt.ylabel('Loss', fontsize=12)
        plt.xlabel('Epochs', fontsize=12)
        plt.text(0.05, 0.95, f'epoch: {epoch} final loss: {list_loss[-1]:.10f}', transform=ax.transAxes,)
        plt.tight_layout()
        ax = fig.add_subplot(1, 5, 2)
        embedding = to_numpy(model.a[0])
        type_list = to_numpy(x[:,14])
        for n in range(n_particle_types):
            plt.scatter(embedding[type_list == n, 0], embedding[type_list == n, 1], s=1,
                        c=cmap.color(n), label=f'type {n}', alpha=0.5)

        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()


