import os
import time
import glob

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

    sub_sampling = config.simulation.sub_sampling
    rotation_augmentation = config.training.rotation_augmentation

    if rotation_augmentation & (sub_sampling > 1):
        assert (False), 'rotation_augmentation does not work with sub_sampling > 1'

    dataset_name = config.dataset
    print('')
    print(f'dataset_name: {dataset_name}')

    data_train_material(config, erase, best_model, device)


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

    coeff_Jp_norm = train_config.coeff_Jp_norm
    coeff_F_norm = train_config.coeff_F_norm
    coeff_det_F = train_config.coeff_det_F

    log_dir, logger = create_log_dir(config, erase)
    print(f'graph files N: {n_runs}')
    logger.info(f'graph files N: {n_runs}')

    os.makedirs(f"./{log_dir}/tmp_training/movie", exist_ok=True)
    files = os.listdir(f"./{log_dir}/tmp_training/movie")
    for file in files:
        os.remove(f"./{log_dir}/tmp_training/movie/{file}")

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

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, n_epochs + 1):


        logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        batch_size = int(get_batch_size(epoch))
        logger.info(f'batch_size: {batch_size}')

        # if (epoch == 1):
        #     lr_embedding = 1E-4
        #     optimizer, n_total_params = set_trainable_parameters(model, lr_embedding, lr)

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
                x_next = torch.tensor(x_list[run][k+1], dtype=torch.float32, device=device).clone().detach()

                if 'next_S' in trainer:
                    y = x_next[:, 12 + dimension * 2: 16 + dimension * 2].clone().detach()
                elif 'next_C_F_Jp' in trainer:
                    y = x_next[:, 1 + dimension * 2: 10 + dimension * 2].clone().detach()  # C
                elif 'C_F_Jp' in trainer:
                    # For k-nearest, we need positions and velocities for neighbor loss
                    pos = x[:, 1:3].clone().detach()  # positions
                    vel = x[:, 3:5].clone().detach()  # velocities
                    y = None  # No direct target needed

                if 'GNN' in trainer:
                    distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
                    adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
                    edges = adj_t.nonzero().t().contiguous()
                else:
                    edges = []

                dataset = data.Data(x=x, edge_index=edges, num_nodes=x.shape[0])
                dataset_batch.append(dataset)

                if batch == 0:
                    data_id = torch.ones((n_particles, 1), dtype=torch.float32, device=device) * run
                    x_batch = x
                    if trainer == 'C_F_Jp':
                        pos_batch = pos
                        vel_batch = vel
                    else:
                        y_batch = y

                    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k
                else:
                    data_id = torch.cat(
                        (data_id, torch.ones((n_particles, 1), dtype=torch.float32, device=device) * run), dim=0)
                    x_batch = torch.cat((x_batch, x), dim=0)
                    if trainer == 'C_F_Jp':
                        pos_batch = torch.cat((pos_batch, pos), dim=0)
                        vel_batch = torch.cat((vel_batch, vel), dim=0)
                    else:
                        y_batch = torch.cat((y_batch, y), dim=0)

                    k_batch = torch.cat((k_batch, torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k),
                                        dim=0)

            batch_loader = DataLoader(dataset_batch, batch_size=batch_size, shuffle=False)
            optimizer.zero_grad()

            for batch_loader_data in batch_loader:

                pred_C, pred_F, pred_Jp, pred_S = model(batch_loader_data, data_id=data_id, k=k_batch, trainer=trainer)

            if 'next_S' in trainer:
                loss = F.mse_loss(pred_S.reshape(-1, 4), y_batch)
            elif 'next_C_F_Jp' in trainer:
                loss = F.mse_loss(torch.cat((pred_C.reshape(-1,4), pred_F.reshape(-1,4), pred_Jp), dim=1), y_batch)
            elif 'C_F_Jp' in trainer:
                k_neighbors = 5
                pred_C = pred_C.reshape(-1, 2, 2)
                for k in range(batch_size):
                    batch_indices = np.arange(k * n_particles, (k + 1) * n_particles)
                    positions = to_numpy(dataset_batch[k].x[:, 1:3])  # shape: [N, 2]
                    velocities = to_numpy(dataset_batch[k].x[:, 3:5])  # shape: [N, 2]

                    tree = cKDTree(positions)
                    dists, indices = tree.query(positions, k=k_neighbors + 1)  # +1 to skip self
                    neighbor_indices = indices[:, 1:]  # shape: [N, k]

                    # position and velocity diffs
                    pos_diff_np = positions[neighbor_indices] - positions[:, None, :]  # [N, k, 2]
                    vel_diff_np = velocities[neighbor_indices] - velocities[:, None, :]  # [N, k, 2]

                    pos_diff = torch.tensor(pos_diff_np, dtype=torch.float32, device=device)  # [N, k, 2]
                    vel_diff = torch.tensor(vel_diff_np, dtype=torch.float32, device=device)  # [N, k, 2]

                    C = pred_C[batch_indices]  # [N, 2, 2]
                    pred_vel_diff = torch.bmm(pos_diff, C.transpose(1, 2))  # [N, k, 2]
                    loss = loss + F.mse_loss(pred_vel_diff, vel_diff)

            # if (epoch < 1) & (N < Niter // 25) & (trainer != 'C_F_Jp'):
            if coeff_Jp_norm >0 :
                loss = loss + coeff_Jp_norm * F.mse_loss(pred_Jp, torch.ones_like(pred_Jp).detach())
            if coeff_F_norm >0 :
                F_norm = torch.norm(pred_F.view(-1, 4), dim=1)
                loss = loss + coeff_F_norm * F.mse_loss(F_norm, torch.ones_like(F_norm).detach() * 1.4141)
            if coeff_det_F > 0:
                det_F = torch.det(pred_F.view(-1, 2, 2))
                loss = loss + coeff_det_F * F.relu(-det_F + 0.1).mean()

            loss.backward()
            optimizer.step()

            total_loss += loss.item()

            if ((epoch < 30) & (N % plot_frequency == 0)) | (N == 0):

                if ('next_C_F_Jp' in trainer) | ('next_S' in trainer):
                    plot_training_C_F_Jp_S(x_list, run, device, dimension, trainer, model, max_radius, min_radius, n_particles, n_particle_types, x_next, epoch, N, log_dir, cmap)

                elif 'C_F_Jp' in trainer:
                    plot_training_C(x_list, run, device, dimension, trainer, model, max_radius, min_radius, n_particles, n_particle_types, epoch, N, log_dir)

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
        plt.text(0.05, 0.95, f'epoch: {epoch} final loss: {list_loss[-1]:.10f}', transform=ax.transAxes, )
        plt.tight_layout()
        ax = fig.add_subplot(1, 5, 2)
        if ('PDE_MPM_A' in model_config.particle_model_name) and ('GNN_C' in trainer) :
            embedding = to_numpy(model.GNN_C.a)
        else:
            embedding = to_numpy(model.a[0])
        type_list = to_numpy(x[:, 14])
        for n in range(n_particle_types):
            plt.scatter(embedding[type_list == n, 0], embedding[type_list == n, 1], s=1,
                        c=cmap.color(n), label=f'type {n}', alpha=0.5)

        plt.savefig(f"./{log_dir}/tmp_training/Fig_{epoch}.tif")
        plt.close()

        plt.style.use('dark_background')

        # net = os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt')
        # state_dict = torch.load(net, map_location=device)
        # model.load_state_dict(state_dict['model_state_dict'])
        # print(f'reload best_model: {net}')
        #
        # for k in trange(0, n_frames-10, n_frames // 50):
        #
        #     with torch.no_grad():
        #         x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
        #         x_next = torch.tensor(x_list[run][k+1], dtype=torch.float32, device=device).clone().detach()
        #         if 'F' in trainer:
        #             y = x_next[:, 5 + dimension * 2: 9 + dimension * 2].clone().detach()  # F
        #         elif trainer == 'S':
        #             y = x[:, 12 + dimension * 2: 16 + dimension * 2].clone().detach()  # S
        #         elif 'C' in trainer:
        #             y = x[:, 1 + dimension * 2: 5 + dimension * 2].clone().detach()
        #
        #         if 'GNN_C' in trainer:
        #             distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
        #             adj_t = ((distance < max_radius ** 2) & (distance >= min_radius ** 2)).float() * 1
        #             edges = adj_t.nonzero().t().contiguous()
        #             dataset = data.Data(x=x, edge_index=edges, num_nodes=x.shape[0])
        #             pred = model.GNN_C(dataset, training=False)
        #         else:
        #             data_id = torch.ones((n_particles, 1), dtype=torch.float32, device=device) * run
        #             k_list_tensor = torch.ones((n_particles, 1), dtype=torch.int, device=device) * k
        #             dataset = data.Data(x=x, edge_index=[], num_nodes=x.shape[0])
        #             pred = model(dataset, data_id=data_id, k=k_list_tensor, trainer=trainer)
        #
        #         error = F.mse_loss(pred, y).item()
        #
        #     fig = plt.figure(figsize=(20, 6))
        #     plt.subplot(1, 3, 1)
        #     if 'F' in trainer:
        #         f_norm = torch.norm(y.view(n_particles, -1), dim=1).cpu().numpy()
        #         plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=f_norm, s=1, cmap='coolwarm', vmin=1.44 - 0.2,
        #                     vmax=1.44 + 0.2)
        #         plt.colorbar(fraction=0.046, pad=0.04)
        #     elif trainer == 'S':
        #         stress_norm = torch.norm(y.view(n_particles, -1), dim=1)
        #         stress_norm = stress_norm[:, None]
        #         plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=stress_norm[:, 0].cpu(), s=1, cmap='hot', vmin=0, vmax=6E-3)
        #         plt.colorbar(fraction=0.046, pad=0.04)
        #     elif 'C' in trainer:
        #         c_norm = torch.norm(y.view(n_particles, -1), dim=1)
        #         c_norm = c_norm[:, None]
        #         c_norm_np = c_norm[:, 0].cpu().numpy()
        #         x_cpu = x.cpu()
        #         topk_indices = c_norm_np.argsort()[-250:]
        #         plt.scatter(x_cpu[:, 1], x_cpu[:, 2], c=c_norm_np, s=1, cmap='viridis', vmin=0, vmax=80)
        #         plt.colorbar(fraction=0.046, pad=0.04)
        #         plt.scatter(x_cpu[topk_indices, 1], x_cpu[topk_indices, 2], c='red', s=1)
        #     plt.xlim([0, 1])
        #     plt.ylim([0, 1])
        #
        #     plt.subplot(1, 3, 2)
        #     if 'F' in trainer:
        #         f_norm = torch.norm(pred.view(n_particles, -1), dim=1).cpu().numpy()
        #         plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=f_norm, s=1, cmap='coolwarm', vmin=1.44 - 0.2,
        #                     vmax=1.44 + 0.2)
        #         plt.colorbar(fraction=0.046, pad=0.04)
        #     elif trainer == 'S':
        #         stress_norm = torch.norm(pred.view(n_particles, -1), dim=1)
        #         stress_norm = stress_norm[:, None]
        #         plt.scatter(x[:, 1].cpu(), x[:, 2].cpu(), c=stress_norm[:, 0].cpu(), s=1, cmap='hot', vmin=0, vmax=6E-3)
        #         plt.colorbar(fraction=0.046, pad=0.04)
        #     elif 'C' in trainer:
        #         c_norm = torch.norm(pred.view(n_particles, -1), dim=1)
        #         c_norm = c_norm[:, None]
        #         c_norm_np = c_norm[:, 0].cpu().numpy()
        #         x_cpu = x.cpu()
        #         topk_indices = c_norm_np.argsort()[-250:]
        #         plt.scatter(x_cpu[:, 1], x_cpu[:, 2], c=c_norm_np, s=1, cmap='viridis', vmin=0, vmax=80)
        #         plt.colorbar(fraction=0.046, pad=0.04)
        #         plt.scatter(x_cpu[topk_indices, 1], x_cpu[topk_indices, 2], c='red', s=1)
        #
        #     plt.xlim([0, 1])
        #     plt.ylim([0, 1])
        #     plt.text(0.05, 0.95,
        #              f'epoch: {epoch} iteration: {N} error: {error:.2f}', )
        #
        #     plt.subplot(1, 3, 3)
        #     if 'C' in trainer:
        #         # c_norm = torch.norm(y.view(n_particles, -1), dim=1)
        #         # c_norm_pred = torch.norm(pred.view(n_particles, -1), dim=1)
        #         # plt.scatter(c_norm.cpu(), c_norm_pred.cpu(), s=1, c='w', alpha=0.5, edgecolors='none')
        #         for m in range(4):
        #             plt.scatter(y[:, m].cpu(), pred[:, m].cpu(), s=1, c='w', alpha=0.5, edgecolors='none')
        #         plt.xlim([-200, 200])
        #         plt.ylim([-200, 200])
        #
        #     plt.tight_layout()
        #     plt.savefig(f"./{log_dir}/tmp_training/movie/pred_{k}.tif", dpi=87)
        #     plt.close()


def data_train_INR(config=None, device=None, field_name='C', total_steps=50000, erase=False, log_file=None):
    """
    Train INR network on MPM fields (C, F, Jp, S) from generated_data.

    This function loads MPM simulation data and trains an implicit neural representation
    to learn a specific field as a function of (time, particle_id).

    Args:
        config: Configuration object
        device: torch device
        field_name: Which field to train on ('C', 'F', 'Jp', 'S')
        total_steps: Number of training steps
        erase: Whether to erase existing log files
        log_file: Optional file handle for writing analysis metrics
    """

    log_dir, logger = create_log_dir(config, erase)
    output_folder = os.path.join(log_dir, 'tmp_training', 'external_input')
    os.makedirs(output_folder, exist_ok=True)

    # Empty output folder at the beginning
    files = glob.glob(f"{output_folder}/*")
    for file in files:
        try:
            os.remove(file)
        except:
            pass

    dataset_name = config.dataset
    data_folder = f"graphs_data/{dataset_name}/"

    x_list = np.load(f"{data_folder}generated_data_0.npy")
    print(f"x_list shape: {x_list.shape}")  # (n_frames, n_particles, n_features)

    n_frames, n_particles, n_features = x_list.shape
    n_training_frames = config.training.n_training_frames
    if n_training_frames >0 and n_training_frames < n_frames:
        x_list = x_list[n_frames//2-n_training_frames//2:n_frames//2+n_training_frames//2, :, :]
        n_frames = n_training_frames
        print(f"using only {n_training_frames} frames for training.")

    field_indices = {
        'id': (0, 1, 'Particle number'),
        'pos': (1, 3, 'Position'),
        'dpos': (3, 5, 'Velocity'),
        'C': (5, 9, 'APIC matrix'),
        'F': (9, 13, 'Deformation gradient'),
        'Jp': (13, 14, 'Plastic deformation'),
        'type': (14, 15, 'Material type'),
        'S': (16, 20, 'Stress tensor'),
    }
    start_idx, end_idx, field_desc = field_indices[field_name]
    n_components = end_idx - start_idx

    print(f"field info:")
    print(f"  name: {field_name} - {field_desc}")
    print(f"  indices: [{start_idx}:{end_idx}]")

    field_data = x_list[:, :, start_idx:end_idx]  # shape: (n_frames, n_particles, n_components)

    print(f"field statistics:")
    print(f"  shape: {field_data.shape}")
    print(f"  range: [{field_data.min():.4f}, {field_data.max():.4f}]")
    print(f"  mean: {field_data.mean():.4f}, std: {field_data.std():.4f}")

    # get INR configuration
    model_config = config.graph_model
    training_config = config.training
    inr_type = config.graph_model.inr_type

    # get nnr_f config parameters
    hidden_dim_nnr_f = getattr(model_config, 'hidden_dim_nnr_f', 1024)
    n_layers_nnr_f = getattr(model_config, 'n_layers_nnr_f', 3)
    outermost_linear_nnr_f = getattr(model_config, 'outermost_linear_nnr_f', True)
    omega_f = getattr(model_config, 'omega_f', 1024)
    nnr_f_xy_period = getattr(model_config, 'nnr_f_xy_period', 1.0)

    # get training config parameters
    training_config = config.training
    batch_size = getattr(training_config, 'batch_size', 8)
    learning_rate = getattr(training_config, 'learning_rate_NNR_f', 1e-6)

    # determine input/output dimensions based on inr_type
    if inr_type == 'siren_t':
        input_size_nnr_f = 1
        output_size_nnr_f = n_particles * n_components  # outputs all particles at once
    elif inr_type == 'siren_id':
        input_size_nnr_f = 2  # (t, id)
        output_size_nnr_f = n_components  # outputs one particle at a time
    elif inr_type == 'siren_txy':
        input_size_nnr_f = 3  # (t, x, y)
        output_size_nnr_f = n_components  # outputs one particle at a time
    elif inr_type == 'ngp':
        input_size_nnr_f = getattr(model_config, 'input_size_nnr_f', 1)
        output_size_nnr_f = n_particles * n_components  # outputs all particles at once
    else:
        raise ValueError(f"unknown inr_type: {inr_type}")

    # create INR model based on type
    if inr_type == 'ngp':

        # get NGP config parameters
        ngp_n_levels = getattr(model_config, 'ngp_n_levels', 24)
        ngp_n_features_per_level = getattr(model_config, 'ngp_n_features_per_level', 2)
        ngp_log2_hashmap_size = getattr(model_config, 'ngp_log2_hashmap_size', 22)
        ngp_base_resolution = getattr(model_config, 'ngp_base_resolution', 16)
        ngp_per_level_scale = getattr(model_config, 'ngp_per_level_scale', 1.4)
        ngp_n_particles = getattr(model_config, 'ngp_n_particles', 128)
        ngp_n_hidden_layers = getattr(model_config, 'ngp_n_hidden_layers', 4)

        nnr_f = HashEncodingMLP(
            n_input_dims=input_size_nnr_f,
            n_output_dims=output_size_nnr_f,
            n_levels=ngp_n_levels,
            n_features_per_level=ngp_n_features_per_level,
            log2_hashmap_size=ngp_log2_hashmap_size,
            base_resolution=ngp_base_resolution,
            per_level_scale=ngp_per_level_scale,
            n_particles=ngp_n_particles,
            n_hidden_layers=ngp_n_hidden_layers,
            output_activation='none'
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        encoding_params = sum(p.numel() for p in nnr_f.encoding.parameters())
        mlp_params = sum(p.numel() for p in nnr_f.mlp.parameters())
        total_params = encoding_params + mlp_params

        print(f"\nusing HashEncodingMLP (instantNGP):")
        print(f"  hash encoding: {ngp_n_levels} levels × {ngp_n_features_per_level} features")
        print(f"  hash table: 2^{ngp_log2_hashmap_size} = {2**ngp_log2_hashmap_size:,} entries")
        print(f"  mlp: {ngp_n_particles} × {ngp_n_hidden_layers} hidden → {output_size_nnr_f}")
        print(f"  parameters: {total_params:,} (encoding: {encoding_params:,}, mlp: {mlp_params:,})")
        print(f"  compression ratio: {data_dims / total_params:.2f}x")

    elif inr_type in ['siren_t', 'siren_id', 'siren_txy']:
        # create SIREN model for nnr_f
        omega_f_learning = getattr(model_config, 'omega_f_learning', False)
        nnr_f = Siren(
            in_features=input_size_nnr_f,
            hidden_features=hidden_dim_nnr_f,
            hidden_layers=n_layers_nnr_f,
            out_features=output_size_nnr_f,
            outermost_linear=outermost_linear_nnr_f,
            first_omega_0=omega_f,
            hidden_omega_0=omega_f,
            learnable_omega=omega_f_learning
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print(f"\nusing SIREN ({inr_type}):")
        print(f"  architecture: {input_size_nnr_f} → {hidden_dim_nnr_f} × {n_layers_nnr_f} hidden → {output_size_nnr_f}")
        print(f"  omega_f: {omega_f} (learnable: {omega_f_learning})")
        if omega_f_learning and hasattr(nnr_f, 'get_omegas'):
            print(f"  initial omegas: {nnr_f.get_omegas()}")
        print(f"  parameters: {total_params:,}")

    print(f"\ntraining: batch_size={batch_size}, learning_rate={learning_rate}")

    ground_truth = torch.tensor(field_data, dtype=torch.float32, device=device)  # (n_frames, n_particles)

    # prepare inputs based on inr_type
    if inr_type == 'siren_t':
        # input: normalized time
        time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames

    elif inr_type == 'siren_id':
        # input: (t, id)
        # create particle IDs and normalize by n_particles
        neuron_ids = np.arange(n_particles)
        neuron_ids_norm = torch.tensor(neuron_ids / n_particles, dtype=torch.float32, device=device)  # (n_particles,)

    elif inr_type == 'siren_txy':
        # input: (t, x, y)
        start_idx, end_idx, _ = field_indices['pos']
        particle_pos = torch.tensor(x_list[:, :, start_idx:end_idx], dtype=torch.float32, device=device)  / nnr_f_xy_period# (n_particles, 2)
        time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames


    steps_til_summary = total_steps // 10

    # separate omega parameters from other parameters for different learning rates
    omega_f_learning = getattr(model_config, 'omega_f_learning', False)
    learning_rate_omega_f = getattr(training_config, 'learning_rate_omega_f', learning_rate)
    omega_params = [p for name, p in nnr_f.named_parameters() if 'omega' in name]
    other_params = [p for name, p in nnr_f.named_parameters() if 'omega' not in name]
    if omega_params and omega_f_learning:
        optim = torch.optim.Adam([
            {'params': other_params, 'lr': learning_rate},
            {'params': omega_params, 'lr': learning_rate_omega_f}
        ])
        print(f"using separate learning rates: network={learning_rate}, omega={learning_rate_omega_f}")
    else:
        optim = torch.optim.Adam(lr=learning_rate, params=nnr_f.parameters())

    print(f"training nnr_f for {total_steps} steps...")

    loss_list = []
    pbar = trange(total_steps+1, ncols=150)
    for step in pbar:

        if inr_type == 'siren_t':
            # sample batch_size time frames
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = time_input[sample_ids]  # (batch_size, 1)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_particles, n_components)
            pred = nnr_f(time_batch)  # (batch_size, n_particles * n_components)
            pred = pred.reshape(batch_size, n_particles, n_components)  # (batch_size, n_particles, n_components)

        elif inr_type == 'siren_id':
            # sample batch_size time frames, predict all particles for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / n_frames, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all particles: (batch_size, n_particles, 2)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_particles, 1)
            id_expanded = neuron_ids_norm[None, :, None].expand(batch_size, n_particles, 1)
            input_batch = torch.cat([t_expanded, id_expanded], dim=2)  # (batch_size, n_particles, 2)
            input_batch = input_batch.reshape(batch_size * n_particles, 2)  # (batch_size * n_particles, 2)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_particles, n_components)  # (batch_size * n_particles, n_components)
            pred = nnr_f(input_batch)  # (batch_size * n_particles, n_components)

        elif inr_type == 'siren_txy':
            # sample batch_size time frames, predict all particles for each frame
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            t_norm = torch.tensor(sample_ids / n_frames, dtype=torch.float32, device=device)  # (batch_size,)
            # expand to all particles: (batch_size, n_particles, 3)
            t_expanded = t_norm[:, None, None].expand(batch_size, n_particles, 1)
            pos_expanded = particle_pos[sample_ids, :, :]  # (batch_size, n_particles, 2) - use positions at sampled frames
            input_batch = torch.cat([t_expanded, pos_expanded], dim=2)  # (batch_size, n_particles, 3)
            input_batch = input_batch.reshape(batch_size * n_particles, 3)  # (batch_size * n_particles, 3)
            gt_batch = ground_truth[sample_ids].reshape(batch_size * n_particles, n_components)  # (batch_size * n_particles, n_components)
            pred = nnr_f(input_batch)  # (batch_size * n_particles, n_components)

        elif inr_type == 'ngp':
            # sample batch_size time frames (same as siren_t)
            sample_ids = np.random.choice(n_frames, batch_size, replace=False)
            time_batch = torch.tensor(sample_ids / n_frames, dtype=torch.float32, device=device).unsqueeze(1)
            gt_batch = ground_truth[sample_ids]  # (batch_size, n_particles, n_components)
            pred = nnr_f(time_batch)  # (batch_size, n_particles * n_components)
            pred = pred.reshape(batch_size, n_particles, n_components)  # (batch_size, n_particles, n_components)

        # compute loss
        if inr_type == 'ngp':
            # relative L2 error - convert targets to match output dtype (tcnn uses float16)
            relative_l2_error = (pred - gt_batch.to(pred.dtype)) ** 2 / (pred.detach() ** 2 + 0.01)
            loss = relative_l2_error.mean()
        else:
            # standard MSE for SIREN
            loss = ((pred - gt_batch) ** 2).mean()

        # omega L2 regularization for learnable omega in SIREN (encourages smaller omega)
        coeff_omega_f_L2 = getattr(training_config, 'coeff_omega_f_L2', 0.0)
        if omega_f_learning and coeff_omega_f_L2 > 0 and hasattr(nnr_f, 'get_omega_L2_loss'):
            omega_L2_loss = nnr_f.get_omega_L2_loss()
            loss = loss + coeff_omega_f_L2 * omega_L2_loss

        optim.zero_grad()
        loss.backward()
        optim.step()

        loss_list.append(loss.item())
        pbar.set_postfix(loss=f"{loss.item():.2f}")

        if step % steps_til_summary == 0:
            with torch.no_grad():
                # compute predictions for all frames
                if inr_type == 'siren_t':
                    pred_all = nnr_f(time_input)  # (n_frames, n_particles * n_components)
                    pred_all = pred_all.reshape(n_frames, n_particles, n_components)  # (n_frames, n_particles, n_components)

                elif inr_type == 'siren_id':
                    # predict all (t, id) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                        input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)  # (n_particles, 2)
                        pred_t = nnr_f(input_t)  # (n_particles, n_components)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_particles, n_components)

                elif inr_type == 'siren_txy':
                    # predict all (t, x, y) combinations
                    pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                        pos_t = particle_pos[t_idx, :, :]  # (n_particles, 2) - positions at frame t_idx
                        input_t = torch.cat([t_val, pos_t], dim=1)  # (n_particles, 3)
                        pred_t = nnr_f(input_t)  # (n_particles, n_components)
                        pred_list.append(pred_t)
                    pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_particles, n_components)

                elif inr_type == 'ngp':
                    time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames
                    pred_all = nnr_f(time_all)  # (n_frames, n_particles * n_components)
                    pred_all = pred_all.reshape(n_frames, n_particles, n_components)  # (n_frames, n_particles, n_components)

                gt_np = ground_truth.cpu().numpy()
                pred_np = pred_all.cpu().numpy()

                # Create 2-row figure: top row for time series, bottom row for spatial plots
                fig = plt.figure(figsize=(18, 10))
                fig.patch.set_facecolor('black')

                # ROW 1: Loss and time traces
                # loss plot
                ax0 = plt.subplot(2, 3, 1)
                ax0.set_facecolor('black')
                ax0.plot(loss_list, color='white', lw=0.1)
                ax0.set_xlabel('step', color='white', fontsize=12)
                loss_label = 'Relative L2 Loss' if inr_type == 'ngp' else 'MSE Loss'
                ax0.set_ylabel(loss_label, color='white', fontsize=12)
                ax0.set_yscale('log')
                ax0.tick_params(colors='white', labelsize=11)
                for spine in ax0.spines.values():
                    spine.set_color('white')

                # traces plot (10 particles, darkgreen=GT, white=pred)
                # For multi-component fields, plot first component only
                ax1 = plt.subplot(2, 3, 2)
                ax1.set_facecolor('black')
                ax1.set_axis_off()
                n_traces = 10
                trace_ids = np.linspace(0, n_particles - 1, n_traces, dtype=int)

                # Extract first component for plotting if multi-component field
                if n_components > 1:
                    gt_plot = gt_np[:, :, 0]  # (n_frames, n_particles) - first component
                    pred_plot = pred_np[:, :, 0]  # (n_frames, n_particles) - first component
                else:
                    gt_plot = gt_np[:, :, 0]  # (n_frames, n_particles)
                    pred_plot = pred_np[:, :, 0]  # (n_frames, n_particles)

                offset = np.abs(gt_plot).max() * 1.5
                t = np.arange(n_frames)

                for j, n_idx in enumerate(trace_ids):
                    y0 = j * offset
                    ax1.plot(t, gt_plot[:, n_idx] + y0, color='darkgreen', lw=2.0, alpha=0.95)
                    ax1.plot(t, pred_plot[:, n_idx] + y0, color='white', lw=0.5, alpha=0.95)

                ax1.set_xlim(0, min(20000, n_frames))
                ax1.set_ylim(-offset * 0.5, offset * (n_traces + 0.5))
                mse = ((pred_np - gt_np) ** 2).mean()
                omega_str = ''
                if hasattr(nnr_f, 'get_omegas'):
                    omegas = nnr_f.get_omegas()
                    if omegas:
                        omega_str = f'  ω: {omegas[0]:.1f}'
                ax1.text(0.02, 0.98, f'MSE: {mse:.6f}{omega_str}',
                            transform=ax1.transAxes, va='top', ha='left',
                            fontsize=12, color='white')

                # ROW 2: Spatial plots at fixed frame (n_frames // 2)
                frame_idx = n_frames // 2

                # Get positions at this frame
                pos_data = x_list[frame_idx, :, 1:3]  # (n_particles, 2)

                # Compute field norm for coloring
                if n_components > 1:
                    gt_frame = gt_np[frame_idx, :, :]  # (n_particles, n_components)
                    pred_frame = pred_np[frame_idx, :, :]  # (n_particles, n_components)
                    gt_norm = np.linalg.norm(gt_frame, axis=1)  # (n_particles,)
                    pred_norm = np.linalg.norm(pred_frame, axis=1)  # (n_particles,)
                else:
                    gt_frame = gt_np[frame_idx, :, 0]  # (n_particles,)
                    pred_frame = pred_np[frame_idx, :, 0]  # (n_particles,)
                    gt_norm = gt_frame
                    pred_norm = pred_frame

                # Determine colormap and range based on field type
                vmin, vmax = gt_norm.min(), gt_norm.max()
                if field_name == 'F':
                    cmap_name = 'coolwarm'
                    vmin, vmax = 1.44 - 0.2, 1.44 + 0.2
                elif field_name == 'S':
                    cmap_name = 'hot'
                    vmin, vmax = 0, max(6e-3, gt_norm.max())
                elif field_name == 'Jp':
                    cmap_name = 'viridis'
                    vmin, vmax = 0.75, 1.25
                else:
                    cmap_name = 'viridis'

                # Ground truth spatial plot
                ax2 = plt.subplot(2, 3, 4)
                ax2.set_facecolor('black')
                sc = ax2.scatter(pos_data[:, 0], pos_data[:, 1], c=gt_norm, s=3, cmap=cmap_name, vmin=vmin, vmax=vmax)
                plt.colorbar(sc, ax=ax2, fraction=0.046, pad=0.04)
                ax2.set_title(f'GT {field_name} (frame {frame_idx})', color='white', fontsize=12)
                ax2.set_xlim([0, 1])
                ax2.set_ylim([0, 1])
                ax2.set_aspect('equal')
                ax2.tick_params(colors='white', labelsize=10)
                for spine in ax2.spines.values():
                    spine.set_color('white')

                # Prediction spatial plot
                ax3 = plt.subplot(2, 3, 5)
                ax3.set_facecolor('black')
                sc = ax3.scatter(pos_data[:, 0], pos_data[:, 1], c=pred_norm, s=3, cmap=cmap_name, vmin=vmin, vmax=vmax)
                plt.colorbar(sc, ax=ax3, fraction=0.046, pad=0.04)
                ax3.set_title(f'Pred {field_name} (frame {frame_idx})', color='white', fontsize=12)
                ax3.set_xlim([0, 1])
                ax3.set_ylim([0, 1])
                ax3.set_aspect('equal')
                ax3.tick_params(colors='white', labelsize=10)
                for spine in ax3.spines.values():
                    spine.set_color('white')

                # Scatter plot: pred vs gt
                ax4 = plt.subplot(2, 3, 6)
                ax4.set_facecolor('black')
                ax4.scatter(gt_norm, pred_norm, c='white', s=1, alpha=0.5)

                # Compute statistics
                from scipy.stats import linregress
                slope, intercept, r_value, p_value, std_err = linregress(gt_norm, pred_norm)
                r2 = r_value ** 2
                frame_mse = ((pred_norm - gt_norm) ** 2).mean()

                # Add diagonal line (ideal)
                lims = [min(gt_norm.min(), pred_norm.min()), max(gt_norm.max(), pred_norm.max())]
                ax4.plot(lims, lims, 'r--', alpha=0.5, lw=1, label='ideal')

                # Add regression line
                x_line = np.array([gt_norm.min(), gt_norm.max()])
                y_line = slope * x_line + intercept
                ax4.plot(x_line, y_line, 'g-', alpha=0.7, lw=1, label=f'fit (slope={slope:.3f})')

                # Recenter on data range with some padding
                x_margin = (gt_norm.max() - gt_norm.min()) * 0.05
                y_margin = (pred_norm.max() - pred_norm.min()) * 0.05
                ax4.set_xlim([gt_norm.min() - x_margin, gt_norm.max() + x_margin])
                ax4.set_ylim([pred_norm.min() - y_margin, pred_norm.max() + y_margin])

                ax4.set_xlabel('Ground Truth', color='white', fontsize=11)
                ax4.set_ylabel('Prediction', color='white', fontsize=11)
                ax4.set_title(f'Pred vs GT (frame {frame_idx})', color='white', fontsize=12)
                ax4.tick_params(colors='white', labelsize=10)
                for spine in ax4.spines.values():
                    spine.set_color('white')

                # Add statistics text
                stats_text = f'N: {n_particles}\n'
                stats_text += f'R²: {r2:.4f}\n'
                stats_text += f'slope: {slope:.4f}\n'
                stats_text += f'MSE: {frame_mse:.6f}'
                ax4.text(0.05, 0.95, stats_text,
                        transform=ax4.transAxes, va='top', ha='left',
                        fontsize=9, color='white', family='monospace')

                # MSE text on top row
                ax5 = plt.subplot(2, 3, 3)
                ax5.set_facecolor('black')
                ax5.set_axis_off()
                info_text = f'Field: {field_name} ({field_desc})\n'
                info_text += f'Step: {step}\n'
                info_text += f'Components: {n_components}\n'
                info_text += f'Particles: {n_particles}\n'
                info_text += f'Frames: {n_frames}\n'
                info_text += f'Overall MSE: {mse:.6f}\n'
                info_text += f'Frame {frame_idx} MSE: {frame_mse:.6f}'
                ax5.text(0.1, 0.5, info_text, transform=ax5.transAxes,
                        va='center', ha='left', fontsize=11, color='white',
                        family='monospace')

                plt.tight_layout()
                plt.savefig(f"{output_folder}/{inr_type}_{step}.png", dpi=150)
                plt.close()

    # save trained model
    # save_path = f"{output_folder}/nnr_f_{inr_type}_pretrained.pt"
    # torch.save(nnr_f.state_dict(), save_path)
    # print(f"\nsaved pretrained nnr_f to: {save_path}")

    # compute final MSE
    with torch.no_grad():
        if inr_type == 'siren_t':
            pred_all = nnr_f(time_input)  # (n_frames, n_particles * n_components)
            pred_all = pred_all.reshape(n_frames, n_particles, n_components)
        elif inr_type == 'siren_id':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)
                pred_t = nnr_f(input_t)  # (n_particles, n_components)
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_particles, n_components)
        elif inr_type == 'siren_txy':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                pos_t = particle_pos[t_idx, :, :]  # (n_particles, 2) - positions at frame t_idx
                input_t = torch.cat([t_val, pos_t], dim=1)
                pred_t = nnr_f(input_t)  # (n_particles, n_components)
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)  # (n_frames, n_particles, n_components)
        elif inr_type == 'ngp':
            time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames
            pred_all = nnr_f(time_all)  # (n_frames, n_particles * n_components)
            pred_all = pred_all.reshape(n_frames, n_particles, n_components)

        final_mse = ((pred_all - ground_truth) ** 2).mean().item()
        print(f"final MSE: {final_mse:.6f}")
        if hasattr(nnr_f, 'get_omegas'):
            print(f"final omegas: {nnr_f.get_omegas()}")

    # Compute R² score for analysis
    with torch.no_grad():
        if inr_type == 'siren_t':
            pred_all = nnr_f(time_input)
            pred_all = pred_all.reshape(n_frames, n_particles, n_components)
        elif inr_type == 'siren_id':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)
                pred_t = nnr_f(input_t)
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'siren_txy':
            pred_list = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                pos_t = particle_pos[t_idx, :, :]
                input_t = torch.cat([t_val, pos_t], dim=1)
                pred_t = nnr_f(input_t)
                pred_list.append(pred_t)
            pred_all = torch.stack(pred_list, dim=0)
        elif inr_type == 'ngp':
            time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames
            pred_all = nnr_f(time_all)
            pred_all = pred_all.reshape(n_frames, n_particles, n_components)

        # Compute R² (coefficient of determination)
        pred_flat = pred_all.reshape(-1).cpu().numpy()
        gt_flat = ground_truth.reshape(-1).cpu().numpy()
        from scipy.stats import linregress
        slope, intercept, r_value, p_value, std_err = linregress(gt_flat, pred_flat)
        final_r2 = r_value ** 2

    # Write analysis.log if log_file provided
    if log_file is not None:
        log_file.write(f"field_name: {field_name}\n")
        log_file.write(f"inr_type: {inr_type}\n")
        log_file.write(f"final_mse: {final_mse:.6e}\n")
        log_file.write(f"final_r2: {final_r2:.6f}\n")
        log_file.write(f"slope: {slope:.6f}\n")
        log_file.write(f"total_params: {total_params}\n")
        log_file.write(f"n_particles: {n_particles}\n")
        log_file.write(f"n_frames: {n_frames}\n")
        log_file.write(f"n_components: {n_components}\n")
        log_file.write(f"total_steps: {total_steps}\n")
        log_file.write(f"batch_size: {batch_size}\n")
        log_file.write(f"learning_rate: {learning_rate:.6e}\n")
        log_file.write(f"hidden_dim_nnr_f: {hidden_dim_nnr_f}\n")
        log_file.write(f"n_layers_nnr_f: {n_layers_nnr_f}\n")
        log_file.write(f"omega_f: {omega_f}\n")
        if hasattr(nnr_f, 'get_omegas'):
            omegas = nnr_f.get_omegas()
            if omegas:
                log_file.write(f"final_omega_f: {omegas[0]:.2f}\n")

    return nnr_f, loss_list