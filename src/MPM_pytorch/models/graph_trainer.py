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