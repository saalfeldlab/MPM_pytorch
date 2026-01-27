import os
import sys
from subprocess import run
import time
import glob
import shutil
from tqdm import tqdm

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F
import random

# Optional import - not needed for subprocess training
try:
    from run_MPM import *
except ModuleNotFoundError:
    pass  # run_MPM not available when running as subprocess
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

from matplotlib.animation import FFMpegWriter
from collections import deque 
 

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
    if train_config.small_init_batch_size:
        get_batch_size = increasing_batch_size(target_batch_size)
    else:
        get_batch_size = constant_batch_size(target_batch_size)
    batch_ratio = train_config.batch_ratio
    batch_size = get_batch_size(0)
    cmap = CustomColorMap(config=config)  # create colormap for given model_config
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

    for run in trange(n_runs, ncols=50, mininterval=1.0, file=sys.stdout, ascii=True):
        x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
        if np.isnan(x).any():
            print('Pb isnan')
        if x[0].shape[0] > n_particles_max:
            n_particles_max = x[0].shape[0]
        x_list.append(x)
        run_lengths.append(len(x))
    x = torch.tensor(x_list[0][0], dtype=torch.float32, device=device)

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

    # results = recommend_amplitude_values(
    #         x_list=x_list,
    #         run=0,
    #         n_frames=1000,
    #         device='cuda',
    #         sample_every=100,
    #         verbose=True
    #     )


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

    print("start training ...")
    check_and_clear_memory(device=device, iteration_number=0, every_n_iterations=1, memory_percentage_threshold=0.6)

    list_loss = []
    time.sleep(1)

    torch.autograd.set_detect_anomaly(True)
    for epoch in range(start_epoch, n_epochs + 1):


        logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
        logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

        batch_size = int(get_batch_size(epoch))
        logger.info(f'batch_size: {batch_size}')

        if batch_ratio < 1:
            Niter = int(n_frames * data_augmentation_loop // batch_size / batch_ratio)
        else:
            Niter = n_frames * data_augmentation_loop // batch_size

        plot_frequency = int(Niter // 2)
        if epoch == 0:
            print(f'{Niter} iterations per epoch, plot every {plot_frequency} iterations')
            logger.info(f'{Niter} iterations per epoch, plot every {plot_frequency} iterations')

        time.sleep(1)
        total_loss = 0

        run = 0
        data_id = torch.ones((n_particles,1), dtype = torch.float32, device=device) * run

        for N in trange(Niter, ncols=150, mininterval=1.0, file=sys.stdout, ascii=True):

            loss = 0
            optimizer.zero_grad()


            for batch in range(batch_size):


                k = time_window + np.random.randint(run_lengths[run] - 1 - time_window - time_step - recursive_loop)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                x_next = torch.tensor(x_list[run][k+1], dtype=torch.float32, device=device).clone().detach()
                k_batch = torch.ones((n_particles,1), dtype = torch.float32, device=device) * k

                y = x_next[:, 1:1 + dimension ].clone().detach()  

                pred_x, pred_C, pred_F, pred_Jp, pred_S = model(x, data_id=data_id, k=k_batch, trainer=trainer, batch_size=batch_size)
                
                loss = loss + 1E4 *(pred_x - y).norm(2)

                # if coeff_Jp_norm > 0 :
                #     loss = loss + coeff_Jp_norm * F.mse_loss(pred_Jp, torch.ones_like(pred_Jp).detach())
                # if coeff_F_norm > 0 :
                #     F_norm = torch.norm(pred_F.view(-1, 4), dim=1)
                #     loss = loss + coeff_F_norm * (F_norm - torch.ones_like(F_norm).detach() * 1.4141).norm(2)

                if coeff_det_F > 0:
                    det_F = torch.det(pred_F.view(-1, 2, 2))
                    loss = loss + coeff_det_F * F.relu(-det_F + 0.1).norm(2)

            try:
                    
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()
                total_loss += loss.item()
                
            except RuntimeError as e:

                print(f'iteration {N} skipped due to error: {e}')   
                optimizer.zero_grad()  # Clear any partial gradients
                continue



            total_loss += loss.item()


            if (N % plot_frequency == 0):
                
                # Save model checkpoint
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                # plots all fields present in trainer (F, S, C, and/or Jp)
                plot_fields_movie(trainer, model, x_list, 0, 500, n_particles, n_frames, device, log_dir, epoch, N, dataset_name)
                check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_particles))
        logger.info("Epoch {}. Loss: {:.10f}".format(epoch, total_loss / n_particles))
        list_loss.append(total_loss / n_particles)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))
        plot_fields_movie(trainer, model, x_list, 0, 500, n_particles, n_frames, device, log_dir, epoch, N, dataset_name)
                

        plt.style.use('default')
        fig = plt.figure(figsize=(22, 5))
        ax = fig.add_subplot(1, 5, 1)
        plt.plot(list_loss, color='black')
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


def data_train_INR(config=None, device=None, field_name='C', total_steps=None, erase=False, log_file=None, current_iteration=None):
    """
    Train INR network on MPM fields (C, F, Jp, S) from generated_data.

    This function loads MPM simulation data and trains an implicit neural representation
    to learn a specific field as a function of (time, particle_id).

    Args:
        config: Configuration object
        device: torch device
        field_name: Which field to train on ('C', 'F', 'Jp', 'S')
        total_steps: Number of training steps (if None, reads from config.training.total_steps)
        erase: Whether to erase existing log files
        log_file: Optional file handle for writing analysis metrics
    """
    # Read total_steps from config if not provided
    if total_steps is None:
        total_steps = getattr(config.training, 'total_steps', 50000)

    log_dir, logger = create_log_dir(config, erase)
    output_folder = os.path.join(log_dir, 'tmp_training', 'field')
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
    field_data = x_list[:, :, start_idx:end_idx]  # shape: (n_frames, n_particles, n_components)

    print(f"field info:")
    print(f"  name: {field_name} - {field_desc}")
    print(f"  indices: [{start_idx}:{end_idx}]")
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
    nnr_f_T_period = getattr(model_config, 'nnr_f_T_period', 1.0)

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
        # LayerNorm for S field stabilization - reduces stochastic variance
        use_layer_norm = getattr(model_config, 'use_layer_norm', False)
        nnr_f = Siren(
            in_features=input_size_nnr_f,
            hidden_features=hidden_dim_nnr_f,
            hidden_layers=n_layers_nnr_f,
            out_features=output_size_nnr_f,
            outermost_linear=outermost_linear_nnr_f,
            first_omega_0=omega_f,
            hidden_omega_0=omega_f,
            learnable_omega=omega_f_learning,
            use_layer_norm=use_layer_norm
        )
        nnr_f = nnr_f.to(device)

        # count parameters
        total_params = sum(p.numel() for p in nnr_f.parameters())

        print(f"using SIREN ({inr_type}):")
        print(f"  architecture: {input_size_nnr_f} → {hidden_dim_nnr_f} × {n_layers_nnr_f} hidden → {output_size_nnr_f}")
        print(f"  omega_f: {omega_f} (learnable: {omega_f_learning})")
        print(f"  layer_norm: {use_layer_norm}")
        if omega_f_learning and hasattr(nnr_f, 'get_omegas'):
            print(f"  initial omegas: {nnr_f.get_omegas()}")
        print(f"  parameters: {total_params:,}")

    print(f"\ntraining: batch_size={batch_size}, learning_rate_NNR_f={learning_rate}")

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
        particle_pos = torch.tensor(x_list[:, :, start_idx:end_idx], dtype=torch.float32, device=device) / nnr_f_xy_period  # (n_particles, 2)
        time_input = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames / nnr_f_T_period


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

    # clamp batch_size to not exceed number of frames
    if batch_size > n_frames:
        print(f"warning: batch_size ({batch_size}) > n_frames ({n_frames}), clamping to {n_frames}")
        batch_size = n_frames

    import time
    training_start_time = time.time()

    loss_list = []
    # Calculate reporting interval (report 10 times during training)
    report_interval = total_steps // 10
    if report_interval == 0:
        report_interval = 1

    for step in range(total_steps+1):

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
            t_norm = torch.tensor(sample_ids / n_frames / nnr_f_T_period, dtype=torch.float32, device=device)  # (batch_size,)
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
        # Gradient clipping to stabilize training (especially for S field with high variance)
        # COMPLETE CLIPPING MAP: 0.25->0.810, 0.5->[0.828,0.181](HIGH variance), 0.75->0.774, 1.0->[0.785,0.787](LOW variance), 1.5->0.075(FAIL), 2.0->0.118(FAIL), 5.0->0.128(FAIL)
        # CONCLUSION: max_norm=1.0 is MOST STABLE for S field (range=0.002 vs 0.647 at max_norm=0.5)
        # LayerNorm INCOMPATIBLE with SIREN (R²=0.022 catastrophic failure)
        # Robustness test at max_norm=1.0 to confirm stability with 3rd sample
        torch.nn.utils.clip_grad_norm_(nnr_f.parameters(), max_norm=1.0)
        optim.step()

        loss_list.append(loss.item())

        # Report progress 10 times during training
        if step > 0 and step % report_interval == 0:
            elapsed = time.time() - training_start_time
            progress_pct = (step / total_steps) * 100
            steps_per_sec = step / elapsed if elapsed > 0 else 0
            eta_seconds = (total_steps - step) / steps_per_sec if steps_per_sec > 0 else 0
            # Compute R² on ALL data (consistent with final R² and plot)
            with torch.no_grad():
                if inr_type == 'siren_t':
                    prog_pred = nnr_f(time_input).reshape(n_frames, n_particles, n_components)
                elif inr_type == 'siren_id':
                    prog_pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                        input_t = torch.cat([t_val, neuron_ids_norm[:, None]], dim=1)
                        prog_pred_list.append(nnr_f(input_t))
                    prog_pred = torch.stack(prog_pred_list, dim=0)
                elif inr_type == 'siren_txy':
                    prog_pred_list = []
                    for t_idx in range(n_frames):
                        t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                        pos_t = particle_pos[t_idx, :, :]
                        input_t = torch.cat([t_val, pos_t], dim=1)
                        prog_pred_list.append(nnr_f(input_t))
                    prog_pred = torch.stack(prog_pred_list, dim=0)
                elif inr_type == 'ngp':
                    time_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames
                    prog_pred = nnr_f(time_all).reshape(n_frames, n_particles, n_components)
                # Compute R² on all data
                prog_pred_flat = prog_pred.reshape(-1).cpu().numpy()
                gt_all_flat = ground_truth.reshape(-1).cpu().numpy()
                corr_matrix = np.corrcoef(prog_pred_flat, gt_all_flat)
                all_r2 = corr_matrix[0, 1] ** 2 if not np.isnan(corr_matrix[0, 1]) else 0.0
            print(f"  {progress_pct:5.1f}% | step {step:6d}/{total_steps} | loss: {loss.item():.4f} | R²: {all_r2:.4f} | {steps_per_sec:.1f} it/s | eta: {eta_seconds/60:.1f}m", flush=True)

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

                # Create figure: 3x4 layout for 4-component fields, 2x3 for single component
                if n_components == 4:
                    fig = plt.figure(figsize=(20, 12))
                    fig.patch.set_facecolor('black')
                    # Top row uses 3x4 grid positions 1-4
                    ax0 = plt.subplot(3, 4, 1)  # loss
                else:
                    fig = plt.figure(figsize=(18, 10))
                    fig.patch.set_facecolor('black')
                    ax0 = plt.subplot(2, 3, 1)  # loss

                # ROW 1: Loss plot
                ax0.set_facecolor('black')
                ax0.plot(loss_list, color='white', lw=0.1, alpha=0.5)
                # smoothed average window curve (red)
                if len(loss_list) > 100:
                    window_size = min(500, len(loss_list) // 10)
                    loss_array = np.array(loss_list)
                    smoothed_loss = np.convolve(loss_array, np.ones(window_size)/window_size, mode='valid')
                    # offset x-axis to align with original data
                    x_smoothed = np.arange(window_size//2, window_size//2 + len(smoothed_loss))
                    ax0.plot(x_smoothed, smoothed_loss, color='red', lw=1.5, alpha=0.9)
                ax0.set_xlabel('step', color='white', fontsize=12)
                loss_label = 'Relative L2 Loss' if inr_type == 'ngp' else 'MSE Loss'
                ax0.set_ylabel(loss_label, color='white', fontsize=12)
                ax0.set_yscale('log')
                ax0.tick_params(colors='white', labelsize=11)
                for spine in ax0.spines.values():
                    spine.set_color('white')

                # Compute statistics on ALL data first (need slope for correction in traces)
                from scipy.stats import linregress
                gt_all_flat = gt_np.reshape(-1)
                pred_all_flat = pred_np.reshape(-1)
                slope, intercept, r_value, p_value, std_err = linregress(gt_all_flat, pred_all_flat)
                r2 = r_value ** 2
                frame_mse = ((pred_all_flat - gt_all_flat) ** 2).mean()
                mse = ((pred_np - gt_np) ** 2).mean()

                # Compute per-frame MSE for temporal analysis
                per_frame_mse = ((pred_np - gt_np) ** 2).mean(axis=(1, 2))  # (n_frames,)

                # Per-frame MSE plot (top middle panel)
                ax1 = plt.subplot(3, 4, 2) if n_components == 4 else plt.subplot(2, 3, 2)
                ax1.set_facecolor('black')
                frame_indices = np.arange(n_frames)
                ax1.plot(frame_indices, per_frame_mse, color='white', lw=0.5, alpha=0.7)
                # Smoothed curve
                if len(per_frame_mse) > 20:
                    window = min(50, len(per_frame_mse) // 5)
                    smoothed = np.convolve(per_frame_mse, np.ones(window)/window, mode='valid')
                    x_smooth = np.arange(window//2, window//2 + len(smoothed))
                    ax1.plot(x_smooth, smoothed, color='cyan', lw=1.5, alpha=0.9)
                # Mark the frame shown in spatial plots
                ax1.axvline(x=n_frames // 2, color='red', linestyle='--', lw=1, alpha=0.7)
                ax1.set_xlabel('frame', color='white', fontsize=10)
                ax1.set_ylabel('MSE', color='white', fontsize=10)
                ax1.set_title('Per-frame MSE', color='white', fontsize=11)
                ax1.tick_params(colors='white', labelsize=9)
                for spine in ax1.spines.values():
                    spine.set_color('white')

                # Correct predictions with slope (pred_corrected ≈ gt when R² is good)
                pred_np_corrected = (pred_np - intercept) / slope if slope != 0 else pred_np

                # ROW 2: Spatial plots at fixed frame (n_frames // 2)
                frame_idx = n_frames // 2
                pos_data = x_list[frame_idx, :, 1:3]  # (n_particles, 2)

                # Helper function to compute SSIM on gridded data
                def compute_ssim_gridded(gt_vals, pred_vals, pos, grid_size=64):
                    """Grid scattered data and compute SSIM."""
                    from scipy.interpolate import griddata
                    # Create regular grid
                    xi = np.linspace(0, 1, grid_size)
                    yi = np.linspace(0, 1, grid_size)
                    xi, yi = np.meshgrid(xi, yi)
                    # Interpolate to grid
                    gt_grid = griddata(pos, gt_vals, (xi, yi), method='linear', fill_value=np.nan)
                    pred_grid = griddata(pos, pred_vals, (xi, yi), method='linear', fill_value=np.nan)
                    # Mask NaN values
                    valid = ~(np.isnan(gt_grid) | np.isnan(pred_grid))
                    if valid.sum() < 100:
                        return np.nan
                    # Normalize to [0, 1] for SSIM
                    gt_min, gt_max = np.nanmin(gt_grid), np.nanmax(gt_grid)
                    if gt_max - gt_min > 1e-10:
                        gt_norm = (gt_grid - gt_min) / (gt_max - gt_min)
                        pred_norm = (pred_grid - gt_min) / (gt_max - gt_min)  # use gt range
                    else:
                        return np.nan
                    # Simple SSIM computation (mean-based)
                    gt_norm = np.nan_to_num(gt_norm, nan=0.5)
                    pred_norm = np.nan_to_num(pred_norm, nan=0.5)
                    # SSIM formula components
                    mu_x, mu_y = gt_norm.mean(), pred_norm.mean()
                    sig_x, sig_y = gt_norm.std(), pred_norm.std()
                    sig_xy = ((gt_norm - mu_x) * (pred_norm - mu_y)).mean()
                    C1, C2 = 0.01**2, 0.03**2
                    ssim = ((2*mu_x*mu_y + C1) * (2*sig_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sig_x**2 + sig_y**2 + C2))
                    return ssim

                if n_components == 4:
                    # 4-component field (F, S, C): show 2x2 panels for GT and Pred
                    comp_labels = ['00', '01', '10', '11']
                    gt_frame = gt_np[frame_idx, :, :]  # (n_particles, 4)
                    pred_frame = pred_np_corrected[frame_idx, :, :]  # slope-corrected predictions

                    cmap_name = 'coolwarm' if field_name in ['F', 'C'] else 'hot'

                    # Compute vmin/vmax from 98th percentile of GT values (per component)
                    vmin_vmax_per_comp = []
                    for c_idx in range(4):
                        gt_c = gt_frame[:, c_idx]
                        vmin_vmax_per_comp.append((np.percentile(gt_c, 2), np.percentile(gt_c, 98)))

                    # GT panels (2x2)
                    for c_idx in range(4):
                        row = c_idx // 2
                        col = c_idx % 2
                        ax_gt = plt.subplot(3, 4, 5 + col + row * 4)
                        ax_gt.set_facecolor('black')
                        gt_c = gt_frame[:, c_idx]
                        vmin, vmax = vmin_vmax_per_comp[c_idx]
                        ax_gt.scatter(pos_data[:, 0], pos_data[:, 1], c=gt_c, s=1, cmap=cmap_name, vmin=vmin, vmax=vmax)
                        ax_gt.set_title(f'GT {comp_labels[c_idx]}', color='white', fontsize=9)
                        ax_gt.set_xlim([0, 1]); ax_gt.set_ylim([0, 1])
                        ax_gt.set_aspect('equal')
                        ax_gt.set_xticks([]); ax_gt.set_yticks([])

                    # Pred panels (2x2) - use GT vmin/vmax
                    for c_idx in range(4):
                        row = c_idx // 2
                        col = c_idx % 2
                        ax_pred = plt.subplot(3, 4, 7 + col + row * 4)
                        ax_pred.set_facecolor('black')
                        gt_c = gt_frame[:, c_idx]
                        pred_c = pred_frame[:, c_idx]
                        vmin, vmax = vmin_vmax_per_comp[c_idx]  # use GT range
                        ax_pred.scatter(pos_data[:, 0], pos_data[:, 1], c=pred_c, s=1, cmap=cmap_name, vmin=vmin, vmax=vmax)
                        ax_pred.set_title(f'Pred {comp_labels[c_idx]}', color='white', fontsize=9)
                        ax_pred.set_xlim([0, 1]); ax_pred.set_ylim([0, 1])
                        ax_pred.set_aspect('equal')
                        ax_pred.set_xticks([]); ax_pred.set_yticks([])

                else:
                    # Single component field (Jp): original layout
                    gt_frame = gt_np[frame_idx, :, 0]
                    pred_frame = pred_np_corrected[frame_idx, :, 0]  # slope-corrected predictions

                    # Use 98th percentile of GT values for vmin/vmax
                    cmap_name = 'viridis'
                    vmin, vmax = np.percentile(gt_frame, 2), np.percentile(gt_frame, 98)

                    # GT panel (no SSIM here - only on Pred panel)
                    ax2 = plt.subplot(2, 3, 4)
                    ax2.set_facecolor('black')
                    ax2.scatter(pos_data[:, 0], pos_data[:, 1], c=gt_frame, s=3, cmap=cmap_name, vmin=vmin, vmax=vmax)
                    ssim_val = compute_ssim_gridded(gt_frame, pred_frame, pos_data)
                    ax2.set_title(f'GT {field_name} (frame {frame_idx})', color='white', fontsize=12)
                    ax2.set_xlim([0, 1]); ax2.set_ylim([0, 1])
                    ax2.set_aspect('equal')
                    ax2.set_xticks([]); ax2.set_yticks([])

                    # Pred panel (corrected) - use GT vmin/vmax
                    ax3 = plt.subplot(2, 3, 5)
                    ax3.set_facecolor('black')
                    ax3.scatter(pos_data[:, 0], pos_data[:, 1], c=pred_frame, s=3, cmap=cmap_name, vmin=vmin, vmax=vmax)
                    ax3.set_title(f'Pred {field_name} (corrected)', color='white', fontsize=12)
                    ax3.set_xlim([0, 1]); ax3.set_ylim([0, 1])
                    ax3.set_aspect('equal')
                    ax3.set_xticks([]); ax3.set_yticks([])

                # Scatter plot: pred vs gt (use ALL data, uncorrected for honest R²)
                ax4 = plt.subplot(2, 3, 6) if n_components != 4 else plt.subplot(3, 4, 4)
                ax4.set_facecolor('black')

                # Subsample for plotting
                n_plot_points = min(50000, len(gt_all_flat))
                plot_indices = np.random.choice(len(gt_all_flat), n_plot_points, replace=False)
                ax4.scatter(gt_all_flat[plot_indices], pred_all_flat[plot_indices], c='white', s=1, alpha=0.3)

                # Use 98th percentile to handle hot spots/outliers, always start at 0
                gt_p98 = np.percentile(gt_all_flat, 98)
                pred_p98 = np.percentile(pred_all_flat, 98)
                upper_bound = max(gt_p98, pred_p98) * 1.05
                lims = [0, upper_bound]

                ax4.plot(lims, lims, 'r--', alpha=0.5, lw=1, label='ideal')
                x_line = np.array([0, upper_bound])
                y_line = slope * x_line + intercept
                ax4.plot(x_line, y_line, 'g-', alpha=0.7, lw=1, label=f'fit')

                ax4.set_xlim(lims); ax4.set_ylim(lims)
                ax4.set_aspect('equal', adjustable='box')
                ax4.set_xlabel('GT', color='white', fontsize=9)
                ax4.set_ylabel('Pred', color='white', fontsize=9)
                ax4.set_title('Pred vs GT', color='white', fontsize=10)
                ax4.tick_params(colors='white', labelsize=8)
                for spine in ax4.spines.values():
                    spine.set_color('white')

                n_total_values = len(gt_all_flat)
                stats_text = f'N: {n_total_values:,}\n'
                stats_text += f'R²: {r2:.4f}\n'
                stats_text += f'slope: {slope:.4f}\n'
                stats_text += f'MSE: {frame_mse:.6f}'
                ax4.text(0.05, 0.95, stats_text, transform=ax4.transAxes, va='top', ha='left', fontsize=7, color='white', family='monospace')

                # MSE text on top row
                ax5 = plt.subplot(3, 4, 3) if n_components == 4 else plt.subplot(2, 3, 3)
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

    # Calculate training time
    training_time = time.time() - training_start_time
    training_time_min = training_time / 60.0
    print(f"training completed in {training_time_min:.1f} minutes")

    # Copy final field visualization to Claude exploration archive
    try:
        # Find the latest file in output_folder
        output_files = sorted(glob.glob(f"{output_folder}/*.png"))
        if output_files:
            latest_file = output_files[-1]
            # Create destination directory
            final_field_dir = os.path.join('log', 'Claude_exploration', 'instruction_multimaterial_1_discs_3types', 'final_field')
            os.makedirs(final_field_dir, exist_ok=True)
            # Generate unique filename with timestamp
            timestamp = time.strftime("%Y%m%d_%H%M%S")
            dest_filename = f"final_field_{field_name}_{timestamp}.png"
            dest_path = os.path.join(final_field_dir, dest_filename)
            shutil.copy2(latest_file, dest_path)
            print(f"final field saved to: {dest_path}")
    except Exception as e:
        print(f"warning: could not copy final field: {e}")

    # Generate MP4 video showing GT vs Pred for all training frames
    print("generating field comparison video...")
    video_frames_dir = os.path.join(output_folder, 'video_frames')
    os.makedirs(video_frames_dir, exist_ok=True)

    # Clear existing frames
    for f in glob.glob(f"{video_frames_dir}/*.png"):
        os.remove(f)

    # Helper function to compute SSIM on gridded data
    def compute_ssim_for_video(gt_vals, pred_vals, pos, grid_size=64):
        """Grid scattered data and compute SSIM."""
        from scipy.interpolate import griddata
        xi = np.linspace(0, 1, grid_size)
        yi = np.linspace(0, 1, grid_size)
        xi, yi = np.meshgrid(xi, yi)
        gt_grid = griddata(pos, gt_vals, (xi, yi), method='linear', fill_value=np.nan)
        pred_grid = griddata(pos, pred_vals, (xi, yi), method='linear', fill_value=np.nan)
        valid = ~(np.isnan(gt_grid) | np.isnan(pred_grid))
        if valid.sum() < 100:
            return np.nan
        gt_min, gt_max = np.nanmin(gt_grid), np.nanmax(gt_grid)
        if gt_max - gt_min > 1e-10:
            gt_norm = (gt_grid - gt_min) / (gt_max - gt_min)
            pred_norm = (pred_grid - gt_min) / (gt_max - gt_min)
        else:
            return np.nan
        gt_norm = np.nan_to_num(gt_norm, nan=0.5)
        pred_norm = np.nan_to_num(pred_norm, nan=0.5)
        mu_x, mu_y = gt_norm.mean(), pred_norm.mean()
        sig_x, sig_y = gt_norm.std(), pred_norm.std()
        sig_xy = ((gt_norm - mu_x) * (pred_norm - mu_y)).mean()
        C1, C2 = 0.01**2, 0.03**2
        ssim = ((2*mu_x*mu_y + C1) * (2*sig_xy + C2)) / ((mu_x**2 + mu_y**2 + C1) * (sig_x**2 + sig_y**2 + C2))
        return ssim

    # Get predictions and ground truth
    with torch.no_grad():
        gt_np = ground_truth.cpu().numpy()
        if inr_type == 'siren_t':
            time_input_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames
            pred_all_video = nnr_f(time_input_all).reshape(n_frames, n_particles, n_components)
        elif inr_type == 'siren_id':
            neuron_ids_norm_v = torch.tensor(np.arange(n_particles) / n_particles, dtype=torch.float32, device=device)
            pred_list_v = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                input_t = torch.cat([t_val, neuron_ids_norm_v[:, None]], dim=1)
                pred_list_v.append(nnr_f(input_t))
            pred_all_video = torch.stack(pred_list_v, dim=0)
        elif inr_type == 'siren_txy':
            pred_list_v = []
            for t_idx in range(n_frames):
                t_val = torch.full((n_particles, 1), t_idx / n_frames, device=device)
                pos_t = particle_pos[t_idx, :, :]
                input_t = torch.cat([t_val, pos_t], dim=1)
                pred_list_v.append(nnr_f(input_t))
            pred_all_video = torch.stack(pred_list_v, dim=0)
        elif inr_type == 'ngp':
            time_input_all = torch.arange(0, n_frames, dtype=torch.float32, device=device).unsqueeze(1) / n_frames
            pred_all_video = nnr_f(time_input_all).reshape(n_frames, n_particles, n_components)
        pred_np_video = pred_all_video.cpu().numpy()

    # Compute global vmin/vmax from GT for consistent coloring
    if n_components == 4:
        vmin_vmax_global = [(np.percentile(gt_np[:, :, c], 2), np.percentile(gt_np[:, :, c], 98)) for c in range(4)]
        cmap_name = 'coolwarm' if field_name in ['F', 'C'] else 'hot'
    else:
        vmin_global, vmax_global = np.percentile(gt_np[:, :, 0], 2), np.percentile(gt_np[:, :, 0], 98)
        cmap_name = 'viridis'

    # Generate frames
    for frame_idx in range(n_frames):
        pos_data = x_list[frame_idx, :, 1:3]

        if n_components == 4:
            fig, axes = plt.subplots(2, 4, figsize=(16, 8))
            fig.patch.set_facecolor('black')
            comp_labels = ['00', '01', '10', '11']

            for c_idx in range(4):
                gt_c = gt_np[frame_idx, :, c_idx]
                pred_c = pred_np_video[frame_idx, :, c_idx]
                vmin, vmax = vmin_vmax_global[c_idx]
                ssim_c = compute_ssim_for_video(gt_c, pred_c, pos_data)

                # GT
                ax_gt = axes[0, c_idx]
                ax_gt.set_facecolor('black')
                ax_gt.scatter(pos_data[:, 0], pos_data[:, 1], c=gt_c, s=1, cmap=cmap_name, vmin=vmin, vmax=vmax)
                ax_gt.set_title(f'GT {comp_labels[c_idx]}', color='white', fontsize=10)
                ax_gt.set_xlim([0, 1]); ax_gt.set_ylim([0, 1])
                ax_gt.set_aspect('equal')
                ax_gt.set_xticks([]); ax_gt.set_yticks([])

                # Pred
                ax_pred = axes[1, c_idx]
                ax_pred.set_facecolor('black')
                ax_pred.scatter(pos_data[:, 0], pos_data[:, 1], c=pred_c, s=1, cmap=cmap_name, vmin=vmin, vmax=vmax)
                ax_pred.set_title(f'Pred {comp_labels[c_idx]}  SSIM:{ssim_c:.3f}', color='white', fontsize=10)
                ax_pred.set_xlim([0, 1]); ax_pred.set_ylim([0, 1])
                ax_pred.set_aspect('equal')
                ax_pred.set_xticks([]); ax_pred.set_yticks([])

            fig.suptitle(f'Frame {frame_idx}/{n_frames}  Field: {field_name}', color='white', fontsize=14)
        else:
            fig, axes = plt.subplots(1, 2, figsize=(10, 5))
            fig.patch.set_facecolor('black')

            gt_frame = gt_np[frame_idx, :, 0]
            pred_frame = pred_np_video[frame_idx, :, 0]
            ssim_val = compute_ssim_for_video(gt_frame, pred_frame, pos_data)

            # GT
            axes[0].set_facecolor('black')
            axes[0].scatter(pos_data[:, 0], pos_data[:, 1], c=gt_frame, s=3, cmap=cmap_name, vmin=vmin_global, vmax=vmax_global)
            axes[0].set_title(f'GT {field_name}', color='white', fontsize=12)
            axes[0].set_xlim([0, 1]); axes[0].set_ylim([0, 1])
            axes[0].set_aspect('equal')
            axes[0].set_xticks([]); axes[0].set_yticks([])

            # Pred
            axes[1].set_facecolor('black')
            axes[1].scatter(pos_data[:, 0], pos_data[:, 1], c=pred_frame, s=3, cmap=cmap_name, vmin=vmin_global, vmax=vmax_global)
            axes[1].set_title(f'Pred {field_name}  SSIM:{ssim_val:.3f}', color='white', fontsize=12)
            axes[1].set_xlim([0, 1]); axes[1].set_ylim([0, 1])
            axes[1].set_aspect('equal')
            axes[1].set_xticks([]); axes[1].set_yticks([])

            fig.suptitle(f'Frame {frame_idx}/{n_frames}', color='white', fontsize=14)

        plt.tight_layout()
        plt.savefig(f"{video_frames_dir}/frame_{frame_idx:06d}.png", dpi=100, facecolor='black')
        plt.close()

    # Generate MP4 using ffmpeg
    video_output_path = os.path.join(output_folder, f'field_comparison_{field_name}.mp4')
    input_pattern = os.path.join(video_frames_dir, "frame_%06d.png")

    ffmpeg_cmd = [
        "ffmpeg", "-y",
        "-loglevel", "error",
        "-framerate", "30",
        "-i", input_pattern,
        "-vf", "scale=trunc(iw/2)*2:trunc(ih/2)*2",
        "-c:v", "libx264",
        "-crf", "23",
        "-pix_fmt", "yuv420p",
        video_output_path
    ]

    # Check if ffmpeg is available
    if shutil.which('ffmpeg') is None:
        print("ffmpeg not found - install with: conda install -c conda-forge ffmpeg")
    else:
        try:
            import subprocess
            result = subprocess.run(ffmpeg_cmd, capture_output=True, text=True, timeout=300)
            if result.returncode == 0:
                print(f"generated video: {video_output_path}")
                # Note: video copying to exploration folder is handled by run_MPM.py
            else:
                print(f"video generation failed (returncode={result.returncode})")
                print(f"  stderr: {result.stderr}")
        except subprocess.TimeoutExpired:
            print("video generation timeout")
        except Exception as e:
            print(f"video generation error: {e}")

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
        log_file.write(f"learning_rate_NNR_f: {learning_rate:.6e}\n")
        log_file.write(f"hidden_dim_nnr_f: {hidden_dim_nnr_f}\n")
        log_file.write(f"n_layers_nnr_f: {n_layers_nnr_f}\n")
        log_file.write(f"omega_f: {omega_f}\n")
        log_file.write(f"training_time_min: {training_time_min:.1f}\n")
        if hasattr(nnr_f, 'get_omegas'):
            omegas = nnr_f.get_omegas()
            if omegas:
                log_file.write(f"final_omega_f: {omegas[0]:.2f}\n")

    return nnr_f, loss_list