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

from matplotlib.animation import FFMpegWriter
from collections import deque  # Only if using the rolling buffer version
 

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

    for run in trange(n_runs, ncols=50):
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


    if False:
        print("\n=== Analyzing Ground Truth F ===")
        
        all_F_gt = []
        for k in range(0, n_frames, 100):  # Sample every 100 frames for analysis
            x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
            F_gt = x[:, 9:13].reshape(-1, 2, 2)  # [n_particles, 2, 2]
            all_F_gt.append(F_gt)
        
        all_F_gt = torch.stack(all_F_gt)  # [n_sampled_frames, n_particles, 2, 2]
        
        # Overall statistics
        print(f"\nOverall F statistics:")
        print(f"  Min: {all_F_gt.min():.6f}")
        print(f"  Max: {all_F_gt.max():.6f}")
        print(f"  Mean: {all_F_gt.mean():.6f}")
        print(f"  Std: {all_F_gt.std():.6f}")
        
        # Print each component (F[0,0], F[0,1], F[1,0], F[1,1])
        print(f"\nPer-component statistics:")
        component_names = ["F[0,0]", "F[0,1]", "F[1,0]", "F[1,1]"]
        for i in range(2):
            for j in range(2):
                component = all_F_gt[:, :, i, j]
                idx = i * 2 + j
                print(f"  {component_names[idx]}: min={component.min():.6f}, max={component.max():.6f}, mean={component.mean():.6f}, std={component.std():.6f}")
        
        # Analyze determinants
        print(f"\nDeterminant statistics:")
        all_det = torch.det(all_F_gt.view(-1, 2, 2))
        print(f"  Min det(F): {all_det.min():.6f}")
        print(f"  Max det(F): {all_det.max():.6f}")
        print(f"  Mean det(F): {all_det.mean():.6f}")
        print(f"  # det(F) < 0.1: {(all_det < 0.1).sum().item()}/{all_det.numel()}")
        print(f"  # det(F) < 0.01: {(all_det < 0.01).sum().item()}/{all_det.numel()}")
        
        # Analyze norms
        print(f"\nFrobenius norm statistics:")
        all_norms = torch.norm(all_F_gt.view(-1, 4), dim=1)
        print(f"  Min ||F||: {all_norms.min():.6f}")
        print(f"  Max ||F||: {all_norms.max():.6f}")
        print(f"  Mean ||F||: {all_norms.mean():.6f}")
        print(f"  Expected ||F|| for identity: {np.sqrt(2):.6f}")




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

        for N in trange(Niter+1, ncols=150):

            loss = 0
            optimizer.zero_grad()


            for batch in range(batch_size):


                k = time_window + np.random.randint(run_lengths[run] - 1 - time_window - time_step - recursive_loop)
                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device).clone().detach()
                x_next = torch.tensor(x_list[run][k+1], dtype=torch.float32, device=device).clone().detach()
                k_batch = torch.ones((n_particles,1), dtype = torch.float32, device=device) * k

                if 'F' in trainer:
                    y = x_next[:, 1:1 + dimension ].clone().detach()  

                pred_x, pred_C, pred_F, pred_Jp, pred_S = model(x, data_id=data_id, k=k_batch, trainer=trainer, batch_size=batch_size)
                
                if 'F' in trainer:
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

                optimizer.zero_grad()  # Clear any partial gradients
                continue



            total_loss += loss.item()




            if ((N % plot_frequency == 0)) & (N > 0):
                
                # Save model checkpoint
                torch.save({'model_state_dict': model.state_dict(), 'optimizer_state_dict': optimizer.state_dict()},
                        os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}_{N}.pt'))
                
                # Create SIREN F field movie
                if 'F' in trainer:
                    with torch.no_grad():
                        plt.style.use('dark_background')
                        
                        # MP4 writer setup
                        fps = 30
                        metadata = dict(title='SIREN F Field Evolution', artist='Matplotlib', comment='F field over time')
                        writer = FFMpegWriter(fps=fps, metadata=metadata)
                        
                        fig = plt.figure(figsize=(15, 5))
                        
                        # Output path
                        out_dir = f"./{log_dir}/tmp_training/siren_F"
                        os.makedirs(out_dir, exist_ok=True)
                        out_path = f"{out_dir}/F_field_movie_{epoch}_{N}.mp4"
                        if os.path.exists(out_path):
                            os.remove(out_path)
                        
                        # Video parameters
                        step_video = 10  # Sample every 10 frames
                        n_frames_to_plot = min(200, n_frames)  # Limit to 200 frames for speed
                        
                        with writer.saving(fig, out_path, dpi=300):
                            
                            for k in range(n_frames-1000, n_frames, step_video):
                                
                                # Clear the figure
                                fig.clear()
                                
                                # Load particle data for frame k
                                x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                                pos = x[:, 1:3]  # Particle positions
                                frame_normalized = torch.tensor(k / n_frames, dtype=torch.float32, device=device)
                                
                                # Get SIREN prediction for F
                                features = torch.cat([
                                    pos,
                                    frame_normalized.expand(n_particles, 1)
                                ], dim=1)
                                
                                # Apply your identity + tanh formulation
                                identity = torch.eye(2, device=device).unsqueeze(0).expand(n_particles, -1, -1)
                                F_pred = identity + torch.tanh(model.siren_F(features).reshape(-1, 2, 2))
                                
                                # Calculate F norm for visualization
                                f_norm_pred = torch.norm(F_pred.view(n_particles, -1), dim=1).cpu().numpy()
                                
                                # Get ground truth F if available
                                F_gt = x[:, 9:13].reshape(-1, 2, 2)
                                f_norm_gt = torch.norm(F_gt.view(n_particles, -1), dim=1).cpu().numpy()
                                
                                # Create subplots
                                ax1 = fig.add_subplot(1, 3, 1)
                                ax2 = fig.add_subplot(1, 3, 2)
                                ax3 = fig.add_subplot(1, 3, 3)
                                
                                # Plot ground truth F
                                scatter1 = ax1.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), 
                                                    c=f_norm_gt, s=1, cmap='coolwarm', 
                                                    vmin=np.sqrt(2)-0.1, vmax=np.sqrt(2)+0.1)
                                ax1.set_title(f'ground truth F (Frame {k})', fontsize=10)
                                ax1.set_xlabel('X')
                                ax1.set_ylabel('Y')
                                ax1.set_aspect('equal')
                                plt.colorbar(scatter1, ax=ax1, label='||F||')
                                ax1.set_xlim([0, 1])
                                ax1.set_ylim([0, 1])
                                
                                # Plot SIREN predicted F
                                scatter2 = ax2.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), 
                                                    c=f_norm_pred, s=1, cmap='coolwarm',
                                                    vmin=np.sqrt(2)-0.1, vmax=np.sqrt(2)+0.1)
                                ax2.set_title(f'SIREN predicted F (Frame {k})', fontsize=10)
                                ax2.set_xlabel('X')
                                ax2.set_ylabel('Y')
                                ax2.set_aspect('equal')
                                plt.colorbar(scatter2, ax=ax2, label='||F||')
                                ax2.set_xlim([0, 1])
                                ax2.set_ylim([0, 1])
                                
                                # Plot error
                                f_error = np.abs(f_norm_pred - f_norm_gt)
                                scatter3 = ax3.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), 
                                                    c=f_error, s=1, cmap='viridis',
                                                    vmin=-0.01, vmax=0.01)
                                ax3.set_title(f'absolute error (Frame {k})', fontsize=10)
                                ax3.set_xlabel('X')
                                ax3.set_ylabel('Y')
                                ax3.set_aspect('equal')
                                plt.colorbar(scatter3, ax=ax3, label='|error|')
                                ax3.set_xlim([0, 1])
                                ax3.set_ylim([0, 1])
                                
                                # Add training info
                                fig.suptitle(f'epoch {epoch}, iter {N} - training SIREN F Field', fontsize=12)
                                
                                plt.tight_layout()
                                
                                # Write frame to video
                                writer.grab_frame()
                        
                        plt.close(fig)


                # check_and_clear_memory(device=device, iteration_number=N, every_n_iterations=Niter // 50, memory_percentage_threshold=0.6)

        torch.save({'model_state_dict': model.state_dict(),
                    'optimizer_state_dict': optimizer.state_dict()},
                   os.path.join(log_dir, 'models', f'best_model_with_{n_runs - 1}_graphs_{epoch}.pt'))

        print("Epoch {}. Loss: {:.6f}".format(epoch, total_loss / n_particles))
        logger.info("Epoch {}. Loss: {:.10f}".format(epoch, total_loss / n_particles))
        list_loss.append(total_loss / n_particles)
        torch.save(list_loss, os.path.join(log_dir, 'loss.pt'))

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
