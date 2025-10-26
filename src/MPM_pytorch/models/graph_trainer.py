import os
from subprocess import run
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

        for N in trange(Niter, ncols=150):

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



def data_test(config=None, config_file=None, visualize=False, style='color frame', verbose=True, best_model=20, step=15, run=0, test_mode='', device=[]):
    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting
    trainer = train_config.MPM_trainer

    dimension = simulation_config.dimension
    n_particles = simulation_config.n_particles
    n_particle_types = simulation_config.n_particle_types
    n_grid = simulation_config.n_grid

    delta_t = simulation_config.delta_t
    dataset_name = config.dataset
    n_frames = simulation_config.n_frames

    cmap = CustomColorMap(config=config)  # create colormap for given model_config
    n_runs = train_config.n_runs

    os.makedirs(f"./{log_dir}/tmp_training/tmp_recons", exist_ok=True)
    files = os.listdir(f"./{log_dir}/tmp_training/tmp_recons")
    for file in files:
        os.remove(f"./{log_dir}/tmp_training/tmp_recons/{file}")

    print('load data ...')
    x_list = []
    x = np.load(f'graphs_data/{dataset_name}/x_list_{run}.npy')
    if np.isnan(x).any():
        print('Pb isnan in x')
    x_list.append(x)
    x = torch.tensor(x_list[0][0], dtype=torch.float32, device=device)

    vnorm = torch.tensor(1, device=device)
    ynorm = torch.tensor(1, device=device)

    print(f'N particles: {n_particles}')
    print(f'N grid: {n_grid}')
    print(f'vnorm: {to_numpy(vnorm)}, ynorm: {to_numpy(ynorm)}')

    print('create models ...')
    model, bc_pos, bc_dpos = choose_training_model(config, device)
    model.ynorm = ynorm
    model.vnorm = vnorm

    if best_model == 'best':
        files = glob.glob(f"{log_dir}/models/*")
        files.sort(key=sort_key)
        filename = files[-1]
        filename = filename.split('/')[-1]
        filename = filename.split('graphs')[-1][1:-3]
        best_model = filename
        print(f'best model: {best_model}')
    net = f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_{best_model}.pt"
    print(f'network: {net}')
    state_dict = torch.load(net, map_location=device)
    model.load_state_dict(state_dict['model_state_dict'])
    model.eval()

    time.sleep(1)

    start_it = 0

    x = x_list[0][start_it].clone().detach()
    n_particles = x.shape[0]


    data_id = torch.ones((n_particles,1), dtype = torch.float32, device=device) * run

    for it in trange(start_it,start_it+800, ncols=150):

        x = torch.tensor(x_list[run][it], dtype=torch.float32, device=device).clone().detach()
        x_next = torch.tensor(x_list[run][it+1], dtype=torch.float32, device=device).clone().detach()

        y = x_next[:, 1:1 + dimension ].clone().detach()  

        with torch.no_grad():  
            pred_x, pred_C, pred_F, pred_Jp, pred_S = model(x, data_id=data_id, k=k, trainer=trainer, batch_size=1)

        N = x[:,0:1]
        X = pred_x
        V = (pred_x - x[:,1:1+dimension]) / delta_t
        C = pred_C
        F = pred_F
        Jp = pred_Jp
        T = x[:,13:14]
        M = x[:,14:15]
        S = pred_S
        ID = x[:,15:16]

        x = torch.cat([N, X, V, C.view(-1, 4), F.view(-1, 4), Jp, T, M, S.view(-1,4), ID], dim=1)
        x = x.clone().detach()

