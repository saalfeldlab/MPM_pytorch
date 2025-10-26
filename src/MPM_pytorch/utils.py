import glob
import logging
import os

import GPUtil
import imageio
import matplotlib.pyplot as plt
import numpy as np
import torch
from matplotlib.ticker import FormatStrFormatter
from skimage.metrics import structural_similarity as ssim
from torch_geometric.data import Data
from torchvision.transforms import CenterCrop
import gc
from torch import cuda
import subprocess
import re
from tqdm import *
from scipy.fft import fft, ifft
import networkx as nx
from collections import defaultdict
from sklearn.linear_model import LinearRegression
from scipy import stats

import warnings
from collections import defaultdict
import matplotlib.patches as patches
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.animation import FFMpegWriter
from collections import deque 


warnings.filterwarnings('ignore')

def sort_key(filename):
            # Extract the numeric parts using regular expressions
            if filename.split('_')[-2] == 'graphs':
                return 0
            else:
                return 1E7 * int(filename.split('_')[-2]) + int(filename.split('_')[-1][:-3])


def to_numpy(tensor: torch.Tensor) -> np.ndarray:
    """
    Convert a PyTorch tensor to a NumPy array.

    Args:
        tensor (torch.Tensor): The PyTorch tensor to convert.

    Returns:
        np.ndarray: The NumPy array.
    """
    return tensor.detach().cpu().numpy()


def set_device(device: str = 'auto'):
    """
    Set the device to use for computations. If 'auto' is specified, the device is chosen automatically:
     * if GPUs are available, the GPU with the most free memory is chosen
     * if MPS is available, MPS is used
     * otherwise, the CPU is used
    :param device: The device to use for computations. Automatically chosen if 'auto' is specified (default).
    :return: The torch.device object that is used for computations.
    """
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ.pop('CUDA_VISIBLE_DEVICES', None)  # Unset CUDA_VISIBLE_DEVICES

    if device == 'auto':
        if torch.cuda.is_available():
            try:
                # Use nvidia-smi to get free memory of each GPU
                result = subprocess.check_output(
                    ['nvidia-smi', '--query-gpu=index,memory.free', '--format=csv,noheader,nounits'],
                    encoding='utf-8'
                )
                # Parse the output
                free_mem_list = []
                for line in result.strip().split('\n'):
                    index_str, mem_str = line.strip().split(',')
                    index = int(index_str)
                    free_mem = float(mem_str) * 1024 * 1024  # Convert MiB to bytes
                    free_mem_list.append((index, free_mem))
                # Ensure the device count matches
                num_gpus = torch.cuda.device_count()
                if num_gpus != len(free_mem_list):
                    print(f"Mismatch in GPU count between PyTorch ({num_gpus}) and nvidia-smi ({len(free_mem_list)})")
                    device = 'cpu'
                    print(f"Using device: {device}")
                else:
                    # Find the GPU with the most free memory
                    max_free_memory = -1
                    best_device_id = -1
                    for index, free_mem in free_mem_list:
                        if free_mem > max_free_memory:
                            max_free_memory = free_mem
                            best_device_id = index
                    if best_device_id == -1:
                        raise ValueError("Could not determine the GPU with the most free memory.")

                    device = f'cuda:{best_device_id}'
                    torch.cuda.set_device(best_device_id)  # Set the chosen device globally
                    total_memory_gb = torch.cuda.get_device_properties(best_device_id).total_memory / 1024 ** 3
                    free_memory_gb = max_free_memory / 1024 ** 3
                    print(
                        f"Using device: {device}, name: {torch.cuda.get_device_name(best_device_id)}, "
                        f"total memory: {total_memory_gb:.2f} GB, free memory: {free_memory_gb:.2f} GB")
            except Exception as e:
                print(f"Failed to get GPU information: {e}")
                device = 'cpu'
                print(f"Using device: {device}")
        elif torch.backends.mps.is_available():
            device = 'mps'
            print(f"Using device: {device}")
        else:
            device = 'cpu'
            print(f"Using device: {device}")
    return device


def set_size(x, particles, mass_distrib_index):
    # particles = index_particles[n]

    #size = 5 * np.power(3, ((to_numpy(x[index_particles[n] , -2]) - 200)/100)) + 10
    size = np.power((to_numpy(x[particles, mass_distrib_index])), 1.2) / 1.5

    return size


def get_gpu_memory_map(device=None):
    t = np.round(torch.cuda.get_device_properties(device).total_memory / 1E9, 2)
    r = np.round(torch.cuda.memory_reserved(device) / 1E9, 2)
    a = np.round(torch.cuda.memory_allocated(device) / 1E9, 2)

    return t, r, a


def symmetric_cutoff(x, percent=1):
    """
    Minimum and maximum values if a certain percentage of the data is cut off from both ends.
    """
    x_lower = np.percentile(x, percent)
    x_upper = np.percentile(x, 100 - percent)
    return x_lower, x_upper


def norm_area(xx, device):

    pos = torch.argwhere(xx[:, -1]<1.0)
    ax = torch.std(xx[pos, -1])

    return torch.tensor([ax], device=device)


def norm_velocity(xx, dimension, device):
    if dimension == 2:
        vx = torch.std(xx[:, 3])
        vy = torch.std(xx[:, 4])
        nvx = np.array(xx[:, 3].detach().cpu())
        vx01, vx99 = symmetric_cutoff(nvx)
        nvy = np.array(xx[:, 4].detach().cpu())
        vy01, vy99 = symmetric_cutoff(nvy)
    else:
        vx = torch.std(xx[:, 4])
        vy = torch.std(xx[:, 5])
        vz = torch.std(xx[:, 6])
        nvx = np.array(xx[:, 4].detach().cpu())
        vx01, vx99 = symmetric_cutoff(nvx)
        nvy = np.array(xx[:, 5].detach().cpu())
        vy01, vy99 = symmetric_cutoff(nvy)
        nvz = np.array(xx[:, 6].detach().cpu())
        vz01, vz99 = symmetric_cutoff(nvz)

    # return torch.tensor([vx01, vx99, vy01, vy99, vx, vy], device=device)

    return torch.tensor([vx], device=device)


def norm_position(xx, dimension, device):
    if dimension == 2:
        bounding_box = get_2d_bounding_box(xx[:, 1:3]* 1.1)
        posnorm = max(bounding_box.values())

        return torch.tensor(posnorm, dtype=torch.float32, device=device), torch.tensor([bounding_box['x_max']/posnorm, bounding_box['y_max']/posnorm], dtype=torch.float32, device=device)
    else:

        bounding_box = get_3d_bounding_box(xx[:, 1:4]* 1.1)
        posnorm = max(bounding_box.values())

        return torch.tensor(posnorm, dtype=torch.float32, device=device), torch.tensor([bounding_box['x_max']/posnorm, bounding_box['y_max']/posnorm, bounding_box['z_max']/posnorm], dtype=torch.float32, device=device)


def norm_acceleration(yy, device):
    ax = torch.std(yy[:, 0])
    ay = torch.std(yy[:, 1])
    nax = np.array(yy[:, 0].detach().cpu())
    ax01, ax99 = symmetric_cutoff(nax)
    nay = np.array(yy[:, 1].detach().cpu())
    ay01, ay99 = symmetric_cutoff(nay)

    # return torch.tensor([ax01, ax99, ay01, ay99, ax, ay], device=device)

    return torch.tensor([ax], device=device)


def choose_boundary_values(bc_name):
    def identity(x):
        return x

    def periodic(x):
        return torch.remainder(x, 1.0)

    def periodic_wall(x):
        y = torch.remainder(x[:,0:1], 1.0)
        return torch.cat((y,x[:,1:2]), 1)

    def shifted_periodic(x):
        return torch.remainder(x - 0.5, 1.0) - 0.5

    def shifted_periodic_wall(x):
        y = torch.remainder(x[:,0:1] - 0.5, 1.0) - 0.5
        return torch.cat((y,x[:,1:2]), 1)


    match bc_name:
        case 'no':
            return identity, identity
        case 'periodic':
            return periodic, shifted_periodic
        case 'wall':
            return periodic_wall, shifted_periodic_wall
        case _:
            raise ValueError(f'unknown boundary condition {bc_name}')


def grads2D(params):
    params_sx = torch.roll(params, -1, 0)
    params_sy = torch.roll(params, -1, 1)

    sx = -(params - params_sx)
    sy = -(params - params_sy)

    sx[-1, :] = 0
    sy[:, -1] = 0

    return [sx, sy]


def tv2D(params):
    nb_voxel = (params.shape[0]) * (params.shape[1])
    sx, sy = grads2D(params)
    tvloss = torch.sqrt(sx.cuda() ** 2 + sy.cuda() ** 2 + 1e-8).sum()
    return tvloss / nb_voxel


def density_laplace(y, x):
    grad = density_gradient(y, x)
    return density_divergence(grad, x)


def density_divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def density_gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def compute_angle(tensor1, tensor2):
    # Ensure the tensors are 2D
    assert tensor1.shape == tensor2.shape == (2,), "Tensors must be 2D vectors"

    # Compute the dot product
    dot_product = torch.dot(tensor1, tensor2)

    # Compute the magnitudes (norms) of the tensors
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (norm1 * norm2)

    # Compute the angle in radians
    angle = torch.acos(cos_angle)

    return angle * 180 / np.pi


def compute_signed_angle(tensor1, tensor2):
    # Ensure the tensors are 2D
    assert tensor1.shape == tensor2.shape == (2,), "Tensors must be 2D vectors"

    # Compute the dot product
    dot_product = torch.dot(tensor1, tensor2)

    # Compute the magnitudes (norms) of the tensors
    norm1 = torch.norm(tensor1)
    norm2 = torch.norm(tensor2)

    # Compute the cosine of the angle
    cos_angle = dot_product / (norm1 * norm2)

    # Compute the angle in radians
    angle = torch.acos(cos_angle)

    # Compute the sign of the angle using the cross product
    cross_product = tensor1[0] * tensor2[1] - tensor1[1] * tensor2[0]
    sign = torch.sign(cross_product)

    # Return the signed angle
    return angle * sign * 180 / np.pi


def get_r2_numpy_corrcoef(x, y):
    return np.corrcoef(x, y)[0, 1] ** 2


class CustomColorMap:
    def __init__(self, config):
        self.cmap_name = config.plotting.colormap
        self.model_name = config.graph_model.particle_model_name

        if self.cmap_name == 'tab10':
            self.nmap = 8
        else:
            self.nmap = config.simulation.n_particles

        self.has_mesh = 'Mesh' in self.model_name

    def color(self, index):

        if ('PDE_MPM' in self.model_name) :
            match index:
                case 0:
                    color = (0, 0.5, 0.75)
                case 1:
                    color = (1, 0, 0)
                case 2:
                    color = (0.75, 0.75, 0.75)
                case 3:
                    color = (0.5, 0.75, 0)
                case 4:
                    color = (0, 0.75, 0)
                case 5:
                    color = (0.5, 0, 0.25)
                case _:
                    color = (1, 1, 1)
        elif ('PDE_F' in self.model_name) | ('PDE_M' in self.model_name) | ('PDE_MLPs' in self.model_name):
            match index:
                case 0:
                    color = (0.75, 0.75, 0.75)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.5, 0.75, 0)
                case 4:
                    color = (0, 0.75, 0)
                case 5:
                    color = (0.5, 0, 0.25)
                case _:
                    color = (1, 1, 1)
        elif self.model_name == 'PDE_E':
            match index:
                case 0:
                    color = (1, 1, 1)
                case 1:
                    color = (0, 0.5, 0.75)
                case 2:
                    color = (1, 0, 0)
                case 3:
                    color = (0.75, 0, 0)
                case _:
                    color = (0.5, 0.5, 0.5)
        elif self.has_mesh:
            if index == 0:
                color = (0, 0, 0)
            else:
                color_map = plt.colormaps.get_cmap(self.cmap_name)
                color = color_map(index / self.nmap)
        else:
            color_map = plt.colormaps.get_cmap(self.cmap_name)
            if self.cmap_name == 'tab20':
                color = color_map(index % 20)
            else:
                color = color_map(index)

        return color


def load_image(path, crop_width=None, device='cpu'):
    target = imageio.v2.imread(path).astype(np.float32)
    target = target / np.max(target)
    target = torch.tensor(target).unsqueeze(0).to(device)
    if crop_width is not None:
        target = CenterCrop(crop_width)(target)
    return target


def get_mgrid(sidelen, dim=2):
    """Generates a flattened grid of (x,y,...) coordinates in a range of -1 to 1.
    sidelen: int
    dim: int
    """
    tensors = tuple(dim * [torch.linspace(-1, 1, steps=sidelen)])
    mgrid = torch.stack(torch.meshgrid(*tensors), dim=-1)
    mgrid = mgrid.reshape(-1, dim)
    return mgrid


def divergence(y, x):
    div = 0.
    for i in range(y.shape[-1]):
        div += torch.autograd.grad(y[..., i], x, torch.ones_like(y[..., i]), create_graph=True)[0][..., i:i + 1]
    return div


def gradient(y, x, grad_outputs=None):
    if grad_outputs is None:
        grad_outputs = torch.ones_like(y)
    grad = torch.autograd.grad(y, [x], grad_outputs=grad_outputs, create_graph=True)[0]
    return grad


def laplace(y, x):
    grad = gradient(y, x)
    return divergence(grad, x)


def calculate_psnr(img1, img2, max_value=255):
    """Calculating peak signal-to-noise ratio (PSNR) between two images."""
    mse = np.mean((np.array(img1, dtype=np.float32) - np.array(img2, dtype=np.float32)) ** 2)
    if mse == 0:
        return 100
    return 20 * np.log10(max_value / (np.sqrt(mse)))


def calculate_ssim(img1, img2):
    ssim_score = ssim(img1, img2, data_range=img2.max() - img2.min())
    return ssim_score


def add_pre_folder(config_file_):

    if 'arbitrary' in config_file_:
        config_file = os.path.join('arbitrary', config_file_)
        pre_folder = 'arbitrary/'
    elif 'boids' in config_file_:
        config_file = os.path.join('boids', config_file_)
        pre_folder = 'boids/'
    elif 'Coulomb' in config_file_:
        config_file = os.path.join('Coulomb', config_file_)
        pre_folder = 'Coulomb/'
    elif 'fluids' in config_file_:
        config_file = os.path.join('fluids', config_file_)
        pre_folder = 'fluids/'
    elif 'gravity' in config_file_:
        config_file = os.path.join('gravity', config_file_)
        pre_folder = 'gravity/'
    elif 'springs' in config_file_:
        config_file = os.path.join('springs', config_file_)
        pre_folder = 'springs/'
    elif 'CElegans' in config_file_:
        config_file = os.path.join('CElegans', config_file_)
        pre_folder = 'CElegans/'
    elif 'fly' in config_file_:
        config_file = os.path.join('fly', config_file_)
        pre_folder = 'fly/'
    elif 'signal' in config_file_:
        config_file = os.path.join('signal', config_file_)
        pre_folder = 'signal/'
    elif 'falling_water_ramp' in config_file_:
        config_file = os.path.join('falling_water_ramp', config_file_)
        pre_folder = 'falling_water_ramp/'
    elif 'multimaterial' in config_file_:
        config_file = os.path.join('multimaterial', config_file_)
        pre_folder = 'multimaterial/'
    elif 'RD_RPS' in config_file_:
        config_file = os.path.join('reaction_diffusion', config_file_)
        pre_folder = 'reaction_diffusion/'
    elif 'wave' in config_file_:
        config_file = os.path.join('wave', config_file_)
        pre_folder = 'wave/'
    elif ('cell' in config_file_) | ('cardio' in config_file_) | ('U2OS' in config_file_):
        config_file = os.path.join('cell', config_file_)
        pre_folder = 'cell/'
    elif 'mouse' in config_file_:
        config_file = os.path.join('mouse_city', config_file_)
        pre_folder = 'mouse_city/'
    elif 'rat' in config_file_:
        config_file = os.path.join('rat_city', config_file_)
        pre_folder = 'rat_city/'

    return config_file, pre_folder


def get_log_dir(config=[]):

    if 'PDE_A' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/arbitrary/')
    elif 'PDE_B' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/boids/')
    elif 'PDE_E' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/Coulomb/')
    elif 'PDE_F' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/fluids/')
    elif 'PDE_G' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/gravity/')
    elif 'PDE_K' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/springs/')
    elif 'PDE_N' in config.graph_model.signal_model_name:
        l_dir = os.path.join('./log/signal/')
    elif 'PDE_M' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/multimaterial/')
    elif 'PDE_MLPs' in config.graph_model.particle_model_name:
        l_dir = os.path.join('./log/multimaterial/')
    elif 'RD_RPS' in config.graph_model.mesh_model_name:
        l_dir = os.path.join('./log/reaction_diffusion/')
    elif 'Wave' in config.graph_model.mesh_model_name:
        l_dir = os.path.join('./log/wave/')
    elif 'cell' in config.dataset:
        l_dir = os.path.join('./log/cell/')
    elif 'mouse' in config.dataset:
        l_dir = os.path.join('./log/mouse/')
    elif 'rat' in config.dataset:
        l_dir = os.path.join('./log/rat/')
    elif 'celegans' in config.dataset:
        l_dir = os.path.join('./log/celegans/')

    return l_dir


def create_log_dir(config=[], erase=True):

    log_dir = os.path.join('.', 'log', config.config_file)
    print('log_dir: {}'.format(log_dir))

    os.makedirs(log_dir, exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'models'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'results'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/particle'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/field'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/matrix'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/prediction'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/lin_phi'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/function/lin_edge'), exist_ok=True)
    os.makedirs(os.path.join(log_dir, 'tmp_training/embedding'), exist_ok=True)
    if config.training.n_ghosts > 0:
        os.makedirs(os.path.join(log_dir, 'tmp_training/ghost'), exist_ok=True)

    if erase:
        files = glob.glob(f"{log_dir}/results/*")
        for f in files:
            if ('all' not in f) & ('field' not in f):
                os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/particle/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/field/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/matrix/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/function/lin_edge/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/function/lin_phis/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/embedding/*")
        for f in files:
            os.remove(f)
        files = glob.glob(f"{log_dir}/tmp_training/ghost/*")
        for f in files:
            os.remove(f)
    os.makedirs(os.path.join(log_dir, 'tmp_recons'), exist_ok=True)

    logging.basicConfig(filename=os.path.join(log_dir, 'training.log'),
                        format='%(asctime)s %(message)s',
                        filemode='w')
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)
    logger.info(config)

    return log_dir, logger


def bundle_fields(data: Data, *names: str) -> torch.Tensor:
    tensors = []
    for name in names:
        tensor = data[name]
        if tensor.dim() == 1:
            tensor = tensor.unsqueeze(-1)
        tensors.append(tensor)
    return torch.concatenate(tensors, dim=-1)


def fig_init(fontsize=48, formatx='%.2f', formaty='%.2f'):
    # from matplotlib import rc, font_manager
    # from numpy import arange, cos, pi
    # from matplotlib.pyplot import figure, axes, plot, xlabel, ylabel, title, \
    #     grid, savefig, show
    # sizeOfFont = 12
    # fontProperties = {'family': 'sans-serif', 'sans-serif': ['Helvetica'],
    #                   'weight': 'normal', 'size': sizeOfFont}
    # ticks_font = font_manager.FontProperties(family='sans-serif', style='normal',
    #                                          size=sizeOfFont, weight='normal', stretch='normal')
    # rc('text', usetex=True)
    # rc('font', **fontProperties)
    # figure(1, figsize=(6, 4))
    # ax = axes([0.1, 0.1, 0.8, 0.7])
    # t = arange(0.0, 1.0 + 0.01, 0.01)
    # s = cos(2 * 2 * pi * t) + 2
    # plot(t, s)
    # for label in ax.get_xticklabels():
    #     label.set_fontproperties(ticks_font)
    # for label in ax.get_yticklabels():
    #     label.set_fontproperties(ticks_font)
    # xlabel(r'\textbf{time (s)}')
    # ylabel(r'\textit{voltage (mV)}', fontsize=16, family='Helvetica')
    # title(r"\TeX\ is Number $\displaystyle\sum_{n=1}^\infty\frac{-e^{i\pi}}{2^n}$!",
    #       fontsize=16, color='r')

    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1, 1, 1)
    plt.xticks([])
    plt.yticks([])
    # ax.xaxis.get_major_formatter()._usetex = False
    # ax.yaxis.get_major_formatter()._usetex = False
    ax.tick_params(axis='both', which='major', pad=15)
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(3))
    ax.xaxis.set_major_formatter(FormatStrFormatter(formatx))
    ax.yaxis.set_major_formatter(FormatStrFormatter(formaty))
    plt.xticks(fontsize=fontsize)
    plt.yticks(fontsize=fontsize)

    return fig, ax


def get_time_series(x_list, cell_id, feature):
    match feature:
        case 'mass':
            feature = 10
        case 'velocity_x':
            feature = 3
        case 'velocity_y':
            feature = 4
        case "type":
            feature = 5
        case "stage":
            feature = 9
        case _:  # default
            feature = 0

    time_series = []
    for it in range(len(x_list)):
        x = x_list[it].clone().detach()
        pos_cell = torch.argwhere(x[:, 0] == cell_id)
        if len(pos_cell) > 0:
            time_series.append(x[pos_cell, feature].squeeze())
        else:
            time_series.append(torch.tensor([0.0]))
    return to_numpy(torch.stack(time_series))


def reparameterize(mu, logvar):
    std = torch.exp(0.5*logvar)
    eps = torch.randn_like(std)
    return eps.mul(std).add_(mu)


def get_sorted_image_files(pic_folder, pic_format):
    # Check if the folder exists
    if not os.path.exists(pic_folder):
        raise FileNotFoundError(f"The folder `{pic_folder}` does not exist.")

    # Get the list of image files with the specified format
    image_files = glob.glob(os.path.join(pic_folder, f"*.{pic_format}"))

    # Sort the files based on the number in the filename
    image_files.sort(key=lambda f: int(re.search(r'(\d+)', os.path.basename(f)).group(1)))

    return image_files


def extract_number(filename):
    match = re.search(r'0-(\d+)\.jpg$', filename)
    return int(match.group(1)) if match else None


def check_and_clear_memory(
        device: str = None,
        iteration_number: int = None,
        every_n_iterations: int = 100,
        memory_percentage_threshold: float = 0.6
):
    """
    Check the memory usage of a GPU and clear the cache every n iterations or if it exceeds a certain threshold.
    :param device: The device to check the memory usage for.
    :param iteration_number: The current iteration number.
    :param every_n_iterations: Clear the cache every n iterations.
    :param memory_percentage_threshold: Percentage of memory usage that triggers a clearing.
    """

    if device and 'cuda' in device:
        logger = logging.getLogger(__name__)

        if (iteration_number % every_n_iterations == 0):

            # logger.info(f"Recurrent cuda cleanining")
            # logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")

            torch.cuda.memory_allocated(device)
            gc.collect()
            torch.cuda.empty_cache()

            if (iteration_number==0):
                logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
                logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")


        if torch.cuda.memory_allocated(device) > memory_percentage_threshold * torch.cuda.get_device_properties(device).total_memory:
            print ("Memory usage is high. Calling garbage collector and clearing cache.")
            # logger.info(f"Total allocated memory: {torch.cuda.memory_allocated(device) / 1024 ** 3:.2f} GB")
            # logger.info(f"Total reserved memory:  {torch.cuda.memory_reserved(device) / 1024 ** 3:.2f} GB")
            gc.collect()
            torch.cuda.empty_cache()

def large_tensor_nonzero(tensor, chunk_size=2**30):
    indices = []
    num_chunks = (tensor.numel() + chunk_size - 1) // chunk_size
    for i in range(num_chunks):
        chunk = tensor.flatten()[i * chunk_size:(i + 1) * chunk_size]
        chunk_indices = chunk.nonzero(as_tuple=True)[0] + i * chunk_size
        indices.append(chunk_indices)
    indices = torch.cat(indices)
    row_indices = indices // tensor.size(1)
    col_indices = indices % tensor.size(1)
    return torch.stack([row_indices, col_indices])

def get_equidistant_points(n_points=1024):
    indices = np.arange(0, n_points, dtype=float) + 0.5
    r = np.sqrt(indices / n_points)
    theta = np.pi * (1 + 5 ** 0.5) * indices
    x, y = r * np.cos(theta), r * np.sin(theta)

    return x, y

def get_matrix_rank(matrix):
    return np.linalg.matrix_rank(matrix)


def map_matrix(neuron_list, neuron_names, matrix):

    map_list = np.zeros(len(neuron_list), dtype=int)
    for i, neuron_name in enumerate(neuron_list):
        if neuron_name in list(neuron_names):
            index = list(neuron_names).index(neuron_name)
            map_list[i] = index
        else:
            map_list[i] = 0

    mapped_matrix = matrix[np.ix_(map_list, map_list)]

    for i, neuron_name in enumerate(neuron_list):
        if neuron_name not in list(neuron_names):
            mapped_matrix[i, :] = 0
            mapped_matrix[:, i] = 0

    return mapped_matrix, map_list


def get_neuron_index(neuron_name, activity_neuron_list):
    """
    Returns the index of the neuron_name in activity_neuron_list.
    Raises ValueError if not found.
    """
    try:
        return activity_neuron_list.index(neuron_name)
    except ValueError:
        raise ValueError(f"Neuron '{neuron_name}' not found in activity_neuron_list.")


def compute_spectral_density(matrix, bins=100):
    # Compute eigenvalues
    eigenvalues = np.linalg.eigvals(matrix)

    # Create histogram
    density, edges = np.histogram(eigenvalues, bins=bins, density=True)

    # Compute bin centers
    centers = (edges[:-1] + edges[1:]) / 2

    return centers, density


def get_2d_bounding_box(xx):

    x_min, y_min = torch.min(xx, dim=0).values
    x_max, y_max = torch.max(xx, dim=0).values

    bounding_box = {
        'x_min': x_min.item(),
        'x_max': x_max.item(),
        'y_min': y_min.item(),
        'y_max': y_max.item()
    }

    return bounding_box


def get_3d_bounding_box(xx):

    x_min, y_min, z_min = torch.min(xx, dim=0).values
    x_max, y_max, z_max = torch.max(xx, dim=0).values

    bounding_box = {
        'x_min': x_min.item(),
        'x_max': x_max.item(),
        'y_min': y_min.item(),
        'y_max': y_max.item(),
        'z_min': z_min.item(),
        'z_max': z_max.item()
    }

    return bounding_box


def get_top_fft_modes_per_pixel(im0, dt=1.0, top_n=3):
    """
    Compute the top N Fourier modes for each pixel and channel in a 4D time series image stack.

    Parameters:
        im0 (ndarray): shape (T, H, W, C)
        dt (float): time step between frames
        top_n (int): number of top frequency modes to return

    Returns:
        top_freqs (ndarray): shape (top_n, H, W, C), top frequencies per pixel/channel
        top_amps (ndarray): shape (top_n, H, W, C), corresponding amplitudes
    """
    T, H, W, C = im0.shape

    # Compute FFT frequencies
    freqs = np.fft.fftfreq(T, d=dt)
    pos_mask = freqs > 0
    pos_freqs = freqs[pos_mask]

    # Compute FFT along time axis
    fft_vals = np.fft.fft(im0, axis=0)              # shape: (T, H, W, C)
    fft_mag = np.abs(fft_vals)[pos_mask, :, :, :]   # shape: (T_pos, H, W, C)

    # Get indices of top N frequencies per pixel/channel
    top_indices = np.argsort(fft_mag, axis=0)[-top_n:][::-1]  # shape: (top_n, H, W, C)

    # Gather top frequencies and amplitudes
    top_freqs = pos_freqs[top_indices]      # broadcast: top_n x H x W x C
    top_amps = np.take_along_axis(fft_mag, top_indices, axis=0)

    return top_freqs, top_amps


def total_variation_norm(im):
    # Compute the differences along the x-axis (horizontal direction)
    dx = im[:, 1:, :] - im[:, :-1, :]  # (batch, height-1, width, channels)

    # Compute the differences along the y-axis (vertical direction)
    dy = im[1:, :, :] - im[:-1, :, :]  # (batch, height, width-1, channels)

    # Sum squared differences and take the square root (L2 norm)
    tv_x = torch.sqrt(torch.sum(dx ** 2))  # Sum along channels
    tv_y = torch.sqrt(torch.sum(dy ** 2))  # Sum along channels

    # Total variation is the sum of x and y contributions
    return tv_x + tv_y


def check_file_exists(dataset_name):
    file_path = f'graphs_data/graphs_{dataset_name}/connection_matrix_list.pt'
    return os.path.isfile(file_path)


def find_suffix_pairs_with_index(neuron_list, suffix1, suffix2):
    pairs = []
    for i, neuron in enumerate(neuron_list):
        if neuron.endswith(suffix1):
            base_name = neuron[:-1]
            target_name = base_name + suffix2
            for j, other_neuron in enumerate(neuron_list):
                if other_neuron == target_name:
                    pairs.append(((i, neuron), (j, other_neuron)))
                    break  # Stop after finding the first match
    return pairs

def fit_polynomial(x, y, degree=3):
    """
    Fit a polynomial of given degree to data x, y.

    Parameters:
        x: numpy array or tensor of shape (N,) or (N, 1)
        y: numpy array or tensor of shape (N,) or (N, 1)
        degree: int, degree of polynomial to fit (default=3)

    Returns:
        poly: numpy.poly1d object representing the fitted polynomial
    """
    # Convert to 1D numpy arrays
    x_np = np.ravel(x)
    y_np = np.ravel(y)

    # Fit polynomial
    coeffs = np.polyfit(x_np, y_np, deg=degree)

    # Create polynomial object
    poly = np.poly1d(coeffs)

    return poly

def fit_polynomial_with_latex(x, y, degree=3):
    """
    Fit a polynomial of given degree to data x, y,
    and return a LaTeX formatted string of the polynomial.

    Parameters:
        x: numpy array or tensor of shape (N,) or (N, 1)
        y: numpy array or tensor of shape (N,) or (N, 1)
        degree: int, degree of polynomial to fit (default=3)

    Returns:
        poly: numpy.poly1d object representing the fitted polynomial
        latex_str: str, LaTeX formatted polynomial expression
    """
    x_np = np.ravel(x)
    y_np = np.ravel(y)

    coeffs = np.polyfit(x_np, y_np, deg=degree)
    poly = np.poly1d(coeffs)

    # Build LaTeX string
    terms = []
    for power, coeff in enumerate(coeffs[::-1]):
        # power = 0 means constant term
        # coeff for the current power
        if abs(coeff) < 1e-12:
            continue  # skip negligible coefficients

        coeff_str = f"{abs(coeff):.4g}"
        if power == 0:
            term = f"{coeff_str}"
        elif power == 1:
            term = f"{coeff_str} x"
        else:
            term = f"{coeff_str} x^{power}"

        # Add sign
        sign = "-" if coeff < 0 else "+"
        terms.append((sign, term))

    # First term sign handling
    if terms:
        first_sign, first_term = terms[0]
        if first_sign == "+":
            latex_expr = first_term
        else:
            latex_expr = first_sign + " " + first_term

        for sign, term in terms[1:]:
            latex_expr += f" {sign} {term}"
    else:
        latex_expr = "0"

    latex_str = f"${latex_expr}$"

    return poly, latex_str

def reconstruct_time_series_from_xlist(x_list):
    """
    Reconstruct time series data from x_list with proper track ID mapping.

    Parameters:
    x_list: List of arrays, each containing frame data with columns:
            [track_ID, y_pos, x_pos, vel_y, vel_x, frame, fluo, fluo_unused1, fluo_unused2]

    Returns:
    time_series_dict: Dictionary {track_ID: [[frame, fluo_value], ...]}
    track_info_dict: Dictionary {track_ID: {'positions': [[y,x], ...], 'frames': [...]}}
    """

    time_series_dict = defaultdict(list)
    track_info_dict = defaultdict(lambda: {'positions': [], 'frames': []})

    for frame_idx, x_frame in enumerate(x_list):
        if x_frame.size == 0:  # Skip empty frames
            continue

        for i in range(x_frame.shape[0]):
            track_id = int(x_frame[i, 0])

            # Skip invalid track IDs
            if track_id < 0:
                continue

            frame_number = int(x_frame[i, 5])
            fluo_value = x_frame[i, 6]  # Only column 6 contains actual fluorescence data
            y_pos = x_frame[i, 1]
            x_pos = x_frame[i, 2]

            # Append to time series
            time_series_dict[track_id].append([frame_number, fluo_value])

            # Store position and frame info
            track_info_dict[track_id]['positions'].append([y_pos, x_pos])
            track_info_dict[track_id]['frames'].append(frame_number)

    time_series_dict = dict(time_series_dict)
    track_info_dict = dict(track_info_dict)

    for track_id in time_series_dict:
        # Sort by frame number
        time_series_dict[track_id] = sorted(time_series_dict[track_id], key=lambda x: x[0])

        # Convert to numpy arrays for easier handling
        time_series_dict[track_id] = np.array(time_series_dict[track_id])
        track_info_dict[track_id]['positions'] = np.array(track_info_dict[track_id]['positions'])
        track_info_dict[track_id]['frames'] = np.array(track_info_dict[track_id]['frames'])

    print(f'found {len(time_series_dict)} time_series')

    return time_series_dict, track_info_dict


def filter_tracks_by_length(time_series_dict, min_length=100, required_frame=None):
    """
    Filter tracks to keep only those with sufficient length for Granger analysis.

    Parameters:
    time_series_dict: Dictionary from reconstruct_time_series_from_xlist
    min_length: Minimum number of time points required
    required_frame: Frame number that must be present in the track (optional)

    Returns:
    filtered_dict: Dictionary with only tracks meeting requirements
    """
    filtered_dict = {}

    for track_id, time_series in time_series_dict.items():
        # Check length requirement
        if len(time_series) < min_length:
            continue

        # Check required frame if specified
        if required_frame is not None:
            frames = time_series[:, 0]  # First column contains frame numbers
            if required_frame not in frames:
                continue

        filtered_dict[track_id] = time_series

    print(f"kept {len(filtered_dict)} tracks out of {len(time_series_dict)} total tracks")
    if required_frame is not None:
        print(f"all tracks contain frame {required_frame}")


    return filtered_dict


def find_average_spatial_neighbors(filtered_time_series, track_info_dict, max_radius=50, min_radius=0, device='cpu', save_path=None):
    """
    Find spatial nearest neighbors between filtered tracks using average positions.

    Parameters:
    filtered_time_series: Dictionary {track_id: time_series_array} - only tracks to analyze
    track_info_dict: Dictionary with track positions from reconstruct_time_series_from_xlist
    max_radius: Maximum distance for neighbors (pixels)
    min_radius: Minimum distance for neighbors (pixels)
    device: 'cpu' or 'cuda'

    Returns:
    neighbor_pairs: List of (track_id1, track_id2) tuples
    track_positions: Dictionary {track_id: [avg_y, avg_x]}
    """

    # Only use tracks that are in filtered_time_series
    track_ids = list(filtered_time_series.keys())
    positions = []

    for track_id in track_ids:
        if track_id in track_info_dict:
            pos_array = track_info_dict[track_id]['positions']
            avg_pos = np.mean(pos_array, axis=0)  # [avg_y, avg_x]
            positions.append(avg_pos)
        else:
            # This shouldn't happen but handle gracefully
            positions.append([0, 0])

    # Convert to torch tensors
    track_ids = np.array(track_ids)
    positions = torch.tensor(np.array(positions), dtype=torch.float32, device=device)

    print(f"computing distances for {len(track_ids)} filtered tracks...")

    # Your distance calculation approach (adapted)
    dimension = 2  # y, x coordinates
    distance = torch.sum((positions[:, None, :] - positions[None, :, :]) ** 2, dim=2)

    # Create adjacency matrix
    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float()

    # Get edge indices
    edge_index = adj_t.nonzero().t().contiguous()

    # Convert back to track ID pairs
    neighbor_pairs = []
    for i in range(edge_index.shape[1]):
        idx1, idx2 = edge_index[0, i].item(), edge_index[1, i].item()
        track_id1, track_id2 = track_ids[idx1], track_ids[idx2]
        neighbor_pairs.append((track_id1, track_id2))

    # Create position lookup
    track_positions = {track_ids[i]: positions[i].cpu().numpy() for i in range(len(track_ids))}

    print(f"found {len(neighbor_pairs)} neighbor pairs within radius {max_radius}")

    # Plot neighbor connections if save_path provided
    if save_path is not None:

        plt.figure(figsize=(12, 10))
        plt.axis('off')
        # Plot all track positions
        pos_array = np.array(list(track_positions.values()))
        plt.scatter(pos_array[:, 1], pos_array[:, 0], s=30, c='green', alpha=0.6, label=f'{len(track_positions)} tracks')

        # Plot connections
        for track1, track2 in neighbor_pairs:
            pos1 = track_positions[track1]
            pos2 = track_positions[track2]
            plt.plot([pos1[1], pos2[1]], [pos1[0], pos2[0]], 'gray', alpha=0.7, linewidth=0.5)

        plt.xlabel('x position')
        plt.ylabel('y position')
        plt.title(
            f'spatial neighbor connections (radius={max_radius})\n{len(neighbor_pairs)} connections between {len(track_positions)} tracks')
        # plt.legend()
        # plt.grid(True, alpha=0.3)
        plt.axis('equal')
        plt.tight_layout()

        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()

    return neighbor_pairs, track_positions


def fit_ar_model_with_bic(time_series, max_order=10):
    """
    Fit AR model with optimal order selection using BIC.

    Parameters:
    time_series: 1D array of time series values
    max_order: Maximum AR order to test

    Returns:
    best_order: Optimal AR order
    coefficients: AR coefficients for best model
    residuals: Residuals from best model
    bic_scores: BIC scores for all tested orders
    """

    if len(time_series) < max_order + 10:  # Need sufficient data
        return None, None, None, None

    bic_scores = []
    models = []

    # Test different AR orders
    for p in range(1, min(max_order + 1, len(time_series) // 3)):
        try:
            # Create lagged features
            X, y = create_ar_features(time_series, p)

            if X.shape[0] < p + 5:  # Need sufficient samples
                bic_scores.append(np.inf)
                models.append(None)
                continue

            # Fit linear regression (AR model)
            model = LinearRegression(fit_intercept=True)
            model.fit(X, y)

            # Calculate predictions and residuals
            y_pred = model.predict(X)
            residuals = y - y_pred

            # Calculate BIC
            n = len(y)
            mse = np.mean(residuals ** 2)
            log_likelihood = -0.5 * n * np.log(2 * np.pi * mse) - 0.5 * np.sum(residuals ** 2) / mse
            bic = -2 * log_likelihood + (p + 1) * np.log(n)  # +1 for intercept

            bic_scores.append(bic)
            models.append((model, residuals))

        except Exception as e:
            bic_scores.append(np.inf)
            models.append(None)

    if not any(score != np.inf for score in bic_scores):
        return None, None, None, None

    # Find best order
    best_idx = np.argmin(bic_scores)
    best_order = best_idx + 1
    best_model, best_residuals = models[best_idx]

    return best_order, best_model.coef_, best_residuals, bic_scores


def create_ar_features(time_series, order):
    """
    Create lagged features for AR model.

    Parameters:
    time_series: 1D array
    order: AR order (number of lags)

    Returns:
    X: Feature matrix [n_samples, order]
    y: Target vector [n_samples]
    """
    n = len(time_series)
    X = np.zeros((n - order, order))

    for i in range(order):
        X[:, i] = time_series[order - 1 - i:n - 1 - i]

    y = time_series[order:]

    return X, y


def fit_granger_models(ts1, ts2, max_order=10):
    """
    Fit both restricted and unrestricted models for Granger causality test.

    Parameters:
    ts1: Time series 1 (potential cause)
    ts2: Time series 2 (potential effect)
    max_order: Maximum AR order to test

    Returns:
    results: Dictionary with model results
    """

    # Ensure same length
    min_len = min(len(ts1), len(ts2))
    ts1 = ts1[:min_len]
    ts2 = ts2[:min_len]

    # Find optimal order for restricted model (ts2 only)
    best_order, _, _, _ = fit_ar_model_with_bic(ts2, max_order)

    if best_order is None:
        return None

    try:
        # Restricted model: ts2(t) = f(ts2(t-1), ts2(t-2), ..., ts2(t-p))
        X_restricted, y = create_ar_features(ts2, best_order)
        model_restricted = LinearRegression(fit_intercept=True)
        model_restricted.fit(X_restricted, y)
        residuals_restricted = y - model_restricted.predict(X_restricted)

        # Unrestricted model: ts2(t) = f(ts2(t-1), ..., ts2(t-p), ts1(t-1), ..., ts1(t-p))
        X_ts1, _ = create_ar_features(ts1, best_order)
        X_unrestricted = np.column_stack([X_restricted, X_ts1])

        model_unrestricted = LinearRegression(fit_intercept=True)
        model_unrestricted.fit(X_unrestricted, y)
        residuals_unrestricted = y - model_unrestricted.predict(X_unrestricted)

        return {
            'order': best_order,
            'residuals_restricted': residuals_restricted,
            'residuals_unrestricted': residuals_unrestricted,
            'n_samples': len(y)
        }

    except Exception as e:
        return None


def calculate_granger_causality(residuals_restricted, residuals_unrestricted):
    """
    Calculate Granger causality metric: GC = log(RSS_restricted / RSS_unrestricted)

    Parameters:
    residuals_restricted: Residuals from AR model (effect only)
    residuals_unrestricted: Residuals from VAR model (effect + cause)

    Returns:
    gc_value: Granger causality metric
    """
    rss_restricted = np.sum(residuals_restricted ** 2)
    rss_unrestricted = np.sum(residuals_unrestricted ** 2)

    if rss_unrestricted == 0:
        return 0

    gc_value = np.log(rss_restricted / rss_unrestricted)
    return gc_value


def compute_granger_difference(ts1, ts2, max_order=10):
    """
    Compute bidirectional Granger causality and difference.

    Parameters:
    ts1, ts2: Time series arrays (fluorescence values)
    max_order: Maximum AR order for BIC selection

    Returns:
    gc_12: Granger causality ts1 -> ts2
    gc_21: Granger causality ts2 -> ts1
    granger_diff: |gc_12 - gc_21|
    direction: 1 if ts1->ts2 stronger, 2 if ts2->ts1 stronger
    """

    # Test ts1 -> ts2
    result_12 = fit_granger_models(ts1, ts2, max_order)
    if result_12 is None:
        return None, None, None, None

    gc_12 = calculate_granger_causality(
        result_12['residuals_restricted'],
        result_12['residuals_unrestricted']
    )

    # Test ts2 -> ts1
    result_21 = fit_granger_models(ts2, ts1, max_order)
    if result_21 is None:
        return None, None, None, None

    gc_21 = calculate_granger_causality(
        result_21['residuals_restricted'],
        result_21['residuals_unrestricted']
    )

    # Compute difference and direction
    granger_diff = abs(gc_12 - gc_21)
    direction = 1 if gc_12 > gc_21 else 2

    return gc_12, gc_21, granger_diff, direction


def analyze_neighbor_pairs(neighbor_pairs, filtered_time_series, max_order=10):
    """
    Compute Granger causality for all neighbor pairs.

    Parameters:
    neighbor_pairs: List of (track_id1, track_id2) tuples
    filtered_time_series: Dictionary {track_id: time_series_array}
    max_order: Maximum AR order

    Returns:
    granger_results: Dictionary with results for each pair
    """

    granger_results = {}

    print(f"analyzing {len(neighbor_pairs)} neighbor pairs...")

    for i, (track1, track2) in enumerate(tqdm(neighbor_pairs, desc="processing pairs")):
        if track1 not in filtered_time_series or track2 not in filtered_time_series:
            continue

        # Extract fluorescence values (column 1)
        ts1 = filtered_time_series[track1][:, 1]
        ts2 = filtered_time_series[track2][:, 1]

        # Compute Granger causality
        gc_12, gc_21, granger_diff, direction = compute_granger_difference(ts1, ts2, max_order)

        if granger_diff is not None:
            granger_results[(track1, track2)] = {
                'gc_12': gc_12,
                'gc_21': gc_21,
                'granger_diff': granger_diff,
                'direction': direction,
                'stronger_direction': track1 if direction == 1 else track2
            }

    print(f"successfully analyzed {len(granger_results)}")
    return granger_results


def iaaft_surrogate_gpu(time_series, n_surrogates=1, max_iter=50, device='cuda'):
    """
    GPU-accelerated IAAFT: generates multiple surrogates simultaneously

    Parameters:
    time_series: 1D array or torch tensor
    n_surrogates: Number of surrogates to generate
    max_iter: Maximum iterations for IAAFT algorithm
    device: 'cuda' or 'cpu'

    Returns:
    surrogates: torch tensor [n_surrogates, n] or [n] if n_surrogates=1
    """

    # Convert to torch tensor and move to GPU
    if isinstance(time_series, np.ndarray):
        original = torch.tensor(time_series, dtype=torch.float32, device=device)
    else:
        original = time_series.to(device=device, dtype=torch.float32)
    n = len(original)

    # Sort original amplitudes
    sorted_amplitudes = torch.sort(original)[0]

    # Get original power spectrum
    fft_original = torch.fft.fft(original)
    amplitudes = torch.abs(fft_original)

    # Batch processing
    phases = torch.rand(n_surrogates, n, device=device) * 2 * torch.pi
    surrogates = original.unsqueeze(0).repeat(n_surrogates, 1)

    # Expand for batch
    amplitudes_batch = amplitudes.unsqueeze(0).repeat(n_surrogates, 1)
    sorted_amplitudes_batch = sorted_amplitudes.unsqueeze(0).repeat(n_surrogates, 1)

    for _ in range(max_iter):
        # Step 1: match power spectrum (vectorized)
        fft_surrogates = amplitudes_batch * torch.exp(1j * phases)
        surrogates = torch.real(torch.fft.ifft(fft_surrogates, dim=1))

        # Step 2: match amplitude distribution (vectorized)
        sorted_indices = torch.argsort(surrogates, dim=1)
        batch_idx = torch.arange(n_surrogates, device=device).unsqueeze(1)
        surrogates[batch_idx, sorted_indices] = sorted_amplitudes_batch

        # Update phases (vectorized)
        phases = torch.angle(torch.fft.fft(surrogates, dim=1))

    return surrogates.cpu().numpy()


def statistical_testing(granger_results, filtered_time_series, n_surrogates=500):
    """Generate surrogates and compute p-values"""
    significant_pairs = {}

    print(f"testing {len(granger_results)} pairs with {n_surrogates} surrogates...")

    for i, (pair, result) in enumerate(tqdm(granger_results.items(), desc="testing pairs")):
        track1, track2 = pair
        original_diff = result['granger_diff']
        direction = result['direction']

        # Get cause and effect time series
        ts1 = filtered_time_series[track1][:, 1]
        ts2 = filtered_time_series[track2][:, 1]
        cause_ts = ts1 if direction == 1 else ts2
        effect_ts = ts2 if direction == 1 else ts1

        all_surrogates = iaaft_surrogate_gpu(cause_ts, n_surrogates=n_surrogates)

        # Test each surrogate
        surrogate_diffs = []
        for i in range(n_surrogates):
            cause_surrogate = all_surrogates[i]
            if direction == 1:
                _, _, surr_diff, _ = compute_granger_difference(cause_surrogate, effect_ts)
            else:
                _, _, surr_diff, _ = compute_granger_difference(effect_ts, cause_surrogate)

            if surr_diff is not None:
                surrogate_diffs.append(surr_diff)

        # Calculate p-value
        if len(surrogate_diffs) > 0:
            p_value = np.mean(np.array(surrogate_diffs) > original_diff)

            if p_value < 0.05:
                significant_pairs[pair] = {
                    **result,
                    'p_value': p_value,
                    'n_surrogates': len(surrogate_diffs)
                }

        # if (i + 1) % 10 == 0:
        #     print(f"tested {i + 1}/{len(granger_results)} pairs")

    print(f"found {len(significant_pairs)} significant pairs (p < 0.05)")
    return significant_pairs


def plot_interesting_causality_pairs(significant_pairs, filtered_time_series, track_positions,
                                     network_scores, dataset_name, n_pairs=20):
    """
    Plot the most interesting leader-follower pairs based on different criteria

    Parameters:
    significant_pairs: Dictionary of significant causal pairs
    filtered_time_series: Dictionary of time series data
    track_positions: Dictionary of track positions
    network_scores: Dictionary with leader/follower scores
    dataset_name: Name of dataset for file paths
    n_pairs: Number of pairs to plot
    """

    print(f"analyzing {len(significant_pairs)} significant pairs to find interesting examples...")

    # Get leader and follower scores
    leader_scores = network_scores['leader']
    follower_scores = network_scores['follower']

    # Categorize pairs by different criteria
    interesting_pairs = []

    for (track1, track2), result in significant_pairs.items():
        # Get scores and metrics
        track1_leader = leader_scores.get(track1, 0)
        track1_follower = follower_scores.get(track1, 0)
        track2_leader = leader_scores.get(track2, 0)
        track2_follower = follower_scores.get(track2, 0)

        stronger_track = result['stronger_direction']
        granger_diff = result['granger_diff']
        p_value = result['p_value']
        gc_12 = result['gc_12']
        gc_21 = result['gc_21']

        # Determine leader and follower based on stronger direction
        if stronger_track == track1:
            leader_id = track1
            follower_id = track2
            leader_score = track1_leader
            follower_score = track2_follower
        else:
            leader_id = track2
            follower_id = track1
            leader_score = track2_leader
            follower_score = track1_follower

        # Calculate spatial distance
        pos1 = track_positions[track1]
        pos2 = track_positions[track2]
        distance = np.sqrt((pos1[0] - pos2[0]) ** 2 + (pos1[1] - pos2[1]) ** 2)

        # Create different categories of interesting pairs
        category = "other"
        interest_score = granger_diff  # Default sorting

        # Category 1: Strong leaders with high Granger causality
        if leader_score > 0.7 and granger_diff > 0.05 and p_value < 0.01:
            category = "strong_leader"
            interest_score = granger_diff * leader_score

        # Category 2: High asymmetry (very directional)
        elif granger_diff > 0.08:
            category = "high_asymmetry"
            interest_score = granger_diff

        # Category 3: Long-range causality
        elif distance > 100 and granger_diff > 0.03:
            category = "long_range"
            interest_score = granger_diff * (distance / 100)

        # Category 4: Short-range strong causality
        elif distance < 50 and granger_diff > 0.04:
            category = "short_range"
            interest_score = granger_diff * (50 / distance)

        # Category 5: Very significant (low p-value)
        elif p_value < 0.001 and granger_diff > 0.03:
            category = "very_significant"
            interest_score = granger_diff * (-np.log10(p_value + 1e-10))

        # Category 6: Bidirectional (both directions strong)
        elif min(gc_12, gc_21) > 0.02 and granger_diff < 0.05:
            category = "bidirectional"
            interest_score = min(gc_12, gc_21)

        interesting_pairs.append({
            'leader_id': leader_id,
            'follower_id': follower_id,
            'category': category,
            'interest_score': interest_score,
            'granger_diff': granger_diff,
            'p_value': p_value,
            'distance': distance,
            'leader_score': leader_score,
            'follower_score': follower_score
        })

    # Sort by interest score and select diverse examples
    interesting_pairs.sort(key=lambda x: x['interest_score'], reverse=True)

    # Select diverse pairs (max 4 per category)
    selected_pairs = []
    category_counts = defaultdict(int)

    for pair in interesting_pairs:
        if len(selected_pairs) >= n_pairs:
            break
        if category_counts[pair['category']] < 4:  # Max 4 per category
            selected_pairs.append(pair)
            category_counts[pair['category']] += 1

    # Fill remaining slots with top pairs regardless of category
    remaining_slots = n_pairs - len(selected_pairs)
    if remaining_slots > 0:
        for pair in interesting_pairs:
            if len(selected_pairs) >= n_pairs:
                break
            if pair not in selected_pairs:
                selected_pairs.append(pair)

    # Print summary of selected pairs
    print(f"\nselected {len(selected_pairs)} interesting pairs:")
    print("category distribution:")
    for category, count in category_counts.items():
        print(f"  {category}: {count}")

    print(f"\ntop examples:")
    for i, pair in enumerate(selected_pairs[:5]):
        print(f"  {i + 1}. {pair['leader_id']}→{pair['follower_id']}: "
              f"{pair['category']}, diff={pair['granger_diff']:.3f}, "
              f"dist={pair['distance']:.1f}px")

    # Plot each selected pair
    print(f"\ngenerating plots...")
    for i, pair in enumerate(selected_pairs):
        l = pair['leader_id']
        f = pair['follower_id']

        print(f"  plotting pair {i + 1}/{len(selected_pairs)}: {l}→{f} ({pair['category']})")

        plot_combined_causality_analysis(
            leader_track_id=l,
            follower_track_id=f,
            filtered_time_series=filtered_time_series,
            track_positions=track_positions,
            significant_pairs=significant_pairs,
            save_path=f'graphs_data/{dataset_name}/causality_pair_{i + 1:02d}_{l}_{f}_{pair["category"]}.png'
        )

    # Create summary report
    summary_filename = f'graphs_data/{dataset_name}/causality_pairs_summary.txt'
    with open(summary_filename, 'w') as f:
        f.write("INTERESTING CAUSALITY PAIRS ANALYSIS\n")
        f.write("=" * 50 + "\n\n")

        for i, pair in enumerate(selected_pairs):
            f.write(f"Pair {i + 1}: {pair['leader_id']} → {pair['follower_id']}\n")
            f.write(f"  Category: {pair['category']}\n")
            f.write(f"  Granger difference: {pair['granger_diff']:.4f}\n")
            f.write(f"  P-value: {pair['p_value']:.4f}\n")
            f.write(f"  Distance: {pair['distance']:.1f} pixels\n")
            f.write(f"  Leader score: {pair['leader_score']:.3f}\n")
            f.write(f"  Follower score: {pair['follower_score']:.3f}\n")
            f.write(f"  Interest score: {pair['interest_score']:.4f}\n")
            f.write("\n")

    print(f"\nsummary saved to {summary_filename}")
    print(f"all plots saved to graphs_data/{dataset_name}/causality_pair_*.png")

    return selected_pairs


def recommend_amplitude_values(
    x_list, 
    run, 
    n_frames, 
    device='cuda',
    sample_every=100,
    verbose=True
):
    """
    Analyze ground truth data and recommend amplitude values for perturbation mode.
    
    For positive amplitude mode:
        F = identity + amplitude * tanh(network_output)
        C = amplitude * tanh(network_output)
        Jp = 1.0 + amplitude * tanh(network_output)
    
    Returns recommended amplitude values based on typical deviations from reference.
    """
    
    results = {}
    
    # ============================================================================
    # ANALYZE F
    # ============================================================================
    if verbose:
        print("\n" + "="*80)
        print("=== Analyzing F (Deformation Gradient) ===")
        print("="*80)
    
    all_F_gt = []
    for k in range(0, n_frames, sample_every):
        x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
        F_gt = x[:, 9:13].reshape(-1, 2, 2)
        all_F_gt.append(F_gt)
    
    all_F_gt = torch.stack(all_F_gt)
    
    # Deviation from identity
    identity = torch.eye(2, device=device)
    F_deviation = all_F_gt - identity.unsqueeze(0).unsqueeze(0)
    F_dev_abs_max = F_deviation.abs().max().item()
    F_dev_norm_mean = torch.norm(F_deviation.reshape(-1, 4), dim=1).mean().item()
    F_dev_norm_max = torch.norm(F_deviation.reshape(-1, 4), dim=1).max().item()
    
    # Singular value deviation from 1.0
    U, sig, Vh = torch.linalg.svd(all_F_gt.reshape(-1, 2, 2))
    sig_dev = (sig - 1.0).abs()
    sig_dev_max = sig_dev.max().item()
    sig_dev_mean = sig_dev.mean().item()
    
    if verbose:
        print(f"\nF deviation from identity:")
        print(f"  Max |F - I|: {F_dev_abs_max:.6f}")
        print(f"  Mean ||F - I||: {F_dev_norm_mean:.6f}")
        print(f"  Max ||F - I||: {F_dev_norm_max:.6f}")
        print(f"\nSingular value deviation from 1.0:")
        print(f"  Max |sig - 1|: {sig_dev_max:.6f}")
        print(f"  Mean |sig - 1|: {sig_dev_mean:.6f}")
    
    # Recommend F amplitude
    # tanh outputs [-1, 1], so amplitude should cover max deviation
    # Add safety margin
    recommended_F_amplitude = F_dev_abs_max * 1.5  # 50% margin
    
    if verbose:
        print(f"\n  Recommended MPM_F_amplitude: {recommended_F_amplitude:.6f}")
        print(f"  (This allows F ∈ [I - {recommended_F_amplitude:.3f}, I + {recommended_F_amplitude:.3f}])")
    
    results['F'] = {
        'dev_abs_max': F_dev_abs_max,
        'dev_norm_mean': F_dev_norm_mean,
        'dev_norm_max': F_dev_norm_max,
        'sig_dev_max': sig_dev_max,
        'recommended_amplitude': recommended_F_amplitude
    }
    
    # ============================================================================
    # ANALYZE C
    # ============================================================================
    if verbose:
        print("\n" + "="*80)
        print("=== Analyzing C (Affine Velocity Gradient) ===")
        print("="*80)
    
    all_C_gt = []
    for k in range(0, n_frames, sample_every):
        x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
        C_gt = x[:, 5:9].reshape(-1, 2, 2)
        all_C_gt.append(C_gt)
    
    all_C_gt = torch.stack(all_C_gt)
    
    # C is naturally around zero (velocity gradients)
    C_abs_max = all_C_gt.abs().max().item()
    C_norm_mean = torch.norm(all_C_gt.reshape(-1, 4), dim=1).mean().item()
    C_norm_max = torch.norm(all_C_gt.reshape(-1, 4), dim=1).max().item()
    C_std = all_C_gt.std().item()
    
    if verbose:
        print(f"\nC statistics:")
        print(f"  Max |C|: {C_abs_max:.6f}")
        print(f"  Mean ||C||: {C_norm_mean:.6f}")
        print(f"  Max ||C||: {C_norm_max:.6f}")
        print(f"  Std(C): {C_std:.6f}")
    
    # Recommend C amplitude
    # Use 3*std or max, whichever is larger, with margin
    recommended_C_amplitude = max(3 * C_std, C_abs_max) * 1.5
    
    if verbose:
        print(f"\n  Recommended MPM_C_amplitude: {recommended_C_amplitude:.6f}")
        print(f"  (This allows C ∈ [-{recommended_C_amplitude:.3f}, {recommended_C_amplitude:.3f}])")
    
    results['C'] = {
        'abs_max': C_abs_max,
        'norm_mean': C_norm_mean,
        'norm_max': C_norm_max,
        'std': C_std,
        'recommended_amplitude': recommended_C_amplitude
    }
    
    # ============================================================================
    # ANALYZE Jp
    # ============================================================================
    if verbose:
        print("\n" + "="*80)
        print("=== Analyzing Jp (Plastic Deformation) ===")
        print("="*80)
    
    all_Jp_gt = []
    for k in range(0, n_frames, sample_every):
        x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
        Jp_gt = x[:, 13:14]
        all_Jp_gt.append(Jp_gt)
    
    all_Jp_gt = torch.stack(all_Jp_gt)
    
    # Deviation from 1.0 (no plastic deformation)
    Jp_deviation = (all_Jp_gt - 1.0).abs()
    Jp_dev_max = Jp_deviation.max().item()
    Jp_dev_mean = Jp_deviation.mean().item()
    Jp_std = all_Jp_gt.std().item()
    Jp_min = all_Jp_gt.min().item()
    Jp_max = all_Jp_gt.max().item()
    
    if verbose:
        print(f"\nJp deviation from 1.0:")
        print(f"  Max |Jp - 1|: {Jp_dev_max:.6f}")
        print(f"  Mean |Jp - 1|: {Jp_dev_mean:.6f}")
        print(f"  Std(Jp): {Jp_std:.6f}")
        print(f"  Range: [{Jp_min:.6f}, {Jp_max:.6f}]")
    
    # Recommend Jp amplitude
    # Use max deviation with margin
    recommended_Jp_amplitude = Jp_dev_max * 1.5
    
    if verbose:
        print(f"\n  Recommended MPM_Jp_amplitude: {recommended_Jp_amplitude:.6f}")
        print(f"  (This allows Jp ∈ [1 - {recommended_Jp_amplitude:.3f}, 1 + {recommended_Jp_amplitude:.3f}])")
        print(f"  = [{1 - recommended_Jp_amplitude:.3f}, {1 + recommended_Jp_amplitude:.3f}]")
    
    results['Jp'] = {
        'dev_max': Jp_dev_max,
        'dev_mean': Jp_dev_mean,
        'std': Jp_std,
        'min': Jp_min,
        'max': Jp_max,
        'recommended_amplitude': recommended_Jp_amplitude
    }
    
    # ============================================================================
    # S (Stress) - if available
    # ============================================================================
    # Note: Stress is typically computed, not stored. 
    # We'll provide a generic recommendation based on material properties
    if verbose:
        print("\n" + "="*80)
        print("=== S (Stress Tensor) Recommendation ===")
        print("="*80)
        print("\nNote: Stress is typically computed from F, not stored.")
        print("Generic recommendation based on typical MPM stress ranges:")
    
    # Generic stress amplitude (can be refined if you have stress data)
    recommended_S_amplitude = 100.0  # Typical stress magnitude in Pa
    
    if verbose:
        print(f"  Recommended MPM_S_amplitude: {recommended_S_amplitude:.6f}")
        print(f"  (Adjust based on your material stiffness: mu, lambda)")
    
    results['S'] = {
        'recommended_amplitude': recommended_S_amplitude
    }
    
    # ============================================================================
    # SUMMARY
    # ============================================================================
    if verbose:
        print("\n" + "="*80)
        print("=== RECOMMENDED CONFIG ===")
        print("="*80)
        print(f"""
MPM_F_amplitude: float = {recommended_F_amplitude:.6f}
MPM_C_amplitude: float = {recommended_C_amplitude:.6f}
MPM_Jp_amplitude: float = {recommended_Jp_amplitude:.6f}
MPM_S_amplitude: float = {recommended_S_amplitude:.6f}
""")
        
        print("\nUsage in model:")
        print(f"""
# F: perturbation around identity
if self.F_amplitude > 0:
    F = self.identity + {recommended_F_amplitude:.6f} * torch.tanh(
        self.siren_F(features).reshape(-1, 2, 2))

# C: perturbation around zero
if self.C_amplitude > 0:
    C = {recommended_C_amplitude:.6f} * torch.tanh(
        self.siren_C(features).reshape(-1, 2, 2))

# Jp: perturbation around 1.0
if self.Jp_amplitude > 0:
    Jp = 1.0 + {recommended_Jp_amplitude:.6f} * torch.tanh(
        self.siren_Jp(features).reshape(-1, 1))

# S: perturbation around zero
if self.S_amplitude > 0:
    S = {recommended_S_amplitude:.6f} * torch.tanh(
        self.siren_S(features).reshape(-1, 2, 2))
""")
    
    return results


def plot_fields_movie(trainer, model, x_list, run, start_frame, n_particles, n_frames, device, 
                      log_dir, epoch, N, dataset_name, step_video=10):
    """
    Create MP4 movies for all fields (F, S, C, Jp) that are present in trainer.
    
    Args:
        trainer: dict containing model configuration (checks for 'F', 'S', 'C', 'Jp' keys)
        model: the neural network model
        x_list: list of particle data arrays
        run: current run index
        start_frame: starting frame for the movie
        n_particles: number of particles
        n_frames: total number of frames
        device: torch device
        log_dir: directory for saving videos
        epoch: current training epoch
        N: current training iteration
        dataset_name: name of the dataset
        step_video: sample every N frames (default: 10)
    """
    
    # Define all possible fields
    field_configs = {
        'F': {
            'indices': (9, 13),
            'shape': (-1, 2, 2),
            'vmin': np.sqrt(2) - 0.1,
            'vmax': np.sqrt(2) + 0.1,
            'cmap': 'coolwarm',
            'title': 'F (deformation)',
            'use_identity': True
        },
        'C': {
            'indices': (5, 9),
            'shape': (-1, 2, 2),
            'vmin': 0,
            'vmax': 80,
            'cmap': 'viridis',
            'title': 'C (Jacobian of velocity)',
            'use_identity': False
        },
        'Jp': {
            'indices': (17, 18),
            'shape': (-1,),
            'vmin': 0.75,
            'vmax': 1.25,
            'cmap': 'viridis',
            'title': 'Jp (volume deformation)',
            'use_identity': False
        }
    }
    
    with torch.no_grad():
        plt.style.use('dark_background')
        
        # Create movie for each field that's in trainer
        for field_name, config in field_configs.items():
            if field_name not in trainer:
                continue
            
            # MP4 writer setup
            fps = 30
            metadata = dict(title=f'SIREN {field_name} Field Evolution', artist='Matplotlib', 
                          comment=f'{field_name} field over time')
            writer = FFMpegWriter(fps=fps, metadata=metadata)
            
            fig = plt.figure(figsize=(15, 5))
            
            # Output path
            out_dir = f"./{log_dir}/tmp_training/siren_{field_name}"
            os.makedirs(out_dir, exist_ok=True)
            out_path = f"{out_dir}/{field_name}_field_movie_{epoch}_{N}.mp4"
            if os.path.exists(out_path):
                os.remove(out_path)
            
            # Get SIREN model for this field
            siren_model = getattr(model, f'siren_{field_name}')
            
            with writer.saving(fig, out_path, dpi=300):
                
                for k in range(start_frame, n_frames, step_video):
                    
                    # Clear the figure
                    fig.clear()
                    
                    # Load particle data for frame k
                    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)
                    pos = x[:, 1:3]  # Particle positions
                    frame_normalized = torch.tensor(k / n_frames, dtype=torch.float32, device=device)
                    
                    # Prepare features for SIREN
                    features = torch.cat([
                        pos,
                        frame_normalized.expand(n_particles, 1)
                    ], dim=1)
                    
                    # Get prediction
                    if config['use_identity']:
                        identity = torch.eye(2, device=device).unsqueeze(0).expand(n_particles, -1, -1)
                        field_pred = identity + torch.tanh(siren_model(features).reshape(*config['shape']))
                    else:
                        field_pred = siren_model(features).reshape(*config['shape'])
                    
                    # Calculate norms
                    if field_name == 'Jp':
                        field_norm_pred = field_pred.cpu().numpy()
                        field_norm_gt = x[:, config['indices'][0]].cpu().numpy()
                    else:
                        field_norm_pred = torch.norm(field_pred.view(n_particles, -1), dim=1).cpu().numpy()
                        field_gt = x[:, config['indices'][0]:config['indices'][1]].reshape(*config['shape'])
                        field_norm_gt = torch.norm(field_gt.view(n_particles, -1), dim=1).cpu().numpy()
                    
                    # Create subplots
                    ax1 = fig.add_subplot(1, 3, 1)
                    ax2 = fig.add_subplot(1, 3, 2)
                    ax3 = fig.add_subplot(1, 3, 3)
                    
                    # Plot ground truth
                    scatter1 = ax1.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), 
                                        c=field_norm_gt, s=1, cmap=config['cmap'], 
                                        vmin=config['vmin'], vmax=config['vmax'])
                    ax1.set_title(f'Ground truth {config["title"]} (Frame {k})', fontsize=10)
                    ax1.set_xlabel('X')
                    ax1.set_ylabel('Y')
                    ax1.set_aspect('equal')
                    plt.colorbar(scatter1, ax=ax1, label=f'||{field_name}||' if field_name != 'Jp' else 'Jp')
                    ax1.set_xlim([0, 1])
                    ax1.set_ylim([0, 1])
                    
                    # Plot SIREN predicted
                    scatter2 = ax2.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), 
                                        c=field_norm_pred, s=1, cmap=config['cmap'],
                                        vmin=config['vmin'], vmax=config['vmax'])
                    ax2.set_title(f'SIREN predicted {config["title"]} (Frame {k})', fontsize=10)
                    ax2.set_xlabel('X')
                    ax2.set_ylabel('Y')
                    ax2.set_aspect('equal')

                    # text min max and mean
                    min_val = field_norm_pred.min()
                    max_val = field_norm_pred.max()
                    mean_val = field_norm_pred.mean()
                    ax2.text(0.05, 0.95, f'min: {min_val:.3f}\nmax: {max_val:.3f}\nmean: {mean_val:.3f}',   transform=ax2.transAxes, fontsize=8, verticalalignment='top')


                    plt.colorbar(scatter2, ax=ax2, label=f'||{field_name}||' if field_name != 'Jp' else 'Jp')
                    ax2.set_xlim([0, 1])
                    ax2.set_ylim([0, 1])
                    
                    # Plot error
                    field_error = np.abs(field_norm_pred - field_norm_gt)
                    scatter3 = ax3.scatter(to_numpy(pos[:, 0]), to_numpy(pos[:, 1]), 
                                        c=field_error, s=1, cmap='viridis',
                                        vmin=-0.01, vmax=0.01)
                    ax3.set_title(f'Absolute error (Frame {k})', fontsize=10)
                    ax3.set_xlabel('X')
                    ax3.set_ylabel('Y')
                    ax3.set_aspect('equal')
                    plt.colorbar(scatter3, ax=ax3, label='|error|')
                    ax3.set_xlim([0, 1])
                    ax3.set_ylim([0, 1])
                    
                    # Add training info
                    fig.suptitle(f'Epoch {epoch}, Iter {N} - Training SIREN {field_name} Field', fontsize=12)
                    
                    plt.tight_layout()
                    
                    # Write frame to video
                    writer.grab_frame()
            
            plt.close(fig)
