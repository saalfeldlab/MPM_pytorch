import matplotlib.pyplot as plt
import torch

from matplotlib.ticker import FormatStrFormatter
from MPM_pytorch.models import *
from MPM_pytorch.utils import *

from run_MPM import *
import matplotlib as mpl
import networkx as nx
from torch_geometric.utils.convert import to_networkx
import warnings
import numpy as np
import time
import tqdm
from tifffile import imread, imwrite as imsave

import torch.nn as nn
import torch.nn.functional as F
import seaborn as sns
from scipy.optimize import curve_fit
import json

class KoLeoLoss(nn.Module):
    """Kozachenko-Leonenko entropic loss regularizer from Sablayrolles et al. - 2018 - Spreading vectors for similarity search"""

    # Copyright (c) Meta Platforms, Inc. and affiliates.
    #
    # This source code is licensed under the Apache License, Version 2.0
    # found in the LICENSE file in the root directory of this source tree.

    def __init__(self):
        super().__init__()
        self.pdist = nn.PairwiseDistance(2, eps=1e-8)

    def pairwise_NNs_inner(self, x):
        """
        Pairwise nearest neighbors for L2-normalized vectors.
        Uses Torch rather than Faiss to remain on GPU.
        """
        # parwise dot products (= inverse distance)
        dots = torch.mm(x, x.t())
        n = x.shape[0]
        dots.view(-1)[:: (n + 1)].fill_(-1)  # Trick to fill diagonal with -1
        # max inner prod -> min distance
        _, I = torch.max(dots, dim=1)  # noqa: E741
        return I

    def forward(self, student_output, eps=1e-8):
        """
        Args:
            student_output (BxD): backbone output of student
        """
        with torch.cuda.amp.autocast(enabled=False):

            # student_output = F.normalize(student_output, eps=eps, p=2, dim=0)
            I = self.pairwise_NNs_inner(student_output)  # noqa: E741
            distances = self.pdist(student_output, student_output[I])  # BxD, BxD -> B
            loss = -torch.log(distances + eps).mean()

        return loss

def linear_model(x, a, b):
    return a * x + b

def get_embedding(model_a=None, dataset_number = 0):
    embedding = []
    embedding.append(model_a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    return embedding

def get_embedding_time_series(model=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):
    embedding = []
    embedding.append(model.a[dataset_number])
    embedding = to_numpy(torch.stack(embedding).squeeze())

    indexes = np.arange(n_frames) * n_particles + cell_id

    return embedding[indexes]

def get_type_time_series(new_labels=None, dataset_number=None, cell_id=None, n_particles=None, n_frames=None, has_cell_division=None):

    indexes = np.arange(n_frames) * n_particles + cell_id

    return new_labels[indexes]

def get_in_features_update(rr=None, model=None, embedding = None, device=None):

    n_neurons = model.n_neurons
    model_update_type = model.update_type

    if embedding == None:
        embedding = model.a[0:n_neurons]
        if model.embedding_trial:
            embedding = torch.cat((embedding, model.b[0].repeat(n_neurons, 1)), dim=1)


    if rr == None:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.zeros((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device),
                    torch.zeros((n_neurons, model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    torch.zeros((n_neurons, 1), device=device),
                    embedding,
                    torch.ones((n_neurons, 1), device=device),
                    torch.ones((n_neurons, 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((torch.zeros((n_neurons, 1), device=device), embedding), dim=1)
    else:
        if 'generic' in model_update_type:
            if 'excitation' in model_update_type:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.zeros((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.zeros((rr.shape[0], model.excitation_dim), device=device)
                ), dim=1)
            else:
                in_features = torch.cat((
                    rr,
                    embedding,
                    torch.ones((rr.shape[0], 1), device=device),
                    torch.ones((rr.shape[0], 1), device=device)
                ), dim=1)
        else:
            in_features = torch.cat((rr, embedding), dim=1)

    return in_features

def get_in_features_lin_edge(x, model, model_config, xnorm, n_neurons, device):

    if model_config.signal_model_name in ['PDE_N4', 'PDE_N7']:
        in_features = torch.cat((x[:n_neurons, 6:7], model.a[:n_neurons]), dim=1)
        in_features_next = torch.cat((x[:n_neurons, 6:7] + xnorm / 150, model.a[:n_neurons]), dim=1)
        if model.embedding_trial:
            in_features_prev = torch.cat((in_features_prev, model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features = torch.cat((in_features, model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((in_features_next, model.b[0].repeat(n_neurons, 1)), dim=1)
    elif model_config.signal_model_name == 'PDE_N5':
        if model.embedding_trial:
            in_features = torch.cat((x[:n_neurons, 6:7], model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[:n_neurons], model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7] + xnorm / 150, model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[:n_neurons], model.b[0].repeat(n_neurons, 1)), dim=1)
        else:
            in_features = torch.cat((x[:n_neurons, 6:7], model.a[:n_neurons], model.a[:n_neurons]), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7] + xnorm / 150, model.a[:n_neurons], model.a[:n_neurons]), dim=1)
    elif (model_config.signal_model_name == 'PDE_N9_A') | (model_config.signal_model_name == 'PDE_N9_C') :
        in_features = torch.cat((x[:, 3:4], model.a), dim=1)
        in_features_next = torch.cat((x[:,3:4] * 1.05, model.a), dim=1)
    elif model_config.signal_model_name == 'PDE_N9_B':
        perm_indices = torch.randperm(n_neurons, device=model.a.device)
        in_features = torch.cat((x[:, 3:4], x[:, 3:4], model.a, model.a[perm_indices]), dim=1)
        in_features_next = torch.cat((x[:, 3:4], x[:, 3:4] * 1.05, model.a, model.a[perm_indices]), dim=1)
    elif model_config.signal_model_name == 'PDE_N8':
        if model.embedding_trial:
            perm_indices = torch.randperm(n_neurons, device=model.a.device)
            in_features = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7], model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[perm_indices[:n_neurons]], model.b[0].repeat(n_neurons, 1)), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7]*1.05, model.a[:n_neurons], model.b[0].repeat(n_neurons, 1), model.a[perm_indices[:n_neurons]], model.b[0].repeat(n_neurons, 1)), dim=1)
        else:
            perm_indices = torch.randperm(n_neurons, device=model.a.device)
            in_features = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7], model.a[:n_neurons], model.a[perm_indices[:n_neurons]]), dim=1)
            in_features_next = torch.cat((x[:n_neurons, 6:7], x[:n_neurons, 6:7] * 1.05, model.a[:n_neurons], model.a[perm_indices[:n_neurons]]), dim=1)
    else:
        in_features = x[:n_neurons, 6:7]
        in_features_next = x[:n_neurons, 6:7] + xnorm / 150
        in_features_prev = x[:n_neurons, 6:7] - xnorm / 150

    return in_features, in_features_next

def get_in_features(rr=None, embedding=None, model=[], model_name = [], max_radius=[]):

    if model.embedding_trial:
        embedding = torch.cat((embedding, model.b[0].repeat(embedding.shape[0], 1)), dim=1)

    match model_name:
        case 'PDE_A' | 'PDE_Cell_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_ParticleField_A':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding), dim=1)
        case 'PDE_A_bis':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_B' | 'PDE_Cell_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_ParticleField_B':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_GS':
            in_features = torch.cat(
                (rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, 10 ** embedding), dim=1)
        case 'PDE_G':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, 0 * rr[:, None],
                                     0 * rr[:, None],
                                     0 * rr[:, None], 0 * rr[:, None], embedding), dim=1)
        case 'PDE_E':
            in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                     rr[:, None] / max_radius, embedding, embedding), dim=1)
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N6' :
            in_features = rr[:, None]
        case 'PDE_N4' | 'PDE_N7':
            in_features = torch.cat((rr[:, None], embedding), dim=1)
        case 'PDE_N8':
            in_features = torch.cat((rr[:, None]*0, rr[:, None], embedding, embedding), dim=1)
        case 'PDE_N5':
            in_features = torch.cat((rr[:, None], embedding, embedding), dim=1)
        case 'PDE_K':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius), dim=1)
        case 'PDE_F':
            in_features = torch.cat((0 * rr[:, None], rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)
        case 'PDE_M':
            in_features = torch.cat((rr[:, None] / max_radius, rr[:, None] / max_radius, embedding, embedding), dim=-1)

    return in_features


def plot_training_flyvis(model, config, epoch, N, log_dir, device, cmap, type_list,
                         gt_weights, n_neurons=None, n_neuron_types=None):

    if n_neurons is None:
        n_neurons = len(type_list)

    # Plot 1: Embedding scatter plot
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_neuron_types):
        pos = torch.argwhere(type_list == n)
        plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=5, color=cmap.color(n), alpha=0.1, edgecolors='none')
    plt.xlabel('embedding 0', fontsize=18)
    plt.ylabel('embedding 1', fontsize=18)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    x_data = to_numpy(gt_weights)
    y_data = to_numpy(model.W.squeeze())
    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
    residuals = y_data - linear_model(x_data, *lin_fit)
    ss_res = np.sum(residuals ** 2)
    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
    r_squared = 1 - (ss_res / ss_tot)
    # print(f'R^2$: {np.round(r_squared, 3)}  slope: {np.round(lin_fit[0], 2)}')


    # Plot 2: Weight comparison scatter plot
    fig = plt.figure(figsize=(8, 8))
    plt.scatter(to_numpy(gt_weights), to_numpy(model.W.squeeze()), s=0.1, c='k', alpha=0.01)
    plt.xlabel(r'true $W_{ij}$', fontsize=18)
    plt.ylabel(r'learned $W_{ij}$', fontsize=18)
    plt.text(-0.19, 0.65, f'R^2: {np.round(r_squared, 3)}\nslope: {np.round(lin_fit[0], 2)}', fontsize=12)
    plt.xlim([-0.2, 0.2])
    plt.ylim([-1.25, 1.25])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif",
                dpi=87, bbox_inches='tight', pad_inches=0)
    plt.close()

    # Plot 3: Edge function visualization
    fig = plt.figure(figsize=(8, 8))
    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if ('PDE_N9_A' in config.graph_model.signal_model_name) | ('PDE_N9_C' in config.graph_model.signal_model_name):
            in_features = torch.cat((rr[:, None], embedding_,), dim=1)
        elif ('PDE_N9_B' in config.graph_model.signal_model_name):
            in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        if config.graph_model.lin_edge_positive:
            func = func ** 2
        if (n % 20 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
    plt.close()

    # Plot 4: Phi function visualization
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if ('PDE_N9_C' in config.graph_model.signal_model_name):
            in_features = torch.cat((rr[:, None], embedding_, torch.zeros_like(rr[:, None])), dim=1)
        else:
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, torch.zeros_like(rr[:, None])), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        if (n % 20 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_phi/func_{epoch}_{N}.tif", dpi=87)
    plt.close()


# Example usage:
# plot_training_flyvis(model, config, epoch, N, log_dir, device, cmap,
#                     type_list, gt_weights, to_numpy, fig_init)

def plot_training_signal(config, model, x, adjacency, log_dir, epoch, N, n_neurons, type_list, cmap, device):

    if 'PDE_N3' in config.graph_model.signal_model_name:

        fig, ax = fig_init()
        plt.scatter(to_numpy(model.a[:-200, 0]), to_numpy(model.a[:-200, 1]), s=1, color='k', alpha=0.1, edgecolor='none')

    else:
        fig = plt.figure(figsize=(8, 8))
        for n in range(n_neurons):
            if x[n, 6] != config.simulation.baseline_value:
                plt.scatter(to_numpy(model.a[n, 0]), to_numpy(model.a[n, 1]), s=100,
                            color=cmap.color(int(type_list[n])), alpha=1.0, edgecolors='none')

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    gt_weight = to_numpy(adjacency)

    if config.training.multi_connectivity:
        pred_weight = to_numpy(model.W[0, :n_neurons, :n_neurons].clone().detach())
    else:
        pred_weight = to_numpy(model.W[:n_neurons, :n_neurons].clone().detach())

    if n_neurons<1000:

        # fig = plt.figure(figsize=(8, 8))
        # plt.scatter(gt_weight, pred_weight, s=10, c='k',alpha=0.2)
        # plt.xticks([])
        # plt.yticks([])
        # plt.tight_layout()
        # plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif", dpi=87)
        # plt.close()

        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(pred_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        ax = fig.add_subplot(122)
        ax = sns.heatmap(gt_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        plt.xticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.yticks([0, n_neurons - 1], [1, n_neurons], fontsize=8)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/matrix_{epoch}_{N}.tif", dpi=87)
        plt.close()
    else:
        fig = plt.figure(figsize=(8, 8))
        fig, ax = fig_init()
        plt.scatter(gt_weight, pred_weight / 10, s=0.1, c='k', alpha=0.1)
        plt.xlabel(r'true $W_{ij}$', fontsize=68)
        plt.ylabel(r'learned $W_{ij}$', fontsize=68)
        if n_neurons == 8000:
            plt.xlim([-0.05, 0.05])
        else:
            plt.xlim([-0.2, 0.2])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/comparison_{epoch}_{N}.tif", dpi=87)
        plt.close()

    if ('PDE_N8' in config.graph_model.signal_model_name):
        dataset = config.dataset
        os.makedirs(f"./{log_dir}/tmp_training/matrix/larynx", exist_ok=True)
        data_folder_name = f'./graphs_data/{config.dataset}/'
        with open(data_folder_name+"all_neuron_list.json", "r") as f:
            all_neuron_list = json.load(f)
        with open(data_folder_name+"larynx_neuron_list.json", "r") as f:
            larynx_neuron_list = json.load(f)
        larynx_pred_weight, index_larynx = map_matrix(larynx_neuron_list, all_neuron_list, pred_weight)
        larynx_gt_weight, _ = map_matrix(larynx_neuron_list, all_neuron_list, gt_weight)
        fig = plt.figure(figsize=(16, 8))
        ax = fig.add_subplot(121)
        ax = sns.heatmap(larynx_pred_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.set_xticks(range(len(larynx_neuron_list)))
        ax.set_xticklabels(larynx_neuron_list, fontsize=12, rotation=90)
        ax.set_yticks(range(len(larynx_neuron_list)))
        ax.set_yticklabels(larynx_neuron_list, fontsize=12)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        ax = fig.add_subplot(122)
        ax = sns.heatmap(larynx_gt_weight, center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
        ax.set_xticks(range(len(larynx_neuron_list)))
        ax.set_xticklabels(larynx_neuron_list, fontsize=12, rotation=90)
        ax.set_yticks(range(len(larynx_neuron_list)))
        ax.set_yticklabels(larynx_neuron_list, fontsize=12)
        plt.ylabel('postsynaptic')
        plt.xlabel('presynaptic')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/matrix/larynx/matrix_{epoch}_{N}.tif", dpi=87)
        plt.close()

        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
        fig = plt.figure(figsize=(8, 8))
        for idx, k in enumerate(np.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 10)):  # Corrected step size to generate 13 evenly spaced values
            for n in range(0, n_neurons, 4):
                embedding_i = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                embedding_j = model.a[np.random.randint(n_neurons), :] * torch.ones(
                    (1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat((torch.ones_like(rr[:, None]) * k, rr[:, None], embedding_i, embedding_j,model.b[0].repeat(1000, 1)), dim=1)
                else:
                    in_features = torch.cat((rr[:, None], torch.ones_like(rr[:, None]) * k, embedding_i, embedding_j), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                if config.graph_model.lin_edge_positive:
                    func = func ** 2
                plt.plot(to_numpy(rr - k), to_numpy(func), 2, color=cmap.color(idx), linewidth=2, alpha=0.25)
        plt.xlabel(r'$x_i-x_j$', fontsize=18)
        # plt.ylabel(r'learned $\psi^*(a_i, x_i)$', fontsize=68)
        plt.ylabel(r'$MLP_1(a_i, a_j, x_i, x_j)$', fontsize=18)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    else:

        fig = plt.figure(figsize=(8, 8))
        rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
        for n in range(n_neurons):
            if ('PDE_N4' in config.graph_model.signal_model_name) | ('PDE_N7' in config.graph_model.signal_model_name):
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    embedding_ = torch.cat((embedding_, model.b[0].repeat(1000, 1)), dim=1)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
            elif 'PDE_N5' in config.graph_model.signal_model_name:
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat(
                        (rr[:, None], embedding_, model.b[0].repeat(1000, 1), embedding_, model.b[0].repeat(1000, 1)),
                        dim=1)
                else:
                    in_features = torch.cat((rr[:, None], embedding_, embedding_), dim=1)
            elif ('PDE_N8' in config.graph_model.signal_model_name):
                embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
                if model.embedding_trial:
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, model.b[0].repeat(1000, 1),
                                             embedding_, model.b[0].repeat(1000, 1)), dim=1)
                else:
                    in_features = torch.cat((rr[:, None] * 0, rr[:, None], embedding_, embedding_), dim=1)
            else:
                in_features = rr[:, None]
            with torch.no_grad():
                func = model.lin_edge(in_features.float())
            if config.graph_model.lin_edge_positive:
                func = func ** 2
            if (n % 2 == 0):
                plt.plot(to_numpy(rr), to_numpy(func), 2, color=cmap.color(to_numpy(type_list)[n].astype(int)),
                         linewidth=2, alpha=0.25)
        plt.xlim(config.plotting.xlim)
        plt.ylim(config.plotting.ylim)
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
        plt.close()

    rr = torch.linspace(config.plotting.xlim[0], config.plotting.xlim[1], 1000, device=device)
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_neurons):
        embedding_ = model.a[n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if 'generic' in config.graph_model.update_type:
            in_features = torch.cat((rr[:, None], embedding_, rr[:, None] * 0, rr[:, None] * 0), dim=1)
        else:
            in_features = torch.cat((rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_phi(in_features.float())
        if (n % 2 == 0):
            plt.plot(to_numpy(rr), to_numpy(func), 2,
                     color=cmap.color(to_numpy(type_list)[n].astype(int)),
                     linewidth=1, alpha=0.1)
    plt.xlim(config.plotting.xlim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_phi/func_{epoch}_{N}.tif", dpi=87)
    plt.close()



def plot_training_signal_field(x, n_nodes, recursive_loop, kk, time_step, x_list, run, model, field_type, model_f, edges, y_list, ynorm, delta_t, n_frames, log_dir, epoch, N, recursive_parameters, modulation, device):
    if recursive_loop > 1:
        x = torch.tensor(x_list[run][kk], device=device).clone().detach()
        ids = np.arange(kk, kk + recursive_loop * time_step, time_step)
        true_activity_list = np.transpose(x_list[run][ids.astype(int), :, 6:7].squeeze())
        true_modulation_list = np.transpose(x_list[run][ids.astype(int), :, 8:9].squeeze())
        loss = 0
        pred_activity_list = list([])
        pred_modulation_list = list([])
        for loop in range(recursive_loop):
            pred_activity_list.append(x[:, 6:7].clone().detach())
            if (loop == 0) & ('learnable_short_term_plasticity' in field_type):
                alpha = (kk % model.embedding_step) / model.embedding_step
                x[:, 8] = alpha * model.b[:, kk // model.embedding_step + 1] ** 2 + (1 - alpha) * model.b[:, kk // model.embedding_step] ** 2
            elif ('short_term_plasticity' in field_type):
                t = torch.zeros((1, 1, 1), dtype=torch.float32, device=device)
                t[:, 0, :] = torch.tensor(kk / n_frames, dtype=torch.float32, device=device)
                x[:, 8] = model_f(t.clone().detach()) ** 2
            pred_modulation_list.append(x[:, 8:9].clone().detach())
            dataset = data.Data(x=x, edge_index=edges)
            y = torch.tensor(y_list[run][kk], device=device) / ynorm
            pred = model(dataset)
            loss = loss + (pred - y).norm(2)
            kk = kk + time_step
            if 'learnable_short_term_plasticity' in field_type:
                in_modulation = torch.cat((x[:, 6:7], x[:, 8:9]), dim=1)
                pred_modulation = model.lin_modulation(in_modulation)
                x[:, 8:9] = x[:, 8:9] + delta_t * time_step * pred_modulation
            x[:, 6:7] = x[:, 6:7] + delta_t * time_step * pred
        pred_activity_list = torch.stack(pred_activity_list).squeeze().t()
        pred_modulation_list = torch.stack(pred_modulation_list).squeeze().t()
        kk = kk - time_step * recursive_loop
        fig = plt.figure(figsize=(12, 12))
        ind_list = [10, 124, 148, 200, 250, 300]
        ax = fig.add_subplot(2, 1, 1)
        ids = np.arange(0, recursive_loop * time_step, time_step)
        for ind in ind_list:
            plt.plot(ids, true_activity_list[ind, :], c='k', alpha=0.5, linewidth=8)
            plt.plot(ids, to_numpy(pred_activity_list[ind, :]))
        plt.text(0.05, 0.95, f'k: {kk}   loss: {np.round(loss.item(), 3)}', ha='left', va='top', transform=ax.transAxes, fontsize=10)
        if 'learnable_short_term_plasticity' in field_type:
            ax = fig.add_subplot(2, 1, 2)
            for ind in ind_list:
                plt.plot(ids, true_modulation_list[ind, :], c='k', alpha=0.5, linewidth=8)
                plt.plot(ids, to_numpy(pred_modulation_list[ind, :]))
        plt.savefig(f"./{log_dir}/tmp_training/field/Field_{epoch}_{N}.tif")
        plt.close()

    if 'learnable_short_term_plasticity' in field_type:
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(to_numpy(modulation), aspect='auto')
        ax = fig.add_subplot(2, 2, 2)
        plt.imshow(to_numpy(model.b ** 2), aspect='auto')
        ax.text(0.01, 0.99, f'recursive_parameter {recursive_parameters[0]:0.3f} ', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', color='w')
        ax.text(0.01, 0.95, f'loop {recursive_loop} ', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left', color='w')
        ax = fig.add_subplot(2, 2, 3)
        plt.scatter(to_numpy(modulation[:, np.arange(0, n_frames, n_frames//1000)]), to_numpy(model.b[:, 0:1000] ** 2), s=0.1, color='k', alpha=0.01)
        x_data = to_numpy(modulation[:, np.arange(0, n_frames, n_frames//1000)]).flatten()
        y_data = to_numpy(model.b[:, 0:1000] ** 2).flatten()
        lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        residuals = y_data - linear_model(x_data, *lin_fit)
        ss_res = np.sum(residuals ** 2)
        ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        r_squared = 1 - (ss_res / ss_tot)
        ax.text(0.01, 0.99, f'$R^2$ {r_squared:0.3f}   slope {lin_fit[0]:0.3f}', transform=ax.transAxes,
                verticalalignment='top', horizontalalignment='left')
        ind_list = [10, 124, 148, 200, 250, 300]
        ax = fig.add_subplot(4, 2, 6)
        for ind in ind_list:
            plt.plot(to_numpy(modulation[ind, :]))
        ax = fig.add_subplot(4, 2, 8)
        for ind in ind_list:
            plt.plot(to_numpy(model.b[ind, :] ** 2))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
        plt.close()

    elif ('short_term_plasticity' in field_type) | ('modulation' in field_type):
        fig = plt.figure(figsize=(12, 12))
        ax = fig.add_subplot(2, 2, 1)
        plt.imshow(to_numpy(modulation), aspect='auto')
        ax = fig.add_subplot(2, 2, 2)
        if n_frames > 1000:
            t = torch.linspace(0, 1, n_frames//100, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
        prediction = model_f[0](t) ** 2
        prediction = prediction.t()
        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
        ax = fig.add_subplot(2, 2, 3)
        # if n_frames > 1000:
        #     ids = np.arange(0, n_frames, 100).astype(int)
        # else:
        #     ids = np.arange(0, n_frames, 1).astype(int)
        # plt.scatter(to_numpy(modulation[:, ids[:-1]]), to_numpy(prediction[:modulation.shape[0], :]), s=0.1, color='k', alpha=0.01)
        # x_data = to_numpy(modulation[:, ids[:-1]]).flatten()
        # y_data = to_numpy(prediction[:modulation.shape[0], :]).flatten()
        # lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
        # residuals = y_data - linear_model(x_data, *lin_fit)
        # ss_res = np.sum(residuals ** 2)
        # ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
        # r_squared = 1 - (ss_res / ss_tot)
        # ax.text(0.01, 0.99, f'$R^2$ {r_squared:0.3f}   slope {lin_fit[0]:0.3f}', transform=ax.transAxes,
        #         verticalalignment='top', horizontalalignment='left')
        # ind_list = [10, 24, 48, 120, 150, 180]
        # ax = fig.add_subplot(4, 2, 6)
        # for ind in ind_list:
        #     plt.plot(to_numpy(modulation[ind, :]))
        # ax = fig.add_subplot(4, 2, 8)
        # for ind in ind_list:
        #     plt.plot(to_numpy(prediction[ind, :]))
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
        plt.close()

    else:
        n_nodes_per_axis = int(np.sqrt(n_nodes))
        if 'visual' in field_type:
            tmp = torch.reshape(x[:n_nodes, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
        else:
            tmp = torch.reshape(x[:, 8:9], (n_nodes_per_axis, n_nodes_per_axis))
        tmp = to_numpy(torch.sqrt(tmp))
        tmp = np.rot90(tmp, k=1)
        fig = plt.figure(figsize=(12, 12))
        plt.imshow(tmp, cmap='grey')
        plt.xticks([])
        plt.yticks([])
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/field_{epoch}_{N}.tif", dpi=80)
        plt.close()

def plot_training_signal_missing_activity(n_frames, k, x_list, baseline_value, model_missing_activity, log_dir, epoch, N, device):

        if n_frames > 1000:
            t = torch.linspace(0, 1, n_frames//100, dtype=torch.float32, device=device).unsqueeze(1)
        else:
            t = torch.linspace(0, 1, n_frames, dtype=torch.float32, device=device).unsqueeze(1)
        prediction = model_missing_activity[0](t)
        prediction = prediction.t()
        fig = plt.figure(figsize=(16, 16))
        ax = fig.add_subplot(2, 2, 1)
        plt.title('neural field')
        plt.imshow(to_numpy(prediction), aspect='auto', cmap='viridis')
        ax = fig.add_subplot(2, 2, 2)
        plt.title('true activity')
        activity = torch.tensor(x_list[0][:, :, 6:7], device=device)
        activity = activity.squeeze()
        activity = activity.t()
        plt.imshow(to_numpy(activity), aspect='auto', cmap='viridis')
        plt.tight_layout()
        ax = fig.add_subplot(2, 2, 3)
        plt.title('learned missing activity')
        pos = np.argwhere(x_list[0][k][:, 6] != baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        ax = fig.add_subplot(2, 2, 4)
        plt.title('learned observed activity')
        pos = np.argwhere(x_list[0][k][:, 6] == baseline_value)
        prediction_ = prediction.clone().detach()
        prediction_[pos[:,0]]=0
        plt.imshow(to_numpy(prediction_), aspect='auto', cmap='viridis')
        plt.tight_layout()
        plt.savefig(f"./{log_dir}/tmp_training/field/missing_activity_{epoch}_{N}.tif", dpi=80)
        plt.close()

def plot_training_particle_field(config, has_siren, has_siren_time, model_f,  n_frames, model_name, log_dir, epoch, N, x, x_mesh, index_particles, n_neurons, n_neuron_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    max_radius = simulation_config.max_radius

    n_nodes = simulation_config.n_nodes
    n_nodes_per_axis = int(np.sqrt(n_nodes))

    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        ax.yaxis.set_major_formatter(FormatStrFormatter('%.2f'))
        # plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}$', fontsize=64)
        # plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}$', fontsize=64)
        plt.xticks(fontsize=32.0)
        plt.yticks(fontsize=32.0)
    else:
        plt.axis('off')
    embedding = get_embedding(model.a, dataset_num)
    if n_neuron_types > 1000:
        plt.scatter(embedding[:, 0], embedding[:, 1], c=to_numpy(x[:, 5]) / n_neurons, s=1, cmap='viridis')
    else:
        for n in range(n_neuron_types):
            plt.scatter(embedding[index_particles[n], 0],
                        embedding[index_particles[n], 1], color=cmap.color(n), s=1)  #

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{model_name}_embedding_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    fig = plt.figure(figsize=(12, 12))
    if axis:
        ax = fig.add_subplot(1, 1, 1)
        # ax.xaxis.get_major_formatter()._usetex = False
        # ax.yaxis.get_major_formatter()._usetex = False
        ax.xaxis.set_major_locator(plt.MaxNLocator(3))
        ax.yaxis.set_major_locator(plt.MaxNLocator(3))
        # plt.xlabel(r'$d_{ij}$', fontsize=64)
        # plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=64)
        plt.xticks(fontsize=32)
        plt.yticks(fontsize=32)
        plt.xlim([0, simulation_config.max_radius])
        # plt.ylim([-0.15, 0.15])
        # plt.ylim([-0.04, 0.03])
        # plt.ylim([-0.1, 0.1])
        plt.tight_layout()

    match model_config.particle_model_name:
        case 'PDE_ParticleField_A':
            rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 200)).to(device)
        case 'PDE_ParticleField_B':
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 200)).to(device)
    for n in range(n_neurons):
        embedding_ = model.a[dataset_num, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
        match model_config.particle_model_name:
            case 'PDE_ParticleField_A':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None], rr[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_ParticleField_B':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        if n % 5 == 0:
            plt.plot(to_numpy(rr),
                     to_numpy(func * ynorm),
                     linewidth=8,
                     color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
    # match model_config.particle_model_name:
    #     case 'PDE_ParticleField_A':
    #         plt.ylim([-0.04, 0.03])
    #     case 'PDE_ParticleField_B':
    #         plt.ylim([-1E-4, 1E-4])

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/{model_name}_function_{epoch}_{N}.tif", dpi=170.7)
    plt.close()

    if has_siren:
        if has_siren_time:
            # frame_list = [n_frames//4, 2*n_frames//4, 3*n_frames//4, n_frames-1]
            frame_list = [54, 58, 62, 66]
        else:
            frame_list = [0]

        for frame in frame_list:

            if has_siren_time:
                with torch.no_grad():
                    tmp = model_f(time=frame / n_frames) ** 2
            else:
                with torch.no_grad():
                    tmp = model_f() ** 2
            tmp = torch.reshape(tmp, (n_nodes_per_axis, n_nodes_per_axis))
            tmp = to_numpy(torch.sqrt(tmp))
            if has_siren_time:
                tmp= np.rot90(tmp,k=1)
            fig_ = plt.figure(figsize=(14, 12))
            axf = fig_.add_subplot(1, 1, 1)
            plt.imshow(tmp, cmap='grey')
            plt.colorbar()
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/field/{model_name}_{epoch}_{N}_{frame}.tif", dpi=80)
            plt.close()

    # else:
    #     im = to_numpy(model_field[dataset_num])
    #     im = np.reshape(im, (n_nodes_per_axis, n_nodes_per_axis))
    #     plt.imshow(im)
    #     plt.gca().invert_yaxis()
    #     plt.tight_layout()
    #     plt.savefig(f"./{log_dir}/tmp_training/field/{model_name}_field_{epoch}_{N}.tif", dpi=87)
    #     plt.close()

    # im = np.flipud(im)
    # io.imsave(f"./{log_dir}/tmp_training/field_pic_{epoch}_{N}.tif", im)

def plot_training(config, pred, gt, log_dir, epoch, N, x, index_particles, n_particles, n_particle_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting
    do_tracking = train_config.do_tracking
    max_radius = simulation_config.max_radius
    n_runs = train_config.n_runs

    matplotlib.rcParams['savefig.pad_inches'] = 0

    if n_runs == 3:

        fig = plt.figure(figsize=(24, 8))
        ax = fig.add_subplot(1, 3, 1)
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        embedding = get_embedding(model.a, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(1, 3, 3)
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0)
        embedding = get_embedding(model.a, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        plt.xticks([])
        plt.yticks([])
        ax = fig.add_subplot(1, 3, 2)
        embedding = get_embedding(model.a, 1)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0.5)
        embedding = get_embedding(model.a, 2)
        plt.scatter(embedding[:, 0], embedding[:, 1], s=5, alpha=0)
    else:
        fig = plt.figure(figsize=(8, 8))
        if do_tracking:
            embedding = to_numpy(model.a)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=1)
        elif simulation_config.state_type == 'sequence':
            embedding = to_numpy(model.a[1].squeeze())
            plt.scatter(embedding[:-200, 0], embedding[:-200, 1], color='k', s=0.1)
        else:
            embedding = get_embedding(model.a, plot_config.data_embedding)
            for n in range(n_particle_types):
                plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1], color=cmap.color(n), s=1)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif",dpi=87)
    plt.close()

    fig = plt.figure(figsize=(8, 8))
    plt.scatter(to_numpy(gt[:, 0]), to_numpy(pred[:, 0]), c='r', s=1)
    plt.scatter(to_numpy(gt[:, 1]), to_numpy(pred[:, 1]), c='g', s=1)
    plt.xlabel('true value', fontsize=14)
    plt.ylabel('pred value', fontsize=14)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/prediction/{epoch}_{N}.tif", dpi=87)
    plt.close()

    match model_config.particle_model_name:

        case 'PDE_A' | 'PDE_A_bis' | 'PDE_ParticleField_A' | 'PDE_E' | 'PDE_G':
            fig = plt.figure(figsize=(12, 12))
            if axis:
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                plt.xticks(fontsize=32)
                plt.yticks(fontsize=32)
                plt.xlim([0, simulation_config.max_radius])
                plt.tight_layout()
            rr = torch.tensor(np.linspace(0, simulation_config.max_radius, 1000)).to(device)
            for n in range(n_particles):
                if do_tracking:
                    embedding_ = model.a[n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                else:
                    embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)

                in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config.graph_model.particle_model_name,
                                              max_radius=simulation_config.max_radius)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                if (n % 5 == 0):
                    plt.plot(to_numpy(rr),
                             to_numpy(func * ynorm),
                             linewidth=2,
                             color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
            # plt.ylim(config.plotting.ylim)
            if (model_config.particle_model_name == 'PDE_G') | (model_config.particle_model_name == 'PDE_E'):
                plt.xlim([0, 0.02])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/function_{epoch}_{N}.tif", dpi=87)
            plt.close()

        case 'PDE_B' | 'PDE_ParticleField_B':
            max_radius = 0.04
            fig = plt.figure(figsize=(12, 12))
            # plt.rcParams['text.usetex'] = True
            # rc('font', **{'family': 'serif', 'serif': ['Palatino']})
            ax = fig.add_subplot(1,1,1)
            rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
            func_list = []
            for n in range(n_particles):
                if do_tracking:
                    embedding_ = model.a[n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                else:
                    embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
                in_features = get_in_features(rr, embedding_, config.graph_model.particle_model_name, max_radius)
                # in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                #                          torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                #                          0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                func_list.append(func)
                if n % 5 == 0:
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(int(n // (n_particles / n_particle_types))), linewidth=2)
            if not(do_tracking):
                plt.ylim(config.plotting.ylim)
            # plt.xlabel(r'$x_j-x_i$', fontsize=64)
            # plt.ylabel(r'$f_{ij}$', fontsize=64)
            ax.xaxis.set_major_locator(plt.MaxNLocator(3))
            ax.yaxis.set_major_locator(plt.MaxNLocator(5))
            ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
            fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
            ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
            plt.xticks(fontsize=32.0)
            plt.yticks(fontsize=32.0)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/function_{epoch}_{N}.tif",dpi=170.7)
            plt.close()

        case 'PDE_GS':
            fig = plt.figure(figsize=(8, 4))
            ax = fig.add_subplot(1, 2, 1)
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
            for n in range(n_particles):
                embedding_ = model.mass[n] * torch.ones((1000, model_config.embedding_dim), device=device)
                in_features = torch.cat((rr[:, None] / simulation_config.max_radius, 0 * rr[:, None],
                                         rr[:, None] / simulation_config.max_radius, embedding_), dim=1)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                         color=cmap.color(to_numpy(x[n, 5]).astype(int)), linewidth=1)
            plt.xlabel('Distance [a.u]', fontsize=14)
            plt.ylabel('MLP [a.u]', fontsize=14)
            plt.xscale('log')
            plt.yscale('log')
            plt.tight_layout()
            ax = fig.add_subplot(1, 2, 2)
            plt.scatter(np.log(np.abs(to_numpy(y_batch[:, 0]))), np.log(np.abs(to_numpy(pred[:, 0]))), c='k', s=1,
                        alpha=0.15)
            plt.scatter(np.log(np.abs(to_numpy(y_batch[:, 1]))), np.log(np.abs(to_numpy(pred[:, 1]))), c='k', s=1,
                        alpha=0.15)
            # plt.xlim([-10, 4])
            # plt.ylim([-10, 4])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/func_{epoch}_{N}.tif", dpi=87)
            plt.close()

        case 'PDE_K':
            fig = plt.figure(figsize=(12, 12))
            if axis:
                ax = fig.add_subplot(1, 1, 1)
                ax.xaxis.set_major_locator(plt.MaxNLocator(3))
                ax.yaxis.set_major_locator(plt.MaxNLocator(3))
                plt.xticks(fontsize=32)
                plt.yticks(fontsize=32)
                plt.xlim([0, simulation_config.max_radius])
                plt.tight_layout()
            rr = torch.tensor(np.linspace(-1, 1, 200)).to(device)
            for n in range(n_particles):
                if do_tracking:
                    embedding_ = model.a[n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                else:
                    embedding_ = model.a[1, n, :] * torch.ones((200, model_config.embedding_dim), device=device)
                in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config.graph_model.particle_model_name,
                                              max_radius=simulation_config.max_radius)
                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                if (n % 5 == 0) :
                    plt.plot(to_numpy(rr),
                             to_numpy(func*ynorm),
                             linewidth=2,
                             color=cmap.color(to_numpy(x[n, 5]).astype(int)), alpha=0.25)
            # if not (do_tracking):
            #     plt.ylim(config.plotting.ylim)
            plt.ylim(config.plotting.ylim)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/function_{epoch}_{N}.tif", dpi=87)
            plt.close()

            if len(model.connection_matrix)>5:
                i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
                fig = plt.figure(figsize=(9, 15))
                for n in range(5):

                    A = torch.zeros(n_particles, n_particles, device=device, requires_grad=False, dtype=torch.float32)
                    A[i, j] = model.vals[n+1] ** 2
                    A.T[i, j] = model.vals[n+1] ** 2
                    A[i, i] = 0
                    ax = plt.subplot(5, 3, 1+n*3)
                    ax = sns.heatmap(to_numpy(model.connection_matrix[n+1]), center=0, square=True, cmap='bwr',
                                     cbar_kws={'fraction': 0.046})
                    ax = plt.subplot(5, 3, 2+n*3)
                    ax = sns.heatmap(to_numpy(A), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
                    plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
                    plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)

                    ax = plt.subplot(5, 3, 3+n*3)
                    gt_weight = to_numpy(model.connection_matrix[n+1])
                    pred_weight = to_numpy(A)
                    plt.scatter(gt_weight, pred_weight, s=40, c='k', alpha=0.1)
                    plt.xlabel(r'true $W_{ij}$', fontsize=12)
                    plt.ylabel(r'learned $W_{ij}$', fontsize=12)

                    x_data = np.reshape(gt_weight, (n_particles * n_particles))
                    y_data = np.reshape(pred_weight, (n_particles * n_particles))
                    lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                    residuals = y_data - linear_model(x_data, *lin_fit)
                    ss_res = np.sum(residuals ** 2)
                    ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                    r_squared = 1 - (ss_res / ss_tot)
                    plt.text(0.1, 0.1, f'R2: {r_squared:0.4f}', fontsize=8, alpha=0.5)
                    plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/matrix/M_{epoch}_{N}.tif", dpi=87)
                plt.close()

            else:

                i, j = torch.triu_indices(n_particles, n_particles, requires_grad=False, device=device)
                A = torch.zeros(n_particles, n_particles, device=device, requires_grad=False, dtype=torch.float32)
                A[i, j] = model.vals[1]**2
                A.T[i, j] = model.vals[1]**2
                A[i, i] = 0

                fig = plt.figure(figsize=(15, 5))
                ax = plt.subplot(1, 3, 1)
                ax = sns.heatmap(to_numpy(model.connection_matrix[1]), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
                ax = plt.subplot(1, 3, 2)
                ax = sns.heatmap(to_numpy(A), center=0, square=True, cmap='bwr', cbar_kws={'fraction': 0.046})
                plt.xticks([0, n_particles - 1], [1, n_particles], fontsize=8)
                plt.yticks([0, n_particles - 1], [1, n_particles], fontsize=8)

                ax = plt.subplot(1, 3, 3)
                gt_weight = to_numpy(model.connection_matrix[1])
                pred_weight = to_numpy(A)
                plt.scatter(gt_weight, pred_weight , s=40, c='k', alpha=0.1)
                plt.xlabel(r'true $W_{ij}$', fontsize=12)
                plt.ylabel(r'learned $W_{ij}$', fontsize=12)

                x_data = np.reshape(gt_weight, (n_particles * n_particles))
                y_data = np.reshape(pred_weight, (n_particles * n_particles))
                lin_fit, lin_fitv = curve_fit(linear_model, x_data, y_data)
                residuals = y_data - linear_model(x_data, *lin_fit)
                ss_res = np.sum(residuals ** 2)
                ss_tot = np.sum((y_data - np.mean(y_data)) ** 2)
                r_squared = 1 - (ss_res / ss_tot)
                plt.text(0.1, 0.1, f'R2: {r_squared:0.4f}', fontsize=8, alpha=0.5)
                plt.tight_layout()

                plt.savefig(f"./{log_dir}/tmp_training/matrix/M_{epoch}_{N}.tif", dpi=87)
                plt.close()

def plot_training_mesh(config, pred, has_field, field, gt, log_dir, epoch, N, x, index_particles, n_particles, n_particle_types, model, n_nodes, n_node_types, index_nodes, dataset_num, ynorm, cmap, axis, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    plot_config = config.plotting
    do_tracking = train_config.do_tracking
    max_radius = simulation_config.max_radius

    matplotlib.rcParams['savefig.pad_inches'] = 0

    match model_config.mesh_model_name:

        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' | 'RD_Mesh4':

            fig = plt.figure(figsize=(8, 8))
            embedding = get_embedding(model.a, 1)
            plt.scatter(embedding[:, 0], embedding[:, 1], c='k', s=1, alpha=0.5, edgecolors='none')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            plt.scatter(to_numpy(gt[:,0]), to_numpy(pred[:,0]), c='r', s=1)
            plt.scatter(to_numpy(gt[:,1]), to_numpy(pred[:,1]), c='g', s=1)
            plt.scatter(to_numpy(gt[:,2]), to_numpy(pred[:,2]), c='b', s=1)
            plt.xlabel('true value', fontsize=14)
            plt.ylabel('pred value', fontsize=14)
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/prediction/{epoch}_{N}.tif", dpi=87)
            plt.close()

            if has_field:
                fig = plt.figure(figsize=(10, 10))
                n_nodes_per_axis = int(np.sqrt(n_nodes))
                plt.imshow(np.reshape(to_numpy(field), (n_nodes_per_axis, n_nodes_per_axis)))
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field/{epoch}_{N}.tif", dpi=87)
                plt.close()

            if False:
                rr = torch.tensor(np.linspace(-1, 1, 200)).to(device)
                rr = rr[:,None]
                rr = torch.cat((rr, torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr)), dim=1)
                popt_list = []
                for n in trange(n_nodes):
                    embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
                    in_features = torch.cat((rr, embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    h = h[:, 0]
                    popt, pcov = curve_fit(linear_model, to_numpy(rr[:,0].squeeze()), to_numpy(h.squeeze()))
                    popt_list.append(popt)
                t = np.array(popt_list)
                t = t[:, 0]
                n_nodes_per_axis = int(np.sqrt(n_nodes))
                fig = plt.figure(figsize=(8, 8))
                t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
                t = np.flipud(t)
                plt.imshow(t, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_LR_{epoch}_{N}.tif",dpi=87)
                plt.close()

                rr = torch.tensor(np.linspace(-1, 1, 200)).to(device)
                rr = rr[:,None]
                rr = torch.cat((torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr), rr, torch.zeros_like(rr), torch.zeros_like(rr)), dim=1)
                popt_list = []
                for n in trange(n_nodes):
                    embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
                    in_features = torch.cat((rr, embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    h = h[:, 0]
                    popt, pcov = curve_fit(linear_model, to_numpy(rr[:,3].squeeze()), to_numpy(h.squeeze()))
                    popt_list.append(popt)
                t = np.array(popt_list)
                t = t[:, 0]
                n_nodes_per_axis = int(np.sqrt(n_nodes))
                fig = plt.figure(figsize=(8, 8))
                t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
                t = np.flipud(t)
                plt.imshow(t, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_uR_{epoch}_{N}.tif",dpi=87)
                plt.close()

                rr = torch.tensor(np.linspace(-1, 1, 200)).to(device)
                rr = rr[:,None]
                rr = torch.cat((torch.zeros_like(rr), rr, torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr)), dim=1)
                popt_list = []
                for n in trange(n_nodes):
                    embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
                    in_features = torch.cat((rr, embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    h = h[:, 1]
                    popt, pcov = curve_fit(linear_model, to_numpy(rr[:,1].squeeze()), to_numpy(h.squeeze()))
                    popt_list.append(popt)
                t = np.array(popt_list)
                t = t[:, 0]
                n_nodes_per_axis = int(np.sqrt(n_nodes))
                fig = plt.figure(figsize=(8, 8))
                t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
                t = np.flipud(t)
                plt.imshow(t, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_LG_{epoch}_{N}.tif",dpi=87)
                plt.close()


                rr = torch.tensor(np.linspace(-1, 1, 200)).to(device)
                rr = rr[:,None]
                rr = torch.cat((torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr), torch.zeros_like(rr), rr, torch.zeros_like(rr)), dim=1)
                popt_list = []
                for n in trange(n_nodes):
                    embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
                    in_features = torch.cat((rr, embedding_), dim=1)
                    h = model.lin_phi(in_features.float())
                    h = h[:, 1]
                    popt, pcov = curve_fit(linear_model, to_numpy(rr[:,4].squeeze()), to_numpy(h.squeeze()))
                    popt_list.append(popt)
                t = np.array(popt_list)
                t = t[:, 0]
                n_nodes_per_axis = int(np.sqrt(n_nodes))
                fig = plt.figure(figsize=(8, 8))
                t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
                t = np.flipud(t)
                plt.imshow(t, cmap='gray')
                plt.xticks([])
                plt.yticks([])
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_vG_{epoch}_{N}.tif",dpi=87)
                plt.close()

        case 'WaveMesh' | 'WaveMeshSmooth':

            if model_config.mesh_model_name == 'WaveMeshSmooth':

                fig = plt.figure(figsize=(10, 4.3))
                ax = fig.add_subplot(121)
                indices = torch.randperm(x.shape[0])[:10000]
                plt.scatter(to_numpy(model.delta_pos[indices, 0]), to_numpy(model.delta_pos[indices, 1]), s=50,
                            c=to_numpy(model.modulation[indices]))
                plt.xlim([-max_radius, max_radius])
                plt.ylim([-max_radius, max_radius])
                plt.colorbar()
                ax = fig.add_subplot(122)
                plt.scatter(to_numpy(model.delta_pos[indices, 0]), to_numpy(model.delta_pos[indices, 1]), s=50,
                            c=to_numpy(model.kernel_operators[indices, 3:4]))
                plt.xlim([-max_radius, max_radius])
                plt.ylim([-max_radius, max_radius])
                plt.colorbar()
                plt.tight_layout()
                plt.savefig(f"./{log_dir}/tmp_training/matrix/kernel_{epoch}_{N}.tif",
                            dpi=87)
                plt.close()


            fig = plt.figure(figsize=(8, 8))
            rr = torch.tensor(np.linspace(-150, 150, 200)).to(device)
            popt_list = []
            for n in range(n_nodes):
                embedding_ = model.a[dataset_num, n, :] * torch.ones((200, 2), device=device)
                in_features = torch.cat((rr[:, None], embedding_), dim=1)
                h = model.lin_phi(in_features.float())
                h = h[:, 0]
                popt, pcov = curve_fit(linear_model, to_numpy(rr.squeeze()), to_numpy(h.squeeze()))
                popt_list.append(popt)
            t = np.array(popt_list)
            t = t[:, 0]
            plt.close()

            n_nodes_per_axis = int(np.sqrt(n_nodes))
            fig = plt.figure(figsize=(8, 8))
            t = np.reshape(t, (n_nodes_per_axis, n_nodes_per_axis))
            t = np.flipud(t)
            plt.imshow(t, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_R_{epoch}_{N}.tif",dpi=87)
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            embedding = get_embedding(model.a, 1)
            plt.scatter(embedding[:, 0], embedding[:, 1], c=t[:, None], s=20, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif",dpi=87)
            plt.close()

            fig = plt.figure(figsize=(8, 8))
            t = np.reshape(t, (100, 100))
            t = np.flipud(t)
            plt.imshow(t, cmap='gray')
            plt.xticks([])
            plt.yticks([])
            plt.tight_layout()
            plt.savefig(f"./{log_dir}/tmp_training/field/mesh_map_{epoch}_{N}.tif",
                        dpi=87)
            plt.close()

def plot_training_mouse(config, log_dir, epoch, N, model):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    model_a = model.a.clone().detach()
    model_a = torch.reshape(model_a, (model_a.shape[0]*model_a.shape[1], model_config.embedding_dim))

    amax = torch.max(model_a, dim=0)[0]
    amin = torch.min(model_a, dim=0)[0]
    model_a = (model_a - amin) / (amax - amin)

    step = model_a.shape[0]*model_a.shape[1]//simulation_config.n_particles

    plt.figure(figsize=(8, 8))
    for n in range(simulation_config.n_particles):
        plt.scatter(to_numpy(model_a[step*n:step*(n+1), 0]), to_numpy(model_a[step*n:step*(n+1), 1]), s=1, alpha=0.25)
    plt.xlim([-0.1,1.1])
    plt.ylim([-0.1,1.1])
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

def plot_training_state(config, id_list,  log_dir, epoch, N, model, n_particle_types, type_list, type_stack, ynorm, cmap, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model

    fig, ax = fig_init()
    for n in range(n_particle_types):
        pos = torch.argwhere(type_stack == n).squeeze()
        if len(pos) > 0:
            plt.scatter(to_numpy(model.a[pos, 0]), to_numpy(model.a[pos, 1]), s=10, color=cmap.color(n), alpha=0.5, edgecolor='none')
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    max_radius = 0.1
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1,1,1)

    if model_config.particle_model_name == 'PDE_Cell_B':
        rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
    elif model_config.particle_model_name == 'PDE_Cell_A_PSC':
        rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
    else:
        rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    if len(type_list) > 1E5:
        nk = len(type_list) // 1000
    else:
        nk = 40
    if len(type_list[0])>1000:
        nn =  10
    else:
        nn = 1

    for k in range(1,len(type_list), nk):
        for n in range(1,len(type_list[k]), nn):
                embedding_ = model.a[to_numpy(id_list[k][n]).astype(int)]
                embedding_ = embedding_ * torch.ones((1000, model_config.embedding_dim), device=device)

                match model_config.particle_model_name:
                    case 'PDE_Cell_A':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_Cell_A_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, torch.ones_like(rr[:, None])*0.1, torch.ones_like(rr[:, None])*0.4, embedding_, embedding_), dim=1)
                    case 'PDE_Cell_B':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    case 'PDE_Cell_B_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], torch.ones_like(rr[:, None])*0.001, torch.ones_like(rr[:, None])*0.001, embedding_, embedding_), dim=1)

                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                type = to_numpy(type_list[k][n])
                plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                             color=cmap.color(int(type)), linewidth=2,alpha=0.5)

    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/{epoch}_{N}.tif",dpi=87)
    plt.close()

def plot_training_cell(config,  log_dir, epoch, N, model, n_particle_types, type_list, ynorm, cmap, device):

    simulation_config = config.simulation
    train_config = config.training
    model_config = config.graph_model
    matplotlib.rcParams['savefig.pad_inches'] = 0

    embedding = get_embedding(model.a, 1)
    fig = plt.figure(figsize=(8, 8))
    for n in range(n_particle_types):
        pos =torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            plt.scatter(embedding[pos, 0], embedding[pos, 1], s=10, alpha=1)

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/embedding/{epoch}_{N}.tif", dpi=87)
    plt.close()

    max_radius = 0.04
    fig = plt.figure(figsize=(12, 12))
    ax = fig.add_subplot(1,1,1)
    rr = torch.tensor(np.linspace(-max_radius, max_radius, 1000)).to(device)
    func_list = []

    if len(type_list) > 10000:
        step = len(type_list) // 1000
    else:
        step = 5

    for n in range(1,len(type_list),step):
        embedding_ = model.a[1, n, :] * torch.ones((1000, model_config.embedding_dim), device=device)
        match model_config.particle_model_name:
            case 'PDE_Cell_A':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, embedding_), dim=1)
            case 'PDE_Cell_A_area':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         rr[:, None] / max_radius, torch.ones_like(rr[:, None])*0.1, torch.ones_like(rr[:, None])*0.4, embedding_, embedding_), dim=1)
            case 'PDE_Cell_B':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
            case 'PDE_Cell_B_area':
                in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                         torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                         0 * rr[:, None], 0 * rr[:, None], torch.ones_like(rr[:, None])*0.001, torch.ones_like(rr[:, None])*0.001, embedding_, embedding_), dim=1)

        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color( int(type_list[n])   ), linewidth=2)
    plt.xlim([-max_radius, max_radius])
    ax.xaxis.set_major_locator(plt.MaxNLocator(3))
    ax.yaxis.set_major_locator(plt.MaxNLocator(5))
    ax.xaxis.set_major_formatter(FormatStrFormatter('%.2f'))
    fmt = lambda x, pos: '{:.1f}e-5'.format((x) * 1e5, pos)
    ax.yaxis.set_major_formatter(mpl.ticker.FuncFormatter(fmt))
    plt.xticks(fontsize=32.0)
    plt.yticks(fontsize=32.0)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/tmp_training/function/lin_edge/{epoch}_{N}.tif",dpi=87)
    plt.close()

def analyze_edge_function_tracking(rr=[], vizualize=False, config=None, model_MLP=[], model_a=None, n_particles=None, ynorm=None, indexes=None, type_list=None, cmap=None, dimension=2, embedding_type=0, device=None):

    model_config = config.graph_model
    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius

    if rr==[]:
        if model_config.particle_model_name == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        elif model_config.particle_model_name == 'PDE_E':
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    if embedding_type == 1:
        n_list = indexes
    else:
        n_list = range(n_particles)

    func_list = []
    for n, k  in enumerate(n_list):
        embedding_ = model_a[int(k), :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        if config.graph_model.particle_model_name != '':
            config_model = config.graph_model.particle_model_name
        elif config.graph_model.signal_model_name != '':
            config_model = config.graph_model.signal_model_name
        elif config.graph_model.mesh_model_name != '':
            config_model = config.graph_model.mesh_model_name
        in_features = get_in_features(rr, embedding_, config_model, max_radius)
        with torch.no_grad():
            func = model_MLP(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        if ((n % 5 == 0) | (config.graph_model.particle_model_name=='PDE_GS')) & vizualize:
            plt.plot(to_numpy(rr),
                     to_numpy(func) * to_numpy(ynorm),
                     color=cmap.color(int(type_list[int(n)])), linewidth=2, alpha=0.25)
    func_list = torch.stack(func_list)
    coeff_norm = to_numpy(func_list)

    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if coeff_norm.shape[0] > 1000:
            new_index = np.random.permutation(coeff_norm.shape[0])
            new_index = new_index[0:min(1000, coeff_norm.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(coeff_norm[new_index])
            proj_interaction = trans.transform(coeff_norm)
        else:
            trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(coeff_norm)
            proj_interaction = trans.transform(coeff_norm)

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xscale('log')
            plt.yscale('log')
            plt.xlim([1E-3, 0.2])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        plt.xlabel('Distance [a.u]', fontsize=12)
        plt.ylabel('MLP [a.u]', fontsize=12)

    return func_list, proj_interaction

def analyze_edge_function_state(rr=[], config=None, model=None, id_list=None, type_list=None, cmap=None, ynorm=None, visualize=False, device=None):

    max_radius = config.simulation.max_radius

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    func_list = []
    true_type_list = []
    short_model_a_list = []
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
    for k in range(1,len(type_list), 10):
        for n in range(1,len(type_list[k]),10):
                short_model_a_list.append(model.a[to_numpy(id_list[k][n]).astype(int)])
                if config.training.use_hot_encoding:
                    embedding_ = torch.matmul(torch.sigmoid((model.a[to_numpy(id_list[k][n]).astype(int)] - 0.5) * 10), model.b.clone().detach()).squeeze()
                else:
                    embedding_ = model.a[to_numpy(id_list[k][n]).astype(int)]
                embedding_ = embedding_ * torch.ones((1000, config.simulation.dimension), device=device)

                match config_model:
                    case 'PDE_Cell_A':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, embedding_), dim=1)
                    case 'PDE_Cell_A_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 rr[:, None] / max_radius, torch.ones_like(rr[:, None])*0.1, torch.ones_like(rr[:, None])*0.4, embedding_, embedding_), dim=1)
                    case 'PDE_Cell_B':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], embedding_), dim=1)
                    case 'PDE_Cell_B_area':
                        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                                 torch.abs(rr[:, None]) / max_radius, 0 * rr[:, None], 0 * rr[:, None],
                                                 0 * rr[:, None], 0 * rr[:, None], torch.ones_like(rr[:, None])*0.001, torch.ones_like(rr[:, None])*0.001, embedding_, embedding_), dim=1)

                with torch.no_grad():
                    func = model.lin_edge(in_features.float())
                func = func[:, 0]
                func_list.append(func)
                true_type_list.append(type_list[k][n])
                if visualize:
                    plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),color=cmap.color(int(type_list[k][n])), linewidth=2, alpha=0.25)

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)
    true_type_list = torch.stack(true_type_list)
    true_type_list = to_numpy(true_type_list)
    short_model_a_list = torch.stack(short_model_a_list)

    print('UMAP reduction ...')
    start_time = time.time()
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        trans = umap.UMAP(n_neighbors=100, n_components=2, transform_queue_size=0).fit(func_list_)
    computation_time = time.time() - start_time
    print(f"UMAP computation time is {np.round(computation_time)} seconds.")

    proj_interaction = trans.transform(func_list_)
    proj_interaction = (proj_interaction - np.min(proj_interaction)) / (np.max(proj_interaction) - np.min(proj_interaction) + 1e-10)

    computation_time = time.time() - start_time
    print(f"dimension reduction computation time is {np.round(computation_time)} seconds.")

    return func_list, true_type_list, short_model_a_list, proj_interaction

def analyze_edge_function(rr=[], vizualize=False, config=None, model_MLP=[], model=None, n_nodes=0, n_particles=None, ynorm=None, type_list=None, cmap=None, update_type=None, device=None):

    max_radius = config.simulation.max_radius
    min_radius = config.simulation.min_radius
    dimension = config.simulation.dimension

    if config.graph_model.particle_model_name != '':
        config_model = config.graph_model.particle_model_name
    elif config.graph_model.signal_model_name != '':
        config_model = config.graph_model.signal_model_name
    elif config.graph_model.mesh_model_name != '':
        config_model = config.graph_model.mesh_model_name

    if rr==[]:
        if config_model == 'PDE_G':
            rr = torch.tensor(np.linspace(0, max_radius * 1.3, 1000)).to(device)
        elif config_model == 'PDE_GS':
            rr = torch.tensor(np.logspace(7, 9, 1000)).to(device)
        elif config_model == 'PDE_E':
            rr = torch.tensor(np.linspace(min_radius, max_radius, 1000)).to(device)
        elif 'PDE_N' in config_model:
            rr = torch.tensor(np.linspace(-5, 5, 1000)).to(device)
        else:
            rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    print('interaction functions ...')
    func_list = []
    for n in range(n_particles):

        if len(model.a.shape)==3:
            model_a= model.a[1, n, :]
        else:
            model_a = model.a[n, :]

        if config.training.do_tracking:
            embedding_ = model_a * torch.ones((1000, dimension), device=device)
        else:
            if (update_type != 'NA') & model.embedding_trial:
                embedding_ = torch.cat((model_a, model.b[0].clone().detach().repeat(n_particles, 1)), dim=1) * torch.ones((1000, 2*dimension), device=device)
            else:
                embedding_ = model_a * torch.ones((1000, dimension), device=device)

        if update_type == 'NA':
            in_features = get_in_features(rr=rr, embedding=embedding_, model=model, model_name=config_model, max_radius=max_radius)
        else:
            in_features = get_in_features_update(rr=rr[:, None], embedding=embedding_, model=model, device=device)
        with torch.no_grad():
            func = model_MLP(in_features.float())[:, 0]

        func_list.append(func)

        should_plot = vizualize and (
                n_particles <= 200 or
                (n % (n_particles // 200) == 0) or
                (config.graph_model.particle_model_name == 'PDE_GS') or
                ('PDE_N' in config_model)
        )

        if should_plot:
            plt.plot(
                to_numpy(rr),
                to_numpy(func) * to_numpy(ynorm),
                2,
                color=cmap.color(type_list[n].astype(int)),
                linewidth=1,
                alpha=0.25
            )

    func_list = torch.stack(func_list)
    func_list_ = to_numpy(func_list)

    if vizualize:
        if config.graph_model.particle_model_name == 'PDE_GS':
            plt.xscale('log')
            plt.yscale('log')
        if config.graph_model.particle_model_name == 'PDE_G':
            plt.xlim([1E-3, 0.02])
        if config.graph_model.particle_model_name == 'PDE_E':
            plt.xlim([0, 0.05])
        if 'PDE_N' in config.graph_model.particle_model_name:
            plt.xlim(config.plotting.xlim)


        # ylim = [np.min(func_list_)/1.05, np.max(func_list_)*1.05]
        plt.ylim(config.plotting.ylim)

    print('UMAP reduction ...')
    with warnings.catch_warnings():
        warnings.simplefilter('ignore')
        if func_list_.shape[0] > 1000:
            new_index = np.random.permutation(func_list_.shape[0])
            new_index = new_index[0:min(1000, func_list_.shape[0])]
            trans = umap.UMAP(n_neighbors=500, n_components=2, transform_queue_size=0, random_state=config.training.seed).fit(func_list_[new_index])
            proj_interaction = trans.transform(func_list_)
        else:
            trans = umap.UMAP(n_neighbors=50, n_components=2, transform_queue_size=0).fit(func_list_)
            proj_interaction = trans.transform(func_list_)

    return func_list, proj_interaction

def choose_training_model(model_config=None, device=None, projections=None):

    dataset_name = model_config.dataset
    aggr_type = model_config.graph_model.aggr_type
    n_particle_types = model_config.simulation.n_particle_types
    n_particles = model_config.simulation.n_particles
    dimension = model_config.simulation.dimension
    do_tracking = model_config.training.do_tracking

    bc_pos, bc_dpos = choose_boundary_values(model_config.simulation.boundary)

    model=[]
    model_name = model_config.graph_model.particle_model_name
    match model_name:
        case 'PDE_R':
            model = Interaction_Mouse(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                     dimension=dimension)
        case 'PDE_MPM':
            model = Interaction_MPM(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos,
                                    dimension=dimension)
        case  'PDE_Cell' | 'PDE_Cell_area':
            model = Interaction_Cell(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_ParticleField_A' | 'PDE_ParticleField_B':
            model = Interaction_Particle_Field(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
        case 'PDE_Agents' | 'PDE_Agents_A' | 'PDE_Agents_B' | 'PDE_Agents_C':
            model = Interaction_Agent(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_A' | 'PDE_A_bis' | 'PDE_B' | 'PDE_B_mass' | 'PDE_B_bis' | 'PDE_E' | 'PDE_G' | 'PDE_K' | 'PDE_T':
            model = Interaction_Particle(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
            model.edges = []
            if 'PDE_K' in model_name:
                model.connection_matrix = torch.load(f'./graphs_data/{dataset_name}/connection_matrix_list.pt', map_location=device)
        case 'PDE_GS':
            model = Interaction_Planet(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_particles)
            t1 = np.repeat(t, model_config.simulation.n_particles)
            t2 = np.tile(t, model_config.simulation.n_particles)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_GS2':
            model = Interaction_Planet2(aggr_type=aggr_type, config=model_config, device=device)
            t = np.arange(model_config.simulation.n_particles)
            t1 = np.repeat(t, model_config.simulation.n_particles)
            t2 = np.tile(t, model_config.simulation.n_particles)
            e = np.stack((t1, t2), axis=0)
            pos = np.argwhere(e[0, :] - e[1, :] != 0)
            e = e[:, pos]
            model.edges = torch.tensor(e, dtype=torch.long, device=device)
        case 'PDE_Cell_A' | 'PDE_Cell_B' | 'PDE_Cell_B_area' | 'PDE_Cell_A_area':
            model = Interaction_Cell(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_F_A' |'PDE_F_B'|'PDE_F_C'|'PDE_F_D'|'PDE_F_E' :
            model = Interaction_Smooth_Particle(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_MLPs' | 'PDE_MLPs_A' | 'PDE_MLPs_A_bis' | 'PDE_MLPs_A_ter' | 'PDE_MLPs_B'| 'PDE_MLPs_B_0' |'PDE_MLPs_B_1' | 'PDE_MLPs_B_4'| 'PDE_MLPs_B_10' |'PDE_MLPs_C' | 'PDE_MLPs_D' | 'PDE_MLPs_E' | 'PDE_MLPs_F':
            model = Interaction_PDE_Particle(aggr_type=aggr_type, config=model_config, device=device,
                                                bc_dpos=bc_dpos, dimension=dimension)
        case 'PDE_M' | 'PDE_M2':
            model = Interaction_Particle2(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MM' | 'PDE_MM_1layer' | 'PDE_MM_2layers' | 'PDE_MM_3layers':
            model = Interaction_Particle3(aggr_type=aggr_type, config=model_config, bc_dpos=bc_dpos, dimension=dimension, device=device)
        case 'PDE_MS':
            model = Interaction_Falling_Water_Smooth(aggr_type=aggr_type, config=model_config,bc_dpos=bc_dpos, dimension=dimension, device=device)

    model_name = model_config.graph_model.mesh_model_name
    match model_name:
        case 'DiffMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMesh':
            model = Mesh_Laplacian(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'WaveMeshSmooth':
            model = Mesh_Smooth(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'RD_Mesh' | 'RD_Mesh2' | 'RD_Mesh3' | 'RD_Mesh4':
            model = Mesh(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    model_name = model_config.graph_model.signal_model_name
    match model_name:
        case 'PDE_N2' | 'PDE_N3' | 'PDE_N4' | 'PDE_N5' | 'PDE_N6' | 'PDE_N7' | 'PDE_N9' | 'PDE_N8':
            model = Signal_Propagation2(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []
        case 'PDE_WBI':
            from MPM_pytorch.models import WBI_Communication
            model = WBI_Communication(aggr_type=aggr_type, config=model_config, device=device, bc_dpos=bc_dpos)
            model.edges = []

    if model==[]:
        raise ValueError(f'Unknown model {model_name}')

    return model, bc_pos, bc_dpos

def constant_batch_size(batch_size):
    def get_batch_size(epoch):
        return batch_size

    return get_batch_size

def increasing_batch_size(batch_size):
    def get_batch_size(epoch):
        return 1 if epoch < 1 else batch_size

    return get_batch_size

def set_trainable_parameters(model=[], lr_embedding=[], lr=[],  lr_update=[], lr_W=[], lr_modulation=[]):

    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.a)

    # embedding = model.a
    # optimizer = torch.optim.Adam([embedding], lr=lr_embedding)
    #
    # _, *parameters = trainable_params
    # for parameter in parameters:
    #     optimizer.add_param_group({'params': parameter, 'lr': lr})

    if lr_update==[]:
        lr_update = lr

    optimizer = torch.optim.Adam([model.a], lr=lr_embedding)
    for name, parameter in model.named_parameters():
        if (parameter.requires_grad) & (name!='a'):
            if (name=='b') or ('lin_modulation' in name):
                optimizer.add_param_group({'params': parameter, 'lr': lr_modulation})
                # print(f'lr_modulation: {name} {lr_modulation}')
            elif 'lin_phi' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_update})
                # print(f'lr_W: {name} {lr_W}')
            elif 'W' in name:
                optimizer.add_param_group({'params': parameter, 'lr': lr_W})
                # print(f'lr_W: {name} {lr_W}')
            else:
                optimizer.add_param_group({'params': parameter, 'lr': lr})
                # print(f'lr: {name} {lr}')

    return optimizer, n_total_params

def set_trainable_division_parameters(model, lr):
    trainable_params = [param for _, param in model.named_parameters() if param.requires_grad]
    n_total_params = sum(p.numel() for p in trainable_params) + torch.numel(model.t)

    embedding = model.t
    optimizer = torch.optim.Adam([embedding], lr=lr)

    _, *parameters = trainable_params
    for parameter in parameters:
        optimizer.add_param_group({'params': parameter, 'lr': lr})

    return optimizer, n_total_params

def get_index_particles(x, n_particle_types, dimension):
    index_particles = []
    for n in range(n_particle_types):
        if dimension == 2:
            index = np.argwhere(x[:, 5].detach().cpu().numpy() == n)
        elif dimension == 3:
            index = np.argwhere(x[:, 7].detach().cpu().numpy() == n)
        index_particles.append(index.squeeze())
    return index_particles

def get_type_list(x, dimension):
    type_list = x[:, 1 + 2 * dimension:2 + 2 * dimension].clone().detach()
    return type_list

def sample_synaptic_data_and_predict(model, x_list, edges, n_runs, n_frames, time_step, device,
                            has_missing_activity=False, model_missing_activity=None,
                            has_neural_field=False, model_f=None,
                            run=None, k=None):
    """
    Sample data from x_list and get model predictions

    Args:
        model: trained GNN model
        x_list: list of data arrays [n_runs][n_frames]
        edges: edge indices for graph
        n_runs, n_frames, time_step: data dimensions
        device: torch device
        has_missing_activity: whether to fill missing activity
        model_missing_activity: model for missing activity (if needed)
        has_neural_field: whether to compute neural field
        model_f: field model (if needed)
        run: specific run index (if None, random)
        k: specific frame index (if None, random)

    Returns:
        dict with pred, in_features, x, dataset, data_id, k_batch
    """
    # Sample random run and frame if not specified
    if run is None:
        run = np.random.randint(n_runs)
    if k is None:
        k = np.random.randint(n_frames - 4 - time_step)

    # Get data
    x = torch.tensor(x_list[run][k], dtype=torch.float32, device=device)

    # Handle missing activity if needed
    if has_missing_activity and model_missing_activity is not None:
        pos = torch.argwhere(x[:, 6] == 6)
        if len(pos) > 0:
            t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
            missing_activity = model_missing_activity[run](t).squeeze()
            x[pos, 6] = missing_activity[pos]

    # Handle neural field if needed
    if has_neural_field and model_f is not None:
        t = torch.tensor([k / n_frames], dtype=torch.float32, device=device)
        x[:, 8] = model_f[run](t) ** 2

    # Create dataset
    dataset = data.Data(x=x, edge_index=edges)
    data_id = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * run
    k_batch = torch.ones((x.shape[0], 1), dtype=torch.int, device=device) * k

    # Get predictions
    pred, in_features = model(dataset, data_id=data_id, k=k_batch, return_all=True)

    return {
        'pred': pred,
        'in_features': in_features,
        'x': x,
        'dataset': dataset,
        'data_id': data_id,
        'k_batch': k_batch,
        'run': run,
        'k': k
    }


def analyze_odor_responses_by_neuron(model, x_list, edges, n_runs, n_frames, time_step, device,
                                     all_neuron_list, has_missing_activity=False, model_missing_activity=None,
                                     has_neural_field=False, model_f=None, n_samples=50, run=0):
    """
    Analyze odor responses by comparing lin_phi output with and without excitation
    Returns top responding neurons by name for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']

    # Store responses: difference between excitation and baseline
    odor_responses = {odor: [] for odor in odor_list}
    embeddings_by_neuron = []
    valid_samples = 0

    model.eval()
    with torch.no_grad():
        sample = 0
        while valid_samples < n_samples:
            result = sample_synaptic_data_and_predict(
                model, x_list, edges, n_runs, n_frames, time_step, device,
                has_missing_activity, model_missing_activity,
                has_neural_field, model_f, run
            )

            if not (torch.isnan(result['x']).any()):
                # Get baseline response (no excitation)
                x_baseline = result['x'].clone()
                x_baseline[:, 10:13] = 0  # no excitation
                dataset_baseline = data.Data(x=x_baseline, edge_index=edges)
                pred_baseline = model(dataset_baseline, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                for i, odor in enumerate(odor_list):
                    x_odor = result['x'].clone()
                    x_odor[:, 10:13] = 0
                    x_odor[:, 10 + i] = 1  # activate specific odor

                    dataset_odor = data.Data(x=x_odor, edge_index=edges)
                    pred_odor = model(dataset_odor, data_id=result['data_id'],
                                      k=result['k_batch'], return_all=False)

                    odor_diff = pred_odor - pred_baseline
                    odor_responses[odor].append(odor_diff.cpu())

                valid_samples += 1

            sample += 1
            if sample > n_samples * 10:
                break

        # Convert to tensors [n_samples, n_neurons]
        for odor in odor_list:
            odor_responses[odor] = torch.stack(odor_responses[odor]).squeeze()

    # Identify top responding neurons for each odor
    top_neurons = {}
    for odor in odor_list:
        # Calculate mean response across samples for each neuron
        mean_response = torch.mean(odor_responses[odor], dim=0)  # [n_neurons]

        # Get top 3 responding neurons (highest positive response)
        top_20_indices = torch.topk(mean_response, k=20).indices.cpu().numpy()
        top_20_names = [all_neuron_list[idx] for idx in top_20_indices]
        top_20_values = [mean_response[idx].item() for idx in top_20_indices]

        top_neurons[odor] = {
            'names': top_20_names,
            'indices': top_20_indices.tolist(),
            'values': top_20_values
        }

        print(f"\ntop 20 responding neurons for {odor}:")
        for i, (name, idx, val) in enumerate(zip(top_20_names, top_20_indices, top_20_values)):
            print(f"  {i + 1}. {name} : {val:.4f}")

    return odor_responses  # Return only odor_responses to match original function signature


def plot_odor_heatmaps(odor_responses):
    """
    Plot 3 separate heatmaps showing mean response per neuron for each odor
    """
    odor_list = ['butanone', 'pentanedione', 'NaCL']
    n_neurons = odor_responses['butanone'].shape[1]

    fig, axes = plt.subplots(1, 3, figsize=(18, 6))

    for i, odor in enumerate(odor_list):
        # Compute mean response per neuron
        mean_responses = torch.mean(odor_responses[odor], dim=0).numpy()  # [n_neurons]

        # Reshape to 2D for heatmap (assuming square-ish layout)
        side_length = int(np.ceil(np.sqrt(n_neurons)))
        padded_responses = np.pad(mean_responses, (0, side_length ** 2 - n_neurons), 'constant')
        response_matrix = padded_responses.reshape(side_length, side_length)

        # Plot heatmap
        sns.heatmap(response_matrix, ax=axes[i], cmap='bwr', center=0,
                    cbar=False, square=True, xticklabels=False, yticklabels=False)
        axes[i].set_title(f'{odor} mean response')

    plt.tight_layout()
    return fig








