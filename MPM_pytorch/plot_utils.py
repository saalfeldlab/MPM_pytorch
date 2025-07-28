import os
import torch
import matplotlib.pyplot as plt
import matplotlib
import numpy as np
from tqdm import trange
from matplotlib import rc
from matplotlib.pyplot import imread
import glob
from MPM_pytorch.utils import fig_init, to_numpy
from MPM_pytorch.models.utils import get_embedding
from sklearn import metrics


def setup_plot_style(style):
    if 'black' in style:
        plt.style.use('dark_background')
        mc = 'w'
    else:
        plt.style.use('default')
        mc = 'k'

    if 'latex' in style:
        plt.rcParams['text.usetex'] = True
        rc('font', **{'family': 'serif', 'serif': ['Palatino']})
    else:
        plt.rcParams['text.usetex'] = False
        plt.rc('font', family='sans-serif')
        plt.rc('text', usetex=False)

    matplotlib.rcParams['savefig.pad_inches'] = 0
    return mc


def sort_key(filename):
    try:
        parts = filename.split('_')
        for part in parts:
            if part.replace('.', '').isdigit():
                return float(part.replace('.pt', ''))
        return 0
    except:
        return 0


def get_training_files_optimized(log_dir, n_runs):
    files = glob.glob(f"{log_dir}/models/best_model_with_{n_runs - 1}_graphs_*.pt")
    files.sort(key=sort_key)

    if not files:
        return [], []

    flag = True
    file_id = 0
    while flag and file_id < len(files):
        if sort_key(files[file_id]) > 0:
            flag = False
            file_id = file_id - 1
        file_id += 1

    files = files[file_id:]
    files_with_0 = [file for file in files if f'_0_' in file]
    files_without_0 = [file for file in files if '_0_' not in file]

    indices_with_0 = np.arange(0, len(files_with_0) - 1, dtype=int)
    indices_without_0 = np.linspace(0, len(files_without_0) - 1, 50, dtype=int) if files_without_0 else []

    selected_files_with_0 = [files_with_0[i] for i in indices_with_0]
    selected_files_without_0 = [files_without_0[i] for i in indices_without_0]
    selected_files = selected_files_with_0 + selected_files_without_0

    return selected_files, np.arange(0, len(selected_files), 1)


def create_embedding_plot(model, type_list, n_particle_types, cmap, config, style, alpha=0.1):
    fig, ax = fig_init(fontsize=24)
    params = {'mathtext.default': 'regular'}
    plt.rcParams.update(params)
    embedding = get_embedding(model.a, 1)

    for n in range(n_particle_types - 1, -1, -1):
        pos = torch.argwhere(type_list == n)
        pos = to_numpy(pos)
        if len(pos) > 0:
            plt.scatter(embedding[pos, 0], embedding[pos, 1], color=cmap.color(n), s=100, alpha=alpha)

    plt.xlabel(r'$a_{0}$', fontsize=48)
    plt.ylabel(r'$a_{1}$', fontsize=48)

    if 'arbitrary_16' in config.dataset:
        plt.xlim([-2.5, 2.5])
        plt.ylim([-2.5, 2.5])
    else:
        plt.xlim([0.5, 1.5])
        plt.ylim([0.5, 1.5])

    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    return fig


def create_function_plot(model, config, type_list, cmap, ynorm, n_particles, max_radius, device):
    fig, ax = fig_init(fontsize=24)
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)

    for n in range(int(n_particles * (1 - getattr(config.training, 'particle_dropout', 0)))):
        embedding_ = model.a[1, n] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
            func = func[:, 0]
        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)

    plt.xlabel('$d_{ij}$', fontsize=48)
    plt.ylabel('$f(a_i, d_{ij})$', fontsize=48)
    plt.xlim([0, max_radius])
    plt.ylim(getattr(config.plotting, 'ylim', [-0.04, 0.03]))
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()
    return fig


def create_combined_plot(embedding_fig, function_fig, log_dir, epoch, it):
    embedding_fig.savefig(f"./{log_dir}/results/all/embedding_{epoch}.tif", dpi=80)
    function_fig.savefig(f"./{log_dir}/results/all/function_{epoch}.tif", dpi=80)
    plt.close(embedding_fig)
    plt.close(function_fig)

    im0 = imread(f"./{log_dir}/results/all/embedding_{epoch}.tif")
    im1 = imread(f"./{log_dir}/results/all/function_{epoch}.tif")

    fig = plt.figure(figsize=(16, 8))
    plt.axis('off')
    plt.subplot(1, 2, 1)
    plt.axis('off')
    plt.imshow(im0)
    plt.xticks([])
    plt.yticks([])
    plt.subplot(1, 2, 2)
    plt.axis('off')
    plt.imshow(im1)
    plt.xticks([])
    plt.yticks([])
    plt.tight_layout()

    num = str(it).zfill(4)
    plt.savefig(f"./{log_dir}/results/training/fig_{num}.tif", dpi=80)
    plt.close(fig)


def process_all_epochs(files, file_id_list, model, config, log_dir, type_list, n_particle_types,
                       cmap, ynorm, style, device, n_particles, max_radius):
    with torch.no_grad():
        it = 0
        for file_id_ in trange(0, len(file_id_list)):
            epoch = files[file_id_].split('graphs')[1][1:-3]
            print(epoch)

            net = f"{log_dir}/models/best_model_with_1_graphs_{epoch}.pt"
            state_dict = torch.load(net, map_location=device)
            model.load_state_dict(state_dict['model_state_dict'])
            model.eval()

            embedding_fig = create_embedding_plot(model, type_list, n_particle_types, cmap, config, style)
            function_fig = create_function_plot(model, config, type_list, cmap, ynorm, n_particles, max_radius, device)
            create_combined_plot(embedding_fig, function_fig, log_dir, epoch, it)

            it += 1


def plot_learned_vs_true_functions(model, config, type_list, n_particles, n_particle_types, ynorm,
                                   max_radius, log_dir, epoch, style, cmap, logger, device):
    fig, ax = fig_init()
    p = torch.load(f'graphs_data/{config.dataset}/model_p.pt', map_location=device)
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
    rmserr_list = []

    for n in range(int(n_particles * (1 - getattr(config.training, 'particle_dropout', 0)))):
        embedding_ = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, embedding_), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
            func = func[:, 0]
        true_func = model.psi(rr, p[to_numpy(type_list[n]).astype(int)].squeeze(),
                              p[to_numpy(type_list[n]).astype(int)].squeeze())
        rmserr_list.append(torch.sqrt(torch.mean((func * ynorm - true_func.squeeze()) ** 2)))
        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(to_numpy(type_list[n]).astype(int)), linewidth=8, alpha=0.1)

    if 'latex' in style:
        plt.xlabel(r'$d_{ij}$', fontsize=68)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
    else:
        plt.xlabel(r'$d_{ij}$', fontsize=68)
        plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)

    plt.xlim([0, max_radius])
    plt.ylim(config.plotting.ylim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/learned_function_{epoch}.tif", dpi=170.7)

    rmserr_list = torch.stack(rmserr_list)
    rmserr_array = to_numpy(rmserr_list)
    print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_array), np.std(rmserr_array)))
    logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_array), np.std(rmserr_array)))
    plt.close()

    fig, ax = fig_init()
    for n in range(n_particle_types):
        plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n], p[n])), color=cmap.color(n), linewidth=8)

    plt.xlim([0, max_radius])
    plt.ylim(config.plotting.ylim)
    if 'latex' in style:
        plt.xlabel(r'$d_{ij}$', fontsize=68)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})$', fontsize=68)
    else:
        plt.xlabel(r'$d_{ij}$', fontsize=68)
        plt.ylabel(r'$f(a_i, d_{ij})$', fontsize=68)

    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
    plt.close()


def analyze_specific_epoch(model, config, embedding_cluster, cmap, index_particles, type_list,
                           n_particle_types, n_particles, ynorm, epoch, log_dir, style, device, logger,
                           max_radius):
    from MPM_pytorch.plot_utils import plot_embedding_func_cluster

    config.training.cluster_method = 'distance_plot'
    config.training.cluster_distance_threshold = 0.01
    accuracy1, n_clusters1, new_labels1 = plot_embedding_func_cluster(
        model, config, embedding_cluster, cmap, index_particles, type_list,
        n_particle_types, n_particles, ynorm, epoch, log_dir, 0.1, style, device)

    print(f'result accuracy: {np.round(accuracy1, 2)} n_clusters: {n_clusters1} '
          f'method: {config.training.cluster_method} threshold: {config.training.cluster_distance_threshold}')
    logger.info(f'result accuracy: {np.round(accuracy1, 2)} n_clusters: {n_clusters1} '
                f'method: {config.training.cluster_method} threshold: {config.training.cluster_distance_threshold}')

    config.training.cluster_method = 'distance_embedding'
    accuracy2, n_clusters2, new_labels2 = plot_embedding_func_cluster(
        model, config, embedding_cluster, cmap, index_particles, type_list,
        n_particle_types, n_particles, ynorm, epoch, log_dir, 0.1, style, device)

    print(f'result accuracy: {np.round(accuracy2, 2)} n_clusters: {n_clusters2} '
          f'method: {config.training.cluster_method} threshold: {config.training.cluster_distance_threshold}')
    logger.info(f'result accuracy: {np.round(accuracy2, 2)} n_clusters: {n_clusters2} '
                f'method: {config.training.cluster_method} threshold: {config.training.cluster_distance_threshold}')

    plot_learned_vs_true_functions(model, config, type_list, n_particles, n_particle_types, ynorm,
                                   max_radius, log_dir, epoch, style, cmap, logger, device)


def plot_asymmetric_functions(model, config, x_list, bc_dpos, dimension, max_radius, min_radius,
                              n_particle_types, ynorm, log_dir, epoch, style, cmap, logger, device):
    x = x_list[0][100].clone().detach()
    type_list = to_numpy(get_type_list(x, dimension))
    distance = torch.sum(bc_dpos(x[:, None, 1:dimension + 1] - x[None, :, 1:dimension + 1]) ** 2, dim=2)
    adj_t = ((distance < max_radius ** 2) & (distance > min_radius ** 2)).float() * 1
    edges = adj_t.nonzero().t().contiguous()
    indexes = np.random.randint(0, edges.shape[1], 5000)
    edges = edges[:, indexes]

    fig, ax = fig_init()
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
    func_list = []

    for n in trange(edges.shape[1]):
        embedding_1 = model.a[1, edges[0, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        embedding_2 = model.a[1, edges[1, n], :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        type_idx = type_list[to_numpy(edges[0, n])].astype(int)

        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, embedding_1, embedding_2), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm), color=cmap.color(type_idx), linewidth=8)

    if 'latex' in style:
        plt.xlabel(r'$d_{ij}
                   , fontsize=68)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})
                   , fontsize=68)
    else:
        plt.xlabel('$d_{ij}
                   , fontsize=68)
        plt.ylabel('$f(a_i, d_{ij})
                   , fontsize=68)
    plt.ylim(config.plotting.ylim)
    plt.xlim([0, max_radius])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/func_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    p = torch.load(f'graphs_data/{config.dataset}/model_p.pt', map_location=device)
    true_func = []

    for n in range(n_particle_types):
        for m in range(n_particle_types):
            true_func.append(model.psi(rr, p[n, m].squeeze(), p[n, m].squeeze()))
            plt.plot(to_numpy(rr), to_numpy(model.psi(rr, p[n, m], p[n, m]).squeeze()),
                     color=cmap.color(n), linewidth=8)

    if 'latex' in style:
        plt.xlabel(r'$d_{ij}
                   , fontsize=68)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})
                   , fontsize=68)
    else:
        plt.xlabel('$d_{ij}
                   , fontsize=68)
        plt.ylabel('$f(a_i, d_{ij})
                   , fontsize=68)
    plt.ylim(config.plotting.ylim)
    plt.xlim([0, max_radius])
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
    plt.close()

    true_func_list = []
    for k in trange(edges.shape[1]):
        n = type_list[to_numpy(edges[0, k])].astype(int)
        m = type_list[to_numpy(edges[1, k])].astype(int)
        true_func_list.append(true_func[3 * n.squeeze() + m.squeeze()])

    func_list = torch.stack(func_list) * ynorm
    true_func_list = torch.stack(true_func_list)
    rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
    rmserr_list = to_numpy(rmserr_list)

    print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
    logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))


def plot_continuous_embedding_and_functions(model, config, n_particles, max_radius, ynorm,
                                            log_dir, epoch, style, cmap, logger, device):
    n_particle_types = 3
    index_particles = []
    for n in range(n_particle_types):
        index_particles.append(np.arange((n_particles // n_particle_types) * n,
                                         (n_particles // n_particle_types) * (n + 1)))

    fig, ax = fig_init()
    embedding = get_embedding(model.a, 1)
    for n in range(n_particle_types):
        plt.scatter(embedding[index_particles[n], 0], embedding[index_particles[n], 1],
                    color=cmap.color(n), s=400, alpha=0.1)

    if 'latex' in style:
        plt.xlabel(r'$\ensuremath{\mathbf{a}}_{i0}
                   , fontsize=68)
        plt.ylabel(r'$\ensuremath{\mathbf{a}}_{i1}
                   , fontsize=68)
    else:
        plt.xlabel(r'$a_{0}
                   , fontsize=68)
        plt.ylabel(r'$a_{1}
                   , fontsize=68)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/first_embedding_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    rr = torch.tensor(np.linspace(0, max_radius, 1000)).to(device)
    func_list = []

    for n in range(n_particles):
        embedding = model.a[1, n, :] * torch.ones((1000, config.graph_model.embedding_dim), device=device)
        in_features = torch.cat((rr[:, None] / max_radius, 0 * rr[:, None],
                                 rr[:, None] / max_radius, embedding), dim=1)
        with torch.no_grad():
            func = model.lin_edge(in_features.float())
        func = func[:, 0]
        func_list.append(func)
        plt.plot(to_numpy(rr), to_numpy(func) * to_numpy(ynorm),
                 color=cmap.color(n // 1600), linewidth=2, alpha=0.1)

    if 'latex' in style:
        plt.xlabel(r'$d_{ij}
                   , fontsize=68)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})
                   , fontsize=68)
    else:
        plt.xlabel('$d_{ij}
                   , fontsize=68)
        plt.ylabel('$f(a_i, d_{ij})
                   , fontsize=68)
    plt.xlim([0, max_radius])
    plt.ylim(config.plotting.ylim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/func_{epoch}.tif", dpi=170.7)
    plt.close()

    fig, ax = fig_init()
    p = torch.load(f'graphs_data/{config.dataset}/model_p.pt')
    true_func_list = []
    csv_ = [to_numpy(rr)]

    for n in range(n_particles):
        true_func = model.psi(rr, p[n], p[n])
        plt.plot(to_numpy(rr), to_numpy(true_func), color=cmap.color(n // 1600),
                 linewidth=2, alpha=0.1)
        true_func_list.append(true_func)
        csv_.append(to_numpy(true_func.squeeze()))

    if 'latex' in style:
        plt.xlabel(r'$d_{ij}
                   , fontsize=68)
        plt.ylabel(r'$f(\ensuremath{\mathbf{a}}_i, d_{ij})
                   , fontsize=68)
    else:
        plt.xlabel(r'$d_{ij}
                   , fontsize=68)
        plt.ylabel('$f(a_i, d_{ij})
                   , fontsize=68)
    plt.xticks(fontsize=32)
    plt.yticks(fontsize=32)
    plt.xlim([0, max_radius])
    plt.ylim(config.plotting.ylim)
    plt.tight_layout()
    plt.savefig(f"./{log_dir}/results/true_func.tif", dpi=170.7)
    np.save(f"./{log_dir}/results/true_func_{epoch}.npy", csv_)
    np.savetxt(f"./{log_dir}/results/true_func_{epoch}.txt", csv_)
    plt.close()

    func_list = torch.stack(func_list) * ynorm
    true_func_list = torch.stack(true_func_list)
    rmserr_list = torch.sqrt(torch.mean((func_list - true_func_list) ** 2, axis=1))
    rmserr_list = to_numpy(rmserr_list)

    print("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))
    logger.info("all function RMS error: {:.1e}+/-{:.1e}".format(np.mean(rmserr_list), np.std(rmserr_list)))