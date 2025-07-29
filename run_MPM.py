import time
from shutil import copyfile
import argparse
import networkx as nx
import os
import scipy.io
import umap
import torch
import torch.nn as nn
import torch_geometric.data as data
from sklearn import metrics
from tifffile import imread
from torch_geometric.loader import DataLoader
from torch_geometric.utils.convert import to_networkx
from scipy.optimize import curve_fit
from scipy.spatial import Delaunay
from torchvision.transforms import GaussianBlur
import matplotlib

matplotlib.use("Agg")  # Non-interactive backend
import matplotlib.pyplot as plt

from matplotlib import rc
from matplotlib.ticker import FuncFormatter
from prettytable import PrettyTable

from MPM_pytorch.config import ParticleGraphConfig
from MPM_pytorch.generators.graph_data_generator import *
from MPM_pytorch.models.graph_trainer import *
from MPM_pytorch.models.Siren_Network import *
from MPM_pytorch.models.utils import *

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    parser = argparse.ArgumentParser(description="ParticleGraph")
    parser.add_argument(
        "-o", "--option", nargs="+", help="Option that takes multiple values"
    )

    args = parser.parse_args()

    if args.option:
        print(f"Options: {args.option}")
    if args.option != None:
        task = args.option[0]
        config_list = [args.option[1]]
        if len(args.option) > 2:
            best_model = args.option[2]
        else:
            best_model = None
    else:
        task = 'generate'  # 'generate', 'train', 'test'
        best_model = ''
        # config_list = ['multimaterial_4_0']
        config_list = ['multimaterial_2_2_stars']

        # viz in MPM/graphs_data/multimaterial/multimaterial_4_0_3D/Fig
        # other config files to be found in ./config/*.yaml
        # out of memory: diminish n_particles

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = ParticleGraphConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")
        print(f"data folder  {config.dataset}")

        if "generate" in task:
            data_generate(
                config,
                device=device,
                visualize=True,
                run_vizualized=0,
                style="black color",
                alpha=1,
                erase=False,
                bSave=True,
                step=20,
            )  # config.simulation.n_frames // 100)



# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"
