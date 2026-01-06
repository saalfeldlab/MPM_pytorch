import time
from shutil import copyfile
import shutil
import yaml
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
import subprocess

from MPM_pytorch.config import MPM_pytorchConfig
from MPM_pytorch.generators.graph_data_generator import *
from MPM_pytorch.models.graph_trainer import *
from MPM_pytorch.models.Siren_Network import *
from MPM_pytorch.models.utils import *
from MPM_pytorch.models.exploration_tree import compute_ucb_scores, save_exploration_artifacts
from MPM_pytorch.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    # try:
    #     matplotlib.use("Qt5Agg")
    # except:
    #     pass

    parser = argparse.ArgumentParser(description="MPM_pytorch")
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
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        task = 'train_INR_Claude'  # 'generate', 'train', 'test', 'train_INR', 'Claude'
        best_model = ''
        config_list = ['multimaterial_1_discs_3types']
        task_params = {'iterations': 128} 

        # ouput in MPM/graphs_data/multimaterial/
        # other config files to be found in ./config/*.yaml
        # out of memory: diminish n_particles

    # resume support: start_iteration parameter (default 1)
    start_iteration = task_params.get('start_iteration', 1)

    # Claude task configuration
    n_iterations = task_params.get('iterations', 5)
    base_config_name = config_list[0] if config_list else 'multimaterial'
    experiment_name = task_params.get('experiment', f'experiment_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # Claude task: duplicate and modify config files
    if 'Claude' in task:
        iteration_range = range(start_iteration, n_iterations + 1)
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"

        for cfg in config_list:
            cfg_file, pre = add_pre_folder(cfg)
            source_config = f"{config_root}/{pre}{cfg}.yaml"
            llm_task_name = cfg + "_Claude"  # You can customize this naming
            target_config = f"{config_root}/{pre}{llm_task_name}.yaml"

            if os.path.exists(source_config):
                shutil.copy2(source_config, target_config)
                print(f"\033[93mcopied {source_config} -> {target_config}\033[0m")

                with open(target_config, 'r') as f:
                    config_data = yaml.safe_load(f)

                # Get Claude-specific parameters from config if they exist
                claude_cfg = config_data.get('claude', {})
                claude_n_epochs = claude_cfg.get('n_epochs', 1)
                claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)
                claude_total_steps = claude_cfg.get('total_steps', 100000)

                # Modify config for Claude task
                config_data['dataset'] = llm_task_name
                config_data['training']['n_epochs'] = claude_n_epochs
                config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
                config_data['description'] = 'designed by Claude'

                with open(target_config, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

                print(f"\033[93mmodified {target_config}: dataset='{llm_task_name}', n_epochs={claude_n_epochs}, data_augmentation_loop={claude_data_augmentation_loop}, total_steps={claude_total_steps}\033[0m")

        # Update config_list to use the Claude-modified config
        config_list = [llm_task_name]
    else:
        iteration_range = range(1, 2)

    for config_file_ in config_list:
        print(" ")
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"
        config_file, pre_folder = add_pre_folder(config_file_)
        config = MPM_pytorchConfig.from_yaml(f"{config_root}/{config_file}.yaml")
        config.dataset = pre_folder + config.dataset
        config.config_file = pre_folder + config_file_
        device = set_device(config.training.device)

        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")

        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            experiment_path = f"{root_dir}/{experiment_name}.md"
            analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
            memory_path = f"{root_dir}/{llm_task_name}_memory.md"

            # check experiment file exists
            if not os.path.exists(experiment_path):
                print(f"\033[91merror: experiment file not found: {experiment_path}\033[0m")
                print(f"\033[93mavailable experiment files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # clear analysis, memory, and reasoning files at start (only if not resuming)
            if start_iteration == 1:
                with open(analysis_path, 'w') as f:
                    f.write(f"# Experiment Log: {config_file_}\n\n")
                print(f"\033[93mcleared {analysis_path}\033[0m")
                # clear reasoning.log for Claude tasks
                reasoning_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                open(reasoning_path, 'w').close()
                print(f"\033[93mcleared {reasoning_path}\033[0m")
                # initialize working memory file
                with open(memory_path, 'w') as f:
                    f.write(f"# Working Memory: {config_file_}\n\n")
                    f.write("## Knowledge Base (accumulated across all blocks)\n\n")
                    f.write("### Regime Comparison Table\n")
                    f.write("| Block | INR Type | Field | n_frames | Best R² | Best slope | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |\n")
                    f.write("|-------|----------|-------|----------|---------|------------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|\n\n")
                    f.write("### Established Principles\n\n")
                    f.write("### Open Questions\n\n")
                    f.write("---\n\n")
                    f.write("## Previous Block Summary\n\n")
                    f.write("---\n\n")
                    f.write("## Current Block (Block 1)\n\n")
                    f.write("### Block Info\n\n")
                    f.write("### Hypothesis\n\n")
                    f.write("### Iterations This Block\n\n")
                    f.write("### Emerging Observations\n\n")
                print(f"\033[93mcleared {memory_path}\033[0m")
            else:
                print(f"\033[93mpreserving {analysis_path} (resuming from iter {start_iteration})\033[0m")
                print(f"\033[93mpreserving {memory_path} (resuming from iter {start_iteration})\033[0m")
            print(f"\033[93m{experiment_name} ({n_iterations} iterations, starting at {start_iteration})\033[0m")
        else:
            iteration_range = range(1, 2)
            
        root_dir = os.path.dirname(os.path.abspath(__file__))
        analysis_log_path = f"{root_dir}/{llm_task_name}_analysis.log"

        for iteration in iteration_range:

            if 'Claude' in task:
                print(f"\n\n\n\033[94miteration {iteration}/{n_iterations}: {config_file_} ===\033[0m")
                # block boundary: erase UCB at start of each n_iter_block-iteration block
                # Read n_iter_block from training config (with fallback to claude section for backward compat)
                with open(f"{config_root}/{config_file}.yaml", 'r') as f:
                    raw_config = yaml.safe_load(f)
                n_iter_block = raw_config.get('training', {}).get('n_iter_block',
                               raw_config.get('claude', {}).get('n_iter_block', 16))
                if iteration > 1 and (iteration - 1) % n_iter_block == 0:
                    ucb_file = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
                    if os.path.exists(ucb_file):
                        os.remove(ucb_file)
                        print(f"\033[93mblock boundary: deleted {ucb_file} (new block)\033[0m")

            # reload config to pick up any changes from previous iteration
            config = MPM_pytorchConfig.from_yaml(f"{config_root}/{config_file}.yaml")
            config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + config_file_

            # open analysis.log for this iteration (write mode)
            log_file = open(analysis_log_path, 'w')

            if "generate" in task:
                data_generate(
                    config,
                    device=device,
                    visualize=True,
                    run_vizualized=0,
                    style="black M grid",   # style options: "black", "latex", "F", "M", "grid"
                                            # - "black": dark background
                                            # - "latex": use LaTeX rendering
                                            # - "F": color by deformation gradient magnitude
                                            # - "M": color by material type
                                            # - "grid": create detailed grid visualization (2D only)
                                            # - default (no "F" or "M"): color by material type
                                            # can combine: e.g., "black F" or "black latex M grid"
                    alpha=1,
                    erase=False,
                    bSave=True,
                    step=10,
                ) # config.simulation.n_frames // 100)

            if "train_INR" in task:
                # Get field_name from raw YAML if Claude task
                if 'Claude' in task:
                    with open(f"{config_root}/{config_file}.yaml", 'r') as f:
                        raw_config = yaml.safe_load(f)
                    field_name = raw_config.get('claude', {}).get('field_name', 'Jp')
                else:
                    field_name = 'Jp'

                # total_steps is now read from config.training.total_steps inside data_train_INR
                data_train_INR(
                    config=config,
                    device=device,
                    field_name=field_name,
                    erase='Claude' in task,
                    log_file=log_file
                )

            log_file.close()

            if 'Claude' in task:
                # Read n_iter_block from raw YAML
                with open(f"{config_root}/{config_file}.yaml", 'r') as f:
                    raw_config = yaml.safe_load(f)
                n_iter_block = raw_config.get('claude', {}).get('n_iter_block', 12)
                block_number = (iteration - 1) // n_iter_block + 1
                iter_in_block = (iteration - 1) % n_iter_block + 1
                is_block_end = iter_in_block == n_iter_block

                exploration_dir = f"{root_dir}/log/Claude_exploration/{experiment_name}"
                artifact_paths = save_exploration_artifacts(
                    root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                    iter_in_block=iter_in_block, block_number=block_number
                )

                # compute UCB scores for Claude to read
                config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
                ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"

                compute_ucb_scores(analysis_path, ucb_path,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)
                print(f"\033[92mUCB scores computed: {ucb_path}\033[0m")

                # plot UCB tree
                ucb_tree_path = f"{artifact_paths['tree_save_dir']}/iter_{iteration:03d}_ucb_tree.png"
                nodes = parse_ucb_scores(ucb_path)
                if nodes:
                    # Get field info, inr_type, and n_training_frames from config
                    with open(f"{config_root}/{config_file}.yaml", 'r') as f:
                        raw_config = yaml.safe_load(f)
                    field_name = raw_config.get('claude', {}).get('field_name', 'Jp')
                    inr_type = raw_config.get('graph_model', {}).get('inr_type', 'siren_txy')
                    n_training_frames = raw_config.get('training', {}).get('n_training_frames', None)
                    field_info = f"{inr_type}, Field: {field_name}, Block {block_number}, Iter {iter_in_block}/{n_iter_block}"

                    plot_ucb_tree(nodes, ucb_tree_path,
                                  title=f"UCB Tree - Iter {iteration}",
                                  field_info=field_info,
                                  n_training_frames=n_training_frames)

                # check files are ready
                time.sleep(2)  # pause to ensure files are written
                activity_path = artifact_paths['activity_path']
                if not os.path.exists(activity_path):
                    print(f"\033[91mwarning: activity image not found at {activity_path}\033[0m")
                if not os.path.exists(analysis_log_path):
                    print(f"\033[91merror: analysis.log not found at {analysis_log_path}\033[0m")
                    continue
                if not os.path.exists(ucb_path):
                    print(f"\033[91merror: ucb_scores.txt not found at {ucb_path}\033[0m")
                    continue
                print(f"\033[92mfiles ready: analysis.log, ucb_scores.txt\033[0m")

                # call Claude CLI for analysis
                print(f"\033[93mClaude analysis...\033[0m")

                claude_prompt = f"""Iteration {iteration}/{n_iterations}
Block info: block {block_number}, iteration {iter_in_block}/{n_iter_block} within block
{">>> BLOCK END <<<" if is_block_end else ""}

Protocol (follow all instructions): {experiment_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config: {config_path}"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '20',
                    '--allowedTools', 'Read', 'Edit'
                ]

                # run with real-time output streaming and token expiry detection
                output_lines = []
                process = subprocess.Popen(
                    claude_cmd,
                    cwd=root_dir,
                    stdout=subprocess.PIPE,
                    stderr=subprocess.STDOUT,
                    text=True,
                    bufsize=1
                )

                # stream output line by line
                for line in process.stdout:
                    print(line, end='', flush=True)
                    output_lines.append(line)

                process.wait()

                # check for OAuth token expiration error
                output_text = ''.join(output_lines)
                if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
                    print(f"\n\033[91m{'='*60}\033[0m")
                    print(f"\033[91mOAuth token expired at iteration {iteration}\033[0m")
                    print(f"\033[93mTo resume:\033[0m")
                    print(f"\033[93m  1. Run: claude /login\033[0m")
                    print(f"\033[93m  2. Then: python run_MPM.py -o {task} {config_file_} start_iteration={iteration}\033[0m")
                    print(f"\033[91m{'='*60}\033[0m")
                    raise SystemExit(1)

                # Save Claude's terminal output to reasoning log (separate from analysis.md)
                reasoning_log_path = analysis_path.replace('_analysis.md', '_reasoning.log')
                if output_text.strip():
                    with open(reasoning_log_path, 'a') as f:
                        f.write(f"\n{'='*60}\n")
                        f.write(f"=== Iteration {iteration} ===\n")
                        f.write(f"{'='*60}\n")
                        f.write(output_text.strip())
                        f.write("\n\n")

                # save protocol file at first iteration of each block
                if iter_in_block == 1:
                    protocol_save_dir = artifact_paths['protocol_save_dir']
                    dst_protocol = f"{protocol_save_dir}/block_{block_number:03d}.md"
                    if os.path.exists(experiment_path):
                        shutil.copy2(experiment_path, dst_protocol)

                # save config snapshot for each iteration
                config_save_dir = f"{exploration_dir}/configs"
                os.makedirs(config_save_dir, exist_ok=True)
                dst_config = f"{config_save_dir}/iter_{iteration:03d}_config.yaml"
                if os.path.exists(config_path):
                    shutil.copy2(config_path, dst_config)

                # save memory file at end of each block (after Claude updates it)
                if is_block_end:
                    memory_save_dir = f"{exploration_dir}/memory"
                    os.makedirs(memory_save_dir, exist_ok=True)
                    dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
                    if os.path.exists(memory_path):
                        shutil.copy2(memory_path, dst_memory)
                        print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

                # recompute UCB scores after Claude (for next iteration)
                compute_ucb_scores(analysis_path, ucb_path,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)





# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"
