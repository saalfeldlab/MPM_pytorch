import time
from shutil import copyfile
import shutil
import re
import yaml
import argparse
import networkx as nx
import os
import sys
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

# Import git tracking functionality
from MPM_pytorch.git_code_tracker import track_code_modifications, git_push, is_git_repo, get_modified_code_files

import warnings
warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")


def detect_last_iteration(analysis_path):
    """Detect the last completed iteration from analysis.md.

    Scans for '## Iter N:' entries written by Claude after each iteration.
    Returns the next iteration to run (1-indexed), or 1 if nothing found.
    """
    found_iters = set()
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))
    if not found_iters:
        return 1
    return max(found_iters) + 1


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
    parser.add_argument(
        "--fresh", action="store_true", default=True, help="start from iteration 1 (default)"
    )
    parser.add_argument(
        "--resume", action="store_true", help="auto-resume from last completed iteration"
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
        task = 'train_INR_Claude_code'  # 'generate', 'train', 'test', 'train_INR', 'Claude', 'Claude_code'
        best_model = ''
        config_list = ['multimaterial_1_discs_3types']
        task_params = {'iterations': 1024} 

        # ouput in MPM/graphs_data/multimaterial/
        # other config files to be found in ./config/*.yaml
        # out of memory: diminish n_particles

    # Claude task configuration
    n_iterations = task_params.get('iterations', 5)
    base_config_name = config_list[0] if config_list else 'multimaterial'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # Auto-resume or fresh start
    _root = os.path.dirname(os.path.abspath(__file__))
    if args.resume:
        start_iteration = detect_last_iteration(f"{_root}/{llm_task_name}_analysis.md")
        if start_iteration > 1:
            print(f"\033[93mResuming from iteration {start_iteration}\033[0m")
        else:
            print(f"\033[93mNo previous iterations found, starting fresh\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{_root}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print(f"\033[91mWARNING: Fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            print(f"\033[91m  {_root}/{llm_task_name}_memory.md\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print(f"\033[93mFresh start\033[0m")

    # Claude task: duplicate and modify config files
    if 'Claude' in task:
        iteration_range = range(start_iteration, n_iterations + 1)
        config_root = os.path.dirname(os.path.abspath(__file__)) + "/config"

        for cfg in config_list:
            cfg_file, pre = add_pre_folder(cfg)
            source_config = f"{config_root}/{pre}{cfg}.yaml"
            llm_task_name = cfg + "_Claude"  # You can customize this naming
            target_config = f"{config_root}/{pre}{llm_task_name}.yaml"

            # Only copy and initialize config on fresh start (not when resuming)
            if start_iteration == 1 and not args.resume:
                if os.path.exists(source_config):
                    shutil.copy2(source_config, target_config)
                    print(f"\033[93mcopied {source_config} -> {target_config}\033[0m")

                    with open(target_config, 'r') as f:
                        config_data = yaml.safe_load(f)

                    # Get Claude-specific parameters from config if they exist
                    claude_cfg = config_data.get('claude', {})
                    claude_n_epochs = claude_cfg.get('n_epochs', 1)
                    claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)

                    # Modify config for Claude task
                    # Add folder prefix to dataset name (same as done at runtime in line 150)
                    # This is needed for subprocess to find the correct data path
                    if pre:  # pre is the folder prefix (e.g., 'multimaterial/')
                        config_data['dataset'] = pre + llm_task_name
                    else:
                        config_data['dataset'] = llm_task_name
                    config_data['training']['n_epochs'] = claude_n_epochs
                    config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
                    config_data['description'] = 'designed by Claude'

                    with open(target_config, 'w') as f:
                        yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)

                    print(f"\033[93mmodified {target_config}: dataset='{llm_task_name}', n_epochs={claude_n_epochs}, data_augmentation_loop={claude_data_augmentation_loop}\033[0m")
            else:
                # Check if resuming from a parallel run (_00.yaml exists)
                parallel_slot0_config = f"{config_root}/{pre}{llm_task_name}_00.yaml"
                if os.path.exists(parallel_slot0_config):
                    shutil.copy2(parallel_slot0_config, target_config)
                    print(f"\033[93mcopied {parallel_slot0_config} -> {target_config} (resuming from parallel run)\033[0m")
                else:
                    print(f"\033[93mpreserving {target_config} (resuming from iter {start_iteration})\033[0m")

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
        device = set_device('cuda:1')

        print(f"config_file  {config.config_file}")
        print(f"\033[92mdevice  {device}\033[0m")

        if 'Claude' in task:
            root_dir = os.path.dirname(os.path.abspath(__file__))
            instruction_path = f"{root_dir}/{instruction_name}.md"
            analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
            memory_path = f"{root_dir}/{llm_task_name}_memory.md"

            # check instruction file exists
            if not os.path.exists(instruction_path):
                print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
                print(f"\033[93mavailable instruction files:\033[0m")
                for f in os.listdir(root_dir):
                    if f.endswith('.md') and not f.startswith('analysis_') and not f.startswith('README'):
                        print(f"  - {f[:-3]}")
                continue

            # clear analysis, memory, and reasoning files at start (only if not resuming)
            if start_iteration == 1 and not args.resume:
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
                    f.write("| Block | INR Type | Field | n_training_frames | Best R² | Best slope | kino_R2 | kino_SSIM | Optimal lr_NNR_f | Optimal hidden_dim | Optimal n_layers | Optimal omega_f | Optimal total_steps | Training time (min) | Key finding |\n")
                    f.write("|-------|----------|-------|-------------------|---------|------------|---------|-----------|------------------|--------------------|--------------------|-----------------|---------------------|---------------------|-------------|\n\n")
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
                # Note: videos folder is created by save_exploration_artifacts in instruction-specific dir
            else:
                print(f"\033[93mpreserving {analysis_path} (resuming from iter {start_iteration})\033[0m")
                print(f"\033[93mpreserving {memory_path} (resuming from iter {start_iteration})\033[0m")
            print(f"\033[93m{instruction_name} ({n_iterations} iterations, starting at {start_iteration})\033[0m")
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

                # For Claude tasks, run training in subprocess to reload modified code
                # This ensures that any changes to Siren_Network.py or graph_trainer.py
                # made by Claude in previous iterations are picked up
                if 'Claude' in task:
                    print(f"\033[93mrunning INR training in subprocess...\033[0m")

                    # Construct subprocess command
                    train_script = os.path.join(root_dir, 'train_INR_subprocess.py')
                    config_path = f"{config_root}/{config_file}.yaml"

                    # Create log directory and error log paths (overwrite for each iteration to only keep latest)
                    log_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
                    os.makedirs(log_dir, exist_ok=True)
                    error_log_path = f"{log_dir}/training_output_latest.log"
                    error_details_path = f"{log_dir}/training_error_latest.log"

                    train_cmd = [
                        sys.executable,  # Use same Python interpreter
                        '-u',  # Force unbuffered output for real-time streaming
                        train_script,
                        '--config', config_path,
                        '--field_name', field_name,
                        '--device', str(device),
                        '--log_file', analysis_log_path,
                        '--config_file', config.config_file,  # Pass config_file for proper log directory
                        '--error_log', error_details_path  # Pass error log path for detailed error capture
                    ]
                    if 'Claude' in task:
                        train_cmd.append('--erase')

                    # Run training subprocess with repair loop
                    # Set environment to force tqdm to work in non-interactive mode
                    env = os.environ.copy()
                    env['PYTHONUNBUFFERED'] = '1'
                    env['TQDM_MININTERVAL'] = '1.0'  # Force tqdm to update less frequently

                    # Code files that Claude might modify
                    code_files = [
                        'src/MPM_pytorch/models/Siren_Network.py',
                        'src/MPM_pytorch/models/graph_trainer.py',
                        'src/MPM_pytorch/generators/graph_data_generator.py',
                    ]

                    max_repair_attempts = 10
                    training_success = False
                    error_traceback = None

                    for repair_attempt in range(max_repair_attempts + 1):
                        process = subprocess.Popen(
                            train_cmd,
                            stdout=subprocess.PIPE,
                            stderr=subprocess.STDOUT,
                            text=True,
                            bufsize=1,
                            env=env
                        )

                        # Capture all output for logging while also streaming to console
                        output_lines = []
                        with open(error_log_path, 'w') as output_file:
                            for line in process.stdout:
                                print(line, end='', flush=True)
                                output_file.write(line)
                                output_file.flush()
                                output_lines.append(line.rstrip())

                        process.wait()

                        if process.returncode == 0:
                            training_success = True
                            break

                        # Training failed - capture error info
                        error_traceback = '\n'.join(output_lines[-50:])  # Last 50 lines

                        if repair_attempt == 0:
                            print(f"\033[91m\ntraining subprocess failed with code {process.returncode}\033[0m")
                            print(f"\033[93mthis may indicate a code modification error.\033[0m\n")

                            # Show last 20 lines of output for context
                            print(f"\033[93mLast 20 lines of output:\033[0m")
                            print("-" * 80)
                            for line in output_lines[-20:]:
                                print(line)
                            print("-" * 80)

                            # Show paths to log files
                            print(f"\nFull output logged to: {error_log_path}")
                            if os.path.exists(error_details_path):
                                print(f"Error details logged to: {error_details_path}")
                                try:
                                    with open(error_details_path, 'r') as f:
                                        error_details = f.read()
                                    if error_details.strip():
                                        print(f"\n\033[91mDetailed error information:\033[0m")
                                        print(error_details)
                                        error_traceback = error_details + '\n' + error_traceback
                                except Exception as e:
                                    print(f"Could not read error details: {e}")

                        # Check if code was modified (only attempt repair for code errors)
                        modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []

                        if not modified_code and repair_attempt == 0:
                            print(f"\033[93mNo code modifications detected - skipping repair attempts\033[0m")
                            break

                        # Attempt repair only if code was modified
                        if repair_attempt < max_repair_attempts and modified_code:
                            print(f"\033[93mRepair attempt {repair_attempt + 1}/{max_repair_attempts}: Asking Claude to fix the code error...\033[0m")

                            repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Attempt {repair_attempt + 1}/{max_repair_attempts}

Error traceback:
```
{error_traceback[-3000:] if error_traceback else 'No traceback available'}
```

Modified code files that may contain the bug:
{chr(10).join(f'- {root_dir}/{f}' for f in modified_code)}

Instructions:
1. Read the error traceback carefully
2. Identify the bug in the modified code
3. Fix the bug using the Edit tool
4. Do NOT make other changes, only fix the crash

If you cannot fix it, say "CANNOT_FIX" and explain why."""

                            repair_cmd = [
                                'claude',
                                '-p', repair_prompt,
                                '--output-format', 'text',
                                '--max-turns', '10',
                                '--allowedTools', 'Read', 'Edit', 'Write'
                            ]

                            repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                            repair_output = repair_result.stdout

                            if 'CANNOT_FIX' in repair_output:
                                print(f"\033[91mClaude cannot fix the error\033[0m")
                                break

                            print(f"\033[92mRepair attempt {repair_attempt + 1} complete, retrying training...\033[0m")

                    # If still failing after all attempts, rollback and skip iteration
                    if not training_success:
                        print(f"\033[91mAll repair attempts failed - rolling back code changes\033[0m")

                        # Rollback modified files using git
                        if is_git_repo(root_dir):
                            for file_path in code_files:
                                try:
                                    subprocess.run(['git', 'checkout', 'HEAD', '--', file_path],
                                                  cwd=root_dir, capture_output=True, timeout=10)
                                except:
                                    pass
                            print(f"\033[93mRolled back code to last working state\033[0m")

                        # Log failed modification to memory
                        if os.path.exists(memory_path):
                            with open(memory_path, 'a') as f:
                                f.write(f"\n### Failed Code Modification (Iter {iteration})\n")
                                f.write(f"Error: {error_traceback[-500:] if error_traceback else 'Unknown'}\n")
                                f.write(f"**DO NOT retry this modification**\n\n")

                        continue  # Skip to next iteration

                    print(f"\033[92mtraining subprocess completed successfully\033[0m")

                    # Copy videos and montages to iteration-specific names
                    dataset_dir = f"{root_dir}/graphs_data/{config.dataset}"

                    # Copy simulation video if it exists
                    input_video = f"{dataset_dir}/fig.mp4"
                    video_path = f"{dataset_dir}/video_iter_{iteration:03d}.mp4"
                    if os.path.exists(input_video):
                        shutil.copy2(input_video, video_path)
                        print(f"\033[92mCopied video: {video_path}\033[0m")

                    # Copy grid video if it exists
                    grid_video = f"{dataset_dir}/grid.mp4"
                    grid_video_path = f"{dataset_dir}/grid_iter_{iteration:03d}.mp4"
                    if os.path.exists(grid_video):
                        shutil.copy2(grid_video, grid_video_path)
                        print(f"\033[92mCopied grid video: {grid_video_path}\033[0m")

                    # Copy first frame as montage/thumbnail if it exists
                    input_image = f"{dataset_dir}/input_fig.png"
                    montage_path = f"{dataset_dir}/montage_iter_{iteration:03d}.png"
                    if os.path.exists(input_image):
                        shutil.copy2(input_image, montage_path)
                        print(f"\033[92mCopied montage: {montage_path}\033[0m")

                else:
                    # For non-Claude tasks, run directly (no code modifications expected)
                    data_train_INR(
                        config=config,
                        device=device,
                        field_name=field_name,
                        erase=False,
                        log_file=log_file,
                        current_iteration=iteration
                    )

            log_file.close()

            if 'Claude' in task:
                # Read n_iter_block from raw YAML
                with open(f"{config_root}/{config_file}.yaml", 'r') as f:
                    raw_config = yaml.safe_load(f)
                n_iter_block = raw_config.get('claude', {}).get('n_iter_block', 12)
                ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)
                block_number = (iteration - 1) // n_iter_block + 1
                iter_in_block = (iteration - 1) % n_iter_block + 1
                is_block_end = iter_in_block == n_iter_block

                code_changes_enabled = 'code' in task

                exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}"
                artifact_paths = save_exploration_artifacts(
                    root_dir, exploration_dir, config, config_file_, pre_folder, iteration,
                    iter_in_block=iter_in_block, block_number=block_number
                )

                # Copy videos to exploration videos folder
                # Video is generated at: ./log/{config_file}/tmp_training/field/field_comparison_{field_name}.mp4
                videos_save_dir = artifact_paths['videos_save_dir']
                log_dir = f"{root_dir}/log/{config.config_file}"
                with open(f"{config_root}/{config_file}.yaml", 'r') as f:
                    raw_config = yaml.safe_load(f)
                field_name = raw_config.get('claude', {}).get('field_name', 'Jp')
                video_src = f"{log_dir}/tmp_training/field/field_comparison_{field_name}.mp4"
                if os.path.exists(video_src):
                    video_dst = f"{videos_save_dir}/iter_{iteration:03d}_{config_file_}.mp4"
                    shutil.copy2(video_src, video_dst)
                    print(f"\033[92mCopied video to: {video_dst}\033[0m")
                else:
                    print(f"\033[93mVideo not found: {video_src}\033[0m")

                # compute UCB scores for Claude to read
                config_path = f"{root_dir}/config/{pre_folder}{config_file_}.yaml"
                ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"

                compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)
                print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

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

Instructions (follow all instructions): {instruction_path}
Working memory: {memory_path}
Full log (append only): {analysis_path}
Metrics log: {analysis_log_path}
UCB scores: {ucb_path}
Current config: {config_path}"""

                # Add code file paths when code changes are enabled
                if code_changes_enabled:
                    claude_prompt += f"""

Code files you can modify (see Step 5.2 in instructions):
- {root_dir}/src/MPM_pytorch/models/Siren_Network.py
- {root_dir}/src/MPM_pytorch/models/graph_trainer.py"""

                claude_cmd = [
                    'claude',
                    '-p', claude_prompt,
                    '--output-format', 'text',
                    '--max-turns', '100',
                    '--allowedTools', 'Read', 'Edit', 'Write'
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

                # Git tracking: commit any code modifications made by Claude
                if is_git_repo(root_dir):
                    print(f"\n\033[96mchecking for code modifications to commit\033[0m")
                    git_results = track_code_modifications(
                        root_dir=root_dir,
                        iteration=iteration,
                        analysis_path=analysis_path,
                        reasoning_path=reasoning_log_path
                    )

                    if git_results:
                        for file_path, success, message in git_results:
                            if success:
                                print(f"\033[92m✓ Git: {message}\033[0m")
                            else:
                                print(f"\033[93m⚠ Git: {message}\033[0m")
                    else:
                        print(f"\033[90m  No code modifications detected\033[0m")
                else:
                    if iteration == 1:
                        print(f"\033[90m  Not a git repository - code modifications will not be version controlled\033[0m")

                # save instruction file at first iteration of each block
                if iter_in_block == 1:
                    instruction_save_dir = artifact_paths['protocol_save_dir']
                    dst_instruction = f"{instruction_save_dir}/block_{block_number:03d}.md"
                    if os.path.exists(instruction_path):
                        shutil.copy2(instruction_path, dst_instruction)

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
                compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                                   current_log_path=analysis_log_path,
                                   current_iteration=iteration,
                                   block_size=n_iter_block)





# bsub -n 4 -gpu "num=1" -q gpu_h100 -Is "python GNN_particles_Ntype.py"

