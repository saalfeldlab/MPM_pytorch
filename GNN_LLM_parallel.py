"""
GNN_LLM_parallel.py — Parallel LLM-guided INR exploration loop for MPM-pytorch.

Runs N_PARALLEL=4 INR training jobs per batch (cluster or local), then calls
Claude to analyze results and propose next mutations.

Usage:
    # Local (sequential within batch):
    python GNN_LLM_parallel.py --resume -o train_INR_Claude_code multimaterial_1_discs_3types

    # Cluster (parallel on A100):
    python GNN_LLM_parallel.py --resume -o train_INR_Claude_code_cluster multimaterial_1_discs_3types
"""

import time
import shutil
import re
import yaml
import argparse
import os
import sys
import subprocess
import warnings

import matplotlib
matplotlib.use("Agg")

from MPM_pytorch.config import MPM_pytorchConfig
from MPM_pytorch.models.utils import set_device, add_pre_folder
from MPM_pytorch.models.exploration_tree import compute_ucb_scores, save_exploration_artifacts
from MPM_pytorch.models.plot_exploration_tree import parse_ucb_scores, plot_ucb_tree
from MPM_pytorch.git_code_tracker import track_code_modifications, git_push, is_git_repo, get_modified_code_files

N_PARALLEL = 4


# ---------------------------------------------------------------------------
# Resume helpers
# ---------------------------------------------------------------------------

def detect_last_iteration(analysis_path, config_save_dir, n_parallel):
    """Detect where to resume from, handling sequential→parallel transition.

    Two modes:
    1. No parallel snapshots exist (resuming from sequential GNN_LLM.py):
       → return last_completed_iter + 1
    2. Parallel snapshots exist (resuming from interrupted parallel run):
       → check if last batch is complete; redo incomplete batch

    Returns the iteration number to START FROM (next batch start).
    """
    found_iters = set()

    # Source 1: analysis.md — look for "## Iter N:" lines
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            for line in f:
                match = re.match(r'^##+ Iter (\d+):', line)
                if match:
                    found_iters.add(int(match.group(1)))

    # Source 2: parallel config snapshots (iter_NNN_slot_SS.yaml)
    parallel_snapshot_iters = set()
    if config_save_dir and os.path.isdir(config_save_dir):
        for fname in os.listdir(config_save_dir):
            m = re.match(r'iter_(\d+)_slot_\d+\.yaml', fname)
            if m:
                parallel_snapshot_iters.add(int(m.group(1)))
                found_iters.add(int(m.group(1)))

    if not found_iters:
        return 1

    last_iter = max(found_iters)

    if not parallel_snapshot_iters:
        # No parallel runs yet — transitioning from sequential exploration.
        # Simply start from the next iteration after the last completed one.
        return last_iter + 1

    # Parallel resume: check if the last batch is complete.
    # Batch alignment is anchored to the first parallel iteration.
    first_parallel = min(parallel_snapshot_iters)
    batch_idx = (last_iter - first_parallel) // n_parallel
    batch_start = first_parallel + batch_idx * n_parallel

    # Check: do all iterations in this batch have analysis entries?
    batch_complete = all(
        i in found_iters
        for i in range(batch_start, batch_start + n_parallel)
    )

    if batch_complete:
        return batch_start + n_parallel  # Next batch
    else:
        return batch_start  # Redo incomplete batch


# ---------------------------------------------------------------------------
# Cluster helpers
# ---------------------------------------------------------------------------

CLUSTER_HOME = "/groups/saalfeld/home/allierc"
CLUSTER_ROOT_DIR = f"{CLUSTER_HOME}/Graph/MPM_pytorch"


def submit_cluster_job(slot, config_path, analysis_log_path, config_file_field,
                       field_name, log_dir, root_dir, erase=True, node_name='a100'):
    """Submit a single INR training job to the cluster WITHOUT -K (non-blocking).

    Returns the LSF job ID string, or None if submission failed.
    """
    cluster_script_path = f"{log_dir}/cluster_train_{slot:02d}.sh"
    error_details_path = f"{log_dir}/training_error_{slot:02d}.log"

    # Build cluster-side paths
    cluster_config_path = config_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_analysis_log = analysis_log_path.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_error_log = error_details_path.replace(root_dir, CLUSTER_ROOT_DIR)

    cluster_train_cmd = (
        f"python train_INR_subprocess.py"
        f" --config '{cluster_config_path}'"
        f" --field_name '{field_name}'"
        f" --device cuda"
        f" --log_file '{cluster_analysis_log}'"
        f" --config_file '{config_file_field}'"
        f" --error_log '{cluster_error_log}'"
    )
    if erase:
        cluster_train_cmd += " --erase"

    with open(cluster_script_path, 'w') as f:
        f.write("#!/bin/bash\n")
        f.write(f"cd {CLUSTER_ROOT_DIR}\n")
        f.write(f"conda run -n MPM-pytorch {cluster_train_cmd}\n")
    os.chmod(cluster_script_path, 0o755)

    cluster_script = cluster_script_path.replace(root_dir, CLUSTER_ROOT_DIR)

    # Cluster-side log paths
    cluster_log_dir = log_dir.replace(root_dir, CLUSTER_ROOT_DIR)
    cluster_stdout = f"{cluster_log_dir}/cluster_train_{slot:02d}.out"
    cluster_stderr = f"{cluster_log_dir}/cluster_train_{slot:02d}.err"

    # Submit WITHOUT -K (non-blocking); A100 queue
    ssh_cmd = (
        f"ssh allierc@login1 \"cd {CLUSTER_ROOT_DIR} && "
        f"bsub -n 8 -gpu 'num=1' -q gpu_{node_name} -W 6000 "
        f"-o '{cluster_stdout}' -e '{cluster_stderr}' "
        f"'bash {cluster_script}'\""
    )
    print(f"\033[96m  slot {slot}: submitting via SSH\033[0m")
    result = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

    match = re.search(r'Job <(\d+)>', result.stdout)
    if match:
        job_id = match.group(1)
        print(f"\033[92m  slot {slot}: job {job_id} submitted\033[0m")
        return job_id
    else:
        print(f"\033[91m  slot {slot}: submission FAILED\033[0m")
        print(f"    stdout: {result.stdout.strip()}")
        print(f"    stderr: {result.stderr.strip()}")
        return None


def wait_for_cluster_jobs(job_ids, log_dir=None, poll_interval=60):
    """Poll bjobs via SSH until all jobs finish.

    Args:
        job_ids: dict {slot: job_id_string}
        log_dir: local directory where cluster_train_XX.err files are written
        poll_interval: seconds between polls

    Returns:
        dict {slot: bool} — True if DONE, False if EXIT/failed
    """
    pending = dict(job_ids)
    results = {}

    while pending:
        ids_str = ' '.join(pending.values())
        ssh_cmd = f'ssh allierc@login1 "bjobs {ids_str} 2>/dev/null"'
        out = subprocess.run(ssh_cmd, shell=True, capture_output=True, text=True)

        for slot, jid in list(pending.items()):
            for line in out.stdout.splitlines():
                if jid in line:
                    if 'DONE' in line:
                        results[slot] = True
                        del pending[slot]
                        print(f"\033[92m  slot {slot} (job {jid}): DONE\033[0m")
                    elif 'EXIT' in line:
                        results[slot] = False
                        del pending[slot]
                        print(f"\033[91m  slot {slot} (job {jid}): FAILED (EXIT)\033[0m")
                        if log_dir:
                            err_file = f"{log_dir}/cluster_train_{slot:02d}.err"
                            if os.path.exists(err_file):
                                try:
                                    with open(err_file, 'r') as ef:
                                        err_content = ef.read().strip()
                                    if err_content:
                                        print(f"\033[91m  --- slot {slot} error log ---\033[0m")
                                        for eline in err_content.splitlines()[-30:]:
                                            print(f"\033[91m    {eline}\033[0m")
                                        print(f"\033[91m  --- end error log ---\033[0m")
                                except Exception:
                                    pass

            # Job disappeared from bjobs — assume DONE
            if slot in pending and jid not in out.stdout:
                results[slot] = True
                del pending[slot]
                print(f"\033[93m  slot {slot} (job {jid}): no longer in queue (assuming DONE)\033[0m")

        if pending:
            statuses = [f"slot {s}" for s in pending]
            print(f"\033[90m  ... waiting for {', '.join(statuses)} ({poll_interval}s)\033[0m")
            time.sleep(poll_interval)

    return results


# ---------------------------------------------------------------------------
# Claude CLI helper
# ---------------------------------------------------------------------------

def run_claude_cli(prompt, root_dir, max_turns=500):
    """Run Claude CLI with real-time output streaming. Returns output text."""
    claude_cmd = [
        'claude',
        '-p', prompt,
        '--output-format', 'text',
        '--max-turns', str(max_turns),
        '--allowedTools', 'Read', 'Edit', 'Write'
    ]

    output_lines = []
    process = subprocess.Popen(
        claude_cmd,
        cwd=root_dir,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        bufsize=1
    )

    for line in process.stdout:
        print(line, end='', flush=True)
        output_lines.append(line)

    process.wait()
    return ''.join(output_lines)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    warnings.filterwarnings("ignore", category=FutureWarning)
    parser = argparse.ArgumentParser(description="MPM-pytorch — Parallel LLM INR Loop")
    parser.add_argument("-o", "--option", nargs="+", help="option: task config_name [iterations=N]")
    parser.add_argument("--fresh", action="store_true", help="start from iteration 1 (erases existing)")
    parser.add_argument("--resume", action="store_true", default=True, help="auto-resume from last completed batch (default)")

    print()
    device = []
    args = parser.parse_args()

    # Parse options
    if args.option:
        task = args.option[0]
        config_list = [args.option[1]]
        task_params = {}
        for arg in args.option[2:]:
            if '=' in arg:
                key, value = arg.split('=', 1)
                task_params[key] = int(value) if value.isdigit() else value
    else:
        task = 'train_INR_Claude_code_cluster'
        config_list = ['multimaterial_1_discs_3types']
        task_params = {'iterations': 1024}

    n_iterations = task_params.get('iterations', 1024)
    base_config_name = config_list[0] if config_list else 'multimaterial_1_discs_3types'
    instruction_name = task_params.get('instruction', f'instruction_{base_config_name}')
    llm_task_name = task_params.get('llm_task', f'{base_config_name}_Claude')

    # Resolve paths
    root_dir = os.path.dirname(os.path.abspath(__file__))
    config_root = f"{root_dir}/config"
    cfg_file, pre_folder = add_pre_folder(base_config_name)
    source_config = f"{config_root}/{cfg_file}.yaml"

    # Resume detection
    exploration_dir = f"{root_dir}/log/Claude_exploration/{instruction_name}_parallel"
    config_save_dir = f"{exploration_dir}/configs"

    if args.fresh:
        args.resume = False

    if not args.fresh:
        analysis_path_probe = f"{root_dir}/{llm_task_name}_analysis.md"
        start_iteration = detect_last_iteration(analysis_path_probe, config_save_dir, N_PARALLEL)
        if start_iteration > 1:
            print(f"\033[93mResuming from iteration {start_iteration}\033[0m")
        else:
            print(f"\033[93mNo previous iterations found, starting fresh\033[0m")
    else:
        start_iteration = 1
        _analysis_check = f"{root_dir}/{llm_task_name}_analysis.md"
        if os.path.exists(_analysis_check):
            print(f"\033[91mWARNING: Fresh start will erase existing results in:\033[0m")
            print(f"\033[91m  {_analysis_check}\033[0m")
            answer = input("\033[91mContinue? (y/n): \033[0m").strip().lower()
            if answer != 'y':
                print("Aborted.")
                sys.exit(0)
        print(f"\033[93mFresh start\033[0m")

    # Task flags
    cluster_enabled = 'cluster' in task
    code_changes_enabled = 'code' in task

    # Read claude params from source config
    with open(source_config, 'r') as f:
        source_data = yaml.safe_load(f)
    claude_cfg = source_data.get('claude', {})
    n_iter_block = claude_cfg.get('n_iter_block', 8)
    ucb_c = claude_cfg.get('ucb_c', 1.414)
    claude_node_name = claude_cfg.get('node_name', 'a100')

    if start_iteration == 1 and not args.resume:
        print(f"\033[94mCluster node: gpu_{claude_node_name}\033[0m")

    # -----------------------------------------------------------------------
    # Initialize N_PARALLEL slot configs
    # -----------------------------------------------------------------------
    config_paths = {}
    analysis_log_paths = {}
    slot_names = {}

    # Check if we're resuming from sequential (no slot configs exist yet)
    sequential_config = f"{config_root}/{cfg_file.replace(base_config_name, llm_task_name)}.yaml"
    if not os.path.exists(sequential_config):
        sequential_config = f"{config_root}/{pre_folder}{llm_task_name}.yaml"

    for slot in range(N_PARALLEL):
        slot_name = f"{llm_task_name}_{slot:02d}"
        slot_names[slot] = slot_name
        target = f"{config_root}/{pre_folder}{slot_name}.yaml"
        config_paths[slot] = target
        analysis_log_paths[slot] = f"{root_dir}/{slot_name}_analysis.log"

        if start_iteration == 1 and not args.resume:
            # Fresh start: copy source config to each slot
            shutil.copy2(source_config, target)
            with open(target, 'r') as f:
                config_data = yaml.safe_load(f)

            claude_n_epochs = claude_cfg.get('n_epochs', 1)
            claude_data_augmentation_loop = claude_cfg.get('data_augmentation_loop', 100)

            # Set dataset to Claude data path (all slots read same shared input data)
            # Same logic as sequential GNN_LLM.py line 164-165
            if pre_folder:
                config_data['dataset'] = pre_folder + llm_task_name
            else:
                config_data['dataset'] = llm_task_name
            config_data['training']['n_epochs'] = claude_n_epochs
            config_data['training']['data_augmentation_loop'] = claude_data_augmentation_loop
            # Set n_training_frames to 400 (minimum for high-frame exploration)
            config_data['training']['n_training_frames'] = 400
            config_data['description'] = 'designed by Claude (parallel)'

            with open(target, 'w') as f:
                yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
            print(f"\033[93m  slot {slot}: created {target} (dataset='{config_data['dataset']}')\033[0m")

        elif not os.path.exists(target):
            # Resume from sequential: slot configs don't exist yet — create from sequential config
            if os.path.exists(sequential_config):
                shutil.copy2(sequential_config, target)
                with open(target, 'r') as f:
                    config_data = yaml.safe_load(f)
                # Keep dataset unchanged — all slots share same input data
                config_data['description'] = 'designed by Claude (parallel)'
                with open(target, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"\033[93m  slot {slot}: created from sequential config → {target}\033[0m")
            else:
                # Fallback: copy source and fix dataset path
                shutil.copy2(source_config, target)
                with open(target, 'r') as f:
                    config_data = yaml.safe_load(f)
                if pre_folder:
                    config_data['dataset'] = pre_folder + llm_task_name
                else:
                    config_data['dataset'] = llm_task_name
                config_data['description'] = 'designed by Claude (parallel)'
                with open(target, 'w') as f:
                    yaml.dump(config_data, f, default_flow_style=False, sort_keys=False)
                print(f"\033[93m  slot {slot}: created from source → {target}\033[0m")
        else:
            print(f"\033[93m  slot {slot}: preserving {target} (resuming)\033[0m")

    # When resuming, read node_name from slot 0 config (may have been modified)
    if start_iteration > 1 or args.resume:
        with open(config_paths[0], 'r') as f:
            slot0_data = yaml.safe_load(f)
        claude_node_name = slot0_data.get('claude', {}).get('node_name', claude_node_name)
        print(f"\033[94mCluster node: gpu_{claude_node_name}\033[0m")

    # -----------------------------------------------------------------------
    # Shared file paths
    # -----------------------------------------------------------------------
    analysis_path = f"{root_dir}/{llm_task_name}_analysis.md"
    memory_path = f"{root_dir}/{llm_task_name}_memory.md"
    ucb_path = f"{root_dir}/{llm_task_name}_ucb_scores.txt"
    instruction_path = f"{root_dir}/{instruction_name}.md"
    parallel_instruction_path = f"{root_dir}/{instruction_name}_parallel.md"
    reasoning_log_path = f"{root_dir}/{llm_task_name}_reasoning.log"

    os.makedirs(exploration_dir, exist_ok=True)

    # Check instruction files exist
    if not os.path.exists(instruction_path):
        print(f"\033[91merror: instruction file not found: {instruction_path}\033[0m")
        sys.exit(1)
    if not os.path.exists(parallel_instruction_path):
        print(f"\033[93mwarning: parallel instruction file not found: {parallel_instruction_path}\033[0m")
        parallel_instruction_path = None

    # Initialize shared files on fresh start
    if start_iteration == 1 and not args.resume:
        with open(analysis_path, 'w') as f:
            f.write(f"# Experiment Log: {base_config_name} (parallel)\n\n")
        print(f"\033[93mcleared {analysis_path}\033[0m")
        open(reasoning_log_path, 'w').close()

        with open(memory_path, 'w') as f:
            f.write(f"# Working Memory: {base_config_name} (parallel)\n\n")
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

        if os.path.exists(ucb_path):
            os.remove(ucb_path)
    else:
        print(f"\033[93mpreserving shared files (resuming from iter {start_iteration})\033[0m")

    print(f"\033[93m{instruction_name} PARALLEL (N={N_PARALLEL}, {n_iterations} iterations, starting at {start_iteration})\033[0m")

    # -----------------------------------------------------------------------
    # BATCH 0: Claude "start" call — initialize 4 config variations
    # -----------------------------------------------------------------------
    if start_iteration == 1 and not args.resume:
        print(f"\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH 0: Claude initializing {N_PARALLEL} config variations\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        slot_list = "\n".join(
            f"  Slot {s}: {config_paths[s]}"
            for s in range(N_PARALLEL)
        )

        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        start_prompt = f"""PARALLEL START: Initialize {N_PARALLEL} config variations for the first batch.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}

Config files to edit (all {N_PARALLEL}):
{slot_list}

Read the instructions and the appendix, then create {N_PARALLEL} diverse initial INR training
parameter variations. Each config already has a unique dataset name — do NOT change the
dataset field. Vary training parameters (e.g. learning_rate_NNR_f, hidden_dim_nnr_f,
omega_f, total_steps, n_layers_nnr_f, n_training_frames) across the {N_PARALLEL} slots to
explore different starting points. n_training_frames starts at 400 — use the appendix
starting configurations (Section 6) as a foundation. Do NOT use n_training_frames ≤ 200.

Write the planned mutations to the working memory file."""

        print("\033[93mClaude start call...\033[0m")
        output_text = run_claude_cli(start_prompt, root_dir, max_turns=100)

        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91mOAuth token expired during start call\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Then re-run this script\033[0m")
            sys.exit(1)

        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== BATCH 0 (start call) ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

    # -----------------------------------------------------------------------
    # Main batch loop
    # -----------------------------------------------------------------------
    for batch_start in range(start_iteration, n_iterations + 1, N_PARALLEL):
        iterations = [batch_start + s for s in range(N_PARALLEL)
                      if batch_start + s <= n_iterations]

        batch_first = iterations[0]
        batch_last = iterations[-1]
        n_slots = len(iterations)

        block_number = (batch_first - 1) // n_iter_block + 1
        iter_in_block_first = (batch_first - 1) % n_iter_block + 1
        iter_in_block_last = (batch_last - 1) % n_iter_block + 1
        is_block_end = any((it - 1) % n_iter_block + 1 == n_iter_block for it in iterations)

        # Block boundary: erase UCB at start of new block
        if batch_first > 1 and (batch_first - 1) % n_iter_block == 0:
            if os.path.exists(ucb_path):
                os.remove(ucb_path)
                print(f"\033[93mblock boundary: deleted {ucb_path}\033[0m")

        print(f"\n\n\033[94m{'='*60}\033[0m")
        print(f"\033[94mBATCH: iterations {batch_first}-{batch_last} / {n_iterations}  (block {block_number})\033[0m")
        print(f"\033[94m{'='*60}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 1: Load configs and submit INR training jobs
        # -------------------------------------------------------------------
        configs = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            config = MPM_pytorchConfig.from_yaml(config_paths[slot])
            if not config.dataset.startswith(pre_folder):
                config.dataset = pre_folder + config.dataset
            config.config_file = pre_folder + slot_names[slot]
            configs[slot] = config

            if device == []:
                device = set_device(config.training.device)

        job_results = {}

        if cluster_enabled:
            print(f"\n\033[93mPHASE 1: Submitting {n_slots} INR training jobs to cluster (A100)\033[0m")

            job_ids = {}
            for slot_idx, iteration in enumerate(iterations):
                slot = slot_idx
                config = configs[slot]

                # Read field_name from slot config
                with open(config_paths[slot], 'r') as f:
                    raw_cfg = yaml.safe_load(f)
                field_name = raw_cfg.get('claude', {}).get('field_name', 'Jp')

                jid = submit_cluster_job(
                    slot=slot,
                    config_path=config_paths[slot],
                    analysis_log_path=analysis_log_paths[slot],
                    config_file_field=config.config_file,
                    field_name=field_name,
                    log_dir=exploration_dir,
                    root_dir=root_dir,
                    erase=True,
                    node_name=claude_node_name
                )
                if jid:
                    job_ids[slot] = jid
                else:
                    job_results[slot] = False

            # Wait for all submitted jobs
            if job_ids:
                print(f"\n\033[93mPHASE 1b: Waiting for {len(job_ids)} cluster jobs\033[0m")
                cluster_results = wait_for_cluster_jobs(job_ids, log_dir=exploration_dir, poll_interval=60)
                job_results.update(cluster_results)

            # Check for training errors — attempt auto-repair
            for slot_idx in range(n_slots):
                if job_results.get(slot_idx) == False:
                    err_content = None
                    err_file = f"{exploration_dir}/training_error_{slot_idx:02d}.log"
                    lsf_err_file = f"{exploration_dir}/cluster_train_{slot_idx:02d}.err"

                    for ef_path in [err_file, lsf_err_file]:
                        if os.path.exists(ef_path):
                            try:
                                with open(ef_path, 'r') as ef:
                                    content = ef.read()
                                if 'TRAINING SUBPROCESS ERROR' in content or 'Traceback' in content:
                                    err_content = content
                                    break
                            except Exception:
                                pass

                    if not err_content:
                        continue

                    print(f"\033[91m  slot {slot_idx}: TRAINING ERROR detected — attempting auto-repair\033[0m")

                    code_files = [
                        'src/MPM_pytorch/models/Siren_Network.py',
                        'src/MPM_pytorch/models/graph_trainer.py',
                        'src/MPM_pytorch/generators/graph_data_generator.py',
                    ]
                    # Include per-slot config file
                    slot_config_rel = os.path.relpath(config_paths[slot_idx], root_dir)
                    code_files.append(slot_config_rel)
                    modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []
                    # Always include config as repairable
                    repair_files = modified_code if modified_code else [slot_config_rel]

                    max_repair_attempts = 3
                    repaired = False
                    for attempt in range(max_repair_attempts):
                        print(f"\033[93m  slot {slot_idx}: repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
                        repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Error traceback:
```
{err_content[-3000:]}
```

Files to check: {chr(10).join(f'- {root_dir}/{f}' for f in repair_files)}

Fix the bug. Do NOT make other changes."""

                        repair_cmd = [
                            'claude', '-p', repair_prompt,
                            '--output-format', 'text', '--max-turns', '10',
                            '--allowedTools', 'Read', 'Edit', 'Write'
                        ]
                        repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                        if 'CANNOT_FIX' in repair_result.stdout:
                            print(f"\033[91m  slot {slot_idx}: Claude cannot fix — stopping repair\033[0m")
                            break

                        # Resubmit repaired slot
                        print(f"\033[96m  slot {slot_idx}: resubmitting after repair\033[0m")
                        with open(config_paths[slot_idx], 'r') as f:
                            raw_cfg = yaml.safe_load(f)
                        field_name = raw_cfg.get('claude', {}).get('field_name', 'Jp')

                        jid = submit_cluster_job(
                            slot=slot_idx,
                            config_path=config_paths[slot_idx],
                            analysis_log_path=analysis_log_paths[slot_idx],
                            config_file_field=configs[slot_idx].config_file,
                            field_name=field_name,
                            log_dir=exploration_dir,
                            root_dir=root_dir,
                            erase=True,
                            node_name=claude_node_name
                        )
                        if jid:
                            retry_results = wait_for_cluster_jobs(
                                {slot_idx: jid}, log_dir=exploration_dir, poll_interval=60
                            )
                            if retry_results.get(slot_idx):
                                job_results[slot_idx] = True
                                repaired = True
                                print(f"\033[92m  slot {slot_idx}: repair successful!\033[0m")
                                break
                            # Reload error for next attempt
                            for ef_path in [err_file, lsf_err_file]:
                                if os.path.exists(ef_path):
                                    try:
                                        with open(ef_path, 'r') as ef:
                                            err_content = ef.read()
                                        break
                                    except Exception:
                                        pass

                    if not repaired:
                        print(f"\033[91m  slot {slot_idx}: repair failed after {max_repair_attempts} attempts — skipping\033[0m")
                        if is_git_repo(root_dir):
                            for fp in code_files:
                                try:
                                    subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                                  cwd=root_dir, capture_output=True, timeout=10)
                                except Exception:
                                    pass

        else:
            # --- Local mode: run INR training sequentially ---
            print(f"\n\033[93mPHASE 1: Running {n_slots} INR trainings locally (sequential)\033[0m")

            env = os.environ.copy()
            env['PYTHONUNBUFFERED'] = '1'
            env['TQDM_MININTERVAL'] = '1.0'

            for slot_idx, iteration in enumerate(iterations):
                slot = slot_idx
                config = configs[slot]
                print(f"\n\033[90m  slot {slot} (iter {iteration}): training INR...\033[0m")

                # Read field_name from slot config
                with open(config_paths[slot], 'r') as f:
                    raw_cfg = yaml.safe_load(f)
                field_name = raw_cfg.get('claude', {}).get('field_name', 'Jp')

                train_script = os.path.join(root_dir, 'train_INR_subprocess.py')
                train_cmd = [
                    sys.executable, '-u', train_script,
                    '--config', config_paths[slot],
                    '--field_name', field_name,
                    '--device', str(device),
                    '--log_file', analysis_log_paths[slot],
                    '--config_file', config.config_file,
                    '--error_log', f"{exploration_dir}/training_error_{slot:02d}.log",
                    '--erase'
                ]

                code_files = [
                    'src/MPM_pytorch/models/Siren_Network.py',
                    'src/MPM_pytorch/models/graph_trainer.py',
                    'src/MPM_pytorch/generators/graph_data_generator.py',
                ]
                slot_config_rel = os.path.relpath(config_paths[slot], root_dir)
                code_files.append(slot_config_rel)

                max_repair_attempts = 3
                success = False

                for attempt in range(max_repair_attempts + 1):
                    process = subprocess.Popen(
                        train_cmd,
                        stdout=subprocess.PIPE,
                        stderr=subprocess.STDOUT,
                        text=True,
                        bufsize=1,
                        env=env
                    )

                    output_lines = []
                    for line in process.stdout:
                        print(line, end='', flush=True)
                        output_lines.append(line.rstrip())

                    process.wait()

                    if process.returncode == 0:
                        success = True
                        break

                    error_traceback = '\n'.join(output_lines[-50:])

                    if attempt == 0:
                        print(f"\033[91m  slot {slot}: training failed\033[0m")

                    modified_code = get_modified_code_files(root_dir, code_files) if is_git_repo(root_dir) else []
                    repair_files = modified_code if modified_code else [slot_config_rel]

                    if attempt < max_repair_attempts:
                        print(f"\033[93m  slot {slot}: repair attempt {attempt + 1}/{max_repair_attempts}\033[0m")
                        repair_prompt = f"""TRAINING CRASHED - Please fix the code error.

Error traceback:
```
{error_traceback[-3000:] if error_traceback else 'No traceback'}
```

Files to check: {chr(10).join(f'- {root_dir}/{f}' for f in repair_files)}

Fix the bug. Do NOT make other changes."""

                        repair_cmd = [
                            'claude', '-p', repair_prompt,
                            '--output-format', 'text', '--max-turns', '10',
                            '--allowedTools', 'Read', 'Edit', 'Write'
                        ]
                        repair_result = subprocess.run(repair_cmd, cwd=root_dir, capture_output=True, text=True)
                        if 'CANNOT_FIX' in repair_result.stdout:
                            break

                if not success:
                    if is_git_repo(root_dir):
                        rollback_files = list(code_files)
                        for fp in rollback_files:
                            try:
                                subprocess.run(['git', 'checkout', 'HEAD', '--', fp],
                                              cwd=root_dir, capture_output=True, timeout=10)
                            except Exception:
                                pass

                job_results[slot] = success

        # -------------------------------------------------------------------
        # PHASE 2: Post-processing — artifacts, videos, config snapshots
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 2: Post-processing successful slots\033[0m")

        activity_paths = {}
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            if not job_results.get(slot, False):
                print(f"\033[90m  slot {slot} (iter {iteration}): skipping (failed)\033[0m")
                continue

            config = configs[slot]
            iter_in_block = (iteration - 1) % n_iter_block + 1

            # Save exploration artifacts
            artifact_paths = save_exploration_artifacts(
                root_dir, exploration_dir, config, slot_names[slot],
                pre_folder, iteration,
                iter_in_block=iter_in_block, block_number=block_number
            )
            activity_paths[slot] = artifact_paths['activity_path']

            # Copy video to exploration folder
            with open(config_paths[slot], 'r') as f:
                raw_cfg = yaml.safe_load(f)
            field_name = raw_cfg.get('claude', {}).get('field_name', 'Jp')
            log_subdir = f"{root_dir}/log/{config.config_file}"
            video_src = f"{log_subdir}/tmp_training/field/field_comparison_{field_name}.mp4"
            if os.path.exists(video_src):
                videos_save_dir = artifact_paths.get('videos_save_dir', f"{exploration_dir}/videos")
                os.makedirs(videos_save_dir, exist_ok=True)
                video_dst = f"{videos_save_dir}/iter_{iteration:03d}_{slot_names[slot]}.mp4"
                shutil.copy2(video_src, video_dst)

            # Save config snapshot
            os.makedirs(config_save_dir, exist_ok=True)
            dst_config = f"{config_save_dir}/iter_{iteration:03d}_slot_{slot:02d}.yaml"
            shutil.copy2(config_paths[slot], dst_config)

        # -------------------------------------------------------------------
        # PHASE 3: Batch UCB update
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 3: Computing UCB scores\033[0m")

        # Re-read ucb_c from first slot config
        with open(config_paths[0], 'r') as f:
            raw_config = yaml.safe_load(f)
        ucb_c = raw_config.get('claude', {}).get('ucb_c', 1.414)

        # Build temporary analysis file with stub entries for all successful slots
        existing_content = ""
        if os.path.exists(analysis_path):
            with open(analysis_path, 'r') as f:
                existing_content = f.read()

        stub_entries = ""
        for slot_idx, iteration in enumerate(iterations):
            if not job_results.get(slot_idx, False):
                continue
            log_path = analysis_log_paths[slot_idx]
            if not os.path.exists(log_path):
                continue
            with open(log_path, 'r') as f:
                log_content = f.read()

            r2_m = re.search(r'final_r2[=:]\s*([\d.eE+-]+|nan)', log_content)
            slope_m = re.search(r'slope[=:]\s*([\d.eE+-]+|nan)', log_content)
            time_m = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            kino_r2_m = re.search(r'kinograph_R2[=:]\s*([\d.eE+-]+|nan)', log_content)
            kino_ssim_m = re.search(r'kinograph_SSIM[=:]\s*([\d.eE+-]+|nan)', log_content)

            if r2_m:
                r2_val = r2_m.group(1)
                slope_val = slope_m.group(1) if slope_m else '0.0'
                time_val = time_m.group(1) if time_m else '0.0'
                kino_r2_val = kino_r2_m.group(1) if kino_r2_m else '0.0'
                kino_ssim_val = kino_ssim_m.group(1) if kino_ssim_m else '0.0'
                # Check if this iteration already exists (resume case)
                if f'## Iter {iteration}:' not in existing_content:
                    stub_entries += (
                        f"\n## Iter {iteration}: pending\n"
                        f"Node: id={iteration}, parent=root\n"
                        f"Metrics: final_r2={r2_val}, slope={slope_val}, "
                        f"kinograph_R2={kino_r2_val}, kinograph_SSIM={kino_ssim_val}, "
                        f"training_time_min={time_val}\n"
                    )

        tmp_analysis = analysis_path + '.tmp_ucb'
        with open(tmp_analysis, 'w') as f:
            f.write(existing_content + stub_entries)

        compute_ucb_scores(
            tmp_analysis, ucb_path, c=ucb_c,
            current_log_path=None,
            current_iteration=batch_last,
            block_size=n_iter_block
        )
        os.remove(tmp_analysis)
        print(f"\033[92mUCB scores computed (c={ucb_c}): {ucb_path}\033[0m")

        # -------------------------------------------------------------------
        # PHASE 4: Claude analyzes results + proposes next N_PARALLEL mutations
        # -------------------------------------------------------------------
        print(f"\n\033[93mPHASE 4: Claude analysis + next mutations\033[0m")

        # Build per-slot info
        slot_info_lines = []
        for slot_idx, iteration in enumerate(iterations):
            slot = slot_idx
            status = "COMPLETED" if job_results.get(slot, False) else "FAILED"
            act_path = activity_paths.get(slot, "N/A")
            slot_info_lines.append(
                f"Slot {slot} (iteration {iteration}) [{status}]:\n"
                f"  Metrics: {analysis_log_paths[slot]}\n"
                f"  Activity: {act_path}\n"
                f"  Config: {config_paths[slot]}"
            )
        slot_info = "\n\n".join(slot_info_lines)

        block_end_marker = "\n>>> BLOCK END <<<" if is_block_end else ""
        parallel_ref = f"\nParallel instructions: {parallel_instruction_path}" if parallel_instruction_path else ""

        claude_prompt = f"""Batch iterations {batch_first}-{batch_last} / {n_iterations}
Block info: block {block_number}, iterations {iter_in_block_first}-{iter_in_block_last}/{n_iter_block} within block{block_end_marker}

PARALLEL MODE: Analyze {n_slots} results, then propose next {N_PARALLEL} mutations.

Instructions (follow all instructions): {instruction_path}{parallel_ref}
Working memory: {memory_path}
Full log (append only): {analysis_path}
UCB scores: {ucb_path}

{slot_info}

Analyze all {n_slots} results. For each successful slot, write a separate iteration entry
(## Iter N: ...) to the full log and memory file. Then edit all {N_PARALLEL} config files
to set up the next batch of {N_PARALLEL} experiments.

IMPORTANT: Do NOT change the 'dataset' field in any config — it must stay as-is for each slot."""

        # Add code file paths when enabled (BLOCK END only)
        if code_changes_enabled and is_block_end:
            claude_prompt += f"""

Code files you can modify (BLOCK END only — for next block):
- {root_dir}/src/MPM_pytorch/models/Siren_Network.py
- {root_dir}/src/MPM_pytorch/models/graph_trainer.py"""

        print("\033[93mClaude analysis...\033[0m")
        output_text = run_claude_cli(claude_prompt, root_dir)

        # Check for OAuth expiration
        if 'OAuth token has expired' in output_text or 'authentication_error' in output_text:
            print(f"\n\033[91m{'='*60}\033[0m")
            print(f"\033[91mOAuth token expired at batch {batch_first}-{batch_last}\033[0m")
            print("\033[93mTo resume:\033[0m")
            print("\033[93m  1. Run: claude /login\033[0m")
            print(f"\033[93m  2. Then: python GNN_LLM_parallel.py --resume -o {task} {base_config_name}\033[0m")
            print(f"\033[91m{'='*60}\033[0m")
            sys.exit(1)

        # Save reasoning
        if output_text.strip():
            with open(reasoning_log_path, 'a') as f:
                f.write(f"\n{'='*60}\n")
                f.write(f"=== Batch {batch_first}-{batch_last} ===\n")
                f.write(f"{'='*60}\n")
                f.write(output_text.strip())
                f.write("\n\n")

        # Git tracking: commit any code modifications made by Claude
        if is_git_repo(root_dir):
            git_results = track_code_modifications(
                root_dir=root_dir,
                iteration=batch_last,
                analysis_path=analysis_path,
                reasoning_path=reasoning_log_path
            )
            if git_results:
                for file_path, success, message in git_results:
                    if success:
                        print(f"\033[92m  Git: {message}\033[0m")
                    else:
                        print(f"\033[93m  Git: {message}\033[0m")

        # Recompute UCB after Claude writes iteration entries to analysis.md
        compute_ucb_scores(analysis_path, ucb_path, c=ucb_c,
                           current_log_path=None,
                           current_iteration=batch_last,
                           block_size=n_iter_block)

        # UCB tree visualization
        should_save_tree = (block_number == 1) or is_block_end
        if should_save_tree:
            tree_save_dir = f"{exploration_dir}/exploration_tree"
            os.makedirs(tree_save_dir, exist_ok=True)
            ucb_tree_path = f"{tree_save_dir}/ucb_tree_iter_{batch_last:03d}.png"
            nodes = parse_ucb_scores(ucb_path)
            if nodes:
                with open(config_paths[0], 'r') as f:
                    raw_config = yaml.safe_load(f)
                field_name = raw_config.get('claude', {}).get('field_name', 'Jp')
                inr_type = raw_config.get('graph_model', {}).get('inr_type', 'siren_txy')
                n_training_frames = raw_config.get('training', {}).get('n_training_frames', None)
                field_info = f"{inr_type}, Field: {field_name}, Block {block_number}"

                plot_ucb_tree(nodes, ucb_tree_path,
                              title=f"UCB Tree - Batch {batch_first}-{batch_last}",
                              field_info=field_info,
                              n_training_frames=n_training_frames)

        # Save instruction file at first iteration of each block
        protocol_save_dir = f"{exploration_dir}/protocol"
        os.makedirs(protocol_save_dir, exist_ok=True)
        if iter_in_block_first == 1:
            dst_instruction = f"{protocol_save_dir}/block_{block_number:03d}.md"
            if os.path.exists(instruction_path):
                shutil.copy2(instruction_path, dst_instruction)

        # Save memory file at end of block
        if is_block_end:
            memory_save_dir = f"{exploration_dir}/memory"
            os.makedirs(memory_save_dir, exist_ok=True)
            dst_memory = f"{memory_save_dir}/block_{block_number:03d}_memory.md"
            if os.path.exists(memory_path):
                shutil.copy2(memory_path, dst_memory)
                print(f"\033[92msaved memory snapshot: {dst_memory}\033[0m")

        # Batch summary
        n_success = sum(1 for v in job_results.values() if v)
        n_failed = sum(1 for v in job_results.values() if not v)
        print(f"\n\033[94mBatch {batch_first}-{batch_last} complete: "
              f"{n_success} succeeded, {n_failed} failed\033[0m")
