"""
Exploration Tree Utilities for MPM INR Training Experiments

Adapted from NeuralGraph exploration_tree.py for MPM INR training experiments.
Computes UCB scores for hyperparameter exploration using PUCT algorithm.
"""

import math
import os
import re
from collections import defaultdict

def compute_ucb_scores(analysis_path, ucb_path, c=1.0, current_log_path=None, current_iteration=None, block_size=12):
    """
    Parse analysis file, build exploration tree, compute UCB scores.

    Args:
        analysis_path: Path to analysis_experiment_*.md file
        ucb_path: Path to write UCB scores output
        c: Exploration constant (default 1.0)
        current_log_path: Path to current iteration's analysis.log (optional)
        current_iteration: Current iteration number (optional)
        block_size: Size of each simulation block (default 12)

    Returns:
        True if UCB scores were computed, False if no nodes found

    Note:
        When block_size > 0 and current_iteration is provided, only nodes
        from the current block are included in UCB scores. Block N covers
        iterations (N*block_size)+1 to (N+1)*block_size.
    """
    nodes = {}
    next_parent_map = {}  # maps iteration N -> parent for iteration N+1 (from "Next: parent=P")

    # parse previous iterations from analysis markdown file
    if os.path.exists(analysis_path):
        with open(analysis_path, 'r') as f:
            content = f.read()

        # Parse nodes from analysis file
        # Format: Node: id=N, parent=P
        # Metrics: ..., final_r2=V, ...
        # Next: parent=P (specifies parent for next iteration)
        current_node = None

        lines = content.split('\n')
        for i, line in enumerate(lines):
            # Match iteration header: ## Iter N: [status] or ### Iter N: [status]
            iter_match = re.match(r'##+ Iter (\d+):', line)
            if iter_match:
                # Save previous node if it has required fields
                if current_node is not None and 'id' in current_node and 'final_r2' in current_node:
                    nodes[current_node['id']] = current_node
                current_iter = int(iter_match.group(1))
                current_node = {'iter': current_iter}
                continue

            # Match Node line
            node_match = re.match(r'Node: id=(\d+), parent=(\d+|None|root)', line)
            if node_match and current_node is not None:
                parsed_id = int(node_match.group(1))
                # Warning only - don't auto-correct as it breaks parent relationships
                if parsed_id != current_iter:
                    print(f"Warning: Node ID mismatch at iteration {current_iter}: found id={parsed_id} (expected id={current_iter})")
                current_node['id'] = parsed_id
                parent_str = node_match.group(2)
                # Treat parent=0, parent=None, or parent=root as root (no parent)
                if parent_str in ('None', '0', 'root'):
                    current_node['parent'] = None
                else:
                    current_node['parent'] = int(parent_str)
                continue

            # Match Next line: specifies parent for the NEXT iteration
            next_match = re.match(r'Next: parent=(\d+|root)', line)
            if next_match and current_node is not None:
                next_parent_str = next_match.group(1)
                if next_parent_str == 'root':
                    next_parent_map[current_node['iter']] = None
                else:
                    next_parent_map[current_node['iter']] = int(next_parent_str)
                continue

            # Match Mutation line
            mutation_match = re.match(r'Mutation: (.+)', line)
            if mutation_match and current_node is not None:
                current_node['mutation'] = mutation_match.group(1).strip()
                continue

            # Match CODE MODIFICATION section (check if code was modified)
            code_mod_match = re.match(r'CODE MODIFICATION:', line)
            if code_mod_match and current_node is not None:
                current_node['has_code_mod'] = True
                # Try to extract the change description from next few lines
                for j in range(i+1, min(i+10, len(lines))):
                    change_match = re.match(r'\s*Change:\s*(.+)', lines[j])
                    if change_match:
                        current_node['code_change'] = change_match.group(1).strip()
                        break
                continue

            # Match Metrics line for final_r2
            metrics_match = re.search(r'final_r2=([\d.]+|nan)', line)
            if metrics_match and current_node is not None:
                r2_str = metrics_match.group(1)
                current_node['final_r2'] = float(r2_str) if r2_str != 'nan' else 0.0
                # Also extract final_mse and slope from same line
                mse_match = re.search(r'final_mse=([\d.eE+-]+|nan)', line)
                if mse_match:
                    mse_str = mse_match.group(1)
                    current_node['final_mse'] = float(mse_str) if mse_str != 'nan' else 0.0
                else:
                    current_node['final_mse'] = 0.0
                slope_match = re.search(r'slope=([\d.]+|nan)', line)
                if slope_match:
                    slope_str = slope_match.group(1)
                    current_node['slope'] = float(slope_str) if slope_str != 'nan' else 0.0
                else:
                    current_node['slope'] = 0.0
                training_time_match = re.search(r'training_time[=:]([\d.]+|N/A)', line)
                if training_time_match:
                    time_str = training_time_match.group(1)
                    current_node['training_time_min'] = float(time_str) if time_str != 'N/A' else 0.0
                else:
                    current_node['training_time_min'] = 0.0
                # Parse kinograph metrics
                kino_r2_match = re.search(r'kinograph_R2=([\d.]+|nan)', line)
                if kino_r2_match:
                    current_node['kinograph_R2'] = float(kino_r2_match.group(1)) if kino_r2_match.group(1) != 'nan' else 0.0
                kino_ssim_match = re.search(r'kinograph_SSIM=([\d.]+|nan)', line)
                if kino_ssim_match:
                    current_node['kinograph_SSIM'] = float(kino_ssim_match.group(1)) if kino_ssim_match.group(1) != 'nan' else 0.0
                continue

        # Save the last node if complete
        if current_node is not None and 'id' in current_node and 'final_r2' in current_node:
            nodes[current_node['id']] = current_node

    # Apply next_parent_map: if iteration N-1 specified "Next: parent=P", use P as parent for node N
    for node_id, node in nodes.items():
        prev_iter = node_id - 1
        if prev_iter in next_parent_map:
            node['parent'] = next_parent_map[prev_iter]

    # Add current iteration from analysis.log if not yet in markdown
    if current_log_path and current_iteration and os.path.exists(current_log_path):
        with open(current_log_path, 'r') as f:
            log_content = f.read()

        # parse final_r2 from analysis.log
        r2_match = re.search(r'final_r2[=:]\s*([\d.]+|nan)', log_content)
        if r2_match:
            r2_str = r2_match.group(1)
            r2_value = float(r2_str) if r2_str != 'nan' else 0.0

            # parse final_mse from analysis.log
            mse_value = 0.0
            mse_match = re.search(r'final_mse[=:]\s*([\d.eE+-]+|nan)', log_content)
            if mse_match:
                mse_str = mse_match.group(1)
                mse_value = float(mse_str) if mse_str != 'nan' else 0.0

            # parse slope from analysis.log
            slope_value = 0.0
            slope_match = re.search(r'slope[=:]\s*([\d.]+|nan)', log_content)
            if slope_match:
                slope_str = slope_match.group(1)
                slope_value = float(slope_str) if slope_str != 'nan' else 0.0

            # parse training_time_min from analysis.log
            training_time_value = 0.0
            time_match = re.search(r'training_time_min[=:]\s*([\d.]+)', log_content)
            if time_match:
                training_time_value = float(time_match.group(1))

            # parse kinograph metrics from analysis.log
            kino_r2_value = 0.0
            kino_r2_m = re.search(r'kinograph_R2[=:]\s*([\d.]+|nan)', log_content)
            if kino_r2_m:
                kino_r2_value = float(kino_r2_m.group(1)) if kino_r2_m.group(1) != 'nan' else 0.0
            kino_ssim_value = 0.0
            kino_ssim_m = re.search(r'kinograph_SSIM[=:]\s*([\d.]+|nan)', log_content)
            if kino_ssim_m:
                kino_ssim_value = float(kino_ssim_m.group(1)) if kino_ssim_m.group(1) != 'nan' else 0.0

            if current_iteration in nodes:
                # Update existing node's metrics
                nodes[current_iteration]['final_r2'] = r2_value
                nodes[current_iteration]['final_mse'] = mse_value
                nodes[current_iteration]['slope'] = slope_value
                nodes[current_iteration]['training_time_min'] = training_time_value
                nodes[current_iteration]['kinograph_R2'] = kino_r2_value
                nodes[current_iteration]['kinograph_SSIM'] = kino_ssim_value
            else:
                # Create new node using parent from previous iteration's "Next: parent=P"
                prev_iter = current_iteration - 1
                parent = next_parent_map.get(prev_iter, prev_iter if prev_iter in nodes else None)
                nodes[current_iteration] = {
                    'iter': current_iteration,
                    'id': current_iteration,
                    'parent': parent,
                    'final_r2': r2_value,
                    'final_mse': mse_value,
                    'slope': slope_value,
                    'training_time_min': training_time_value,
                    'kinograph_R2': kino_r2_value,
                    'kinograph_SSIM': kino_ssim_value
                }

    if not nodes:
        return False

    # Filter nodes to current block if block_size > 0 and current_iteration is provided
    if block_size > 0 and current_iteration is not None:
        current_block = (current_iteration - 1) // block_size
        block_start = current_block * block_size + 1
        block_end = (current_block + 1) * block_size

        # Filter nodes to only include those in current block
        nodes = {node_id: node for node_id, node in nodes.items()
                 if block_start <= node_id <= block_end}

        # Update parent references: if parent is outside block, set to None (root)
        for node_id, node in nodes.items():
            if node['parent'] is not None and node['parent'] not in nodes:
                node['parent'] = None

    if not nodes:
        return False

    # Build tree structure: for each node, track children
    children = defaultdict(list)
    for node_id, node in nodes.items():
        if node['parent'] is not None:
            children[node['parent']].append(node_id)

    # Total number of nodes
    n_total = len(nodes)

    # Compute visits using PUCT backpropagation semantics:
    # - Each node starts with V=1 (its own creation visit)
    # - When a child is created, parent and all ancestors get V += 1
    visits = {node_id: 1 for node_id in nodes}

    # Sort nodes by id to process in creation order (children after parents)
    sorted_node_ids = sorted(nodes.keys())

    # Backpropagate: for each node, increment all ancestors
    for node_id in sorted_node_ids:
        parent_id = nodes[node_id]['parent']
        while parent_id is not None and parent_id in nodes:
            visits[parent_id] += 1
            parent_id = nodes[parent_id]['parent']

    # Compute UCB for each node
    # PUCT formula: UCB(u) = RankScore(u) + c * sqrt(N_total) / (1 + V(u))
    ucb_scores = []
    for node_id, node in nodes.items():
        v = visits[node_id]
        reward = node.get('final_r2', 0.0)

        # PUCT exploration term: c * sqrt(N_total) / (1 + V)
        exploration_term = c * math.sqrt(n_total) / (1 + v)

        ucb = reward + exploration_term

        ucb_scores.append({
            'id': node_id,
            'parent': node['parent'],
            'visits': v,
            'mean_R2': reward,
            'ucb': ucb,
            'final_r2': reward,
            'final_mse': node.get('final_mse', 0.0),
            'slope': node.get('slope', 0.0),
            'training_time_min': node.get('training_time_min', 0.0),
            'kinograph_R2': node.get('kinograph_R2', 0.0),
            'kinograph_SSIM': node.get('kinograph_SSIM', 0.0),
            'mutation': node.get('mutation', ''),
            'has_code_mod': node.get('has_code_mod', False),
            'code_change': node.get('code_change', ''),
            'is_current': node_id == current_iteration
        })

    # Sort by UCB descending (highest UCB = most promising to explore)
    ucb_scores.sort(key=lambda x: x['ucb'], reverse=True)

    # Write UCB scores to file
    with open(ucb_path, 'w') as f:
        # Include block information if block_size > 0
        if block_size > 0 and current_iteration is not None:
            current_block = (current_iteration - 1) // block_size
            block_start = current_block * block_size + 1
            block_end = (current_block + 1) * block_size
            f.write(f"=== UCB Scores (Block {current_block}, iters {block_start}-{block_end}, N={n_total}, c={c}) ===\n\n")
        else:
            f.write(f"=== UCB Scores (N_total={n_total}, c={c}) ===\n\n")
        for score in ucb_scores:
            parent_str = score['parent'] if score['parent'] is not None else 'root'
            mutation_str = score.get('mutation', '')
            has_code_mod = score.get('has_code_mod', False)
            code_change = score.get('code_change', '')

            line = (f"Node {score['id']}: UCB={score['ucb']:.3f}, "
                    f"parent={parent_str}, visits={score['visits']}, "
                    f"R2={score['final_r2']:.3f}, "
                    f"slope={score['slope']:.3f}, "
                    f"kino_R2={score.get('kinograph_R2', 0.0):.3f}, "
                    f"kino_SSIM={score.get('kinograph_SSIM', 0.0):.3f}, "
                    f"time={score['training_time_min']:.1f}min")

            # Add code modification indicator
            if has_code_mod:
                line += f" [CODE]"
                if code_change:
                    line += f" {code_change}"

            # Add config mutation
            if mutation_str:
                line += f", Mutation={mutation_str}"

            f.write(line + "\n")

    return True


def save_exploration_artifacts(root_dir, exploration_dir, config, config_file, pre_folder, iteration, iter_in_block=1, block_number=1):
    """
    Save artifacts for exploration tree visualization.

    Args:
        root_dir: Root directory of the project
        exploration_dir: Directory to save exploration artifacts
        config: Configuration object
        config_file: Config file name
        pre_folder: Pre-folder prefix
        iteration: Current iteration number
        iter_in_block: Iteration within current block
        block_number: Current block number

    Returns:
        Dictionary with paths to saved artifacts
    """
    # Create directories
    tree_save_dir = f"{exploration_dir}/ucb_trees"
    protocol_save_dir = f"{exploration_dir}/protocols"
    videos_save_dir = f"{exploration_dir}/videos"
    os.makedirs(tree_save_dir, exist_ok=True)
    os.makedirs(protocol_save_dir, exist_ok=True)
    os.makedirs(videos_save_dir, exist_ok=True)

    # Path to activity visualization (placeholder - create in data_train_INR if needed)
    log_dir = f"./log/{config.dataset}/"
    output_folder = os.path.join(log_dir, 'tmp_training', 'external_input')

    # Find the most recent visualization image
    import glob
    images = sorted(glob.glob(f"{output_folder}/*.png"))
    if images:
        activity_path = images[-1]  # Most recent image
    else:
        activity_path = f"{output_folder}/placeholder.png"

    # Copy kinograph montage to exploration directory
    kinograph_save_dir = f"{exploration_dir}/kinograph"
    os.makedirs(kinograph_save_dir, exist_ok=True)
    field_output = os.path.join(log_dir, 'tmp_training', 'field')
    src_montage = os.path.join(field_output, 'kinograph_montage.png')
    kinograph_path = f"{kinograph_save_dir}/iter_{iteration:03d}.png"
    if os.path.exists(src_montage):
        import shutil
        shutil.copy2(src_montage, kinograph_path)

    return {
        'tree_save_dir': tree_save_dir,
        'protocol_save_dir': protocol_save_dir,
        'videos_save_dir': videos_save_dir,
        'kinograph_save_dir': kinograph_save_dir,
        'kinograph_path': kinograph_path,
        'activity_path': activity_path
    }
