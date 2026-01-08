"""
Plot UCB Exploration Tree from ucb_scores.txt for MPM INR Training

Visualizes the exploration tree structure with:
- Circles for regular nodes, crosses for leaf nodes
- Node ID, visits, and R² displayed
- MSE metric instead of Pearson correlation

Adapted from NeuralGraph for MPM INR optimization.
"""

import re
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.patches import Ellipse
import numpy as np


@dataclass
class UCBNode:
    """Represents a node from ucb_scores.txt."""
    id: int
    ucb: float
    parent: Optional[int]
    visits: int
    r2: float
    slope: float = 0.0
    training_time_min: float = 0.0
    mutation: str = ""
    has_code_mod: bool = False
    code_change: str = ""


def parse_ucb_scores(filepath: str) -> list[UCBNode]:
    """Parse ucb_scores.txt into a list of UCBNode objects."""
    nodes = []

    with open(filepath, 'r') as f:
        content = f.read()

    # Parse each line - looking for [CODE] marker and extracting code change
    for line in content.split('\n'):
        if not line.startswith('Node '):
            continue

        # Basic node info pattern
        basic_pattern = r'Node (\d+): UCB=([\d.]+), parent=(\d+|root), visits=(\d+), R2=([\d.]+)(?:, slope=([\d.]+))?(?:, time=([\d.]+)min)?'
        match = re.match(basic_pattern, line)

        if not match:
            continue

        node_id = int(match.group(1))
        ucb = float(match.group(2))
        parent_str = match.group(3)
        parent = None if parent_str == 'root' else int(parent_str)
        visits = int(match.group(4))
        r2 = float(match.group(5))
        slope = float(match.group(6)) if match.group(6) else 0.0
        training_time_min = float(match.group(7)) if match.group(7) else 0.0

        # Check for [CODE] marker
        has_code_mod = '[CODE]' in line
        code_change = ""

        if has_code_mod:
            # Extract code change description (text after [CODE])
            code_match = re.search(r'\[CODE\]\s+([^,]+)', line)
            if code_match:
                code_change = code_match.group(1).strip()

        # Extract mutation (config changes)
        mutation = ""
        mutation_match = re.search(r', Mutation=(.+)', line)
        if mutation_match:
            mutation = mutation_match.group(1).strip()
            # Remove [CODE] part from mutation if present
            mutation = re.sub(r'\[CODE\].*', '', mutation).strip()

        nodes.append(UCBNode(
            id=node_id,
            ucb=ucb,
            parent=parent,
            visits=visits,
            r2=r2,
            slope=slope,
            training_time_min=training_time_min,
            mutation=mutation,
            has_code_mod=has_code_mod,
            code_change=code_change
        ))

    return nodes


def build_tree(nodes: list[UCBNode]) -> dict:
    """Build tree structure: children dict and find roots."""
    children = defaultdict(list)
    node_map = {n.id: n for n in nodes}

    for node in nodes:
        if node.parent is not None:
            children[node.parent].append(node.id)

    # Sort children by id for consistent layout
    for parent_id in children:
        children[parent_id].sort()

    # Find root nodes (nodes with no parent or parent not in node_map)
    roots = [n.id for n in nodes if n.parent is None or n.parent not in node_map]

    return {
        'children': children,
        'node_map': node_map,
        'roots': roots
    }


def compute_layout(tree: dict) -> dict[int, tuple[float, float]]:
    """
    Compute x,y positions for tree visualization.
    x = depth from root
    y = vertical spread within depth level
    """
    children = tree['children']
    roots = tree['roots']

    depth_map = {}
    y_positions = {}

    # Compute depths using BFS
    def compute_depth(node_id, current_depth=0):
        depth_map[node_id] = current_depth
        for child_id in children.get(node_id, []):
            compute_depth(child_id, current_depth + 1)

    for root in roots:
        compute_depth(root, 0)

    # Assign y positions: leaves get sequential positions, parents center on children
    leaf_counter = [0]

    def assign_y_dfs(node_id):
        child_list = children.get(node_id, [])
        if not child_list:
            # Leaf node
            y_positions[node_id] = leaf_counter[0]
            leaf_counter[0] += 1
        else:
            # Process children first
            for child_id in child_list:
                assign_y_dfs(child_id)
            # Parent y = center of children
            y_positions[node_id] = np.mean([y_positions[c] for c in child_list])

    for root in roots:
        assign_y_dfs(root)

    # Combine into positions dict
    positions = {}
    for node_id in depth_map:
        if node_id in y_positions:
            positions[node_id] = (depth_map[node_id], y_positions[node_id])

    return positions


def plot_ucb_tree(nodes: list[UCBNode],
                  output_path: Optional[str] = None,
                  title: str = "UCB Exploration Tree",
                  field_info: Optional[str] = None,
                  n_training_frames: Optional[int] = None):
    """
    Plot the UCB exploration tree for MPM INR training.

    - Circle (o) for nodes with children
    - Cross (x) for leaf nodes
    - Shows node ID, visits, R², and MSE
    - Shows field configuration near root node
    """
    if not nodes:
        print("No nodes to plot")
        return

    tree = build_tree(nodes)
    positions = compute_layout(tree)
    children = tree['children']
    node_map = tree['node_map']

    # Color based on R2 value
    def get_color(r2):
        if r2 >= 0.9:
            return '#2ecc71'  # green
        elif r2 >= 0.5:
            return '#f39c12'  # orange
        else:
            return '#e74c3c'  # red

    fig, ax = plt.subplots(figsize=(16, 12))

    # Set white background explicitly
    fig.patch.set_facecolor('white')
    ax.set_facecolor('white')

    # Draw edges first (so they're behind nodes)
    for node in nodes:
        if node.parent is not None and node.parent in positions and node.id in positions:
            x1, y1 = positions[node.parent]
            x2, y2 = positions[node.id]
            ax.plot([x1, x2], [y1, y2], color='#34495e', linestyle='-',
                   linewidth=1.5, alpha=0.6, zorder=1)

    # Compute UCB range for size scaling
    ucb_values = [n.ucb for n in nodes]
    min_ucb = min(ucb_values)
    max_ucb = max(ucb_values)
    ucb_range = max_ucb - min_ucb if max_ucb > min_ucb else 1.0

    # Draw nodes
    for node in nodes:
        if node.id not in positions:
            continue

        x, y = positions[node.id]
        color = get_color(node.r2)

        # Size proportional to UCB (larger base size)
        size = 150 + 150 * (node.ucb - min_ucb) / ucb_range

        # Determine if leaf node (no children)
        is_leaf = len(children.get(node.id, [])) == 0

        if is_leaf:
            # Cross marker for leaf nodes
            ax.scatter(x, y, c=color, s=size, marker='x', linewidths=3, zorder=2)
        else:
            # Circle marker for internal nodes
            ax.scatter(x, y, c=color, s=size, marker='o',
                      edgecolors='black', linewidths=0.5, zorder=2)

        # Add a special indicator for code modifications (double circle border)
        if node.has_code_mod:
            # Draw an outer circle to create double-border effect
            ax.scatter(x, y, c='none', s=size*1.5, marker='o',
                      edgecolors='#9b59b6', linewidths=2.5, zorder=2, alpha=0.8)

        # Label: node id inside/near the marker (always black)
        ax.annotate(str(node.id), (x, y), ha='center', va='center',
                   fontsize=9,
                   color='black', zorder=3)

        # Code modification indicator above the node
        if node.has_code_mod and node.code_change:
            code_text = f"[CODE] {node.code_change}"
            ax.annotate(code_text, (x, y), ha='center', va='bottom',
                       fontsize=7, xytext=(0, 22), textcoords='offset points',
                       color='#9b59b6', fontweight='bold', zorder=3,
                       bbox=dict(boxstyle='round,pad=0.3', facecolor='#f3e5f5',
                                edgecolor='#9b59b6', alpha=0.7))

        # Mutation above the node (for nodes with id > 1)
        elif node.id > 1 and node.mutation:
            # Remove parenthesis part from mutation text
            mutation_text = re.sub(r'\s*\([^)]*\)\s*$', '', node.mutation).strip()
            # Skip simulation change messages (they clutter the plot)
            if not mutation_text.startswith('simulation changed'):
                # Shorten common parameter names
                mutation_text = mutation_text.replace('hidden_dim_nnr_f', 'h')
                mutation_text = mutation_text.replace('n_layers_nnr_f', 'L')
                mutation_text = mutation_text.replace('learning_rate_NNR_f', 'lr')
                mutation_text = mutation_text.replace('total_steps', 'steps')
                mutation_text = mutation_text.replace('omega_f', 'ω')
                mutation_text = mutation_text.replace('n_training_frames', 'frames')
                # Extract just the new value (after →) if present, format as "param=value"
                if '→' in mutation_text:
                    parts = mutation_text.split('→')
                    if len(parts) == 2:
                        # Get param name from left side (before :)
                        left = parts[0].strip()
                        new_val = parts[1].strip()
                        if ':' in left:
                            param_name = left.split(':')[0].strip()
                            mutation_text = f"{param_name}={new_val}"
                        else:
                            mutation_text = new_val
                ax.annotate(mutation_text, (x, y), ha='center', va='bottom',
                           fontsize=7, xytext=(0, 12), textcoords='offset points',
                           color='#333333', zorder=3)

        # Annotation: UCB/V and R²/slope/time below the node
        label_text = f"UCB={node.ucb:.2f} V={node.visits}\nR²={node.r2:.3f}"
        if node.slope > 0:
            label_text += f" slope={node.slope:.3f}"
        if node.training_time_min > 0:
            label_text += f"\n{node.training_time_min:.1f}min"
        ax.annotate(label_text, (x, y), ha='center', va='top',
                   fontsize=8, xytext=(0, -14), textcoords='offset points',
                   color='#555555', zorder=3)

    # Add field info and n_training_frames at top left of figure
    info_lines = []
    if field_info:
        # Format: remove extra prefixes, replace underscores with spaces
        field_text = field_info.strip()
        field_text = field_text.replace('_', ' ')
        info_lines.append(field_text)
    if n_training_frames is not None:
        info_lines.append(f"n_frames={n_training_frames}")
    if info_lines:
        # Place at top left using axes coordinates (0,1 = top left)
        ax.text(0.02, 0.98, '\n'.join(info_lines), transform=ax.transAxes,
                fontsize=11, ha='left', va='top', color='#333333')

    # Remove axis labels and ticks
    ax.set_xlabel('')
    ax.set_ylabel('')
    ax.set_xticks([])
    ax.set_yticks([])

    # Set axis limits with padding
    if positions:
        x_vals = [p[0] for p in positions.values()]
        y_vals = [p[1] for p in positions.values()]
        ax.set_xlim(min(x_vals) - 0.5, max(x_vals) + 0.5)
        ax.set_ylim(min(y_vals) - 1, max(y_vals) + 1)

    ax.grid(False)
    ax.axis('off')

    # Draw green circle around the best node (highest R²)
    # Use Ellipse to appear circular regardless of aspect ratio
    if nodes:
        best_node = max(nodes, key=lambda n: n.r2)
        if best_node.id in positions:
            x, y = positions[best_node.id]
            # Calculate ellipse dimensions to appear as circle on screen
            x_range = ax.get_xlim()[1] - ax.get_xlim()[0]
            y_range = ax.get_ylim()[1] - ax.get_ylim()[0]
            # Base radius in data units, scaled to axes
            radius_x = 0.15 * x_range / 4  # scale factor for width
            radius_y = 0.15 * y_range / 4  # scale factor for height
            best_ellipse = Ellipse((x, y), width=radius_x*2, height=radius_y*2,
                                   fill=False, color='#228B22',
                                   linewidth=3, alpha=0.5, zorder=4)
            ax.add_patch(best_ellipse)

    # Legend
    legend_elements = [
        mpatches.Patch(color='#2ecc71', label='R² ≥ 0.9'),
        mpatches.Patch(color='#f39c12', label='R² ≥ 0.5'),
        mpatches.Patch(color='#e74c3c', label='R² < 0.5'),
        plt.Line2D([0], [0], marker='o', color='gray', label='Internal node',
                   markerfacecolor='gray', markersize=8, linestyle='None'),
        plt.Line2D([0], [0], marker='x', color='gray', label='Leaf node',
                   markerfacecolor='gray', markersize=8, linestyle='None', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='#228B22', label='Best R² node',
                   markerfacecolor='none', markersize=10, linestyle='None', markeredgewidth=2),
        plt.Line2D([0], [0], marker='o', color='#9b59b6', label='Code modification',
                   markerfacecolor='none', markersize=10, linestyle='None', markeredgewidth=2),
    ]
    ax.legend(handles=legend_elements, loc='upper right', fontsize=9)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        # print(f"saved UCB tree to {output_path}")
    else:
        plt.show()

    plt.close()


def print_summary(nodes: list[UCBNode]):
    """Print summary statistics."""
    if not nodes:
        print("No nodes found")
        return

    print(f"\n=== UCB Tree Summary ===")
    print(f"Total nodes: {len(nodes)}")
    print(f"UCB range: {min(n.ucb for n in nodes):.3f} - {max(n.ucb for n in nodes):.3f}")
    print(f"Visits range: {min(n.visits for n in nodes)} - {max(n.visits for n in nodes)}")
    print(f"R² range: {min(n.r2 for n in nodes):.3f} - {max(n.r2 for n in nodes):.3f}")

    # Find highest UCB nodes (most promising to explore)
    sorted_by_ucb = sorted(nodes, key=lambda n: n.ucb, reverse=True)[:5]
    print(f"\nTop 5 by UCB (most promising):")
    for n in sorted_by_ucb:
        print(f"  Node {n.id}: UCB={n.ucb:.3f}, visits={n.visits}, R²={n.r2:.3f}")
