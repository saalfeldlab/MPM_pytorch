"""
Knowledge Graph Visualization for Epistemic Analysis

Parses epistemic_analysis_*.md files to visualize:
- Discovered principles as nodes (sized by confidence)
- Relationships between principles as edges
- Reasoning mode distribution over iterations
- Confidence evolution curves

Usage:
    python plot_knowledge_graph.py <epistemic_analysis_file> [output_dir]

Example:
    python plot_knowledge_graph.py epistemic_analysis_multimaterial_1_discs_3types.md ./plots/
"""

import re
import math
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional, List, Dict
from collections import defaultdict

import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for file output
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import numpy as np

try:
    import networkx as nx
    HAS_NETWORKX = True
except ImportError:
    HAS_NETWORKX = False
    print("Warning: networkx not installed. Install with: pip install networkx")


@dataclass
class Principle:
    """Discovered principle from epistemic analysis."""
    id: int
    name: str
    prior: str
    discovery: str
    n_tests: int
    n_alt_rejected: int
    n_blocks: int
    confidence: float
    category: str = ""


@dataclass
class ReasoningMode:
    """Reasoning mode statistics."""
    name: str
    count: int
    validation_rate: Optional[float] = None
    first_iter: int = 0


@dataclass
class ReasoningEvent:
    """Individual reasoning event."""
    iteration: int
    mode: str
    observation: str
    pattern: str = ""
    validated: Optional[bool] = None


@dataclass
class TimelineMilestone:
    """Timeline milestone."""
    iteration: int
    milestone: str
    mode: str


@dataclass
class EpistemicAnalysis:
    """Parsed epistemic analysis data."""
    experiment_name: str = ""
    total_iterations: int = 0
    n_blocks: int = 0

    principles: List[Principle] = field(default_factory=list)
    reasoning_modes: List[ReasoningMode] = field(default_factory=list)
    reasoning_events: List[ReasoningEvent] = field(default_factory=list)
    timeline: List[TimelineMilestone] = field(default_factory=list)

    # Detailed events by mode
    induction_events: List[ReasoningEvent] = field(default_factory=list)
    abduction_events: List[ReasoningEvent] = field(default_factory=list)
    deduction_events: List[ReasoningEvent] = field(default_factory=list)
    falsification_events: List[ReasoningEvent] = field(default_factory=list)
    transfer_events: List[ReasoningEvent] = field(default_factory=list)
    boundary_events: List[ReasoningEvent] = field(default_factory=list)


def parse_epistemic_file(filepath: str) -> EpistemicAnalysis:
    """Parse epistemic_analysis_*.md file."""

    ea = EpistemicAnalysis()

    with open(filepath, 'r') as f:
        content = f.read()

    # Extract experiment name
    name_match = re.search(r'# Epistemic Analysis:\s*(.+)', content)
    if name_match:
        ea.experiment_name = name_match.group(1).strip()

    # Extract iterations and blocks
    iter_match = re.search(r'\*\*Iterations\*\*:\s*(\d+)\s*\((\d+)\s*blocks', content)
    if iter_match:
        ea.total_iterations = int(iter_match.group(1))
        ea.n_blocks = int(iter_match.group(2))

    # Parse Reasoning Modes Summary table
    modes_section = re.search(
        r'#### Reasoning Modes Summary\s*\n\n\|[^\n]+\n\|[-|\s]+\n(.*?)(?=\n---|\n####)',
        content, re.DOTALL
    )
    if modes_section:
        for line in modes_section.group(1).strip().split('\n'):
            if '|' in line:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 4 and cols[0]:
                    mode_name = cols[0]
                    try:
                        count = int(cols[1])
                    except:
                        count = 0

                    # Parse validation rate
                    val_rate = None
                    if '%' in cols[2]:
                        val_match = re.search(r'(\d+)%', cols[2])
                        if val_match:
                            val_rate = int(val_match.group(1)) / 100

                    # Parse first appearance
                    first_iter = 0
                    iter_match = re.search(r'Iter (\d+)', cols[3])
                    if iter_match:
                        first_iter = int(iter_match.group(1))

                    ea.reasoning_modes.append(ReasoningMode(
                        name=mode_name,
                        count=count,
                        validation_rate=val_rate,
                        first_iter=first_iter
                    ))

    # Parse Induction table
    induction_section = re.search(
        r'#### 1\. Induction.*?\n\n\|[^\n]+\n\|[-|\s]+\n(.*?)(?=\n####|\n---)',
        content, re.DOTALL
    )
    if induction_section:
        for line in induction_section.group(1).strip().split('\n'):
            if '|' in line:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 4:
                    iter_str = cols[0]
                    # Parse iteration (might be range like "19-23")
                    iter_match = re.search(r'(\d+)', iter_str)
                    iteration = int(iter_match.group(1)) if iter_match else 0

                    ea.induction_events.append(ReasoningEvent(
                        iteration=iteration,
                        mode='induction',
                        observation=cols[1],
                        pattern=cols[2]
                    ))

    # Parse Deduction table
    deduction_section = re.search(
        r'#### 3\. Deduction.*?\n\n\|[^\n]+\n\|[-|\s]+\n(.*?)(?=\n####|\n---)',
        content, re.DOTALL
    )
    if deduction_section:
        for line in deduction_section.group(1).strip().split('\n'):
            if '|' in line:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 5:
                    iter_str = cols[0]
                    iter_match = re.search(r'(\d+)', iter_str)
                    iteration = int(iter_match.group(1)) if iter_match else 0

                    validated = '✓' in cols[4]

                    ea.deduction_events.append(ReasoningEvent(
                        iteration=iteration,
                        mode='deduction',
                        observation=cols[1],
                        pattern=cols[2],
                        validated=validated
                    ))

    # Parse Falsification table
    falsification_section = re.search(
        r'#### 4\. Falsification.*?\n\n\|[^\n]+\n\|[-|\s]+\n(.*?)(?=\n####|\n---)',
        content, re.DOTALL
    )
    if falsification_section:
        for line in falsification_section.group(1).strip().split('\n'):
            if '|' in line:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 3:
                    iter_str = cols[0]
                    iter_match = re.search(r'(\d+)', iter_str)
                    iteration = int(iter_match.group(1)) if iter_match else 0

                    ea.falsification_events.append(ReasoningEvent(
                        iteration=iteration,
                        mode='falsification',
                        observation=cols[1],
                        pattern=cols[2]
                    ))

    # Parse Timeline table
    timeline_section = re.search(
        r'#### Timeline\s*\n\n\|[^\n]+\n\|[-|\s]+\n(.*?)(?=\n\*\*|\n---)',
        content, re.DOTALL
    )
    if timeline_section:
        for line in timeline_section.group(1).strip().split('\n'):
            if '|' in line:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 3:
                    try:
                        iteration = int(cols[0])
                    except:
                        continue

                    ea.timeline.append(TimelineMilestone(
                        iteration=iteration,
                        milestone=cols[1],
                        mode=cols[2]
                    ))

    # Parse Principles table
    principles_section = re.search(
        r'#### 10 Discovered Principles.*?\n\n\|[^\n]+\n\|[-|\s]+\n(.*?)(?=\n####|\n---)',
        content, re.DOTALL
    )
    if principles_section:
        for line in principles_section.group(1).strip().split('\n'):
            if '|' in line:
                cols = [c.strip() for c in line.split('|')[1:-1]]
                if len(cols) >= 6:
                    try:
                        pid = int(cols[0])
                    except:
                        continue

                    # Parse confidence
                    conf_match = re.search(r'(\d+)%', cols[5])
                    confidence = int(conf_match.group(1)) if conf_match else 0

                    # Parse evidence for n_tests, n_alt, n_blocks
                    evidence = cols[4]
                    tests_match = re.search(r'(\d+)\s*tests?', evidence)
                    alt_match = re.search(r'(\d+)\s*alt', evidence)
                    blocks_match = re.search(r'(\d+)\s*blocks?', evidence)

                    n_tests = int(tests_match.group(1)) if tests_match else 0
                    n_alt = int(alt_match.group(1)) if alt_match else 0
                    n_blocks = int(blocks_match.group(1)) if blocks_match else 1

                    # Categorize
                    name = cols[1]
                    category = categorize_principle(name)

                    ea.principles.append(Principle(
                        id=pid,
                        name=name,
                        prior=cols[2],
                        discovery=cols[3],
                        n_tests=n_tests,
                        n_alt_rejected=n_alt,
                        n_blocks=n_blocks,
                        confidence=confidence,
                        category=category
                    ))

    return ea


def categorize_principle(name: str) -> str:
    """Categorize principle by name."""
    name_lower = name.lower()

    if any(x in name_lower for x in ['field', 'jp', 'agnostic', 'efficient', 'easier']):
        return 'cross-domain'
    elif any(x in name_lower for x in ['layer', 'depth', '×', 'minimum', 'capacity']):
        return 'architectural'
    else:
        return 'quantitative'


def build_principle_graph(ea: EpistemicAnalysis):
    """Build NetworkX graph from principles."""

    if not HAS_NETWORKX:
        return None

    G = nx.Graph()

    for p in ea.principles:
        G.add_node(p.id,
                   name=p.name,
                   confidence=p.confidence,
                   category=p.category,
                   n_tests=p.n_tests,
                   n_alt=p.n_alt_rejected,
                   n_blocks=p.n_blocks)

    # Add edges for related principles
    for i, p1 in enumerate(ea.principles):
        for p2 in ea.principles[i+1:]:
            # Same category = related
            if p1.category == p2.category:
                G.add_edge(p1.id, p2.id, relation='same_category')

            # Related by parameter mentions
            p1_name = p1.name.lower()
            p2_name = p2.name.lower()

            if 'lr' in p1_name and 'lr' in p2_name:
                G.add_edge(p1.id, p2.id, relation='lr_related')
            if 'layer' in p1_name and 'layer' in p2_name:
                G.add_edge(p1.id, p2.id, relation='depth_related')
            if 'field' in p1_name and 'field' in p2_name:
                G.add_edge(p1.id, p2.id, relation='field_related')

    return G


def plot_knowledge_graph(ea: EpistemicAnalysis, output_path: str = None):
    """Plot knowledge graph with principles as nodes."""

    if not HAS_NETWORKX:
        print("Cannot plot graph: networkx not installed")
        return

    G = build_principle_graph(ea)
    if G is None or len(G.nodes()) == 0:
        print("No principles to plot")
        return

    fig, ax = plt.subplots(figsize=(12, 10))

    # Colors by category
    category_colors = {
        'quantitative': '#3498db',
        'architectural': '#e74c3c',
        'cross-domain': '#2ecc71'
    }

    # Node properties
    node_colors = [category_colors.get(G.nodes[n]['category'], '#95a5a6') for n in G.nodes()]
    node_sizes = [200 + G.nodes[n]['confidence'] * 8 for n in G.nodes()]

    # Layout
    pos = nx.spring_layout(G, k=2.5, iterations=50, seed=42)

    # Draw edges
    nx.draw_networkx_edges(G, pos, ax=ax, alpha=0.2, edge_color='gray', width=1)

    # Draw nodes
    nx.draw_networkx_nodes(G, pos, ax=ax, node_color=node_colors,
                           node_size=node_sizes, alpha=0.85)

    # Labels
    labels = {}
    for n in G.nodes():
        name = G.nodes[n]['name']
        conf = G.nodes[n]['confidence']
        short_name = name[:18] + '...' if len(name) > 18 else name
        labels[n] = f"{n}. {short_name}\n({conf:.0f}%)"

    nx.draw_networkx_labels(G, pos, labels, ax=ax, font_size=7)

    # Title and legend
    ax.set_title(f"Knowledge Graph: {ea.experiment_name}\n"
                 f"({len(ea.principles)} principles from {ea.total_iterations} iterations)",
                 fontsize=12)

    legend_elements = [
        mpatches.Patch(color='#3498db', label='Quantitative'),
        mpatches.Patch(color='#e74c3c', label='Architectural'),
        mpatches.Patch(color='#2ecc71', label='Cross-domain'),
    ]
    ax.legend(handles=legend_elements, loc='upper left', fontsize=9)
    ax.axis('off')

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_reasoning_distribution(ea: EpistemicAnalysis, output_path: str = None):
    """Plot reasoning mode distribution."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Bar chart of reasoning modes
    ax1 = axes[0]

    mode_colors = {
        'Induction': '#9b59b6',
        'Abduction': '#1abc9c',
        'Deduction': '#3498db',
        'Falsification': '#e74c3c',
        'Analogy/Transfer': '#2ecc71',
        'Boundary Probing': '#f39c12'
    }

    if ea.reasoning_modes:
        modes = [m.name for m in ea.reasoning_modes]
        counts = [m.count for m in ea.reasoning_modes]
        colors = [mode_colors.get(m, '#95a5a6') for m in modes]

        bars = ax1.bar(modes, counts, color=colors, alpha=0.8)
        ax1.set_ylabel('Count')
        ax1.set_title('Reasoning Mode Distribution')
        ax1.tick_params(axis='x', rotation=45)

        # Add validation rates where available
        for i, m in enumerate(ea.reasoning_modes):
            if m.validation_rate is not None:
                ax1.text(i, counts[i] + 0.5, f'{m.validation_rate*100:.0f}%',
                        ha='center', fontsize=8, color='green')

    # Right: Timeline of first appearances
    ax2 = axes[1]

    if ea.timeline:
        iters = [t.iteration for t in ea.timeline]
        milestones = [t.milestone for t in ea.timeline]
        modes = [t.mode for t in ea.timeline]

        colors = [mode_colors.get(m, '#95a5a6') for m in modes]

        ax2.scatter(iters, range(len(iters)), c=colors, s=100, alpha=0.8)

        for i, (it, ms) in enumerate(zip(iters, milestones)):
            ax2.annotate(f"Iter {it}: {ms}", (it, i),
                        xytext=(5, 0), textcoords='offset points',
                        fontsize=8, va='center')

        ax2.set_xlabel('Iteration')
        ax2.set_title('Reasoning Milestones')
        ax2.set_yticks([])
        ax2.set_xlim(0, max(iters) + 10)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_confidence_by_evidence(ea: EpistemicAnalysis, output_path: str = None):
    """Plot principles by evidence strength."""

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    # Left: Scatter plot of n_tests vs confidence
    ax1 = axes[0]

    if ea.principles:
        category_colors = {
            'quantitative': '#3498db',
            'architectural': '#e74c3c',
            'cross-domain': '#2ecc71'
        }

        for cat in ['quantitative', 'architectural', 'cross-domain']:
            ps = [p for p in ea.principles if p.category == cat]
            if ps:
                x = [p.n_tests for p in ps]
                y = [p.confidence for p in ps]
                sizes = [100 + p.n_alt_rejected * 50 for p in ps]

                ax1.scatter(x, y, s=sizes, c=category_colors[cat],
                           alpha=0.7, label=cat)

                # Label points
                for p in ps:
                    ax1.annotate(f"{p.id}", (p.n_tests, p.confidence),
                                fontsize=8, ha='center', va='bottom')

        ax1.set_xlabel('Number of Tests')
        ax1.set_ylabel('Confidence (%)')
        ax1.set_title('Confidence vs Evidence\n(size = alternatives rejected)')
        ax1.legend(fontsize=8)
        ax1.axhline(y=90, color='green', linestyle='--', alpha=0.3, label='Very High')
        ax1.axhline(y=75, color='orange', linestyle='--', alpha=0.3, label='High')
        ax1.axhline(y=60, color='red', linestyle='--', alpha=0.3, label='Medium')

    # Right: Horizontal bar chart of principles by confidence
    ax2 = axes[1]

    if ea.principles:
        # Sort by confidence
        sorted_ps = sorted(ea.principles, key=lambda p: p.confidence)

        names = [f"{p.id}. {p.name[:20]}" for p in sorted_ps]
        confs = [p.confidence for p in sorted_ps]
        colors = [category_colors.get(p.category, '#95a5a6') for p in sorted_ps]

        bars = ax2.barh(names, confs, color=colors, alpha=0.8)
        ax2.set_xlabel('Confidence (%)')
        ax2.set_title('Principles Ranked by Confidence')
        ax2.axvline(x=90, color='green', linestyle='--', alpha=0.3)
        ax2.axvline(x=75, color='orange', linestyle='--', alpha=0.3)
        ax2.axvline(x=60, color='red', linestyle='--', alpha=0.3)
        ax2.set_xlim(0, 105)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close(fig)


def plot_confidence_evolution_curves(output_path: str = None):
    """Plot theoretical confidence evolution curves."""

    fig, ax = plt.subplots(figsize=(10, 6))

    x_tests = np.arange(1, 101)

    scenarios = [
        (1, 0, 'gray', '-', '1 block, 0 alt'),
        (2, 0, 'blue', '-', '2 blocks, 0 alt'),
        (3, 0, 'green', '-', '3 blocks, 0 alt'),
        (2, 5, 'blue', '--', '2 blocks, 5 alt'),
        (3, 5, 'green', '--', '3 blocks, 5 alt'),
        (3, 10, 'green', ':', '3 blocks, 10 alt'),
    ]

    for n_blocks, n_alt, color, ls, label in scenarios:
        conf = []
        for n in x_tests:
            c = 30 + 5 * math.log2(n + 1) + 10 * math.log2(n_alt + 1) + 15 * n_blocks
            conf.append(min(100, c))
        ax.plot(x_tests, conf, color=color, linestyle=ls, label=label, alpha=0.8)

    ax.axhline(y=90, color='green', linestyle='--', alpha=0.3)
    ax.axhline(y=75, color='orange', linestyle='--', alpha=0.3)
    ax.axhline(y=60, color='red', linestyle='--', alpha=0.3)

    ax.text(95, 92, 'Very High', fontsize=8, color='green')
    ax.text(95, 77, 'High', fontsize=8, color='orange')
    ax.text(95, 62, 'Medium', fontsize=8, color='red')

    ax.set_xlabel('Number of Confirmations')
    ax.set_ylabel('Confidence (%)')
    ax.set_title('Confidence Evolution Curves\n'
                 'Formula: 30% + 5%×log2(n+1) + 10%×log2(alt+1) + 15%×blocks')
    ax.legend(loc='lower right', fontsize=8)
    ax.set_xlim(1, 100)
    ax.set_ylim(20, 105)
    ax.grid(True, alpha=0.3)

    plt.tight_layout()

    if output_path:
        plt.savefig(output_path, dpi=150, bbox_inches='tight')
        print(f"Saved: {output_path}")
    plt.close(fig)


def print_summary(ea: EpistemicAnalysis):
    """Print analysis summary."""

    print("\n" + "="*60)
    print(f"EPISTEMIC ANALYSIS: {ea.experiment_name}")
    print("="*60)

    print(f"\nIterations: {ea.total_iterations} ({ea.n_blocks} blocks)")

    print(f"\nReasoning Modes:")
    for m in ea.reasoning_modes:
        val_str = f", {m.validation_rate*100:.0f}% validated" if m.validation_rate else ""
        print(f"  {m.name}: {m.count} (first: iter {m.first_iter}{val_str})")

    print(f"\nPrinciples by Confidence:")
    for p in sorted(ea.principles, key=lambda x: -x.confidence):
        print(f"  {p.id}. {p.name}: {p.confidence:.0f}% ({p.category})")

    print(f"\nPrinciples by Category:")
    for cat in ['quantitative', 'architectural', 'cross-domain']:
        count = sum(1 for p in ea.principles if p.category == cat)
        print(f"  {cat}: {count}")


def main():
    import sys

    if len(sys.argv) < 2:
        print("Usage: python plot_knowledge_graph.py <epistemic_analysis_file> [output_dir]")
        print("\nExample:")
        print("  python plot_knowledge_graph.py epistemic_analysis_experiment.md ./plots/")
        return

    epistemic_file = sys.argv[1]
    output_dir = sys.argv[2] if len(sys.argv) > 2 else "."

    Path(output_dir).mkdir(parents=True, exist_ok=True)

    print(f"Parsing: {epistemic_file}")
    ea = parse_epistemic_file(epistemic_file)

    print_summary(ea)

    # Generate plots
    base_name = Path(epistemic_file).stem

    print("\nGenerating plots...")

    plot_knowledge_graph(ea, f"{output_dir}/{base_name}_graph.png")
    plot_reasoning_distribution(ea, f"{output_dir}/{base_name}_reasoning.png")
    plot_confidence_by_evidence(ea, f"{output_dir}/{base_name}_confidence.png")
    plot_confidence_evolution_curves(f"{output_dir}/{base_name}_evolution.png")

    print("\nDone!")


if __name__ == "__main__":
    main()
