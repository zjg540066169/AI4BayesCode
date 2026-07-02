"""Visualize the model PREDICTION DAG with networkx + matplotlib.

Python analogue of the R helper's `ai4bayescode_plot_dag(model)`. Renders ONLY
the prediction DAG (generative / causal direction), per
codegen.md §2(b). Gibbs / refresh edges are intentionally hidden —
those are the internal full-conditional graph and confuse users who
expect to see the model's data-flow structure. Call
`model.get_dag()["gibbs_reads"]` directly if you need to debug
Gibbs dependencies.

Node coloring matches the R helper:

  - Data inputs                (green circle)
  - Hyperparameters            (grey square)
  - Sampled parameters         (blue circle)
  - Derived keys               (purple diamond)
  - Terminal / leaf            (orange circle)

Plate detection: names matching `<prefix>_<digits>` collapse into a
single node labeled `<prefix>_i [n=K]`.
"""

from __future__ import annotations

import re
from collections import defaultdict
from pathlib import Path
from typing import Any, Optional


_PLATE_RE = re.compile(r"^(.*)_([0-9]+)$")


def _dag_to_edges(dag: dict) -> tuple[list, list]:
    """Flatten the dag dict into (edges, data_inputs).

    PREDICTION-ONLY: gibbs_reads / gibbs_invalidates are intentionally
    skipped — see module docstring.
    """
    edges = []  # list of (from, to, type)
    predict = dag.get("predict_edges", {}) or {}
    for src, dst_list in predict.items():
        for dst in dst_list:
            edges.append((src, dst, "predict"))
    data_inputs = list(dag.get("data_inputs", []) or [])
    return edges, data_inputs


def _collapse_plates(edges: list, data_inputs: list) -> tuple[list, list, dict]:
    """Return (new_edges, new_data_inputs, plate_sizes)."""
    all_nodes = set()
    for (a, b, _) in edges:
        all_nodes.add(a); all_nodes.add(b)
    all_nodes.update(data_inputs)

    by_prefix: dict[str, list[str]] = defaultdict(list)
    for n in all_nodes:
        m = _PLATE_RE.match(n)
        if m:
            by_prefix[m.group(1)].append(n)

    # A "plate" is any prefix with 2+ numbered instances
    plate_of: dict[str, str] = {}
    plate_sizes: dict[str, int] = {}
    for prefix, members in by_prefix.items():
        if len(members) < 2:
            continue
        label = f"{prefix}_i [n={len(members)}]"
        plate_sizes[label] = len(members)
        for m in members:
            plate_of[m] = label

    def canon(n):
        return plate_of.get(n, n)

    new_edges = []
    seen = set()
    for (a, b, t) in edges:
        ca, cb = canon(a), canon(b)
        key = (ca, cb, t)
        if key in seen:
            continue
        seen.add(key)
        new_edges.append(key)
    new_data_inputs = [canon(n) for n in data_inputs]
    return new_edges, new_data_inputs, plate_sizes


def ai4bayescode_plot_dag(
    model: Any,
    *,
    out_path: Optional[str | Path] = None,
    title: Optional[str] = None,
    figsize: tuple[float, float] = (12, 8),
    dpi: int = 150,
    plate: bool = True,
) -> Optional[Path]:
    """Render the model DAG as a PNG (or show interactively if out_path is None).

    Parameters
    ----------
    model : object
        An AI4BayesCode model instance exposing `get_dag() -> dict`.
    out_path : path-like, optional
        Where to write the PNG. If None, writes to a tempfile and returns
        the path.
    title : str, optional
        Plot title (defaults to the model's class name).
    figsize, dpi : matplotlib figure config.
    plate : bool, default True
        Collapse `<prefix>_<digits>` nodes into one plate node.

    Returns
    -------
    Path to the generated PNG (or None if shown interactively).
    """
    try:
        import networkx as nx
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "ai4bayescode_plot_dag requires networkx and matplotlib.\n"
            "Install with: pip install networkx matplotlib"
        ) from e

    if not hasattr(model, "get_dag"):
        raise AttributeError(
            "model has no `get_dag()` method — is this an AI4BayesCode sampler?"
        )
    dag = model.get_dag()
    edges, data_inputs = _dag_to_edges(dag)
    if plate:
        edges, data_inputs, _ = _collapse_plates(edges, data_inputs)

    G = nx.DiGraph()
    for (a, b, t) in edges:
        G.add_edge(a, b, type=t)
    for di in data_inputs:
        G.add_node(di)

    # Classify nodes
    gibbs_reads = dag.get("gibbs_reads", {}) or {}
    gibbs_invalidates = dag.get("gibbs_invalidates", {}) or {}
    sampled_blocks = set(gibbs_reads.keys())
    derived_keys = {k for keys in gibbs_invalidates.values() for k in keys}

    node_colors, node_shapes, node_labels = {}, {}, {}
    for n in G.nodes():
        if n in data_inputs:
            node_colors[n] = "#2ecc71"  # green
            node_shapes[n] = "o"
        elif n in sampled_blocks:
            node_colors[n] = "#3498db"  # blue
            node_shapes[n] = "o"
        elif n in derived_keys:
            node_colors[n] = "#9b59b6"  # purple
            node_shapes[n] = "d"
        elif G.out_degree(n) == 0:
            node_colors[n] = "#e67e22"  # orange — terminal
            node_shapes[n] = "o"
        else:
            node_colors[n] = "#95a5a6"  # grey — hyperparam / other
            node_shapes[n] = "s"
        node_labels[n] = n

    # Sugiyama-like hierarchical layout via networkx's dot fallback
    try:
        pos = nx.nx_agraph.graphviz_layout(G, prog="dot")
    except Exception:
        pos = nx.spring_layout(G, seed=42)

    fig, ax = plt.subplots(figsize=figsize, dpi=dpi)
    # Draw nodes by shape
    for shape in set(node_shapes.values()):
        nodes_of_shape = [n for n, s in node_shapes.items() if s == shape]
        colors_of_shape = [node_colors[n] for n in nodes_of_shape]
        nx.draw_networkx_nodes(G, pos, nodelist=nodes_of_shape,
                               node_color=colors_of_shape, node_shape=shape,
                               node_size=600, ax=ax, edgecolors="black",
                               linewidths=0.8)

    # Edges by type
    edge_styles = {
        "gibbs": dict(edge_color="black", style="solid", arrows=True),
        "refresh": dict(edge_color="purple", style="dashed", arrows=True),
        "predict": dict(edge_color="red", style="solid", arrows=True),
    }
    for t, style in edge_styles.items():
        subset = [(u, v) for u, v, data in G.edges(data=True)
                  if data.get("type") == t]
        if not subset:
            continue
        nx.draw_networkx_edges(G, pos, edgelist=subset, ax=ax,
                               width=1.3, arrowsize=15, **style)

    nx.draw_networkx_labels(G, pos, labels=node_labels, font_size=8, ax=ax)

    if title is None:
        title = f"{type(model).__name__} prediction DAG"
    ax.set_title(title, fontsize=12)
    ax.axis("off")

    # Legend
    from matplotlib.patches import Patch
    legend_items = [
        Patch(color="#2ecc71", label="Data input"),
        Patch(color="#3498db", label="Sampled param"),
        Patch(color="#9b59b6", label="Derived key"),
        Patch(color="#e67e22", label="Terminal"),
        Patch(color="#95a5a6", label="Hyperparam"),
    ]
    ax.legend(handles=legend_items, loc="upper right", fontsize=8,
              frameon=True, title="Node role")

    if out_path is None:
        import tempfile
        out_path = Path(tempfile.gettempdir()) / f"ai4bayescode_dag_{id(model)}.png"
    out_path = Path(out_path)
    fig.tight_layout()
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)
    return out_path
