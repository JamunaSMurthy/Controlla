"""Graph construction utilities for Controlla experiments.

This package builds the graph artifacts used by Controlla:

- emotion / affective graph
- reference-identity graph
- cross-modal graph
- topology variants for sensitivity studies

The generated graph artifacts are used by graph-OT alignment, traversal, and
paper-level ablation/sensitivity experiments.
"""

from .graph_utils import (
    adjacency_to_distance,
    cosine_similarity_matrix,
    edge_list_from_adjacency,
    infer_graph_output_dir,
    knn_adjacency,
    load_manifest,
    save_graph,
)
from .topology_variants import apply_topology_variant, perturb_adjacency

__all__ = [
    "adjacency_to_distance",
    "apply_topology_variant",
    "cosine_similarity_matrix",
    "edge_list_from_adjacency",
    "infer_graph_output_dir",
    "knn_adjacency",
    "load_manifest",
    "perturb_adjacency",
    "save_graph",
]