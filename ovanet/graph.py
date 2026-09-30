"""
C4 — Spatial Graph construction from follicle detections.

build_graph(follicles, ovary_mask=None) -> GraphResult

Constructs a spatial graph of detected follicles:
  - Nodes: 5 normalized features [x_rel, y_rel, area_rel, r_rel, eccentricity]
  - Adjacency: k-NN (k = min(3, n-1)), symmetrized, with Gaussian weights
    w_ij = exp(-d_ij^2 / tau), where tau = median(d^2)
  - 8 summary statistics in fixed order (GRAPH_STAT_NAMES):
    n_follicles, mean_area_rel, std_area_rel, mean_dist_to_center_rel,
    std_dist_to_center_rel, peripheral_index, mean_degree, clustering_coeff

Edge case: n < 3 returns empty/zero arrays, valid=False, and stats=0.0
except n_follicles per frozen contract.
"""

import math
import numpy as np

from ovanet.contracts import (
    GRAPH_FEATURE_DIM,
    GRAPH_STAT_NAMES,
    Follicle,
    GraphEdge,
    GraphNode,
    GraphResult,
)


def _compute_ovary_properties(ovary_mask: np.ndarray | None):
    """Compute center (cx, cy) and characteristic radius R = sqrt(area / pi)."""
    if ovary_mask is None or ovary_mask.sum() == 0:
        return 128.0, 128.0, 128.0 * 128.0 * math.pi, 128.0

    area_px = float(np.sum(ovary_mask > 0))
    # Moments of the ovary mask for centroid
    y_coords, x_coords = np.where(ovary_mask > 0)
    cx_ovary = float(np.mean(x_coords))
    cy_ovary = float(np.mean(y_coords))
    r_ovary = math.sqrt(max(1.0, area_px / math.pi))
    return cx_ovary, cy_ovary, area_px, r_ovary


def _clustering_coefficient(binary_adj: np.ndarray) -> float:
    """Compute average clustering coefficient for an undirected unweighted graph."""
    n = binary_adj.shape[0]
    if n < 3:
        return 0.0

    coeffs = []
    for i in range(n):
        neighbors = np.where(binary_adj[i] > 0)[0]
        k_i = len(neighbors)
        if k_i < 2:
            coeffs.append(0.0)
            continue
        # Subgraph among neighbors
        subgraph = binary_adj[np.ix_(neighbors, neighbors)]
        # Number of edges between neighbors (each edge counted twice in symmetric matrix)
        triangles = np.sum(subgraph) / 2.0
        c_i = (2.0 * triangles) / (k_i * (k_i - 1))
        coeffs.append(c_i)

    return float(np.mean(coeffs)) if coeffs else 0.0


def build_graph(
    follicles: list[Follicle],
    ovary_mask: np.ndarray | None = None,
) -> GraphResult:
    """
    Contract C4.

    Parameters
    ----------
    follicles : list[Follicle] from detector / C2
    ovary_mask : uint8 [256, 256], optional

    Returns
    -------
    GraphResult with (n, x, adj, edges, nodes, stats)
    """
    n = len(follicles)

    # Base nodes
    nodes = [GraphNode(id=f.id, x=float(f.cx), y=float(f.cy)) for f in follicles]

    # Handle n < 3 edge case (contract requirement)
    if n < 3:
        x = np.zeros((n, GRAPH_FEATURE_DIM), dtype=np.float32)
        adj = np.zeros((n, n), dtype=np.float32)
        edges: list[GraphEdge] = []
        stats = {name: 0.0 for name in GRAPH_STAT_NAMES}
        stats["n_follicles"] = float(n)
        return GraphResult(n=n, x=x, adj=adj, edges=edges, nodes=nodes, stats=stats)

    cx_ov, cy_ov, area_ov, r_ov = _compute_ovary_properties(ovary_mask)

    # 1. Compute node features [n, 5]:
    # [x_rel, y_rel, area_rel, r_rel, eccentricity]
    x_mat = np.zeros((n, GRAPH_FEATURE_DIM), dtype=np.float32)
    coords = np.zeros((n, 2), dtype=np.float32)

    for i, f in enumerate(follicles):
        coords[i, 0] = f.cx
        coords[i, 1] = f.cy
        x_mat[i, 0] = (f.cx - cx_ov) / r_ov
        x_mat[i, 1] = (f.cy - cy_ov) / r_ov
        x_mat[i, 2] = f.area_px / max(1.0, area_ov)
        x_mat[i, 3] = f.eq_radius_px / r_ov
        x_mat[i, 4] = f.eccentricity

    # 2. Pairwise distances
    # D_ij = sqrt((x_i - x_j)^2 + (y_i - y_j)^2)
    diff = coords[:, np.newaxis, :] - coords[np.newaxis, :, :]
    dist_matrix = np.sqrt(np.sum(diff ** 2, axis=-1))

    # 3. k-NN adjacency with k = min(3, n-1)
    k = min(3, n - 1)
    k_adj = np.zeros((n, n), dtype=bool)

    edge_dist_sq = []
    for i in range(n):
        # Sort indices by distance, excluding self (dist=0)
        sorted_indices = np.argsort(dist_matrix[i])
        # First index is self (0)
        neighbors = sorted_indices[1 : k + 1]
        for nb in neighbors:
            k_adj[i, nb] = True
            edge_dist_sq.append(dist_matrix[i, nb] ** 2)

    # Symmetrize
    sym_adj = np.logical_or(k_adj, k_adj.T)
    np.fill_diagonal(sym_adj, False)

    # Kernel bandwidth tau = median(d^2)
    if len(edge_dist_sq) > 0:
        tau = float(np.median(edge_dist_sq))
        if tau <= 1e-4:
            tau = 1.0
    else:
        tau = 1.0

    # Continuous weights: w_ij = exp(-d_ij^2 / tau)
    adj_weights = np.zeros((n, n), dtype=np.float32)
    for i in range(n):
        for j in range(n):
            if sym_adj[i, j]:
                d2 = dist_matrix[i, j] ** 2
                adj_weights[i, j] = float(np.exp(-d2 / tau))

    # Construct edge list
    edges = []
    for i in range(n):
        for j in range(i + 1, n):
            if sym_adj[i, j]:
                edges.append(
                    GraphEdge(source=i, target=j, weight=float(adj_weights[i, j]))
                )

    # 4. Summary statistics per GRAPH_STAT_NAMES:
    # 1. n_follicles
    # 2. mean_nn_dist_rel, 3. std_nn_dist_rel: 1-NN distance in R units
    if n > 1:
        # Distance to closest distinct follicle (first column after self in sorted distances)
        nn_dists = np.sort(dist_matrix, axis=1)[:, 1] / r_ov
        mean_nn_dist = float(np.mean(nn_dists))
        std_nn_dist = float(np.std(nn_dists))
    else:
        mean_nn_dist = 0.0
        std_nn_dist = 0.0

    # 4. mean_degree in unweighted graph
    degrees = np.sum(sym_adj, axis=1)
    mean_degree = float(np.mean(degrees))

    # 5. clustering_coeff
    clustering_coeff = _clustering_coefficient(sym_adj.astype(np.uint8))

    # 6. peripheral_index
    dist_to_center = np.sqrt(x_mat[:, 0] ** 2 + x_mat[:, 1] ** 2)
    max_d = max(1.0, float(np.max(dist_to_center)))
    peripheral_count = np.sum(dist_to_center >= 0.5 * max_d)
    peripheral_index = float(peripheral_count / n)

    # 7. size_cv: Coefficient of variation of follicle areas (std / mean)
    areas = np.array([f.area_px for f in follicles], dtype=np.float32)
    mean_area = float(np.mean(areas))
    std_area = float(np.std(areas))
    size_cv = float(std_area / mean_area) if mean_area > 1e-4 else 0.0

    # 8. follicle_area_fraction: total follicle area / ovary area
    total_follicle_area = float(np.sum(areas))
    follicle_area_fraction = float(min(1.0, total_follicle_area / max(1.0, area_ov)))

    stats = {
        "n_follicles": float(n),
        "mean_nn_dist_rel": mean_nn_dist,
        "std_nn_dist_rel": std_nn_dist,
        "mean_degree": mean_degree,
        "clustering_coeff": clustering_coeff,
        "peripheral_index": peripheral_index,
        "size_cv": size_cv,
        "follicle_area_fraction": follicle_area_fraction,
    }

    return GraphResult(
        n=n,
        x=x_mat,
        adj=adj_weights,
        edges=edges,
        nodes=nodes,
        stats=stats,
    )
