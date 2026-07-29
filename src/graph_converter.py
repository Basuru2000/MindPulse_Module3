"""
MindPulse — Module 3  |  Layer 2: Graph Converter
==================================================
Converts the Layer 1 NetworkX Knowledge Graph into PyTorch Geometric's
tensor format, ready for GCN training.

This is the bridge between Layer 1 and Layer 2. The same Knowledge Graph
that Layer 1 traverses with hand-written rules becomes the graph topology
that Layer 2's GCN learns over — which is precisely why NetworkX was chosen
over a graph database in Layer 1 (see Module 3 spec, Section 2.1).

What this module produces:
  • x           — node feature matrix, shape (num_nodes, NODE_FEATURE_DIM)
  • edge_index  — edge connectivity, shape (2, num_edges)
  • edge_weight — clinical edge weights, shape (num_edges,)
  • node_to_idx — mapping from KG node ID string to tensor row index
  • idx_to_node — reverse mapping, for interpreting model output

Node feature encoding (18 dimensions, all static graph properties):
  [0:6]   node type one-hot        — StressState, TriggerContext, LocationContext,
                                     SocialContext, GestureProfile, Intervention
  [6:11]  intervention category    — breathing, physical, cognitive, social, sensory
                                     (all zero for non-intervention nodes)
  [11]    tier index (normalised)  — StressState nodes only, 0.0-1.0
  [12]    duration (normalised)    — Intervention nodes only, 0.0-1.0
  [13:16] target tiers multi-hot   — Intervention nodes only: mild, moderate, acute
  [16]    has social exclusions    — Intervention nodes only, binary
  [17]    has location exclusions  — Intervention nodes only, binary

Note on what is NOT encoded here: per-episode values (baseline_deviation,
support_score, trigger_confidence) are dynamic — they change with every stress
episode and are not properties of the graph itself. They are injected at
scoring time in the Layer 2 model, not baked into these static node features.

Author : Module 3 — MindPulse (Team MindForge)
"""

import torch
import networkx as nx
from torch_geometric.data import Data

from knowledge_graph import build_knowledge_graph


# ── Feature encoding constants ────────────────────────────────────────────────
NODE_TYPES = [
    "StressState",
    "TriggerContext",
    "LocationContext",
    "SocialContext",
    "GestureProfile",
    "Intervention",
]

INTERVENTION_CATEGORIES = ["breathing", "physical", "cognitive", "social", "sensory"]

TARGET_TIERS = ["mild", "moderate", "acute"]

# Longest intervention in the library is 5 minutes — used to normalise duration
MAX_DURATION = 5.0

# Total feature dimensionality (see module docstring for the layout)
NODE_FEATURE_DIM = (
    len(NODE_TYPES)               # 6  — node type one-hot
    + len(INTERVENTION_CATEGORIES)  # 5  — intervention category one-hot
    + 1                             # 1  — tier index
    + 1                             # 1  — duration
    + len(TARGET_TIERS)             # 3  — target tiers multi-hot
    + 2                             # 2  — exclusion flags
)  # = 18


# ══════════════════════════════════════════════════════════════════════════════
# NODE FEATURE ENCODING
# ══════════════════════════════════════════════════════════════════════════════

def encode_node_features(node_id: str, attrs: dict) -> list:
    """
    Encode a single KG node's attributes into a fixed-length feature vector.

    Every node gets the same NODE_FEATURE_DIM-length vector; dimensions that
    don't apply to a given node type are left at 0.0. This is standard practice
    for heterogeneous graphs represented in a homogeneous GCN — the node-type
    one-hot lets the model learn type-specific behaviour from the shared space.
    """
    feat = [0.0] * NODE_FEATURE_DIM

    # ── [0:6] Node type one-hot ────────────────────────────────────────────────
    node_type = attrs.get("node_type", "")
    if node_type in NODE_TYPES:
        feat[NODE_TYPES.index(node_type)] = 1.0

    offset = len(NODE_TYPES)  # 6

    # ── [6:11] Intervention category one-hot ──────────────────────────────────
    if node_type == "Intervention":
        inv_type = attrs.get("intervention_type", "")
        if inv_type in INTERVENTION_CATEGORIES:
            feat[offset + INTERVENTION_CATEGORIES.index(inv_type)] = 1.0

    offset += len(INTERVENTION_CATEGORIES)  # 11

    # ── [11] Tier index, normalised to 0.0-1.0 ────────────────────────────────
    # calm=0, mild=1, moderate=2, acute=3  →  0.0, 0.33, 0.67, 1.0
    if node_type == "StressState":
        tier_index = attrs.get("tier_index", 0)
        feat[offset] = tier_index / 3.0

    offset += 1  # 12

    # ── [12] Duration, normalised to 0.0-1.0 ──────────────────────────────────
    if node_type == "Intervention":
        duration = attrs.get("duration", 0)
        feat[offset] = duration / MAX_DURATION

    offset += 1  # 13

    # ── [13:16] Target tiers multi-hot ────────────────────────────────────────
    if node_type == "Intervention":
        target_tiers = attrs.get("target_tiers", [])
        for i, tier in enumerate(TARGET_TIERS):
            if tier in target_tiers:
                feat[offset + i] = 1.0

    offset += len(TARGET_TIERS)  # 16

    # ── [16:18] Exclusion flags ────────────────────────────────────────────────
    if node_type == "Intervention":
        feat[offset]     = 1.0 if attrs.get("excludes_social",   []) else 0.0
        feat[offset + 1] = 1.0 if attrs.get("excludes_location", []) else 0.0

    return feat


# ══════════════════════════════════════════════════════════════════════════════
# GRAPH CONVERSION
# ══════════════════════════════════════════════════════════════════════════════

def convert_kg_to_pyg(G: nx.DiGraph, make_undirected: bool = True) -> tuple:
    """
    Convert the NetworkX Knowledge Graph into a PyTorch Geometric Data object.

    Parameters
    ----------
    G : nx.DiGraph
        The Module 3 Knowledge Graph from knowledge_graph.build_knowledge_graph()
    make_undirected : bool
        If True (default), each directed KG edge is added in both directions.

        Why undirected by default: the KG is built directed (context → intervention)
        because Layer 1's rule traversal only ever reads in that direction. But a
        GCN learns node embeddings by aggregating from neighbours, and a purely
        directed graph would mean context nodes never receive any signal back from
        the interventions they connect to — their embeddings would be based on
        nothing but their own initial features. Making edges bidirectional lets
        representations co-adapt in both directions, which is standard practice
        for KG-based GCN recommenders. The original edge direction is preserved
        in the returned edge_type list for reference.

    Returns
    -------
    data : torch_geometric.data.Data
        With .x (node features), .edge_index, .edge_weight
    node_to_idx : dict
        KG node ID string → tensor row index
    idx_to_node : dict
        Tensor row index → KG node ID string
    """
    # ── Build a stable node ordering ───────────────────────────────────────────
    # Sorted for determinism — the same KG must always produce the same index
    # mapping, so trained models remain valid across runs.
    nodes = sorted(G.nodes())
    node_to_idx = {node: i for i, node in enumerate(nodes)}
    idx_to_node = {i: node for node, i in node_to_idx.items()}

    # ── Node feature matrix ────────────────────────────────────────────────────
    x = torch.tensor(
        [encode_node_features(node, G.nodes[node]) for node in nodes],
        dtype=torch.float,
    )

    # ── Edge index and edge weights ────────────────────────────────────────────
    src_list, dst_list, weight_list = [], [], []

    for u, v, attrs in G.edges(data=True):
        w = float(attrs.get("weight", 0.0))
        ui, vi = node_to_idx[u], node_to_idx[v]

        src_list.append(ui)
        dst_list.append(vi)
        weight_list.append(w)

        if make_undirected:
            src_list.append(vi)
            dst_list.append(ui)
            weight_list.append(w)

    edge_index  = torch.tensor([src_list, dst_list], dtype=torch.long)
    edge_weight = torch.tensor(weight_list, dtype=torch.float)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)

    return data, node_to_idx, idx_to_node


# ══════════════════════════════════════════════════════════════════════════════
# INSPECTION / VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def print_conversion_summary(data: Data, node_to_idx: dict, G: nx.DiGraph) -> None:
    """Print a structured summary confirming the conversion is correct."""
    print("\n" + "═" * 65)
    print("  LAYER 2 — KG → PYTORCH GEOMETRIC CONVERSION SUMMARY")
    print("═" * 65)

    print(f"\n  Source KG (NetworkX):")
    print(f"    Nodes                  : {G.number_of_nodes()}")
    print(f"    Edges (directed)       : {G.number_of_edges()}")

    print(f"\n  Converted (PyTorch Geometric):")
    print(f"    Node feature matrix    : {tuple(data.x.shape)}")
    print(f"    Edge index             : {tuple(data.edge_index.shape)}")
    print(f"    Edge weight vector     : {tuple(data.edge_weight.shape)}")
    print(f"    Feature dimensions     : {data.x.shape[1]}")

    # ── Consistency checks ─────────────────────────────────────────────────────
    print(f"\n  Consistency checks:")

    node_match = data.x.shape[0] == G.number_of_nodes()
    print(f"    Node count matches KG              : {'✓ PASS' if node_match else '✗ FAIL'}")

    expected_edges = G.number_of_edges() * 2  # undirected doubling
    edge_match = data.edge_index.shape[1] == expected_edges
    print(f"    Edge count = 2x KG (bidirectional) : {'✓ PASS' if edge_match else '✗ FAIL'}"
          f"  ({data.edge_index.shape[1]} vs expected {expected_edges})")

    weight_match = data.edge_weight.shape[0] == data.edge_index.shape[1]
    print(f"    Edge weights align with edges      : {'✓ PASS' if weight_match else '✗ FAIL'}")

    idx_valid = data.edge_index.max().item() < data.x.shape[0]
    print(f"    All edge indices within node range : {'✓ PASS' if idx_valid else '✗ FAIL'}")

    no_nan = not torch.isnan(data.x).any().item()
    print(f"    No NaN values in node features     : {'✓ PASS' if no_nan else '✗ FAIL'}")

    # ── Node type distribution in the tensor ───────────────────────────────────
    print(f"\n  Node type distribution (from feature matrix):")
    for i, ntype in enumerate(NODE_TYPES):
        count = int(data.x[:, i].sum().item())
        print(f"    {ntype:<20} {count:3d} nodes")

    # ── Sample encodings, for eyeball verification ─────────────────────────────
    print(f"\n  Sample node encodings:")
    for sample in ["StressState:acute", "Intervention:I02", "GestureProfile:support_detected"]:
        if sample in node_to_idx:
            idx = node_to_idx[sample]
            vec = data.x[idx]
            nonzero = [(i, round(v.item(), 3)) for i, v in enumerate(vec) if v.item() != 0.0]
            print(f"    {sample:<35} idx={idx:2d}  nonzero dims: {nonzero}")

    print(f"\n  Edge weight statistics:")
    print(f"    Min                    : {data.edge_weight.min().item():.3f}")
    print(f"    Max                    : {data.edge_weight.max().item():.3f}")
    print(f"    Mean                   : {data.edge_weight.mean().item():.3f}")
    zero_weights = int((data.edge_weight == 0.0).sum().item())
    neg_weights  = int((data.edge_weight < 0.0).sum().item())
    print(f"    Zero-weight edges      : {zero_weights}  (hard feasibility exclusions)")
    print(f"    Negative-weight edges  : {neg_weights}  (context penalties, e.g. walk at work)")

    # ── Degree safety check for negative edge weights ──────────────────────────
    # GCNConv's symmetric normalisation computes per-node weighted degree, then
    # raises it to the power -0.5. A node whose weighted degree went negative
    # would produce NaN. Layer 1's KG contains a small number of deliberately
    # negative weights (contextual penalties like "brisk walk while at work"),
    # so this check confirms every node's total degree stays safely positive.
    num_nodes = data.x.shape[0]
    deg = torch.zeros(num_nodes)
    deg.index_add_(0, data.edge_index[0], data.edge_weight)
    deg += 1.0  # GCNConv adds self-loops with weight 1.0 by default
    min_deg = deg.min().item()
    deg_safe = min_deg > 0
    print(f"\n    Min per-node weighted degree : {min_deg:.3f}")
    print(f"    Degree safety (no NaN risk)  : {'✓ PASS' if deg_safe else '✗ FAIL — negative degree'}")

    print("\n" + "═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT (standalone test)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building Knowledge Graph from Layer 1...")
    G = build_knowledge_graph()
    print(f"[✓] KG built: {G.number_of_nodes()} nodes, {G.number_of_edges()} edges")

    print("\nConverting to PyTorch Geometric format...")
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)
    print("[✓] Conversion complete")

    print_conversion_summary(data, node_to_idx, G)

    print("[✓] Graph is ready for GCN training (Layer 2 Step 2).\n")