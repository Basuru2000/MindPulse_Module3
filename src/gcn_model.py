"""
MindPulse — Module 3  |  Layer 2: GCN Model Architecture
=========================================================
Defines the 2-layer Graph Convolutional Network and the dot-product scoring
head that together form the Layer 2 recommender.

THIS FILE DEFINES THE MODEL ONLY — no training, no data loading, no
evaluation. Training is Step 4 (`train_gcn.py`), evaluation is Step 5.
Keeping architecture separate from training means the same model definition
is imported by the training script, the evaluation script, and the
GNNExplainer integration, so all three are guaranteed to use an identical
network.

WHAT THE MODEL DOES
-------------------
Layer 1 scores (context → intervention) links with hand-written rules.
Layer 2 learns to score the same links from the graph's structure instead.

    1. ENCODE   Every one of the 46 KG nodes is passed through two GCN
                layers, producing a 32-dimensional embedding per node.
                A node's embedding is shaped by its own features AND by
                its neighbours' features, propagated along the clinically
                weighted edges — this is the "structural learning" that
                a hand-written rule cannot express.

                    (46, 18) --GCNConv--> (46, 64) --GCNConv--> (46, 32)

    2. POOL     A stress episode activates five context nodes at once
                (StressState, TriggerContext, LocationContext,
                SocialContext, GestureProfile). Their five embeddings are
                mean-pooled into a single 32-dim "current situation" vector.

    3. SCORE    That situation vector is compared against an intervention's
                embedding by dot product, yielding one raw score (logit)
                per (context, intervention) pair.

Higher score = the model believes this intervention suits this situation.

WHY MEAN-POOLING THE FIVE CONTEXT NODES
---------------------------------------
The Module 3 spec (Section 2.2) left one design decision open: how to inject
per-episode context into a GCN that only knows static graph properties. Two
options were considered — mean-pooling the active context nodes' embeddings,
or concatenating a separate per-episode feature vector at the scoring head.

Mean-pooling was chosen because it keeps every learned quantity inside the
graph. The situation vector is composed entirely of node embeddings, so
GNNExplainer (Step 6) can attribute a recommendation back to specific nodes
and edges in the Knowledge Graph. A concatenated side-channel would sit
outside the graph and be invisible to graph-based explanation — which would
undermine the explainability contribution that is central to Module 3's
research framing. This resolves the open question in spec Section 2.2.

WHY A DOT PRODUCT RATHER THAN AN MLP SCORING HEAD
-------------------------------------------------
A learned MLP head would add hundreds of parameters on top of a dataset with
only 172 training positives. The dot product adds two (a bias and a scale)
and forces all the learning into the embeddings themselves, which is exactly
what GNNExplainer needs in order to produce meaningful attributions. This
also matches the architecture stated in the spec.

Author : Module 3 — MindPulse (Team MindForge)
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv


# ── Architecture constants (spec Section 2.2) ────────────────────────────────
IN_DIM     = 18   # must equal graph_converter.NODE_FEATURE_DIM
HIDDEN_DIM = 64   # GCN layer 1 output
EMBED_DIM  = 32   # GCN layer 2 output — final node embedding size
DROPOUT    = 0.3  # applied between the two GCN layers, training only


# ══════════════════════════════════════════════════════════════════════════════
# THE MODEL
# ══════════════════════════════════════════════════════════════════════════════

class Layer2GCN(nn.Module):
    """
    2-layer GCN encoder + dot-product scoring head.

    Parameters
    ----------
    in_dim : int
        Input node feature dimensionality. Must match the node feature matrix
        produced by graph_converter.convert_kg_to_pyg() — currently 18.
    hidden_dim : int
        Width of the first GCN layer's output.
    embed_dim : int
        Final node embedding size. Context and intervention embeddings are
        compared in this space.
    dropout : float
        Dropout probability applied between the GCN layers. Active only in
        training mode; model.eval() disables it automatically.
    normalize_embeddings : bool
        If False (default), scores are raw dot products, as specified.
        If True, embeddings are L2-normalised first, making the score a
        cosine similarity bounded to [-1, 1].
    use_residual : bool
        If True (default), a learned linear projection of the raw node
        features is ADDED to the GCN output.

        This is not decoration — without it this model does not work on
        this graph. The Module 3 KG is ~90% of a complete bipartite graph
        (TriggerContext, LocationContext and SocialContext each connect to
        all 22 interventions by construction). On a graph that dense, two
        rounds of neighbourhood averaging drive every intervention
        embedding to the same vector: measured mean pairwise cosine between
        the 22 interventions rises from 0.58 in the raw features to 0.89
        after one GCN layer and 0.998 after two. Identical embeddings make
        ranking impossible, and the model scores at chance.

        This is oversmoothing (Li, Han & Wu, AAAI 2018). The residual path
        preserves each node's own identity alongside the aggregated
        neighbourhood signal, restoring distinguishability to 0.74.
        Measured effect, 5-fold CV AUC on the training sessions:
        0.653 without, 0.730 with.
    """

    def __init__(self,
                 in_dim: int = IN_DIM,
                 hidden_dim: int = HIDDEN_DIM,
                 embed_dim: int = EMBED_DIM,
                 dropout: float = DROPOUT,
                 normalize_embeddings: bool = False,
                 use_residual: bool = True):
        super().__init__()

        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.dropout = dropout
        self.normalize_embeddings = normalize_embeddings
        self.use_residual = use_residual

        # ── The two graph convolutions ────────────────────────────────────────
        # add_self_loops=True (the default) matters here: it guarantees every
        # node keeps some of its own signal, and it adds +1.0 to each node's
        # weighted degree, which is what keeps the KG's 24 negative-weight
        # edges from driving any degree to zero or below. graph_converter.py
        # verifies this on every run (min weighted degree 0.870).
        self.conv1 = GCNConv(in_dim, hidden_dim, add_self_loops=True)
        self.conv2 = GCNConv(hidden_dim, embed_dim, add_self_loops=True)

        # ── Residual path (see use_residual in the class docstring) ───────────
        # bias=False because conv2 already carries a bias; a second one here
        # would be redundant and would only add an unidentifiable parameter.
        self.skip = nn.Linear(in_dim, embed_dim, bias=False) if use_residual else None

        # ── Scoring head: two scalars, that is the whole head ─────────────────
        # score_bias absorbs the class base rate. Only 27.5% of training pairs
        # are positive, so the optimal "know nothing" logit is
        # log(0.275/0.725) = -0.97, not 0. Without a bias the embeddings would
        # have to encode that offset themselves, wasting capacity that should
        # be learning context-intervention structure.
        # score_scale is a learnable temperature that lets the model sharpen
        # or soften its confidence independently of embedding magnitude.
        self.score_bias  = nn.Parameter(torch.tensor(0.0))
        self.score_scale = nn.Parameter(torch.tensor(1.0))

    # ── 1. ENCODE ─────────────────────────────────────────────────────────────
    def encode(self, x, edge_index, edge_weight=None):
        """
        Run both GCN layers to produce an embedding for every node.

        Returns a tensor of shape (num_nodes, embed_dim) — i.e. (46, 32).

        Note there is no activation after conv2. The output is an embedding,
        not a prediction; squashing it here would needlessly restrict the
        space the dot product operates in.
        """
        h = self.conv1(x, edge_index, edge_weight)
        h = F.relu(h)
        h = F.dropout(h, p=self.dropout, training=self.training)
        z = self.conv2(h, edge_index, edge_weight)

        # Residual: re-inject each node's own features so identity survives
        # the neighbourhood averaging. Applied BEFORE normalisation so that,
        # when normalisation is on, it acts on the combined representation.
        if self.skip is not None:
            z = z + self.skip(x)

        if self.normalize_embeddings:
            z = F.normalize(z, p=2, dim=-1)
        return z

    # ── 2. POOL ───────────────────────────────────────────────────────────────
    def pool_context(self, z, context_indices):
        """
        Mean-pool the five active context node embeddings into one vector.

        context_indices : LongTensor (num_pairs, 5)
        returns         : FloatTensor (num_pairs, embed_dim)
        """
        return z[context_indices].mean(dim=1)

    # ── 3. SCORE ──────────────────────────────────────────────────────────────
    def score_pairs(self, z, context_indices, intervention_indices):
        """
        Score one intervention per context — used during training.

        context_indices      : LongTensor (num_pairs, 5)
        intervention_indices : LongTensor (num_pairs,)
        returns              : FloatTensor (num_pairs,)  RAW LOGITS

        These are logits, NOT probabilities. Step 4 pairs them with
        BCEWithLogitsLoss, which applies the sigmoid internally in a
        numerically stable way. Applying sigmoid here and then using plain
        BCELoss would be mathematically equivalent but prone to overflow —
        a classic and hard-to-diagnose training bug.
        """
        ctx = self.pool_context(z, context_indices)          # (P, D)
        inv = z[intervention_indices]                        # (P, D)
        dot = (ctx * inv).sum(dim=-1)                        # (P,)
        return dot * self.score_scale + self.score_bias

    def score_all_interventions(self, z, context_indices, intervention_idx):
        """
        Score EVERY intervention for each context — used for ranking at
        evaluation time (Step 5) and at inference.

        context_indices  : LongTensor (num_pairs, 5)
        intervention_idx : LongTensor (num_interventions,)  all 22 node indices
        returns          : FloatTensor (num_pairs, num_interventions)

        Computing the full matrix in one matmul rather than looping is what
        makes Precision@K / NDCG@K over the test set fast — every candidate
        gets scored against every test context in a single operation.
        """
        ctx = self.pool_context(z, context_indices)          # (P, D)
        inv = z[intervention_idx]                            # (M, D)
        dot = ctx @ inv.t()                                  # (P, M)
        return dot * self.score_scale + self.score_bias

    # ── Full forward pass ─────────────────────────────────────────────────────
    def forward(self, x, edge_index, edge_weight=None,
                context_indices=None, intervention_indices=None):
        """
        Encode, then optionally score.

        If context_indices is None, returns node embeddings (46, 32).
        Otherwise returns pair logits (num_pairs,).

        The argument ORDER here is deliberate: PyTorch Geometric's Explainer
        (Step 6) calls model(x, edge_index, **kwargs), so x and edge_index
        must come first and everything else must be keyword-passable. Getting
        this right now avoids having to restructure the model later to make
        GNNExplainer work.
        """
        z = self.encode(x, edge_index, edge_weight)
        if context_indices is None:
            return z
        return self.score_pairs(z, context_indices, intervention_indices)

    # ── Introspection ─────────────────────────────────────────────────────────
    def count_parameters(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)

    def describe(self) -> str:
        return (
            f"Layer2GCN(\n"
            f"    conv1        : GCNConv({self.in_dim} -> {self.hidden_dim})\n"
            f"    activation   : ReLU\n"
            f"    dropout      : {self.dropout} (training only)\n"
            f"    conv2        : GCNConv({self.hidden_dim} -> {self.embed_dim})\n"
            f"    residual     : {self.use_residual}\n"
            f"    pooling      : mean over 5 context nodes\n"
            f"    scoring head : dot product * scale + bias\n"
            f"    normalised   : {self.normalize_embeddings}\n"
            f"    parameters   : {self.count_parameters():,}\n"
            f")"
        )


# ══════════════════════════════════════════════════════════════════════════════
# NUMERICAL SAFETY GUARD
# ══════════════════════════════════════════════════════════════════════════════

def check_degree_safety(edge_index, edge_weight, num_nodes: int) -> tuple:
    """
    Confirm no node's weighted degree can go negative, which would make
    GCNConv's symmetric normalisation produce NaN.

    GCNConv normalises by deg^(-0.5). Raising a negative number to a
    fractional power yields NaN, and that NaN then propagates silently
    through every subsequent layer — training appears to run, the loss
    prints as nan, and the cause is far from obvious. The Module 3 KG
    contains 24 deliberately negative edge weights (contextual penalties
    such as "brisk walk while at work"), so this is a live concern rather
    than a theoretical one.

    IMPORTANT — which index this scatters onto:
    PyTorch Geometric computes degree by scattering edge weights onto
    edge_index[1] (the destination/column index). graph_converter.py's
    equivalent check scatters onto edge_index[0] (source). Today those two
    give identical answers, but ONLY because convert_kg_to_pyg() is called
    with make_undirected=True, which mirrors every edge with the same
    weight and makes in-degree equal out-degree. This function deliberately
    matches PyG's convention so the guard stays correct even if that
    assumption ever changes.

    Returns (is_safe, min_degree).
    """
    deg = torch.zeros(num_nodes, device=edge_index.device)
    deg.index_add_(0, edge_index[1], edge_weight)
    deg = deg + 1.0  # GCNConv adds self-loops with weight 1.0
    min_deg = deg.min().item()
    return min_deg > 0, min_deg


# ══════════════════════════════════════════════════════════════════════════════
# GNNEXPLAINER WRAPPER  (used in Step 6, defined here to keep it beside the model)
# ══════════════════════════════════════════════════════════════════════════════

class ExplainableWrapper(nn.Module):
    """
    Adapts Layer2GCN for PyTorch Geometric's Explainer API.

    The Explainer expects a model whose forward takes (x, edge_index) and
    returns a prediction tensor. Layer2GCN needs the context and intervention
    indices as well, so this wrapper freezes one specific episode's indices
    and exposes the simple signature the Explainer requires.

    Not used until Step 6 — included now only because the argument-order
    decision it depends on is part of this file's design.
    """

    def __init__(self, model: Layer2GCN, context_indices, intervention_indices):
        super().__init__()
        self.model = model
        self.register_buffer("context_indices", context_indices)
        self.register_buffer("intervention_indices", intervention_indices)

    def forward(self, x, edge_index, edge_weight=None, **kwargs):
        return self.model(x, edge_index, edge_weight,
                          self.context_indices, self.intervention_indices)


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST
# ══════════════════════════════════════════════════════════════════════════════

def _self_test():
    """Verify the architecture against the real graph, without training it."""
    from knowledge_graph import build_knowledge_graph
    from graph_converter import convert_kg_to_pyg, NODE_FEATURE_DIM

    print("\n" + "=" * 65)
    print("  LAYER 2 — GCN MODEL ARCHITECTURE SELF-TEST")
    print("=" * 65)

    G = build_knowledge_graph()
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)

    torch.manual_seed(42)
    model = Layer2GCN(in_dim=NODE_FEATURE_DIM)

    print("\n  Architecture:")
    for line in model.describe().split("\n"):
        print(f"  {line}")

    # ── Build a small batch of fake pairs purely to exercise the shapes ───────
    intervention_idx = torch.tensor(
        [node_to_idx[n] for n in sorted(node_to_idx) if n.startswith("Intervention:")],
        dtype=torch.long,
    )
    context_pool = torch.tensor(
        [node_to_idx[n] for n in sorted(node_to_idx) if not n.startswith("Intervention:")],
        dtype=torch.long,
    )
    P = 8
    ctx_idx = context_pool[torch.randint(0, len(context_pool), (P, 5))]
    inv_idx = intervention_idx[torch.randint(0, len(intervention_idx), (P,))]

    print("\n  Shape checks:")

    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_weight)
    ok_embed = tuple(z.shape) == (data.x.shape[0], EMBED_DIM)
    print(f"    Node embeddings  (46, 32)          : "
          f"{'PASS' if ok_embed else 'FAIL'}  got {tuple(z.shape)}")

    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight, ctx_idx, inv_idx)
    ok_logits = tuple(logits.shape) == (P,)
    print(f"    Pair logits      ({P},)             : "
          f"{'PASS' if ok_logits else 'FAIL'}  got {tuple(logits.shape)}")

    with torch.no_grad():
        allscores = model.score_all_interventions(z, ctx_idx, intervention_idx)
    ok_all = tuple(allscores.shape) == (P, len(intervention_idx))
    print(f"    Ranking matrix   ({P}, 22)          : "
          f"{'PASS' if ok_all else 'FAIL'}  got {tuple(allscores.shape)}")

    print("\n  Numerical safety:")
    safe, min_deg = check_degree_safety(data.edge_index, data.edge_weight, data.x.shape[0])
    print(f"    Min weighted degree (PyG convention): {min_deg:.3f}")
    print(f"    Degree safe (no NaN risk)          : {'PASS' if safe else 'FAIL'}")
    finite_z = bool(torch.isfinite(z).all().item())
    finite_l = bool(torch.isfinite(logits).all().item())
    print(f"    Embeddings finite (no NaN/Inf)     : {'PASS' if finite_z else 'FAIL'}")
    print(f"    Logits finite (no NaN/Inf)         : {'PASS' if finite_l else 'FAIL'}")
    print(f"    Negative edge weights survived     : "
          f"{'PASS' if finite_z else 'FAIL'}  "
          f"({int((data.edge_weight < 0).sum().item())} negative edges present)")

    print("\n  Gradient flow:")
    model.train()
    out = model(data.x, data.edge_index, data.edge_weight, ctx_idx, inv_idx)
    loss = F.binary_cross_entropy_with_logits(out, torch.rand(P).round())
    loss.backward()
    missing = [n for n, p in model.named_parameters()
               if p.requires_grad and (p.grad is None or not torch.isfinite(p.grad).all())]
    print(f"    All parameters receive gradients    : "
          f"{'PASS' if not missing else 'FAIL ' + str(missing)}")

    print("\n  Dropout behaviour:")
    model.train()
    a = model.encode(data.x, data.edge_index, data.edge_weight)
    b = model.encode(data.x, data.edge_index, data.edge_weight)
    train_differs = not torch.allclose(a, b)
    model.eval()
    with torch.no_grad():
        c = model.encode(data.x, data.edge_index, data.edge_weight)
        d = model.encode(data.x, data.edge_index, data.edge_weight)
    eval_same = torch.allclose(c, d)
    print(f"    Stochastic in train mode           : {'PASS' if train_differs else 'FAIL'}")
    print(f"    Deterministic in eval mode         : {'PASS' if eval_same else 'FAIL'}")

    print("\n" + "=" * 65)
    print("  [OK] Architecture verified. Ready for training (Layer 2 Step 4).")
    print("=" * 65 + "\n")


if __name__ == "__main__":
    _self_test()