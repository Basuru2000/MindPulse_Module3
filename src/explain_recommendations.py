"""
MindPulse — Module 3  |  Layer 2: Explainability
=================================================
Produces human-readable explanations for Layer 2's recommendations, satisfying
FR3-06 ("apply GNNExplainer to the trained GCN to produce a subgraph
explanation for each delivered recommendation").

TWO METHODS, AND WHY BOTH ARE HERE
-----------------------------------
1. EXACT CONTEXT ATTRIBUTION  (primary; used for the user-facing string)

   Layer 2's score is  mean(z_ctx) . z_inv, and mean is linear, so the score
   decomposes EXACTLY into one term per active context node:

       score = (1/5) * SUM_c ( z_c . z_inv ) * scale + bias

   Each of the five context signals therefore has an exact, signed
   contribution. No mask is learned, nothing is approximated, and the result
   is identical on every run. Verified to reconstruct the model's own score to
   ~1e-7. This is what generates "Recommended because: acute stress tier +
   behavioural support detected".

2. GNNEXPLAINER  (spec-required; reported with measured limitations)

   Learns soft masks over nodes and edges to find the subgraph driving the
   prediction. Implemented per FR3-06 and per Ying et al. (NeurIPS 2019).

   On THIS model it does not work well, and this file measures that rather
   than hiding it. The ablation in Step 6 analysis found the KG's edge
   structure contributes +0.011 CV AUC over a no-graph model (0.732 vs 0.721,
   inside one standard deviation), because the edge weights are deterministic
   functions of node attributes that the 18-dim node features already encode.
   The residual path consequently carries ~74% of embedding magnitude.

   Since the graph carries little of the decision, GNNExplainer's edge masks
   have poor fidelity (removing "important" edges perturbs the score no more
   than removing random ones) and poor stability across seeds. The
   run_diagnostics() function below quantifies both, so the limitation is
   evidence in the report rather than an unexamined claim.

   This is a finding about the architecture, not a defect in GNNExplainer: a
   knowledge graph whose edge weights are derived deterministically from node
   attributes gives message passing nothing new to propagate.

Usage
-----
    python src/explain_recommendations.py              # examples + diagnostics
    python src/explain_recommendations.py --no-diag    # examples only (fast)

Requires models/gcn_layer2.pt from src/train_gcn.py.

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import argparse
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import build_knowledge_graph, INTERVENTION_MAP
from graph_converter import convert_kg_to_pyg
from rule_engine import Layer1RuleEngine
from gcn_model import Layer2GCN, ExplainableWrapper
from training_pairs import get_context_node_indices
from layer1_evaluation import DATASET_PATH

MODEL_PATH      = os.path.join("models", "gcn_layer2.pt")
OUTPUT_PATH     = os.path.join("data", "synthetic", "layer2_explanations.csv")
FULL_POOL_K     = 22
EXPLAINER_EPOCHS = 200
N_EXAMPLES      = 5
N_DIAG_SESSIONS = 12
N_DIAG_SEEDS    = 5
EDGES_REMOVED   = 100     # of 976, for the fidelity probe
RANDOM_TRIALS   = 10

# Readable labels for the five context node types
CONTEXT_PHRASING = {
    "StressState":     lambda v: f"{v} stress tier",
    "TriggerContext":  lambda v: (f"{v} trigger" if v != "unknown" else "unspecified trigger"),
    "LocationContext": lambda v: (f"at {v}" if v != "unknown" else "unknown location"),
    "SocialContext":   lambda v: (f"with {v}" if v != "unknown" else "unknown social setting"),
    "GestureProfile":  lambda v: ("behavioural support detected" if v == "support_detected"
                                  else "no behavioural support signal"),
}


def phrase(node_name: str) -> str:
    ntype, _, value = node_name.partition(":")
    fn = CONTEXT_PHRASING.get(ntype)
    return fn(value) if fn else node_name


# ══════════════════════════════════════════════════════════════════════════════
# SETUP
# ══════════════════════════════════════════════════════════════════════════════

def load_everything():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run src/train_gcn.py first.")
    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")

    G = build_knowledge_graph()
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model = Layer2GCN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_weight)
    df = pd.read_csv(DATASET_PATH)
    engine = Layer1RuleEngine(G, top_k=FULL_POOL_K)
    return model, ckpt, data, node_to_idx, idx_to_node, z, df, engine


def layer2_recommend(model, z, engine, row, node_to_idx):
    """Layer 1 gates the feasible pool; Layer 2 reorders it. Returns top pick."""
    result = engine.recommend({
        "tier": row["tier"], "confidence": row["confidence"],
        "baseline_deviation": row["baseline_deviation"],
        "support_detected": bool(row["support_detected"]),
        "support_score": float(row["support_score"]),
        "signal_quality": row["signal_quality"],
        "baseline_mode": row["baseline_mode"],
        "dialogue_completed": bool(row["dialogue_completed"]),
        "trigger_type": row.get("trigger_type", "unknown"),
        "trigger_confidence": row.get("trigger_confidence", 0.70),
        "location_context": row.get("location_context", "unknown"),
        "social_context": row.get("social_context", "unknown"),
        "timestamp": row.get("timestamp", None),
        "time_of_day": row.get("time_of_day", "afternoon"),
    })
    if not result["ranked_candidates"]:
        return None, None, result

    pool = [c["intervention_id"] for c in result["ranked_candidates"]]
    ctx = torch.tensor([get_context_node_indices(row, node_to_idx)], dtype=torch.long)
    cand = torch.tensor([node_to_idx[f"Intervention:{i}"] for i in pool], dtype=torch.long)
    with torch.no_grad():
        scores = model.score_all_interventions(z, ctx, cand)[0].numpy()
    return pool[int(np.argmax(scores))], ctx, result


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 1 — EXACT CONTEXT ATTRIBUTION
# ══════════════════════════════════════════════════════════════════════════════

def exact_attribution(model, z, ctx_indices, inv_index):
    """
    Decompose the score into one exact, signed contribution per context node.

    Because score = mean(z_ctx) . z_inv * scale + bias and mean is linear,
    each context node's term is (1/n) * (z_c . z_inv) * scale. These sum,
    with the bias, to exactly the model's own score — asserted below, since a
    silent mismatch would mean the explanation describes a different quantity
    than the recommendation.
    """
    n = len(ctx_indices)
    scale = model.score_scale.item()
    bias = model.score_bias.item()

    with torch.no_grad():
        contribs = [float((z[c] * z[inv_index]).sum().item()) / n * scale
                    for c in ctx_indices]
        actual = model.score_pairs(
            z, torch.tensor([ctx_indices], dtype=torch.long),
            torch.tensor([inv_index], dtype=torch.long)).item()

    reconstructed = sum(contribs) + bias
    assert abs(reconstructed - actual) < 1e-4, (
        f"Attribution does not reconstruct the score ({reconstructed} vs {actual}). "
        "The explanation would describe a different quantity than the recommendation.")
    return contribs, actual


def build_explanation(idx_to_node, ctx_indices, contribs, inv_id, score, top_n=2):
    """Human-readable string in the format required by spec Section 2.2."""
    inv = INTERVENTION_MAP[inv_id]
    ranked = sorted(zip(ctx_indices, contribs), key=lambda t: -t[1])
    drivers = [phrase(idx_to_node[i]) for i, c in ranked[:top_n] if c > 0]
    if not drivers:
        drivers = [phrase(idx_to_node[ranked[0][0]])]
    # ranked is sorted descending, so the strongest opposing factor is the LAST
    # negative entry, not the first. Taking the first would surface the weakest
    # objection and misdescribe why the score was held down.
    negatives = [i for i, c in ranked if c < 0]
    opposing = [phrase(idx_to_node[negatives[-1]])] if negatives else []

    text = (f"Recommended '{inv['name']}' ({inv['type']}, {inv['duration']} min) "
            f"because: {' + '.join(drivers)}.")
    if opposing:
        text += f" Weighed against: {opposing[0]}."
    return text + f" Layer 2 score: {score:+.4f}."


# ══════════════════════════════════════════════════════════════════════════════
# METHOD 2 — GNNEXPLAINER  (FR3-06)
# ══════════════════════════════════════════════════════════════════════════════

def gnnexplainer_masks(model, data, ctx, inv_index, seed=0, epochs=EXPLAINER_EPOCHS):
    """
    Run GNNExplainer and return (node_mask, edge_mask).

    ExplainableWrapper freezes this episode's context and intervention indices
    so the model exposes the (x, edge_index) signature the Explainer requires.
    The argument ordering in Layer2GCN.forward was chosen at Step 3 for exactly
    this, so no restructuring is needed here.
    """
    from torch_geometric.explain import Explainer, GNNExplainer

    torch.manual_seed(seed)
    wrapper = ExplainableWrapper(model, ctx, torch.tensor([inv_index], dtype=torch.long))
    explainer = Explainer(
        model=wrapper,
        algorithm=GNNExplainer(epochs=epochs),
        explanation_type="model",
        node_mask_type="object",
        edge_mask_type="object",
        model_config=dict(mode="binary_classification",
                          task_level="graph", return_type="raw"),
    )
    exp = explainer(data.x, data.edge_index, edge_weight=data.edge_weight)
    return exp.node_mask.squeeze(-1).detach(), exp.edge_mask.detach()


def top_subgraph(edge_mask, data, idx_to_node, k=5):
    """Return the k highest-scoring edges as readable (source -> target) pairs."""
    out = []
    for e in edge_mask.argsort(descending=True)[:k]:
        s, t = data.edge_index[0, e].item(), data.edge_index[1, e].item()
        out.append((idx_to_node[s], idx_to_node[t], float(edge_mask[e])))
    return out


# ══════════════════════════════════════════════════════════════════════════════
# DIAGNOSTICS
# ══════════════════════════════════════════════════════════════════════════════

def run_diagnostics(model, data, z, df, engine, node_to_idx, idx_to_node):
    """
    Quantify how much to trust each method. Three probes:

      1. Branch decomposition — how much of the embedding is graph-derived
      2. Fidelity  — does removing 'important' edges change the score more
                     than removing random edges?
      3. Stability — do repeated GNNExplainer runs agree?
    """
    import torch.nn.functional as F

    print("\n" + "=" * 72)
    print("  EXPLAINABILITY DIAGNOSTICS")
    print("=" * 72)

    # ── 1. Branch decomposition ───────────────────────────────────────────────
    with torch.no_grad():
        h = F.relu(model.conv1(data.x, data.edge_index, data.edge_weight))
        gcn = model.conv2(h, data.edge_index, data.edge_weight)
        skip = model.skip(data.x) if model.skip is not None else torch.zeros_like(gcn)
    g_norm, s_norm = gcn.norm(dim=1).mean().item(), skip.norm(dim=1).mean().item()
    share = s_norm / (g_norm + s_norm) if (g_norm + s_norm) else 0.0
    print("\n  1. Where does the embedding come from?")
    print(f"     ||GCN(x)||  mean norm         : {g_norm:.4f}   (graph-dependent)")
    print(f"     ||skip(x)|| mean norm         : {s_norm:.4f}   (edge-independent)")
    print(f"     residual share of magnitude   : {share:.1%}")
    print("     GNNExplainer can only mask edges and features, so it explains")
    print("     the graph branch only.")

    # ── Collect eligible sessions ─────────────────────────────────────────────
    sessions = []
    for _, row in df.iterrows():
        inv_id, ctx, _ = layer2_recommend(model, z, engine, row, node_to_idx)
        if inv_id is not None:
            sessions.append((row, inv_id, ctx))
        if len(sessions) >= N_DIAG_SESSIONS:
            break

    # ── 2. Fidelity ───────────────────────────────────────────────────────────
    rng = np.random.default_rng(0)
    imp_deltas, rnd_deltas = [], []
    for row, inv_id, ctx in sessions:
        inv_index = node_to_idx[f"Intervention:{inv_id}"]
        inv_t = torch.tensor([inv_index], dtype=torch.long)
        _, edge_mask = gnnexplainer_masks(model, data, ctx, inv_index, seed=0)
        with torch.no_grad():
            base = model(data.x, data.edge_index, data.edge_weight, ctx, inv_t).item()

        ew = data.edge_weight.clone()
        ew[edge_mask.argsort(descending=True)[:EDGES_REMOVED]] = 0.0
        with torch.no_grad():
            imp_deltas.append(abs(base - model(data.x, data.edge_index, ew, ctx, inv_t).item()))

        trials = []
        for _ in range(RANDOM_TRIALS):
            ew2 = data.edge_weight.clone()
            ew2[torch.tensor(rng.choice(data.edge_weight.shape[0],
                                        EDGES_REMOVED, replace=False))] = 0.0
            with torch.no_grad():
                trials.append(abs(base - model(data.x, data.edge_index, ew2, ctx, inv_t).item()))
        rnd_deltas.append(float(np.mean(trials)))

    mi, mr = float(np.mean(imp_deltas)), float(np.mean(rnd_deltas))
    ratio = mi / mr if mr else float("nan")
    print(f"\n  2. Fidelity ({len(sessions)} sessions, top-{EDGES_REMOVED} edges removed)")
    print(f"     mean |score change|, important edges : {mi:.4f}")
    print(f"     mean |score change|, random edges    : {mr:.4f}")
    print(f"     ratio (want >> 1.0)                  : {ratio:.2f}   "
          f"{'PASS' if ratio > 1.5 else 'FAIL — masks are not identifying influential edges'}")

    # ── 3. Stability ──────────────────────────────────────────────────────────
    jaccards = []
    for row, inv_id, ctx in sessions[:5]:
        inv_index = node_to_idx[f"Intervention:{inv_id}"]
        tops = []
        for s in range(N_DIAG_SEEDS):
            nm, _ = gnnexplainer_masks(model, data, ctx, inv_index, seed=s)
            tops.append(set(nm.argsort(descending=True)[:5].tolist()))
        jaccards.append(len(set.intersection(*tops)) / len(set.union(*tops)))
    mj = float(np.mean(jaccards))
    print(f"\n  3. Stability (top-5 nodes across {N_DIAG_SEEDS} seeds)")
    print(f"     mean Jaccard (want ~1.0)             : {mj:.2f}   "
          f"{'PASS' if mj > 0.6 else 'FAIL — explanations vary run to run'}")

    print("\n  " + "-" * 68)
    print("  VERDICT")
    print("  " + "-" * 68)
    if ratio > 1.5 and mj > 0.6:
        print("  GNNExplainer output is reliable on this model; use it directly.")
    else:
        print("  GNNExplainer output is NOT reliable on this model. Both probes point")
        print("  to the same cause: the KG's edge weights are deterministic functions")
        print("  of node attributes already present in the 18-dim node features, so")
        print("  message passing carries little of the decision (ablation: +0.011 CV")
        print("  AUC over a no-graph model, within one standard deviation).")
        print("  USE EXACT CONTEXT ATTRIBUTION for user-facing explanations. Report")
        print("  the GNNExplainer subgraph as a spec-required component whose")
        print("  limitations were measured, not as a validated explanation method.")
    return {"residual_share": share, "fidelity_ratio": ratio, "stability_jaccard": mj}


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main(run_diag=True):
    print("\n" + "=" * 72)
    print("  MODULE 3 — LAYER 2 EXPLAINABILITY  (FR3-06)")
    print("=" * 72)

    model, ckpt, data, node_to_idx, idx_to_node, z, df, engine = load_everything()
    print(f"\n  Loaded model : {MODEL_PATH}  "
          f"(CV AUC {ckpt['training']['cv_auc_mean']:.3f})")

    rows, shown = [], 0
    for _, row in df.iterrows():
        if shown >= N_EXAMPLES:
            break
        inv_id, ctx, l1 = layer2_recommend(model, z, engine, row, node_to_idx)
        if inv_id is None:
            continue

        ctx_indices = ctx[0].tolist()
        inv_index = node_to_idx[f"Intervention:{inv_id}"]
        contribs, score = exact_attribution(model, z, ctx_indices, inv_index)
        text = build_explanation(idx_to_node, ctx_indices, contribs, inv_id, score)

        print("\n" + "-" * 72)
        print(f"  SESSION {row['session_id']}  (user {row['user_id']})")
        print("-" * 72)
        print(f"  Layer 2 recommends : {inv_id} — {INTERVENTION_MAP[inv_id]['name']}")
        print(f"\n  Exact context attribution (sums to the score):")
        for i, c in sorted(zip(ctx_indices, contribs), key=lambda t: -t[1]):
            bar = "#" * min(int(abs(c) * 8), 28)
            print(f"    {idx_to_node[i]:<36} {c:+.4f}  {bar}")
        print(f"\n  EXPLANATION: {text}")
        print(f"\n  Layer 1 said: {l1['explanation'][:150]}")

        nm, em = gnnexplainer_masks(model, data, ctx, inv_index, seed=0)
        print(f"\n  GNNExplainer subgraph (FR3-06, see diagnostics for reliability):")
        for s, t, w in top_subgraph(em, data, idx_to_node, k=3):
            print(f"    {s:<32} -> {t:<26} {w:.3f}")

        rows.append({"session_id": row["session_id"], "user_id": row["user_id"],
                     "intervention_id": inv_id, "layer2_score": score,
                     "explanation": text,
                     **{f"contrib__{idx_to_node[i]}": c
                        for i, c in zip(ctx_indices, contribs)}})
        shown += 1

    diag = run_diagnostics(model, data, z, df, engine, node_to_idx, idx_to_node) \
        if run_diag else None

    os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(OUTPUT_PATH, index=False)
    print(f"\n  Saved explanations -> {OUTPUT_PATH}")
    print("=" * 72 + "\n")
    return diag


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--no-diag", action="store_true",
                    help="skip diagnostics (faster)")
    args = ap.parse_args()
    main(run_diag=not args.no_diag)