"""
MindPulse — Module 3  |  Layer 2: Evaluation
=============================================
Evaluates the trained Layer 2 GCN against Layer 1 and the content-based
baseline on the 125 held-out test sessions, completing the progressive
improvement comparison:

    Content-Based Baseline  <  Layer 1 (rule-based KG)  <  Layer 2 (GCN)

layer1_evaluation.py IS NOT MODIFIED
------------------------------------
That file is verified and its numbers are already reported. This script
imports the three scoring functions it exposes at module level —
precision_at_k, ndcg_at_k and content_based_score — so the metric
definitions themselves are shared, never reimplemented.

The train/test split, ground-truth construction and episode building live
inside layer1_evaluation.run_evaluation() and are not importable, so they are
reproduced below. Duplicated logic can drift, and drift here would silently
invalidate the whole comparison — so rather than trusting the copy, this
script RE-DERIVES Layer 1's metrics and asserts they match the published
values numerically (see verify_layer1_parity). That is a stronger guarantee
than an import: an import proves the same code ran, whereas this proves the
same numbers came out.

THE CANDIDATE POOL IS SHARED
----------------------------
Both layers rank the SAME pool: the full feasible set Layer 1's eight gating
rules admit for that session (engine instantiated with top_k=22 so nothing is
truncated). Layer 1 orders it by rule score; Layer 2 reorders it by GCN score;
top-K is taken from each.

This matters more than it looks. If Layer 2 merely reordered Layer 1's top-5,
Precision@5 and Hit Rate@5 would be mathematically IDENTICAL for both layers —
same set, different order — and only NDCG could move. The headline metrics
would flatline for reasons unrelated to model quality.

Measured on the real test set the pool averages 11.6 interventions (min 4,
max 15); only 4 of 112 evaluated sessions have a pool of 5 or fewer. Layer 2
can genuinely change the top-5 for 96% of sessions and the top-3 for all.

Sharing the pool also resolves the I22 `requires_time` gap: the GCN's node
features encode no time information, but Layer 1's feasibility gate removes
time-inappropriate interventions before Layer 2 sees them. Feasibility is
Layer 1's responsibility by design.

WHY CONFIDENCE INTERVALS
------------------------
112 evaluated sessions is small, and training-time cross-validation showed
fold AUC ranging 0.582-0.747. Point estimates alone are not trustworthy here.
Every Layer 2 minus Layer 1 difference is reported with a paired bootstrap 95%
interval. Pairing matters: both layers see the same sessions, so sessions are
resampled together and the correlation between layers is preserved.

Usage
-----
    python src/layer2_evaluation.py      (requires models/gcn_layer2.pt)

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch
from collections import defaultdict

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import build_knowledge_graph, INTERVENTION_MAP
from graph_converter import convert_kg_to_pyg
from rule_engine import Layer1RuleEngine
from gcn_model import Layer2GCN
from training_pairs import get_context_node_indices

# Only these exist at module level in layer1_evaluation.py — importing the
# metric definitions is what keeps the two layers measured the same way.
from layer1_evaluation import (
    precision_at_k,
    ndcg_at_k,
    content_based_score,
    DATASET_PATH,
    K_VALUES,
    RELEVANCE_THRESHOLD,
    TRAIN_RATIO,
    RANDOM_SEED,
)

MODEL_PATH     = os.path.join("models", "gcn_layer2.pt")
RESULTS_PATH   = os.path.join("data", "synthetic", "layer2_evaluation_results.csv")
FULL_POOL_K    = 22
N_BOOTSTRAP    = 2000
BOOTSTRAP_SEED = 42

# Layer 1's published results, from layer1_evaluation.py on this dataset.
# Used as a tripwire: if the logic reproduced below ever diverges from the
# original, these stop matching and the run aborts instead of silently
# reporting an invalid comparison.
REFERENCE_LAYER1 = {
    "n_evaluated": 112, "n_suppressed": 13,
    "prec@3": 0.5387, "ndcg@3": 0.5190, "hit@3": 0.2679,
    "prec@5": 0.5732, "ndcg@5": 0.5584, "hit@5": 0.5089,
    "mor": 3.2354,
}
PARITY_TOL = 1e-3


# ══════════════════════════════════════════════════════════════════════════════
# REPRODUCED FROM layer1_evaluation.run_evaluation()
# ══════════════════════════════════════════════════════════════════════════════
# These three mirror logic that is inline inside run_evaluation() and therefore
# cannot be imported. verify_layer1_parity() below proves they behave identically.

def make_test_mask(df: pd.DataFrame) -> np.ndarray:
    """Per-user 80/20 split, seed 42 — identical to layer1_evaluation.py."""
    np.random.seed(RANDOM_SEED)
    test_mask = np.zeros(len(df), dtype=bool)
    for user_id in df["user_id"].unique():
        user_idx = df[df["user_id"] == user_id].index.tolist()
        n_test = max(1, int(len(user_idx) * (1 - TRAIN_RATIO)))
        test_idx = np.random.choice(user_idx, size=n_test, replace=False)
        test_mask[test_idx] = True
    return test_mask


def build_ground_truth(df: pd.DataFrame, test_mask: np.ndarray):
    """Per-user relevance sets and mean ratings from TRAINING sessions only."""
    df_train = df[~test_mask].copy()
    user_relevant = defaultdict(set)
    user_ratings = defaultdict(dict)

    for _, row in df_train.iterrows():
        uid, inv_id = row["user_id"], row["intervention_id"]
        if inv_id not in user_ratings[uid]:
            user_ratings[uid][inv_id] = []
        user_ratings[uid][inv_id].append(row["outcome_rating"])

    for uid, inv_ratings in user_ratings.items():
        for inv_id, ratings in inv_ratings.items():
            avg = np.mean(ratings)
            user_ratings[uid][inv_id] = round(avg, 2)
            if avg >= RELEVANCE_THRESHOLD:
                user_relevant[uid].add(inv_id)

    return user_relevant, user_ratings


def build_episode(row: pd.Series) -> dict:
    """Row -> episode dict, identical to layer1_evaluation.py."""
    return {
        "tier":               row["tier"],
        "confidence":         row["confidence"],
        "baseline_deviation": row["baseline_deviation"],
        "support_detected":   bool(row["support_detected"]),
        "support_score":      float(row["support_score"]),
        "signal_quality":     row["signal_quality"],
        "baseline_mode":      row["baseline_mode"],
        "dialogue_completed": bool(row["dialogue_completed"]),
        "trigger_type":       row.get("trigger_type", "unknown"),
        "trigger_confidence": row.get("trigger_confidence", 0.70),
        "location_context":   row.get("location_context", "unknown"),
        "social_context":     row.get("social_context", "unknown"),
        "timestamp":          row.get("timestamp", None),
        "time_of_day":        row.get("time_of_day", "afternoon"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# SUPPORT
# ══════════════════════════════════════════════════════════════════════════════

def load_trained_model():
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"Trained model not found at {MODEL_PATH}. Run src/train_gcn.py first."
        )
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model = Layer2GCN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def paired_bootstrap_ci(a: np.ndarray, b: np.ndarray,
                        n_boot: int = N_BOOTSTRAP, seed: int = BOOTSTRAP_SEED):
    """
    Paired bootstrap 95% CI for mean(a) - mean(b).

    a and b hold per-session values for the same sessions in the same order.
    Sessions are resampled together so the correlation between the two layers
    on a given session is preserved; resampling independently would inflate
    variance and give misleadingly wide intervals.
    """
    rng = np.random.default_rng(seed)
    observed = float(a.mean() - b.mean())
    idx = rng.integers(0, len(a), size=(n_boot, len(a)))
    diffs = a[idx].mean(axis=1) - b[idx].mean(axis=1)
    return observed, float(np.percentile(diffs, 2.5)), float(np.percentile(diffs, 97.5))


def verify_layer1_parity(res: pd.DataFrame, n_suppressed: int) -> bool:
    """
    Confirm the Layer 1 numbers produced here match layer1_evaluation.py.

    If this fails, the comparison in this file is not measuring the same thing
    layer1_evaluation.py measured, and no result below should be trusted.
    """
    got = {
        "n_evaluated": len(res), "n_suppressed": n_suppressed,
        "prec@3": res["l1_prec@3"].mean(), "ndcg@3": res["l1_ndcg@3"].mean(),
        "hit@3": res["l1_hit@3"].mean(),   "prec@5": res["l1_prec@5"].mean(),
        "ndcg@5": res["l1_ndcg@5"].mean(), "hit@5": res["l1_hit@5"].mean(),
        "mor": res["l1_mor"].mean(),
    }
    print("\n  Layer 1 parity check (vs layer1_evaluation.py):")
    ok = True
    for key, expected in REFERENCE_LAYER1.items():
        actual = got[key]
        match = abs(actual - expected) <= (0 if isinstance(expected, int) else PARITY_TOL)
        ok &= match
        print(f"    {key:<14} expected {expected:<10} got {actual:<10.4f} "
              f"{'PASS' if match else 'FAIL'}")
    return bool(ok)


# ══════════════════════════════════════════════════════════════════════════════
# EVALUATION
# ══════════════════════════════════════════════════════════════════════════════

def run_evaluation():
    print("\n" + "=" * 72)
    print("  MODULE 3 — LAYER 2 EVALUATION  (Baseline vs Layer 1 vs Layer 2)")
    print("=" * 72)

    if not os.path.exists(DATASET_PATH):
        raise FileNotFoundError(f"Dataset not found at {DATASET_PATH}.")

    df = pd.read_csv(DATASET_PATH)
    test_mask = make_test_mask(df)
    df_test = df[test_mask].copy().reset_index(drop=True)

    G = build_knowledge_graph()
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)
    engine = Layer1RuleEngine(G, top_k=FULL_POOL_K)

    model, ckpt = load_trained_model()
    print(f"\n  Loaded model : {MODEL_PATH}")
    print(f"    lr {ckpt['training']['lr']}, {ckpt['training']['epochs']} epochs, "
          f"CV AUC {ckpt['training']['cv_auc_mean']:.3f} "
          f"+/- {ckpt['training']['cv_auc_std']:.3f}")

    # Node embeddings are static — the graph does not change per session.
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_weight)

    user_relevant, user_ratings = build_ground_truth(df, test_mask)

    rows, pool_sizes, suppressed = [], [], 0

    for _, row in df_test.iterrows():
        uid = row["user_id"]
        episode = build_episode(row)
        gt_inv, gt_rating = row["intervention_id"], row["outcome_rating"]

        relevant_ids = user_relevant.get(uid, {gt_inv}) or {gt_inv}
        rating_map = user_ratings.get(uid, {})
        if gt_inv not in rating_map:
            rating_map = dict(rating_map)
            rating_map[gt_inv] = gt_rating

        result = engine.recommend(episode)
        if not result["ranked_candidates"]:
            suppressed += 1
            continue

        pool_ids = [c["intervention_id"] for c in result["ranked_candidates"]]
        pool_sizes.append(len(pool_ids))
        l1_ids = pool_ids                      # Layer 1's own ordering

        # ── Layer 2: same pool, reordered by GCN score ────────────────────────
        # Context nodes resolved via training_pairs.get_context_node_indices so
        # evaluation encodes context exactly as training did, including the
        # dialogue_completed fallback to "unknown".
        ctx = torch.tensor([get_context_node_indices(row, node_to_idx)],
                           dtype=torch.long)
        cand = torch.tensor([node_to_idx[f"Intervention:{i}"] for i in pool_ids],
                            dtype=torch.long)
        with torch.no_grad():
            scores = model.score_all_interventions(z, ctx, cand)[0].numpy()
        l2_ids = [pool_ids[i] for i in np.argsort(-scores)]

        # ── Content-based baseline (unchanged) ────────────────────────────────
        bs = sorted(((inv["id"], content_based_score(inv, row["tier"],
                                                     episode["trigger_type"]))
                     for inv in INTERVENTION_MAP.values()), key=lambda x: -x[1])
        cb_ids = [i for i, _ in bs[:max(K_VALUES)]]

        rec = {"session_id": row["session_id"]}
        for k in K_VALUES:
            for name, ids in (("cb", cb_ids), ("l1", l1_ids), ("l2", l2_ids)):
                rec[f"{name}_prec@{k}"] = precision_at_k(ids, relevant_ids, k)
                rec[f"{name}_ndcg@{k}"] = ndcg_at_k(ids, rating_map, k)
                rec[f"{name}_hit@{k}"] = 1 if gt_inv in ids[:k] else 0
        for name, ids in (("cb", cb_ids), ("l1", l1_ids), ("l2", l2_ids)):
            rec[f"{name}_mor"] = rating_map.get(ids[0], gt_rating) if ids else 0
        rows.append(rec)

    res = pd.DataFrame(rows)
    pool_sizes = np.array(pool_sizes)

    print(f"\n  Test sessions evaluated  : {len(res)}")
    print(f"  Suppressed (poor signal) : {suppressed}")
    print(f"  Shared candidate pool    : mean {pool_sizes.mean():.1f} "
          f"(min {pool_sizes.min()}, max {pool_sizes.max()})")
    print(f"  Pools <= {max(K_VALUES)} (top-{max(K_VALUES)} set fixed) : "
          f"{int((pool_sizes <= max(K_VALUES)).sum())} / {len(res)} sessions")

    # ── Parity gate ───────────────────────────────────────────────────────────
    if not verify_layer1_parity(res, suppressed):
        raise RuntimeError(
            "Layer 1 parity check FAILED. The split / ground-truth / episode logic "
            "reproduced in this file no longer matches layer1_evaluation.py, so the "
            "Layer 1 vs Layer 2 comparison would be invalid. If you regenerated the "
            "dataset, re-run src/layer1_evaluation.py and update REFERENCE_LAYER1 "
            "with its new numbers. Otherwise the two files have genuinely diverged."
        )

    # ── Results ───────────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print("  PROGRESSIVE IMPROVEMENT  (all three ranked on identical sessions)")
    print("-" * 72)
    print(f"  {'Metric':<18}{'Baseline':<12}{'Layer 1':<12}{'Layer 2':<12}{'L2 - L1'}")
    print(f"  {'-'*18}{'-'*12}{'-'*12}{'-'*12}{'-'*10}")

    metrics = []
    for k in K_VALUES:
        for m in ("prec", "ndcg", "hit"):
            label = {"prec": f"Precision@{k}", "ndcg": f"NDCG@{k}",
                     "hit": f"Hit Rate@{k}"}[m]
            cb, l1, l2 = res[f"cb_{m}@{k}"], res[f"l1_{m}@{k}"], res[f"l2_{m}@{k}"]
            print(f"  {label:<18}{cb.mean():<12.4f}{l1.mean():<12.4f}"
                  f"{l2.mean():<12.4f}{l2.mean()-l1.mean():+.4f}")
            metrics.append((label, l1.values, l2.values))
    cb, l1, l2 = res["cb_mor"], res["l1_mor"], res["l2_mor"]
    print(f"  {'Mean Outcome':<18}{cb.mean():<12.4f}{l1.mean():<12.4f}"
          f"{l2.mean():<12.4f}{l2.mean()-l1.mean():+.4f}")
    metrics.append(("Mean Outcome", l1.values, l2.values))

    # ── Significance ──────────────────────────────────────────────────────────
    print("\n" + "-" * 72)
    print(f"  PAIRED BOOTSTRAP 95% CI ON (LAYER 2 - LAYER 1), {N_BOOTSTRAP} resamples")
    print("-" * 72)
    print(f"  {'Metric':<18}{'Difference':<14}{'95% CI':<24}{'Significant?'}")
    print(f"  {'-'*18}{'-'*14}{'-'*24}{'-'*12}")

    n_better, n_worse, summary = 0, 0, []
    for label, a1, a2 in metrics:
        d, lo, hi = paired_bootstrap_ci(a2, a1)
        if lo > 0:
            verdict, n_better = "yes (better)", n_better + 1
        elif hi < 0:
            verdict, n_worse = "yes (WORSE)", n_worse + 1
        else:
            verdict = "no"
        print(f"  {label:<18}{d:<+14.4f}[{lo:+.4f}, {hi:+.4f}]{'':<4}{verdict}")
        summary.append({"metric": label, "difference": d, "ci_low": lo,
                        "ci_high": hi, "significant": verdict})

    # ── Interpretation ────────────────────────────────────────────────────────
    print("\n" + "=" * 72)
    print("  INTERPRETATION")
    print("=" * 72)
    n_total = len(metrics)
    n_up = sum(1 for _, a1, a2 in metrics if a2.mean() > a1.mean())
    print(f"\n  Layer 2 improves on Layer 1 in {n_up} of {n_total} metrics by point estimate.")
    print(f"  Statistically distinguishable at 95%: {n_better} better, "
          f"{n_worse} worse, {n_total - n_better - n_worse} inconclusive.")

    if n_better == 0:
        print("\n  No metric reaches significance. At ~112 test sessions this is a")
        print("  plausible outcome even for a genuinely better model — report point")
        print("  estimates WITH these intervals rather than claiming a win.")
    if n_worse > 0:
        print("\n  WARNING: Layer 2 is significantly WORSE on at least one metric.")
        print("  Do not present the progressive-improvement narrative until this is")
        print("  understood.")

    print("\n  Caveat on Mean Outcome Rating: layer1_evaluation.py falls back to the")
    print("  DELIVERED intervention's rating when the top-1 recommendation is absent")
    print("  from the user's history, so recommending something untried scores as")
    print("  whatever was actually delivered. Treat it as indicative only; Precision@K")
    print("  and NDCG@K are the defensible metrics.")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    pd.DataFrame(summary).to_csv(RESULTS_PATH, index=False)
    res.to_csv(RESULTS_PATH.replace(".csv", "_per_session.csv"), index=False)
    print(f"\n  Saved -> {RESULTS_PATH}")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    run_evaluation()