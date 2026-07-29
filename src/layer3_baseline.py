"""
MindPulse — Module 3  |  Sub-layer 3.1: Recency-Weighted Personalisation
========================================================================
The no-training personalisation baseline (spec Section 2.3, Sub-layer 3.1).
Re-ranks Layer 2's output by blending in a recency-weighted history signal:

    layer3_score = layer2_score + alpha * personalisation_score

`personalisation_score` comes from `user_history.UserHistory` — a leak-safe,
shrunk, recency-weighted residual (see that module for the reasoning).

WHY THIS FILE EXISTS BEFORE THE MLP
-----------------------------------
This is the honest baseline for Sub-layers 3.2 and 3.3. If a trained MLP
cannot beat a weighted average of the user's own history, the MLP is not
doing personalisation — it is doing overfitting. Layer 2 taught this the hard
way: the graph turned out to contribute +0.011 AUC over a no-graph model, and
that only became visible because the ablation was run. The same discipline
applies here, and running the cheap baseline first is how it stays visible.

EVALUATION — WHY IPS IS PRIMARY (see Phase 0 findings)
------------------------------------------------------
Every conventional metric fails on this dataset, each differently:

  * Precision@K / NDCG@K are CIRCULAR for Layer 3. Relevance is defined from
    the user's own training ratings, so a trivial ranker that sorts by those
    same ratings scores 0.97 / 0.92. Any history-based model climbs toward
    that number without learning anything.
  * Hit Rate@K PENALISES personalisation. `dataset_generator.py` selects the
    delivered intervention with `random.choice(feasible)` — uniformly at
    random, independent of user preference. An oracle with perfect knowledge
    of the true latent preference makes Hit Rate monotonically WORSE
    (0.3125 -> 0.2946 -> 0.2857 -> 0.2679 at boosts 0.5/1.0/2.0).
  * Mean Outcome Rating falls back to the delivered intervention's rating
    when the top pick is absent from the user's history.

That same uniform logging policy is, however, the ideal condition for
UNBIASED OFF-POLICY EVALUATION. With propensity mu(a|x) = 1/|feasible|, the
inverse-propensity estimator of the expected outcome under a stochastic
top-K policy is unbiased:

    E[rating under pi] ~= mean_i [ 1{delivered_i in topK_i} * (n_i / K) * rating_i ]

It answers the question that actually matters — would this recommender have
produced better outcomes? — using observed test ratings rather than
training-derived relevance, so it is not circular. Its weakness is variance:
at 112 test sessions the confidence interval is roughly +/- 0.7 rating points.
This file reports that interval rather than hiding it.

TUNING
------
alpha, half-life and shrinkage are selected on the TRAINING sessions only,
by IPS, using history that is itself restricted to earlier training sessions.
The test set is not touched until the final evaluation.

Usage
-----
    python src/layer3_baseline.py

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import numpy as np
import pandas as pd
import torch

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import build_knowledge_graph, INTERVENTION_MAP
from graph_converter import convert_kg_to_pyg
from rule_engine import Layer1RuleEngine
from gcn_model import Layer2GCN
from training_pairs import get_context_node_indices
from user_history import UserHistory
from layer1_evaluation import DATASET_PATH, precision_at_k, ndcg_at_k
from layer2_evaluation import make_test_mask, build_ground_truth, build_episode

MODEL_PATH   = os.path.join("models", "gcn_layer2.pt")
RESULTS_PATH = os.path.join("data", "synthetic", "layer3_baseline_results.csv")
FULL_POOL_K  = 22
IPS_K        = 5
N_BOOTSTRAP  = 4000
SEED         = 42

ALPHA_GRID     = [0.0, 0.5, 1.0, 2.0, 4.0]
HALF_LIFE_GRID = [15.0, 30.0, 90.0]
SHRINK_GRID    = [1.0, 3.0]


# ══════════════════════════════════════════════════════════════════════════════
# SHARED MACHINERY
# ══════════════════════════════════════════════════════════════════════════════

def load_stack():
    """Load KG, Layer 1 engine, and the trained Layer 2 model with its embeddings."""
    G = build_knowledge_graph()
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)
    engine = Layer1RuleEngine(G, top_k=FULL_POOL_K)
    ckpt = torch.load(MODEL_PATH, weights_only=False)
    model = Layer2GCN(**ckpt["config"])
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    with torch.no_grad():
        z = model.encode(data.x, data.edge_index, data.edge_weight)
    return engine, model, z, node_to_idx


def build_session_records(df_rows, engine, model, z, node_to_idx):
    """
    For each session: Layer 1's feasible pool and Layer 2's score per candidate.

    Computed once and reused across every hyperparameter setting — the pool and
    the GCN scores do not depend on alpha, half-life or shrinkage, so
    recomputing them inside the sweep would be pure waste.
    """
    out = []
    for _, row in df_rows.iterrows():
        res = engine.recommend(build_episode(row))
        if not res["ranked_candidates"]:
            continue
        pool = [c["intervention_id"] for c in res["ranked_candidates"]]
        ctx = torch.tensor([get_context_node_indices(row, node_to_idx)], dtype=torch.long)
        cand = torch.tensor([node_to_idx[f"Intervention:{i}"] for i in pool], dtype=torch.long)
        with torch.no_grad():
            l2 = model.score_all_interventions(z, ctx, cand)[0].numpy()
        out.append({"row": row, "pool": pool, "l2": l2,
                    "gt": row["intervention_id"], "rating": float(row["outcome_rating"]),
                    "user": row["user_id"], "ts": row["timestamp"]})
    return out


def rerank(rec, history: UserHistory, alpha: float):
    """Blend Layer 2's score with the personalisation signal."""
    if alpha == 0.0:
        p = np.zeros(len(rec["pool"]))
    else:
        p = history.score_many(rec["user"], rec["pool"], rec["ts"])
    order = np.argsort(-(rec["l2"] + alpha * p))
    return [rec["pool"][i] for i in order]


def ips_estimate(records, rankings, k=IPS_K):
    """
    Unbiased estimate of E[outcome_rating] under a uniform-over-top-K policy.

    Valid because the logging policy is uniform over the feasible set
    (`random.choice(feasible)` in dataset_generator.py). Returns the
    per-session contribution vector so the caller can bootstrap it.
    """
    v = np.array([
        (len(r["pool"]) / k) * r["rating"] if r["gt"] in rank[:k] else 0.0
        for r, rank in zip(records, rankings)
    ])
    matched = int(sum(1 for r, rank in zip(records, rankings) if r["gt"] in rank[:k]))
    return v, matched


def bootstrap_ci(v, n_boot=N_BOOTSTRAP, seed=SEED):
    rng = np.random.default_rng(seed)
    bs = v[rng.integers(0, len(v), (n_boot, len(v)))].mean(axis=1)
    return float(v.mean()), float(np.percentile(bs, 2.5)), float(np.percentile(bs, 97.5))


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 74)
    print("  MODULE 3 — SUB-LAYER 3.1: RECENCY-WEIGHTED PERSONALISATION")
    print("=" * 74)

    df = pd.read_csv(DATASET_PATH)
    tm = make_test_mask(df)
    df_train, df_test = df[~tm].copy(), df[tm].copy()
    engine, model, z, node_to_idx = load_stack()

    print(f"\n  Train sessions {len(df_train)} | Test sessions {len(df_test)} (untouched during tuning)")

    train_recs = build_session_records(df_train, engine, model, z, node_to_idx)
    test_recs = build_session_records(df_test, engine, model, z, node_to_idx)
    print(f"  Usable after Layer 1 suppression: train {len(train_recs)}, test {len(test_recs)}")

    # ── Tuning on TRAINING only ───────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("  HYPERPARAMETER SEARCH  (IPS on TRAINING sessions — test never used)")
    print("-" * 74)
    print(f"  {'alpha':<8}{'half-life':<12}{'shrink':<9}{'IPS E[rating]':<16}{'matched'}")

    results = []
    for hl in HALF_LIFE_GRID:
        for kk in SHRINK_GRID:
            H = UserHistory(df_train, half_life_days=hl, shrinkage_k=kk)
            for a in ALPHA_GRID:
                rk = [rerank(r, H, a) for r in train_recs]
                v, matched = ips_estimate(train_recs, rk)
                results.append({"alpha": a, "half_life": hl, "shrink": kk,
                                "ips": float(v.mean()), "matched": matched})
                if a in (0.0, 1.0, 4.0):
                    print(f"  {a:<8}{hl:<12}{kk:<9}{v.mean():<16.4f}{matched}")

    R = pd.DataFrame(results)
    best = R.loc[R["ips"].idxmax()]
    base = R[(R.alpha == 0.0) & (R.half_life == HALF_LIFE_GRID[0]) &
             (R.shrink == SHRINK_GRID[0])]["ips"].iloc[0]
    print(f"\n  alpha = 0 (pure Layer 2, no personalisation) : {base:.4f}")
    print(f"  best on training : alpha={best['alpha']}, half-life={best['half_life']}, "
          f"shrink={best['shrink']}  -> {best['ips']:.4f}")
    print(f"  training-side gain from personalisation      : {best['ips'] - base:+.4f}")

    # ── Final evaluation on TEST ──────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("  TEST-SET EVALUATION  (single evaluation, tuned settings)")
    print("-" * 74)

    H = UserHistory(df_train, half_life_days=float(best["half_life"]),
                    shrinkage_k=float(best["shrink"]))
    rank_l2 = [rerank(r, H, 0.0) for r in test_recs]
    rank_l3 = [rerank(r, H, float(best["alpha"])) for r in test_recs]

    v2, m2 = ips_estimate(test_recs, rank_l2)
    v3, m3 = ips_estimate(test_recs, rank_l3)
    e2, lo2, hi2 = bootstrap_ci(v2)
    e3, lo3, hi3 = bootstrap_ci(v3)
    d, dlo, dhi = bootstrap_ci(v3 - v2)

    print(f"\n  PRIMARY — IPS off-policy estimate of E[outcome_rating], K={IPS_K}")
    print(f"    {'policy':<34}{'estimate':<12}{'95% CI':<22}{'matched'}")
    print(f"    {'Layer 2 (no personalisation)':<34}{e2:<12.4f}[{lo2:.3f}, {hi2:.3f}]{'':<6}{m2}")
    print(f"    {'Layer 3.1 (recency-weighted)':<34}{e3:<12.4f}[{lo3:.3f}, {hi3:.3f}]{'':<6}{m3}")
    print(f"    {'difference (3.1 - L2)':<34}{d:<+12.4f}[{dlo:+.3f}, {dhi:+.3f}]")
    verdict = "better" if dlo > 0 else ("WORSE" if dhi < 0 else "not distinguishable")
    print(f"    verdict: {verdict}")

    # ── Contaminated metrics, reported with their ceiling ─────────────────────
    ur, urat = build_ground_truth(df, tm)
    def trad(rankings):
        acc = {f"{m}@{k}": [] for k in (3, 5) for m in ("prec", "ndcg", "hit")}
        for r, rank in zip(test_recs, rankings):
            uid, gt = r["user"], r["gt"]
            rel = ur.get(uid, {gt}) or {gt}
            rm = dict(urat.get(uid, {})); rm.setdefault(gt, r["rating"])
            for k in (3, 5):
                acc[f"prec@{k}"].append(precision_at_k(rank, rel, k))
                acc[f"ndcg@{k}"].append(ndcg_at_k(rank, rm, k))
                acc[f"hit@{k}"].append(1 if gt in rank[:k] else 0)
        return {k: float(np.mean(v)) for k, v in acc.items()}

    oracle_rank = [sorted(r["pool"], key=lambda i: -urat.get(r["user"], {}).get(i, 0))
                   for r in test_recs]
    t2, t3, to = trad(rank_l2), trad(rank_l3), trad(oracle_rank)

    print("\n  SECONDARY — conventional metrics (see caveats below)")
    print(f"    {'metric':<12}{'Layer 2':<11}{'Layer 3.1':<12}{'contamination ceiling'}")
    for k in ("prec@3", "ndcg@3", "hit@3", "prec@5", "ndcg@5", "hit@5"):
        print(f"    {k:<12}{t2[k]:<11.4f}{t3[k]:<12.4f}{to[k]:.4f}")

    print("\n  Caveats — do not report these as headline results:")
    print("    * Precision@K / NDCG@K are CIRCULAR for Layer 3: relevance comes from")
    print("      the user's own training ratings, the same source as the personalisation")
    print("      signal. The ceiling column shows what a trivial ranker on that quantity")
    print("      achieves. Movement toward it is contamination, not learning.")
    print("    * Hit Rate@K PENALISES personalisation: the delivered intervention was")
    print("      chosen uniformly at random from the feasible set, so the Hit-Rate-optimal")
    print("      policy is not to personalise at all.")

    pd.DataFrame([{"policy": "layer2", "ips": e2, "ci_low": lo2, "ci_high": hi2, **t2},
                  {"policy": "layer3.1", "ips": e3, "ci_low": lo3, "ci_high": hi3, **t3}]
                 ).to_csv(RESULTS_PATH, index=False)
    print(f"\n  Saved -> {RESULTS_PATH}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()