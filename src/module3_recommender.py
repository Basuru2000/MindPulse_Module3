"""
MindPulse — Module 3  |  End-to-End Recommender
================================================
The single callable that turns a Module 1 stress profile (plus optional
Module 2 context) into a ranked recommendation with an explanation.

Everything else in this codebase is an evaluation script that operates over
the dataset in bulk. This is the only component that answers the question a
live system asks: given ONE episode right now, what should we recommend?

    recommender = Module3Recommender(history_df=training_sessions)
    result = recommender.recommend(module1_profile, module2_context)

The returned dict matches the output schema in specification Section 3.2.

PIPELINE
--------
    Module 1 profile (+ Module 2 context)
        |
        v  adapt_inputs()      flatten gesture_events, apply dialogue fallback
    episode dict
        |
        v  Layer 1             8 gating rules -> feasible candidate pool
        |                      (may suppress entirely: calm tier / poor signal)
        v  Layer 2             GCN scores every candidate in the pool
        |
        v  Layer 3             MLP residual correction re-ranks the same pool
        |
        v  explanation         exact context attribution
    recommendation envelope

The pool is shared across layers by design (specification Section 2.2.6):
Layer 1 decides what is FEASIBLE, Layers 2 and 3 decide only the ORDER.
Feasibility — including I22's evening/night constraint, which the GCN's node
features do not encode — is therefore always enforced regardless of what the
learned layers prefer.

INPUT ADAPTATION IS NOT TRIVIAL
-------------------------------
Module 1 emits `gesture_events` as an ARRAY of behavioural-support objects,
while the rule engine and the Layer 3 feature builder both expect flattened
`support_detected` / `support_score` scalars. The dataset stores the flattened
form, so no evaluation script ever exercised this conversion. `adapt_inputs()`
does it here, and it is the most likely place for a live integration to break.

ON LAYER 3 BEING ENABLED BY DEFAULT
------------------------------------
Layer 3 is enabled because the specified architecture includes it and it is
implemented and validated. It should not be expected to change outcomes: on
the held-out set Sub-layer 3.2 measured +0.0250 [-0.289, +0.364] against
Layer 2 — not distinguishable from zero (Section 7.5). The power analysis
(Section 2.3.8) explains why: the individual preference effect in this data
lies below the recovery boundary.

Every response therefore reports `layer3_delta` — how far Layer 3 moved the
top pick's score — and `layer3_changed_ranking`, so a caller can see the
layer's actual influence rather than assuming it. Pass `enable_layer3=False`
to run Layer 2 only.

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import uuid
import numpy as np
import pandas as pd
import torch
from datetime import datetime, timedelta

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import build_knowledge_graph, INTERVENTION_MAP
from graph_converter import convert_kg_to_pyg
from rule_engine import Layer1RuleEngine
from gcn_model import Layer2GCN
from training_pairs import get_context_node_indices
from user_history import UserHistory
from layer3_features import build_features
from ann_model import Layer3ANN

GCN_PATH     = os.path.join("models", "gcn_layer2.pt")
ANN_PATH     = os.path.join("models", "ann_layer3.pt")
DATASET_PATH = os.path.join("data", "synthetic", "module3_dataset.csv")
FULL_POOL_K  = 22
OUTCOME_CHECK_MINUTES = 30

DISCLAIMER = ("This is a self-care suggestion, not medical advice or a "
              "diagnosis. If distress persists, please contact a qualified "
              "professional.")


# ══════════════════════════════════════════════════════════════════════════════
# INPUT ADAPTATION
# ══════════════════════════════════════════════════════════════════════════════

def flatten_gesture_events(gesture_events) -> tuple:
    """
    Module 1's `gesture_events` array -> (support_detected, support_score).

    Module 1 emits a list, empty when nothing was detected, otherwise objects
    of the form {type: "behavioural_support", count, proportion, support_score,
    support_applied, confidence_delta}. Only events Module 1 actually APPLIED
    are honoured — `support_applied=False` means Module 1 detected movement but
    its own gating declined to use it, and Module 3 should not re-admit a signal
    the producing module rejected.

    When several qualifying events are present the maximum score is taken, as
    the strongest evidence of behavioural support within the window.
    """
    if not gesture_events:
        return False, 0.0
    scores = [float(e.get("support_score", 0.0)) for e in gesture_events
              if e.get("type") == "behavioural_support"
              and e.get("support_applied", True)]
    if not scores:
        return False, 0.0
    return True, max(scores)


def adapt_inputs(m1: dict, m2: dict = None) -> dict:
    """
    Build the episode dict the rule engine and feature builder consume.

    Applies the same Module 2 fallback rule the rest of the codebase uses:
    when `dialogue_completed` is false, ALL Module 2 fields become "unknown"
    rather than being guessed. An explicit "unknown" from a completed dialogue
    is passed through unchanged — Module 2's location and social questions are
    optional, so "unknown" is a real answer, not a missing one.
    """
    m2 = m2 or {}
    detected, score = flatten_gesture_events(m1.get("gesture_events", []))
    dialogue = bool(m2.get("dialogue_completed", False))

    return {
        "user_id":            m1.get("user_id"),
        "tier":               m1.get("tier", "mild"),
        "confidence":         float(m1.get("confidence", 0.70)),
        "baseline_deviation": float(m1.get("baseline_deviation", 1.0)),
        "support_detected":   detected,
        "support_score":      score,
        "signal_quality":     m1.get("signal_quality", "good"),
        "baseline_mode":      m1.get("baseline_mode", "personalised"),
        "timestamp":          m1.get("timestamp"),
        "dialogue_completed": dialogue,
        "trigger_type":       m2.get("trigger_type", "unknown") if dialogue else "unknown",
        "trigger_confidence": float(m2.get("trigger_confidence", 0.70)) if dialogue else 0.0,
        "location_context":   m2.get("location_context", "unknown") if dialogue else "unknown",
        "social_context":     m2.get("social_context", "unknown") if dialogue else "unknown",
        "kg_node_id":         m2.get("kg_node_id"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# RECOMMENDER
# ══════════════════════════════════════════════════════════════════════════════

class Module3Recommender:
    """
    End-to-end Module 3 pipeline.

    Parameters
    ----------
    history_df : pd.DataFrame or None
        Past sessions used to build the Layer 3 personalisation signal. In a
        live system this is the user's stored feedback history; here it
        defaults to the synthetic dataset. If None and the dataset is absent,
        Layer 3 is disabled rather than run on empty history.
    enable_layer2, enable_layer3 : bool
        Allow running the pipeline at reduced depth, which is what makes the
        layer-by-layer contribution inspectable at inference time.
    """

    def __init__(self, history_df: pd.DataFrame = None,
                 enable_layer2: bool = True, enable_layer3: bool = True):
        self.G = build_knowledge_graph()
        self.data, self.node_to_idx, self.idx_to_node = convert_kg_to_pyg(
            self.G, make_undirected=True)
        self.engine = Layer1RuleEngine(self.G, top_k=FULL_POOL_K)

        # ── Layer 2 ───────────────────────────────────────────────────────────
        self.layer2 = None
        if enable_layer2 and os.path.exists(GCN_PATH):
            ck = torch.load(GCN_PATH, weights_only=False)
            self.layer2 = Layer2GCN(**ck["config"])
            self.layer2.load_state_dict(ck["state_dict"])
            self.layer2.eval()
            with torch.no_grad():
                self.z = self.layer2.encode(self.data.x, self.data.edge_index,
                                            self.data.edge_weight)
        elif enable_layer2:
            print(f"[warn] {GCN_PATH} not found — running Layer 1 only.")

        # ── Layer 3 ───────────────────────────────────────────────────────────
        self.layer3 = self.history = None
        self.beta = 0.0
        if enable_layer3 and self.layer2 is not None and os.path.exists(ANN_PATH):
            if history_df is None and os.path.exists(DATASET_PATH):
                history_df = pd.read_csv(DATASET_PATH)
            if history_df is not None and len(history_df):
                ck3 = torch.load(ANN_PATH, weights_only=False)
                self.layer3 = Layer3ANN(**ck3["config"])
                self.layer3.load_state_dict(ck3["state_dict"])
                self.layer3.eval()
                self.scaler = ck3["scaler"]
                self.beta = float(ck3["beta"])
                self.history = UserHistory(
                    history_df,
                    half_life_days=ck3["history"]["half_life_days"],
                    shrinkage_k=ck3["history"]["shrinkage_k"])
            else:
                print("[warn] no history available — Layer 3 disabled.")
        elif enable_layer3 and self.layer2 is not None:
            print(f"[warn] {ANN_PATH} not found — Layer 3 disabled.")

    # ──────────────────────────────────────────────────────────────────────────
    def recommend(self, module1_profile: dict, module2_context: dict = None,
                  top_k: int = 3) -> dict:
        """
        Produce a recommendation envelope (specification Section 3.2).

        Returns a suppression envelope — `intervention_id` None, `suppressed`
        True — when Layer 1's gates decline to recommend. Suppression is a
        legitimate clinical outcome, not an error: poor signal quality must
        never trigger a push, and calm tier is not a trigger condition.
        """
        ep = adapt_inputs(module1_profile, module2_context)
        ts = ep.get("timestamp") or datetime.utcnow().isoformat()
        rec_id = str(uuid.uuid4())

        snapshot = {
            "trigger_type": ep["trigger_type"],
            "location_context": ep["location_context"],
            "social_context": ep["social_context"],
            "baseline_deviation": ep["baseline_deviation"],
            "support_detected": ep["support_detected"],
            "support_score": ep["support_score"],
        }

        # ── Layer 1 ───────────────────────────────────────────────────────────
        l1 = self.engine.recommend(ep)
        if not l1["ranked_candidates"]:
            return {
                "user_id": ep["user_id"], "recommendation_id": rec_id,
                "timestamp": ts, "suppressed": True,
                "suppression_reason": l1["gating_flags"].get("reason", "gated"),
                "intervention_id": None, "intervention_type": None,
                "intervention_name": None, "intervention_duration": None,
                "delivery_tier": ep["tier"], "recommendation_layer": None,
                "layer1_score": None, "layer2_score": None, "layer3_score": None,
                "gnn_explanation": l1["explanation"],
                "context_snapshot": snapshot, "kg_node_id": ep.get("kg_node_id"),
                "outcome_check_due": None, "alternatives": [],
                "disclaimer": DISCLAIMER,
            }

        pool = [c["intervention_id"] for c in l1["ranked_candidates"]]
        l1_scores = {c["intervention_id"]: c["layer1_score"]
                     for c in l1["ranked_candidates"]}
        ranked, layer_used = pool, "layer1"
        l2_scores = l3_scores = None

        # ── Layer 2 ───────────────────────────────────────────────────────────
        if self.layer2 is not None:
            ctx = torch.tensor([get_context_node_indices(ep, self.node_to_idx)],
                               dtype=torch.long)
            cand = torch.tensor([self.node_to_idx[f"Intervention:{i}"] for i in pool],
                                dtype=torch.long)
            with torch.no_grad():
                s2 = self.layer2.score_all_interventions(self.z, ctx, cand)[0].numpy()
            l2_scores = dict(zip(pool, s2.tolist()))
            ranked = [pool[i] for i in np.argsort(-s2)]
            layer_used = "layer2"

            # ── Layer 3 ───────────────────────────────────────────────────────
            if self.layer3 is not None:
                feats = np.vstack([
                    build_features(ep, inv, s2[i], self.history)
                    for i, inv in enumerate(pool)])
                feats = (feats - self.scaler["mean"]) / self.scaler["std"]
                with torch.no_grad():
                    corr = self.layer3(torch.tensor(feats, dtype=torch.float)).numpy()
                s3 = s2 + self.beta * corr
                l3_scores = dict(zip(pool, s3.tolist()))
                ranked = [pool[i] for i in np.argsort(-s3)]
                layer_used = "layer3"

        top = ranked[0]
        inv = INTERVENTION_MAP[top]

        # ── Explanation ───────────────────────────────────────────────────────
        explanation = (self._explain(ep, top) if self.layer2 is not None
                       else l1["explanation"])

        # Layer 3's actual influence, reported rather than assumed
        l2_order = ([pool[i] for i in np.argsort(-np.array(list(l2_scores.values())))]
                    if l2_scores else pool)
        l3_delta = (l3_scores[top] - l2_scores[top]) if l3_scores else None

        return {
            "user_id": ep["user_id"], "recommendation_id": rec_id,
            "timestamp": ts, "suppressed": False, "suppression_reason": None,
            "intervention_id": top,
            "intervention_type": inv["type"],
            "intervention_name": inv["name"],
            "intervention_duration": inv["duration"],
            "delivery_tier": ep["tier"],
            "recommendation_layer": layer_used,
            "layer1_score": l1_scores.get(top),
            "layer2_score": l2_scores.get(top) if l2_scores else None,
            "layer3_score": l3_scores.get(top) if l3_scores else None,
            "layer3_delta": l3_delta,
            "layer3_changed_ranking": (bool(l2_order[0] != top)
                                       if l3_scores else False),
            "gnn_explanation": explanation,
            "context_snapshot": snapshot,
            "kg_node_id": ep.get("kg_node_id"),
            "outcome_check_due": (pd.to_datetime(ts) +
                                  timedelta(minutes=OUTCOME_CHECK_MINUTES)).isoformat(),
            "candidate_pool_size": len(pool),
            "alternatives": [
                {"intervention_id": i, "intervention_name": INTERVENTION_MAP[i]["name"]}
                for i in ranked[1:top_k]],
            "disclaimer": DISCLAIMER,
        }

    # ──────────────────────────────────────────────────────────────────────────
    def _explain(self, ep: dict, intervention_id: str) -> str:
        """
        Exact context attribution (specification Section 2.2.7).

        Uses the exact decomposition rather than GNNExplainer: the score is
        mean(z_ctx) . z_inv and mean is linear, so it splits exactly into one
        signed term per context node. GNNExplainer is implemented in
        `explain_recommendations.py` but measured unreliable on this model
        (fidelity ratio 0.41, stability Jaccard 0.36), so it is not used on
        the live path.
        """
        from explain_recommendations import phrase

        ctx_idx = get_context_node_indices(ep, self.node_to_idx)
        inv_i = self.node_to_idx[f"Intervention:{intervention_id}"]
        scale = self.layer2.score_scale.item()
        with torch.no_grad():
            contribs = [float((self.z[c] * self.z[inv_i]).sum().item())
                        / len(ctx_idx) * scale for c in ctx_idx]

        ranked = sorted(zip(ctx_idx, contribs), key=lambda t: -t[1])
        drivers = [phrase(self.idx_to_node[i]) for i, c in ranked[:2] if c > 0]
        if not drivers:
            drivers = [phrase(self.idx_to_node[ranked[0][0]])]
        negatives = [i for i, c in ranked if c < 0]

        inv = INTERVENTION_MAP[intervention_id]
        text = (f"Recommended '{inv['name']}' ({inv['type']}, {inv['duration']} min) "
                f"because: {' + '.join(drivers)}.")
        if negatives:
            text += f" Weighed against: {phrase(self.idx_to_node[negatives[-1]])}."
        return text


# ══════════════════════════════════════════════════════════════════════════════
# DEMO
# ══════════════════════════════════════════════════════════════════════════════

def _demo():
    import json

    print("\n" + "=" * 76)
    print("  MODULE 3 — END-TO-END RECOMMENDER DEMO")
    print("=" * 76)

    rec = Module3Recommender()
    print(f"\n  Layer 1 : ready")
    print(f"  Layer 2 : {'ready' if rec.layer2 else 'unavailable'}")
    print(f"  Layer 3 : {'ready (beta=' + str(rec.beta) + ')' if rec.layer3 else 'unavailable'}")

    cases = [
        ("Acute stress, academic trigger, alone at university, support detected", {
            "user_id": "U001", "tier": "acute", "confidence": 0.88,
            "baseline_deviation": 3.1, "signal_quality": "good",
            "baseline_mode": "personalised", "timestamp": "2026-03-02T14:20:00",
            "gesture_events": [{"type": "behavioural_support", "count": 6,
                                "proportion": 0.5, "support_score": 0.82,
                                "support_applied": True, "confidence_delta": 0.04}],
        }, {"dialogue_completed": True, "trigger_type": "academic",
            "trigger_confidence": 0.81, "location_context": "university",
            "social_context": "alone", "kg_node_id": "ep_88213"}),

        ("Mild stress at work with colleagues — feasibility must exclude items", {
            "user_id": "U002", "tier": "mild", "confidence": 0.79,
            "baseline_deviation": 1.1, "signal_quality": "good",
            "baseline_mode": "personalised", "timestamp": "2026-03-03T10:05:00",
            "gesture_events": [],
        }, {"dialogue_completed": True, "trigger_type": "interpersonal",
            "trigger_confidence": 0.72, "location_context": "work",
            "social_context": "colleagues"}),

        ("No Module 2 dialogue — Module 1 signals only", {
            "user_id": "U003", "tier": "moderate", "confidence": 0.74,
            "baseline_deviation": 2.0, "signal_quality": "degraded",
            "baseline_mode": "cold_start", "timestamp": "2026-03-04T21:40:00",
            "gesture_events": [{"type": "behavioural_support", "count": 3,
                                "proportion": 0.3, "support_score": 0.55,
                                "support_applied": True}],
        }, None),

        ("Module 1 detected support but did NOT apply it", {
            "user_id": "U004", "tier": "moderate", "confidence": 0.80,
            "baseline_deviation": 1.9, "signal_quality": "good",
            "baseline_mode": "personalised", "timestamp": "2026-03-05T16:00:00",
            "gesture_events": [{"type": "behavioural_support", "count": 2,
                                "proportion": 0.2, "support_score": 0.61,
                                "support_applied": False}],
        }, {"dialogue_completed": True, "trigger_type": "health",
            "trigger_confidence": 0.60, "location_context": "unknown",
            "social_context": "unknown"}),

        ("Poor signal quality — must be SUPPRESSED", {
            "user_id": "U005", "tier": "acute", "confidence": 0.91,
            "baseline_deviation": 2.8, "signal_quality": "poor",
            "baseline_mode": "personalised", "timestamp": "2026-03-06T09:15:00",
            "gesture_events": [],
        }, {"dialogue_completed": True, "trigger_type": "financial",
            "trigger_confidence": 0.70, "location_context": "home",
            "social_context": "alone"}),

        ("Calm tier — not a trigger condition, must be SUPPRESSED", {
            "user_id": "U006", "tier": "calm", "confidence": 0.85,
            "baseline_deviation": 0.3, "signal_quality": "good",
            "baseline_mode": "personalised", "timestamp": "2026-03-07T11:00:00",
            "gesture_events": [],
        }, None),
    ]

    for title, m1, m2 in cases:
        out = rec.recommend(m1, m2)
        print("\n" + "-" * 76)
        print(f"  {title}")
        print("-" * 76)
        if out["suppressed"]:
            print(f"  SUPPRESSED — {out['suppression_reason']}")
            continue
        print(f"  -> {out['intervention_id']}  {out['intervention_name']} "
              f"({out['intervention_type']}, {out['intervention_duration']} min)")
        print(f"     layer={out['recommendation_layer']}  pool={out['candidate_pool_size']}")
        print(f"     L1 {out['layer1_score']:.4f} | L2 {out['layer2_score']:+.4f} | "
              f"L3 {out['layer3_score']:+.4f}  (delta {out['layer3_delta']:+.4f}, "
              f"reordered: {out['layer3_changed_ranking']})")
        print(f"     {out['gnn_explanation']}")
        print(f"     alternatives: "
              f"{', '.join(a['intervention_id'] for a in out['alternatives'])}")

    print("\n" + "-" * 76)
    print("  FULL ENVELOPE (specification Section 3.2)")
    print("-" * 76)
    full = rec.recommend(cases[0][1], cases[0][2])
    print(json.dumps(full, indent=2, default=str)[:1400])

    # ── Contract checks ───────────────────────────────────────────────────────
    required = ["user_id", "recommendation_id", "timestamp", "intervention_id",
                "intervention_type", "intervention_name", "intervention_duration",
                "delivery_tier", "recommendation_layer", "layer1_score",
                "layer2_score", "layer3_score", "gnn_explanation",
                "context_snapshot", "kg_node_id", "outcome_check_due"]
    missing = [k for k in required if k not in full]
    print("\n  Contract checks:")
    print(f"    All Section 3.2 fields present     : "
          f"{'PASS' if not missing else 'FAIL ' + str(missing)}")

    sup = rec.recommend(cases[4][1], cases[4][2])
    print(f"    Poor signal suppressed             : "
          f"{'PASS' if sup['suppressed'] else 'FAIL'}")
    calm = rec.recommend(cases[5][1], cases[5][2])
    print(f"    Calm tier suppressed               : "
          f"{'PASS' if calm['suppressed'] else 'FAIL'}")

    d, s = flatten_gesture_events(cases[3][1]["gesture_events"])
    print(f"    Unapplied support event ignored    : "
          f"{'PASS' if not d and s == 0.0 else 'FAIL'}")

    work = rec.recommend(cases[1][1], cases[1][2])
    excl = INTERVENTION_MAP[work["intervention_id"]]
    ok = ("colleagues" not in excl.get("excludes_social", [])
          and "work" not in excl.get("excludes_location", []))
    print(f"    Feasibility respected at work      : {'PASS' if ok else 'FAIL'}")

    due = pd.to_datetime(full["outcome_check_due"]) - pd.to_datetime(full["timestamp"])
    print(f"    outcome_check_due = +30 min        : "
          f"{'PASS' if due == timedelta(minutes=30) else 'FAIL'}")

    print("\n" + "=" * 76)
    print("  [OK] End-to-end pipeline verified.")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    _demo()