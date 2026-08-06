"""
MindPulse — Module 3  |  Demo UI — Scenario Data Generator
===========================================================
Runs four representative episodes through the real Module 3 pipeline and
writes the results to a JSON file that `outputs/demo_ui.html` renders.

This adds no functionality. Every value it writes comes from
`module3_recommender.recommend()` — the same call the evaluation pipeline
makes. Its only extra work is recording WHY each rejected intervention was
rejected, which the recommender does not report because nothing else needed
it: Layer 1 simply returns the surviving pool.

The exclusion reasons are derived by re-applying the same feasibility
predicates the rule engine uses (tier membership, social and location
exclusion lists, time-of-day requirement) to the interventions absent from
the returned pool. They are therefore descriptions of the rule engine's
decision, not a reimplementation of it — if the two ever disagreed, the pool
would still be whatever the rule engine returned.

Usage
-----
    python src/generate_demo_data.py

Output
------
    outputs/demo_scenarios.json

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import json

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import INTERVENTIONS, INTERVENTION_MAP
from rule_engine import derive_time_of_day
from module3_recommender import Module3Recommender, adapt_inputs

OUT = os.path.join("outputs", "demo_scenarios.json")


# ══════════════════════════════════════════════════════════════════════════════
# SCENARIOS
# ══════════════════════════════════════════════════════════════════════════════

SCENARIOS = [
    {
        "key": "acute",
        "label": "Acute stress at university",
        "blurb": "Full context available from Module 2. The standard path.",
        "m1": {
            "user_id": "U001", "tier": "acute", "confidence": 0.88,
            "baseline_deviation": 3.1, "signal_quality": "good",
            "baseline_mode": "personalised", "timestamp": "2026-03-02T14:20:00",
            "gesture_events": [{"type": "behavioural_support", "count": 6,
                                "support_score": 0.82, "support_applied": True}],
        },
        "m2": {
            "dialogue_completed": True, "trigger_type": "academic",
            "trigger_confidence": 0.81, "location_context": "university",
            "social_context": "alone", "kg_node_id": "ep_88213",
        },
    },
    {
        "key": "work",
        "label": "Mild stress at work, with colleagues",
        "blurb": "Feasibility filtering visibly removes anything needing privacy.",
        "m1": {
            "user_id": "U002", "tier": "mild", "confidence": 0.79,
            "baseline_deviation": 1.1, "signal_quality": "good",
            "baseline_mode": "personalised", "timestamp": "2026-03-03T10:05:00",
            "gesture_events": [],
        },
        "m2": {
            "dialogue_completed": True, "trigger_type": "interpersonal",
            "trigger_confidence": 0.72, "location_context": "work",
            "social_context": "colleagues",
        },
    },
    {
        "key": "nodialogue",
        "label": "No Module 2 dialogue",
        "blurb": "Context unavailable. The module proceeds on Module 1 alone.",
        "m1": {
            "user_id": "U003", "tier": "moderate", "confidence": 0.74,
            "baseline_deviation": 2.0, "signal_quality": "degraded",
            "baseline_mode": "cold_start", "timestamp": "2026-03-04T21:40:00",
            "gesture_events": [{"type": "behavioural_support", "count": 3,
                                "support_score": 0.55, "support_applied": True}],
        },
        "m2": None,
    },
    {
        "key": "suppressed",
        "label": "Poor signal quality",
        "blurb": "The physiological signal cannot be trusted. Nothing is issued.",
        "m1": {
            "user_id": "U005", "tier": "acute", "confidence": 0.91,
            "baseline_deviation": 2.8, "signal_quality": "poor",
            "baseline_mode": "personalised", "timestamp": "2026-03-06T09:15:00",
            "gesture_events": [],
        },
        "m2": {
            "dialogue_completed": True, "trigger_type": "financial",
            "trigger_confidence": 0.70, "location_context": "home",
            "social_context": "alone",
        },
    },
]


# ══════════════════════════════════════════════════════════════════════════════
# EXCLUSION REASONS
# ══════════════════════════════════════════════════════════════════════════════

def exclusion_reason(inv: dict, ep: dict) -> str:
    """
    Why this intervention is not in the feasible pool.

    Re-applies the same predicates the rule engine uses. Checked in the order
    the engine applies them, so the reason shown is the first one that fired
    rather than an arbitrary one.
    """
    if ep["tier"] not in inv["tiers"]:
        return "wrong stress tier"
    if ep["social_context"] in inv.get("excludes_social", []):
        return f"not with {ep['social_context']}"
    if ep["location_context"] in inv.get("excludes_location", []):
        loc = ep["location_context"].replace("_", " ")
        return f"not {loc}"
    tod = derive_time_of_day(ep["timestamp"])
    if inv.get("requires_time") and tod not in inv["requires_time"]:
        return f"{'/'.join(inv['requires_time'])} only"
    return "lower priority"


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  MODULE 3 — DEMO SCENARIO DATA")
    print("=" * 70)

    # History from training sessions only, matching how the module is evaluated.
    import pandas as pd
    from layer2_evaluation import make_test_mask
    df = pd.read_csv(os.path.join("data", "synthetic", "module3_dataset.csv"))
    train = df[~make_test_mask(df)]

    rec = Module3Recommender(history_df=train)
    print(f"\n  Layer 1 ready · Layer 2 {'ready' if rec.layer2 else 'MISSING'} · "
          f"Layer 3 {'ready' if rec.layer3 else 'MISSING'}")

    out = []
    for s in SCENARIOS:
        result = rec.recommend(s["m1"], s["m2"])
        ep = adapt_inputs(s["m1"], s["m2"])

        # Which interventions survived, and why the others did not
        if result["suppressed"]:
            pool_ids = []
        else:
            l1 = rec.engine.recommend(ep)
            pool_ids = [c["intervention_id"] for c in l1["ranked_candidates"]]

        ranked = ([result["intervention_id"]]
                  + [a["intervention_id"] for a in result["alternatives"]]
                  if not result["suppressed"] else [])

        library = []
        for inv in INTERVENTIONS:
            feasible = inv["id"] in pool_ids
            library.append({
                "id": inv["id"], "name": inv["name"], "type": inv["type"],
                "duration": inv["duration"], "feasible": feasible,
                "reason": None if feasible else exclusion_reason(inv, ep),
            })

        out.append({
            "key": s["key"], "label": s["label"], "blurb": s["blurb"],
            "module1": {
                "tier": s["m1"]["tier"],
                "confidence": s["m1"]["confidence"],
                "baseline_deviation": s["m1"]["baseline_deviation"],
                "signal_quality": s["m1"]["signal_quality"],
                "baseline_mode": s["m1"]["baseline_mode"],
                "support_detected": ep["support_detected"],
                "support_score": ep["support_score"],
                "timestamp": s["m1"]["timestamp"],
            },
            "module2": None if s["m2"] is None else {
                "trigger_type": s["m2"]["trigger_type"],
                "trigger_confidence": s["m2"]["trigger_confidence"],
                "location_context": s["m2"]["location_context"],
                "social_context": s["m2"]["social_context"],
            },
            "library": library,
            "pool_size": len(pool_ids),
            "ranked": ranked,
            "result": result,
        })

        if result["suppressed"]:
            print(f"    {s['label']:<42} SUPPRESSED")
        else:
            print(f"    {s['label']:<42} {result['intervention_id']} "
                  f"({len(pool_ids)} feasible of 22)")

    os.makedirs("outputs", exist_ok=True)
    with open(OUT, "w", encoding="utf-8") as f:
        json.dump(out, f, indent=1, default=str)

    print(f"\n  Saved -> {OUT}")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()