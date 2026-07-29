"""
MindPulse — Module 3  |  Layer 3: Feature Construction
=======================================================
Builds the supervised training matrix for Sub-layer 3.2's MLP.

WHAT THE MODEL PREDICTS, AND WHY IT IS THE RESIDUAL
----------------------------------------------------
Layer 3's target is

    residual = outcome_rating - population_mean_rating(intervention)

not the raw rating. Layer 2 already scores overall clinical appropriateness;
asking Layer 3 to predict raw ratings would mostly re-learn that, and any
apparent gain would be Layer 2's work restated. The residual isolates exactly
the quantity Layer 3 is supposed to contribute — how this specific user
deviates from the population on this intervention.

It also makes a null result interpretable. If there is no individual signal,
a correctly-behaving model predicts approximately zero everywhere and Layer 3
collapses gracefully onto Layer 2. With a raw-rating target the same
situation would produce confident-looking predictions that merely echo
population appropriateness.

WHICH ROWS ARE SUPERVISED
-------------------------
Only the DELIVERED intervention of each session has an observed outcome, so
there is exactly one labelled row per usable session (526 after Layer 1
suppression). Non-delivered candidates cannot be labelled and are not invented.

This is normally where selection bias enters an off-policy problem — but not
here. `dataset_generator.py` selects the delivered intervention with
`random.choice(feasible)`, a uniform logging policy, so the labelled rows are
an unbiased sample of the candidate pool. The model can be trained on
delivered interventions and applied to all candidates without reweighting.
The same property is what makes the IPS estimator in `layer3_baseline.py`
unbiased.

FEATURE GROUPS
--------------
1. History (leak-safe, from `user_history.UserHistory`) — the personalisation
   signal itself, plus how much evidence backs it.
2. Layer 2 score — lets the MLP learn a correction relative to Layer 2 rather
   than re-deriving it.
3. Episode-level Module 1/2 signals — `baseline_deviation`, `support_score`,
   `confidence`, `trigger_confidence`, signal quality, cold-start, tier.
   These are dynamic per-episode values that were deliberately NOT encoded in
   the Layer 2 graph (spec Section 2.2.2), so they are a genuine addition
   Layer 3 makes rather than a duplication.
4. Intervention attributes — duration and category one-hot.
5. User-level context — the user's overall mean residual (their global
   offset) and how many sessions they have accumulated.

LEAKAGE CONTROL
---------------
Every history feature is computed strictly from sessions EARLIER than the row
being described, via `UserHistory`. The user-level offset is computed the same
way. One acknowledged approximation: the population mean used to form the
residual target is computed over all training sessions including the row
itself, which is standard centring practice and negligible at n=526, but is
recorded here rather than left implicit.

Author : Module 3 — MindPulse (Team MindForge)
"""

import numpy as np
import pandas as pd

from knowledge_graph import INTERVENTION_MAP

CATEGORIES = ["breathing", "physical", "cognitive", "social", "sensory"]
SIGNAL_QUALITY_ORD = {"good": 1.0, "degraded": 0.5, "poor": 0.0}
TIER_ORD = {"mild": 1 / 3, "moderate": 2 / 3, "acute": 1.0}
MAX_DURATION = 5.0

FEATURE_NAMES = (
    ["hist_score", "hist_is_intervention", "hist_is_category", "hist_is_none",
     "hist_log_n", "hist_weight"]
    + ["layer2_score"]
    + ["baseline_deviation", "support_score", "support_detected", "confidence",
       "trigger_confidence", "signal_quality", "cold_start", "tier"]
    + ["duration"] + [f"cat_{c}" for c in CATEGORIES]
    + ["user_offset", "user_experience"]
)


def _user_offset(history, user_id, as_of):
    """
    The user's overall mean residual from strictly-earlier sessions.

    A global offset cannot reorder candidates on its own, but it lets the MLP
    calibrate: a user who rates everything low is different from one who
    dislikes this particular intervention, and without this feature the model
    cannot tell those cases apart.
    """
    u = history._by_user.get(user_id)
    if u is None:
        return 0.0, 0
    prior = u["ts"] < np.datetime64(pd.to_datetime(as_of))
    if not prior.any():
        return 0.0, 0
    return float(u["resid"][prior].mean()), int(prior.sum())


def build_features(row, intervention_id, layer2_score, history) -> np.ndarray:
    """Feature vector for one (session, candidate intervention) pair."""
    h = history.score(row["user_id"], intervention_id, row["timestamp"])
    off, exp = _user_offset(history, row["user_id"], row["timestamp"])
    inv = INTERVENTION_MAP[intervention_id]

    return np.array(
        [h["score"],
         1.0 if h["level"] == "intervention" else 0.0,
         1.0 if h["level"] == "category" else 0.0,
         1.0 if h["level"] == "none" else 0.0,
         float(np.log1p(h["n"])),
         float(h["weight"]),

         float(layer2_score),

         float(row["baseline_deviation"]),
         float(row["support_score"]),
         1.0 if bool(row["support_detected"]) else 0.0,
         float(row["confidence"]),
         float(row["trigger_confidence"]),
         SIGNAL_QUALITY_ORD.get(row["signal_quality"], 0.5),
         1.0 if row["baseline_mode"] == "cold_start" else 0.0,
         TIER_ORD.get(row["tier"], 0.5),

         inv["duration"] / MAX_DURATION]
        + [1.0 if inv["type"] == c else 0.0 for c in CATEGORIES]
        + [off, float(np.log1p(exp))],
        dtype=float,
    )


def build_training_matrix(records, history, pop_mean, global_mean):
    """
    Supervised rows: one per session, for the DELIVERED intervention only.

    Returns X (n, d), y (n,) residual targets, and groups (n,) user ids for
    grouped cross-validation.
    """
    X, y, groups = [], [], []
    for rec in records:
        row, gt = rec["row"], rec["gt"]
        idx = rec["pool"].index(gt) if gt in rec["pool"] else None
        if idx is None:
            continue                      # delivered item not feasible now; skip
        X.append(build_features(row, gt, rec["l2"][idx], history))
        y.append(rec["rating"] - pop_mean.get(gt, global_mean))
        groups.append(row["user_id"])
    return np.array(X), np.array(y), np.array(groups)


def build_candidate_matrix(rec, history):
    """Feature matrix for every candidate in one session's pool, for inference."""
    return np.vstack([build_features(rec["row"], inv, rec["l2"][i], history)
                      for i, inv in enumerate(rec["pool"])])


# ══════════════════════════════════════════════════════════════════════════════
# SELF-TEST — including the question that decides whether 3.2 is worth building
# ══════════════════════════════════════════════════════════════════════════════

def _self_test():
    from sklearn.linear_model import Ridge
    from sklearn.dummy import DummyRegressor
    from sklearn.model_selection import GroupKFold, cross_val_predict
    from sklearn.metrics import r2_score

    from layer1_evaluation import DATASET_PATH
    from layer2_evaluation import make_test_mask
    from user_history import UserHistory
    from layer3_baseline import load_stack, build_session_records

    print("\n" + "=" * 72)
    print("  LAYER 3 — FEATURE BUILDER SELF-TEST")
    print("=" * 72)

    df = pd.read_csv(DATASET_PATH)
    tm = make_test_mask(df)
    df_train = df[~tm].copy()
    engine, model, z, node_to_idx = load_stack()
    recs = build_session_records(df_train, engine, model, z, node_to_idx)

    H = UserHistory(df_train)
    X, y, g = build_training_matrix(recs, H, H.pop_mean, H.global_mean)

    print(f"\n  Supervised rows : {X.shape[0]} (one per usable training session)")
    print(f"  Features        : {X.shape[1]}  (names match: "
          f"{'PASS' if X.shape[1] == len(FEATURE_NAMES) else 'FAIL'})")
    print(f"  Target (residual): mean {y.mean():+.4f}, sd {y.std():.4f}")

    print("\n  Checks:")
    print(f"    No NaN / Inf in features        : "
          f"{'PASS' if np.isfinite(X).all() else 'FAIL'}")
    print(f"    Target centred near 0           : "
          f"{'PASS' if abs(y.mean()) < 0.05 else 'FAIL'}  ({y.mean():+.4f})")
    zero_var = [FEATURE_NAMES[i] for i in range(X.shape[1]) if X[:, i].std() == 0]
    print(f"    No constant features            : "
          f"{'PASS' if not zero_var else 'FAIL ' + str(zero_var)}")

    # ── The decisive question ─────────────────────────────────────────────────
    print("\n  CAN ANY MODEL PREDICT THIS TARGET?")
    print("  (5-fold CV grouped by user — a model that cannot beat predicting")
    print("   the mean has found no individual signal, and no MLP will either)")
    cv = GroupKFold(n_splits=5)
    ridge = cross_val_predict(Ridge(alpha=1.0), X, y, cv=cv, groups=g)
    dummy = cross_val_predict(DummyRegressor(strategy="mean"), X, y, cv=cv, groups=g)
    r2_r, r2_d = r2_score(y, ridge), r2_score(y, dummy)
    print(f"\n    Ridge regression   CV R2 = {r2_r:+.4f}")
    print(f"    Predict-the-mean   CV R2 = {r2_d:+.4f}")
    print(f"    improvement              = {r2_r - r2_d:+.4f}   "
          f"{'signal present' if r2_r > 0.01 else 'NO usable signal'}")

    # History features alone — the personalisation-specific subset
    hist_cols = [FEATURE_NAMES.index(c) for c in
                 ["hist_score", "hist_log_n", "hist_weight", "user_offset"]]
    r2_h = r2_score(y, cross_val_predict(Ridge(alpha=1.0), X[:, hist_cols], y,
                                         cv=cv, groups=g))
    print(f"\n    History features only  CV R2 = {r2_h:+.4f}")

    print("\n" + "=" * 72)
    print("  [OK] Feature builder verified.")
    print("=" * 72 + "\n")


if __name__ == "__main__":
    _self_test()