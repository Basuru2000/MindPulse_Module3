"""
MindPulse — Module 3  |  Sub-layer 3.2: ANN Training
=====================================================
Trains the Layer3ANN residual predictor and selects the blend weight that
controls how much it is allowed to influence the final ranking.

    layer3_score = layer2_score + beta * predicted_residual

WHAT THIS SCRIPT IS ACTUALLY TESTING
-------------------------------------
Phase 2 established that the residual target is not predictable from these
features: Ridge scored CV R2 = -0.0512 against -0.0022 for predicting the
mean. So the expected outcome here is that the MLP COLLAPSES — predictions
with near-zero variance, and a selected beta near zero, leaving Layer 3
equal to Layer 2.

That is the correct behaviour, not a failure. But "the model predicts zero"
is ambiguous between two very different situations:

    (a) there is no signal in the data, or
    (b) the training code is broken.

So this script runs a POSITIVE CONTROL. It builds a synthetic target that is
a known function of the same feature matrix, retrains from scratch, and
confirms the model recovers it. If the control passes and the real target
still collapses, (b) is ruled out and the null result is a property of the
data.

PROTOCOL
--------
* 5-fold cross-validation GROUPED BY USER. Personalisation must generalise to
  a user's unseen sessions, and grouping by user is the only split that tests
  that. A random split would let the model memorise user-level offsets.
* Features standardised with a scaler fitted on the training fold only.
* Hidden width swept over {8, 16, 32}; at 526 rows this ranges from 2.6 to
  0.66 rows per parameter, so capacity has to be chosen rather than assumed.
* Early stopping on fold validation loss.
* beta selected on TRAINING sessions by the IPS estimator from
  `layer3_baseline.py`. The test set is not touched anywhere in this file.

Usage
-----
    python src/train_ann.py

Outputs
-------
    models/ann_layer3.pt    weights, scaler statistics, beta, CV metadata

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.model_selection import GroupKFold
from sklearn.metrics import r2_score

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from layer1_evaluation import DATASET_PATH
from layer2_evaluation import make_test_mask
from user_history import UserHistory
from layer3_features import (build_training_matrix, build_candidate_matrix,
                             FEATURE_NAMES)
from layer3_baseline import (load_stack, build_session_records, ips_estimate,
                             IPS_K)
from ann_model import Layer3ANN

MODEL_DIR   = "models"
MODEL_PATH  = os.path.join(MODEL_DIR, "ann_layer3.pt")

HIDDEN_GRID = [8, 16, 32]
BETA_GRID   = [0.0, 0.25, 0.5, 1.0, 2.0]
MAX_EPOCHS  = 400
PATIENCE    = 40
LR          = 0.01
WEIGHT_DECAY = 1e-3
N_FOLDS     = 5
SEED        = 42

# History settings carried over from Sub-layer 3.1's training-side tuning
HALF_LIFE   = 90.0
SHRINKAGE   = 1.0


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def standardise(train_X, *others):
    """Fit on the training fold only; apply everywhere. Constant columns pass through."""
    mu, sd = train_X.mean(0), train_X.std(0)
    sd = np.where(sd < 1e-8, 1.0, sd)
    return [(a - mu) / sd for a in (train_X, *others)], mu, sd


def fit_fold(Xtr, ytr, Xva, yva, hidden, seed=SEED):
    """Train one model with early stopping on the fold's validation split."""
    torch.manual_seed(seed)
    model = Layer3ANN(in_dim=Xtr.shape[1], hidden_dim=hidden)
    opt = torch.optim.Adam(model.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)

    Xtr_t = torch.tensor(Xtr, dtype=torch.float)
    ytr_t = torch.tensor(ytr, dtype=torch.float)
    Xva_t = torch.tensor(Xva, dtype=torch.float)
    yva_t = torch.tensor(yva, dtype=torch.float)

    best, best_state, stale = np.inf, None, 0
    for _ in range(MAX_EPOCHS):
        model.train()
        opt.zero_grad()
        loss = F.mse_loss(model(Xtr_t), ytr_t)
        loss.backward()
        opt.step()

        model.eval()
        with torch.no_grad():
            vl = F.mse_loss(model(Xva_t), yva_t).item()
        if vl < best - 1e-6:
            best, best_state, stale = vl, copy.deepcopy(model.state_dict()), 0
        else:
            stale += 1
            if stale >= PATIENCE:
                break

    model.load_state_dict(best_state)
    model.eval()
    with torch.no_grad():
        return model, model(Xva_t).numpy()


def cross_validate(X, y, groups, hidden):
    """Out-of-fold predictions for one hidden width."""
    oof = np.zeros(len(y))
    for tr_i, va_i in GroupKFold(n_splits=N_FOLDS).split(X, y, groups=groups):
        (Xtr, Xva), _, _ = standardise(X[tr_i], X[va_i])
        _, pred = fit_fold(Xtr, y[tr_i], Xva, y[va_i], hidden)
        oof[va_i] = pred
    return oof


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 74)
    print("  MODULE 3 — SUB-LAYER 3.2: ANN TRAINING")
    print("=" * 74)

    df = pd.read_csv(DATASET_PATH)
    tm = make_test_mask(df)
    df_train = df[~tm].copy()

    engine, l2model, z, node_to_idx = load_stack()
    recs = build_session_records(df_train, engine, l2model, z, node_to_idx)
    H = UserHistory(df_train, half_life_days=HALF_LIFE, shrinkage_k=SHRINKAGE)
    X, y, groups = build_training_matrix(recs, H, H.pop_mean, H.global_mean)

    print(f"\n  Supervised rows : {X.shape[0]}   features : {X.shape[1]}")
    print(f"  Target residual : mean {y.mean():+.4f}, sd {y.std():.4f}")
    print(f"  Test sessions   : untouched in this file")

    # ── Hidden-width sweep ────────────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("  CROSS-VALIDATION  (5-fold, grouped by user)")
    print("-" * 74)
    print(f"  {'hidden':<10}{'params':<10}{'CV R2':<12}{'pred sd':<12}{'vs mean'}")

    mean_r2 = r2_score(y, np.full_like(y, y.mean()))
    results = []
    for h in HIDDEN_GRID:
        oof = cross_validate(X, y, groups, h)
        r2 = r2_score(y, oof)
        n_par = Layer3ANN(in_dim=X.shape[1], hidden_dim=h).count_parameters()
        results.append({"hidden": h, "r2": r2, "pred_sd": float(oof.std()), "oof": oof})
        print(f"  {h:<10}{n_par:<10}{r2:<12.4f}{oof.std():<12.4f}{r2 - mean_r2:+.4f}")

    print(f"\n  Predict-the-mean baseline CV R2 : {mean_r2:+.4f}")
    best = max(results, key=lambda r: r["r2"])
    print(f"  Best hidden width               : {best['hidden']}  "
          f"(CV R2 {best['r2']:+.4f})")

    learned = best["r2"] > mean_r2 + 0.01

    # ── Collapse check ────────────────────────────────────────────────────────
    print("\n  Collapse check:")
    ratio = best["pred_sd"] / y.std()
    print(f"    prediction sd / target sd     : {ratio:.4f}")
    print(f"    model learned usable signal   : {'YES' if learned else 'NO'}")
    if not learned:
        print("    -> Expected. Phase 2 measured Ridge at CV R2 -0.0512 on this")
        print("       target. Predictions concentrate near zero, so Layer 3 will")
        print("       degrade gracefully onto Layer 2 rather than adding noise.")

    # ── Positive control ──────────────────────────────────────────────────────
    print("\n" + "-" * 74)
    print("  POSITIVE CONTROL  (can this trainer learn when signal EXISTS?)")
    print("-" * 74)
    print("  Same features, same protocol; target replaced by a known linear")
    print("  function of the features plus noise. If this fails, the null result")
    print("  above is a bug rather than a property of the data.")

    rng = np.random.default_rng(SEED)
    hs = FEATURE_NAMES.index("hist_score")
    uo = FEATURE_NAMES.index("user_offset")
    for noise in (0.2, 0.5):
        y_syn = (2.0 * X[:, hs] + 1.0 * X[:, uo]
                 + rng.normal(0, noise, size=len(y)))
        oof = cross_validate(X, y_syn, groups, best["hidden"])
        r2s = r2_score(y_syn, oof)
        base = r2_score(y_syn, np.full_like(y_syn, y_syn.mean()))
        print(f"    noise sd {noise}: CV R2 {r2s:+.4f}  (mean baseline {base:+.4f})  "
              f"{'PASS' if r2s > 0.2 else 'FAIL'}")

    # ── Blend weight beta, selected on TRAINING via IPS ───────────────────────
    print("\n" + "-" * 74)
    print("  BLEND WEIGHT beta  (IPS on TRAINING sessions)")
    print("-" * 74)

    (Xs,), mu, sd = standardise(X)
    torch.manual_seed(SEED)
    final = Layer3ANN(in_dim=X.shape[1], hidden_dim=best["hidden"])
    opt = torch.optim.Adam(final.parameters(), lr=LR, weight_decay=WEIGHT_DECAY)
    Xt, yt = torch.tensor(Xs, dtype=torch.float), torch.tensor(y, dtype=torch.float)
    for _ in range(200):
        final.train(); opt.zero_grad()
        F.mse_loss(final(Xt), yt).backward(); opt.step()
    final.eval()

    print(f"  {'beta':<10}{'IPS E[rating] on train'}")
    ips_rows = []
    for b in BETA_GRID:
        rankings = []
        for rec in recs:
            Xc = (build_candidate_matrix(rec, H) - mu) / sd
            with torch.no_grad():
                p = final(torch.tensor(Xc, dtype=torch.float)).numpy()
            order = np.argsort(-(rec["l2"] + b * p))
            rankings.append([rec["pool"][i] for i in order])
        v, _ = ips_estimate(recs, rankings)
        ips_rows.append({"beta": b, "ips": float(v.mean())})
        print(f"  {b:<10}{v.mean():.4f}")

    best_beta = max(ips_rows, key=lambda r: r["ips"])
    print(f"\n  Selected beta = {best_beta['beta']}  (train IPS {best_beta['ips']:.4f})")
    if best_beta["beta"] == 0.0:
        print("  beta = 0 means the tuning found no benefit from the ANN correction;")
        print("  Layer 3.2 is then identical to Layer 2 by construction.")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    torch.save({
        "state_dict": final.state_dict(),
        "config": {"in_dim": X.shape[1], "hidden_dim": best["hidden"],
                   "dropout": final.dropout},
        "scaler": {"mean": mu, "std": sd},
        "beta": float(best_beta["beta"]),
        "history": {"half_life_days": HALF_LIFE, "shrinkage_k": SHRINKAGE},
        "cv": {"r2": best["r2"], "mean_baseline_r2": mean_r2,
               "pred_sd_ratio": ratio, "learned_signal": bool(learned)},
        "feature_names": FEATURE_NAMES,
    }, MODEL_PATH)
    print(f"\n  Saved -> {MODEL_PATH}")
    print("=" * 74 + "\n")


if __name__ == "__main__":
    main()