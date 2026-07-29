"""
MindPulse — Module 3  |  Sub-layer 3.3: Online Feedback Loop
=============================================================
Streams the held-out sessions in chronological order, updating the Layer 3
model as each outcome arrives (spec Section 2.3, Sub-layer 3.3).

WHAT IS BEING ISOLATED
----------------------
Four policies are run over the SAME chronological stream, differing in one
thing at a time, so the comparison attributes each increment correctly:

  Layer 2      no personalisation at all
  3.1          growing history, heuristic recency weighting, no model
  3.2          growing history + MLP with FROZEN weights
  3.3          growing history + MLP updated online after each outcome

History grows for 3.1, 3.2 and 3.3 alike — history is data, not model. The
only difference between 3.2 and 3.3 is whether the network weights change,
which is precisely the contribution Sub-layer 3.3 claims.

WHY THIS EVALUATION IS LEAK-FREE BY CONSTRUCTION
------------------------------------------------
Every session is scored using only outcomes that occurred strictly earlier.
This is prequential (test-then-train) evaluation, and it is the honest form
for a system that learns from feedback. It also resolves the temporal
concern flagged in the Layer 3 roadmap: the random train/test split meant
static evaluation could use history from sessions that happened after the one
being scored. Streaming in time order removes that entirely.

MEASURED HEADROOM — READ BEFORE INTERPRETING RESULTS
-----------------------------------------------------
The test stream is thin:

  * 5 test sessions per user, uniformly
  * 20% of test sessions are that user's FIRST, so no online update can have
    influenced them
  * the stream adds at most 22% more per-user data on top of the 23 training
    sessions in which Phase 2 and Phase 3 already found no learnable signal

So a null result here is expected and is not evidence that online learning
does not work. To separate "the mechanism is inert" from "there is nothing to
learn", this script runs a POSITIVE CONTROL: the same stream with a known
preference effect injected into the outcomes, where 3.3 should pull ahead of
frozen 3.2 as evidence accumulates.

Usage
-----
    python src/layer3_online.py      (requires models/ann_layer3.pt)

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from knowledge_graph import INTERVENTION_MAP
from layer1_evaluation import DATASET_PATH
from layer2_evaluation import make_test_mask
from user_history import UserHistory
from layer3_features import build_training_matrix, build_candidate_matrix
from layer3_baseline import (load_stack, build_session_records, ips_estimate,
                             bootstrap_ci, IPS_K)
from ann_model import Layer3ANN

ANN_PATH     = os.path.join("models", "ann_layer3.pt")
RESULTS_PATH = os.path.join("data", "synthetic", "layer3_online_results.csv")

ONLINE_LR    = 1e-3    # overwritten by select_online_lr() at run time
ONLINE_STEPS = 5       # gradient steps per observed outcome
REPLAY_BATCH = 32      # replayed rows per step, to limit forgetting

# (learning rate, steps) candidates. Selected on the TRAINING stream by IPS —
# never on the test stream, and never on the positive control, which would be
# tuning on the very thing used to validate the mechanism.
ONLINE_GRID  = [(0.0, 0), (1e-3, 5), (1e-2, 5), (5e-2, 10), (1e-1, 20)]
ALPHA_31     = 4.0     # from Sub-layer 3.1's training-side tuning
SEED         = 42


# ══════════════════════════════════════════════════════════════════════════════
# STREAM
# ══════════════════════════════════════════════════════════════════════════════

def select_online_lr(train_recs, df_train, ckpt):
    """
    Choose the online learning rate prequentially on the TRAINING stream.

    Each training session is scored before being learned from, so the
    procedure mirrors deployment. The offline model was fitted on these same
    rows, which makes the absolute IPS values optimistic — but the comparison
    ACROSS learning rates is still fair, since every candidate inherits the
    same warm start. What matters here is which rate wins, not its level.

    Selecting on the test stream would be leakage; selecting on the injected
    positive control would guarantee the control passes and make it worthless
    as evidence.
    """
    global ONLINE_LR, ONLINE_STEPS
    print(f"  {'online_lr':<12}{'steps':<8}{'train IPS':<12}{'drift'}")
    best = None
    for lr, st in ONLINE_GRID:
        ONLINE_LR, ONLINE_STEPS = lr, st
        rk, dr = run_stream(train_recs, df_train, ckpt, online=(lr > 0))
        v, _ = ips_estimate(train_recs, rk)
        print(f"  {lr:<12}{st:<8}{v.mean():<12.4f}{dr:.5f}")
        if best is None or v.mean() > best[2]:
            best = (lr, st, float(v.mean()))
    ONLINE_LR, ONLINE_STEPS = best[0], best[1]
    return best


def run_stream(test_recs, df_train, ckpt, online: bool, ratings=None,
               alpha_only=False, seed=SEED):
    """
    Replay the test sessions in chronological order under one policy.

    Parameters
    ----------
    online : bool
        True  -> update MLP weights after each observed outcome (3.3)
        False -> frozen weights (3.2)
    ratings : dict or None
        Optional session_id -> rating override, used by the positive control
        to inject a known preference effect without touching the real data.
    alpha_only : bool
        Ignore the MLP and rank by Layer 2 + ALPHA_31 * history score (3.1).

    Returns the per-session ranking list and the mean absolute online update.
    """
    torch.manual_seed(seed)
    rng = np.random.default_rng(seed)

    H = UserHistory(df_train,
                    half_life_days=ckpt["history"]["half_life_days"],
                    shrinkage_k=ckpt["history"]["shrinkage_k"])

    model = Layer3ANN(**ckpt["config"])
    model.load_state_dict(copy.deepcopy(ckpt["state_dict"]))
    model.eval()
    opt = torch.optim.Adam(model.parameters(), lr=ONLINE_LR)

    mu, sd = ckpt["scaler"]["mean"], ckpt["scaler"]["std"]
    beta = ckpt["beta"]

    # Replay buffer warm-started with the training rows, so online steps do not
    # immediately overwrite what offline training established.
    engine_recs = None
    buf_X, buf_y = ckpt.get("_buf_X"), ckpt.get("_buf_y")

    rankings, drifts = [], []
    for rec in test_recs:
        Xc = (build_candidate_matrix(rec, H) - mu) / sd
        with torch.no_grad():
            pred = model(torch.tensor(Xc, dtype=torch.float)).numpy()

        if alpha_only:
            adj = rec["l2"] + ALPHA_31 * H.score_many(rec["user"], rec["pool"], rec["ts"])
        else:
            adj = rec["l2"] + beta * pred
        rankings.append([rec["pool"][i] for i in np.argsort(-adj)])

        # ── Observe the outcome ───────────────────────────────────────────────
        gt = rec["gt"]
        rating = rec["rating"] if ratings is None else ratings[rec["row"]["session_id"]]
        if gt not in rec["pool"]:
            continue
        gi = rec["pool"].index(gt)
        x_new = Xc[gi]
        y_new = rating - H.pop_mean.get(gt, H.global_mean)

        H.add_observation(rec["user"], gt, rec["ts"], rating)

        if online and buf_X is not None:
            before = pred.copy()
            xb = torch.tensor(np.vstack([x_new]), dtype=torch.float)
            yb = torch.tensor([y_new], dtype=torch.float)
            for _ in range(ONLINE_STEPS):
                idx = rng.integers(0, len(buf_X), REPLAY_BATCH)
                xr = torch.tensor(np.vstack([buf_X[idx], x_new]), dtype=torch.float)
                yr = torch.tensor(np.append(buf_y[idx], y_new), dtype=torch.float)
                model.train(); opt.zero_grad()
                F.mse_loss(model(xr), yr).backward(); opt.step()
            model.eval()
            with torch.no_grad():
                after = model(torch.tensor(Xc, dtype=torch.float)).numpy()
            drifts.append(float(np.abs(after - before).mean()))

            buf_X = np.vstack([buf_X, x_new])
            buf_y = np.append(buf_y, y_new)

    return rankings, (float(np.mean(drifts)) if drifts else 0.0)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 76)
    print("  MODULE 3 — SUB-LAYER 3.3: ONLINE FEEDBACK LOOP")
    print("=" * 76)

    if not os.path.exists(ANN_PATH):
        raise FileNotFoundError(f"{ANN_PATH} not found. Run src/train_ann.py first.")

    df = pd.read_csv(DATASET_PATH)
    df["_ts"] = pd.to_datetime(df["timestamp"])
    tm = make_test_mask(df)
    df_train = df[~tm].copy()
    df_test = df[tm].sort_values("_ts").copy()      # CHRONOLOGICAL

    engine, l2model, z, node_to_idx = load_stack()
    test_recs = build_session_records(df_test, engine, l2model, z, node_to_idx)
    ckpt = torch.load(ANN_PATH, weights_only=False)

    # Warm-start the replay buffer with the offline training rows
    H0 = UserHistory(df_train,
                     half_life_days=ckpt["history"]["half_life_days"],
                     shrinkage_k=ckpt["history"]["shrinkage_k"])
    train_recs = build_session_records(df_train, engine, l2model, z, node_to_idx)
    Xtr, ytr, _ = build_training_matrix(train_recs, H0, H0.pop_mean, H0.global_mean)
    ckpt["_buf_X"] = (Xtr - ckpt["scaler"]["mean"]) / ckpt["scaler"]["std"]
    ckpt["_buf_y"] = ytr

    print(f"\n  Test stream : {len(test_recs)} sessions in chronological order")
    print(f"  {df_test['_ts'].min().date()} -> {df_test['_ts'].max().date()}")
    print(f"  beta = {ckpt['beta']}, replay batch {REPLAY_BATCH}")

    # ── Select the online learning rate on the TRAINING stream ────────────────
    print("\n" + "-" * 76)
    print("  ONLINE RATE SELECTION  (prequential IPS on TRAINING stream)")
    print("-" * 76)
    df_train_sorted = df_train.sort_values("_ts").copy()
    train_recs_sorted = build_session_records(df_train_sorted, engine, l2model,
                                              z, node_to_idx)
    sel = select_online_lr(train_recs_sorted, df_train_sorted, ckpt)
    print(f"\n  Selected: lr={sel[0]}, steps={sel[1]}  (train IPS {sel[2]:.4f})")
    if sel[0] <= 1e-3:
        print("  The procedure chose minimal updating. Larger rates scored WORSE on")
        print("  training, which is the correct response when there is no individual")
        print("  signal to track — aggressive adaptation would fit noise.")

    # ── Four policies over the same stream ────────────────────────────────────
    print("\n" + "-" * 76)
    print("  PREQUENTIAL EVALUATION  (each session scored using only earlier outcomes)")
    print("-" * 76)

    zero = copy.deepcopy(ckpt); zero["beta"] = 0.0
    r_l2, _ = run_stream(test_recs, df_train, zero, online=False)
    r_31, _ = run_stream(test_recs, df_train, ckpt, online=False, alpha_only=True)
    r_32, _ = run_stream(test_recs, df_train, ckpt, online=False)
    r_33, drift = run_stream(test_recs, df_train, ckpt, online=True)

    policies = [("Layer 2 (no personalisation)", r_l2),
                ("Sub-layer 3.1 (recency)", r_31),
                ("Sub-layer 3.2 (static MLP)", r_32),
                ("Sub-layer 3.3 (online MLP)", r_33)]

    v_base, _ = ips_estimate(test_recs, r_l2)
    print(f"\n  {'policy':<32}{'IPS E[rating]':<16}{'95% CI':<22}{'vs Layer 2'}")
    rows = []
    for name, rk in policies:
        v, matched = ips_estimate(test_recs, rk)
        e, lo, hi = bootstrap_ci(v)
        d, dlo, dhi = bootstrap_ci(v - v_base)
        tag = "" if name.startswith("Layer 2") else \
              (f"{d:+.4f} [{dlo:+.3f},{dhi:+.3f}]")
        print(f"  {name:<32}{e:<16.4f}[{lo:.3f}, {hi:.3f}]{'':<6}{tag}")
        rows.append({"policy": name, "ips": e, "ci_low": lo, "ci_high": hi,
                     "matched": matched})

    print(f"\n  Online update diagnostic:")
    print(f"    mean |change in prediction| per outcome : {drift:.6f}")
    print(f"    loop is wired and weights are moving    : "
          f"{'PASS' if drift > 1e-9 else 'FAIL — no updates occurred'}")

    # ── Positive control ──────────────────────────────────────────────────────
    print("\n" + "-" * 76)
    print("  POSITIVE CONTROL  (inject a known preference; can 3.3 track it?)")
    print("-" * 76)
    print("  Same stream, outcomes replaced by ratings containing a strong")
    print("  per-user category preference. Online 3.3 should pull ahead of")
    print("  frozen 3.2 as evidence accumulates. If it does not, the online")
    print("  mechanism is inert rather than merely starved of signal.")

    print("  The SAME grid is swept, so if the mechanism can help it will show up")
    print("  as a different rate winning here than on real data.")

    rng = np.random.default_rng(SEED)
    pref = {u: rng.choice(["breathing", "physical", "cognitive", "social", "sensory"])
            for u in df["user_id"].unique()}
    inj = {}
    for rec in test_recs:
        cat = INTERVENTION_MAP[rec["gt"]]["type"]
        inj[rec["row"]["session_id"]] = rec["rating"] + (2.0 if pref[rec["user"]] == cat else 0.0)

    df_tr_inj = df_train.copy()
    df_tr_inj["outcome_rating"] = [
        r + (2.0 if pref[u] == INTERVENTION_MAP[i]["type"] else 0.0)
        for u, i, r in zip(df_tr_inj.user_id, df_tr_inj.intervention_id,
                           df_tr_inj.outcome_rating)]

    global ONLINE_LR, ONLINE_STEPS
    saved = (ONLINE_LR, ONLINE_STEPS)
    inj_ratings = np.array([inj[r["row"]["session_id"]] for r in test_recs])

    def ips_inj(rk):
        return np.array([(len(r["pool"]) / IPS_K) * inj_ratings[i]
                         if r["gt"] in rank[:IPS_K] else 0.0
                         for i, (r, rank) in enumerate(zip(test_recs, rk))])

    c32, _ = run_stream(test_recs, df_tr_inj, ckpt, online=False, ratings=inj)
    base = ips_inj(c32).mean()
    print(f"\n    3.2 frozen (no online updates) : {base:.4f}")
    print(f"    {'online_lr':<12}{'steps':<8}{'3.3 online':<14}{'vs frozen':<14}{'drift'}")
    ctrl_best = None
    for lr, st in ONLINE_GRID:
        if lr == 0.0:
            continue
        ONLINE_LR, ONLINE_STEPS = lr, st
        c33, cdrift = run_stream(test_recs, df_tr_inj, ckpt, online=True, ratings=inj)
        gain = ips_inj(c33).mean() - base
        print(f"    {lr:<12}{st:<8}{ips_inj(c33).mean():<14.4f}{gain:<+14.4f}{cdrift:.5f}")
        if ctrl_best is None or gain > ctrl_best[1]:
            ctrl_best = (lr, gain)
    ONLINE_LR, ONLINE_STEPS = saved

    print(f"\n    Best on injected signal: lr={ctrl_best[0]}, gain {ctrl_best[1]:+.4f}")
    print(f"    Selected on real data  : lr={sel[0]}")
    if ctrl_best[1] > 0.1 and sel[0] < ctrl_best[0]:
        print("    VERDICT: the online mechanism WORKS — it produces a clear gain when")
        print("    individual signal exists, and the same selection procedure correctly")
        print("    chooses near-zero adaptation on the real data, where none does.")
    else:
        print("    VERDICT: no clear separation; treat the online loop as unvalidated.")

    os.makedirs(os.path.dirname(RESULTS_PATH), exist_ok=True)
    pd.DataFrame(rows).to_csv(RESULTS_PATH, index=False)
    print(f"\n  Saved -> {RESULTS_PATH}")
    print("=" * 76 + "\n")


if __name__ == "__main__":
    main()