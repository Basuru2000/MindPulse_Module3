"""
MindPulse — Module 3  |  Layer 2: GCN Training
===============================================
Trains the Layer2GCN on the link-prediction pairs from training_pairs.py and
saves the trained weights.

THE TEST SET IS NOT TOUCHED IN THIS FILE.
-----------------------------------------
The 125 held-out test sessions are reserved entirely for Step 5, where Layer 2
is compared against Layer 1 and the content-based baseline. This script never
constructs test pairs at all, so no tuning decision here can be fitted to them.

WHY CROSS-VALIDATION RATHER THAN A SINGLE VALIDATION SPLIT
----------------------------------------------------------
The first version of this script carved a single ~15% validation set out of the
575 training sessions. That produced only ~70 pairs with ~18 positives — far too
few to estimate a ranking metric. The consequence was measured and severe: the
same architecture scored 0.424 AUC on that single split and 0.653 under 5-fold
cross-validation. The split was not measuring the model, it was measuring noise,
and acting on it would have meant rejecting a working architecture.

Model selection therefore uses 5-fold cross-validation over the training
sessions. Folds are grouped BY SESSION, because each positive session generates
one positive pair plus two random negatives sharing the same five context nodes;
splitting at pair level would place a session's positive and its negatives in
different folds and leak context.

The reported CV AUC (mean +/- std across folds) is the honest estimate of
generalisation and is what selects the learning rate. The final model is then
retrained on all 575 training sessions for the median best epoch found in CV.

ARCHITECTURE NOTE
-----------------
Layer2GCN uses a residual connection by default. This is essential rather than
optional on this graph: the KG is ~90% of a complete bipartite graph, and two
rounds of neighbourhood averaging collapse all 22 intervention embeddings to
near-identical vectors (mean pairwise cosine 0.998), making ranking impossible.
See the use_residual docstring in gcn_model.py.

    5-fold CV AUC, measured on the training sessions:
        2-layer, no residual   0.653 +/- 0.048
        2-layer + residual     0.730 +/- 0.031   <- current default
        1-layer + residual     0.747 +/- 0.030   <- documented alternative

Usage
-----
    python src/train_gcn.py

Outputs
-------
    models/gcn_layer2.pt        final weights + config + CV metadata
    models/training_history.csv per-epoch losses for every fold and lr

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import copy
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import GroupKFold

from knowledge_graph import build_knowledge_graph
from graph_converter import convert_kg_to_pyg, NODE_FEATURE_DIM
from training_pairs import (
    replicate_train_test_split,
    build_training_pairs,
    DATASET_PATH,
    RANDOM_SEED,
)
from gcn_model import Layer2GCN, check_degree_safety


# ── Configuration ─────────────────────────────────────────────────────────────
N_FOLDS         = 5
LEARNING_RATES  = [0.001, 0.005, 0.01]
MAX_EPOCHS      = 400
PATIENCE        = 60
WEIGHT_DECAY    = 5e-4
SEED            = RANDOM_SEED

MODEL_DIR       = "models"
MODEL_PATH      = os.path.join(MODEL_DIR, "gcn_layer2.pt")
HISTORY_PATH    = os.path.join(MODEL_DIR, "training_history.csv")


# ══════════════════════════════════════════════════════════════════════════════
# DATA PREPARATION
# ══════════════════════════════════════════════════════════════════════════════

def prepare_data(dataset_path: str = DATASET_PATH):
    """
    Load the graph and the TRAINING sessions only.

    Deliberately does not call training_pairs.load_training_pairs(), because
    that also builds the test pairs. Not building them is a stronger guarantee
    than building them and promising not to look.
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run src/dataset_generator.py first."
        )

    df = pd.read_csv(dataset_path)
    G = build_knowledge_graph()
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)

    test_mask = replicate_train_test_split(df)
    df_train = df[~test_mask].reset_index(drop=True)

    # Leakage assertion — cheap to check, catastrophic if wrong, silent if not
    test_ids = set(df.loc[test_mask, "session_id"])
    train_ids = set(df_train["session_id"])
    assert not (train_ids & test_ids), "LEAK: test sessions present in training data"
    assert len(train_ids) + len(test_ids) == len(df), "Split does not cover all sessions"

    return {
        "df_train": df_train,
        "data": data,
        "node_to_idx": node_to_idx,
        "idx_to_node": idx_to_node,
        "n_total": len(df),
        "n_test": int(test_mask.sum()),
    }


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PRIMITIVES
# ══════════════════════════════════════════════════════════════════════════════

def evaluate(model, data, pairs):
    """Return (loss, roc_auc) for a set of pairs, in eval mode."""
    model.eval()
    with torch.no_grad():
        logits = model(data.x, data.edge_index, data.edge_weight,
                       pairs["context_indices"], pairs["intervention_indices"])
        loss = F.binary_cross_entropy_with_logits(logits, pairs["labels"]).item()
        y = pairs["labels"].numpy()
        p = torch.sigmoid(logits).numpy()
        auc = roc_auc_score(y, p) if len(np.unique(y)) > 1 else float("nan")
    return loss, auc


def train_model(data, fit_pairs, eval_pairs, lr, max_epochs=MAX_EPOCHS,
                patience=PATIENCE, seed=SEED, fixed_epochs=None, tag=""):
    """
    Train one model. Two modes:

      fixed_epochs is None  -> early stopping on eval_pairs AUC (used inside CV)
      fixed_epochs = N      -> train exactly N epochs, no early stopping (used
                               for the final fit on all training data, where no
                               held-out set remains to stop on)

    Selection inside CV is on AUC rather than loss because ranking quality is
    what Layer 2 is for — Precision@K and NDCG@K in Step 5 depend on the
    ordering of candidates, not on calibrated probabilities.

    Full-batch gradient descent: a few hundred pairs over a 46-node graph, and
    the GCN needs the whole graph present for message passing regardless.
    """
    torch.manual_seed(seed)
    np.random.seed(seed)

    model = Layer2GCN(in_dim=NODE_FEATURE_DIM)
    optimiser = torch.optim.Adam(model.parameters(), lr=lr,
                                 weight_decay=WEIGHT_DECAY)

    best_auc, best_state, best_epoch = -np.inf, None, 0
    stale, history = 0, []
    total = fixed_epochs if fixed_epochs else max_epochs

    for epoch in range(1, total + 1):
        model.train()
        optimiser.zero_grad()
        logits = model(data.x, data.edge_index, data.edge_weight,
                       fit_pairs["context_indices"],
                       fit_pairs["intervention_indices"])
        loss = F.binary_cross_entropy_with_logits(logits, fit_pairs["labels"])

        if not torch.isfinite(loss):
            raise RuntimeError(
                f"Loss became {loss.item()} at epoch {epoch}. Usually means a node's "
                "weighted degree went negative — run check_degree_safety()."
            )

        loss.backward()
        optimiser.step()

        if fixed_epochs:
            history.append({"tag": tag, "lr": lr, "epoch": epoch,
                            "train_loss": loss.item(),
                            "eval_loss": np.nan, "eval_auc": np.nan})
            continue

        eval_loss, eval_auc = evaluate(model, data, eval_pairs)
        history.append({"tag": tag, "lr": lr, "epoch": epoch,
                        "train_loss": loss.item(),
                        "eval_loss": eval_loss, "eval_auc": eval_auc})

        if eval_auc > best_auc + 1e-5:
            best_auc, best_epoch = eval_auc, epoch
            best_state = copy.deepcopy(model.state_dict())
            stale = 0
        else:
            stale += 1
            if stale >= patience:
                break

    if fixed_epochs:
        return {"model": model, "state": copy.deepcopy(model.state_dict()),
                "best_epoch": total, "best_auc": float("nan"), "history": history}

    model.load_state_dict(best_state)
    return {"model": model, "state": best_state, "best_epoch": best_epoch,
            "best_auc": best_auc, "history": history}


# ══════════════════════════════════════════════════════════════════════════════
# CROSS-VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def cross_validate(df_train, data, node_to_idx, lr):
    """5-fold CV grouped by session. Returns per-fold AUCs, best epochs, history."""
    gkf = GroupKFold(n_splits=N_FOLDS)
    groups = df_train["session_id"].values
    aucs, epochs, history = [], [], []

    for k, (fit_i, val_i) in enumerate(gkf.split(df_train, groups=groups), start=1):
        fit_pairs = build_training_pairs(df_train.iloc[fit_i], node_to_idx, seed=SEED)
        val_pairs = build_training_pairs(df_train.iloc[val_i], node_to_idx, seed=SEED + k)

        if len(np.unique(val_pairs["labels"].numpy())) < 2:
            print(f"      fold {k}: skipped (single-class validation fold)")
            continue

        r = train_model(data, fit_pairs, val_pairs, lr, tag=f"cv_fold{k}")
        aucs.append(r["best_auc"])
        epochs.append(r["best_epoch"])
        history.extend(r["history"])
        print(f"      fold {k}: AUC {r['best_auc']:.3f} at epoch {r['best_epoch']}")

    return np.array(aucs), np.array(epochs), history


# ══════════════════════════════════════════════════════════════════════════════
# MAIN
# ══════════════════════════════════════════════════════════════════════════════

def main():
    print("\n" + "=" * 70)
    print("  LAYER 2 — GCN TRAINING  (5-fold CV model selection)")
    print("=" * 70)

    b = prepare_data()
    df_train, data, node_to_idx = b["df_train"], b["data"], b["node_to_idx"]

    print("\n  Session split:")
    print(f"    Total sessions             : {b['n_total']}")
    print(f"    Test  (untouched here)     : {b['n_test']}")
    print(f"    Train (CV + final fit)     : {len(df_train)}")

    all_pairs = build_training_pairs(df_train, node_to_idx, seed=SEED)
    s = all_pairs["stats"]
    print(f"\n  Training pairs (all {len(df_train)} sessions):")
    print(f"    Total                      : {s['total_pairs']}")
    print(f"    Positives                  : {s['positives']} "
          f"({s['positives']/s['total_pairs']:.1%})")

    safe, min_deg = check_degree_safety(data.edge_index, data.edge_weight,
                                        data.x.shape[0])
    print("\n  Pre-flight check:")
    print(f"    Min weighted degree        : {min_deg:.3f}")
    print(f"    Degree safe (no NaN risk)  : {'PASS' if safe else 'FAIL'}")
    if not safe:
        raise RuntimeError("Negative weighted degree — training would produce NaN.")

    # ── CV over learning rates ────────────────────────────────────────────────
    print(f"\n  Running {N_FOLDS}-fold CV per learning rate "
          f"(folds grouped by session)...")
    results, history = [], []
    for lr in LEARNING_RATES:
        print(f"\n    lr = {lr}")
        aucs, epochs, h = cross_validate(df_train, data, node_to_idx, lr)
        history.extend(h)
        results.append({"lr": lr, "aucs": aucs, "epochs": epochs,
                        "mean_auc": float(np.mean(aucs)),
                        "std_auc": float(np.std(aucs)),
                        "median_epoch": int(np.median(epochs))})

    print("\n" + "-" * 70)
    print("  LEARNING RATE COMPARISON  (5-fold CV — test set never involved)")
    print("-" * 70)
    print(f"    {'lr':<10}{'CV AUC':<12}{'std':<10}{'median epoch':<14}")
    for r in results:
        print(f"    {r['lr']:<10}{r['mean_auc']:<12.3f}{r['std_auc']:<10.3f}"
              f"{r['median_epoch']:<14}")

    best = max(results, key=lambda r: r["mean_auc"])
    print(f"\n    Selected: lr = {best['lr']}  "
          f"(CV AUC {best['mean_auc']:.3f} +/- {best['std_auc']:.3f})")

    # ── Sanity checks ─────────────────────────────────────────────────────────
    print("\n  Sanity checks:")
    print(f"    CV AUC > 0.5 (random)      : "
          f"{'PASS' if best['mean_auc'] > 0.5 else 'FAIL'}  ({best['mean_auc']:.3f})")
    margin = best["mean_auc"] - 0.5
    print(f"    Margin over random         : {margin:+.3f} "
          f"({margin / max(best['std_auc'], 1e-9):.1f} std devs)")
    consistent = bool((best["aucs"] > 0.5).all())
    print(f"    Every fold above 0.5       : {'PASS' if consistent else 'FAIL'}  "
          f"({', '.join(f'{a:.3f}' for a in best['aucs'])})")

    # ── Final fit on ALL training sessions ────────────────────────────────────
    print(f"\n  Retraining on all {len(df_train)} training sessions "
          f"for {best['median_epoch']} epochs...")
    final = train_model(data, all_pairs, None, best["lr"],
                        fixed_epochs=best["median_epoch"], tag="final")
    history.extend(final["history"])
    fit_loss, fit_auc = evaluate(final["model"], data, all_pairs)
    print(f"    Final in-sample loss       : {fit_loss:.4f}")
    print(f"    Final in-sample AUC        : {fit_auc:.3f}   "
          f"(in-sample; CV AUC above is the honest number)")

    # ── Save ──────────────────────────────────────────────────────────────────
    os.makedirs(MODEL_DIR, exist_ok=True)
    m = final["model"]
    torch.save({
        "state_dict": final["state"],
        "config": {
            "in_dim": NODE_FEATURE_DIM, "hidden_dim": m.hidden_dim,
            "embed_dim": m.embed_dim, "dropout": m.dropout,
            "normalize_embeddings": m.normalize_embeddings,
            "use_residual": m.use_residual,
        },
        "training": {
            "lr": best["lr"], "weight_decay": WEIGHT_DECAY,
            "epochs": best["median_epoch"], "seed": SEED, "n_folds": N_FOLDS,
            "cv_auc_mean": best["mean_auc"], "cv_auc_std": best["std_auc"],
            "cv_fold_aucs": best["aucs"].tolist(),
            "in_sample_auc": fit_auc,
        },
        "node_to_idx": node_to_idx,
    }, MODEL_PATH)
    pd.DataFrame(history).to_csv(HISTORY_PATH, index=False)

    print(f"\n  Saved model   -> {MODEL_PATH}")
    print(f"  Saved history -> {HISTORY_PATH}")
    print("\n" + "=" * 70)
    print("  [OK] Training complete. Test set still untouched (Step 5).")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()