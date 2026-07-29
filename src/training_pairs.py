"""
MindPulse — Module 3  |  Layer 2: Training Pair Construction
=============================================================
Converts the 700-session synthetic dataset into positive and negative
link-prediction training pairs for the Layer 2 GCN.

THE LEARNING TASK
-----------------
Layer 2 learns to score (stress-context → intervention) links. For each
historical session we know which intervention was delivered and how well it
worked, so we can teach the model which links should score highly.

Each training pair is:
    (context node indices, intervention node index, label)

The "context" is not a single node — a stress episode activates one node of
each context type simultaneously. Each session therefore contributes FIVE
context node indices:

    StressState:{tier}
    TriggerContext:{trigger_type}
    LocationContext:{location_context}
    SocialContext:{social_context}
    GestureProfile:{support_detected ? support_detected : no_support}

The GCN model (Step 3) mean-pools the embeddings of these five nodes into a
single context embedding, then scores it against each intervention embedding
via dot product.

POSITIVE / NEGATIVE DEFINITION
------------------------------
The dataset's outcome ratings are heavily concentrated at the neutral value
(rating 3 accounts for ~51% of sessions). Treating "neutral" as either a
success or a failure would inject a large amount of label noise, so this
module uses a three-way split instead:

  • Positive      — delivered AND outcome_rating >= POSITIVE_THRESHOLD (4)
                    The intervention demonstrably helped. Label 1.0.
  • Hard negative — delivered AND outcome_rating <= NEGATIVE_THRESHOLD (2)
                    The intervention was tried in this exact context and
                    demonstrably did NOT help. These are the most informative
                    negatives available — far more useful than random ones,
                    because the model learns a real distinction rather than
                    just "delivered vs not delivered". Label 0.0.
  • Excluded      — outcome_rating == 3. Genuinely ambiguous; excluded rather
                    than guessed at.

  • Random negative — an intervention NOT delivered in this session, sampled
                    per positive. Standard link-prediction practice: without
                    these, the model only ever sees interventions that were
                    actually chosen and cannot learn to rank the full library.
                    Label 0.0.

Thresholds are module-level constants so they can be relaxed (e.g. positives
at >= 3) if training turns out to be data-starved — see PAIR_STATS output.

TRAIN / TEST SPLIT
------------------
The split replicates layer1_evaluation.py EXACTLY (same seed, same per-user
80/20 stratification, same iteration order). This is essential: Layer 2 must
be evaluated on the identical held-out sessions as Layer 1, or the
"Layer 2 > Layer 1" comparison in the progressive-improvement narrative would
be measuring different test sets rather than genuine improvement.

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import numpy as np
import pandas as pd
import torch

from knowledge_graph import build_knowledge_graph
from graph_converter import convert_kg_to_pyg


# ── Configuration ─────────────────────────────────────────────────────────────
DATASET_PATH = "data/synthetic/module3_dataset.csv"

POSITIVE_THRESHOLD = 4    # outcome_rating >= this → positive pair
NEGATIVE_THRESHOLD = 2    # outcome_rating <= this → hard negative pair
                          # (rating == 3 is excluded as ambiguous)

NEG_SAMPLES_PER_POSITIVE = 2   # random negatives drawn per positive pair

# Must match layer1_evaluation.py exactly — see module docstring
TRAIN_RATIO = 0.80
RANDOM_SEED = 42

# The five context node types active in every session, in fixed order
CONTEXT_NODE_TYPES = [
    "StressState",
    "TriggerContext",
    "LocationContext",
    "SocialContext",
    "GestureProfile",
]
NUM_CONTEXT_NODES = len(CONTEXT_NODE_TYPES)


# ══════════════════════════════════════════════════════════════════════════════
# TRAIN / TEST SPLIT  (must stay identical to layer1_evaluation.py)
# ══════════════════════════════════════════════════════════════════════════════

def replicate_train_test_split(df: pd.DataFrame) -> np.ndarray:
    """
    Reproduce layer1_evaluation.py's exact train/test split.

    Returns a boolean mask, True where the row belongs to the TEST set.

    This deliberately duplicates the logic in layer1_evaluation.py rather than
    importing it, because that function is embedded inside run_evaluation().
    If either copy is ever changed, BOTH must be updated — otherwise Layer 1
    and Layer 2 would be evaluated on different held-out sessions and the
    comparison between them would be invalid.
    """
    np.random.seed(RANDOM_SEED)
    test_mask = np.zeros(len(df), dtype=bool)
    for user_id in df["user_id"].unique():
        user_idx = df[df["user_id"] == user_id].index.tolist()
        n_test   = max(1, int(len(user_idx) * (1 - TRAIN_RATIO)))
        test_idx = np.random.choice(user_idx, size=n_test, replace=False)
        test_mask[test_idx] = True
    return test_mask


# ══════════════════════════════════════════════════════════════════════════════
# CONTEXT NODE RESOLUTION
# ══════════════════════════════════════════════════════════════════════════════

def get_context_node_indices(row: pd.Series, node_to_idx: dict) -> list:
    """
    Resolve a session row into its five active context node indices.

    Applies the SAME Module 2 fallback logic as rule_engine.py: when
    dialogue_completed is False, trigger/location/social are all treated as
    "unknown" rather than using whatever value happens to sit in the column.
    Keeping this consistent with Layer 1 matters — otherwise Layer 2 would be
    learning from context the live system would never actually have.
    """
    dialogue_completed = bool(row["dialogue_completed"])

    tier = row["tier"]

    if dialogue_completed:
        trigger  = row.get("trigger_type",     "unknown")
        location = row.get("location_context", "unknown")
        social   = row.get("social_context",   "unknown")
    else:
        trigger, location, social = "unknown", "unknown", "unknown"

    gesture = "support_detected" if bool(row["support_detected"]) else "no_support"

    node_names = [
        f"StressState:{tier}",
        f"TriggerContext:{trigger}",
        f"LocationContext:{location}",
        f"SocialContext:{social}",
        f"GestureProfile:{gesture}",
    ]

    indices = []
    for name in node_names:
        if name not in node_to_idx:
            raise KeyError(
                f"Context node '{name}' not found in the Knowledge Graph. "
                f"This usually means the dataset contains a value the KG has no "
                f"node for — check that dataset_generator.py and knowledge_graph.py "
                f"are in sync."
            )
        indices.append(node_to_idx[name])

    return indices


# ══════════════════════════════════════════════════════════════════════════════
# TRAINING PAIR CONSTRUCTION
# ══════════════════════════════════════════════════════════════════════════════

def build_training_pairs(df: pd.DataFrame, node_to_idx: dict,
                          seed: int = RANDOM_SEED) -> dict:
    """
    Build link-prediction training pairs from a set of session rows.

    Parameters
    ----------
    df : pd.DataFrame
        Session rows (already filtered to train or test split).
    node_to_idx : dict
        KG node name → tensor index, from graph_converter.convert_kg_to_pyg()
    seed : int
        Seed for reproducible negative sampling.

    Returns
    -------
    dict with:
      context_indices     : LongTensor (num_pairs, 5)
      intervention_indices: LongTensor (num_pairs,)
      labels              : FloatTensor (num_pairs,)
      stats               : dict of counts, for the summary report
    """
    rng = np.random.default_rng(seed)

    # All intervention node indices, for random negative sampling
    all_intervention_nodes = sorted(
        [name for name in node_to_idx if name.startswith("Intervention:")]
    )
    all_intervention_idx = np.array([node_to_idx[n] for n in all_intervention_nodes])

    context_rows, intervention_rows, label_rows = [], [], []

    n_positive      = 0
    n_hard_negative = 0
    n_random_negative = 0
    n_excluded      = 0

    for _, row in df.iterrows():
        rating = int(row["outcome_rating"])

        # ── Ambiguous neutral rating → excluded entirely ──────────────────────
        if NEGATIVE_THRESHOLD < rating < POSITIVE_THRESHOLD:
            n_excluded += 1
            continue

        ctx_idx     = get_context_node_indices(row, node_to_idx)
        delivered   = f"Intervention:{row['intervention_id']}"
        delivered_i = node_to_idx[delivered]

        if rating >= POSITIVE_THRESHOLD:
            # ── Positive pair ──────────────────────────────────────────────────
            context_rows.append(ctx_idx)
            intervention_rows.append(delivered_i)
            label_rows.append(1.0)
            n_positive += 1

            # ── Random negatives, drawn only for positives ────────────────────
            # (Drawing negatives for hard-negative rows too would over-weight
            #  the negative class; positives are the scarcer signal here.)
            candidates = all_intervention_idx[all_intervention_idx != delivered_i]
            n_draw = min(NEG_SAMPLES_PER_POSITIVE, len(candidates))
            sampled = rng.choice(candidates, size=n_draw, replace=False)
            for neg_i in sampled:
                context_rows.append(ctx_idx)
                intervention_rows.append(int(neg_i))
                label_rows.append(0.0)
                n_random_negative += 1

        else:
            # ── Hard negative: delivered in this context, but it didn't work ──
            context_rows.append(ctx_idx)
            intervention_rows.append(delivered_i)
            label_rows.append(0.0)
            n_hard_negative += 1

    pairs = {
        "context_indices":      torch.tensor(context_rows, dtype=torch.long),
        "intervention_indices": torch.tensor(intervention_rows, dtype=torch.long),
        "labels":               torch.tensor(label_rows, dtype=torch.float),
        "stats": {
            "sessions_seen":     len(df),
            "sessions_excluded": n_excluded,
            "positives":         n_positive,
            "hard_negatives":    n_hard_negative,
            "random_negatives":  n_random_negative,
            "total_pairs":       len(label_rows),
        },
    }
    return pairs


# ══════════════════════════════════════════════════════════════════════════════
# SUMMARY / VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def print_pair_summary(train_pairs: dict, test_pairs: dict,
                        df: pd.DataFrame, test_mask: np.ndarray) -> None:
    """Print a structured summary confirming the pairs were built correctly."""
    print("\n" + "═" * 65)
    print("  LAYER 2 — TRAINING PAIR CONSTRUCTION SUMMARY")
    print("═" * 65)

    print(f"\n  Dataset split (matches layer1_evaluation.py exactly):")
    print(f"    Total sessions         : {len(df)}")
    print(f"    Train sessions         : {(~test_mask).sum()}")
    print(f"    Test sessions          : {test_mask.sum()}")

    for name, pairs in [("TRAIN", train_pairs), ("TEST", test_pairs)]:
        s = pairs["stats"]
        print(f"\n  {name} pairs:")
        print(f"    Sessions seen          : {s['sessions_seen']}")
        print(f"    Excluded (rating == 3) : {s['sessions_excluded']}"
              f"  ({s['sessions_excluded']/max(s['sessions_seen'],1)*100:.0f}% — ambiguous)")
        print(f"    Positives   (label 1)  : {s['positives']}")
        print(f"    Hard negatives (0)     : {s['hard_negatives']}")
        print(f"    Random negatives (0)   : {s['random_negatives']}")
        print(f"    ─────────────────────────────")
        print(f"    Total pairs            : {s['total_pairs']}")
        if s["total_pairs"] > 0:
            pos_frac = s["positives"] / s["total_pairs"]
            print(f"    Positive class balance : {pos_frac:.1%}")

    print(f"\n  Tensor shapes (TRAIN):")
    print(f"    context_indices        : {tuple(train_pairs['context_indices'].shape)}")
    print(f"    intervention_indices   : {tuple(train_pairs['intervention_indices'].shape)}")
    print(f"    labels                 : {tuple(train_pairs['labels'].shape)}")

    # ── Consistency checks ─────────────────────────────────────────────────────
    print(f"\n  Consistency checks:")

    ci = train_pairs["context_indices"]
    ii = train_pairs["intervention_indices"]
    lb = train_pairs["labels"]

    shape_ok = ci.shape[0] == ii.shape[0] == lb.shape[0]
    print(f"    All tensors same length            : {'✓ PASS' if shape_ok else '✗ FAIL'}")

    ctx_ok = ci.shape[1] == NUM_CONTEXT_NODES
    print(f"    5 context nodes per pair           : {'✓ PASS' if ctx_ok else '✗ FAIL'}")

    label_ok = bool(((lb == 0.0) | (lb == 1.0)).all().item())
    print(f"    Labels are strictly 0.0 or 1.0     : {'✓ PASS' if label_ok else '✗ FAIL'}")

    has_both = bool((lb == 1.0).any().item() and (lb == 0.0).any().item())
    print(f"    Both classes present               : {'✓ PASS' if has_both else '✗ FAIL'}")

    # No train/test leakage: the split is by session, so verify disjointness
    # by confirming the counts add up rather than comparing rows directly.
    total_sessions = train_pairs["stats"]["sessions_seen"] + test_pairs["stats"]["sessions_seen"]
    no_leak = total_sessions == len(df)
    print(f"    Train + test = all sessions        : {'✓ PASS' if no_leak else '✗ FAIL'}"
          f"  ({total_sessions} vs {len(df)})")

    # ── Data sufficiency warning ───────────────────────────────────────────────
    n_pos_train = train_pairs["stats"]["positives"]
    print(f"\n  Data sufficiency:")
    if n_pos_train < 100:
        print(f"    ⚠  Only {n_pos_train} training positives — this is thin.")
        print(f"       If the GCN underfits, consider lowering POSITIVE_THRESHOLD")
        print(f"       from {POSITIVE_THRESHOLD} to 3 to include neutral-rated sessions.")
    else:
        print(f"    ✓  {n_pos_train} training positives — sufficient for a graph this size.")

    print("\n" + "═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# PUBLIC ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

def load_training_pairs(dataset_path: str = DATASET_PATH, verbose: bool = True) -> tuple:
    """
    Full pipeline: load dataset → build KG → convert to PyG → build pairs.

    Returns
    -------
    (train_pairs, test_pairs, data, node_to_idx, idx_to_node, df, test_mask)
    """
    if not os.path.exists(dataset_path):
        raise FileNotFoundError(
            f"Dataset not found at {dataset_path}. Run src/dataset_generator.py first."
        )

    df = pd.read_csv(dataset_path)

    G = build_knowledge_graph()
    data, node_to_idx, idx_to_node = convert_kg_to_pyg(G, make_undirected=True)

    test_mask = replicate_train_test_split(df)
    df_train = df[~test_mask].copy()
    df_test  = df[test_mask].copy()

    train_pairs = build_training_pairs(df_train, node_to_idx, seed=RANDOM_SEED)
    test_pairs  = build_training_pairs(df_test,  node_to_idx, seed=RANDOM_SEED + 1)

    if verbose:
        print_pair_summary(train_pairs, test_pairs, df, test_mask)

    return train_pairs, test_pairs, data, node_to_idx, idx_to_node, df, test_mask


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE TEST
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Loading dataset and building training pairs...")
    (train_pairs, test_pairs, data,
     node_to_idx, idx_to_node, df, test_mask) = load_training_pairs()

    # ── Show a concrete example pair, for eyeball verification ─────────────────
    print("  Example TRAIN pair (first positive):")
    labels = train_pairs["labels"]
    pos_positions = (labels == 1.0).nonzero(as_tuple=True)[0]
    if len(pos_positions) > 0:
        i = pos_positions[0].item()
        ctx = train_pairs["context_indices"][i].tolist()
        inv = train_pairs["intervention_indices"][i].item()
        print(f"    Context nodes : {[idx_to_node[c] for c in ctx]}")
        print(f"    Intervention  : {idx_to_node[inv]}")
        print(f"    Label         : {labels[i].item()}  (positive — this worked)")

    print("\n  Example TRAIN pair (first hard negative):")
    neg_positions = (labels == 0.0).nonzero(as_tuple=True)[0]
    if len(neg_positions) > 0:
        i = neg_positions[0].item()
        ctx = train_pairs["context_indices"][i].tolist()
        inv = train_pairs["intervention_indices"][i].item()
        print(f"    Context nodes : {[idx_to_node[c] for c in ctx]}")
        print(f"    Intervention  : {idx_to_node[inv]}")
        print(f"    Label         : {labels[i].item()}  (negative)")

    print("\n[✓] Training pairs are ready for GCN training (Layer 2 Step 3).\n")