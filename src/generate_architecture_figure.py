"""
MindPulse — Module 3  |  Report Figure: Architecture Diagram (§5.3.3)
=====================================================================
Generates the Module 3 architecture diagram for Chapter 5 of the final report.

Drawn to match the visual register of Module 01's architecture figure in
§5.3.1 — plain rectangles, directed arrows, no colour fills beyond light
greys, so the two figures sit together in the same chapter without one
looking like it came from a different document.

The diagram makes three things explicit that prose alone conveys poorly:

  1. The SPLIT between what Layer 1 decides (feasibility) and what Layers 2-3
     decide (order). This is Module 3's central design commitment and the
     reason no learned component can surface an unsafe intervention.
  2. The SUPPRESSION path, which exits before any learned layer runs.
  3. The SHARED CANDIDATE POOL — all three layers rank the same set.

Usage
-----
    python src/generate_architecture_figure.py

Output
------
    outputs/figures/fig_m3_architecture.png   (300 dpi)

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch

FIG_DIR = os.path.join("outputs", "figures")
OUT = os.path.join(FIG_DIR, "fig_m3_architecture.png")

# Muted palette; Module 01's figure is essentially greyscale line-art, so
# colour here is used only to group the three layers, never for decoration.
C_INPUT = "#EDEDED"
C_L1 = "#DCE6F1"
C_L2 = "#FCE4D6"
C_L3 = "#E2EFDA"
C_OUT = "#EDEDED"
C_SUPP = "#F5F5F5"
EDGE = "#444444"

plt.rcParams.update({
    "font.size": 8.4,
    "font.family": "sans-serif",
})


def box(ax, x, y, w, h, text, fc, fontsize=8.4, weight="normal", style="round,pad=0.02"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle=style,
                                facecolor=fc, edgecolor=EDGE, linewidth=0.9))
    ax.text(x + w / 2, y + h / 2, text, ha="center", va="center",
            fontsize=fontsize, fontweight=weight, linespacing=1.35)


def arrow(ax, x1, y1, x2, y2, style="-|>", ls="-", lw=1.0, color=EDGE):
    ax.add_patch(FancyArrowPatch((x1, y1), (x2, y2), arrowstyle=style,
                                 mutation_scale=11, linewidth=lw,
                                 linestyle=ls, color=color,
                                 shrinkA=0, shrinkB=0))


def label(ax, x, y, text, fontsize=7.6, style="italic", color="#333333", ha="center"):
    ax.text(x, y, text, ha=ha, va="center", fontsize=fontsize,
            style=style, color=color, linespacing=1.3)


def main():
    fig, ax = plt.subplots(figsize=(9.2, 10.4))
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 11.6)
    ax.axis("off")

    CX, W = 2.55, 4.9          # main column x-position and width

    # ── Inputs ────────────────────────────────────────────────────────────────
    box(ax, 0.55, 10.75, 4.0, 0.62,
        "Module 1 — Stress Profile\n(tier, confidence, baseline deviation,\n"
        "signal quality, gesture_events)", C_INPUT, fontsize=7.8)
    box(ax, 5.45, 10.75, 4.0, 0.62,
        "Module 2 — Context  (optional)\n(trigger, location, social,\n"
        "dialogue_completed, kg_node_id)", C_INPUT, fontsize=7.8)

    arrow(ax, 2.55, 10.75, 3.6, 10.28)
    arrow(ax, 7.45, 10.75, 6.4, 10.28)

    # ── Input adaptation ──────────────────────────────────────────────────────
    box(ax, CX, 9.66, W, 0.62,
        "Input Adaptation\nflatten gesture_events → support signal;\n"
        "unknown-context fallback when dialogue incomplete", C_INPUT, fontsize=7.8)
    arrow(ax, 5.0, 9.66, 5.0, 9.16)

    # ── Layer 1 ───────────────────────────────────────────────────────────────
    box(ax, 1.95, 7.30, 6.1, 1.86, "", C_L1)
    ax.text(2.15, 8.94, "LAYER 1 — Rule-Based Knowledge Graph Recommender",
            fontsize=8.6, fontweight="bold", va="center")
    label(ax, 2.15, 8.66, "decides WHICH interventions are permissible", ha="left")

    box(ax, 2.25, 7.94, 2.6, 0.56,
        "Domain Knowledge Graph\n46 nodes · 488 weighted edges", "#FFFFFF", fontsize=7.6)
    box(ax, 5.15, 7.94, 2.6, 0.56,
        "Eight Sequential Gating Rules\nconfidence · signal · cold-start ·\n"
        "tier · feasibility · support · urgency", "#FFFFFF", fontsize=6.9)
    arrow(ax, 4.85, 8.22, 5.15, 8.22)

    box(ax, 2.85, 7.44, 5.3, 0.40,
        "Feasible candidate pool  —  mean 11.6 interventions (min 4, max 15)",
        "#FFFFFF", fontsize=7.6)

    # Suppression exit
    box(ax, 8.35, 7.48, 1.5, 0.66,
        "SUPPRESSED\nno recommendation\n(poor signal /\ncalm tier)",
        C_SUPP, fontsize=7.0)
    arrow(ax, 8.05, 7.81, 8.35, 7.81, ls="--")
    label(ax, 9.10, 7.16, "13 of 125 held-out\nsessions", fontsize=6.8)

    arrow(ax, 5.0, 7.30, 5.0, 6.80)

    # ── Layer 2 ───────────────────────────────────────────────────────────────
    box(ax, 1.95, 5.28, 6.1, 1.52, "", C_L2)
    ax.text(2.15, 6.58, "LAYER 2 — GCN Re-Ranker",
            fontsize=8.6, fontweight="bold", va="center")
    label(ax, 2.15, 6.32, "decides the ORDER of the permitted set", ha="left")

    box(ax, 2.25, 5.42, 2.35, 0.72,
        "KG → PyTorch Geometric\n(46 × 18) node features\n976 bidirectional edges",
        "#FFFFFF", fontsize=7.2)
    box(ax, 4.85, 5.42, 3.0, 0.72,
        "2-layer GCN + residual\n18 → 64 → 32 · 3,874 params\n"
        "mean-pool 5 context nodes · dot product",
        "#FFFFFF", fontsize=7.2)
    arrow(ax, 4.60, 5.78, 4.85, 5.78)

    arrow(ax, 5.0, 5.28, 5.0, 4.78)

    # ── Layer 3 ───────────────────────────────────────────────────────────────
    box(ax, 1.95, 3.26, 6.1, 1.52, "", C_L3)
    ax.text(2.15, 4.56, "LAYER 3 — ANN Personalisation",
            fontsize=8.6, fontweight="bold", va="center")
    label(ax, 2.15, 4.30, "adapts the order to the individual user", ha="left")

    box(ax, 2.25, 3.40, 2.35, 0.72,
        "Leak-safe user history\nrecency-weighted residual\n"
        "intervention → category → none",
        "#FFFFFF", fontsize=7.2)
    box(ax, 4.85, 3.40, 3.0, 0.72,
        "MLP residual predictor\n23 features → 8 → 1\n"
        "score = L2 + β · predicted residual",
        "#FFFFFF", fontsize=7.2)
    arrow(ax, 4.60, 3.76, 4.85, 3.76)

    arrow(ax, 5.0, 3.26, 5.0, 2.76)

    # ── Explanation ───────────────────────────────────────────────────────────
    box(ax, CX, 2.14, W, 0.62,
        "Explanation — Exact Context Attribution\n"
        "score decomposes exactly into one signed term per context node",
        C_INPUT, fontsize=7.8)
    arrow(ax, 5.0, 2.14, 5.0, 1.64)

    # ── Output ────────────────────────────────────────────────────────────────
    box(ax, CX, 0.92, W, 0.72,
        "Module 3 Recommendation JSON\n"
        "intervention · layer1/2/3 scores · explanation ·\n"
        "context snapshot · outcome_check_due (+30 min) · disclaimer",
        C_OUT, fontsize=7.8)

    # ── Feedback loop ─────────────────────────────────────────────────────────
    box(ax, 0.30, 1.62, 1.42, 0.62,
        "Outcome record\nat +30 min\nrating · tier_delta", C_SUPP, fontsize=7.0)
    arrow(ax, 2.55, 1.28, 1.72, 1.60, style="-|>", ls="--")
    arrow(ax, 1.01, 2.24, 1.01, 3.76, ls="--")
    arrow(ax, 1.01, 3.76, 2.25, 3.76, ls="--")
    label(ax, 0.62, 3.06, "feedback\nloop", fontsize=7.0)

    # ── Side annotation: the shared pool ──────────────────────────────────────
    ax.plot([8.22, 8.62, 8.62, 8.22], [7.44, 7.44, 3.40, 3.40],
            color="#888888", lw=0.9)
    ax.text(8.74, 5.42,
            "All three layers rank\nthe SAME candidate pool.\n\n"
            "Layer 1 sets feasibility;\nLayers 2 and 3 change\nonly the order.",
            fontsize=7.2, va="center", ha="left", style="italic",
            color="#333333", linespacing=1.45)

    os.makedirs(FIG_DIR, exist_ok=True)
    fig.savefig(OUT, dpi=300, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"saved -> {OUT}")


if __name__ == "__main__":
    main()