"""
MindPulse — Module 3  |  Demo UI Builder
=========================================
Builds the two self-contained HTML pages used to demonstrate Module 3:

    outputs/demo_ui.html         the recommendation workflow
    outputs/demo_results.html    the evaluation and research findings

Both are single files with no server, no dependencies and no external
assets — figures are downscaled and embedded as base64, so the pages can be
copied anywhere and opened by double-clicking.

WHAT THIS DOES AND DOES NOT DO
-------------------------------
It adds no functionality. Every value on either page is read from an existing
artefact:

  * recommendations           module3_recommender.recommend()
  * layer comparison table    data/synthetic/layer2_evaluation_results*.csv
  * Layer 3 off-policy table  data/synthetic/layer3_online_results.csv
  * figures                   outputs/figures/*.png

Nothing is retyped into the page by hand. If a result changes, re-running the
evaluation and then this script updates the pages.

The one exception is the per-intervention exclusion reason shown in the
recommendation funnel: Layer 1 returns only the surviving pool, never the
reasons, so this script re-applies the same feasibility predicates in the
same order to describe the engine's decision. It is a description, not a
reimplementation — the pool itself is always whatever the engine returned.

Usage
-----
    python src/build_demo_ui.py

Author : Module 3 — MindPulse (Team MindForge)
"""

import os
import sys
import json
import base64
import io

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import pandas as pd

FIG_DIR = os.path.join("outputs", "figures")
SYN = os.path.join("data", "synthetic")
OUT_DIR = "outputs"
TPL_DIR = "src"

# Figures are 300 dpi for print; screens need far less. Downscaling keeps the
# self-contained pages a reasonable size without visible quality loss.
FIG_MAX_WIDTH = 1500


# ══════════════════════════════════════════════════════════════════════════════
# ASSETS
# ══════════════════════════════════════════════════════════════════════════════

def embed_figure(name: str) -> str:
    """Downscale a figure and return it as a base64 data URI."""
    path = os.path.join(FIG_DIR, name)
    if not os.path.exists(path):
        print(f"    [warn] missing figure: {name}")
        return ""
    try:
        from PIL import Image
        im = Image.open(path)
        if im.width > FIG_MAX_WIDTH:
            h = round(im.height * FIG_MAX_WIDTH / im.width)
            im = im.resize((FIG_MAX_WIDTH, h), Image.LANCZOS)
        buf = io.BytesIO()
        im.convert("RGB").save(buf, format="PNG", optimize=True)
        raw = buf.getvalue()
    except ImportError:
        raw = open(path, "rb").read()
    return "data:image/png;base64," + base64.b64encode(raw).decode("ascii")


# ══════════════════════════════════════════════════════════════════════════════
# RESULTS — read from files, never hardcoded
# ══════════════════════════════════════════════════════════════════════════════

def load_results() -> dict:
    per = pd.read_csv(os.path.join(SYN, "layer2_evaluation_results_per_session.csv"))
    diffs = pd.read_csv(os.path.join(SYN, "layer2_evaluation_results.csv"))
    online = pd.read_csv(os.path.join(SYN, "layer3_online_results.csv"))

    keys = [("Precision@3", "prec@3"), ("NDCG@3", "ndcg@3"), ("Hit Rate@3", "hit@3"),
            ("Precision@5", "prec@5"), ("NDCG@5", "ndcg@5"), ("Hit Rate@5", "hit@5"),
            ("Mean Outcome", "mor")]

    layers = []
    for label, k in keys:
        row = diffs[diffs["metric"] == label]
        layers.append({
            "metric": label,
            "cb": float(per[f"cb_{k}"].mean()),
            "l1": float(per[f"l1_{k}"].mean()),
            "l2": float(per[f"l2_{k}"].mean()),
            "diff": float(row.iloc[0]["difference"]) if len(row) else None,
            "lo": float(row.iloc[0]["ci_low"]) if len(row) else None,
            "hi": float(row.iloc[0]["ci_high"]) if len(row) else None,
            "sig": bool(len(row) and str(row.iloc[0]["significant"]).startswith("yes")),
        })

    ndcg3 = next(m for m in layers if m["metric"] == "NDCG@3")
    return {
        "n_sessions": int(len(per)),
        "layers": layers,
        "n_sig": int(sum(m["sig"] for m in layers)),
        "ndcg3_rel": round(ndcg3["diff"] / ndcg3["l1"] * 100),
        "ndcg3_l2": ndcg3["l2"],
        "online": online.to_dict("records"),
    }


# ══════════════════════════════════════════════════════════════════════════════
# BUILD
# ══════════════════════════════════════════════════════════════════════════════

def build(template: str, payload_name: str, payload, out_name: str):
    tpl = open(os.path.join(TPL_DIR, template), encoding="utf-8").read()
    marker = "__%s__" % payload_name
    assert marker in tpl, f"{marker} not found in {template}"
    html = tpl.replace(marker, json.dumps(payload, separators=(",", ":"), default=str))
    assert marker not in html
    path = os.path.join(OUT_DIR, out_name)
    open(path, "w", encoding="utf-8").write(html)
    print(f"    {out_name:<22} {len(html)//1024:>5} KB")
    return path


def main():
    print("\n" + "=" * 70)
    print("  MODULE 3 — DEMO UI BUILD")
    print("=" * 70)

    os.makedirs(OUT_DIR, exist_ok=True)

    # ── Page 1: recommendation workflow ───────────────────────────────────────
    print("\n  Recommendation workflow")
    scen_path = os.path.join(OUT_DIR, "demo_scenarios.json")
    if not os.path.exists(scen_path):
        print("    demo_scenarios.json not found — running generator...")
        import generate_demo_data
        generate_demo_data.main()
    scenarios = json.load(open(scen_path, encoding="utf-8"))
    print(f"    {len(scenarios)} scenarios loaded")
    build("demo_ui_template.html", "PAYLOAD", scenarios, "demo_ui.html")

    # ── Page 2: evaluation and findings ───────────────────────────────────────
    print("\n  Evaluation and findings")
    res = load_results()
    print(f"    {res['n_sessions']} held-out sessions · "
          f"{res['n_sig']} significant improvements · "
          f"NDCG@3 +{res['ndcg3_rel']}%")

    print("    embedding figures...")
    res["figures"] = {
        "layers":       embed_figure("fig1_progressive_improvement.png"),
        "oversmooth":   embed_figure("fig2_oversmoothing.png"),
        "training":     embed_figure("fig3_gcn_training.png"),
        "ips":          embed_figure("fig4_layer3_ips.png"),
        "power":        embed_figure("fig5_power_analysis.png"),
    }
    build("demo_results_template.html", "RESULTS", res, "demo_results.html")

    print("\n" + "=" * 70)
    print("  Open outputs/demo_ui.html to begin.")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    main()