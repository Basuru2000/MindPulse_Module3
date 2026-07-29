"""
MindPulse — Module 3  |  Layer 1: Knowledge Graph
==================================================
Constructs the domain Knowledge Graph (KG) used as the structural
substrate for both Layer 1 rule-based filtering and Layer 2 GCN training.

The KG encodes clinical and behavioural domain knowledge as a directed,
weighted graph of typed nodes and edges. Edge weights reflect clinical
appropriateness of relationships between stress states, contexts,
gesture profiles, and evidence-based interventions.

Node types (6):
  • StressState      — calm, mild, moderate, acute
  • TriggerContext   — academic, interpersonal, financial, health, other, unknown
  • LocationContext  — home, university, work, in_transit, other, unknown
  • SocialContext    — alone, colleagues, friends, family, other, unknown
  • GestureProfile   — support_detected, no_support
  • Intervention     — 22 evidence-based interventions (I01–I22)

  CHANGE LOG (aligned to updated Module 1 / Module 2 specs):
  - GestureProfile: previously 4 nodes (scratch_dominant, fidget_dominant,
    walk_dominant, still), based on a gesture taxonomy Module 1 no longer
    produces. Module 1's revised design outputs a single behavioural-support
    signal (detected/not detected + support_score) rather than classifying
    gesture type. GestureProfile is now 2 nodes: support_detected, no_support.
    The node_type name "GestureProfile" is kept for naming consistency across
    the codebase; only the possible node values changed.
  - LocationContext / SocialContext: added an "unknown" node to each, since
    Module 2's dialogue can complete (dialogue_completed=true) while location
    and/or social context remain unanswered (these are optional questions,
    asked at most once per day per Module 2's spec).

Reference: Module 3 Technical Specification, Section 2.1 & Section 3
Author   : Module 3 — MindPulse (Team MindForge)
"""

import networkx as nx


# ══════════════════════════════════════════════════════════════════════════════
# INTERVENTION LIBRARY  (mirrors dataset_generator.py exactly — unchanged)
# ══════════════════════════════════════════════════════════════════════════════

INTERVENTIONS = [
    {"id": "I01", "name": "4-7-8 Breathing",                   "type": "breathing",  "tiers": ["mild", "moderate"],         "duration": 2, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I02", "name": "Box Breathing (4-4-4-4)",            "type": "breathing",  "tiers": ["moderate", "acute"],        "duration": 3, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I03", "name": "Diaphragmatic Breathing",            "type": "breathing",  "tiers": ["mild","moderate","acute"],  "duration": 2, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I04", "name": "5-Minute Brisk Walk",                "type": "physical",   "tiers": ["mild", "moderate"],         "duration": 5, "excludes_social": [],                                 "excludes_location": ["in_transit", "work"]},
    {"id": "I05", "name": "Progressive Muscle Relaxation",      "type": "physical",   "tiers": ["moderate", "acute"],        "duration": 5, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I06", "name": "Cold Water Face Splash",             "type": "physical",   "tiers": ["acute"],                    "duration": 1, "excludes_social": [],                                 "excludes_location": ["in_transit"]},
    {"id": "I07", "name": "Grounding Technique (5-4-3-2-1)",   "type": "cognitive",  "tiers": ["acute"],                    "duration": 3, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I08", "name": "Cognitive Reframing (3 Good Things)","type": "cognitive",  "tiers": ["mild"],                     "duration": 3, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I09", "name": "Worry Postponement",                 "type": "cognitive",  "tiers": ["mild", "moderate"],         "duration": 3, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I10", "name": "Brief Mindfulness Check-in",         "type": "cognitive",  "tiers": ["mild", "moderate"],         "duration": 2, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I11", "name": "Social Contact Prompt",              "type": "social",     "tiers": ["mild", "moderate"],         "duration": 2, "excludes_social": ["colleagues","friends","family"],   "excludes_location": []},
    {"id": "I12", "name": "Gratitude Message Prompt",           "type": "social",     "tiers": ["mild"],                     "duration": 2, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I13", "name": "Nature Sound / White Noise",         "type": "sensory",    "tiers": ["moderate"],                 "duration": 5, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I14", "name": "Hydration Reminder",                 "type": "physical",   "tiers": ["mild"],                     "duration": 1, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I15", "name": "Stretching (desk-based)",            "type": "physical",   "tiers": ["mild", "moderate"],         "duration": 3, "excludes_social": [],                                 "excludes_location": ["in_transit"]},
    {"id": "I16", "name": "Single-Focus Task Prompt",           "type": "cognitive",  "tiers": ["moderate"],                 "duration": 3, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I17", "name": "Visual Focus Break (20-20-20)",      "type": "physical",   "tiers": ["mild"],                     "duration": 1, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I18", "name": "Journalling Micro-Prompt",           "type": "cognitive",  "tiers": ["mild"],                     "duration": 3, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I19", "name": "Self-Compassion Pause",              "type": "cognitive",  "tiers": ["mild","moderate","acute"],  "duration": 2, "excludes_social": [],                                 "excludes_location": []},
    {"id": "I20", "name": "Power Posture Reset",                "type": "physical",   "tiers": ["mild", "moderate"],         "duration": 1, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I21", "name": "Rhythmic Tapping (EFT-lite)",        "type": "physical",   "tiers": ["moderate", "acute"],        "duration": 2, "excludes_social": ["colleagues"],                     "excludes_location": []},
    {"id": "I22", "name": "Sleep Hygiene Reminder",             "type": "cognitive",  "tiers": ["mild"],                     "duration": 1, "excludes_social": [],                                 "excludes_location": [], "requires_time": ["evening", "night"]},
]

INTERVENTION_MAP = {inv["id"]: inv for inv in INTERVENTIONS}

# ── Clinical tier–intervention appropriateness weights (unchanged) ─────────────
TIER_INTERVENTION_WEIGHTS = {
    ("mild",     "breathing"):  0.70,
    ("mild",     "cognitive"):  0.85,
    ("mild",     "physical"):   0.75,
    ("mild",     "social"):     0.80,
    ("mild",     "sensory"):    0.60,
    ("moderate", "breathing"):  0.85,
    ("moderate", "cognitive"):  0.70,
    ("moderate", "physical"):   0.75,
    ("moderate", "social"):     0.60,
    ("moderate", "sensory"):    0.65,
    ("acute",    "breathing"):  0.95,
    ("acute",    "cognitive"):  0.60,
    ("acute",    "physical"):   0.75,
    ("acute",    "social"):     0.40,
    ("acute",    "sensory"):    0.50,
}

# ── Trigger–intervention affinity weights (unchanged) ──────────────────────────
TRIGGER_INTERVENTION_WEIGHTS = {
    ("academic",      "cognitive"):  0.90,
    ("academic",      "breathing"):  0.80,
    ("academic",      "physical"):   0.65,
    ("academic",      "social"):     0.50,
    ("academic",      "sensory"):    0.55,
    ("interpersonal", "social"):     0.85,
    ("interpersonal", "cognitive"):  0.75,
    ("interpersonal", "breathing"):  0.70,
    ("interpersonal", "physical"):   0.60,
    ("interpersonal", "sensory"):    0.55,
    ("financial",     "cognitive"):  0.80,
    ("financial",     "breathing"):  0.75,
    ("financial",     "physical"):   0.60,
    ("financial",     "social"):     0.65,
    ("financial",     "sensory"):    0.50,
    ("health",        "breathing"):  0.85,
    ("health",        "physical"):   0.70,
    ("health",        "cognitive"):  0.65,
    ("health",        "social"):     0.60,
    ("health",        "sensory"):    0.55,
    ("other",         "breathing"):  0.70,
    ("other",         "cognitive"):  0.70,
    ("other",         "physical"):   0.65,
    ("other",         "social"):     0.60,
    ("other",         "sensory"):    0.55,
    ("unknown",       "breathing"):  0.70,
    ("unknown",       "cognitive"):  0.65,
    ("unknown",       "physical"):   0.65,
    ("unknown",       "social"):     0.55,
    ("unknown",       "sensory"):    0.50,
}

# ── Location feasibility weights ────────────────────────────────────────────────
# UPDATED: added "unknown" row — same treatment as "other" (small neutral weight),
# used when Module 2 dialogue completes but the location question was skipped.
LOCATION_WEIGHTS = {
    ("home",        "breathing"):  0.10,
    ("home",        "physical"):   0.10,
    ("home",        "cognitive"):  0.10,
    ("home",        "social"):     0.10,
    ("home",        "sensory"):    0.10,
    ("university",  "breathing"):  0.05,
    ("university",  "cognitive"):  0.10,
    ("university",  "physical"):   0.05,
    ("university",  "social"):     0.05,
    ("university",  "sensory"):    0.03,
    ("work",        "cognitive"):  0.10,
    ("work",        "breathing"):  0.05,
    ("work",        "physical"):  -0.10,  # walk excluded at work context
    ("work",        "social"):     0.03,
    ("work",        "sensory"):    0.03,
    ("in_transit",  "cognitive"):  0.05,
    ("in_transit",  "breathing"):  0.05,
    ("in_transit",  "physical"):  -0.15,  # walk excluded in transit
    ("in_transit",  "social"):     0.02,
    ("in_transit",  "sensory"):    0.03,
    ("other",       "breathing"):  0.05,
    ("other",       "cognitive"):  0.05,
    ("other",       "physical"):   0.05,
    ("other",       "social"):     0.05,
    ("other",       "sensory"):    0.03,
    ("unknown",     "breathing"):  0.05,
    ("unknown",     "cognitive"):  0.05,
    ("unknown",     "physical"):   0.05,
    ("unknown",     "social"):     0.05,
    ("unknown",     "sensory"):    0.03,
}

# ── Social context feasibility weights ─────────────────────────────────────────
# UPDATED: added "unknown" row — same treatment as "other" (small neutral weight),
# used when Module 2 dialogue completes but the social-context question was skipped.
SOCIAL_WEIGHTS = {
    ("alone",      "breathing"):  0.10,
    ("alone",      "physical"):   0.10,
    ("alone",      "cognitive"):  0.10,
    ("alone",      "social"):     0.05,   # social prompt less relevant if already alone
    ("alone",      "sensory"):    0.10,
    ("colleagues", "cognitive"):  0.10,
    ("colleagues", "physical"):   0.05,
    ("colleagues", "breathing"):  0.00,   # audio breathing excluded with colleagues
    ("colleagues", "social"):     0.02,
    ("colleagues", "sensory"):    0.00,   # audio excluded with colleagues
    ("friends",    "breathing"):  0.08,
    ("friends",    "physical"):   0.10,
    ("friends",    "cognitive"):  0.08,
    ("friends",    "social"):     0.10,
    ("friends",    "sensory"):    0.05,
    ("family",     "breathing"):  0.08,
    ("family",     "physical"):   0.08,
    ("family",     "cognitive"):  0.08,
    ("family",     "social"):     0.10,
    ("family",     "sensory"):    0.05,
    ("other",      "breathing"):  0.05,
    ("other",      "physical"):   0.05,
    ("other",      "cognitive"):  0.08,
    ("other",      "social"):     0.05,
    ("other",      "sensory"):    0.05,
    ("unknown",    "breathing"):  0.05,
    ("unknown",    "physical"):   0.05,
    ("unknown",    "cognitive"):  0.08,
    ("unknown",    "social"):     0.05,
    ("unknown",    "sensory"):    0.05,
}


# ══════════════════════════════════════════════════════════════════════════════
# KNOWLEDGE GRAPH BUILDER
# ══════════════════════════════════════════════════════════════════════════════

def build_knowledge_graph() -> nx.DiGraph:
    """
    Construct and return the Module 3 Knowledge Graph (KG).

    The KG is a directed weighted graph (DiGraph). Node IDs use the format
    "NodeType:value" for unambiguous identification (e.g. "StressState:moderate",
    "Intervention:I02"). Each node carries a 'node_type' attribute for filtering.
    Each edge carries a 'weight' attribute (0.0–1.0) representing clinical
    appropriateness or feasibility of the relationship.

    Returns
    -------
    G : nx.DiGraph
        The fully constructed Knowledge Graph ready for Layer 1 rule traversal
        and Layer 2 GCN training.
    """
    G = nx.DiGraph()

    # ── 1. StressState nodes (unchanged) ───────────────────────────────────────
    stress_states = ["calm", "mild", "moderate", "acute"]
    for state in stress_states:
        G.add_node(f"StressState:{state}",
                   node_type="StressState",
                   tier=state,
                   tier_index=stress_states.index(state))

    # ── 2. TriggerContext nodes (unchanged — "unknown" was already present) ────
    trigger_types = ["academic", "interpersonal", "financial", "health", "other", "unknown"]
    for ttype in trigger_types:
        G.add_node(f"TriggerContext:{ttype}",
                   node_type="TriggerContext",
                   trigger_type=ttype)

    # ── 3. LocationContext nodes — UPDATED: added "unknown" ────────────────────
    locations = ["home", "university", "work", "in_transit", "other", "unknown"]
    for loc in locations:
        G.add_node(f"LocationContext:{loc}",
                   node_type="LocationContext",
                   location=loc)

    # ── 4. SocialContext nodes — UPDATED: added "unknown" ───────────────────────
    social_contexts = ["alone", "colleagues", "friends", "family", "other", "unknown"]
    for soc in social_contexts:
        G.add_node(f"SocialContext:{soc}",
                   node_type="SocialContext",
                   social=soc)

    # ── 5. GestureProfile nodes — REPLACED ──────────────────────────────────────
    # Previously 4 nodes (scratch/fidget/walk/still-dominant), matching a gesture
    # taxonomy Module 1 no longer produces. Module 1's revised design outputs a
    # single behavioural-support signal instead — detected or not, with a
    # support_score (0.0-1.0). The KG now reflects that with 2 nodes.
    #
    # Base weights below represent the MAXIMUM influence behavioural support can
    # have on each intervention category. The rule engine (Layer 1) scales
    # "support_detected" by the episode's actual support_score at runtime,
    # mirroring Module 1's own alpha-controlled, small-and-bounded confidence
    # boost philosophy — behavioural support nudges scoring, it does not drive it.
    gesture_profiles = ["support_detected", "no_support"]
    gesture_inv_affinity = {
        "support_detected": {"breathing": 0.80, "physical": 0.75, "cognitive": 0.50, "social": 0.40, "sensory": 0.55},
        "no_support":       {"breathing": 0.55, "physical": 0.50, "cognitive": 0.60, "social": 0.55, "sensory": 0.50},
    }
    for gp in gesture_profiles:
        G.add_node(f"GestureProfile:{gp}",
                   node_type="GestureProfile",
                   profile=gp,
                   inv_affinity=gesture_inv_affinity[gp])

    # ── 6. Intervention nodes (unchanged) ──────────────────────────────────────
    for inv in INTERVENTIONS:
        G.add_node(f"Intervention:{inv['id']}",
                   node_type="Intervention",
                   intervention_id=inv["id"],
                   intervention_name=inv["name"],
                   intervention_type=inv["type"],
                   target_tiers=inv["tiers"],
                   duration=inv["duration"],
                   excludes_social=inv.get("excludes_social", []),
                   excludes_location=inv.get("excludes_location", []),
                   requires_time=inv.get("requires_time", []))

    # ── 7. StressState → Intervention edges (tier-appropriateness, unchanged) ──
    for inv in INTERVENTIONS:
        for tier in inv["tiers"]:
            weight = TIER_INTERVENTION_WEIGHTS.get((tier, inv["type"]), 0.50)
            G.add_edge(f"StressState:{tier}",
                       f"Intervention:{inv['id']}",
                       weight=weight,
                       edge_type="TIER_APPROPRIATENESS")

    # ── 8. TriggerContext → Intervention edges (trigger affinity, unchanged) ───
    for inv in INTERVENTIONS:
        for trigger in trigger_types:
            weight = TRIGGER_INTERVENTION_WEIGHTS.get((trigger, inv["type"]), 0.55)
            G.add_edge(f"TriggerContext:{trigger}",
                       f"Intervention:{inv['id']}",
                       weight=weight,
                       edge_type="TRIGGER_AFFINITY")

    # ── 9. LocationContext → Intervention edges — now includes "unknown" ───────
    for inv in INTERVENTIONS:
        for loc in locations:
            if loc in inv.get("excludes_location", []):
                w = 0.0
            else:
                w = LOCATION_WEIGHTS.get((loc, inv["type"]), 0.05)
            G.add_edge(f"LocationContext:{loc}",
                       f"Intervention:{inv['id']}",
                       weight=w,
                       edge_type="LOCATION_FEASIBILITY")

    # ── 10. SocialContext → Intervention edges — now includes "unknown" ────────
    for inv in INTERVENTIONS:
        for soc in social_contexts:
            if soc in inv.get("excludes_social", []):
                w = 0.0
            else:
                w = SOCIAL_WEIGHTS.get((soc, inv["type"]), 0.05)
            G.add_edge(f"SocialContext:{soc}",
                       f"Intervention:{inv['id']}",
                       weight=w,
                       edge_type="SOCIAL_FEASIBILITY")

    # ── 11. GestureProfile → Intervention edges — REPLACED logic ───────────────
    for inv in INTERVENTIONS:
        for gp in gesture_profiles:
            affinity = gesture_inv_affinity[gp].get(inv["type"], 0.50)
            G.add_edge(f"GestureProfile:{gp}",
                       f"Intervention:{inv['id']}",
                       weight=affinity,
                       edge_type="GESTURE_URGENCY")

    # ── 12. StressState → TriggerContext context edges (unchanged) ─────────────
    trigger_tier_cooccurrence = {
        ("mild",     "academic"):      0.80,
        ("mild",     "interpersonal"): 0.70,
        ("mild",     "financial"):     0.60,
        ("mild",     "health"):        0.55,
        ("moderate", "academic"):      0.75,
        ("moderate", "interpersonal"): 0.75,
        ("moderate", "financial"):     0.70,
        ("moderate", "health"):        0.65,
        ("acute",    "interpersonal"): 0.80,
        ("acute",    "academic"):      0.70,
        ("acute",    "health"):        0.75,
        ("acute",    "financial"):     0.65,
    }
    for (tier, trigger), weight in trigger_tier_cooccurrence.items():
        G.add_edge(f"StressState:{tier}",
                   f"TriggerContext:{trigger}",
                   weight=weight,
                   edge_type="TIER_TRIGGER_COOCCURRENCE")

    return G


# ══════════════════════════════════════════════════════════════════════════════
# KG INSPECTION UTILITIES  (unchanged — generic, work with any node/edge types)
# ══════════════════════════════════════════════════════════════════════════════

def print_kg_summary(G: nx.DiGraph) -> None:
    """Print a structured summary of the constructed Knowledge Graph."""
    print("\n" + "═" * 65)
    print("  MODULE 3 KNOWLEDGE GRAPH — SUMMARY")
    print("═" * 65)

    node_type_counts = {}
    for node, data in G.nodes(data=True):
        ntype = data.get("node_type", "Unknown")
        node_type_counts[ntype] = node_type_counts.get(ntype, 0) + 1

    print(f"\n  Total nodes : {G.number_of_nodes()}")
    for ntype, count in sorted(node_type_counts.items()):
        print(f"    {ntype:<20} {count:3d} nodes")

    edge_type_counts = {}
    for u, v, data in G.edges(data=True):
        etype = data.get("edge_type", "Unknown")
        edge_type_counts[etype] = edge_type_counts.get(etype, 0) + 1

    print(f"\n  Total edges : {G.number_of_edges()}")
    for etype, count in sorted(edge_type_counts.items()):
        print(f"    {etype:<30} {count:4d} edges")

    print(f"\n  Top-5 StressState:acute → Intervention edges (by weight):")
    acute_edges = [
        (v, d["weight"]) for u, v, d in G.edges(data=True)
        if u == "StressState:acute" and d.get("edge_type") == "TIER_APPROPRIATENESS"
    ]
    for inv_node, weight in sorted(acute_edges, key=lambda x: -x[1])[:5]:
        inv_id   = G.nodes[inv_node]["intervention_id"]
        inv_name = G.nodes[inv_node]["intervention_name"]
        print(f"    {inv_id} — {inv_name:<40} weight={weight:.2f}")

    print("\n" + "═" * 65 + "\n")


def get_intervention_neighbors(G: nx.DiGraph, intervention_id: str) -> None:
    """Show all nodes connected to a given intervention (useful for debugging)."""
    node_key = f"Intervention:{intervention_id}"
    if node_key not in G:
        print(f"Intervention {intervention_id} not found in KG.")
        return
    inv = G.nodes[node_key]
    print(f"\n  {intervention_id} — {inv['intervention_name']} ({inv['intervention_type']})")
    in_edges = [(u, d["weight"], d["edge_type"]) for u, v, d in G.in_edges(node_key, data=True)]
    for u, w, etype in sorted(in_edges, key=lambda x: -x[1]):
        print(f"    ← {u:<35}  weight={w:.2f}  [{etype}]")


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT (standalone test)
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("Building Knowledge Graph...")
    G = build_knowledge_graph()
    print_kg_summary(G)

    print("  Spot-check — Intervention I02 (Box Breathing):")
    get_intervention_neighbors(G, "I02")
    print("\n  Spot-check — Intervention I07 (Grounding Technique):")
    get_intervention_neighbors(G, "I07")
    print()