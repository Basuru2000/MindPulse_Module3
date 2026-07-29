"""
MindPulse — Module 3  |  Layer 1: Rule Engine
==============================================
Applies the Layer 1 rule-based gating procedure over the Knowledge Graph
to produce a context-filtered, tier-appropriate, ranked candidate set of
interventions for each triggered stress episode.

Processing steps (per Module 3 spec Section 2.1):
  1. Confidence gating        — below 0.55 → conservative fallback
  2. Signal quality gating    — poor quality → no high-urgency push
  3. Cold-start handling      — no personal baseline → generic tier routing
  4. Dialogue fallback        — Module 2 unavailable → Module 1 only
  5. Tier-based routing       — activates StressState node, traverses edges
  6. Context feasibility      — excludes infeasible (location, social) interventions
  7. Behavioural support      — scales somatic vs cognitive priority weights
  8. Baseline deviation       — scales urgency (active vs passive nudge)
  9. Trigger affinity         — boosts trigger-aligned intervention types

  CHANGE LOG (aligned to updated Module 1 / Module 2 specs):
  - Behavioural support (was "gesture urgency"): Module 1's revised design no
    longer classifies gesture type (scratch/fidget/walk/still). It emits a
    single behavioural-support signal instead: whether repetitive movement was
    detected, plus a support_score (0.0-1.0). The rule engine now reads
    episode["support_detected"] and episode["support_score"], and scales the
    KG's base GestureProfile edge weight by the actual per-episode
    support_score — mirroring Module 1's own alpha-controlled, small-and-
    bounded confidence-boost design (support nudges scoring, it doesn't drive it).
  - Unknown context handling: previously, whenever dialogue_completed was
    False, location_context and social_context defaulted to "other"/"alone".
    Module 2's spec confirms location/social questions are optional and can be
    skipped even when dialogue_completed=True. The engine now: (a) passes
    through whatever Module 2 actually sent when dialogue completed —
    including an explicit "unknown" value — and (b) falls back to "unknown"
    (not "other"/"alone") only when no dialogue happened at all. This makes
    the "no data" case honest rather than assuming a specific context.
  - time_of_day: previously assumed to be supplied by the caller (Module 2's
    own spec confirms time_of_day is NOT exposed to Module 3 — it is scoped to
    Module 4 only). The engine now derives time_of_day locally from Module 1's
    timestamp field whenever a timestamp is present, removing an unnecessary
    dependency on Module 2 for a field it was never actually contracted to
    provide. If no timestamp is available, an explicitly-passed time_of_day
    (or a neutral default) is used instead — kept for backward compatibility
    with hand-built test episodes.

Output: ranked list of candidate interventions with Layer 1 relevance scores.
        Each candidate includes the score breakdown for explainability.

Author : Module 3 — MindPulse (Team MindForge)
"""

from datetime import datetime
import networkx as nx
from knowledge_graph import build_knowledge_graph, INTERVENTION_MAP


# ── Urgency thresholds (from Module 3 spec Section 2.1) ───────────────────────
CONFIDENCE_THRESHOLD    = 0.55   # below this → conservative mode
TRIGGER_CONF_THRESHOLD  = 0.65   # below this → context fields down-weighted
DEVIATION_ACTIVE        = 2.50   # z-score above this → active push
DEVIATION_PASSIVE       = 1.00   # z-score above this → passive nudge

# ── Score component weights (sum = 1.0) ───────────────────────────────────────
# How much each signal contributes to the final Layer 1 relevance score
W_TIER      = 0.40   # Tier-appropriateness (primary signal)
W_TRIGGER   = 0.20   # Trigger-context affinity
W_GESTURE   = 0.20   # Behavioural-support alignment
W_LOCATION  = 0.10   # Location feasibility bonus
W_SOCIAL    = 0.10   # Social context feasibility bonus


# ══════════════════════════════════════════════════════════════════════════════
# LOCAL TIME-OF-DAY DERIVATION
# ══════════════════════════════════════════════════════════════════════════════

def derive_time_of_day(timestamp_str: str) -> str:
    """
    Derive a time-of-day bucket locally from an ISO 8601 timestamp string.

    This removes Module 3's dependency on Module 2 for time_of_day — Module 2's
    documented interface scopes time_of_day to Module 4 only, so Module 3 was
    never actually contracted to receive it. Module 1's timestamp is always
    available and is the correct local source for this derivation.

    Buckets: morning 05:00-11:59, afternoon 12:00-16:59,
             evening 17:00-20:59, night 21:00-04:59 (UTC).

    Returns "afternoon" as a safe neutral default if the timestamp is missing
    or cannot be parsed.
    """
    if not timestamp_str:
        return "afternoon"
    try:
        ts = timestamp_str.replace("Z", "+00:00") if timestamp_str.endswith("Z") else timestamp_str
        hour = datetime.fromisoformat(ts).hour
    except (ValueError, AttributeError, TypeError):
        return "afternoon"

    if 5 <= hour < 12:
        return "morning"
    elif 12 <= hour < 17:
        return "afternoon"
    elif 17 <= hour < 21:
        return "evening"
    else:
        return "night"


# ══════════════════════════════════════════════════════════════════════════════
# RULE ENGINE
# ══════════════════════════════════════════════════════════════════════════════

class Layer1RuleEngine:
    """
    Rule-based recommender that traverses the Knowledge Graph to produce
    a ranked, context-filtered candidate list of interventions.

    Parameters
    ----------
    G : nx.DiGraph
        The Module 3 Knowledge Graph (from knowledge_graph.build_knowledge_graph())
    top_k : int
        Number of top-ranked candidates to return (default: 5)
    """

    def __init__(self, G: nx.DiGraph, top_k: int = 5):
        self.G     = G
        self.top_k = top_k

    # ──────────────────────────────────────────────────────────────────────────
    # PUBLIC METHOD: recommend()
    # ──────────────────────────────────────────────────────────────────────────

    def recommend(self, episode: dict) -> dict:
        """
        Apply the full Layer 1 gating and scoring procedure to a stress episode.

        Parameters
        ----------
        episode : dict
            A stress episode input containing Module 1 and (optionally) Module 2 fields.
            Required Module 1 fields: tier, confidence, baseline_deviation,
              signal_quality, baseline_mode
            Optional Module 1 fields: timestamp (ISO 8601 — used to derive
              time_of_day locally), support_detected (bool), support_score
              (float 0.0-1.0) — behavioural-support signal (replaces the old
              gesture_dominant field).
            Optional Module 2 fields: trigger_type, trigger_confidence,
              location_context, social_context, dialogue_completed.
            Optional fallback field: time_of_day — only used if no timestamp
              is present (kept for backward-compatible manual test episodes).

        Returns
        -------
        result : dict
            {
              "ranked_candidates": list of dicts (intervention + scores),
              "top_recommendation": dict (highest-ranked candidate),
              "urgency_mode": str ("active" | "passive" | "conservative" | "suppressed"),
              "gating_flags": dict (which rules fired and why),
              "explanation": str (human-readable explanation of top recommendation),
            }
        """
        # ── Extract fields with safe defaults ─────────────────────────────────
        tier               = episode.get("tier", "mild")
        confidence         = float(episode.get("confidence", 0.70))
        baseline_deviation = float(episode.get("baseline_deviation", 1.0))
        signal_quality     = episode.get("signal_quality", "good")
        baseline_mode      = episode.get("baseline_mode", "personalised")
        dialogue_completed = episode.get("dialogue_completed", False)

        # Module 2 context — pass through actual values when dialogue completed
        # (including an explicit "unknown" if Module 2 skipped that question);
        # fall back to "unknown" (not a guessed context) when no dialogue occurred.
        trigger_type       = episode.get("trigger_type", "unknown") if dialogue_completed else "unknown"
        trigger_confidence = float(episode.get("trigger_confidence", 0.70)) if dialogue_completed else 0.0
        location_context   = episode.get("location_context", "unknown") if dialogue_completed else "unknown"
        social_context      = episode.get("social_context", "unknown") if dialogue_completed else "unknown"

        # Behavioural support signal (replaces old gesture_dominant)
        support_detected = bool(episode.get("support_detected", False))
        support_score    = float(episode.get("support_score", 0.0)) if support_detected else 0.0

        # time_of_day — derive locally from Module 1's timestamp when available;
        # fall back to an explicitly-passed value only if no timestamp is given
        if episode.get("timestamp"):
            time_of_day = derive_time_of_day(episode["timestamp"])
        else:
            time_of_day = episode.get("time_of_day", "afternoon")

        # ── Gate 1: Calm tier → no recommendation ─────────────────────────────
        if tier == "calm":
            return self._empty_result("calm tier — no recommendation triggered", episode)

        # ── Gate 2: Signal quality → suppress high-urgency if poor ────────────
        if signal_quality == "poor":
            return self._empty_result("signal quality poor — event logged, no push", episode)

        # ── Gate 3: Confidence → conservative mode if below threshold ─────────
        conservative_mode = confidence < CONFIDENCE_THRESHOLD

        # ── Gate 4: Baseline deviation → urgency mode ─────────────────────────
        if baseline_mode == "cold_start":
            urgency_mode = "passive"   # no personal baseline → conservative nudge
        elif conservative_mode:
            urgency_mode = "conservative"
        elif baseline_deviation >= DEVIATION_ACTIVE:
            urgency_mode = "active"
        elif baseline_deviation >= DEVIATION_PASSIVE:
            urgency_mode = "passive"
        else:
            urgency_mode = "passive"

        # ── Gate 5: Trigger confidence → down-weight context if low ───────────
        trigger_weight_mult = 1.0 if trigger_confidence >= TRIGGER_CONF_THRESHOLD else 0.40

        # ── Determine behavioural-support KG node ──────────────────────────────
        gesture_profile = "support_detected" if support_detected else "no_support"

        gating_flags = {
            "conservative_mode":     conservative_mode,
            "urgency_mode":          urgency_mode,
            "dialogue_available":    dialogue_completed,
            "trigger_type":          trigger_type,
            "trigger_down_weighted": trigger_confidence < TRIGGER_CONF_THRESHOLD,
            "cold_start":            baseline_mode == "cold_start",
            "behavioural_support_detected": support_detected,
            "derived_time_of_day":   time_of_day,
        }

        candidates = self._score_all_interventions(
            tier, trigger_type, location_context, social_context,
            gesture_profile, support_score, time_of_day, trigger_weight_mult,
            conservative_mode, baseline_deviation
        )

        if not candidates:
            # Guaranteed fallback: Diaphragmatic Breathing (no constraints)
            candidates = [self._fallback_candidate(tier)]

        # ── Sort and select top-K ──────────────────────────────────────────────
        candidates = sorted(candidates, key=lambda x: -x["layer1_score"])

        if conservative_mode:
            # Conservative mode: take lowest-intensity intervention that passes filters
            candidates = sorted(candidates, key=lambda x: x["duration"])[:self.top_k]

        top_k_candidates = candidates[:self.top_k]
        top              = top_k_candidates[0]

        return {
            "ranked_candidates":  top_k_candidates,
            "top_recommendation": top,
            "urgency_mode":       urgency_mode,
            "gating_flags":       gating_flags,
            "explanation":        self._build_explanation(top, tier, trigger_type,
                                                          support_detected, location_context,
                                                          social_context, urgency_mode),
        }

    # ──────────────────────────────────────────────────────────────────────────
    # PRIVATE HELPERS
    # ──────────────────────────────────────────────────────────────────────────

    def _score_all_interventions(self, tier, trigger_type, location,
                                  social, gesture_profile, support_score,
                                  time_of_day, trigger_weight_mult,
                                  conservative_mode, baseline_deviation) -> list:
        """
        Score every intervention node in the KG using weighted edge traversal.
        Returns only feasible candidates (feasibility-excluded ones are dropped).
        """
        G = self.G
        candidates = []

        for inv in INTERVENTION_MAP.values():
            inv_node = f"Intervention:{inv['id']}"

            # ── Tier eligibility: must be in intervention's target tiers ───────
            if tier not in inv["tiers"]:
                continue

            # ── Hard feasibility exclusions ────────────────────────────────────
            if social   in inv.get("excludes_social",   []):
                continue
            if location in inv.get("excludes_location", []):
                continue
            if inv.get("requires_time") and time_of_day not in inv.get("requires_time", []):
                continue

            # ── KG edge-weight retrieval ───────────────────────────────────────
            tier_w     = self._get_edge_weight(G, f"StressState:{tier}",           inv_node)
            trigger_w  = self._get_edge_weight(G, f"TriggerContext:{trigger_type}", inv_node) * trigger_weight_mult
            location_w = self._get_edge_weight(G, f"LocationContext:{location}",    inv_node)
            social_w   = self._get_edge_weight(G, f"SocialContext:{social}",        inv_node)

            # Behavioural-support weight: the KG stores the MAXIMUM possible
            # influence for "support_detected"; scale it by the actual
            # per-episode support_score (0.0-1.0), mirroring Module 1's own
            # bounded, small-influence confidence-boost design. "no_support"
            # uses its base weight directly (no scaling — it's a stable state).
            gesture_w_base = self._get_edge_weight(G, f"GestureProfile:{gesture_profile}", inv_node)
            if gesture_profile == "support_detected":
                gesture_w = gesture_w_base * support_score
            else:
                gesture_w = gesture_w_base

            # ── Weighted composite score ───────────────────────────────────────
            score = (W_TIER     * tier_w    +
                     W_TRIGGER  * trigger_w +
                     W_GESTURE  * gesture_w +
                     W_LOCATION * location_w +
                     W_SOCIAL   * social_w)

            # ── Baseline deviation urgency scaling ────────────────────────────
            # High deviation boosts short, immediate interventions slightly
            if baseline_deviation >= DEVIATION_ACTIVE and inv["duration"] <= 3:
                score *= 1.10
            elif baseline_deviation < DEVIATION_PASSIVE and inv["duration"] > 3:
                score *= 0.90

            candidates.append({
                "intervention_id":   inv["id"],
                "intervention_name": inv["name"],
                "intervention_type": inv["type"],
                "duration":          inv["duration"],
                "layer1_score":      round(score, 4),
                "score_breakdown": {
                    "tier_score":     round(tier_w,     4),
                    "trigger_score":  round(trigger_w,  4),
                    "gesture_score":  round(gesture_w,  4),
                    "location_score": round(location_w, 4),
                    "social_score":   round(social_w,   4),
                },
            })

        return candidates

    def _get_edge_weight(self, G, source_node: str, target_node: str) -> float:
        """Safely retrieve edge weight from KG (returns 0.0 if edge absent)."""
        if G.has_edge(source_node, target_node):
            return G[source_node][target_node].get("weight", 0.0)
        return 0.0

    def _fallback_candidate(self, tier: str) -> dict:
        """Guaranteed fallback: Diaphragmatic Breathing (I03) — no constraints."""
        return {
            "intervention_id":   "I03",
            "intervention_name": "Diaphragmatic Breathing",
            "intervention_type": "breathing",
            "duration":          2,
            "layer1_score":      0.50,
            "score_breakdown":   {"tier_score": 0.50, "trigger_score": 0.50,
                                  "gesture_score": 0.50, "location_score": 0.10,
                                  "social_score": 0.10},
        }

    def _empty_result(self, reason: str, episode: dict) -> dict:
        """Return a structured empty result with a reason (for gated-out episodes)."""
        return {
            "ranked_candidates":  [],
            "top_recommendation": None,
            "urgency_mode":       "suppressed",
            "gating_flags":       {"suppressed": True, "reason": reason},
            "explanation":        f"No recommendation issued: {reason}",
        }

    def _build_explanation(self, top: dict, tier: str, trigger: str,
                            support_detected: bool, location: str, social: str,
                            urgency: str) -> str:
        """
        Build a human-readable GNNExplainer-style explanation for the top recommendation.
        Format: "Recommended because: [tier] stress + [trigger] context + [support] detected."
        """
        trigger_str  = f"{trigger} context" if trigger != "unknown" else "unspecified context"
        support_str  = "repetitive movement support detected" if support_detected else "no behavioural support signal"
        location_str = f"feasible at {location}" if location != "unknown" else "location not specified"
        urgency_str  = {"active": "active intervention mode", "passive": "gentle nudge mode",
                         "conservative": "conservative mode (low confidence)"}.get(urgency, urgency)

        return (
            f"Recommended '{top['intervention_name']}' ({top['intervention_type']}, "
            f"{top['duration']} min) because: {tier} stress tier + {trigger_str} + "
            f"{support_str} + {location_str} [{urgency_str}]. "
            f"Layer 1 score: {top['layer1_score']:.4f}."
        )


# ══════════════════════════════════════════════════════════════════════════════
# STANDALONE DEMO
# ══════════════════════════════════════════════════════════════════════════════

def print_recommendation_result(result: dict) -> None:
    """Pretty-print a recommendation result for demo/debugging."""
    print("\n" + "─" * 65)
    print(f"  URGENCY MODE  : {result['urgency_mode'].upper()}")
    print(f"  EXPLANATION   : {result['explanation']}")
    print(f"\n  TOP {len(result['ranked_candidates'])} CANDIDATES:")
    print(f"  {'Rank':<5} {'ID':<5} {'Score':<7} {'Type':<12} {'Min':<5} {'Name'}")
    print(f"  {'─'*5} {'─'*5} {'─'*7} {'─'*12} {'─'*5} {'─'*35}")
    for i, c in enumerate(result["ranked_candidates"], 1):
        print(f"  {i:<5} {c['intervention_id']:<5} {c['layer1_score']:<7.4f} "
              f"{c['intervention_type']:<12} {c['duration']:<5} {c['intervention_name']}")
    if result["gating_flags"]:
        flags_str = ", ".join(f"{k}={v}" for k, v in result["gating_flags"].items()
                              if v is not False and v is not None and v != "")
        print(f"\n  GATING FLAGS  : {flags_str}")
    print("─" * 65)


if __name__ == "__main__":
    print("Building Knowledge Graph...")
    G = build_knowledge_graph()

    engine = Layer1RuleEngine(G, top_k=5)

    # ── Test Case 1: Acute stress, academic trigger, alone, university ─────────
    # Strong behavioural support signal (mirrors old high-urgency "fidget" case)
    print("\n" + "═" * 65)
    print("  TEST CASE 1 — Acute stress | Academic | Alone | University | Strong support signal")
    print("═" * 65)
    ep1 = {
        "tier": "acute", "confidence": 0.85, "baseline_deviation": 3.2,
        "support_detected": True, "support_score": 0.85, "signal_quality": "good",
        "baseline_mode": "personalised", "dialogue_completed": True,
        "trigger_type": "academic", "trigger_confidence": 0.82,
        "location_context": "university", "social_context": "alone",
        "time_of_day": "afternoon",
    }
    print_recommendation_result(engine.recommend(ep1))

    # ── Test Case 2: Moderate stress, interpersonal, with colleagues, work ─────
    print("\n" + "═" * 65)
    print("  TEST CASE 2 — Moderate stress | Interpersonal | Colleagues | Work | Support signal")
    print("═" * 65)
    ep2 = {
        "tier": "moderate", "confidence": 0.78, "baseline_deviation": 2.1,
        "support_detected": True, "support_score": 0.60, "signal_quality": "good",
        "baseline_mode": "personalised", "dialogue_completed": True,
        "trigger_type": "interpersonal", "trigger_confidence": 0.75,
        "location_context": "work", "social_context": "colleagues",
        "time_of_day": "morning",
    }
    print_recommendation_result(engine.recommend(ep2))

    # ── Test Case 3: Mild stress, cold-start (new user), no Module 2 dialogue ──
    # dialogue_completed=False → location/social should now show "unknown",
    # not the old hardcoded "other"/"alone" defaults.
    print("\n" + "═" * 65)
    print("  TEST CASE 3 — Mild stress | Cold-start | No Module 2 dialogue at all")
    print("═" * 65)
    ep3 = {
        "tier": "mild", "confidence": 0.72, "baseline_deviation": 1.3,
        "support_detected": False, "support_score": 0.0, "signal_quality": "degraded",
        "baseline_mode": "cold_start", "dialogue_completed": False,
        "time_of_day": "evening",
    }
    print_recommendation_result(engine.recommend(ep3))

    # ── Test Case 4: Low confidence → conservative mode ───────────────────────
    print("\n" + "═" * 65)
    print("  TEST CASE 4 — Moderate stress | Low confidence → conservative")
    print("═" * 65)
    ep4 = {
        "tier": "moderate", "confidence": 0.48, "baseline_deviation": 1.8,
        "support_detected": True, "support_score": 0.55, "signal_quality": "degraded",
        "baseline_mode": "personalised", "dialogue_completed": True,
        "trigger_type": "unknown", "trigger_confidence": 0.50,
        "location_context": "home", "social_context": "alone",
        "time_of_day": "night",
    }
    print_recommendation_result(engine.recommend(ep4))

    # ── Test Case 5: Poor signal quality → suppressed ─────────────────────────
    print("\n" + "═" * 65)
    print("  TEST CASE 5 — Poor signal quality → suppressed (no push)")
    print("═" * 65)
    ep5 = {
        "tier": "moderate", "confidence": 0.81, "baseline_deviation": 2.5,
        "support_detected": True, "support_score": 0.70, "signal_quality": "poor",
        "baseline_mode": "personalised",
    }
    print_recommendation_result(engine.recommend(ep5))

    # ── Test Case 6 (NEW): Dialogue completed, but location/social skipped ─────
    # Validates the "unknown" KG nodes added in Step 1 — distinct from Test 3,
    # where there was no dialogue at all. Here Module 2 DID run and DID
    # classify the trigger, it just didn't ask the optional location/social
    # questions that day. Also demonstrates local time_of_day derivation from
    # a Module 1 timestamp instead of an explicitly-passed value.
    print("\n" + "═" * 65)
    print("  TEST CASE 6 — Dialogue completed | Trigger known | Location & social skipped")
    print("  (also demonstrates time_of_day derived locally from timestamp)")
    print("═" * 65)
    ep6 = {
        "tier": "moderate", "confidence": 0.74, "baseline_deviation": 1.9,
        "support_detected": False, "support_score": 0.0, "signal_quality": "good",
        "baseline_mode": "personalised", "dialogue_completed": True,
        "trigger_type": "academic", "trigger_confidence": 0.71,
        "location_context": "unknown", "social_context": "unknown",
        "timestamp": "2026-07-21T22:15:00Z",   # no time_of_day supplied — derived below
    }
    print_recommendation_result(engine.recommend(ep6))

    print("\n[✓] Rule Engine demo complete.\n")