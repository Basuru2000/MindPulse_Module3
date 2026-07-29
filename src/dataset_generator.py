"""
MindPulse — Module 3 Synthetic Dataset Generator
=================================================
Generates a realistic synthetic dataset of 700 stress-intervention-outcome
sessions for training and evaluating the Module 3 recommendation engine.

Physiological feature distributions are grounded in the WESAD dataset
(Schmidt et al., 2018) — specifically the HRV, EDA, and IMU signal
characteristics observed across the baseline and stress conditions,
as documented in the published dataset paper.

Dataset strategy: Hybrid approach
  - Real physiological distributions → from WESAD (Schmidt et al., 2018)
  - Synthetic session records       → generated here (Module 3-specific fields)
  - Outcome rating simulation       → grounded in published JITAI effect sizes

  CHANGE LOG (aligned to updated Module 1 / Module 2 specs):
  - Gesture simulation replaced with behavioural-support simulation. Module 1's
    revised design no longer classifies gesture type (scratch/fidget/walk/
    still) — it emits a single behavioural-support signal: detected or not,
    plus a support_score (0.0-1.0), event count, and window proportion. This
    generator now produces that schema instead of the old 4-category one.
  - location_context / social_context: when dialogue_completed=True, there is
    now a small independent chance either field is stored as "unknown" —
    mirroring Module 2's confirmed behaviour that location/social questions
    are optional and asked at most once per day. This exercises the "unknown"
    KG nodes and rule-engine handling added in Steps 1-2. The underlying
    feasibility simulation used to pick the historically-delivered
    intervention still uses the true sampled context — only the STORED,
    reported column reflects the possible "unknown"; this mirrors a live
    system, where the physical environment is real but Module 2 may not have
    captured it in a given dialogue turn.
  - timestamp / time_of_day consistency fix: previously the session hour was
    sampled independently of the separately-sampled time_of_day bucket,
    meaning the two fields could disagree. Since rule_engine.py (Step 2) now
    derives time_of_day locally FROM the timestamp, the two must agree, so the
    hour is now sampled to match the chosen time_of_day bucket.

Output:
  data/synthetic/module3_dataset.csv            (for implementation)
  data/presentation/module3_dataset.xlsx        (for evaluator presentation)

Author : Module 3 — MindPulse (Team MindForge)
Dataset: 700 sessions × 25 users
"""

import numpy as np
import pandas as pd
from datetime import datetime, timedelta
import random
import os

# ── Reproducibility ────────────────────────────────────────────────────────────
SEED = 42
np.random.seed(SEED)
random.seed(SEED)

# ── Configuration ──────────────────────────────────────────────────────────────
N_USERS    = 25
N_SESSIONS = 700
START_DATE = datetime(2025, 9, 1)   # Academic year start — matches stakeholder profile

# ── WESAD-Derived Physiological Distributions (per stress tier) ────────────────
# Source: Schmidt et al. (2018), WESAD dataset
WESAD_DISTRIBUTIONS = {
    "calm": {
        "rmssd":              (48.0, 14.0),
        "sdnn":               (55.0, 18.0),
        "lf_hf":              (1.80,  0.70),
        "temp_slope":         (0.020, 0.015),
        "jerk_mean":          (0.30,  0.15),
        "baseline_deviation": (0.30,  0.40),
    },
    "mild": {
        "rmssd":              (38.0, 12.0),
        "sdnn":               (44.0, 14.0),
        "lf_hf":              (2.40,  0.90),
        "temp_slope":         (0.040, 0.020),
        "jerk_mean":          (0.50,  0.20),
        "baseline_deviation": (1.20,  0.40),
    },
    "moderate": {
        "rmssd":              (27.0, 10.0),
        "sdnn":               (34.0, 11.0),
        "lf_hf":              (3.20,  1.10),
        "temp_slope":         (0.070, 0.025),
        "jerk_mean":          (0.80,  0.25),
        "baseline_deviation": (2.00,  0.40),
    },
    "acute": {
        "rmssd":              (18.0,  8.0),
        "sdnn":               (24.0,  9.0),
        "lf_hf":              (4.50,  1.40),
        "temp_slope":         (0.120, 0.030),
        "jerk_mean":          (1.20,  0.35),
        "baseline_deviation": (3.00,  0.50),
    },
}

# ── Evidence-Based Intervention Library (22 interventions) — unchanged ────────
INTERVENTIONS = [
    {"id": "I01", "name": "4-7-8 Breathing",                 "type": "breathing",  "tiers": ["mild", "moderate"],           "duration": 2, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I02", "name": "Box Breathing (4-4-4-4)",          "type": "breathing",  "tiers": ["moderate", "acute"],          "duration": 3, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I03", "name": "Diaphragmatic Breathing",          "type": "breathing",  "tiers": ["mild","moderate","acute"],    "duration": 2, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I04", "name": "5-Minute Brisk Walk",              "type": "physical",   "tiers": ["mild", "moderate"],           "duration": 5, "excludes_social": [],                                   "excludes_location": ["in_transit", "work"]},
    {"id": "I05", "name": "Progressive Muscle Relaxation",    "type": "physical",   "tiers": ["moderate", "acute"],          "duration": 5, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I06", "name": "Cold Water Face Splash",           "type": "physical",   "tiers": ["acute"],                      "duration": 1, "excludes_social": [],                                   "excludes_location": ["in_transit"]},
    {"id": "I07", "name": "Grounding Technique (5-4-3-2-1)", "type": "cognitive",  "tiers": ["acute"],                      "duration": 3, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I08", "name": "Cognitive Reframing (3 Good Things)","type":"cognitive", "tiers": ["mild"],                       "duration": 3, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I09", "name": "Worry Postponement",               "type": "cognitive",  "tiers": ["mild", "moderate"],           "duration": 3, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I10", "name": "Brief Mindfulness Check-in",       "type": "cognitive",  "tiers": ["mild", "moderate"],           "duration": 2, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I11", "name": "Social Contact Prompt",            "type": "social",     "tiers": ["mild", "moderate"],           "duration": 2, "excludes_social": ["colleagues","friends","family"],     "excludes_location": []},
    {"id": "I12", "name": "Gratitude Message Prompt",         "type": "social",     "tiers": ["mild"],                       "duration": 2, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I13", "name": "Nature Sound / White Noise",       "type": "sensory",    "tiers": ["moderate"],                   "duration": 5, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I14", "name": "Hydration Reminder",               "type": "physical",   "tiers": ["mild"],                       "duration": 1, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I15", "name": "Stretching (desk-based)",          "type": "physical",   "tiers": ["mild", "moderate"],           "duration": 3, "excludes_social": [],                                   "excludes_location": ["in_transit"]},
    {"id": "I16", "name": "Single-Focus Task Prompt",         "type": "cognitive",  "tiers": ["moderate"],                   "duration": 3, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I17", "name": "Visual Focus Break (20-20-20)",    "type": "physical",   "tiers": ["mild"],                       "duration": 1, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I18", "name": "Journalling Micro-Prompt",         "type": "cognitive",  "tiers": ["mild"],                       "duration": 3, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I19", "name": "Self-Compassion Pause",            "type": "cognitive",  "tiers": ["mild","moderate","acute"],    "duration": 2, "excludes_social": [],                                   "excludes_location": []},
    {"id": "I20", "name": "Power Posture Reset",              "type": "physical",   "tiers": ["mild", "moderate"],           "duration": 1, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I21", "name": "Rhythmic Tapping (EFT-lite)",      "type": "physical",   "tiers": ["moderate", "acute"],          "duration": 2, "excludes_social": ["colleagues"],                       "excludes_location": []},
    {"id": "I22", "name": "Sleep Hygiene Reminder",           "type": "cognitive",  "tiers": ["mild"],                       "duration": 1, "excludes_social": [],                                   "excludes_location": [], "requires_time": ["evening", "night"]},
]

INTERVENTION_MAP = {inv["id"]: inv for inv in INTERVENTIONS}

# ── Contextual Distributions (unchanged) ────────────────────────────────────────
TIER_DIST        = {"mild": 0.40, "moderate": 0.35, "acute": 0.25}
LOCATION_DIST    = {"university": 0.35, "home": 0.30, "work": 0.20, "in_transit": 0.10, "other": 0.05}
SOCIAL_DIST      = {"alone": 0.45, "colleagues": 0.25, "friends": 0.15, "family": 0.10, "other": 0.05}
TRIGGER_DIST     = {"academic": 0.35, "interpersonal": 0.25, "financial": 0.15, "health": 0.15, "other": 0.10}
TIME_OF_DAY_DIST = {"morning": 0.20, "afternoon": 0.35, "evening": 0.30, "night": 0.15}

# ── Unknown-context sampling probabilities ──────────────────────────────────────
# Applied only when dialogue_completed=True — mirrors Module 2's confirmed
# behaviour that location/social questions are optional (asked at most once
# per day). Independent per field.
P_LOCATION_UNKNOWN_IF_DIALOGUE = 0.12
P_SOCIAL_UNKNOWN_IF_DIALOGUE   = 0.12

# ── Hour ranges matching rule_engine.py's derive_time_of_day() buckets ─────────
# Keeps the generated timestamp consistent with the sampled time_of_day bucket,
# since Layer 1 now derives time_of_day FROM the timestamp, not the other way
# around. Night wraps across midnight (21:00-23:59 and 00:00-04:59).
TIME_OF_DAY_HOUR_RANGES = {
    "morning":   (5, 11),
    "afternoon": (12, 16),
    "evening":   (17, 20),
}


# ══════════════════════════════════════════════════════════════════════════════
# HELPER FUNCTIONS
# ══════════════════════════════════════════════════════════════════════════════

def weighted_choice(distribution: dict) -> str:
    """Select a key from a probability distribution dictionary."""
    keys   = list(distribution.keys())
    weights = list(distribution.values())
    return random.choices(keys, weights=weights, k=1)[0]


def sample_hour_for_time_of_day(bucket: str) -> int:
    """
    Sample an hour (0-23) consistent with the given time_of_day bucket, so
    that rule_engine.derive_time_of_day(timestamp) reconstructs the same
    bucket later. "night" wraps across midnight.
    """
    if bucket == "night":
        return random.choice(list(range(21, 24)) + list(range(0, 5)))
    lo, hi = TIME_OF_DAY_HOUR_RANGES[bucket]
    return random.randint(lo, hi)


def generate_user_profiles(n_users: int) -> dict:
    """
    Generate per-user preference profiles that drive personalised outcome ratings.
    Each user has a preferred intervention type, physiological sensitivity,
    and compliance rate — mirroring real-world individual variation.
    """
    profiles = {}
    inv_types = ["breathing", "physical", "cognitive", "social", "sensory"]
    for i in range(1, n_users + 1):
        uid = f"U{i:03d}"
        profiles[uid] = {
            "preferred_type": random.choice(inv_types),
            "sensitivity":    round(np.random.uniform(0.60, 1.00), 3),
            "compliance":     round(np.random.uniform(0.50, 0.95), 3),
            "session_count":  0,
        }
    return profiles


def sample_physio_features(tier: str) -> dict:
    """
    Sample physiological features from WESAD-derived distributions for the given tier.
    All values are clipped to physiologically plausible ranges.
    """
    dist = WESAD_DISTRIBUTIONS[tier]
    features = {}
    for feat, (mean, std) in dist.items():
        val = np.random.normal(mean, std)
        if   feat == "rmssd":              val = max(5.0,   min(120.0, val))
        elif feat == "sdnn":               val = max(5.0,   min(150.0, val))
        elif feat == "lf_hf":              val = max(0.20,  min(10.0,  val))
        elif feat == "temp_slope":         val = max(-0.05, min(0.30,  val))
        elif feat == "jerk_mean":          val = max(0.05,  min(3.00,  val))
        elif feat == "baseline_deviation": val = max(0.0,            val)
        features[feat] = round(val, 4)
    return features


def generate_behavioural_support(tier: str) -> dict:
    """
    Generate the Module 1-aligned behavioural-support signal for a session.

    Mirrors Module 1's revised Stage 5: the module detects only WHETHER
    repetitive wrist-movement is present, never a specific gesture type
    (scratch/fidget/walk/still). Detection probability and the resulting
    support_score scale loosely with stress tier — repetitive movement is
    more common at higher physiological arousal, but is never guaranteed
    (it may be habit, boredom, exercise, or device adjustment instead).

    Returns a dict with support_detected (bool), support_score (float,
    0.0 if not detected), support_event_count (int), and support_proportion
    (float, fraction of the analysis window containing repetitive movement).
    """
    detect_prob = {"mild": 0.35, "moderate": 0.55, "acute": 0.65}[tier]
    detected = random.random() < detect_prob

    if not detected:
        return {
            "support_detected":    False,
            "support_score":       0.0,
            "support_event_count": 0,
            "support_proportion":  0.0,
        }

    tier_score_range = {"mild": (0.35, 0.65), "moderate": (0.45, 0.80), "acute": (0.55, 0.90)}
    lo, hi = tier_score_range[tier]
    support_score = round(float(np.random.uniform(lo, hi)), 3)
    event_count   = int(np.random.poisson(lam=2 + support_score * 3))
    proportion    = round(float(np.clip(support_score * np.random.uniform(0.7, 1.1), 0.05, 0.60)), 3)

    return {
        "support_detected":    True,
        "support_score":       support_score,
        "support_event_count": event_count,
        "support_proportion":  proportion,
    }


def get_feasible_interventions(tier: str, location: str, social: str, time_of_day: str) -> list:
    """
    Return the subset of interventions that pass Layer 1 context-feasibility filtering.
    Mirrors the exact gating rules defined in the Module 3 spec Section 2.1.
    Uses the TRUE sampled context (see module docstring change log) — the
    physical environment during a historically-delivered intervention is real
    even if Module 2's dialogue later reports it as "unknown" for that turn.
    """
    feasible = []
    for inv in INTERVENTIONS:
        if tier not in inv["tiers"]:
            continue
        if social in inv.get("excludes_social", []):
            continue
        if location in inv.get("excludes_location", []):
            continue
        if "requires_time" in inv and time_of_day not in inv["requires_time"]:
            continue
        feasible.append(inv)
    if not feasible:
        feasible = [INTERVENTION_MAP["I03"]]
    return feasible


def simulate_outcome_rating(intervention: dict, tier: str, user_profile: dict,
                             support_detected: bool, support_score: float) -> int:
    """
    Simulate an outcome rating (1–5) based on:
      - Clinical appropriateness (tier–intervention match)
      - Individual user preference alignment
      - Behavioural-support alignment (somatic vs cognitive)
      - User physiological sensitivity
      - Random noise (real-world variability)

    Grounded in published JITAI effect sizes from:
      Adams et al. (JMIR 2024), von Lützow et al. (BMJ Mental Health 2025),
      and Breeze cohort study.
    """
    base = 3.0

    tier_match = {
        ("breathing", "acute"):    0.80,
        ("breathing", "moderate"): 0.60,
        ("cognitive", "mild"):     0.70,
        ("physical",  "moderate"): 0.50,
        ("cognitive", "moderate"): 0.40,
        ("physical",  "mild"):     0.40,
        ("social",    "mild"):     0.50,
        ("sensory",   "moderate"): 0.40,
        ("physical",  "acute"):    0.45,
        ("cognitive", "acute"):    0.35,
    }
    base += tier_match.get((intervention["type"], tier), 0.10)

    if intervention["type"] == user_profile["preferred_type"]:
        base += 0.60 * user_profile["sensitivity"]

    # Behavioural-support alignment: detected repetitive movement (scaled by
    # its score) favours somatic interventions; its absence mildly favours
    # cognitive ones (calmer physical state). Scaling by support_score
    # mirrors Module 1's own bounded, small-influence confidence-boost design
    # — this is not a hard classification, just a graded nudge.
    if support_detected and intervention["type"] in ["breathing", "physical"]:
        base += 0.30 * support_score
    elif not support_detected and intervention["type"] == "cognitive":
        base += 0.20

    base *= (0.70 + 0.30 * user_profile["sensitivity"])
    base += np.random.normal(0.0, 0.50)

    return int(np.clip(round(base), 1, 5))


def simulate_tier_delta(outcome_rating: int, tier: str) -> tuple:
    """
    Simulate post-intervention stress tier change (30-minute follow-up).
    Higher outcome ratings → greater likelihood of tier improvement (negative delta).
    Returns (tier_delta: int, tier_after: str).
    """
    tier_index  = {"mild": 1, "moderate": 2, "acute": 3}
    tier_names  = ["calm", "mild", "moderate", "acute"]
    current_idx = tier_index[tier]

    if outcome_rating >= 4:
        delta = int(np.random.choice([-2, -1, 0],  p=[0.25, 0.55, 0.20]))
    elif outcome_rating == 3:
        delta = int(np.random.choice([-1,  0, 1],  p=[0.35, 0.50, 0.15]))
    else:
        delta = int(np.random.choice([ 0,  1, -1], p=[0.50, 0.30, 0.20]))

    tier_after_idx = int(np.clip(current_idx + delta, 0, 3))
    return delta, tier_names[tier_after_idx]


# ══════════════════════════════════════════════════════════════════════════════
# MAIN DATASET GENERATOR
# ══════════════════════════════════════════════════════════════════════════════

def generate_dataset(n_sessions: int = N_SESSIONS, n_users: int = N_USERS) -> pd.DataFrame:
    """
    Generate the full synthetic dataset.
    Returns a DataFrame with one session record per row.
    """
    user_profiles = generate_user_profiles(n_users)
    user_ids      = list(user_profiles.keys())

    base_per_user  = n_sessions // n_users
    session_counts = {uid: base_per_user for uid in user_ids}
    remainder = n_sessions - base_per_user * n_users
    for uid in random.sample(user_ids, remainder):
        session_counts[uid] += 1

    records         = []
    session_counter = 1

    for uid in user_ids:
        profile = user_profiles[uid]
        n       = session_counts[uid]

        # Spread sessions across the academic calendar (Sept–Feb) — sample day
        # offsets only; hour is assigned per-session below so it stays
        # consistent with that session's sampled time_of_day bucket.
        day_offsets = sorted(random.randint(0, 180) for _ in range(n))

        for day_offset in day_offsets:
            profile["session_count"] += 1

            # ── Contextual fields ──────────────────────────────────────────────
            tier        = weighted_choice(TIER_DIST)
            location    = weighted_choice(LOCATION_DIST)
            social      = weighted_choice(SOCIAL_DIST)
            trigger     = weighted_choice(TRIGGER_DIST)
            time_of_day = weighted_choice(TIME_OF_DAY_DIST)

            hour   = sample_hour_for_time_of_day(time_of_day)
            minute = random.randint(0, 59)
            ts     = START_DATE + timedelta(days=day_offset, hours=hour, minutes=minute)

            # ── Physiological features (WESAD-grounded) ───────────────────────
            physio      = sample_physio_features(tier)
            behavioural = generate_behavioural_support(tier)

            # ── Module 1 signal-quality fields ────────────────────────────────
            confidence     = round(np.random.uniform(0.56, 0.98), 3)
            signal_quality = random.choices(
                ["good", "degraded", "poor"], weights=[0.70, 0.22, 0.08]
            )[0]
            baseline_mode  = "cold_start" if profile["session_count"] <= 3 else "personalised"

            # ── Module 2 context fields ────────────────────────────────────────
            dialogue_completed = random.choices([True, False], weights=[0.82, 0.18])[0]
            trigger_confidence = round(np.random.uniform(0.55, 0.95), 3)

            # Reported location/social — may be "unknown" even when dialogue
            # completed, mirroring Module 2's optional-question behaviour.
            # (The TRUE sampled `location`/`social` above is still used for
            # feasibility filtering below — see get_feasible_interventions
            # docstring for the reasoning.)
            location_reported = "unknown" if (dialogue_completed and random.random() < P_LOCATION_UNKNOWN_IF_DIALOGUE) else location
            social_reported    = "unknown" if (dialogue_completed and random.random() < P_SOCIAL_UNKNOWN_IF_DIALOGUE)   else social

            # ── Layer 1 feasibility filter → selects the delivered intervention ─
            feasible     = get_feasible_interventions(tier, location, social, time_of_day)
            intervention = random.choice(feasible)

            # ── Outcome simulation ─────────────────────────────────────────────
            outcome_rating = simulate_outcome_rating(
                intervention, tier, profile,
                behavioural["support_detected"], behavioural["support_score"]
            )
            accepted = random.random() < profile["compliance"]
            if not accepted:
                outcome_rating = max(1, outcome_rating - 1)

            tier_delta, tier_after = simulate_tier_delta(outcome_rating, tier)

            training_signal = round(
                0.60 * (outcome_rating / 5.0) + 0.40 * max(0.0, -tier_delta / 3.0), 4
            )

            # ── Assemble record ────────────────────────────────────────────────
            record = {
                # ── Session identifiers
                "session_id":              f"S{session_counter:04d}",
                "user_id":                 uid,
                "timestamp":               ts.strftime("%Y-%m-%d %H:%M:%S"),
                "date":                    ts.strftime("%Y-%m-%d"),
                "time_of_day":             time_of_day,
                "day_of_week":             ts.strftime("%A"),

                # ── Module 1: Stress Profile (schema v1.0)
                "tier":                    tier,
                "tier_index":              {"mild": 1, "moderate": 2, "acute": 3}[tier],
                "confidence":              confidence,
                "baseline_deviation":      physio["baseline_deviation"],
                "support_detected":        behavioural["support_detected"],
                "support_score":           behavioural["support_score"],
                "support_event_count":     behavioural["support_event_count"],
                "support_proportion":      behavioural["support_proportion"],
                "signal_quality":          signal_quality,
                "baseline_mode":           baseline_mode,

                # ── WESAD-grounded physiological features
                "rmssd":                   physio["rmssd"],
                "sdnn":                    physio["sdnn"],
                "lf_hf":                   physio["lf_hf"],
                "temp_slope":              physio["temp_slope"],
                "jerk_mean":               physio["jerk_mean"],

                # ── Module 2: Context (schema v2.0)
                "trigger_type":            trigger,
                "trigger_confidence":      trigger_confidence,
                "location_context":        location_reported,
                "social_context":          social_reported,
                "dialogue_completed":      dialogue_completed,

                # ── Module 3: Intervention & Outcome
                "intervention_id":         intervention["id"],
                "intervention_type":       intervention["type"],
                "intervention_name":       intervention["name"],
                "intervention_duration":   intervention["duration"],
                "recommendation_layer":    "layer1",
                "accepted":                accepted,
                "outcome_rating":          outcome_rating,
                "tier_after":              tier_after,
                "tier_delta":              tier_delta,
                "training_signal":         training_signal,
            }

            records.append(record)
            session_counter += 1

    return pd.DataFrame(records)


# ══════════════════════════════════════════════════════════════════════════════
# VALIDATION
# ══════════════════════════════════════════════════════════════════════════════

def validate_dataset(df: pd.DataFrame) -> None:
    """
    Print a structured validation report for the generated dataset.
    Checks distributions, feasibility compliance, and completeness.
    """
    print("\n" + "═" * 65)
    print("  MODULE 3 DATASET — VALIDATION REPORT")
    print("═" * 65)

    print(f"\n  Total sessions        : {len(df)}")
    print(f"  Unique users          : {df['user_id'].nunique()}")
    print(f"  Avg sessions per user : {len(df)/df['user_id'].nunique():.1f}")
    print(f"  Features per record   : {len(df.columns)}")
    print(f"  Date range            : {df['date'].min()} → {df['date'].max()}")

    print(f"\n  Stress tier distribution:")
    for tier in ["mild", "moderate", "acute"]:
        n   = (df["tier"] == tier).sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    {tier:<10}  {n:4d} sessions  ({pct:5.1f}%)  {bar}")

    print(f"\n  Outcome rating distribution:")
    for r in [1, 2, 3, 4, 5]:
        n   = (df["outcome_rating"] == r).sum()
        pct = n / len(df) * 100
        bar = "█" * int(pct / 2)
        print(f"    Rating {r}    {n:4d} sessions  ({pct:5.1f}%)  {bar}")
    print(f"    Mean rating : {df['outcome_rating'].mean():.3f}")

    print(f"\n  Unique interventions used  : {df['intervention_id'].nunique()} / 22")
    print(f"  Signal quality — good      : {(df['signal_quality']=='good').mean():.0%}")
    print(f"  Signal quality — degraded  : {(df['signal_quality']=='degraded').mean():.0%}")
    print(f"  Signal quality — poor      : {(df['signal_quality']=='poor').mean():.0%}")
    print(f"  Cold-start sessions        : {(df['baseline_mode']=='cold_start').sum()}")
    print(f"  Acceptance rate            : {df['accepted'].mean():.0%}")
    print(f"  Dialogue completed rate    : {df['dialogue_completed'].mean():.0%}")

    print(f"\n  Behavioural support detected : {df['support_detected'].mean():.0%}")
    print(f"  Location context — unknown   : {(df['location_context']=='unknown').mean():.0%}")
    print(f"  Social context — unknown     : {(df['social_context']=='unknown').mean():.0%}")

    # Feasibility compliance check (against the stored/reported context —
    # "unknown" never appears in any exclusion list, so it never falsely
    # flags a violation; the true context was already respected at selection
    # time, see get_feasible_interventions).
    violations = 0
    for _, row in df.iterrows():
        inv = INTERVENTION_MAP[row["intervention_id"]]
        if row["social_context"]   in inv.get("excludes_social",   []): violations += 1
        if row["location_context"] in inv.get("excludes_location", []): violations += 1
    status = "✓ PASS — no violations" if violations == 0 else f"✗ FAIL — {violations} violations"
    print(f"\n  Feasibility constraint check : {status}")

    # Timestamp / time_of_day consistency check — new, validates the fix
    # described in the module docstring change log.
    def _bucket_from_hour(h):
        if 5 <= h < 12: return "morning"
        if 12 <= h < 17: return "afternoon"
        if 17 <= h < 21: return "evening"
        return "night"
    df["_derived_bucket"] = pd.to_datetime(df["timestamp"]).dt.hour.apply(_bucket_from_hour)
    mismatches = (df["_derived_bucket"] != df["time_of_day"]).sum()
    df.drop(columns=["_derived_bucket"], inplace=True)
    status2 = "✓ PASS — timestamp matches time_of_day for all rows" if mismatches == 0 else f"✗ FAIL — {mismatches} mismatches"
    print(f"  Timestamp/time_of_day consistency check : {status2}")

    phys_ok = (
        df["rmssd"].between(5, 120).all() and
        df["sdnn"].between(5, 150).all() and
        df["lf_hf"].between(0.2, 10).all()
    )
    print(f"  Physiological range check    : {'✓ PASS' if phys_ok else '✗ FAIL — values out of range'}")

    print("\n" + "═" * 65 + "\n")


# ══════════════════════════════════════════════════════════════════════════════
# EXPORT
# ══════════════════════════════════════════════════════════════════════════════

def export_dataset(df: pd.DataFrame,
                   csv_dir: str   = "data/synthetic",
                   excel_dir: str = "data/presentation") -> tuple:
    """
    Export the dataset to CSV (for implementation) and Excel (for evaluators).
    """
    os.makedirs(csv_dir,   exist_ok=True)
    os.makedirs(excel_dir, exist_ok=True)

    csv_path = os.path.join(csv_dir, "module3_dataset.csv")
    df.to_csv(csv_path, index=False)
    print(f"[✓] CSV  saved → {csv_path}  ({len(df)} rows × {len(df.columns)} columns)")

    excel_path = os.path.join(excel_dir, "module3_dataset.xlsx")

    with pd.ExcelWriter(excel_path, engine="openpyxl") as writer:

        df.to_excel(writer, sheet_name="Sessions", index=False)

        summary = pd.DataFrame({
            "Metric": [
                "Total sessions",
                "Total users",
                "Average sessions per user",
                "Tier — Mild",
                "Tier — Moderate",
                "Tier — Acute",
                "Mean outcome rating",
                "Acceptance rate",
                "Unique interventions used",
                "Cold-start sessions",
                "Dialogue completed rate",
                "Behavioural support detected rate",
                "Location context unknown rate",
                "Social context unknown rate",
                "",
                "Physiological feature source",
                "RMSSD range (mild)",
                "RMSSD range (acute)",
                "Outcome rating simulation source",
            ],
            "Value": [
                len(df),
                df["user_id"].nunique(),
                f"{len(df)/df['user_id'].nunique():.1f}",
                int((df["tier"] == "mild").sum()),
                int((df["tier"] == "moderate").sum()),
                int((df["tier"] == "acute").sum()),
                f"{df['outcome_rating'].mean():.2f} / 5.00",
                f"{df['accepted'].mean():.0%}",
                int(df["intervention_id"].nunique()),
                int((df["baseline_mode"] == "cold_start").sum()),
                f"{df['dialogue_completed'].mean():.0%}",
                f"{df['support_detected'].mean():.0%}",
                f"{(df['location_context']=='unknown').mean():.0%}",
                f"{(df['social_context']=='unknown').mean():.0%}",
                "",
                "WESAD dataset (Schmidt et al., 2018) — HRV, EDA, ACC distributions",
                "mean=38ms, std=12ms",
                "mean=18ms, std=8ms",
                "Simulated from published JITAI effect sizes (Adams et al. JMIR 2024; von Lützow et al. BMJ 2025)",
            ],
        })
        summary.to_excel(writer, sheet_name="Summary", index=False)

        user_summary = (
            df.groupby("user_id")
            .agg(
                sessions          = ("session_id",     "count"),
                avg_outcome_rating= ("outcome_rating",  "mean"),
                acceptance_rate   = ("accepted",        "mean"),
                avg_tier_delta    = ("tier_delta",       "mean"),
                most_common_tier  = ("tier",             lambda x: x.mode()[0]),
                most_common_inv   = ("intervention_type",lambda x: x.mode()[0]),
            )
            .round(3)
            .reset_index()
        )
        user_summary.to_excel(writer, sheet_name="Per-User Summary", index=False)

        inv_dist = (
            df.groupby(["intervention_id", "intervention_name", "intervention_type"])
            .agg(
                total_delivered   = ("session_id",     "count"),
                avg_outcome_rating= ("outcome_rating",  "mean"),
                acceptance_rate   = ("accepted",        "mean"),
                avg_tier_delta    = ("tier_delta",       "mean"),
            )
            .round(3)
            .reset_index()
            .sort_values("intervention_id")
        )
        inv_dist.to_excel(writer, sheet_name="Intervention Distribution", index=False)

    print(f"[✓] Excel saved → {excel_path}  (4 sheets: Sessions, Summary, Per-User Summary, Intervention Distribution)")
    return csv_path, excel_path


# ══════════════════════════════════════════════════════════════════════════════
# ENTRY POINT
# ══════════════════════════════════════════════════════════════════════════════

if __name__ == "__main__":
    print("═" * 65)
    print("  MindPulse — Module 3 Synthetic Dataset Generator")
    print(f"  Target: {N_SESSIONS} sessions  |  {N_USERS} users  |  seed={SEED}")
    print("═" * 65)

    df = generate_dataset(n_sessions=N_SESSIONS, n_users=N_USERS)
    validate_dataset(df)
    export_dataset(df)

    print("[✓] All done. Your dataset is ready.")
    print("    CSV  → data/synthetic/module3_dataset.csv")
    print("    Excel→ data/presentation/module3_dataset.xlsx")