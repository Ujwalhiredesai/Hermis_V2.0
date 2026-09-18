# ==================== RANKED FAILURE MODE CLASSIFIER ====================
# Replaces classify_failure() with a multi-factor, ranked-by-probability version.
#
# WHAT CHANGED AND WHY:
# The old classify_failure() checked trend conditions with if/elif and returned
# the FIRST match as a single label. That means it could never say "70% likely
# thermal degradation, 20% likely imbalance" — it just returned one guess.
#
# This version scores EVERY failure mode candidate against multiple signals
# (not just one trend threshold each), then normalizes the scores into
# probabilities and returns them ranked. This is still fully rule-based and
# explainable (no black-box model here) — but it is now genuinely a ranked,
# multi-factor system, which is what "ranked by probability" should mean.

import numpy as np


# Each failure mode is defined by which signals push its likelihood up.
# weight = how strongly that signal contributes if the condition is met.
# condition(trends, anomaly_score) -> returns 0.0 to 1.0 (how strongly this
# failure mode's signature matches the current trend pattern)

def _clamp01(x):
    return max(0.0, min(1.0, x))


def _score_thermal_degradation(trends, anomaly_score):
    # Thermal degradation: temperature trending up steadily, vibration relatively flat
    t = trends['temperature_c']
    v = trends['vibration']
    score = _clamp01(t / 0.25)          # scales up to 1.0 as temp trend approaches +25%
    if v < 0.05:                         # low vibration change supports "pure" thermal cause
        score *= 1.15
    return _clamp01(score)


def _score_mechanical_imbalance(trends, anomaly_score):
    # Mechanical imbalance: vibration trending up sharply, temp secondary/lagging
    v = trends['vibration']
    t = trends['temperature_c']
    score = _clamp01(v / 0.35)
    if t > 0.05:                         # some heat rise from friction supports this too
        score *= 1.1
    return _clamp01(score)


def _score_power_surge_overheating(trends, anomaly_score):
    # Combined electrical + thermal stress
    p = trends['power_watts']
    t = trends['temperature_c']
    score = _clamp01((p + t) / 0.35)
    return _clamp01(score)


def _score_electrical_overload(trends, anomaly_score):
    # Power trending up on its own, without matching thermal/vibration rise
    p = trends['power_watts']
    t = trends['temperature_c']
    v = trends['vibration']
    score = _clamp01(p / 0.25)
    if t < 0.05 and v < 0.05:
        score *= 1.2                     # isolated power rise strengthens this specific label
    return _clamp01(score)


def _score_bearing_wear(trends, anomaly_score):
    # Slow-building vibration + mild temp rise, classic early bearing wear signature
    v = trends['vibration']
    t = trends['temperature_c']
    score = _clamp01((v * 0.7 + t * 0.3) / 0.20)
    return _clamp01(score)


def _score_normal_operation(trends, anomaly_score):
    # High when nothing is trending meaningfully
    max_trend = max(abs(trends['power_watts']), abs(trends['temperature_c']), abs(trends['vibration']))
    score = _clamp01(1.0 - (max_trend / 0.15))
    if anomaly_score is not None:
        score *= _clamp01(1.0 - anomaly_score)  # anomaly model pulls this down if it's suspicious
    return _clamp01(score)


FAILURE_MODES = {
    "🔥 Thermal Degradation": _score_thermal_degradation,
    "📳 Mechanical Imbalance": _score_mechanical_imbalance,
    "⚡ Power Surge + Overheating": _score_power_surge_overheating,
    "⚡ Electrical Overload": _score_electrical_overload,
    "⚙️ Early Bearing Wear": _score_bearing_wear,
    "✅ Normal Operation": _score_normal_operation,
}


def classify_failure_ranked(df, anomaly_score=None, top_n=3):
    """
    Returns a ranked list of (failure_mode_label, probability_percent) tuples,
    highest probability first. Replaces the old single-label classify_failure().

    df: sensor dataframe with power_watts, temperature_c, vibration columns
    anomaly_score: optional float 0-1 from the Isolation Forest model
                    (higher = more anomalous) to factor into scoring
    top_n: how many ranked candidates to return
    """
    if len(df) < 20:
        return [("Insufficient data", 0.0)]

    recent = df.tail(20)
    trends = {}
    for col in ['power_watts', 'temperature_c', 'vibration']:
        start, end = recent[col].iloc[0], recent[col].iloc[-1]
        trends[col] = (end - start) / start if start != 0 else 0.0

    raw_scores = {}
    for label, score_fn in FAILURE_MODES.items():
        raw_scores[label] = score_fn(trends, anomaly_score)

    total = sum(raw_scores.values())
    if total == 0:
        # Nothing scored anything meaningfully -> default to Normal Operation at 100%
        return [("✅ Normal Operation", 100.0)]

    probabilities = {label: (score / total) * 100 for label, score in raw_scores.items()}
    ranked = sorted(probabilities.items(), key=lambda x: x[1], reverse=True)

    return ranked[:top_n]


# ==================== EXAMPLE: HOW TO USE THIS IN THE DASHBOARD ====================
#
# Replace this old code:
#   pattern = classify_failure(df)
#   st.markdown(f"**🔍 Pattern:** {pattern}")
#
# With this:
#   ranked = classify_failure_ranked(df, anomaly_score=confidence if prediction == -1 else None)
#   st.markdown("**🔍 Likely Failure Mode (ranked):**")
#   for label, prob in ranked:
#       st.markdown(f"- {label}: **{prob:.0f}%**")
#
# This will show something like:
#   🔍 Likely Failure Mode (ranked):
#   - 📳 Mechanical Imbalance: 58%
#   - ⚙️ Early Bearing Wear: 27%
#   - 🔥 Thermal Degradation: 15%
#
# That IS genuinely "ranked by probability" now — multiple candidate failure
# modes scored against the same signals, normalized, and ordered.
