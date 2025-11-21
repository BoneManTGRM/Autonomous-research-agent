"""Biomarker analysis module for anti-aging / longevity mode.

This is a simple rule-based analyzer that:
- Takes a dict of biomarker values (e.g., labs, vitals),
- Computes rough risk / longevity flags,
- Produces a structured summary.

It is intentionally conservative and generic. It is a toy scoring system,
not medical advice. Real labs would plug in more detailed models here.

Example input:
    {
      "age": 45,
      "bmi": 28.0,
      "ldl": 140,
      "hdl": 40,
      "triglycerides": 180,
      "fasting_glucose": 105,
      "crp": 3.5
    }
"""

from __future__ import annotations

from typing import Any, Dict


def _score_range(value: float, low: float, high: float) -> int:
    """Return a simple risk score: 0=good, 1=borderline, 2=high-risk."""
    if value < low:
        return 1  # low isn't always good, mark as borderline
    if low <= value <= high:
        return 0
    return 2


def analyze_biomarkers(snapshot: Dict[str, Any]) -> Dict[str, Any]:
    """Analyze a biomarker snapshot and return a structured summary."""
    age = float(snapshot.get("age", 0) or 0)
    bmi = float(snapshot.get("bmi", 0) or 0)
    ldl = float(snapshot.get("ldl", 0) or 0)
    hdl = float(snapshot.get("hdl", 0) or 0)
    tg = float(snapshot.get("triglycerides", 0) or 0)
    glucose = float(snapshot.get("fasting_glucose", 0) or 0)
    crp = float(snapshot.get("crp", 0) or 0)

    scores: Dict[str, int] = {}

    # Very rough, non-medical ranges
    scores["bmi"] = _score_range(bmi, 18.5, 25.0)
    scores["ldl"] = _score_range(ldl, 0, 100)  # lower is better
    scores["hdl"] = _score_range(hdl, 40, 60)  # higher is better
    scores["triglycerides"] = _score_range(tg, 0, 150)
    scores["fasting_glucose"] = _score_range(glucose, 70, 99)
    scores["crp"] = _score_range(crp, 0, 3)

    total_score = sum(scores.values())
    if total_score <= 2:
        risk_band = "low"
    elif total_score <= 5:
        risk_band = "moderate"
    else:
        risk_band = "elevated"

    summary_lines = []
    if scores["bmi"] > 0:
        summary_lines.append("Body weight or composition may be suboptimal (BMI outside 18.5–25).")
    if scores["ldl"] > 0:
        summary_lines.append("LDL cholesterol is above ideal range; cardiovascular risk may be higher.")
    if scores["hdl"] > 0:
        summary_lines.append("HDL cholesterol is not in the optimal zone.")
    if scores["triglycerides"] > 0:
        summary_lines.append("Triglycerides are higher than ideal; metabolic health may be stressed.")
    if scores["fasting_glucose"] > 0:
        summary_lines.append("Fasting glucose suggests impaired metabolic flexibility or early dysregulation.")
    if scores["crp"] > 0:
        summary_lines.append("CRP indicates some degree of inflammation, which may impact aging and repair.")

    if not summary_lines:
        summary_lines.append("All tracked biomarkers are within basic reference ranges.")

    return {
        "age": age,
        "scores": scores,
        "total_score": total_score,
        "risk_band": risk_band,
        "summary": summary_lines,
    }
