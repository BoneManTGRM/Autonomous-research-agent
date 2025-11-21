"""Metrics for computing RYE (Repair Yield per Energy).

This module defines helper functions to compute ΔR (improvement) and E (effort)
for each cycle of the research agent. The RYE metric is central to the
reparodynamic analysis of the agent: higher RYE means more efficient repairs.
"""

from typing import Dict, List


def compute_delta_r(issues_before: int, issues_after: int, repairs_applied: int) -> float:
    """Compute the improvement ΔR for a cycle.

    Args:
        issues_before (int): Number of issues detected before repairs.
        issues_after (int): Number of issues detected after repairs.
        repairs_applied (int): Count of repair actions applied.

    Returns:
        float: The improvement score ΔR. This is a simple heuristic; users may
            refine it based on specific tasks. Here ΔR is the number of
            issues resolved or, if none, the number of repairs applied.
    """
    if issues_before > 0:
        delta = issues_before - issues_after
    else:
        # If there were no issues to begin with, improvements come from
        # repair actions themselves. A minimal improvement of 0.1 per repair.
        delta = repairs_applied * 0.1
    return float(delta)


def compute_energy(actions_taken: List[Dict[str, str]]) -> float:
    """Estimate the effort (E) expended during the cycle.

    Args:
        actions_taken (List[Dict[str, str]]): A list of action dictionaries
            representing the operations executed during the cycle.

    Returns:
        float: A numeric cost representing the energy. Each action
            contributes 1 unit of cost by default.
    """
    # Each action counts as 1 unit of effort. Different weighting schemes
    # could be implemented to account for more expensive operations.
    return float(len(actions_taken)) if actions_taken else 1.0


def compute_rye(delta_r: float, energy_e: float) -> float:
    """Compute the Repair Yield per Energy (RYE) metric.

    Args:
        delta_r (float): Improvement achieved during the cycle.
        energy_e (float): Effort spent during the cycle.

    Returns:
        float: The RYE value. Defaults to 0 if energy is zero.
    """
    if energy_e == 0:
        return 0.0
    return delta_r / energy_e
