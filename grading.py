"""Deterministic graders for task-level and episode-level scoring."""

from __future__ import annotations

from typing import Dict, Iterable, List, Tuple

try:
    from .models import SupportOpsTriageAction, TicketSnapshot
except ImportError:
    from models import SupportOpsTriageAction, TicketSnapshot

FIELD_WEIGHTS: Dict[str, float] = {
    "classification": 0.2,
    "priority": 0.2,
    "assigned_queue": 0.2,
    "escalated": 0.15,
    "response_template": 0.1,
    "resolved": 0.15,
}

PASS_THRESHOLDS = {
    "easy": 0.82,
    "medium": 0.78,
    "hard": 0.74,
}


def ticket_score(ticket: TicketSnapshot, expected: Dict[str, object]) -> float:
    """Compute weighted score for a single ticket in [0.0, 1.0]."""
    points = 0.0

    if ticket.classification == expected.get("classification"):
        points += FIELD_WEIGHTS["classification"]
    if ticket.priority == expected.get("priority"):
        points += FIELD_WEIGHTS["priority"]
    if ticket.assigned_queue == expected.get("assigned_queue"):
        points += FIELD_WEIGHTS["assigned_queue"]
    if ticket.escalated == bool(expected.get("escalated", False)):
        points += FIELD_WEIGHTS["escalated"]
    if ticket.response_template == expected.get("response_template"):
        points += FIELD_WEIGHTS["response_template"]

    resolve_required = bool(expected.get("resolve_required", False))
    is_resolved = ticket.status == "resolved"
    if resolve_required == is_resolved:
        points += FIELD_WEIGHTS["resolved"]

    return min(max(points, 0.0), 1.0)


def queue_progress(
    tickets: Iterable[TicketSnapshot],
    expected_by_id: Dict[str, Dict[str, object]],
) -> float:
    """Compute aggregate progress over all tickets in [0.0, 1.0]."""
    ticket_list = list(tickets)
    if not ticket_list:
        return 0.0

    total = 0.0
    for ticket in ticket_list:
        expected = expected_by_id.get(ticket.ticket_id, {})
        total += ticket_score(ticket, expected)
    return total / len(ticket_list)


def _efficiency_factor(step_count: int, ticket_count: int) -> float:
    """Return efficiency factor in [0.0, 1.0] based on action budget usage."""
    # Around 5 actions per ticket is efficient for this environment.
    optimal = max(ticket_count * 5, 1)
    if step_count <= optimal:
        return 1.0
    if step_count >= 2 * optimal:
        return 0.0
    return max(0.0, 1.0 - ((step_count - optimal) / float(optimal)))


def grade_episode(
    task_id: str,
    difficulty: str,
    tickets: List[TicketSnapshot],
    expected_by_id: Dict[str, Dict[str, object]],
    step_count: int,
) -> Tuple[float, Dict[str, float], bool]:
    """Compute final deterministic score, breakdown, and pass/fail."""
    quality = queue_progress(tickets, expected_by_id)
    efficiency = _efficiency_factor(step_count=step_count, ticket_count=len(tickets))
    score = min(max((quality * 0.85) + (efficiency * 0.15), 0.0), 1.0)

    threshold = PASS_THRESHOLDS.get(difficulty, 0.8)
    passed = score >= threshold
    breakdown = {
        "quality": round(quality, 6),
        "efficiency": round(efficiency, 6),
        "threshold": round(threshold, 6),
    }
    return score, breakdown, passed


def trajectory_to_actions(actions: List[dict]) -> List[SupportOpsTriageAction]:
    """Convert untyped dictionaries into validated action models."""
    parsed: List[SupportOpsTriageAction] = []
    for payload in actions:
        parsed.append(SupportOpsTriageAction.model_validate(payload))
    return parsed
