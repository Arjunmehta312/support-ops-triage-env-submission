"""Deterministic task bank for support operations triage training."""

from __future__ import annotations

from copy import deepcopy
from typing import Dict, List

TaskData = Dict[str, object]

_TASKS: Dict[str, TaskData] = {
    "easy_inbox_hygiene": {
        "task_id": "easy_inbox_hygiene",
        "title": "Inbox Hygiene for SMB Queue",
        "difficulty": "easy",
        "objective": (
            "Triage all inbound tickets with correct classification, priority, queue assignment, "
            "and close tickets with a clear response template."
        ),
        "max_steps": 18,
        "tickets": [
            {
                "ticket_id": "E-101",
                "subject": "Refund charged twice on invoice INV-4412",
                "body": "Customer was charged twice after card retry. Wants immediate refund.",
                "customer_tier": "pro",
                "sla_minutes_remaining": 180,
                "status": "open",
                "expected": {
                    "classification": "billing",
                    "priority": "P2",
                    "assigned_queue": "billing_ops",
                    "escalated": False,
                    "response_template": "billing_refund_workflow",
                    "resolve_required": True,
                },
            },
            {
                "ticket_id": "E-102",
                "subject": "Cannot reset MFA after changing phone",
                "body": "User cannot login after replacing phone and losing OTP app.",
                "customer_tier": "free",
                "sla_minutes_remaining": 240,
                "status": "open",
                "expected": {
                    "classification": "account",
                    "priority": "P3",
                    "assigned_queue": "account_ops",
                    "escalated": False,
                    "response_template": "account_recovery_mfa",
                    "resolve_required": True,
                },
            },
            {
                "ticket_id": "E-103",
                "subject": "Service status question about yesterday incident",
                "body": "Customer asks if outage is resolved and if RCA is available.",
                "customer_tier": "enterprise",
                "sla_minutes_remaining": 120,
                "status": "open",
                "expected": {
                    "classification": "request",
                    "priority": "P3",
                    "assigned_queue": "platform_ops",
                    "escalated": False,
                    "response_template": "incident_status_update",
                    "resolve_required": True,
                },
            },
        ],
    },
    "medium_vip_sla_mix": {
        "task_id": "medium_vip_sla_mix",
        "title": "VIP + SLA Breach Prevention",
        "difficulty": "medium",
        "objective": (
            "Prevent SLA breaches while handling enterprise incidents and correctly escalating security-sensitive "
            "tickets."
        ),
        "max_steps": 28,
        "tickets": [
            {
                "ticket_id": "M-201",
                "subject": "EU customers cannot access dashboards",
                "body": "Multiple tenants report HTTP 503 from Frankfurt region.",
                "customer_tier": "enterprise",
                "sla_minutes_remaining": 35,
                "status": "open",
                "expected": {
                    "classification": "incident",
                    "priority": "P1",
                    "assigned_queue": "incident_command",
                    "escalated": True,
                    "response_template": "major_incident_ack",
                    "resolve_required": False,
                },
            },
            {
                "ticket_id": "M-202",
                "subject": "Suspicious admin login from unknown country",
                "body": "Audit log shows impossible travel and new API tokens created.",
                "customer_tier": "enterprise",
                "sla_minutes_remaining": 55,
                "status": "open",
                "expected": {
                    "classification": "security",
                    "priority": "P1",
                    "assigned_queue": "security_ops",
                    "escalated": True,
                    "response_template": "security_breach_containment",
                    "resolve_required": False,
                },
            },
            {
                "ticket_id": "M-203",
                "subject": "Need monthly invoice split by cost center",
                "body": "Finance wants two cost centers reflected in next billing cycle.",
                "customer_tier": "pro",
                "sla_minutes_remaining": 220,
                "status": "open",
                "expected": {
                    "classification": "billing",
                    "priority": "P3",
                    "assigned_queue": "billing_ops",
                    "escalated": False,
                    "response_template": "billing_invoice_customization",
                    "resolve_required": True,
                },
            },
            {
                "ticket_id": "M-204",
                "subject": "Abusive content reported in shared workspace",
                "body": "Customer attached screenshots of harassment in public project comments.",
                "customer_tier": "pro",
                "sla_minutes_remaining": 80,
                "status": "open",
                "expected": {
                    "classification": "abuse",
                    "priority": "P2",
                    "assigned_queue": "trust_safety",
                    "escalated": True,
                    "response_template": "trust_safety_intake",
                    "resolve_required": False,
                },
            },
        ],
    },
    "hard_incident_storm": {
        "task_id": "hard_incident_storm",
        "title": "Cross-Region Incident Storm",
        "difficulty": "hard",
        "objective": (
            "Handle a cascading incident storm with duplicate reports, VIP pressure, abuse noise, and near-term "
            "SLA deadlines while keeping triage quality high."
        ),
        "max_steps": 40,
        "tickets": [
            {
                "ticket_id": "H-301",
                "subject": "APAC checkout API timing out for enterprise tenants",
                "body": "Error rates jumped to 24 percent. Revenue impact reported.",
                "customer_tier": "enterprise",
                "sla_minutes_remaining": 20,
                "status": "open",
                "expected": {
                    "classification": "incident",
                    "priority": "P1",
                    "assigned_queue": "incident_command",
                    "escalated": True,
                    "response_template": "major_incident_ack",
                    "resolve_required": False,
                },
            },
            {
                "ticket_id": "H-302",
                "subject": "Duplicate: checkout failures in Singapore",
                "body": "Likely same root cause as platform outage, asks for ETA.",
                "customer_tier": "pro",
                "sla_minutes_remaining": 28,
                "status": "open",
                "expected": {
                    "classification": "incident",
                    "priority": "P1",
                    "assigned_queue": "incident_command",
                    "escalated": True,
                    "response_template": "incident_status_update",
                    "resolve_required": False,
                },
            },
            {
                "ticket_id": "H-303",
                "subject": "Unauthorized OAuth app added to org",
                "body": "Possible compromised admin account and data exfil concern.",
                "customer_tier": "enterprise",
                "sla_minutes_remaining": 24,
                "status": "open",
                "expected": {
                    "classification": "security",
                    "priority": "P1",
                    "assigned_queue": "security_ops",
                    "escalated": True,
                    "response_template": "security_breach_containment",
                    "resolve_required": False,
                },
            },
            {
                "ticket_id": "H-304",
                "subject": "Quarterly true-up invoice mismatch",
                "body": "Customer sees overage charges not matching seat logs.",
                "customer_tier": "enterprise",
                "sla_minutes_remaining": 95,
                "status": "open",
                "expected": {
                    "classification": "billing",
                    "priority": "P2",
                    "assigned_queue": "billing_ops",
                    "escalated": False,
                    "response_template": "billing_refund_workflow",
                    "resolve_required": True,
                },
            },
            {
                "ticket_id": "H-305",
                "subject": "Workspace owner locked out after SSO migration",
                "body": "SCIM sync removed owner role by mistake.",
                "customer_tier": "pro",
                "sla_minutes_remaining": 60,
                "status": "open",
                "expected": {
                    "classification": "account",
                    "priority": "P2",
                    "assigned_queue": "account_ops",
                    "escalated": False,
                    "response_template": "account_recovery_mfa",
                    "resolve_required": True,
                },
            },
            {
                "ticket_id": "H-306",
                "subject": "Harassment spam campaign in public templates",
                "body": "Bot accounts posting hate speech links repeatedly.",
                "customer_tier": "free",
                "sla_minutes_remaining": 70,
                "status": "open",
                "expected": {
                    "classification": "abuse",
                    "priority": "P2",
                    "assigned_queue": "trust_safety",
                    "escalated": True,
                    "response_template": "trust_safety_intake",
                    "resolve_required": False,
                },
            },
        ],
    },
}


def list_task_briefs() -> List[Dict[str, object]]:
    """Return public task metadata for endpoint responses."""
    briefs: List[Dict[str, object]] = []
    for task in _TASKS.values():
        briefs.append(
            {
                "task_id": task["task_id"],
                "title": task["title"],
                "difficulty": task["difficulty"],
                "objective": task["objective"],
                "max_steps": task["max_steps"],
            }
        )
    return briefs


def get_task(task_id: str) -> TaskData:
    """Return deep-copied task data by id."""
    if task_id not in _TASKS:
        available = ", ".join(sorted(_TASKS.keys()))
        raise KeyError(f"Unknown task_id '{task_id}'. Available: {available}")
    return deepcopy(_TASKS[task_id])


def default_task_id() -> str:
    """Return deterministic default task id."""
    return "easy_inbox_hygiene"


def task_ids() -> List[str]:
    """Return all task identifiers."""
    return sorted(_TASKS.keys())
