"""Baseline inference runner for support triage tasks."""

from __future__ import annotations

import json
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Literal, TYPE_CHECKING

try:
    from .grading import grade_episode
    from .models import BaselineResponse, SupportOpsTriageAction
    from .server.support_ops_triage_env_environment import SupportOpsTriageEnvironment
    from .task_bank import list_task_briefs
except ImportError:
    from grading import grade_episode
    from models import BaselineResponse, SupportOpsTriageAction
    from server.support_ops_triage_env_environment import SupportOpsTriageEnvironment
    from task_bank import list_task_briefs

if TYPE_CHECKING:
    from openai import OpenAI as OpenAIClient
else:
    OpenAIClient = Any

try:
    from openai import OpenAI
except ImportError:  # pragma: no cover
    OpenAI = None  # type: ignore[assignment]


@dataclass
class BaselineConfig:
    model: str = "gpt-4.1-mini"
    max_steps_per_task: int = 36
    run_seed: int = 42


def _heuristic_action(observation: dict) -> SupportOpsTriageAction:
    """Fallback deterministic policy when API key is not provided."""
    tickets = observation.get("queue_snapshot", [])
    unresolved = [t for t in tickets if t.get("status") != "resolved"]
    if not unresolved:
        return SupportOpsTriageAction(operation="finish")

    target = sorted(unresolved, key=lambda t: t.get("sla_minutes_remaining", 999))[0]
    ticket_id = target["ticket_id"]

    if not target.get("classification"):
        subject = (target.get("subject") or "").lower()
        if "security" in subject or "unauthorized" in subject:
            cls = "security"
        elif "invoice" in subject or "refund" in subject:
            cls = "billing"
        elif "abuse" in subject or "harassment" in subject or "spam" in subject:
            cls = "abuse"
        elif "incident" in subject or "outage" in subject or "503" in subject:
            cls = "incident"
        elif "login" in subject or "mfa" in subject or "sso" in subject:
            cls = "account"
        else:
            cls = "request"
        return SupportOpsTriageAction(operation="classify", ticket_id=ticket_id, classification=cls)

    if not target.get("priority"):
        sla = int(target.get("sla_minutes_remaining", 999))
        if sla <= 30:
            priority = "P1"
        elif sla <= 90:
            priority = "P2"
        elif sla <= 180:
            priority = "P3"
        else:
            priority = "P4"
        return SupportOpsTriageAction(operation="set_priority", ticket_id=ticket_id, priority=priority)

    if not target.get("assigned_queue"):
        cls = target.get("classification")
        queue_map = {
            "billing": "billing_ops",
            "security": "security_ops",
            "incident": "incident_command",
            "abuse": "trust_safety",
            "account": "account_ops",
            "technical": "platform_ops",
            "request": "platform_ops",
        }
        return SupportOpsTriageAction(
            operation="assign_queue",
            ticket_id=ticket_id,
            queue=queue_map.get(cls, "platform_ops"),
        )

    if target.get("escalated") is False and int(target.get("sla_minutes_remaining", 999)) <= 80:
        return SupportOpsTriageAction(operation="escalate", ticket_id=ticket_id, escalate=True)

    if not target.get("response_template"):
        cls = target.get("classification")
        template_map = {
            "billing": "billing_refund_workflow",
            "security": "security_breach_containment",
            "incident": "major_incident_ack",
            "abuse": "trust_safety_intake",
            "account": "account_recovery_mfa",
            "request": "incident_status_update",
            "technical": "incident_status_update",
        }
        return SupportOpsTriageAction(
            operation="respond",
            ticket_id=ticket_id,
            response_template=template_map.get(cls, "incident_status_update"),
        )

    if target.get("status") != "resolved":
        return SupportOpsTriageAction(
            operation="resolve",
            ticket_id=ticket_id,
            note="Resolved by baseline heuristic after triage completion.",
        )

    return SupportOpsTriageAction(operation="finish")


def _extract_json_object(text: str) -> dict:
    """Best-effort extraction of one JSON object from model output."""
    text = text.strip()
    if text.startswith("{") and text.endswith("}"):
        return json.loads(text)

    start = text.find("{")
    end = text.rfind("}")
    if start >= 0 and end > start:
        return json.loads(text[start : end + 1])
    raise ValueError("No JSON object found in model output")


def _openai_action(
    client: OpenAIClient,
    model: str,
    task_brief: dict,
    observation: dict,
    run_seed: int,
) -> SupportOpsTriageAction:
    """Call OpenAI model for one policy step."""
    system_prompt = (
        "You are an expert support operations triage agent. "
        "Return exactly one JSON object for the next action with valid fields for the schema."
    )
    user_prompt = {
        "task": task_brief,
        "observation": observation,
        "action_schema": {
            "operation": [
                "focus",
                "classify",
                "set_priority",
                "assign_queue",
                "escalate",
                "respond",
                "resolve",
                "summarize",
                "finish",
            ],
            "required": ["operation"],
            "optional": [
                "ticket_id",
                "classification",
                "priority",
                "queue",
                "escalate",
                "response_template",
                "note",
                "summary",
            ],
        },
    }

    response = client.responses.create(
        model=model,
        temperature=0,
        seed=run_seed,
        input=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": json.dumps(user_prompt)},
        ],
    )
    payload = _extract_json_object(response.output_text)
    return SupportOpsTriageAction.model_validate(payload)


def _play_task(
    env: SupportOpsTriageEnvironment,
    task_id: str,
    provider: Literal["openai", "heuristic"],
    config: BaselineConfig,
    client: OpenAIClient | None,
) -> float:
    """Run a single task episode and return final grade."""
    obs = env.reset(task_id=task_id)
    task_brief = {
        "task_id": obs.task_id,
        "difficulty": obs.difficulty,
        "objective": obs.objective,
        "max_steps": env.state.max_steps,
    }

    for _ in range(config.max_steps_per_task):
        obs_payload = obs.model_dump()
        if provider == "openai" and client is not None:
            action = _openai_action(
                client=client,
                model=config.model,
                task_brief=task_brief,
                observation=obs_payload,
                run_seed=config.run_seed,
            )
        else:
            action = _heuristic_action(obs_payload)

        obs = env.step(action)
        if obs.done:
            return float(obs.metadata.get("final_grade", 0.0))

    # Timeout fallback, grade current state.
    score, _, _ = grade_episode(
        task_id=env.state.task_id,
        difficulty=env.state.difficulty,
        tickets=env.state.tickets,
        expected_by_id=env.expected_by_id,
        step_count=env.state.step_count,
    )
    return score


def run_baseline(config: BaselineConfig | None = None) -> BaselineResponse:
    """Execute baseline evaluation over all task difficulties."""
    cfg = config or BaselineConfig()
    api_key = os.getenv("OPENAI_API_KEY", "").strip()

    provider: Literal["openai", "heuristic"] = "heuristic"
    client = None
    if api_key and OpenAI is not None:
        provider = "openai"
        client = OpenAI(api_key=api_key)

    env = SupportOpsTriageEnvironment()
    scores: Dict[str, float] = {}
    try:
        for brief in list_task_briefs():
            task_id = str(brief["task_id"])
            scores[task_id] = round(
                _play_task(
                    env=env,
                    task_id=task_id,
                    provider=provider,
                    config=cfg,
                    client=client,
                ),
                6,
            )
    finally:
        env.close()

    average = round(sum(scores.values()) / max(len(scores), 1), 6)
    return BaselineResponse(
        model=cfg.model,
        provider=provider,
        task_scores=scores,
        average_score=average,
        run_seed=cfg.run_seed,
    )
