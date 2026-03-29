# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Typed models for the Support Operations Triage environment."""

from typing import Any, Dict, List, Literal, Optional

from openenv.core.env_server.types import Action, Observation, State
from pydantic import BaseModel, Field


TicketClassification = Literal[
    "billing",
    "technical",
    "security",
    "account",
    "abuse",
    "incident",
    "request",
]

TicketPriority = Literal["P1", "P2", "P3", "P4"]
TicketQueue = Literal[
    "billing_ops",
    "platform_ops",
    "security_ops",
    "trust_safety",
    "account_ops",
    "incident_command",
]
AgentOperation = Literal[
    "focus",
    "classify",
    "set_priority",
    "assign_queue",
    "escalate",
    "respond",
    "resolve",
    "summarize",
    "finish",
]


class RewardSignal(BaseModel):
    """Explainable reward decomposition for training diagnostics."""

    delta_progress: float = Field(default=0.0)
    milestone_bonus: float = Field(default=0.0)
    action_penalty: float = Field(default=0.0)
    loop_penalty: float = Field(default=0.0)
    sla_penalty: float = Field(default=0.0)
    total_reward: float = Field(default=0.0)
    reason: str = Field(default="")


class TicketSnapshot(BaseModel):
    """Public ticket view shown to the agent."""

    ticket_id: str
    subject: str
    customer_tier: Literal["free", "pro", "enterprise"]
    sla_minutes_remaining: int
    status: Literal["open", "in_progress", "resolved"]
    classification: Optional[TicketClassification] = None
    priority: Optional[TicketPriority] = None
    assigned_queue: Optional[TicketQueue] = None
    escalated: bool = False
    response_template: Optional[str] = None
    note: Optional[str] = None


class TaskBrief(BaseModel):
    """Task metadata exposed to clients and evaluators."""

    task_id: str
    title: str
    difficulty: Literal["easy", "medium", "hard"]
    objective: str
    max_steps: int


class ObservationMessage(BaseModel):
    """Structured message entry for web/UI rendering."""

    sender_id: str = Field(default="env")
    category: str = Field(default="info")
    content: str


class SupportOpsTriageAction(Action):
    """Structured action schema used by policy models and human agents."""

    operation: AgentOperation = Field(
        ..., description="Action type to apply to ticket workflow state"
    )
    ticket_id: Optional[str] = Field(
        default=None,
        description="Ticket identifier to operate on; optional for summarize/finish",
    )
    classification: Optional[TicketClassification] = Field(default=None)
    priority: Optional[TicketPriority] = Field(default=None)
    queue: Optional[TicketQueue] = Field(default=None)
    escalate: Optional[bool] = Field(default=None)
    response_template: Optional[str] = Field(default=None)
    note: Optional[str] = Field(default=None)
    summary: Optional[str] = Field(default=None)


class SupportOpsTriageObservation(Observation):
    """Observation containing queue state and incremental feedback."""

    task_id: str = Field(default="")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    objective: str = Field(default="")
    queue_snapshot: List[TicketSnapshot] = Field(default_factory=list)
    focus_ticket_id: Optional[str] = Field(default=None)
    pending_count: int = Field(default=0)
    progress_score: float = Field(default=0.0)
    messages: List[ObservationMessage] = Field(default_factory=list)
    reward_signal: RewardSignal = Field(default_factory=RewardSignal)


class SupportOpsTriageState(State):
    """Extended state for transparency, replay, and grading."""

    task_id: str = Field(default="")
    difficulty: Literal["easy", "medium", "hard"] = Field(default="easy")
    objective: str = Field(default="")
    focus_ticket_id: Optional[str] = Field(default=None)
    max_steps: int = Field(default=0)
    total_reward: float = Field(default=0.0)
    progress_score: float = Field(default=0.0)
    tickets: List[TicketSnapshot] = Field(default_factory=list)
    action_history: List[Dict[str, Any]] = Field(default_factory=list)
    task_completed: bool = Field(default=False)
    final_grade: float = Field(default=0.0)


class GraderRequest(BaseModel):
    """Optional direct grading payload for offline trajectory evaluation."""

    task_id: str
    actions: List[SupportOpsTriageAction] = Field(default_factory=list)


class GraderResponse(BaseModel):
    """Programmatic score response in [0.0, 1.0]."""

    task_id: str
    score: float
    breakdown: Dict[str, float] = Field(default_factory=dict)
    passed: bool = False


class BaselineResponse(BaseModel):
    """Baseline benchmark payload for all tasks."""

    model: str
    provider: Literal["openai", "openrouter", "heuristic"]
    task_scores: Dict[str, float]
    average_score: float
    run_seed: int
