"""Core environment for support operations triage and SLA management."""

from __future__ import annotations

from collections import Counter
from typing import Any, Dict, List, Optional
from uuid import uuid4

from openenv.core.env_server.interfaces import Environment

try:
    from support_ops_triage_env.grading import grade_episode, queue_progress
    from support_ops_triage_env.models import (
        RewardSignal,
        SupportOpsTriageAction,
        SupportOpsTriageObservation,
        SupportOpsTriageState,
        TaskBrief,
        TicketSnapshot,
    )
    from support_ops_triage_env.task_bank import (
        default_task_id,
        get_task,
        list_task_briefs,
    )
except ImportError:
    from grading import grade_episode, queue_progress
    from models import (
        RewardSignal,
        SupportOpsTriageAction,
        SupportOpsTriageObservation,
        SupportOpsTriageState,
        TaskBrief,
        TicketSnapshot,
    )
    from task_bank import default_task_id, get_task, list_task_briefs


class SupportOpsTriageEnvironment(Environment):
    """Real-world customer support triage simulator with dense rewards."""

    SUPPORTS_CONCURRENT_SESSIONS: bool = True
    latest_grade: Dict[str, Any] = {}

    def __init__(self):
        self._state = SupportOpsTriageState(episode_id=str(uuid4()), step_count=0)
        self.expected_by_id: Dict[str, Dict[str, object]] = {}
        self._task_brief: Optional[TaskBrief] = None
        self._action_hash_counts: Counter[str] = Counter()

    def _build_snapshot(self, raw: Dict[str, Any]) -> TicketSnapshot:
        return TicketSnapshot(
            ticket_id=raw["ticket_id"],
            subject=raw["subject"],
            customer_tier=raw["customer_tier"],
            sla_minutes_remaining=int(raw["sla_minutes_remaining"]),
            status=raw.get("status", "open"),
        )

    def _task_brief_by_id(self, task_id: str) -> TaskBrief:
        for item in list_task_briefs():
            if item["task_id"] == task_id:
                return TaskBrief.model_validate(item)
        raise KeyError(task_id)

    def _current_messages(self) -> List[str]:
        msgs: List[str] = []
        if self._state.focus_ticket_id:
            msgs.append(f"Focused ticket: {self._state.focus_ticket_id}")
        p1_open = [
            t.ticket_id
            for t in self._state.tickets
            if t.status != "resolved" and t.priority == "P1"
        ]
        if p1_open:
            msgs.append("P1 unresolved tickets: " + ", ".join(sorted(p1_open)))
        unresolved = [t for t in self._state.tickets if t.status != "resolved"]
        msgs.append(f"Unresolved ticket count: {len(unresolved)}")
        return msgs

    def _update_progress(self) -> float:
        progress = queue_progress(self._state.tickets, self.expected_by_id)
        self._state.progress_score = round(progress, 6)
        return progress

    def _decrement_sla(self) -> float:
        """Decay SLA clocks to penalize slow policies and loops."""
        penalty = 0.0
        for ticket in self._state.tickets:
            if ticket.status == "resolved":
                continue
            decay = 5 if ticket.customer_tier == "enterprise" else 4
            ticket.sla_minutes_remaining = max(0, ticket.sla_minutes_remaining - decay)
            if ticket.sla_minutes_remaining == 0:
                penalty -= 0.06
        return penalty

    def _action_fingerprint(self, action: SupportOpsTriageAction) -> str:
        payload = action.model_dump(exclude_none=True)
        return str(sorted(payload.items()))

    def _validate_ticket(self, ticket_id: Optional[str]) -> Optional[TicketSnapshot]:
        if not ticket_id:
            return None
        for ticket in self._state.tickets:
            if ticket.ticket_id == ticket_id:
                return ticket
        return None

    def _apply_action(self, action: SupportOpsTriageAction) -> tuple[List[str], float, float]:
        messages: List[str] = []
        action_penalty = 0.0
        milestone_bonus = 0.0

        op = action.operation
        ticket = self._validate_ticket(action.ticket_id)

        if op in {
            "focus",
            "classify",
            "set_priority",
            "assign_queue",
            "escalate",
            "respond",
            "resolve",
        } and ticket is None:
            messages.append("Invalid ticket_id for operation.")
            return messages, -0.12, 0.0

        if op == "focus":
            self._state.focus_ticket_id = action.ticket_id
            messages.append(f"Focused on {action.ticket_id}.")
        elif op == "classify":
            if action.classification is None:
                return ["Missing classification field."], -0.08, 0.0
            ticket.classification = action.classification
            messages.append(f"Set classification of {ticket.ticket_id} to {action.classification}.")
        elif op == "set_priority":
            if action.priority is None:
                return ["Missing priority field."], -0.08, 0.0
            ticket.priority = action.priority
            messages.append(f"Set priority of {ticket.ticket_id} to {action.priority}.")
        elif op == "assign_queue":
            if action.queue is None:
                return ["Missing queue field."], -0.08, 0.0
            ticket.assigned_queue = action.queue
            messages.append(f"Assigned {ticket.ticket_id} to {action.queue}.")
        elif op == "escalate":
            if action.escalate is None:
                return ["Missing escalate field."], -0.08, 0.0
            ticket.escalated = action.escalate
            messages.append(f"Escalation for {ticket.ticket_id} set to {action.escalate}.")
        elif op == "respond":
            if not action.response_template:
                return ["Missing response_template field."], -0.08, 0.0
            ticket.response_template = action.response_template
            messages.append(f"Applied response template for {ticket.ticket_id}.")
        elif op == "resolve":
            if ticket.status == "resolved":
                return [f"Ticket {ticket.ticket_id} already resolved."], -0.05, 0.0
            ticket.status = "resolved"
            ticket.note = action.note
            milestone_bonus += 0.03
            messages.append(f"Resolved {ticket.ticket_id}.")
        elif op == "summarize":
            if not action.summary or len(action.summary.strip()) < 18:
                return ["Summary too short to be useful."], -0.06, 0.0
            self._state.action_history.append(
                {"summary": action.summary.strip(), "step": self._state.step_count}
            )
            milestone_bonus += 0.04
            messages.append("Stored shift summary.")
        elif op == "finish":
            messages.append("Finish requested.")
        else:
            action_penalty -= 0.08
            messages.append(f"Unsupported operation '{op}'.")

        return messages, action_penalty, milestone_bonus

    def _build_observation(
        self,
        reward_signal: RewardSignal,
        messages: List[str],
        done: bool,
    ) -> SupportOpsTriageObservation:
        return SupportOpsTriageObservation(
            task_id=self._state.task_id,
            difficulty=self._state.difficulty,
            objective=self._state.objective,
            queue_snapshot=self._state.tickets,
            focus_ticket_id=self._state.focus_ticket_id,
            pending_count=len([t for t in self._state.tickets if t.status != "resolved"]),
            progress_score=self._state.progress_score,
            messages=messages,
            done=done,
            reward=reward_signal.total_reward,
            reward_signal=reward_signal,
            metadata={
                "task_id": self._state.task_id,
                "step_count": self._state.step_count,
                "final_grade": self._state.final_grade if done else None,
            },
        )

    def reset(
        self,
        seed: Optional[int] = None,
        episode_id: Optional[str] = None,
        task_id: Optional[str] = None,
        **kwargs: Any,
    ) -> SupportOpsTriageObservation:
        selected_task = task_id or kwargs.get("task_id") or default_task_id()
        task = get_task(selected_task)
        brief = self._task_brief_by_id(selected_task)
        tickets = [self._build_snapshot(raw) for raw in task["tickets"]]

        self._state = SupportOpsTriageState(
            episode_id=episode_id or str(uuid4()),
            step_count=0,
            task_id=selected_task,
            difficulty=brief.difficulty,
            objective=brief.objective,
            focus_ticket_id=tickets[0].ticket_id if tickets else None,
            max_steps=int(task["max_steps"]),
            tickets=tickets,
            total_reward=0.0,
            progress_score=0.0,
            action_history=[],
            task_completed=False,
            final_grade=0.0,
        )
        self.expected_by_id = {
            t["ticket_id"]: t["expected"]  # type: ignore[index]
            for t in task["tickets"]  # type: ignore[index]
        }
        self._task_brief = brief
        self._action_hash_counts.clear()
        self._update_progress()

        reward_signal = RewardSignal(
            reason="Environment reset complete.",
            total_reward=0.0,
        )
        return self._build_observation(
            reward_signal=reward_signal,
            messages=["Support queue initialized."],
            done=False,
        )

    def step(self, action: SupportOpsTriageAction) -> SupportOpsTriageObservation:  # type: ignore[override]
        self._state.step_count += 1
        pre_progress = self._state.progress_score

        action_hash = self._action_fingerprint(action)
        self._action_hash_counts[action_hash] += 1
        loop_penalty = -0.04 if self._action_hash_counts[action_hash] >= 3 else 0.0

        action_messages, action_penalty, milestone_bonus = self._apply_action(action)
        sla_penalty = self._decrement_sla()

        post_progress = self._update_progress()
        delta_progress = post_progress - pre_progress

        unresolved = [t for t in self._state.tickets if t.status != "resolved"]
        done = (
            action.operation == "finish"
            or self._state.step_count >= self._state.max_steps
            or len(unresolved) == 0
        )

        if action.operation == "finish" and post_progress < 0.7:
            action_penalty -= 0.2
            action_messages.append("Finishing too early: critical triage items remain.")

        total_reward = (
            (delta_progress * 1.8)
            + milestone_bonus
            + action_penalty
            + loop_penalty
            + sla_penalty
            - 0.01  # small step cost to discourage random exploration loops
        )
        total_reward = float(round(total_reward, 6))
        self._state.total_reward = round(self._state.total_reward + total_reward, 6)

        self._state.action_history.append(
            {
                "step": self._state.step_count,
                "action": action.model_dump(exclude_none=True),
                "delta_progress": round(delta_progress, 6),
                "reward": total_reward,
            }
        )

        if done:
            score, breakdown, passed = grade_episode(
                task_id=self._state.task_id,
                difficulty=self._state.difficulty,
                tickets=self._state.tickets,
                expected_by_id=self.expected_by_id,
                step_count=self._state.step_count,
            )
            self._state.task_completed = passed
            self._state.final_grade = round(score, 6)
            SupportOpsTriageEnvironment.latest_grade = {
                "task_id": self._state.task_id,
                "score": self._state.final_grade,
                "passed": passed,
                "breakdown": breakdown,
                "step_count": self._state.step_count,
            }
            action_messages.append(
                f"Episode graded. score={self._state.final_grade:.3f}, passed={passed}."
            )

        reward_signal = RewardSignal(
            delta_progress=round(delta_progress, 6),
            milestone_bonus=round(milestone_bonus, 6),
            action_penalty=round(action_penalty, 6),
            loop_penalty=round(loop_penalty, 6),
            sla_penalty=round(sla_penalty, 6),
            total_reward=total_reward,
            reason="; ".join(action_messages) if action_messages else "No-op action.",
        )

        messages = self._current_messages() + action_messages
        return self._build_observation(reward_signal=reward_signal, messages=messages, done=done)

    @property
    def state(self) -> SupportOpsTriageState:
        return self._state
