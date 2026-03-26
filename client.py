# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Support operations triage environment client."""

from typing import Dict

from openenv.core import EnvClient
from openenv.core.client_types import StepResult
from openenv.core.env_server.types import State

from .models import (
    RewardSignal,
    SupportOpsTriageAction,
    SupportOpsTriageObservation,
    SupportOpsTriageState,
    TicketSnapshot,
)


class SupportOpsTriageEnv(
    EnvClient[SupportOpsTriageAction, SupportOpsTriageObservation, SupportOpsTriageState]
):
    """
    Client for the Support Ops Triage Env Environment.

    This client maintains a persistent WebSocket connection to the environment server,
    enabling efficient multi-step interactions with lower latency.
    Each client instance has its own dedicated environment session on the server.

    Example:
        >>> # Connect to a running server
        >>> with SupportOpsTriageEnv(base_url="http://localhost:8000") as client:
        ...     result = client.reset()
        ...     print(result.observation.echoed_message)
        ...
        ...     result = client.step(SupportOpsTriageAction(message="Hello!"))
        ...     print(result.observation.echoed_message)

    Example with Docker:
        >>> # Automatically start container and connect
        >>> client = SupportOpsTriageEnv.from_docker_image("support_ops_triage_env-env:latest")
        >>> try:
        ...     result = client.reset()
        ...     result = client.step(SupportOpsTriageAction(message="Test"))
        ... finally:
        ...     client.close()
    """

    def _step_payload(self, action: SupportOpsTriageAction) -> Dict:
        """
        Convert SupportOpsTriageAction to JSON payload for step message.

        Args:
            action: SupportOpsTriageAction instance

        Returns:
            Dictionary representation suitable for JSON encoding
        """
        return action.model_dump(exclude_none=True)

    def _parse_result(self, payload: Dict) -> StepResult[SupportOpsTriageObservation]:
        """
        Parse server response into StepResult[SupportOpsTriageObservation].

        Args:
            payload: JSON response data from server

        Returns:
            StepResult with SupportOpsTriageObservation
        """
        obs_data = payload.get("observation", {})
        observation = SupportOpsTriageObservation(
            task_id=obs_data.get("task_id", ""),
            difficulty=obs_data.get("difficulty", "easy"),
            objective=obs_data.get("objective", ""),
            queue_snapshot=[
                TicketSnapshot.model_validate(t)
                for t in obs_data.get("queue_snapshot", [])
            ],
            focus_ticket_id=obs_data.get("focus_ticket_id"),
            pending_count=obs_data.get("pending_count", 0),
            progress_score=obs_data.get("progress_score", 0.0),
            messages=obs_data.get("messages", []),
            reward_signal=RewardSignal.model_validate(obs_data.get("reward_signal", {})),
            done=payload.get("done", False),
            reward=payload.get("reward"),
            metadata=obs_data.get("metadata", {}),
        )

        return StepResult(
            observation=observation,
            reward=payload.get("reward"),
            done=payload.get("done", False),
        )

    def _parse_state(self, payload: Dict) -> SupportOpsTriageState:
        """
        Parse server response into State object.

        Args:
            payload: JSON response from state request

        Returns:
            State object with episode_id and step_count
        """
        return SupportOpsTriageState(
            episode_id=payload.get("episode_id"),
            step_count=payload.get("step_count", 0),
            task_id=payload.get("task_id", ""),
            difficulty=payload.get("difficulty", "easy"),
            objective=payload.get("objective", ""),
            focus_ticket_id=payload.get("focus_ticket_id"),
            max_steps=payload.get("max_steps", 0),
            total_reward=payload.get("total_reward", 0.0),
            progress_score=payload.get("progress_score", 0.0),
            tickets=[TicketSnapshot.model_validate(t) for t in payload.get("tickets", [])],
            action_history=payload.get("action_history", []),
            task_completed=payload.get("task_completed", False),
            final_grade=payload.get("final_grade", 0.0),
        )
