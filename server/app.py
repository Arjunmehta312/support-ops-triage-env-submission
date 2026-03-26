# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app exposing OpenEnv APIs plus hackathon evaluation endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Body

try:
    from openenv.core.env_server.http_server import create_app
except Exception as e:  # pragma: no cover
    raise ImportError(
        "openenv is required for the web interface. Install dependencies with '\n    uv sync\n'"
    ) from e

try:
    from support_ops_triage_env.baseline_runner import run_baseline
    from support_ops_triage_env.grading import grade_episode, trajectory_to_actions
    from support_ops_triage_env.models import (
        BaselineResponse,
        GraderRequest,
        GraderResponse,
        SupportOpsTriageAction,
        SupportOpsTriageObservation,
    )
    from support_ops_triage_env.task_bank import list_task_briefs
    from support_ops_triage_env.server.support_ops_triage_env_environment import (
        SupportOpsTriageEnvironment,
    )
except ModuleNotFoundError:
    from baseline_runner import run_baseline
    from grading import grade_episode, trajectory_to_actions
    from models import (
        BaselineResponse,
        GraderRequest,
        GraderResponse,
        SupportOpsTriageAction,
        SupportOpsTriageObservation,
    )
    from task_bank import list_task_briefs
    from server.support_ops_triage_env_environment import SupportOpsTriageEnvironment


_shared_env = SupportOpsTriageEnvironment()


def _env_factory() -> SupportOpsTriageEnvironment:
    # Keep one shared instance for deterministic endpoint introspection.
    return _shared_env


# Create the app with web interface and README integration
app = create_app(
    _env_factory,
    SupportOpsTriageAction,
    SupportOpsTriageObservation,
    env_name="support_ops_triage_env",
    max_concurrent_envs=1,
)


@app.get("/tasks")
def tasks() -> Dict[str, Any]:
    """Return available tasks and action schema for policy runners."""
    schema = SupportOpsTriageAction.model_json_schema()
    required_fields = schema.get("required", [])
    operation_enum = (
        schema.get("properties", {})
        .get("operation", {})
        .get("enum", [])
    )
    return {
        "tasks": list_task_briefs(),
        "action_schema": {
            "required": required_fields,
            "operation_values": operation_enum,
            "full_schema": schema,
        },
    }


@app.get("/grader", response_model=GraderResponse)
def grader_latest() -> GraderResponse:
    """Return latest completed episode grade."""
    latest = SupportOpsTriageEnvironment.latest_grade or {
        "task_id": "",
        "score": 0.0,
        "passed": False,
        "breakdown": {},
    }
    return GraderResponse.model_validate(latest)


@app.post("/grader", response_model=GraderResponse)
def grader_from_trajectory(payload: GraderRequest = Body(...)) -> GraderResponse:
    """Grade a trajectory by replaying actions from reset for deterministic scoring."""
    env = SupportOpsTriageEnvironment()
    try:
        env.reset(task_id=payload.task_id)
        for action in trajectory_to_actions([a.model_dump(exclude_none=True) for a in payload.actions]):
            obs = env.step(action)
            if obs.done:
                break

        score, breakdown, passed = grade_episode(
            task_id=env.state.task_id,
            difficulty=env.state.difficulty,
            tickets=env.state.tickets,
            expected_by_id=env.expected_by_id,
            step_count=env.state.step_count,
        )
        return GraderResponse(
            task_id=payload.task_id,
            score=round(score, 6),
            breakdown=breakdown,
            passed=passed,
        )
    finally:
        env.close()


@app.post("/baseline", response_model=BaselineResponse)
def baseline() -> BaselineResponse:
    """Run baseline policy over all tasks and return reproducible scores."""
    return run_baseline()


def main(host: str = "0.0.0.0", port: int = 8000):
    """
    Entry point for direct execution via uv run or python -m.

    This function enables running the server without Docker:
        uv run --project . server
        uv run --project . server --port 8001
        python -m support_ops_triage_env.server.app

    Args:
        host: Host address to bind to (default: "0.0.0.0")
        port: Port number to listen on (default: 8000)

    For production deployments, consider using uvicorn directly with
    multiple workers:
        uvicorn support_ops_triage_env.server.app:app --workers 4
    """
    import uvicorn

    uvicorn.run(app, host=host, port=port)


if __name__ == '__main__':
    main()
