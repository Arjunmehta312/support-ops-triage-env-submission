# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""FastAPI app exposing OpenEnv APIs plus hackathon evaluation endpoints."""

from __future__ import annotations

from typing import Any, Dict

from fastapi import Body
from fastapi.responses import HTMLResponse, Response

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


_LANDING_HTML = """<!doctype html>
<html lang=\"en\">
    <head>
        <meta charset=\"utf-8\" />
        <meta name=\"viewport\" content=\"width=device-width, initial-scale=1\" />
        <title>Support Ops Triage Environment</title>
        <style>
            :root {
                --bg: #0f172a;
                --panel: #111827;
                --panel-border: #1f2937;
                --text: #e5e7eb;
                --muted: #9ca3af;
                --accent: #38bdf8;
            }
            * { box-sizing: border-box; }
            body {
                margin: 0;
                font-family: "Segoe UI", "Helvetica Neue", Helvetica, sans-serif;
                color: var(--text);
                background:
                    radial-gradient(1100px 600px at 10% -10%, #1d4ed8 0%, rgba(29, 78, 216, 0) 55%),
                    radial-gradient(1000px 700px at 100% 0%, #0ea5e9 0%, rgba(14, 165, 233, 0) 45%),
                    var(--bg);
                min-height: 100vh;
                display: grid;
                place-items: center;
                padding: 24px;
            }
            .card {
                width: min(840px, 100%);
                background: linear-gradient(180deg, rgba(17, 24, 39, 0.92), rgba(15, 23, 42, 0.96));
                border: 1px solid var(--panel-border);
                border-radius: 16px;
                padding: 30px;
                box-shadow: 0 18px 45px rgba(2, 6, 23, 0.45);
            }
            h1 {
                margin: 0 0 10px;
                font-size: clamp(26px, 3vw, 36px);
                letter-spacing: 0.2px;
            }
            p {
                margin: 0;
                color: var(--muted);
                line-height: 1.58;
            }
            .meta {
                margin-top: 18px;
                font-size: 13px;
                color: #cbd5e1;
            }
            .grid {
                margin-top: 22px;
                display: grid;
                gap: 10px;
            }
            .row {
                display: grid;
                grid-template-columns: 150px 1fr;
                gap: 14px;
                align-items: center;
                border: 1px solid #263444;
                border-radius: 10px;
                padding: 10px 12px;
                background: rgba(15, 23, 42, 0.55);
            }
            .method {
                color: var(--accent);
                font-weight: 700;
                font-size: 13px;
            }
            code {
                color: #f8fafc;
                font-size: 13px;
            }
            a {
                color: var(--accent);
                text-decoration: none;
            }
            a:hover { text-decoration: underline; }
        </style>
    </head>
    <body>
        <main class=\"card\">
            <h1>Support Ops Triage Environment</h1>
            <p>
                Production-style OpenEnv API for support ticket triage evaluation. This Space exposes
                deterministic tasks, grading endpoints, and baseline execution endpoints.
            </p>
            <div class=\"meta\">Status: <a href=\"/health\">healthy check</a></div>
            <section class=\"grid\" aria-label=\"endpoint index\">
                <div class=\"row\"><span class=\"method\">GET</span><code>/health</code></div>
                <div class=\"row\"><span class=\"method\">GET</span><code>/tasks</code></div>
                <div class=\"row\"><span class=\"method\">POST</span><code>/reset</code></div>
                <div class=\"row\"><span class=\"method\">POST</span><code>/step</code></div>
                <div class=\"row\"><span class=\"method\">GET</span><code>/state</code></div>
                <div class=\"row\"><span class=\"method\">GET / POST</span><code>/grader</code></div>
                <div class=\"row\"><span class=\"method\">POST</span><code>/baseline</code></div>
            </section>
        </main>
    </body>
</html>
"""


@app.get("/", include_in_schema=False, response_class=HTMLResponse)
def landing() -> str:
        """Human-friendly landing page for Space viewers."""
        return _LANDING_HTML


@app.get("/favicon.ico", include_in_schema=False)
def favicon() -> Response:
        """Return empty favicon to avoid repeated browser 404s."""
        return Response(status_code=204)


@app.get("/web/health", include_in_schema=False)
def web_health_alias() -> Dict[str, str]:
        """Alias for health probes that target /web/health."""
        return {"status": "healthy"}


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
