"""Pre-submission validator for hackathon readiness checks."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import time
from typing import Any

import requests

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
TIMEOUT = 20


def _print(msg: str) -> None:
    print(f"[validator] {msg}")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _get(path: str) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.get(url, timeout=TIMEOUT)
    _check(resp.status_code == 200, f"GET {path} failed: {resp.status_code} {resp.text[:250]}")
    return resp.json()


def _post(path: str, payload: dict | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    resp = requests.post(url, json=payload or {}, timeout=TIMEOUT)
    _check(resp.status_code == 200, f"POST {path} failed: {resp.status_code} {resp.text[:250]}")
    return resp.json()


def _validate_openenv() -> None:
    _print("Running openenv validate against local environment folder...")
    cmd = ["openenv", "validate", "."]
    result = subprocess.run(cmd, capture_output=True, text=True)
    if result.returncode != 0:
        raise AssertionError(
            "openenv validate failed.\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )


def main() -> int:
    _print(f"Base URL: {BASE_URL}")
    _validate_openenv()

    health = _get("/health")
    _check(health.get("status") == "healthy", "Health endpoint did not report healthy")

    tasks_payload = _get("/tasks")
    tasks = tasks_payload.get("tasks", [])
    _check(len(tasks) >= 3, "Expected at least 3 tasks")

    reset_payload = _post("/reset", {"task_id": tasks[0]["task_id"]})
    _check("observation" in reset_payload, "Reset response missing observation")

    step_payload = _post(
        "/step",
        {
            "action": {
                "operation": "focus",
                "ticket_id": reset_payload["observation"]["queue_snapshot"][0]["ticket_id"],
            }
        },
    )
    _check("observation" in step_payload, "Step response missing observation")

    state_payload = _get("/state")
    _check("step_count" in state_payload, "State response missing step_count")

    baseline_payload = _post("/baseline")
    _check("task_scores" in baseline_payload, "Baseline response missing task_scores")

    grader_payload = _get("/grader")
    _check("score" in grader_payload, "Grader response missing score")

    for task_id, score in baseline_payload["task_scores"].items():
        _check(0.0 <= float(score) <= 1.0, f"Score out of range for {task_id}: {score}")

    _print("All checks passed.")
    print(json.dumps({"status": "ok", "base_url": BASE_URL}, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
