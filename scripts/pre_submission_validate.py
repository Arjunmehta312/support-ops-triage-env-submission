"""Pre-submission validator for hackathon readiness checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys
import time
from typing import Any

import requests

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://localhost:8000")
TIMEOUT = int(os.getenv("VALIDATOR_TIMEOUT_SECONDS", "30"))
RETRIES = int(os.getenv("VALIDATOR_HTTP_RETRIES", "3"))


def _print(msg: str) -> None:
    print(f"[validator] {msg}")


def _check(condition: bool, message: str) -> None:
    if not condition:
        raise AssertionError(message)


def _get(path: str) -> Any:
    url = f"{BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.get(url, timeout=TIMEOUT)
            _check(resp.status_code == 200, f"GET {path} failed: {resp.status_code} {resp.text[:250]}")
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(attempt * 2, 5))

    assert last_error is not None
    raise last_error


def _post(path: str, payload: dict | None = None) -> Any:
    url = f"{BASE_URL}{path}"
    last_error: Exception | None = None
    for attempt in range(1, RETRIES + 1):
        try:
            resp = requests.post(url, json=payload or {}, timeout=TIMEOUT)
            _check(resp.status_code == 200, f"POST {path} failed: {resp.status_code} {resp.text[:250]}")
            return resp.json()
        except Exception as exc:
            last_error = exc
            if attempt < RETRIES:
                time.sleep(min(attempt * 2, 5))

    assert last_error is not None
    raise last_error


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


def _validate_inference_script() -> None:
    """Ensure inference.py exists and runs to completion."""
    inference_path = Path("inference.py")
    _check(inference_path.exists(), "inference.py is required at project root")

    env = os.environ.copy()
    # Run in deterministic/no-key mode during validator checks.
    env.setdefault("BASELINE_PROVIDER", "heuristic")
    cmd = [sys.executable, "inference.py"]
    result = subprocess.run(cmd, capture_output=True, text=True, env=env, timeout=900)
    if result.returncode != 0:
        raise AssertionError(
            "inference.py execution failed.\n"
            f"stdout:\n{result.stdout}\n\n"
            f"stderr:\n{result.stderr}"
        )

    try:
        payload = json.loads(result.stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(f"inference.py did not output valid JSON: {exc}") from exc

    _check("task_scores" in payload, "inference.py output missing task_scores")
    _check("average_score" in payload, "inference.py output missing average_score")


def main() -> int:
    _print(f"Base URL: {BASE_URL}")
    _validate_openenv()
    _validate_inference_script()

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
