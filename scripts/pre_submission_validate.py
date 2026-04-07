"""Pre-submission validator for hackathon readiness checks."""

from __future__ import annotations

import json
import os
from pathlib import Path
import re
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

    stdout = result.stdout.strip()

    # Preferred submission format: structured logs on stdout.
    start_lines = [line for line in stdout.splitlines() if line.startswith("[START] ")]
    step_lines = [line for line in stdout.splitlines() if line.startswith("[STEP] ")]
    end_lines = [line for line in stdout.splitlines() if line.startswith("[END] ")]

    if start_lines or step_lines or end_lines:
        _check(bool(start_lines), "inference.py missing [START] lines")
        _check(bool(step_lines), "inference.py missing [STEP] lines")
        _check(bool(end_lines), "inference.py missing [END] lines")

        score_re = re.compile(r"\bscore=([0-9]*\.?[0-9]+)")
        for line in end_lines:
            match = score_re.search(line)
            _check(match is not None, f"[END] line missing score: {line}")
            score = float(match.group(1))
            _check(0.0 <= score <= 1.0, f"[END] score out of range [0,1]: {line}")
        return

    # Backward-compatible fallback: JSON payload on stdout.
    try:
        payload = json.loads(stdout)
    except json.JSONDecodeError as exc:
        raise AssertionError(
            "inference.py output did not match structured stdout or JSON fallback.\n"
            f"stdout:\n{stdout}\n\n"
            f"stderr:\n{result.stderr}\n\n"
            f"json_error: {exc}"
        ) from exc

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
