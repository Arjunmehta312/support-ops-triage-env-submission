"""CI determinism checks for grader and baseline endpoints."""

from __future__ import annotations

import json
import os
from typing import Any

import requests

BASE_URL = os.getenv("OPENENV_BASE_URL", "http://127.0.0.1:8000")
TIMEOUT = 45


def _get(path: str) -> Any:
    resp = requests.get(f"{BASE_URL}{path}", timeout=TIMEOUT)
    if resp.status_code != 200:
        raise AssertionError(f"GET {path} failed: {resp.status_code} {resp.text[:250]}")
    return resp.json()


def _post(path: str, payload: dict | None = None) -> Any:
    resp = requests.post(f"{BASE_URL}{path}", json=payload or {}, timeout=TIMEOUT)
    if resp.status_code != 200:
        raise AssertionError(f"POST {path} failed: {resp.status_code} {resp.text[:250]}")
    return resp.json()


def main() -> int:
    tasks_payload = _get("/tasks")
    tasks = tasks_payload.get("tasks", [])
    if not tasks:
        raise AssertionError("No tasks available from /tasks")

    task_id = tasks[0]["task_id"]
    reset_payload = _post("/reset", {"task_id": task_id})
    queue = reset_payload.get("observation", {}).get("queue_snapshot", [])
    if not queue:
        raise AssertionError("No queue_snapshot entries after reset")

    ticket_id = queue[0]["ticket_id"]
    grader_payload = {
        "task_id": task_id,
        "actions": [
            {"operation": "focus", "ticket_id": ticket_id},
            {
                "operation": "classify",
                "ticket_id": ticket_id,
                "classification": "billing",
            },
        ],
    }

    grader_results = []
    for _ in range(5):
        result = _post("/grader", grader_payload)
        grader_results.append(json.dumps(result, sort_keys=True))

    grader_unique = len(set(grader_results))
    if grader_unique != 1:
        raise AssertionError(f"Non-deterministic grader output detected: unique={grader_unique}")

    baseline_results = []
    for _ in range(5):
        result = _post("/baseline")
        baseline_results.append(json.dumps(result, sort_keys=True))

    baseline_unique = len(set(baseline_results))
    if baseline_unique != 1:
        raise AssertionError(f"Non-deterministic baseline output detected: unique={baseline_unique}")

    print("[determinism] grader_unique_count=1")
    print("[determinism] baseline_unique_count=1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
