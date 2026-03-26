---
title: Support Ops Triage Environment
emoji: 🛟
colorFrom: blue
colorTo: red
sdk: docker
app_port: 8000
base_path: /web
tags:
  - openenv
  - customer-support
  - rl
  - evaluation
---

# Support Ops Triage Environment

A production-style OpenEnv environment for training and evaluating agents on customer support operations triage.

This is not a toy domain. It simulates realistic support workflows used by SaaS operations teams:
- classifying inbound tickets
- setting urgency based on SLA pressure
- routing to the correct operational queue
- escalating incidents and security events
- responding with policy-safe templates
- resolving tickets and producing end-of-shift summaries

## Why This Environment Matters

Most current agent environments benchmark coding or games. Real enterprise support teams need dependable triage policies under time pressure. This environment models that gap with deterministic grading, dense reward shaping, and task difficulty progression that maps to actual on-call support workflows.

## OpenEnv Spec Compliance

The environment implements the standard OpenEnv API through typed models:
- step(action) -> observation with reward and done
- reset(...) -> initial observation for a selected task
- state() -> current episode state

Typed Pydantic models:
- Action: SupportOpsTriageAction
- Observation: SupportOpsTriageObservation
- Reward: RewardSignal
- State: SupportOpsTriageState

Manifest:
- openenv.yaml at project root

Validation command:
- openenv validate .

## Action Space

SupportOpsTriageAction has operation plus optional fields depending on operation.

Operations:
- focus
- classify
- set_priority
- assign_queue
- escalate
- respond
- resolve
- summarize
- finish

Optional fields used by operations:
- ticket_id
- classification
- priority
- queue
- escalate
- response_template
- note
- summary

## Observation Space

SupportOpsTriageObservation includes:
- task metadata: task_id, difficulty, objective
- queue_snapshot: list of TicketSnapshot objects
- focus_ticket_id
- pending_count
- progress_score in [0, 1]
- messages (policy feedback + operational hints)
- reward_signal with decomposed reward terms

## Tasks and Graders

The environment includes 3 deterministic tasks with escalating difficulty.

1. easy_inbox_hygiene (easy)
- Small mixed queue with straightforward billing/account/request tickets.
- Tests basic triage correctness and closure quality.

2. medium_vip_sla_mix (medium)
- Enterprise incident + security + trust/safety mix with tighter SLA windows.
- Tests prioritization and escalation under pressure.

3. hard_incident_storm (hard)
- Cross-region incident storm with duplicates, abuse noise, and VIP pressure.
- Tests robust decision quality with tighter action budget.

### Grader Behavior

Final grader score is deterministic in [0.0, 1.0], combining:
- triage quality score (field-level correctness per ticket)
- efficiency score (action budget usage)

Difficulty thresholds:
- easy: >= 0.82
- medium: >= 0.78
- hard: >= 0.74

## Reward Function Design

Reward is dense and trajectory-aware (not sparse terminal-only).

Each step includes:
- positive: delta_progress when decisions improve ticket correctness
- positive: milestone_bonus for useful actions (resolve/summarize)
- negative: action_penalty for invalid or low-quality actions
- negative: loop_penalty for repeated no-progress behavior
- negative: sla_penalty when unresolved tickets hit SLA zero
- small per-step cost to discourage infinite loops

This gives meaningful partial progress signals while penalizing destructive or stagnant behavior.

## Additional Endpoints

In addition to standard OpenEnv endpoints, this environment exposes:

- GET /tasks
  - Returns task list and action schema

- GET /grader
  - Returns latest completed episode grade

- POST /grader
  - Grades a provided trajectory replay request

- POST /baseline
  - Runs baseline policy over all tasks and returns reproducible score report

## Baseline Inference

Baseline runner lives in:
- scripts/run_baseline.py

It uses the OpenAI client when OPENAI_API_KEY is present.
- model default: gpt-4.1-mini
- seed: 42
- temperature: 0

If OPENAI_API_KEY is absent, it falls back to a deterministic heuristic baseline so evaluation still runs.

Example:

```bash
python scripts/run_baseline.py
```

Typical baseline output format:

```json
{
  "model": "gpt-4.1-mini",
  "provider": "openai",
  "task_scores": {
    "easy_inbox_hygiene": 0.88,
    "medium_vip_sla_mix": 0.74,
    "hard_incident_storm": 0.61
  },
  "average_score": 0.743333,
  "run_seed": 42
}
```

## Local Setup

### 1) Install dependencies

```bash
pip install -e .
```

### 2) Run server

```bash
uvicorn server.app:app --host 0.0.0.0 --port 8000
```

### 3) Quick checks

```bash
curl http://localhost:8000/health
curl http://localhost:8000/tasks
```

## Docker

Build and run:

```bash
docker build -t support-ops-triage-env:latest .
docker run --rm -p 8000:8000 support-ops-triage-env:latest
```

## Hugging Face Space Deployment

This repository is Docker Space compatible.

Required metadata is already present in README front matter:
- sdk: docker
- app_port: 8000
- tag: openenv

Deploy with OpenEnv CLI:

```bash
openenv push --repo-id <your-hf-username>/support-ops-triage-env
```

## Pre-Submission Validation

Run the included validator:

```bash
python scripts/pre_submission_validate.py
```

The validator checks:
- openenv validate passes
- health endpoint
- tasks endpoint and 3+ tasks
- reset/step/state availability
- baseline endpoint returns scores in [0.0, 1.0]
- grader endpoint responds correctly

## Project Structure

- models.py: typed action/observation/reward/state models
- task_bank.py: deterministic task scenarios
- grading.py: deterministic grader logic
- server/support_ops_triage_env_environment.py: environment dynamics + reward shaping
- server/app.py: OpenEnv app + custom endpoints
- baseline_runner.py: baseline policy runner
- scripts/run_baseline.py: baseline entry script
- scripts/pre_submission_validate.py: automated pre-submit checks
- Dockerfile: Hugging Face Docker Space image
