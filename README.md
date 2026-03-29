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

## Important: This Is The Official Submission Environment

This directory (`support_ops_triage_env/`) is the actual environment to be submitted for the hackathon.

Repository-wide clarification:
- This folder contains the real implementation, deployment config, validator flow, and baseline runner used for submission.
- Other repository folders are supplementary (prep scripts, notes, docs, and learning material).

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

## Submission Artifacts

This directory is submission-ready as a standalone environment repository root.

Required files included:
- `openenv.yaml`
- `Dockerfile`
- `README.md`
- `inference.py` (root-level entrypoint)
- `requirements.txt`
- `.env.example`

Primary submission outputs:
- GitHub repository link
- Hugging Face Space URL

## Portal Additional Instruction Compliance

The submission supports the required inference environment variables:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

`inference.py` maps these values to baseline runtime configuration and executes the benchmark with the OpenAI client path used in baseline inference.

### Environment Configuration

For local development:
1. Copy `.env.example` to `.env`
2. Fill in `HF_TOKEN`
3. Run `python inference.py`

For Hugging Face Space secrets, set:
- `API_BASE_URL`
- `MODEL_NAME`
- `HF_TOKEN`

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

Determinism and implementation policy:
- Grading is fully programmatic and rule-based.
- No LLM calls are used in the grader scoring path.
- For identical episode state and action history, grader output is identical.

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
- inference.py (portal submission entrypoint)

Provider behavior is environment-driven and secure-by-default:

- `BASELINE_PROVIDER=heuristic` (default): fully deterministic baseline, no API key required.
- `BASELINE_PROVIDER=openai`: uses OpenAI Chat Completions with `OPENAI_API_KEY`.
- `BASELINE_PROVIDER=openrouter`: uses OpenAI-compatible client against OpenRouter with:
  - `OPENROUTER_API_KEY`
  - optional `OPENROUTER_BASE_URL` (default: `https://openrouter.ai/api/v1`)
  - optional `OPENROUTER_MODEL` (default: `nvidia/nemotron-3-super-120b-a12b:free`)
  - optional `OPENROUTER_REASONING_ENABLED=true` to request reasoning mode
  - optional attribution headers via `OPENROUTER_SITE_URL` and `OPENROUTER_APP_NAME`
- `BASELINE_PROVIDER=auto`: prefers OpenRouter key, then OpenAI key, then heuristic fallback.

Deterministic defaults:
- seed: 42
- temperature: 0

Runtime controls for model-call duration and throughput:
- `BASELINE_MODEL_TIMEOUT_SECONDS` (default: `20`)
- `BASELINE_MAX_MODEL_RETRIES` (default: `3`)
- `BASELINE_MAX_OUTPUT_TOKENS` (default: `96`)
- `BASELINE_MAX_STEPS_PER_TASK` (default: `36`)
- OpenRouter-specific overrides: `OPENROUTER_MODEL_TIMEOUT_SECONDS`, `OPENROUTER_MAX_MODEL_RETRIES`, `OPENROUTER_MAX_OUTPUT_TOKENS`

Recommended low-latency OpenRouter profile:

```bash
BASELINE_PROVIDER=openrouter \
OPENROUTER_API_KEY=... \
OPENROUTER_MODEL_TIMEOUT_SECONDS=8 \
OPENROUTER_MAX_MODEL_RETRIES=0 \
OPENROUTER_MAX_OUTPUT_TOKENS=64 \
python scripts/run_baseline.py
```

Failure handling behavior:
- On provider timeout/failure, the runner automatically falls back to heuristic actions for the remainder of that task.
- This preserves baseline completion and avoids repeated long waits.

Security note:
- Never hardcode API keys in repository files.
- Pass keys only via environment variables or secret managers.
- OpenRouter free endpoints may log prompts for model improvement, so avoid sensitive data.

Portal-compliant inference command:

```bash
API_BASE_URL=https://openrouter.ai/api/v1 \
MODEL_NAME=nvidia/nemotron-3-super-120b-a12b:free \
HF_TOKEN=... \
python inference.py
```

Example:

```bash
# Deterministic submission-safe mode
BASELINE_PROVIDER=heuristic python scripts/run_baseline.py

# OpenRouter mode (Nemotron 3 Super)
BASELINE_PROVIDER=openrouter OPENROUTER_API_KEY=... python scripts/run_baseline.py

# OpenAI mode
BASELINE_PROVIDER=openai OPENAI_API_KEY=... python scripts/run_baseline.py
```

Typical baseline output format:

```json
{
  "model": "nvidia/nemotron-3-super-120b-a12b:free",
  "provider": "openrouter",
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
- inference.py exists and executes successfully
- health endpoint
- tasks endpoint and 3+ tasks
- reset/step/state availability
- baseline endpoint returns scores in [0.0, 1.0]
- grader endpoint responds correctly

## Submission Evidence Checklist

Recommended final checks before portal submission:
- openenv validate passes in the environment root
- local server runs and pre-submission validator passes
- deployed Space health and endpoint checks pass
- determinism check script confirms reproducible grader and baseline outputs

Commands:

```bash
python scripts/pre_submission_validate.py
python scripts/ci_determinism_check.py
python scripts/check_space_status.py
```

## Deployment Verification Snapshot (March 29, 2026)

Current readiness summary:
- Remote validator passed against deployed Space URL.
- Determinism checks passed (`grader_unique_count=1`, `baseline_unique_count=1`).
- OpenRouter integration is active with model `nvidia/nemotron-3-super-120b-a12b:free`.
- Full OpenRouter baseline run completed with fast runtime profile.
- Latest push to HF Space succeeded via `openenv push --repo-id Arjunmehta312/support-ops-triage-env`.
- Post-push Space can be `BUILDING` briefly; confirm readiness with `scripts/check_space_status.py`.

OpenRouter profile used for verification:

```bash
BASELINE_PROVIDER=openrouter \
OPENROUTER_API_KEY=... \
OPENROUTER_MODEL=nvidia/nemotron-3-super-120b-a12b:free \
OPENROUTER_MODEL_TIMEOUT_SECONDS=8 \
OPENROUTER_MAX_MODEL_RETRIES=0 \
OPENROUTER_MAX_OUTPUT_TOKENS=64 \
python scripts/run_baseline.py
```

Result sample from the latest successful OpenRouter run:

```json
{
  "model": "nvidia/nemotron-3-super-120b-a12b:free",
  "provider": "openrouter",
  "task_scores": {
    "easy_inbox_hygiene": 0.735,
    "medium_vip_sla_mix": 0.491875,
    "hard_incident_storm": 0.497083
  },
  "average_score": 0.574653,
  "run_seed": 42
}
```

Secret handling note:
- API keys are not stored in repository files.
- Keys are injected via environment variables only and cleared from session after runs.

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
