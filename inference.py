"""Submission inference entrypoint for support_ops_triage_env.

This script satisfies portal requirements:
- file name is inference.py at project root
- uses OpenAI client path through baseline_runner
- reads API_BASE_URL, MODEL_NAME, HF_TOKEN environment variables
"""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

try:
    from support_ops_triage_env.baseline_runner import (
        BaselineConfig,
        _heuristic_action,
        _openai_action,
        _resolve_provider_client,
    )
    from support_ops_triage_env.grading import grade_episode
    from support_ops_triage_env.server.support_ops_triage_env_environment import SupportOpsTriageEnvironment
    from support_ops_triage_env.task_bank import list_task_briefs
except ModuleNotFoundError:
    from baseline_runner import BaselineConfig, _heuristic_action, _openai_action, _resolve_provider_client
    from grading import grade_episode
    from server.support_ops_triage_env_environment import SupportOpsTriageEnvironment
    from task_bank import list_task_briefs


def _setdefault_env(name: str, value: str | None) -> None:
    if value and not os.getenv(name):
        os.environ[name] = value


def _load_local_env(env_path: str | None = None) -> None:
    """Load a simple KEY=VALUE .env file if present.

    This keeps local execution ergonomic while keeping secrets out of source control.
    """
    path = Path(env_path) if env_path else Path(__file__).resolve().parent / ".env"
    if not path.exists():
        return

    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip("\"'")
        if key and value and not os.getenv(key):
            os.environ[key] = value


def _serialize_action(action: Any) -> str:
    """Serialize action into a compact one-line string for structured logs."""
    if hasattr(action, "model_dump"):
        payload = action.model_dump(exclude_none=True)
    else:
        payload = action
    return json.dumps(payload, separators=(",", ":"), ensure_ascii=True)


def _log_start(task: str, env: str, model: str) -> None:
    print(f"[START] task={task} env={env} model={model}", flush=True)


def _log_step(step: int, action_str: str, reward: float, done: bool, error: str | None) -> None:
    done_str = "true" if done else "false"
    error_str = error if error else "null"
    print(
        f"[STEP] step={step} action={action_str} reward={reward:.2f} done={done_str} error={error_str}",
        flush=True,
    )


def _log_end(success: bool, steps: int, score: float, rewards: list[float]) -> None:
    success_str = "true" if success else "false"
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(
        f"[END] success={success_str} steps={steps} score={score:.3f} rewards={rewards_str}",
        flush=True,
    )


def _run_with_structured_logs(cfg: BaselineConfig) -> dict[str, float]:
    """Run baseline policy and emit required structured logs to stdout."""
    provider, client, model = _resolve_provider_client(cfg)
    benchmark = os.getenv("OPENENV_BENCHMARK", "support_ops_triage")

    env = SupportOpsTriageEnvironment()
    task_scores: dict[str, float] = {}
    try:
        for brief in list_task_briefs():
            task_id = str(brief["task_id"])
            _log_start(task=task_id, env=benchmark, model=model)

            obs = env.reset(task_id=task_id)
            task_brief = {
                "task_id": obs.task_id,
                "difficulty": obs.difficulty,
                "objective": obs.objective,
                "max_steps": env.state.max_steps,
            }

            llm_available = provider in ("openai", "openrouter") and client is not None
            rewards: list[float] = []
            steps_taken = 0
            done = False

            for step in range(1, cfg.max_steps_per_task + 1):
                obs_payload = obs.model_dump()
                if llm_available and client is not None:
                    try:
                        action = _openai_action(
                            client=client,
                            model=model,
                            provider=provider,
                            task_brief=task_brief,
                            observation=obs_payload,
                            run_seed=cfg.run_seed,
                            max_retries=cfg.max_model_retries,
                            max_output_tokens=cfg.max_output_tokens,
                            request_timeout_seconds=cfg.model_timeout_seconds,
                            enable_openrouter_reasoning=cfg.enable_openrouter_reasoning,
                        )
                    except Exception:
                        # Fall back to heuristic after provider errors.
                        llm_available = False
                        action = _heuristic_action(obs_payload)
                else:
                    action = _heuristic_action(obs_payload)

                action_str = _serialize_action(action)
                obs = env.step(action)
                reward = float(obs.reward or 0.0)
                done = bool(obs.done)
                rewards.append(reward)
                steps_taken = step

                # The current environment does not expose a dedicated last_action_error field.
                _log_step(step=step, action_str=action_str, reward=reward, done=done, error=None)

                if done:
                    break

            if done:
                score = float(obs.metadata.get("final_grade", 0.0))
            else:
                score, _, _ = grade_episode(
                    task_id=env.state.task_id,
                    difficulty=env.state.difficulty,
                    tickets=env.state.tickets,
                    expected_by_id=env.expected_by_id,
                    step_count=env.state.step_count,
                )

            score = max(0.0, min(float(score), 1.0))
            task_scores[task_id] = round(score, 6)
            _log_end(success=score >= 0.5, steps=steps_taken, score=score, rewards=rewards)
    finally:
        env.close()

    return task_scores


def main() -> int:
    _load_local_env()

    # Optional placeholder for environments using from_docker_image().
    # Kept for checklist parity; not used by this local in-process environment.
    local_image_name = os.getenv("LOCAL_IMAGE_NAME", "").strip()

    api_base_url = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model_name = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free").strip()
    hf_token = os.getenv("HF_TOKEN", "").strip()

    # Map portal variables to baseline runner variables without hardcoding secrets.
    _setdefault_env("OPENROUTER_BASE_URL", api_base_url)
    _setdefault_env("OPENROUTER_MODEL", model_name)
    _setdefault_env("OPENAI_MODEL", model_name)
    _setdefault_env("OPENROUTER_API_KEY", hf_token)
    _setdefault_env("OPENAI_API_KEY", hf_token)

    # Silence linter for checklist-only variable while keeping behavior unchanged.
    _ = local_image_name

    # Keep runtime bounded for validator infrastructure.
    _setdefault_env("BASELINE_MAX_STEPS_PER_TASK", "24")
    _setdefault_env("BASELINE_MODEL_TIMEOUT_SECONDS", "12")
    _setdefault_env("BASELINE_MAX_MODEL_RETRIES", "1")
    _setdefault_env("BASELINE_MAX_OUTPUT_TOKENS", "96")

    # Default provider selection for submission runs.
    if not os.getenv("BASELINE_PROVIDER"):
        os.environ["BASELINE_PROVIDER"] = "openrouter" if hf_token else "heuristic"

    result = _run_with_structured_logs(BaselineConfig())

    # Keep a machine-readable summary in stderr for local debugging without
    # interfering with required structured stdout lines.
    average = round(sum(result.values()) / max(len(result), 1), 6)
    summary = {
        "task_scores": result,
        "average_score": average,
    }
    print(json.dumps(summary, indent=2), file=os.sys.stderr, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
