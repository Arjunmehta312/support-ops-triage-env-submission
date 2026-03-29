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

try:
    from support_ops_triage_env.baseline_runner import BaselineConfig, run_baseline
except ModuleNotFoundError:
    from baseline_runner import BaselineConfig, run_baseline


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


def main() -> int:
    _load_local_env()

    api_base_url = os.getenv("API_BASE_URL", "https://openrouter.ai/api/v1").strip()
    model_name = os.getenv("MODEL_NAME", "nvidia/nemotron-3-super-120b-a12b:free").strip()
    hf_token = (os.getenv("HF_TOKEN") or os.getenv("API_KEY") or "").strip()

    # Map portal variables to baseline runner variables without hardcoding secrets.
    _setdefault_env("OPENROUTER_BASE_URL", api_base_url)
    _setdefault_env("OPENROUTER_MODEL", model_name)
    _setdefault_env("OPENAI_MODEL", model_name)
    _setdefault_env("OPENROUTER_API_KEY", hf_token)
    _setdefault_env("OPENAI_API_KEY", hf_token)

    # Keep runtime bounded for validator infrastructure.
    _setdefault_env("BASELINE_MAX_STEPS_PER_TASK", "24")
    _setdefault_env("BASELINE_MODEL_TIMEOUT_SECONDS", "12")
    _setdefault_env("BASELINE_MAX_MODEL_RETRIES", "1")
    _setdefault_env("BASELINE_MAX_OUTPUT_TOKENS", "96")

    # Default provider selection for submission runs.
    if not os.getenv("BASELINE_PROVIDER"):
        os.environ["BASELINE_PROVIDER"] = "openrouter" if hf_token else "heuristic"

    result = run_baseline(BaselineConfig())
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
