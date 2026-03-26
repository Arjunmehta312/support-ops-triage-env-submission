"""Run baseline benchmark over all support triage tasks."""

from __future__ import annotations

import json

from support_ops_triage_env.baseline_runner import BaselineConfig, run_baseline


def main() -> int:
    result = run_baseline(BaselineConfig())
    print(json.dumps(result.model_dump(), indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
