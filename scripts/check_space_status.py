"""Quick operator check for Hugging Face Space runtime, endpoints, and logs.

Usage:
  python scripts/check_space_status.py
  python scripts/check_space_status.py --owner Arjunmehta312 --space support-ops-triage-env
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from dataclasses import dataclass
from typing import Optional

import requests

try:
    from huggingface_hub import get_token
except Exception:  # pragma: no cover
    get_token = None  # type: ignore[assignment]


@dataclass
class CheckResult:
    name: str
    ok: bool
    detail: str
    required: bool = True


def _fmt(ok: bool) -> str:
    return "OK" if ok else "FAIL"


def check_runtime(api_url: str, timeout: int) -> CheckResult:
    try:
        resp = requests.get(api_url, timeout=timeout)
        if resp.status_code != 200:
            return CheckResult("runtime_api", False, f"HTTP {resp.status_code}")
        data = resp.json()
        stage = data.get("runtime", {}).get("stage")
        sha = data.get("sha")
        modified = data.get("lastModified")
        return CheckResult(
            "runtime_api",
            True,
            f"stage={stage} sha={sha} lastModified={modified}",
        )
    except Exception as exc:
        return CheckResult("runtime_api", False, f"{type(exc).__name__}: {exc}")


def check_endpoint(base_url: str, path: str, timeout: int) -> CheckResult:
    url = f"{base_url}{path}"
    try:
        resp = requests.get(url, timeout=timeout)
        body = (resp.text or "").replace("\n", " ")
        preview = body[:120]
        ok = resp.status_code < 500
        return CheckResult(path, ok, f"HTTP {resp.status_code} body={preview}")
    except Exception as exc:
        return CheckResult(path, False, f"{type(exc).__name__}: {exc}")


def get_auth_token() -> Optional[str]:
    if get_token is None:
        return None
    try:
        token = get_token()
        return token if token else None
    except Exception:
        return None


def sample_sse(url: str, token: str, timeout: int, max_lines: int = 6) -> CheckResult:
    headers = {
        "Authorization": f"Bearer {token}",
        "Accept": "text/event-stream",
    }

    lines: list[str] = []
    try:
        with requests.get(url, headers=headers, stream=True, timeout=(timeout, timeout)) as resp:
            if resp.status_code != 200:
                return CheckResult(url, False, f"HTTP {resp.status_code}", required=False)

            start = time.time()
            for line in resp.iter_lines(decode_unicode=True):
                if not line:
                    if time.time() - start > timeout:
                        break
                    continue

                if line.startswith("data:") or line.startswith("event:"):
                    lines.append(line[:220])
                if len(lines) >= max_lines:
                    break
                if time.time() - start > timeout:
                    break

        if not lines:
            return CheckResult(url, True, "SSE reachable, no event lines sampled in window", required=False)

        summary = " | ".join(lines[:2])
        return CheckResult(url, True, f"sample={summary}", required=False)
    except Exception as exc:
        return CheckResult(url, False, f"{type(exc).__name__}: {exc}", required=False)


def main() -> int:
    parser = argparse.ArgumentParser(description="Check HF Space runtime and endpoint health")
    parser.add_argument("--owner", default="Arjunmehta312")
    parser.add_argument("--space", default="support-ops-triage-env")
    parser.add_argument("--timeout", type=int, default=20)
    args = parser.parse_args()

    api_url = f"https://huggingface.co/api/spaces/{args.owner}/{args.space}"
    base_url = f"https://{args.owner.lower()}-{args.space}.hf.space"

    checks: list[CheckResult] = []
    checks.append(check_runtime(api_url, args.timeout))

    for path in ["/health", "/tasks", "/", "/favicon.ico"]:
        checks.append(check_endpoint(base_url, path, args.timeout))

    token = get_auth_token()
    if token:
        run_logs = f"https://huggingface.co/api/spaces/{args.owner}/{args.space}/logs/run"
        build_logs = f"https://huggingface.co/api/spaces/{args.owner}/{args.space}/logs/build"
        checks.append(sample_sse(run_logs, token, args.timeout))
        checks.append(sample_sse(build_logs, token, args.timeout))
    else:
        checks.append(CheckResult("logs_auth", False, "No HF token found from huggingface_hub.get_token()", required=False))

    print("=== Space Operator Check ===")
    print(f"space={args.owner}/{args.space}")
    print(f"base_url={base_url}")
    print()

    overall_ok = True
    for c in checks:
        if c.required:
            overall_ok = overall_ok and c.ok
        tag = _fmt(c.ok) if c.required else ("WARN" if not c.ok else "OK")
        suffix = "" if c.required else " (non-blocking)"
        print(f"[{tag}] {c.name}{suffix}: {c.detail}")

    print()
    report = {
        "space": f"{args.owner}/{args.space}",
        "base_url": base_url,
        "overall_ok": overall_ok,
        "checks": [c.__dict__ for c in checks],
    }
    print(json.dumps(report, indent=2))

    return 0 if overall_ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
