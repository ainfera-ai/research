"""AIN-542 · ainfera-labs CLI entrypoint.

Usage:
  ainfera-labs verify --task-type code --response '{"choices":[{"message":{"content":"```python\\nx=1\\n```"}}]}'
  ainfera-labs verify --task-type math --response '{"content":[{"type":"text","text":"2x + 3 = 7\\n2x = 4\\nx = 2\\nFinal answer: 2"}]}' --expected 2
  ainfera-labs verify --task-type qa --response '{"content":[{"type":"text","text":"Paris is the capital of France."}]}' --expected '{"evidence":"Paris is the capital of France.","answer":"Paris"}'
  ainfera-labs batch --days 7 --limit 100 --reward-source verify
  ainfera-labs benchmark
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from typing import Any


def _parse_json_or_string(s: str) -> Any:
    """Try to parse *s* as JSON; if that fails, return the raw string."""
    try:
        return json.loads(s)
    except (ValueError, TypeError):
        return s


def cmd_verify(args: argparse.Namespace) -> int:
    """Run a single sample through the verify harness."""
    from labs.verify_harness import VerifySample, verify

    response_payload = _parse_json_or_string(args.response)
    expected = _parse_json_or_string(args.expected) if args.expected else None

    sample = VerifySample(
        task_type=args.task_type,
        request_payload=_parse_json_or_string(args.request) if args.request else None,
        response_payload=response_payload,
        expected=expected,
    )
    result = verify(sample)

    # Print result
    print(f"reward:        {result.reward}")
    print(f"reward_source: {result.reward_source or '(deferred)'}")
    print(f"verifier:      {result.verifier}")
    print(f"mode:          {result.mode}")
    print(f"verifiable:    {result.verifiable}")
    print(f"detail:        {result.detail}")
    if result.evidence:
        print("evidence:")
        for e in result.evidence:
            print(f"  - {e}")
    return 0 if result.reward is not None else 1


def cmd_batch(args: argparse.Namespace) -> int:
    """Fetch outcomes from api.ainfera.ai and verify them in batch."""
    try:
        from labs.api_client import AinferaClient, ApiError

        client = AinferaClient()
        print(f"Fetching outcomes from {client.base_url}...")
        rows = client.fetch_outcomes(
            days=args.days,
            limit=args.limit,
            reward_source=args.reward_source,
        )
        print(f"Fetched {len(rows)} rows.")
    except ApiError as exc:
        print(f"API error: {exc}", file=sys.stderr)
        return 1
    except Exception as exc:
        print(f"Error: {exc}", file=sys.stderr)
        return 1

    from labs.verify_harness import verify_rows

    results = verify_rows(rows)

    # Summary
    n_verified = sum(1 for r in results if r.reward is not None)
    n_passed = sum(1 for r in results if r.reward is not None and r.reward >= 1.0)
    n_failed = sum(1 for r in results if r.reward is not None and r.reward < 1.0)
    n_deferred = sum(1 for r in results if r.reward is None)

    print(f"\nVerification summary ({len(results)} rows):")
    print(f"  verified:  {n_verified}")
    print(f"  passed:    {n_passed}")
    print(f"  failed:    {n_failed}")
    print(f"  deferred:  {n_deferred} (→ Council)")

    if args.json:
        output = [
            {
                "reward": r.reward,
                "verifier": r.verifier,
                "mode": r.mode,
                "detail": r.detail,
            }
            for r in results
        ]
        print(json.dumps(output, indent=2))

    return 0


def cmd_benchmark(args: argparse.Namespace) -> int:
    """Run the synthetic routing benchmark."""
    # Import lazily so the benchmark module isn't required for other commands
    benchmark_dir = os.path.join(os.path.dirname(__file__), "..", "benchmark")
    sys.path.insert(0, benchmark_dir)
    from run_benchmark import run, summarize

    R, base = run()
    summarize(R, base)
    return 0


def cmd_list_verifiers(args: argparse.Namespace) -> int:
    """List all available verifier families."""
    from labs.verify_harness import _FAMILY_VERIFIERS, _TASK_TYPE_VERIFIERS

    print("Verifier families (task_type → verifier):")
    for task_type, verifier in sorted(_TASK_TYPE_VERIFIERS.items()):
        print(f"  {task_type:15s} → {verifier.__name__}")

    print(f"\nFamily dispatch table ({len(_FAMILY_VERIFIERS)} families):")
    for family, verifier in sorted(_FAMILY_VERIFIERS.items()):
        print(f"  {family:20s} → {verifier.__name__}")

    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="ainfera-labs",
        description="Ainfera Labs — verifiable-reward harness and benchmark CLI",
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # verify
    p_verify = sub.add_parser("verify", help="Verify a single sample")
    p_verify.add_argument("--task-type", required=True, help="Task type (code, math, qa, sql, etc.)")
    p_verify.add_argument("--response", required=True, help="Response payload (JSON or text)")
    p_verify.add_argument("--request", help="Request payload (JSON, optional)")
    p_verify.add_argument("--expected", help="Expected/gold answer (JSON or text)")
    p_verify.set_defaults(func=cmd_verify)

    # batch
    p_batch = sub.add_parser("batch", help="Fetch and verify outcomes from api.ainfera.ai")
    p_batch.add_argument("--days", type=int, default=7, help="Lookback window in days")
    p_batch.add_argument("--limit", type=int, default=100, help="Max rows to fetch")
    p_batch.add_argument("--reward-source", help="Filter by reward_source")
    p_batch.add_argument("--json", action="store_true", help="Output results as JSON")
    p_batch.set_defaults(func=cmd_batch)

    # benchmark
    p_bench = sub.add_parser("benchmark", help="Run the synthetic routing benchmark")
    p_bench.set_defaults(func=cmd_benchmark)

    # list-verifiers
    p_list = sub.add_parser("list-verifiers", help="List available verifier families")
    p_list.set_defaults(func=cmd_list_verifiers)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
