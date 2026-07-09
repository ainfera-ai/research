"""`python3 -m eval_v2_ruler` — run the eval-v2 ruler.

Scores every catalog model on G1-G4 admission gates using a frozen
tool-use task set. Output: CSV + JSON in the eval-v2-results directory.

Gate gates (vault: decisions/2026-07-06-eval-v2-ruler-scope.md):
  G1  tool-call validity ≥99%
  G2  context coherence ≥95% (correct tool + valid args)
  G3  latency p95 < 3s
  G4  cost efficiency (tokens per successful task)

No prod mutations. No routing_outcomes rows (pinned slugs). Build phase only.
"""

from __future__ import annotations

import argparse
import sys

from labs.eval_v2_ruler import config
from labs.eval_v2_ruler.runner import run_ruler
from labs.eval_v2_ruler.taskset import load_frozen


def _gate_or_exit() -> None:
    """Fail closed unless RULER_ENABLED + RULER_LIVE are on."""
    problems: list[str] = []
    if not config.ruler_enabled():
        problems.append(f"  - {config.ENABLED_ENV} is OFF (kill switch).")
    if not config.live_enabled():
        problems.append(f"  - {config.LIVE_ENV} is OFF (live calls cost money).")
    if problems:
        print(
            "Refusing to run: the eval-v2 ruler is INERT.\n" + "\n".join(problems)
            + f"\n\nEnable BOTH flags AND set {config.GATEWAY_KEY_ENV}.\n"
            f"Example: RULER_ENABLED=1 RULER_LIVE=1 python3 -m eval_v2_ruler",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m eval_v2_ruler",
        description="Run the eval-v2 ruler — score catalog models on G1-G4 gates.",
    )
    p.add_argument(
        "--taskset",
        default=config.DEFAULT_TASKSET,
        help=f"Path to a frozen task-set JSON (default: {config.DEFAULT_TASKSET}).",
    )
    p.add_argument(
        "--slugs",
        default=None,
        help="Comma-separated model slugs to score (bypasses catalog fetch).",
    )
    p.add_argument(
        "--max-models",
        type=int,
        default=None,
        help="Score only the first N models (for validation runs).",
    )
    p.add_argument(
        "--output-dir",
        default=config.OUTPUT_DIR,
        help=f"Output directory (default: {config.OUTPUT_DIR}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _gate_or_exit()

    frozen = load_frozen(args.taskset)
    print(f"ruler: task-set {frozen.version} (hash={frozen.hash[:12]}..., {frozen.n} tasks)",
          file=sys.stderr)

    # Build explicit model list if --slugs provided
    models = None
    if args.slugs:
        from labs.eval_v2_ruler.catalog import CatalogModel
        slugs = [s.strip() for s in args.slugs.split(",") if s.strip()]
        models = [CatalogModel(
            slug=s, display_name=s, provider="custom", type="model",
            input_cost_per_million=None, output_cost_per_million=None,
            context_window=None, capabilities=(), routable_status="active",
        ) for s in slugs]
        print(f"ruler: scoring {len(models)} explicit slugs (bypassing catalog)", file=sys.stderr)

    res = run_ruler(
        frozen,
        models=models,
        max_models=args.max_models,
        output_dir=args.output_dir,
    )

    print("=" * 72)
    print("eval-v2 ruler — catalog scoring complete")
    print("=" * 72)
    print(f"models scored  : {res.n_scored}/{res.n_models}")
    print(f"models passed  : {res.n_passed}")
    print(f"total cost     : ${res.total_cost_usd:.2f}")
    print(f"csv output     : {res.csv_path}")
    print(f"json output    : {res.json_path}")
    if res.errors:
        print(f"errors         : {len(res.errors)} models had all-task errors")
        for e in res.errors[:5]:
            print(f"  - {e}")
    print("=" * 72)
    return 0


if __name__ == "__main__":  # pragma: no cover
    raise SystemExit(main())
