"""`python3 -m labs.eval_harness` — run ONE curated synthetic integration cycle.

AIN-459 / Tap-3. This entry point proves the proof-pipeline END-TO-END
(enroll -> 4 arms -> synth -> judge -> metrics -> snapshot) on a CURATED SYNTHETIC
task set — NOT on real customer traffic.

Why synthetic: real customer prompt text is NOT persisted in the schema, so
`loader.freeze_from_rows` (the real-traffic freeze) has no data source. That path
is deferred pending a prompt-persistence decision; this cycle bypasses it entirely
and loads a committed, author-written task set via `taskset.load_frozen`.

HONESTY: the web snapshot this cycle emits is tagged state="illustrative" with a
synthetic source string. It is a PIPELINE proof, NOT a measured real-traffic
result, and must NEVER be published to /proof or read as measured. Publication
(Taps 5/7) stays founder-held.

Fail-closed gates (same as the rest of the harness):
  - LABS_EVAL_HARNESS_ENABLED must be truthy (whole-harness kill switch), AND
  - LABS_EVAL_LIVE must be truthy (live model calls cost money), AND
  - run_cycle itself re-checks the probe-agent id (L2) + the Labs key before it
    builds the real gateway.

This file is the ONLY place a wall-clock timestamp is read (the harness modules
forbid it); `generated_at` is computed here at the CLI boundary.
"""

from __future__ import annotations

import argparse
import sys
from datetime import datetime, timezone
from pathlib import Path

from labs.eval_harness import config, snapshot, taskset
from labs.eval_harness.cycle import run_cycle

# Default curated synthetic task set (committed; public-safe — see its `note`).
DEFAULT_TASKSET = str(Path(__file__).with_name("fixtures") / "curated_integration_taskset.json")

# The honesty tag + source for every snapshot this CLI emits. assert_sanitized
# already allows "illustrative"; "measured" is reserved for the genuine
# real-traffic path and is NEVER set here.
SNAPSHOT_STATE = "illustrative"
SNAPSHOT_SOURCE = (
    "SYNTHETIC integration cycle — pipeline proof only; NOT measured, NOT for publication."
)


def _now_iso_utc() -> str:
    """ISO-8601 UTC timestamp at the CLI boundary (modules forbid wall-clock)."""
    return datetime.now(timezone.utc).isoformat(timespec="seconds").replace("+00:00", "Z")


def _gate_or_exit() -> None:
    """Fail closed unless BOTH the harness flag and the live flag are on."""
    problems: list[str] = []
    if not config.harness_enabled():
        problems.append(f"  - {config.ENABLED_ENV} is OFF (whole-harness kill switch).")
    if not config.live_enabled():
        problems.append(f"  - {config.LIVE_ENV} is OFF (live model calls cost money).")
    if problems:
        print(
            "Refusing to run: the curated synthetic integration cycle is INERT.\n"
            + "\n".join(problems)
            + "\n\nEnable BOTH flags on the Spark Labs tenant only, after the Doppler\n"
            "key-fix + the founder gate. run_cycle additionally requires "
            f"{config.PROBE_AGENT_ID_ENV} (L2 probe-agent) and {config.GATEWAY_KEY_ENV}.",
            file=sys.stderr,
        )
        raise SystemExit(2)


def _parse_args(argv: list[str] | None) -> argparse.Namespace:
    p = argparse.ArgumentParser(
        prog="python3 -m labs.eval_harness",
        description=(
            "Run ONE curated SYNTHETIC integration cycle (AIN-459 / Tap-3). "
            "Illustrative-only; never measured, never published."
        ),
    )
    p.add_argument(
        "--taskset",
        default=DEFAULT_TASKSET,
        help=f"Path to a frozen task-set JSON (default: the curated synthetic set, {DEFAULT_TASKSET}).",
    )
    return p.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    _gate_or_exit()

    # Curated synthetic load — DELIBERATELY bypasses loader.freeze_from_rows (the
    # routing_outcomes path has no persisted prompt data; see loader.py).
    frozen = taskset.load_frozen(args.taskset)
    undersized = taskset.assert_min_per_type(frozen)
    if undersized:
        # Expected for an integration cycle — this is NOT a statistical-power run.
        print(
            f"NOTE: task-set {frozen.version} is below {config.MIN_TASKS_PER_TYPE}/type "
            f"for: {', '.join(undersized)}. Expected for a synthetic integration cycle "
            "(pipeline proof, not statistical power).",
            file=sys.stderr,
        )

    generated_at = _now_iso_utc()
    res = run_cycle(
        frozen,
        generated_at=generated_at,
        snapshot_state=SNAPSHOT_STATE,
        snapshot_source=SNAPSHOT_SOURCE,
    )

    web = res.web_snapshot or {}
    # Defence in depth: the artifact this CLI emits must never claim "measured".
    if web.get("state") != SNAPSHOT_STATE:
        raise SystemExit(
            f"FATAL: snapshot state is {web.get('state')!r}, expected {SNAPSHOT_STATE!r} — refusing."
        )

    print("=" * 72)
    print("Curated SYNTHETIC integration cycle — AIN-459 / Tap-3")
    print("=" * 72)
    print(f"task-set version : {frozen.version}")
    print(f"task-set hash    : {frozen.hash}")
    print(f"tasks            : {frozen.n}  by type: {frozen.by_type}")
    print(f"generated_at     : {generated_at}")
    print(f"status           : {res.status}")
    print(f"n_calls          : {res.n_calls}")
    print(f"cost_usd         : {res.cost_usd}")
    print(f"snapshot state   : {web.get('state')}  (NEVER 'measured' on this path)")
    print(f"win verdict      : {res.win}")
    print("  CAVEAT         : this win/no-win verdict is NOT meaningful — it is")
    print("                   computed on SYNTHETIC data to prove the pipeline,")
    print("                   not on measured real traffic. Do NOT publish.")
    print(f"drift_types      : {res.drift_types or '(none)'}")
    print(f"artifacts        : {res.artifacts}")
    print("=" * 72)
    print("INERT/illustrative — snapshot is tagged so it can never be read as")
    print("measured or published to /proof. Publication stays founder-held.")
    return 0 if res.status == "completed" else 1


if __name__ == "__main__":  # pragma: no cover - CLI entry
    raise SystemExit(main())
