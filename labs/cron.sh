#!/usr/bin/env bash
# labs/cron.sh — L14.2 daily cadence orchestrator (03:00 → 04:45 WIB).
#
# Anchor: 03:00 WIB = 20:00 UTC prior day. Crontab in
# ainfera-os/spark/cron/ainfera-labs.cron invokes this script.
#
# Pipeline:
#   03:00  judge sweep         (judge_worker)
#   03:30  LinUCB refit        (linucb_refit)
#   04:00  replay gate         (replay_gate; CRN harness from eval/replay.py)
#   04:15  policy publish      (api POST /v1/admin/policy/publish — W6-B)
#   04:30  delta log           (delta_logger; append to preprint CSV)
#   04:45  heartbeat           (slack_heartbeat + vault commit runs/<date>.md)
#
# Exits 0 on completion (PROMOTE or HOLD both count as success — HOLD is
# normal when the criterion isn't met). Exits non-zero only on
# infrastructure failure (DB down, gateway 5xx persistent, etc.).
#
# Logs to /opt/ainfera/logs/labs-<date>.log (handled by crontab redirect).

set -euo pipefail

log() { printf '[labs %s] %s\n' "$(date -u +%FT%TZ)" "$*"; }
die() { printf '[labs FATAL] %s\n' "$*" >&2; exit 1; }

WIB_DATE="$(TZ=Asia/Jakarta date +%Y-%m-%d)"
LABS_HOME="${LABS_HOME:-/opt/ainfera/labs}"
ARTIFACTS_DIR="${LABS_ARTIFACTS_DIR:-$LABS_HOME/artifacts/$WIB_DATE}"
mkdir -p "$ARTIFACTS_DIR"

cd "${LABS_REPO:-$HOME/code/ainfera-ai/research}"

log "labs daily cadence — $WIB_DATE WIB · artifacts=$ARTIFACTS_DIR"

# --- 03:00 · judge_worker ---------------------------------------------------
log "[1/6] judge_worker sweep"
python3 -m labs.judge_worker \
    --output "$ARTIFACTS_DIR/judge-summary.json" \
    --date "$WIB_DATE" \
    || die "judge_worker failed"

# --- 03:30 · linucb_refit ---------------------------------------------------
log "[2/6] LinUCB refit"
python3 -m labs.linucb_refit \
    --input-corpus-days 30 \
    --output "$ARTIFACTS_DIR/policy-candidate.json" \
    --date "$WIB_DATE" \
    || die "linucb_refit failed"

# --- 04:00 · replay_gate ----------------------------------------------------
log "[3/6] replay gate"
python3 -m labs.replay_gate \
    --candidate "$ARTIFACTS_DIR/policy-candidate.json" \
    --output "$ARTIFACTS_DIR/replay-verdict.json" \
    || die "replay_gate failed"

DECISION="$(jq -r '.decision' "$ARTIFACTS_DIR/replay-verdict.json")"
log "  → decision: $DECISION"

# --- 04:15 · policy publish (only on PROMOTE) -------------------------------
if [[ "$DECISION" == "PROMOTE" ]]; then
    log "[4/6] policy publish → api/admin/policy/publish"
    CANDIDATE_VERSION="$(jq -r '.candidate_version' "$ARTIFACTS_DIR/replay-verdict.json")"
    curl -sS -X POST "${AINFERA_BASE_URL:-https://api.ainfera.ai}/v1/admin/policy/publish" \
        -H "Authorization: Bearer ${AINFERA_SERVICE_ROLE_KEY}" \
        -H 'Content-Type: application/json' \
        -d "{\"policy_version\":\"${CANDIDATE_VERSION}\"}" \
        > "$ARTIFACTS_DIR/publish-response.json" \
        || die "policy publish failed"
    log "  ✓ published $CANDIDATE_VERSION"
else
    log "[4/6] policy publish SKIPPED (decision=HOLD)"
fi

# --- 04:30 · delta log ------------------------------------------------------
log "[5/6] delta log → preprint corpus"
python3 -m labs.delta_logger \
    --verdict "$ARTIFACTS_DIR/replay-verdict.json" \
    --baseline-corpus 2026-05-15 \
    || die "delta_logger failed"

# --- 04:45 · heartbeat + vault commit --------------------------------------
log "[6/6] heartbeat + vault commit"
python3 -m labs.slack_heartbeat \
    --verdict "$ARTIFACTS_DIR/replay-verdict.json" \
    --judge-summary "$ARTIFACTS_DIR/judge-summary.json" \
    || log "  (heartbeat post failed; not fatal)"

# Commit the run summary to ainfera-os/vault/research-runs/ via the existing
# vault-commit.sh wrapper (path mounted into the container at /opt/ainfera/vault).
VAULT_RUN_PATH="${VAULT_RUNS_DIR:-/opt/ainfera/vault/research/labs/runs}/${WIB_DATE}.md"
mkdir -p "$(dirname "$VAULT_RUN_PATH")"
cat > "$VAULT_RUN_PATH" <<EOF
---
type: labs-run
date: $WIB_DATE
decision: $DECISION
---
# Labs run · $WIB_DATE

See \`$ARTIFACTS_DIR\` for raw JSON.

\`\`\`json
$(jq '.' "$ARTIFACTS_DIR/replay-verdict.json")
\`\`\`
EOF
log "  ✓ vault: $VAULT_RUN_PATH"

log "DONE · decision=$DECISION · artifacts=$ARTIFACTS_DIR"
