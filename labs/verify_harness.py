"""AIN-542 Step 1 · Tier A verifiable-reward harness (the hard anchor).

``r = verify(x, y) ∈ {0, 1}`` — a *substantive* check of the model output, never
HTTP-200 liveness (that is the D2 reward-hack). This is the anchor the whole v2
methodology calibrates against: the Council (Tier B) is scored against these
rewards on the verifiable subset to compute anchor-κ.

Two verification MODES — the load-bearing distinction for observational data:

  intrinsic  — needs NO gold answer; checks the output is well-formed on its own
               terms (code parses, structured output is JSON/schema-valid, a tool
               call is well-formed). USABLE ON LIVE FLEET ROWS.
  reference  — needs a gold ``expected`` (answer / result-set match). Only the
               synthetic / canary stream supplies that; observational rows can't
               use it → those rows defer to the Council.

A verifier returns ``reward=None`` when it cannot substantively check the row
(no gold for a reference check, no code block in a code task, subjective task).
``None`` means "defer to Tier B", NOT "pass". Pure stdlib (json, ast, re).

Substantive-not-liveness invariant (v2 §1): a verifier is handed only the
request/response *payloads* — never an HTTP status — and only ever returns a
non-None reward off a content check. The unit tests pin this: a 200 with an
empty / garbage body scores 0 or None, never 1.
"""

from __future__ import annotations

import ast
import json
import math
import re
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any

from labs.task_verifiability import (
    FAMILY_ANSWER,
    FAMILY_CODE,
    FAMILY_NONE,
    FAMILY_RETRIEVAL,
    FAMILY_SCHEMA,
    FAMILY_TOOL,
    Verifiability,
    verifiability_of,
)

# Mirrors api services.training_scope.REWARD_SOURCE_VERIFY — the Tier-A tag.
REWARD_SOURCE_VERIFY = "verify"


@dataclass(frozen=True)
class VerifySample:
    """One row to verify: the task + the request/response payloads (joined from
    ``inferences``), plus an optional gold ``expected`` (synthetic/canary only)."""

    task_type: str | None
    request_payload: dict[str, Any] | None = None
    response_payload: dict[str, Any] | None = None
    expected: Any | None = None


@dataclass(frozen=True)
class VerifyResult:
    reward: float | None  # [0,1]; None = unverifiable → defer to Council
    reward_source: str  # 'verify' iff reward is not None, else ''
    verifier: str  # family that ran (or 'none')
    mode: str  # 'intrinsic' | 'reference' | 'none'
    verifiable: bool  # did a verifier produce a reward (reward is not None)
    detail: str  # short human reason
    evidence: tuple[str, ...] = field(default_factory=tuple)


def _ok(
    reward: float, family: str, mode: str, detail: str, evidence: tuple[str, ...]
) -> VerifyResult:
    return VerifyResult(
        reward=float(reward),
        reward_source=REWARD_SOURCE_VERIFY,
        verifier=family,
        mode=mode,
        verifiable=True,
        detail=detail,
        evidence=evidence,
    )


def _defer(
    family: str, mode: str, detail: str, evidence: tuple[str, ...] = ()
) -> VerifyResult:
    """Unverifiable → Tier B. reward=None, reward_source='' (never 'verify')."""
    return VerifyResult(
        reward=None,
        reward_source="",
        verifier=family,
        mode=mode,
        verifiable=False,
        detail=detail,
        evidence=evidence,
    )


# ── payload extraction (best-effort over OpenAI- + Anthropic-shaped bodies) ──


def extract_output_text(response: dict[str, Any] | None) -> str:
    """Concatenate the assistant's text output across known response shapes."""
    if not isinstance(response, dict):
        return ""
    parts: list[str] = []
    # OpenAI chat completions
    for choice in response.get("choices", []) or []:
        msg = (choice or {}).get("message") or {}
        content = msg.get("content")
        if isinstance(content, str):
            parts.append(content)
        elif isinstance(content, list):
            for block in content:
                if isinstance(block, dict) and block.get("type") in (None, "text"):
                    parts.append(str(block.get("text", "")))
    # Anthropic messages
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text", "")))
    elif isinstance(content, str):
        parts.append(content)
    # raw fallbacks
    for key in ("text", "output", "completion"):
        val = response.get(key)
        if isinstance(val, str):
            parts.append(val)
    return "\n".join(p for p in parts if p)


def extract_tool_calls(response: dict[str, Any] | None) -> list[dict[str, Any]]:
    """Normalize tool calls to ``[{name, arguments(dict)}]`` across shapes."""
    if not isinstance(response, dict):
        return []
    out: list[dict[str, Any]] = []
    # OpenAI: choices[].message.tool_calls[].function.{name, arguments(json str)}
    for choice in response.get("choices", []) or []:
        for tc in ((choice or {}).get("message") or {}).get("tool_calls", []) or []:
            fn = (tc or {}).get("function") or {}
            args = fn.get("arguments")
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except (ValueError, TypeError):
                    args = None
            out.append(
                {
                    "name": fn.get("name"),
                    "arguments": args,
                    "args_raw": fn.get("arguments"),
                }
            )
    # Anthropic: content[].type == 'tool_use' {name, input(dict)}
    content = response.get("content")
    if isinstance(content, list):
        for block in content:
            if isinstance(block, dict) and block.get("type") == "tool_use":
                out.append(
                    {
                        "name": block.get("name"),
                        "arguments": block.get("input"),
                        "args_raw": block.get("input"),
                    }
                )
    return out


def extract_request_tool_schemas(
    request: dict[str, Any] | None,
) -> dict[str, dict[str, Any]]:
    """Map tool name -> JSON-schema of its arguments, across request shapes."""
    if not isinstance(request, dict):
        return {}
    schemas: dict[str, dict[str, Any]] = {}
    for tool in request.get("tools", []) or []:
        if not isinstance(tool, dict):
            continue
        fn = tool.get("function")
        if isinstance(fn, dict) and fn.get("name"):  # OpenAI
            schemas[str(fn["name"])] = fn.get("parameters") or {}
        elif tool.get("name"):  # Anthropic
            schemas[str(tool["name"])] = tool.get("input_schema") or {}
    return schemas


def extract_request_json_schema(
    request: dict[str, Any] | None,
) -> dict[str, Any] | None:
    """Pull a response_format json_schema if the request asked for one."""
    if not isinstance(request, dict):
        return None
    rf = request.get("response_format")
    if isinstance(rf, dict) and rf.get("type") == "json_schema":
        js = rf.get("json_schema") or {}
        return js.get("schema") if isinstance(js, dict) else None
    return None


_CODE_FENCE = re.compile(r"```([A-Za-z0-9_+-]*)\n(.*?)```", re.DOTALL)
_PY_LANGS = {"", "python", "py", "python3"}


def extract_code_blocks(text: str) -> list[tuple[str, str]]:
    """Return ``[(lang, code)]`` from triple-backtick fences."""
    return [(lang.lower(), code) for lang, code in _CODE_FENCE.findall(text)]


def _brackets_balanced(code: str) -> bool:
    pairs = {")": "(", "]": "[", "}": "{"}
    stack: list[str] = []
    for ch in code:
        if ch in "([{":
            stack.append(ch)
        elif ch in pairs:
            if not stack or stack.pop() != pairs[ch]:
                return False
    return not stack


# ── structural schema check (lightweight; no jsonschema dep) ─────────────────


def _structural_conforms(value: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    """Top-level conformance: type + required keys present. Honest subset of
    JSON-Schema (object/array/string/number/integer/boolean), not a full
    validator — enough to catch a wrong-shaped structured output."""
    jtype = schema.get("type")
    type_ok = {
        "object": isinstance(value, dict),
        "array": isinstance(value, list),
        "string": isinstance(value, str),
        "number": isinstance(value, (int, float)) and not isinstance(value, bool),
        "integer": isinstance(value, int) and not isinstance(value, bool),
        "boolean": isinstance(value, bool),
    }
    if jtype in type_ok and not type_ok[jtype]:
        return False, f"type!={jtype}"
    if jtype == "object" or (jtype is None and isinstance(value, dict)):
        if not isinstance(value, dict):
            return False, "not_object"
        missing = [k for k in (schema.get("required") or []) if k not in value]
        if missing:
            return False, f"missing_keys={','.join(missing)}"
    return True, "ok"


# ── verifiers ────────────────────────────────────────────────────────────────


def verify_code(sample: VerifySample) -> VerifyResult:
    """Intrinsic: does the emitted code PARSE? Python via ast.parse (substantive);
    other languages via a bracket-balance heuristic (flagged). No code block in a
    code task → defer (could be a clarification, not a miss to penalise blindly).

    NOTE: this is the intrinsic subset. Sandboxed execution + unit tests (the full
    ``code_exec`` family) is the Spark-side follow-up; not wired here."""
    text = extract_output_text(sample.response_payload)
    blocks = extract_code_blocks(text)
    if not blocks:
        return _defer(FAMILY_CODE, "intrinsic", "no_code_block", ("n_blocks=0",))
    ev: list[str] = []
    all_ok = True
    for i, (lang, code) in enumerate(blocks):
        if lang in _PY_LANGS:
            try:
                ast.parse(code)
                ev.append(f"block{i}:py:parse=ok")
            except SyntaxError as exc:
                all_ok = False
                ev.append(f"block{i}:py:parse=fail({exc.msg})")
        else:
            ok = _brackets_balanced(code)
            all_ok = all_ok and ok
            ev.append(f"block{i}:{lang}:brackets={'ok' if ok else 'fail'}(heuristic)")
    return _ok(
        1.0 if all_ok else 0.0, FAMILY_CODE, "intrinsic", "code_parse", tuple(ev)
    )


_STRUCT_DEMAND = re.compile(
    r"\bjson\b|structured output|response_format|\bschema\b", re.IGNORECASE
)


def _request_text(request: dict[str, Any] | None) -> str:
    if not isinstance(request, dict):
        return ""
    parts: list[str] = []
    sysm = request.get("system")
    if isinstance(sysm, str):
        parts.append(sysm)
    for m in request.get("messages", []) or []:
        c = m.get("content") if isinstance(m, dict) else None
        if isinstance(c, str):
            parts.append(c)
        elif isinstance(c, list):
            for b in c:
                if isinstance(b, dict) and b.get("type") in (None, "text"):
                    parts.append(str(b.get("text", "")))
    return " ".join(parts)


def structure_demanded(request: dict[str, Any] | None) -> bool:
    """True iff the request asked for structured/JSON output — an explicit
    response_format json_schema, or a clear instruction in the prompt. This is the
    gate for JSON-validity scoring: a correct PROSE answer to a request that never
    asked for JSON is NOT wrong (the AIN-547 constant-anchor bug)."""
    if extract_request_json_schema(request):
        return True
    return bool(_STRUCT_DEMAND.search(_request_text(request)))


def _answer_present(text: str, expected: Any) -> bool:
    """Answer-correctness, format-agnostic: the gold answer appears in the output
    (numeric equivalence for numbers, normalized substring otherwise)."""
    exp = str(expected).strip()
    en = _to_number(exp)
    if en is not None:
        for tok in re.findall(r"-?\d[\d,]*\.?\d*", text):
            tn = _to_number(tok)
            if tn is not None and math.isclose(tn, en, rel_tol=1e-6, abs_tol=1e-9):
                return True
        return False
    return _normalize(exp) in _normalize(text)


def verify_schema(sample: VerifySample) -> VerifyResult:
    """Extraction/structured-output verifier (AIN-547 — correctness over format).

    JSON-validity is the intrinsic check ONLY when structured output was demanded
    (response_format json_schema, or an explicit prompt instruction). Otherwise
    format is incidental: score answer-correctness against gold if present, else
    DEFER to the Council — never penalise a correct prose answer for not being
    JSON (the constant-anchor bug that made anchor-κ a 0-by-construction artifact)."""
    text = extract_output_text(sample.response_payload).strip()
    if not text:
        return _defer(FAMILY_SCHEMA, "intrinsic", "empty_output")
    demanded = structure_demanded(sample.request_payload)
    if demanded:
        obj, why = _loads_lenient(text)
        if obj is _SENTINEL:
            return _ok(
                0.0,
                FAMILY_SCHEMA,
                "intrinsic",
                "structured_demanded:json_parse_fail",
                (f"json={why}",),
            )
        schema = extract_request_json_schema(sample.request_payload)
        if schema:
            conforms, detail = _structural_conforms(obj, schema)
            return _ok(
                1.0 if conforms else 0.0,
                FAMILY_SCHEMA,
                "intrinsic",
                "json+schema",
                (f"schema={detail}",),
            )
        return _ok(
            1.0, FAMILY_SCHEMA, "intrinsic", "structured_demanded:json_ok", ("json=ok",)
        )
    # structure NOT demanded → format is incidental
    if sample.expected is not None:
        ok = _answer_present(text, sample.expected)
        return _ok(
            1.0 if ok else 0.0,
            FAMILY_SCHEMA,
            "reference",
            "answer_match",
            (f"exp={str(sample.expected)[:24]}",),
        )
    return _defer(FAMILY_SCHEMA, "intrinsic", "no_structure_demand_no_gold")


def verify_tool(sample: VerifySample) -> VerifyResult:
    """Intrinsic (partial): is there a well-formed tool call whose name is one of
    the request's tools and whose arguments parse + structurally match that tool's
    schema? 'Partial' because the *side-effect* / right-tool-choice needs a gold
    the observational row lacks — only well-formedness is checked here."""
    calls = extract_tool_calls(sample.response_payload)
    if not calls:
        return _defer(FAMILY_TOOL, "intrinsic", "no_tool_call")
    schemas = extract_request_tool_schemas(sample.request_payload)
    ev: list[str] = []
    all_ok = True
    for i, call in enumerate(calls):
        name = call.get("name")
        args = call.get("arguments")
        name_ok = (name in schemas) if schemas else (name is not None)
        args_ok = isinstance(args, (dict, list))
        detail = "ok"
        if args_ok and schemas and name in schemas:
            conforms, detail = _structural_conforms(args, schemas[name])
            args_ok = conforms
        ok = bool(name_ok and args_ok)
        all_ok = all_ok and ok
        ev.append(
            f"call{i}:name={'ok' if name_ok else 'unknown'}:args={detail if args_ok else 'fail'}"
        )
    return _ok(
        1.0 if all_ok else 0.0, FAMILY_TOOL, "intrinsic", "tool_wellformed", tuple(ev)
    )


def verify_answer(sample: VerifySample) -> VerifyResult:
    """Reference: does the final answer match ``expected``? Exact (normalized) or
    numeric (isclose). Needs gold → defers when ``expected`` is None (the live
    fleet case for reasoning)."""
    if sample.expected is None:
        return _defer(FAMILY_ANSWER, "reference", "no_gold")
    text = extract_output_text(sample.response_payload)
    got = _final_answer(text)
    exp = str(sample.expected).strip()
    # numeric equivalence first
    gn, en = _to_number(got), _to_number(exp)
    if gn is not None and en is not None:
        ok = math.isclose(gn, en, rel_tol=1e-6, abs_tol=1e-9)
        return _ok(
            1.0 if ok else 0.0,
            FAMILY_ANSWER,
            "reference",
            "numeric",
            (f"got={got}", f"exp={exp}"),
        )
    ok = _normalize(got) == _normalize(exp)
    return _ok(
        1.0 if ok else 0.0,
        FAMILY_ANSWER,
        "reference",
        "exact",
        (f"got={got[:40]}", f"exp={exp[:40]}"),
    )


def verify_sql(sample: VerifySample) -> VerifyResult:
    """Reference: result-set match. NOT WIRED — needs an execution sandbox + gold
    rows. Declared so the family is complete-shaped; returns unverifiable (never a
    false pass)."""
    return _defer("sql_result_match", "reference", "sql_result_match_not_wired")


def verify_citation(sample: VerifySample) -> VerifyResult:
    """Reference: citations resolve / answer grounded in source. NOT WIRED — needs
    the source set. Returns unverifiable."""
    return _defer(FAMILY_RETRIEVAL, "reference", "citation_not_wired")


_FAMILY_VERIFIERS = {
    FAMILY_CODE: verify_code,
    FAMILY_SCHEMA: verify_schema,
    FAMILY_TOOL: verify_tool,
    FAMILY_ANSWER: verify_answer,
    FAMILY_RETRIEVAL: verify_citation,
}


def verify(sample: VerifySample) -> VerifyResult:
    """Dispatch a sample to its task_type's verifier family. Subjective tasks (no
    anchor) defer to the Council immediately."""
    tv = verifiability_of(sample.task_type)
    if tv.family == FAMILY_NONE:
        return _defer(FAMILY_NONE, "none", "subjective_no_anchor")
    verifier = _FAMILY_VERIFIERS.get(tv.family)
    if verifier is None:  # pragma: no cover - defensive
        return _defer(tv.family, "none", "no_verifier_for_family")
    return verifier(sample)


def verify_rows(rows: Iterable[dict[str, Any]]) -> list[VerifyResult]:
    """Convenience: verify DB rows shaped ``{task_type, request_payload,
    response_payload, expected?}`` (the inferences-joined corpus)."""
    return [
        verify(
            VerifySample(
                task_type=r.get("task_type"),
                request_payload=r.get("request_payload"),
                response_payload=r.get("response_payload"),
                expected=r.get("expected"),
            )
        )
        for r in rows
    ]


# ── small text helpers ───────────────────────────────────────────────────────

_SENTINEL = object()


def _loads_lenient(text: str) -> tuple[Any, str]:
    """json.loads, but also tolerant of a fenced ```json block or leading prose."""
    try:
        return json.loads(text), "ok"
    except ValueError:
        pass
    m = re.search(r"```(?:json)?\n(.*?)```", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), "fenced"
        except ValueError:
            pass
    # first {...} or [...] span
    m = re.search(r"(\{.*\}|\[.*\])", text, re.DOTALL)
    if m:
        try:
            return json.loads(m.group(1)), "span"
        except ValueError:
            return _SENTINEL, "unparseable"
    return _SENTINEL, "unparseable"


def _final_answer(text: str) -> str:
    m = re.search(r"\\boxed\{([^}]*)\}", text)
    if m:
        return m.group(1).strip()
    m = re.search(r"(?i)(?:final answer|answer)\s*[:=]\s*(.+)", text)
    if m:
        return m.group(1).strip().splitlines()[0].strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else ""


def _to_number(s: str) -> float | None:
    m = re.search(r"-?\d[\d,]*\.?\d*", s.replace(" ", ""))
    if not m:
        return None
    try:
        return float(m.group(0).replace(",", ""))
    except ValueError:
        return None


def _normalize(s: str) -> str:
    return re.sub(r"\s+", " ", s).strip().lower().rstrip(".")


# ── sizing: verifiable vs subjective % of traffic ────────────────────────────


def verifiability_histogram(
    task_types: Iterable[str | None],
) -> dict[str, dict[str, float]]:
    """Bucket task_types into {verifiable, partial, subjective} with count + pct.
    The Step 1 'size the anchor' deliverable, pure twin of the sizing SQL."""
    counts = {t.value: 0 for t in Verifiability}
    total = 0
    for tt in task_types:
        counts[verifiability_of(tt).tier.value] += 1
        total += 1
    return {
        tier: {"count": n, "pct": round(100.0 * n / total, 2) if total else 0.0}
        for tier, n in counts.items()
    }


def _sizing_case_sql() -> str:
    """Build the CASE that maps task_type -> tier from TASK_VERIFIABILITY, so the
    SQL can't drift from the Python map (lockstep test asserts all 7 covered)."""
    from labs.task_verifiability import TASK_VERIFIABILITY

    whens = "\n".join(
        f"             WHEN task_type = '{tt}' THEN '{tv.tier.value}'"
        for tt, tv in sorted(TASK_VERIFIABILITY.items())
    )
    return f"CASE\n{whens}\n             ELSE 'subjective' END"


VERIFIABILITY_SIZING_SQL = (
    "SELECT\n"
    f"  {_sizing_case_sql()} AS tier,\n"
    "  count(*) AS n_rows,\n"
    "  round(100.0 * count(*) / sum(count(*)) OVER (), 2) AS pct\n"
    "FROM routing_outcomes\n"
    "WHERE NOT exclude_from_training\n"  # Step 0 stamp: trainable (customer+fleet) only
    "GROUP BY tier\n"
    "ORDER BY n_rows DESC"
)
