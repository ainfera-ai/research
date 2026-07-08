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
import sqlite3
import urllib.error
import urllib.request
from collections.abc import Iterable
from dataclasses import dataclass, field
from typing import Any
from fractions import Fraction

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


# ── new family constants for the four wired verifiers ────────────────────────

FAMILY_SQL = "sql_result_match"      # reference: result-set match via in-memory SQLite
FAMILY_CITATION = FAMILY_RETRIEVAL    # alias — citations are the retrieval family
FAMILY_JSON_SCHEMA = "json_schema"   # intrinsic+reference: JSON parses + conforms to schema
FAMILY_MATH_EXACT = "math_exact"     # reference: strict numeric equality (no fuzzy tolerance)


# ── SQL verifier (reference: execute + compare rows) ─────────────────────────


_SQL_LANGS = {"", "sql", "sqlite", "sqlite3", "mysql", "postgres", "postgresql"}
_SQL_FENCE_RE = re.compile(
    r"```(?:sql|sqlite|sqlite3)?\n(.*?)```", re.DOTALL | re.IGNORECASE
)
# DDL/DML statements we allow for setup in the sandbox.  SELECT is the only
# statement whose *result rows* we compare.  We explicitly block anything that
# touches the filesystem or attaches external databases.
_SQL_FORBIDDEN = re.compile(
    r"\b(ATTACH|DETACH|PRAGMA\s+(?!table_info)|"
    r"LOAD|VACUUM|REINDEX|"
    r"CREATE\s+(?:VIRTUAL|TRIGGER|VIEW)\b)",
    re.IGNORECASE,
)


def _extract_sql(text: str) -> str | None:
    """Pull the first SQL block from *text*: fenced ```sql first, then any
    statement starting with a SQL keyword."""
    for lang, code in extract_code_blocks(text):
        if lang in _SQL_LANGS:
            return code.strip()
    m = _SQL_FENCE_RE.search(text)
    if m:
        return m.group(1).strip()
    # bare-statement fallback: first line that looks like SQL
    for line in text.splitlines():
        stripped = line.strip()
        if stripped and re.match(
            r"(?:SELECT|WITH|INSERT|CREATE|UPDATE|DELETE)\b", stripped, re.IGNORECASE
        ):
            return stripped
    return None


def _sql_split_statements(sql: str) -> list[str]:
    """Naive statement splitter on semicolons (good enough for the gold corpus;
    we don't need a full parser)."""
    stmts = [s.strip() for s in sql.split(";")]
    return [s for s in stmts if s]


def _execute_sql_sandbox(
    sql_text: str, setup_sql: str | None = None
) -> tuple[list[tuple] | None, str]:
    """Execute *sql_text* in a fresh in-memory SQLite.  Optionally run
    *setup_sql* (DDL/DML seed) first.

    Returns ``(rows, detail)``: *rows* is the result-set of the last SELECT
    (or ``[]`` if the last statement was DDL/DML), or ``None`` on error.
    """
    if _SQL_FORBIDDEN.search(sql_text):
        return None, "forbidden_statement"
    conn = sqlite3.connect(":memory:")
    conn.row_factory = sqlite3.Row
    rows: list[tuple] | None = None
    try:
        if setup_sql:
            if _SQL_FORBIDDEN.search(setup_sql):
                return None, "forbidden_in_setup"
            for stmt in _sql_split_statements(setup_sql):
                conn.execute(stmt)
        last_stmt: str | None = None
        for stmt in _sql_split_statements(sql_text):
            last_stmt = stmt
            if _SQL_FORBIDDEN.search(stmt):
                return None, f"forbidden:{stmt[:40]}"
        cur = conn.execute(last_stmt) if last_stmt else None
        if cur is not None:
            fetched = cur.fetchall()
            rows = [tuple(r) for r in fetched]
        else:
            rows = []
        conn.commit()
        return rows, "ok"
    except sqlite3.Error as exc:
        return None, f"sqlite_error:{exc}"
    except Exception as exc:  # defensive — never let the sandbox crash verify()
        return None, f"sandbox_error:{type(exc).__name__}"
    finally:
        conn.close()


def _rows_match(
    got: list[tuple] | None, expected: list[tuple] | list[list] | None
) -> bool:
    """Order-insensitive row comparison.  ``None`` rows never match."""
    if got is None:
        return False
    if expected is None:
        return False
    # normalise: expected may come as list-of-lists from JSON
    exp_tuples = [tuple(r) for r in expected]
    return sorted(got) == sorted(exp_tuples)


def verify_sql(sample: VerifySample) -> VerifyResult:
    """Reference: extract SQL from the response, execute it against an in-memory
    SQLite sandbox, and compare the result rows (order-insensitive) against
    ``expected``.

    ``expected`` shapes accepted:
      - ``{"rows": [[...], ...], "setup_sql": "CREATE TABLE..."}`` — full spec
      - ``{"rows": [...]}`` — just the expected rows (no setup)
      - ``[[...], ...]`` — bare list of expected rows (no setup)
      - ``"SELECT ..."`` — a gold SQL string to execute and compare rows

    Safety: only ``sqlite3`` stdlib, in-memory DB, forbidden statements
    (ATTACH, LOAD, VACUUM, etc.) blocked.  Never persists, never touches the
    filesystem.
    """
    text = extract_output_text(sample.response_payload)
    sql = _extract_sql(text)
    if not sql:
        return _defer(FAMILY_SQL, "reference", "no_sql_found", ("extract=none",))
    expected = sample.expected
    setup_sql: str | None = None
    expected_rows: list[tuple] | None = None
    if isinstance(expected, dict):
        raw_rows = expected.get("rows")
        setup_sql = expected.get("setup_sql") or expected.get("setup")
        if raw_rows is not None:
            expected_rows = [tuple(r) for r in raw_rows]
        elif expected.get("sql"):
            # gold SQL — execute it in the same sandbox and compare
            gold_rows, gold_detail = _execute_sql_sandbox(
                expected["sql"], setup_sql=setup_sql
            )
            if gold_rows is None:
                return _defer(
                    FAMILY_SQL, "reference", "gold_sql_failed", (gold_detail,)
                )
            expected_rows = gold_rows
    elif isinstance(expected, list):
        expected_rows = [tuple(r) for r in expected]
    elif isinstance(expected, str) and expected.strip().upper().startswith(
        ("SELECT", "WITH")
    ):
        gold_rows, gold_detail = _execute_sql_sandbox(expected, setup_sql=setup_sql)
        if gold_rows is None:
            return _defer(
                FAMILY_SQL, "reference", "gold_sql_failed", (gold_detail,)
            )
        expected_rows = gold_rows
    if expected_rows is None:
        return _defer(FAMILY_SQL, "reference", "no_expected_rows")
    got_rows, detail = _execute_sql_sandbox(sql, setup_sql=setup_sql)
    if got_rows is None:
        return _ok(
            0.0,
            FAMILY_SQL,
            "reference",
            "sql_exec_error",
            (f"detail={detail}", f"sql={sql[:60]}"),
        )
    match = _rows_match(got_rows, expected_rows)
    return _ok(
        1.0 if match else 0.0,
        FAMILY_SQL,
        "reference",
        "sql_row_match" if match else "sql_row_mismatch",
        (
            f"got={len(got_rows)}rows",
            f"exp={len(expected_rows)}rows",
            f"detail={detail}",
        ),
    )


# ── Citation verifier (reference: URL resolve + verify) ──────────────────────


_URL_RE = re.compile(
    r"https?://[^\s<>\")\]]+",  # URL until whitespace or closing bracket/quote
    re.IGNORECASE,
)


def _extract_urls(text: str) -> list[str]:
    """Extract all HTTP(S) URLs from *text*, de-duplicated, preserving order."""
    seen: set[str] = set()
    urls: list[str] = []
    for m in _URL_RE.finditer(text):
        url = m.group(0).rstrip(".,;:!?")  # strip trailing punctuation
        if url not in seen:
            seen.add(url)
            urls.append(url)
    return urls


def _url_resolves(url: str, timeout: float = 10.0) -> tuple[bool, str]:
    """HTTP HEAD *url*, return ``(ok, detail)``.  ``True`` iff status < 400.
    Falls back to GET if HEAD returns 405 (some servers reject HEAD)."""
    for method in ("HEAD", "GET"):
        req = urllib.request.Request(url, method=method)
        req.add_header("User-Agent", "AinferaVerify/1.0")
        try:
            with urllib.request.urlopen(req, timeout=timeout) as resp:
                status = resp.status
                if status < 400:
                    return True, f"{method}:{status}"
                if status == 405 and method == "HEAD":
                    continue
                return False, f"{method}:{status}"
        except urllib.error.HTTPError as exc:
            if exc.code == 405 and method == "HEAD":
                continue
            return False, f"{method}:HTTPError:{exc.code}"
        except urllib.error.URLError as exc:
            return False, f"{method}:URLError:{exc.reason}"
        except Exception as exc:
            return False, f"{method}:{type(exc).__name__}"
    return False, "all_methods_failed"


def verify_citation(sample: VerifySample) -> VerifyResult:
    """Reference: extract URLs from the response and verify each resolves
    (HTTP HEAD, status < 400, with GET fallback).

    ``expected``:
      - ``None`` or not provided → intrinsic mode: just check that URLs in the
        response resolve.  If no URLs found → defer (no citations to verify).
      - ``["url1", "url2"]`` → reference mode: verify those specific URLs are
        present AND resolve.
      - ``{"urls": [...]}`` → same as list.

    Returns reward = fraction of URLs that resolved.  If expected URLs are
    given but some are missing from the response, those count as failures.

    Pure stdlib (urllib).
    """
    text = extract_output_text(sample.response_payload)
    response_urls = _extract_urls(text)
    expected = sample.expected

    if isinstance(expected, dict):
        expected_urls = expected.get("urls") or []
    elif isinstance(expected, list):
        expected_urls = expected
    else:
        expected_urls = []

    if expected_urls:
        # reference mode: every expected URL must appear in response AND resolve
        missing = [u for u in expected_urls if u not in response_urls]
        ev: list[str] = []
        n_ok = 0
        for url in expected_urls:
            if url in missing:
                ev.append(f"missing:{url[:50]}")
                continue
            ok, detail = _url_resolves(url)
            ev.append(f"{'ok' if ok else 'fail'}:{url[:50]}({detail})")
            if ok:
                n_ok += 1
        # missing URLs are failures, so denominator = len(expected_urls)
        total = len(expected_urls)
        reward = n_ok / total if total > 0 else 0.0
        return _ok(
            reward,
            FAMILY_CITATION,
            "reference",
            "citation_url_resolve",
            tuple(ev),
        )

    # intrinsic mode: no expected URLs — check URLs that appear in the response
    if not response_urls:
        return _defer(FAMILY_CITATION, "intrinsic", "no_urls_found")
    ev = []
    n_ok = 0
    for url in response_urls:
        ok, detail = _url_resolves(url)
        ev.append(f"{'ok' if ok else 'fail'}:{url[:50]}({detail})")
        if ok:
            n_ok += 1
    reward = n_ok / len(response_urls)
    return _ok(
        reward,
        FAMILY_CITATION,
        "intrinsic",
        "citation_url_resolve",
        tuple(ev),
    )


# ── JSON Schema verifier (intrinsic: parse + validate against schema) ────────


def _validate_json_schema(value: Any, schema: dict[str, Any]) -> tuple[bool, str]:
    """Validate *value* against a JSON-Schema *schema* dict.

    Supports: type, required, properties, items, enum, minimum, maximum,
    minLength, maxLength, minItems, maxItems, additionalProperties.

    Uses ``jsonschema`` if available (preferred), otherwise falls back to a
    hand-rolled validator covering the common subset above.  The fallback is
    honest about its limitations — it validates what it can and passes what it
    can't, rather than falsely failing.
    """
    try:
        import jsonschema  # type: ignore[import-untyped]

        jsonschema.validate(instance=value, schema=schema)
        return True, "jsonschema:ok"
    except ImportError:
        pass
    except Exception as exc:
        return False, f"jsonschema:{exc}"
    # ── hand-rolled fallback (stdlib only) ──
    return _validate_json_schema_stdlib(value, schema)


def _validate_json_schema_stdlib(
    value: Any, schema: dict[str, Any]
) -> tuple[bool, str]:
    """Minimal JSON-Schema validator (stdlib).  Covers the subset the gold
    corpus and typical response_format schemas use."""
    if not isinstance(schema, dict):
        return True, "no_schema"  # nothing to validate against

    # type
    jtype = schema.get("type")
    if isinstance(jtype, str):
        jtype = [jtype]
    if isinstance(jtype, list):
        type_checks = {
            "object": lambda v: isinstance(v, dict),
            "array": lambda v: isinstance(v, list),
            "string": lambda v: isinstance(v, str),
            "number": lambda v: isinstance(v, (int, float))
            and not isinstance(v, bool),
            "integer": lambda v: isinstance(v, int) and not isinstance(v, bool),
            "boolean": lambda v: isinstance(v, bool),
            "null": lambda v: v is None,
        }
        if not any(type_checks[t](value) for t in jtype if t in type_checks):
            return False, f"type!={'|'.join(jtype)}"

    # enum
    if "enum" in schema and value not in schema["enum"]:
        return False, f"enum:{value}"

    # string constraints
    if isinstance(value, str):
        if "minLength" in schema and len(value) < schema["minLength"]:
            return False, f"minLength:{schema['minLength']}"
        if "maxLength" in schema and len(value) > schema["maxLength"]:
            return False, f"maxLength:{schema['maxLength']}"
        if "pattern" in schema:
            if not re.search(schema["pattern"], value):
                return False, f"pattern:{schema['pattern']}"

    # number constraints
    if isinstance(value, (int, float)) and not isinstance(value, bool):
        if "minimum" in schema and value < schema["minimum"]:
            return False, f"minimum:{schema['minimum']}"
        if "maximum" in schema and value > schema["maximum"]:
            return False, f"maximum:{schema['maximum']}"
        if "exclusiveMinimum" in schema and value <= schema["exclusiveMinimum"]:
            return False, f"exclusiveMinimum:{schema['exclusiveMinimum']}"
        if "exclusiveMaximum" in schema and value >= schema["exclusiveMaximum"]:
            return False, f"exclusiveMaximum:{schema['exclusiveMaximum']}"
        if "multipleOf" in schema:
            mo = schema["multipleOf"]
            if mo and (value / mo) != int(value / mo):
                return False, f"multipleOf:{mo}"

    # object constraints
    if isinstance(value, dict):
        missing = [k for k in (schema.get("required") or []) if k not in value]
        if missing:
            return False, f"missing_keys={','.join(missing)}"
        props = schema.get("properties") or {}
        for key, sub_schema in props.items():
            if key in value and isinstance(sub_schema, dict):
                ok, detail = _validate_json_schema_stdlib(value[key], sub_schema)
                if not ok:
                    return False, f"property:{key}:{detail}"
        ap = schema.get("additionalProperties")
        if ap is False:
            extra = [k for k in value if k not in props]
            if extra:
                return False, f"additionalProperties:{','.join(extra)}"
        elif isinstance(ap, dict):
            for key in value:
                if key not in props:
                    ok, detail = _validate_json_schema_stdlib(value[key], ap)
                    if not ok:
                        return False, f"additional:{key}:{detail}"

    # array constraints
    if isinstance(value, list):
        if "minItems" in schema and len(value) < schema["minItems"]:
            return False, f"minItems:{schema['minItems']}"
        if "maxItems" in schema and len(value) > schema["maxItems"]:
            return False, f"maxItems:{schema['maxItems']}"
        items_schema = schema.get("items")
        if isinstance(items_schema, dict):
            for i, item in enumerate(value):
                ok, detail = _validate_json_schema_stdlib(item, items_schema)
                if not ok:
                    return False, f"items[{i}]:{detail}"

    return True, "ok"


def verify_json_schema(sample: VerifySample) -> VerifyResult:
    """Intrinsic: parse JSON from the response and validate it against a
    JSON-Schema dict.

    The schema is taken from ``sample.expected`` (if it's a dict with
    ``"type"`` or ``"properties"`` or ``"$schema"``), or from the request's
    ``response_format.json_schema``.

    Returns ``reward=1.0`` if the JSON parses AND conforms, ``0.0`` if it
    parses but doesn't conform or doesn't parse, and ``None`` (defer) if no
    schema is available to validate against.
    """
    text = extract_output_text(sample.response_payload).strip()
    if not text:
        return _defer(FAMILY_JSON_SCHEMA, "intrinsic", "empty_output")
    obj, why = _loads_lenient(text)
    if obj is _SENTINEL:
        return _ok(
            0.0,
            FAMILY_JSON_SCHEMA,
            "intrinsic",
            "json_parse_fail",
            (f"json={why}",),
        )
    # find the schema
    schema: dict[str, Any] | None = None
    if isinstance(sample.expected, dict) and (
        "type" in sample.expected
        or "properties" in sample.expected
        or "$schema" in sample.expected
        or "required" in sample.expected
    ):
        schema = sample.expected
    elif isinstance(sample.expected, dict) and "schema" in sample.expected:
        schema = sample.expected["schema"]
    if schema is None:
        schema = extract_request_json_schema(sample.request_payload)
    if schema is None:
        return _defer(
            FAMILY_JSON_SCHEMA, "intrinsic", "no_schema_to_validate_against"
        )
    ok, detail = _validate_json_schema(obj, schema)
    return _ok(
        1.0 if ok else 0.0,
        FAMILY_JSON_SCHEMA,
        "intrinsic",
        "json_schema_valid" if ok else "json_schema_invalid",
        (f"schema={detail}",),
    )


# ── Math exact-match verifier (reference: strict numeric equality) ───────────


def _parse_numeric(s: str) -> float | Fraction | None:
    """Parse a numeric value from *s*, supporting:
      - integers: ``42``
      - decimals: ``3.14``
      - negatives: ``-7``
      - fractions: ``3/4``, ``-2/3``
      - LaTeX fractions: ``\\frac{1}{2}``
      - percentages: ``50%`` → 50.0 (the number, not the ratio)
      - comma-separated thousands: ``1,000``

    Returns a ``float`` (or ``Fraction`` for exact fraction comparison), or
    ``None`` if *s* is not numeric.
    """
    s = s.strip().replace(" ", "").replace(",", "")
    if not s:
        return None
    # strip trailing % — we compare the number, not the ratio
    s = s.rstrip("%")
    # LaTeX \frac{a}{b}
    m = re.match(r"\\frac\{(-?\d+)\}\{(-?\d+)\}", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return None
        return Fraction(num, den)
    # simple fraction a/b
    m = re.match(r"^(-?\d+)/(-?\d+)$", s)
    if m:
        num, den = int(m.group(1)), int(m.group(2))
        if den == 0:
            return None
        return Fraction(num, den)
    # plain integer or decimal
    m = re.match(r"^(-?\d+)(?:\.(\d+))?$", s)
    if m:
        try:
            return float(s)
        except ValueError:
            return None
    # scientific notation
    m = re.match(r"^(-?\d+\.?\d*)[eE]([+-]?\d+)$", s)
    if m:
        try:
            return float(s)
        except ValueError:
            return None
    return None


def _numbers_equal(
    a: float | Fraction, b: float | Fraction
) -> bool:
    """Strict numeric equality — no fuzzy tolerance.  Fractions are compared
    exactly; floats use ``==`` (with a tiny epsilon for representation noise
    from parsing, e.g. ``0.1 + 0.2``)."""
    if isinstance(a, Fraction) and isinstance(b, Fraction):
        return a == b
    # mixed: convert Fraction to float for comparison
    fa = float(a) if isinstance(a, Fraction) else a
    fb = float(b) if isinstance(b, Fraction) else b
    if fa == fb:
        return True
    # guard against float representation noise (e.g. 1/3 vs 0.333...)
    return math.isclose(fa, fb, rel_tol=1e-12, abs_tol=1e-15)


def verify_math_exact(sample: VerifySample) -> VerifyResult:
    """Reference: strict numeric equality between the response's final answer
    and ``expected``.

    Unlike ``verify_answer`` (which uses ``_answer_present`` with a fuzzy
    ``isclose`` tolerance of ``1e-6``), this verifier demands exact numeric
    equality.  Supports fractions (``3/4``, ``\\frac{1}{2}``), decimals,
    negatives, and scientific notation.

    Returns ``reward=1.0`` for exact match, ``0.0`` for mismatch, ``None``
    (defer) if no ``expected`` is provided or if neither side is numeric.
    """
    if sample.expected is None:
        return _defer(FAMILY_MATH_EXACT, "reference", "no_gold")
    text = extract_output_text(sample.response_payload)
    got = _final_answer(text)
    exp_str = str(sample.expected).strip()
    gn = _parse_numeric(got)
    en = _parse_numeric(exp_str)
    if gn is None or en is None:
        # if either side isn't numeric, defer rather than fail — this verifier
        # is specifically for numeric exact match, not string comparison
        return _defer(
            FAMILY_MATH_EXACT,
            "reference",
            "non_numeric",
            (f"got={got[:24]}", f"exp={exp_str[:24]}"),
        )
    ok = _numbers_equal(gn, en)
    return _ok(
        1.0 if ok else 0.0,
        FAMILY_MATH_EXACT,
        "reference",
        "math_exact" if ok else "math_exact_mismatch",
        (f"got={got}", f"exp={exp_str}"),
    )


# ── Math step/derivation verifier (reference: SymPy chain checking) ──────────


FAMILY_MATH_STEPS = "math_steps"  # reference: verify each derivation step via SymPy


def _extract_math_steps(text: str) -> list[tuple[str, str]]:
    """Extract (lhs, rhs) equation steps from *text*.

    Looks for lines that contain '=' and look like mathematical equations
    (not prose assignments). Returns a list of (lhs, rhs) string pairs.

    Handles:
      - ``x = 5``  →  ("x", "5")
      - ``2x + 3 = 7``  →  ("2x + 3", "7")
      - ``x^2 - 1 = (x-1)(x+1)``  →  ("x^2 - 1", "(x-1)(x+1)")
      - ``=> y = 3``  →  ("y", "3")  (arrow prefix stripped)
      - LaTeX ``\\frac{a}{b} = c``  →  ("\\frac{a}{b}", "c")

    Skips lines that are clearly prose (no math operators) or code.
    """
    steps: list[tuple[str, str]] = []
    # Math-ish characters: digits, operators, variables, brackets, etc.
    _math_chars = re.compile(r"[\d+\-*/^=()<>{}\[\].,|\\a-zA-Z]")
    for line in text.splitlines():
        line = line.strip()
        if not line or len(line) < 3:
            continue
        # Skip code fences
        if line.startswith("```") or line.startswith(">>>"):
            continue
        # Strip leading step markers: "1.", "Step 1:", "=>", "→", "∴", "⇒"
        line = re.sub(r"^(?:step\s*\d+[:.]?\s*|\d+[.:)]\s*)", "", line, flags=re.IGNORECASE)
        line = re.sub(r"^(?:=>|→|⇒|∴|⟹)\s*", "", line)
        # Must have exactly one '=' that separates LHS and RHS (not '==' or '>=')
        if "=" not in line:
            continue
        # Skip comparison operators
        if any(op in line for op in ("==", "!=", ">=", "<=", ":=")):
            continue
        # Split on first '='
        eq_idx = line.index("=")
        lhs = line[:eq_idx].strip()
        rhs = line[eq_idx + 1:].strip()
        if not lhs or not rhs:
            continue
        # At least one side should have math-looking content
        if not _math_chars.search(lhs) or not _math_chars.search(rhs):
            continue
        steps.append((lhs, rhs))
    return steps


def _sympy_simplify_safe(expr_str: str) -> Any | None:
    """Try to parse and simplify *expr_str* with SymPy. Returns None on failure."""
    try:
        from sympy import sympify, simplify

        expr = sympify(expr_str, convert_xor=True, rational=True)
        return simplify(expr)
    except Exception:
        return None


def _normalize_math_expr(expr_str: str) -> str:
    """Normalize a math expression string for SymPy parsing.

    SymPy's ``sympify`` doesn't handle implicit multiplication (``2x`` → ``2*x``)
    or ``^`` for power.  This converts common math notation to SymPy-parseable
    form.
    """
    s = expr_str.strip()
    # Convert ^ to ** (power operator)
    s = s.replace("^", "**")
    # Insert * between a digit and a letter (2x → 2*x, 3y → 3*y)
    s = re.sub(r"(\d)([a-zA-Z])", r"\1*\2", s)
    # Insert * between a closing paren and a letter/digit: (x+1)(x-1) → (x+1)*(x-1)
    s = re.sub(r"\)(\w)", r")*\1", s)
    # Insert * between a letter and a closing paren: x(y+1) → x*(y+1)
    # (but not for function names like sin, cos, etc.)
    return s


def _sympy_equal(lhs_str: str, rhs_str: str) -> bool | None:
    """Check if lhs == rhs symbolically using SymPy.

    Returns True/False if the comparison could be made, None if either side
    fails to parse.

    Strategy (in order):
      1. Parse both sides, try ``simplify(LHS - RHS) == 0`` (catches identities).
      2. If that doesn't resolve, try numeric evaluation (for arithmetic steps
         like ``2 + 3 = 5`` where both sides evaluate to numbers).
      3. If neither works (e.g. equation transformations like ``2x + 3 = 7``
         that are only true for a specific x), return None — we can't
         verify the step without knowing x, so defer rather than fail.
    """
    try:
        from sympy import sympify, simplify, Eq, N

        lhs = sympify(_normalize_math_expr(lhs_str), convert_xor=True, rational=True)
        rhs = sympify(_normalize_math_expr(rhs_str), convert_xor=True, rational=True)
        # Direct equality check first (fast path)
        if lhs == rhs:
            return True
        # Try simplification: lhs - rhs == 0
        diff = simplify(lhs - rhs)
        if diff == 0:
            return True
        # Try the equality relation
        eq = Eq(lhs, rhs)
        simplified = simplify(eq)
        if simplified is True:
            return True
        if simplified is False:
            return False
        # If simplification can't resolve it, try numeric evaluation.
        # This catches arithmetic steps (2 + 3 = 5) where both sides are
        # pure numbers, even if simplify doesn't reduce the difference to 0
        # due to form differences.
        try:
            lhs_num = N(lhs)
            rhs_num = N(rhs)
            if lhs_num.is_number and rhs_num.is_number:
                diff_num = abs(float(lhs_num) - float(rhs_num))
                if diff_num < 1e-10:
                    return True
                return False
        except Exception:
            pass
        # Can't verify — it's likely an equation transformation (e.g. 2x+3=7)
        # that's only true for a specific x value. Return None to defer.
        return None
    except Exception:
        return None


def _sympy_equal_substituted(
    lhs_str: str, rhs_str: str, var_name: str, var_value: float
) -> bool | None:
    """Check if lhs == rhs after substituting *var_name* = *var_value*.

    This is for verifying equation-transformation steps like ``2x + 3 = 7``
    where we know x=2 from the final answer.  After substitution both sides
    should evaluate to the same number.
    """
    try:
        from sympy import sympify, simplify, N, Symbol

        lhs = sympify(_normalize_math_expr(lhs_str), convert_xor=True, rational=True)
        rhs = sympify(_normalize_math_expr(rhs_str), convert_xor=True, rational=True)
        var = Symbol(var_name)
        lhs_sub = lhs.subs(var, var_value)
        rhs_sub = rhs.subs(var, var_value)
        # Try numeric evaluation
        try:
            lhs_num = N(lhs_sub)
            rhs_num = N(rhs_sub)
            if lhs_num.is_number and rhs_num.is_number:
                diff_num = abs(float(lhs_num) - float(rhs_num))
                if diff_num < 1e-6:
                    return True
                return False
        except Exception:
            pass
        # Try symbolic simplification after substitution
        diff = simplify(lhs_sub - rhs_sub)
        if diff == 0:
            return True
        return None
    except Exception:
        return None


def _extract_var_assignment(steps: list[tuple[str, str]], text: str) -> tuple[str, float] | None:
    """Try to find a variable assignment from the derivation.

    Looks for ``x = <number>`` in the steps or the \\boxed answer, and returns
    (variable_name, value).  Returns None if no assignment is found.
    """
    # Check steps in reverse (last assignment is the answer)
    for lhs, rhs in reversed(steps):
        # Simple: lhs is a single variable, rhs is a number
        lhs_clean = lhs.strip().replace(" ", "")
        rhs_num = _parse_numeric(rhs.strip())
        if rhs_num is not None and re.fullmatch(r"[a-zA-Z]", lhs_clean):
            return lhs_clean, float(rhs_num)
    # Check \boxed{} for a number
    final = _final_answer(text)
    final_num = _parse_numeric(final)
    if final_num is not None:
        # Try to find the variable from the steps
        for lhs, rhs in steps:
            lhs_clean = lhs.strip().replace(" ", "")
            if re.fullmatch(r"[a-zA-Z]", lhs_clean):
                return lhs_clean, float(final_num)
    return None


def verify_math(sample: VerifySample) -> VerifyResult:
    """Reference: verify mathematical reasoning steps using SymPy.

    Extracts equation steps (``LHS = RHS``) from the response and checks each
    one symbolically: ``simplify(LHS - RHS) == 0``.  This is a *substantive*
    check — a wrong intermediate step (even if the final answer is right) is
    caught, and a correct chain (even if the final answer is formatted oddly)
    passes.

    If ``expected`` is provided, the final step's RHS (or \\boxed answer) must
    also match it numerically (via ``_parse_numeric`` + ``_numbers_equal``).

    Reward:
      - 1.0 if ALL steps verify AND the final answer matches expected (if given)
      - 0.0 if any step fails the SymPy check
      - Fractional reward = (verified steps / total steps) if some pass, some fail
      - None (defer) if no steps found, or SymPy unavailable, or no expected
        and the chain can't be substantively checked

    SymPy is an optional dependency. If unavailable, falls back to
    ``verify_math_exact`` (final-answer-only checking).
    """
    # Check SymPy availability
    try:
        import sympy  # noqa: F401
    except ImportError:
        # Fall back to exact-answer-only checking
        return verify_math_exact(sample)

    text = extract_output_text(sample.response_payload)
    steps = _extract_math_steps(text)

    if not steps:
        # No derivation steps found — try final-answer-only via math_exact
        return verify_math_exact(sample)

    ev: list[str] = []
    n_verified = 0
    n_failed = 0
    n_unparseable = 0

    # First pass: try symbolic/numeric verification
    step_results: list[bool | None] = []
    for lhs, rhs in steps:
        result = _sympy_equal(lhs, rhs)
        step_results.append(result)

    # If some steps couldn't be verified symbolically, try substitution:
    # find the variable assignment (e.g. x=2 from the final step) and
    # re-verify the unparseable steps with that substitution.
    unparseable_indices = [i for i, r in enumerate(step_results) if r is None]
    if unparseable_indices:
        var_assignment = _extract_var_assignment(steps, text)
        if var_assignment:
            var_name, var_value = var_assignment
            ev.append(f"substituting {var_name}={var_value}")
            for i in unparseable_indices:
                lhs, rhs = steps[i]
                result = _sympy_equal_substituted(lhs, rhs, var_name, var_value)
                step_results[i] = result

    for i, (lhs, rhs) in enumerate(steps):
        result = step_results[i]
        if result is True:
            n_verified += 1
            ev.append(f"step{i}: {lhs} = {rhs} ✓")
        elif result is False:
            n_failed += 1
            ev.append(f"step{i}: {lhs} ≠ {rhs} ✗")
        else:
            n_unparseable += 1
            ev.append(f"step{i}: {lhs} = {rhs} ? (unparseable)")

    total = len(steps)
    n_checkable = n_verified + n_failed

    # If no steps could be checked symbolically, defer to math_exact
    if n_checkable == 0:
        return verify_math_exact(sample)

    # Check the final answer against expected, if provided
    final_ok = True
    if sample.expected is not None:
        # The final step's RHS is the answer, or use _final_answer
        final = _final_answer(text)
        exp_str = str(sample.expected).strip()
        gn = _parse_numeric(final)
        en = _parse_numeric(exp_str)
        if gn is not None and en is not None:
            final_ok = _numbers_equal(gn, en)
            ev.append(f"final: {final} vs {exp_str} {'✓' if final_ok else '✗'}")
        else:
            # Non-numeric expected — try string match
            final_ok = _normalize(final) == _normalize(exp_str)
            ev.append(f"final: {final[:24]} vs {exp_str[:24]} {'✓' if final_ok else '✗'}")

    if n_failed > 0:
        # Any failed step → partial credit
        reward = n_verified / total
        return _ok(
            reward,
            FAMILY_MATH_STEPS,
            "reference",
            f"math_steps:{n_verified}/{total}_verified",
            tuple(ev),
        )

    # All checkable steps verified
    if final_ok:
        reward = 1.0
    else:
        # Steps OK but final answer wrong → 0.5 (partial credit for correct work)
        reward = 0.5
    return _ok(
        reward,
        FAMILY_MATH_STEPS,
        "reference",
        f"math_steps:{n_verified}/{total}_verified" + ("" if final_ok else "_final_mismatch"),
        tuple(ev),
    )


# ── QA / evidence verifier (reference: factuality against source evidence) ────


FAMILY_QA = "qa_evidence"  # reference: answer grounded in provided evidence


def _extract_sentences(text: str) -> list[str]:
    """Split *text* into sentences (simple heuristic on ., !, ?)."""
    parts = re.split(r"(?<=[.!?])\s+", text)
    return [s.strip() for s in parts if s.strip() and len(s.strip()) > 2]


def _normalize_text(s: str) -> str:
    """Lowercase, collapse whitespace, strip punctuation for comparison."""
    s = s.lower().strip()
    s = re.sub(r"[^\w\s]", " ", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


def _tokenize(s: str) -> set[str]:
    """Tokenize into a set of lowercase words (stop words removed)."""
    _STOP = {
        "the", "a", "an", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "must", "shall", "can", "need", "dare",
        "ought", "used", "to", "of", "in", "for", "on", "with", "at", "by",
        "from", "as", "into", "through", "during", "before", "after", "above",
        "below", "up", "down", "out", "off", "over", "under", "again", "further",
        "then", "once", "here", "there", "when", "where", "why", "how", "all",
        "each", "few", "more", "most", "other", "some", "such", "no", "nor",
        "not", "only", "own", "same", "so", "than", "too", "very", "just",
        "and", "or", "but", "if", "because", "while", "about", "against",
        "between", "into", "this", "that", "these", "those", "it", "its",
        "i", "you", "he", "she", "we", "they", "them", "his", "her", "their",
        "what", "which", "who", "whom", "whose",
    }
    tokens = re.findall(r"\b[a-z][a-z0-9]*\b", s.lower())
    return {t for t in tokens if t not in _STOP and len(t) > 2}


def _entailment_overlap(
    claim: str, evidence: str, min_ratio: float = 0.5
) -> tuple[bool, float]:
    """Check if *evidence* supports *claim* via token overlap (lightweight
    textual entailment proxy).

    Returns ``(supported, overlap_ratio)``.  *supported* is True iff the
    Jaccard ratio of content tokens (claim ∩ evidence / claim) >= *min_ratio*.
    """
    claim_tokens = _tokenize(claim)
    if not claim_tokens:
        return True, 1.0  # no claim to verify → vacuously supported
    evidence_tokens = _tokenize(evidence)
    if not evidence_tokens:
        return False, 0.0
    overlap = claim_tokens & evidence_tokens
    ratio = len(overlap) / len(claim_tokens)
    return ratio >= min_ratio, ratio


def verify_qa(sample: VerifySample) -> VerifyResult:
    """Reference: verify a factual answer against source evidence.

    ``expected`` shapes:
      - ``{"evidence": "...", "answer": "..."}`` — evidence text + the gold answer
      - ``{"evidence": "..."}`` — evidence only; check the response is grounded
      - ``str`` — treat as the gold answer (no evidence → answer-match only)

    The verifier checks TWO things:
      1. **Grounding**: the response's claims are supported by the evidence
         (token-overlap entailment proxy). Each sentence in the response that
         contains a factual claim must have sufficient token overlap with the
         evidence text.
      2. **Correctness**: the gold answer appears in the response (if provided).

    Reward:
      - 1.0 if both grounding passes AND the answer matches
      - 0.5 if grounding passes but answer doesn't match (or vice versa)
      - 0.0 if grounding fails (response contains unsupported claims)
      - None (defer) if no evidence and no gold answer

    This is a substantive check: a response with a correct answer but
    fabricated reasoning fails grounding. A response with good reasoning but
    wrong final answer gets partial credit.
    """
    text = extract_output_text(sample.response_payload)
    if not text.strip():
        return _defer(FAMILY_QA, "reference", "empty_output")

    expected = sample.expected
    evidence: str | None = None
    gold_answer: str | None = None

    if isinstance(expected, dict):
        evidence = expected.get("evidence") or expected.get("source") or expected.get("context")
        gold_answer = expected.get("answer") or expected.get("expected")
    elif isinstance(expected, str):
        gold_answer = expected

    # If no evidence and no gold answer, defer
    if not evidence and not gold_answer:
        return _defer(FAMILY_QA, "reference", "no_evidence_no_gold")

    ev: list[str] = []

    # ── 1. Grounding check (if evidence provided) ──
    grounding_ok = True
    if evidence:
        sentences = _extract_sentences(text)
        n_supported = 0
        n_unsupported = 0
        for sent in sentences:
            supported, ratio = _entailment_overlap(sent, evidence)
            if supported:
                n_supported += 1
            else:
                n_unsupported += 1
                ev.append(f"unsupported: {sent[:40]}... (overlap={ratio:.2f})")
        # Grounding passes if >= 70% of sentences are supported
        total = n_supported + n_unsupported
        if total > 0:
            grounding_ratio = n_supported / total
            grounding_ok = grounding_ratio >= 0.7
            ev.append(f"grounding: {n_supported}/{total} sentences supported")
        else:
            grounding_ok = True
            ev.append("grounding: no sentences to check")

    # ── 2. Answer correctness (if gold answer provided) ──
    answer_ok = True
    if gold_answer:
        answer_ok = _answer_present(text, gold_answer)
        ev.append(f"answer_match: {answer_ok} (gold={str(gold_answer)[:24]})")

    # ── Combine ──
    if grounding_ok and answer_ok:
        reward = 1.0
        detail = "qa_grounded+correct"
    elif grounding_ok and not answer_ok:
        reward = 0.5
        detail = "qa_grounded_wrong_answer"
    elif not grounding_ok and answer_ok:
        reward = 0.5
        detail = "qa_ungrounded_correct_answer"
    else:
        reward = 0.0
        detail = "qa_ungrounded_wrong_answer"

    return _ok(
        reward,
        FAMILY_QA,
        "reference",
        detail,
        tuple(ev),
    )


_FAMILY_VERIFIERS = {
    FAMILY_CODE: verify_code,
    FAMILY_SCHEMA: verify_schema,
    FAMILY_TOOL: verify_tool,
    FAMILY_ANSWER: verify_answer,
    FAMILY_RETRIEVAL: verify_citation,
    FAMILY_SQL: verify_sql,
    FAMILY_JSON_SCHEMA: verify_json_schema,
    FAMILY_MATH_EXACT: verify_math_exact,
    FAMILY_MATH_STEPS: verify_math,
    FAMILY_QA: verify_qa,
}

# Direct task_type → verifier mapping for the new verifier families.
# These task_types are not in the canonical 7 (which go through
# verifiability_of → family → _FAMILY_VERIFIERS); they dispatch directly.
_TASK_TYPE_VERIFIERS: dict[str, callable] = {  # type: ignore[type-arg]
    "sql": verify_sql,
    "citation": verify_citation,
    "json_schema": verify_json_schema,
    "math_exact": verify_math_exact,
    "math": verify_math,  # SymPy step/derivation checking (was math_exact alias)
    "qa": verify_qa,      # evidence/factuality checking
    "evidence": verify_qa,  # convenience alias
}


def verify(sample: VerifySample) -> VerifyResult:
    """Dispatch a sample to its task_type's verifier family. Subjective tasks (no
    anchor) defer to the Council immediately.

    Dispatch order:
      1. Direct task_type match in _TASK_TYPE_VERIFIERS (sql, citation,
         json_schema, math_exact, math, qa, evidence) → that verifier.
      2. verifiability_of(task_type) → family → _FAMILY_VERIFIERS.
      3. Subjective / unknown → defer to Council.
    """
    # 1. direct task_type dispatch for the new verifiers
    direct = _TASK_TYPE_VERIFIERS.get(sample.task_type or "")
    if direct is not None:
        return direct(sample)
    # 2. family-based dispatch (canonical 7 task types)
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
