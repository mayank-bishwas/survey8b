#!/usr/bin/env python3
"""Survey8B faceted-filter API.

Serves live, jointly-filtered persona counts to the frontend so the three
wired filter dimensions (region, role_function, economic_motivation) behave
as standard faceted search: each field's option counts and the overall
"Total audience" figure reflect the intersection of whatever is currently
selected across all three fields, recomputed on every change.

Single endpoint:

    POST /api/facets
    Body: {"region": [...], "role_function": [...], "economic_motivation": [...]}
    (any/all arrays may be empty or omitted, meaning "no filter on this field")

    Response: {
      "total": <int>,                      # count under the full combined selection
      "facets": {
        "region": [{"value": ..., "count": ...}, ...],
        "role_function": [...],
        "economic_motivation": [...]
      }
    }

Each field's facet counts are computed with that field's OWN selection
excluded (but the other two fields' selections applied) — the standard
faceted-filtering pattern, so a field's option counts show "how many more
matches if I also picked this," not counts that have already collapsed to
zero once you've selected something in that same field.

Second endpoint:

    POST /api/run-survey
    Body: {
      "question": "...",
      "region": [...], "role_function": [...], "economic_motivation": [...],
      "sample_size": <int>
    }

    Runs the real survey8b_step2.py pipeline end to end: validates the
    requested sample size against the actual matched population, derives
    response categories (gpt-4o-mini), samples personas from Neon, runs the
    per-persona Q&A calls (gpt-4o-mini, concurrently), and generates the
    key-takeaway/summary (Gemma via OpenRouter).

    Response: {
      "filters": {...}, "sample_size": <int>, "matched_count": <int>,
      "categories": [...], "distribution": [{"category","count","pct"}, ...],
      "key_takeaway": "...", "summary": "...",
      "results": [{"persona_id","question","persona_description","answer","reasoning"}, ...],
      "cost": {...}, "timing": {...}
    }

    A 400 response means the request itself was invalid (empty question,
    non-positive sample size, or sample size exceeding the matched
    population for the submitted filters) — the frontend should show the
    error message as-is rather than retrying. A 500 means something failed
    mid-pipeline (DB or model-provider error).
"""
from __future__ import annotations

import functools
import json
import os
import re
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from typing import Any

import psycopg
from openai import OpenAI

import survey8b_step2 as s2

TABLE_NAME = "personas_matraix_subset"
FACET_COLUMNS = ["region", "role_function", "economic_motivation"]
PORT = int(os.environ.get("PORT") or os.environ.get("SURVEY8B_API_PORT", "5001"))
# Persona Q&A calls are I/O-bound and independent, so they run concurrently
# rather than sequentially — at the product's real default sample size (50),
# sequential gpt-4o-mini calls (~2s each) would take ~100s on their own,
# well past the ~35-50s total runtime measured earlier (at sample size 10).
PERSONA_CONCURRENCY = int(os.environ.get("SURVEY8B_PERSONA_CONCURRENCY", "10"))


class ValidationError(Exception):
    """Raised for a bad request (400), as opposed to a pipeline failure (500)."""


def _looks_degenerate(text: str) -> bool:
    """True for junk model output like "-0.0" that's non-empty but useless."""
    stripped = text.strip()
    if len(stripped) < 15:
        return True
    return not re.search(r"[a-zA-Z]{3,}", stripped)


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        if line.startswith("export "):
            line = line[7:].lstrip()
        key, sep, value = line.partition("=")
        if not sep:
            continue
        key = key.strip()
        value = value.strip().strip("'").strip('"')
        if key and key not in os.environ:
            os.environ[key] = value


def build_where(selections: dict[str, list[str]], exclude: str | None = None) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    for column in FACET_COLUMNS:
        if column == exclude:
            continue
        values = [v for v in selections.get(column, []) if v]
        if not values:
            continue
        placeholders = ", ".join(["%s"] * len(values))
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)
    where_sql = f" WHERE {' AND '.join(clauses)}" if clauses else ""
    return where_sql, params


def get_facets(conn: psycopg.Connection, selections: dict[str, list[str]]) -> dict[str, Any]:
    with conn.cursor() as cur:
        total_where, total_params = build_where(selections)
        cur.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}{total_where}", total_params)
        total = cur.fetchone()[0]

        facets: dict[str, list[dict[str, Any]]] = {}
        for column in FACET_COLUMNS:
            where_sql, params = build_where(selections, exclude=column)
            sql = (
                f"SELECT {column}, COUNT(*) FROM {TABLE_NAME}{where_sql}"
                f"{' AND ' if where_sql else ' WHERE '}{column} IS NOT NULL"
                f" GROUP BY {column}"
            )
            cur.execute(sql, params)
            facets[column] = [
                {"value": value, "count": count} for value, count in cur.fetchall()
            ]

    return {"total": total, "facets": facets}


def _parse_filters(body: dict[str, Any]) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    for column in FACET_COLUMNS:
        raw = body.get(column) or []
        if not isinstance(raw, list):
            raise ValidationError(f"{column!r} must be a list of strings")
        values = [v.strip() for v in raw if isinstance(v, str) and v.strip()]
        if values:
            filters[column] = values
    return filters


def _run_one_persona(
    row: dict[str, Any],
    question: str,
    categories: list[str],
    persona_call_fn: s2.CallFn,
) -> dict[str, Any] | None:
    persona_text = s2.persona_sentence(row)
    parse_error: Exception | None = None
    for _attempt in range(3):
        try:
            call_start = time.perf_counter()
            parsed, raw_text, usage = s2.call_model_for_persona(
                persona_call_fn, persona_text, question, categories
            )
            usage["elapsed_seconds"] = round(time.perf_counter() - call_start, 3)
            return {
                "persona_id": row["id"],
                "question": question,
                "persona_description": persona_text,
                "answer": parsed["category"],
                "reasoning": parsed["reasoning"],
                "response": parsed,  # kept for summarise()'s expected shape
                "usage": usage,
            }
        except ValueError as exc:
            parse_error = exc
    print(f"[survey8b_api] persona {row['id']} SKIPPED (malformed response 3x): {parse_error}")
    return None


SURVEY_RUNS_TABLE_SQL = """
CREATE TABLE IF NOT EXISTS survey_runs (
    id SERIAL PRIMARY KEY,
    created_at TIMESTAMPTZ NOT NULL DEFAULT now(),
    status TEXT NOT NULL DEFAULT 'success',
    error_message TEXT,
    question TEXT,
    filters JSONB,
    sample_size INT,
    matched_count INT,
    categories JSONB,
    distribution JSONB,
    key_takeaway TEXT,
    summary TEXT,
    category_model TEXT,
    persona_model TEXT,
    summary_model TEXT,
    cost JSONB,
    timing JSONB,
    copy_count INT NOT NULL DEFAULT 0,
    screenshot_count INT NOT NULL DEFAULT 0,
    download_count INT NOT NULL DEFAULT 0
)
"""

# Migrates a table created before status/error_message existed, and relaxes
# NOT NULL constraints that no longer hold once failed/rejected attempts are
# logged too (a validation error can happen before question or filters are
# even parsed). No-ops on a fresh table, since the CREATE above already
# matches this shape.
SURVEY_RUNS_MIGRATE_SQL = """
ALTER TABLE survey_runs ADD COLUMN IF NOT EXISTS status TEXT NOT NULL DEFAULT 'success';
ALTER TABLE survey_runs ADD COLUMN IF NOT EXISTS error_message TEXT;
ALTER TABLE survey_runs ALTER COLUMN question DROP NOT NULL;
ALTER TABLE survey_runs ALTER COLUMN filters DROP NOT NULL;
ALTER TABLE survey_runs ALTER COLUMN sample_size DROP NOT NULL;
ALTER TABLE survey_runs ALTER COLUMN matched_count DROP NOT NULL;
ALTER TABLE survey_runs ALTER COLUMN categories DROP NOT NULL;
ALTER TABLE survey_runs ALTER COLUMN distribution DROP NOT NULL;
ALTER TABLE survey_runs ADD COLUMN IF NOT EXISTS copy_count INT NOT NULL DEFAULT 0;
ALTER TABLE survey_runs ADD COLUMN IF NOT EXISTS screenshot_count INT NOT NULL DEFAULT 0;
ALTER TABLE survey_runs ADD COLUMN IF NOT EXISTS download_count INT NOT NULL DEFAULT 0;
"""

# What the frontend is allowed to increment, mapped to the column it bumps.
# A whitelist rather than string interpolation of whatever the client sends,
# since /api/track is a public unauthenticated endpoint.
TRACKABLE_ACTIONS = {
    "copy": "copy_count",
    "screenshot": "screenshot_count",
    "download": "download_count",
}


def record_action(payload: dict[str, Any]) -> dict[str, Any]:
    """Bump one of the per-run action counters. Analytics only — every
    failure mode here is reported back as a plain response rather than
    raised, because a lost count must never surface as an error in the UI."""
    action = payload.get("action")
    column = TRACKABLE_ACTIONS.get(action)
    if column is None:
        raise ValidationError(f"Unknown action: {action!r}")

    try:
        run_id = int(payload.get("run_id"))
    except (TypeError, ValueError):
        raise ValidationError("run_id must be an integer") from None

    db_url = os.environ["NEON_DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                f"UPDATE survey_runs SET {column} = {column} + 1 WHERE id = %s",
                [run_id],
            )
            updated = cur.rowcount
        conn.commit()

    return {"ok": bool(updated)}


def ensure_survey_runs_table() -> None:
    db_url = os.environ["NEON_DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(SURVEY_RUNS_TABLE_SQL)
            cur.execute(SURVEY_RUNS_MIGRATE_SQL)
        conn.commit()


def log_survey_run(
    status: str,
    error_message: str | None,
    question: str | None,
    filters: dict[str, Any] | None,
    sample_size: int | None,
    matched_count: int | None,
    categories: Any | None,
    distribution: Any | None = None,
    key_takeaway: str | None = None,
    summary: str | None = None,
    category_model: str | None = None,
    persona_model: str | None = None,
    summary_model: str | None = None,
    cost: Any | None = None,
    timing: Any | None = None,
) -> int | None:
    """Best-effort analytics log — failures here must never break the
    user-facing response, so this is called from a broad try/except.
    Logs every attempt (success, validation_error, pipeline_error), not just
    successes, so post-launch issues are visible without relying on users
    to report them.

    Returns the new row's id, which the successful-run path hands back to
    the frontend as run_id so later /api/track calls can attribute copy /
    screenshot / download clicks to this specific run."""
    db_url = os.environ["NEON_DATABASE_URL"]
    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(
                """
                INSERT INTO survey_runs (
                    status, error_message, question, filters, sample_size,
                    matched_count, categories, distribution, key_takeaway,
                    summary, category_model, persona_model, summary_model,
                    cost, timing
                ) VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
                RETURNING id
                """,
                [
                    status,
                    error_message,
                    question,
                    json.dumps(filters) if filters is not None else None,
                    sample_size,
                    matched_count,
                    json.dumps(categories) if categories is not None else None,
                    json.dumps(distribution) if distribution is not None else None,
                    key_takeaway,
                    summary,
                    category_model,
                    persona_model,
                    summary_model,
                    json.dumps(cost) if cost is not None else None,
                    json.dumps(timing) if timing is not None else None,
                ],
            )
            row = cur.fetchone()
        conn.commit()
    return row[0] if row else None


def run_survey_pipeline(body: dict[str, Any]) -> dict[str, Any]:
    try:
        response = _run_survey_pipeline_inner(body)
    except ValidationError as exc:
        _log_failed_run("validation_error", str(exc))
        raise
    except Exception as exc:  # noqa: BLE001
        _log_failed_run("pipeline_error", str(exc))
        raise

    return response


# Per-thread scratch slot for the in-flight request's pipeline state, so a
# failure log can capture as much context as was actually available when it
# happened. Thread-local (not a plain global) because ThreadingHTTPServer
# runs each request on its own thread.
_thread_state = threading.local()


def _log_failed_run(status: str, error_message: str) -> None:
    question, filters, sample_size, matched_count, categories = (
        getattr(_thread_state, "question", None),
        getattr(_thread_state, "filters", None),
        getattr(_thread_state, "sample_size", None),
        getattr(_thread_state, "matched_count", None),
        getattr(_thread_state, "categories", None),
    )
    try:
        log_survey_run(
            status, error_message, question, filters, sample_size,
            matched_count, categories,
        )
    except Exception as log_exc:  # noqa: BLE001
        print(f"[survey8b_api] survey_runs failure logging failed (non-fatal): {log_exc}")


def _run_survey_pipeline_inner(body: dict[str, Any]) -> dict[str, Any]:
    _thread_state.question = _thread_state.filters = _thread_state.sample_size = None
    _thread_state.matched_count = _thread_state.categories = None

    question = str(body.get("question", "")).strip()
    _thread_state.question = question
    if not question:
        raise ValidationError("Question is required.")

    try:
        sample_size = int(body.get("sample_size"))
    except (TypeError, ValueError):
        raise ValidationError("sample_size must be an integer.")
    _thread_state.sample_size = sample_size
    if sample_size <= 0:
        raise ValidationError("sample_size must be greater than 0.")

    filters = _parse_filters(body)
    _thread_state.filters = filters

    # Sampling silently restricts to working-age adults (see
    # s2.WORKING_AGE_BRACKETS) on top of whatever the user picked. Applied
    # only to the query builders below, not to `filters` itself, so the
    # response/log/UI never show this as something the user chose or could
    # have chosen.
    sampling_filters = {**filters, "age_bracket": s2.WORKING_AGE_BRACKETS}

    db_url = os.environ["NEON_DATABASE_URL"]
    try:
        with psycopg.connect(db_url) as conn:
            with conn.cursor() as cur:
                count_sql, count_params = s2.build_count_query(sampling_filters)
                cur.execute(count_sql, count_params)
                matched_count = cur.fetchone()[0]
                _thread_state.matched_count = matched_count

                if sample_size > matched_count:
                    raise ValidationError(
                        f"Requested sample size ({sample_size}) exceeds the matched "
                        f"population ({matched_count}) for these filters. Reduce the "
                        "sample size or broaden the filters."
                    )

                sample_sql, sample_params = s2.build_sample_query(sampling_filters, sample_size)
                cur.execute(sample_sql, sample_params)
                rows = cur.fetchall()
    except ValueError as exc:
        # build_count_query/build_sample_query raise ValueError when no filter
        # is provided at all — a request-shape problem, not a server error.
        raise ValidationError(str(exc))

    sampled_rows = [dict(zip(s2.OUTPUT_COLUMNS, row)) for row in rows]

    openai_client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])
    # Lower temperature specifically for category derivation to reduce
    # run-to-run variance in the number/wording of categories for the same
    # question — persona Q&A and summarization keep their default temperature.
    category_call_fn = functools.partial(
        s2.call_openai, openai_client, s2.CATEGORY_MODEL_NAME, temperature=0.2
    )
    persona_call_fn = functools.partial(s2.call_openai, openai_client, s2.PERSONA_MODEL_NAME)
    summary_call_fn = functools.partial(
        s2.call_openrouter, os.environ["OPENROUTER_API_KEY"], s2.SUMMARY_MODEL
    )
    # Gemma intermittently returns valid JSON with garbage values instead of
    # text (observed in survey_runs: {"key_takeaway": -999}, {"key_takeaway":
    # -5, "summary": -100}, {}), and retrying the same model on the same input
    # tends to reproduce it rather than recover. gpt-4o-mini already drives the
    # category and persona steps reliably, so it's the fallback.
    summary_fallback_call_fn = functools.partial(
        s2.call_openai, openai_client, s2.CATEGORY_MODEL_NAME
    )

    category_start = time.perf_counter()
    categories, category_usage = s2.derive_categories(category_call_fn, question)
    category_elapsed = time.perf_counter() - category_start
    category_usage["elapsed_seconds"] = round(category_elapsed, 3)
    _thread_state.categories = categories

    persona_start = time.perf_counter()
    with ThreadPoolExecutor(max_workers=PERSONA_CONCURRENCY) as pool:
        raw_results = list(
            pool.map(
                lambda row: _run_one_persona(row, question, categories, persona_call_fn),
                sampled_rows,
            )
        )
    persona_elapsed = time.perf_counter() - persona_start
    results = [r for r in raw_results if r is not None]

    summary = s2.summarise(results, sampled_rows, filters, categories, category_usage)

    summary_start = time.perf_counter()
    summary_text = summary_usage = None
    summary_error: Exception | None = None
    summary_model_used = s2.SUMMARY_MODEL
    # (call_fn, model_name, attempts) — exhaust the primary model first, then
    # fall back to a different one. Retrying Gemma on the same input reproduces
    # its garbage-value failures rather than recovering from them, so a
    # same-model retry budget alone isn't enough.
    summary_plan = [
        (summary_call_fn, s2.SUMMARY_MODEL, 3),
        (summary_fallback_call_fn, s2.CATEGORY_MODEL_NAME, 2),
    ]
    for plan_call_fn, plan_model, plan_attempts in summary_plan:
        for _attempt in range(plan_attempts):
            try:
                candidate_text, candidate_usage = s2.generate_summary(
                    plan_call_fn, question, summary["distribution"], results, len(results)
                )
                # s2.generate_summary() only rejects empty strings; a non-empty
                # but useless response (e.g. "-0.0") still needs to be retried,
                # matching the persona-call and category-derivation resilience
                # already applied elsewhere in the pipeline.
                if _looks_degenerate(candidate_text["key_takeaway"]) or _looks_degenerate(
                    candidate_text["summary"]
                ):
                    raise ValueError(f"Degenerate summary response: {candidate_text!r}")
                summary_text, summary_usage = candidate_text, candidate_usage
                summary_model_used = plan_model
                summary_error = None
                break
            except Exception as exc:  # noqa: BLE001
                # ValueError = malformed/degenerate content; anything else is a
                # transport-level failure (timeout, HTTP error) from the
                # provider call, which already retries internally. Both are
                # worth moving on from rather than failing the whole request.
                summary_error = exc
        if summary_text is not None:
            break
    summary_elapsed = time.perf_counter() - summary_start

    if summary_text is None:
        print(f"[survey8b_api] summarization unavailable, returning distribution without AI summary: {summary_error}")
        summary_usage = {"input_tokens": 0, "output_tokens": 0, "total_tokens": 0, "cost_usd": 0.0}
        key_takeaway_value = (
            "AI-written summary unavailable for this run — the response "
            "distribution above is complete and accurate."
        )
        summary_value = ""
    else:
        key_takeaway_value = summary_text["key_takeaway"]
        summary_value = summary_text["summary"]
    summary_usage["elapsed_seconds"] = round(summary_elapsed, 3)

    summary["cost"]["summarization_cost_usd"] = summary_usage["cost_usd"]
    summary["cost"]["total_cost_usd"] = round(
        summary["cost"]["total_cost_usd"] + summary_usage["cost_usd"], 6
    )
    summary["cost"]["total_input_tokens"] += summary_usage["input_tokens"]
    summary["cost"]["total_output_tokens"] += summary_usage["output_tokens"]

    response = {
        "filters": filters,
        "question": question,
        "sample_size": len(results),
        "matched_count": matched_count,
        "categories": categories,
        "distribution": summary["distribution"],
        "key_takeaway": key_takeaway_value,
        "summary": summary_value,
        "results": [
            {
                "persona_id": r["persona_id"],
                "question": r["question"],
                "persona_description": r["persona_description"],
                "answer": r["answer"],
                "reasoning": r["reasoning"],
            }
            for r in results
        ],
        "category_model": s2.CATEGORY_MODEL_NAME,
        "persona_model": s2.PERSONA_MODEL_NAME,
        "summary_model": summary_model_used,
        "cost": summary["cost"],
        "timing": {
            "category_derivation_seconds": round(category_elapsed, 3),
            "persona_calls_total_seconds": round(persona_elapsed, 3),
            "summarization_seconds": round(summary_elapsed, 3),
            "total_seconds": round(category_elapsed + persona_elapsed + summary_elapsed, 3),
        },
    }

    try:
        response["run_id"] = log_survey_run(
            "success" if summary_text is not None else "success_no_summary",
            None if summary_text is not None else str(summary_error),
            response["question"], response["filters"],
            response["sample_size"], response["matched_count"], response["categories"],
            response["distribution"], response["key_takeaway"], response["summary"],
            response["category_model"], response["persona_model"], response["summary_model"],
            response["cost"], response["timing"],
        )
    except Exception as exc:  # noqa: BLE001
        print(f"[survey8b_api] survey_runs logging failed (non-fatal): {exc}")
        response["run_id"] = None

    return response


class Handler(BaseHTTPRequestHandler):
    def _cors_headers(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header("Access-Control-Allow-Methods", "POST, OPTIONS")
        self.send_header("Access-Control-Allow-Headers", "Content-Type")

    def do_OPTIONS(self) -> None:  # noqa: N802
        self.send_response(204)
        self._cors_headers()
        self.end_headers()

    def _send_json(self, status: int, payload: dict[str, Any]) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self._cors_headers()
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_POST(self) -> None:  # noqa: N802
        if self.path not in ("/api/facets", "/api/run-survey", "/api/track"):
            self._send_json(404, {"error": "Not found"})
            return

        try:
            length = int(self.headers.get("Content-Length", "0"))
            raw_body = self.rfile.read(length) if length else b"{}"
            payload = json.loads(raw_body or b"{}")
            if not isinstance(payload, dict):
                raise ValidationError("Request body must be a JSON object")

            if self.path == "/api/facets":
                db_url = os.environ["NEON_DATABASE_URL"]
                with psycopg.connect(db_url) as conn:
                    result = get_facets(conn, payload)
            elif self.path == "/api/track":
                result = record_action(payload)
            else:
                result = run_survey_pipeline(payload)

            self._send_json(200, result)
        except ValidationError as exc:
            self._send_json(400, {"error": str(exc)})
        except Exception as exc:  # noqa: BLE001
            self._send_json(500, {"error": str(exc)})

    def log_message(self, format: str, *args: Any) -> None:  # noqa: A002
        print(f"[survey8b_api] {self.address_string()} - {format % args}")


def main() -> None:
    load_local_env(Path(".env"))
    for var in ("NEON_DATABASE_URL", "OPENAI_API_KEY", "OPENROUTER_API_KEY"):
        if var not in os.environ:
            raise RuntimeError(f"{var} is not set")

    ensure_survey_runs_table()

    server = ThreadingHTTPServer(("0.0.0.0", PORT), Handler)
    print(f"Survey8B API listening on http://localhost:{PORT}")
    print("  POST /api/facets")
    print("  POST /api/run-survey")
    print("  POST /api/track")
    server.serve_forever()


if __name__ == "__main__":
    main()
