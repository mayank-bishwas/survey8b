#!/usr/bin/env python3
from __future__ import annotations

import argparse
import functools
import json
import os
import random
import re
import time
from pathlib import Path
from typing import Any, Callable
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen

import psycopg
from openai import OpenAI, OpenAIError


TABLE_NAME = "personas_matraix_subset"
CATEGORY_MODEL_NAME = "gpt-4o-mini"
PERSONA_MODEL_NAME = "gpt-4o-mini"
SUMMARY_MODEL = "google/gemma-4-31b-it"
DEFAULT_SAMPLE_SIZE = 50

# Published per-token rates (USD) for models called via the direct-OpenAI path,
# used to compute exact per-call cost. OpenRouter reports its own authoritative
# `cost` per call instead (via `usage: {"include": true}`), so this table is
# only consulted on the OpenAI path.
OPENAI_MODEL_RATES_PER_TOKEN = {
    "gpt-5.5": {"input": 5.00 / 1_000_000, "output": 30.00 / 1_000_000},
    "gpt-4o-mini": {"input": 0.15 / 1_000_000, "output": 0.60 / 1_000_000},
}

# call_fn signature shared by both providers: (system_prompt, user_prompt) ->
# (raw_text, usage_dict) where usage_dict has input_tokens/output_tokens/
# total_tokens/cost_usd.
CallFn = Callable[[str, str], tuple[str, dict[str, Any]]]
DEFAULT_QUESTION = (
    "Would you be interested in using an AI-powered tool called Survey8b that lets "
    "you test content and keyword intent against simulated audience personas before "
    "writing content?"
)

FILTER_COLUMNS = [
    "region",
    "age_bracket",
    "demo_household_income",
    "role_function",
    "seniority",
    "pref_early_vs_late",
    "val_social_status",
    "economic_motivation",
    "risk_tolerance",
    "att_subscription_services",
]

OUTPUT_COLUMNS = [
    "id",
    "region",
    "age_bracket",
    "demo_household_income",
    "role_function",
    "seniority",
    "pref_early_vs_late",
    "val_social_status",
    "economic_motivation",
    "risk_tolerance",
    "att_subscription_services",
]

# Sampling is silently restricted to working-age adults, applied on every run
# regardless of user-selected filters - never exposed as a selectable filter
# and never mentioned in persona_sentence().
WORKING_AGE_BRACKETS = ["25-34", "35-44", "45-54"]


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


def build_count_query(filters: dict[str, list[str]]) -> tuple[str, list[str]]:
    clauses: list[str] = []
    params: list[str] = []
    for column in FILTER_COLUMNS:
        values = filters.get(column)
        if not values:
            continue
        placeholders = ", ".join(["%s"] * len(values))
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)
    if not clauses:
        raise ValueError("At least one filter value must be provided.")
    sql = f"SELECT COUNT(*) FROM {TABLE_NAME} WHERE " + " AND ".join(clauses)
    return sql, params


def build_sample_query(filters: dict[str, list[str]], sample_size: int) -> tuple[str, list[Any]]:
    clauses: list[str] = []
    params: list[Any] = []
    for column in FILTER_COLUMNS:
        values = filters.get(column)
        if not values:
            continue
        placeholders = ", ".join(["%s"] * len(values))
        clauses.append(f"{column} IN ({placeholders})")
        params.extend(values)
    if not clauses:
        raise ValueError("At least one filter value must be provided.")
    sql = (
        f"SELECT {', '.join(OUTPUT_COLUMNS)} "
        f"FROM {TABLE_NAME} WHERE " + " AND ".join(clauses) + " ORDER BY random() LIMIT %s"
    )
    params.append(sample_size)
    return sql, params


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey8B Step 2: sample personas, simulate survey responses, and summarize."
    )
    for column in FILTER_COLUMNS:
        parser.add_argument(
            f"--{column.replace('_', '-')}",
            nargs="+",
            required=False,
            help=f"One or more values for {column}; matched with IN (...).",
        )
    parser.add_argument(
        "--sample-size",
        type=int,
        default=DEFAULT_SAMPLE_SIZE,
        help="Number of personas to sample from the matching slice.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=None,
        help="Optional random seed for repeatable sampling.",
    )
    parser.add_argument(
        "--db-url-env",
        default="NEON_DATABASE_URL",
        help="Environment variable containing the Postgres connection string.",
    )
    parser.add_argument(
        "--openai-key-env",
        default="OPENAI_API_KEY",
        help="Environment variable containing the OpenAI API key.",
    )
    parser.add_argument(
        "--openrouter-key-env",
        default="OPENROUTER_API_KEY",
        help="Environment variable containing the OpenRouter API key.",
    )
    parser.add_argument(
        "--category-provider",
        choices=["openai", "openrouter"],
        default="openai",
        help="Which API to call --category-model through.",
    )
    parser.add_argument(
        "--persona-provider",
        choices=["openai", "openrouter"],
        default="openai",
        help="Which API to call --persona-model through.",
    )
    parser.add_argument(
        "--output",
        default="survey8b_step2_results.json",
        help="Path for the per-persona JSON results.",
    )
    parser.add_argument(
        "--summary-output",
        default="survey8b_step2_summary.json",
        help="Path for the aggregate summary JSON.",
    )
    parser.add_argument(
        "--category-model",
        default=CATEGORY_MODEL_NAME,
        help="Model identifier for the selected provider, used for category derivation.",
    )
    parser.add_argument(
        "--persona-model",
        default=PERSONA_MODEL_NAME,
        help="Model identifier for the selected provider, used for the per-persona Q&A calls.",
    )
    parser.add_argument(
        "--summary-model",
        default=SUMMARY_MODEL,
        help=(
            "OpenRouter model identifier used for the final key-takeaway/summary "
            "generation call. Always called via OpenRouter, independent of --provider."
        ),
    )
    parser.add_argument(
        "--question",
        default=DEFAULT_QUESTION,
        help="Survey question to ask each persona.",
    )
    return parser.parse_args()


def collect_filters(args: argparse.Namespace) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    for column in FILTER_COLUMNS:
        value = getattr(args, column)
        if value:
            filters[column] = list(value)
    return filters


ADOPTION_PHRASE = {
    "bleeding edge": "you're on the bleeding edge, trying new things before almost anyone else",
    "early adopter": "you're an early adopter who jumps on promising new products quickly",
    "mainstream": "you adopt new products once they've become mainstream",
    "late adopter": "you're a late adopter, waiting until something's well-proven before trying it",
    "laggard": "you're a laggard, sticking with what you know until you have no choice",
}

STATUS_PHRASE = {
    "core value": "what others think of your choices matters a great deal to you",
    "important": "you care meaningfully about how your choices are perceived by others",
    "moderate": "you care somewhat about status, but it's not a major driver",
    "minor": "status barely factors into your decisions",
    "irrelevant": "you genuinely don't care what others think of your choices",
}

SPENDING_PHRASE = {
    "cost-sensitive": "you're cost-sensitive and always looking for the cheapest workable option",
    "value-driven": "you're value-driven, weighing price against what you actually get",
    "premium-seeking": "you're premium-seeking, willing to pay more for quality or prestige",
    "indifferent": "price is largely irrelevant to how you decide",
}

RISK_PHRASE = {
    "risk-averse": "you're risk-averse, preferring safe, proven choices",
    "cautious": "you're cautious about risk without being fully averse to it",
    "balanced": "you weigh risk and reward in a fairly balanced way",
    "risk-tolerant": "you're risk-tolerant and comfortable with some uncertainty",
    "risk-seeking": "you're risk-seeking, drawn to uncertain or high-upside bets",
}

SUBSCRIPTION_PHRASE = {
    "enthusiast": "you're an enthusiastic fan of subscription-based products",
    "positive": "you're generally positive about subscription services",
    "neutral": "you feel neutral about subscription services, neither for nor against",
    "skeptical": "you're skeptical of subscription services",
    "opposed": "you're opposed to subscription-based purchasing models",
}

TRAIT_PHRASE_MAPS = (
    ("pref_early_vs_late", ADOPTION_PHRASE),
    ("val_social_status", STATUS_PHRASE),
    ("economic_motivation", SPENDING_PHRASE),
    ("risk_tolerance", RISK_PHRASE),
    ("att_subscription_services", SUBSCRIPTION_PHRASE),
)


def persona_sentence(row: dict[str, Any]) -> str:
    role = row.get("role_function")
    region = row.get("region")

    sentence = f"You are a {role} professional" if role else "You are a professional"
    if region:
        sentence += f" in {region}"
    sentence += "."

    trait_clauses: list[str] = []
    for column, phrase_map in TRAIT_PHRASE_MAPS:
        value = row.get(column)
        if not value:
            continue
        phrase = phrase_map.get(value.strip().lower())
        if phrase:
            trait_clauses.append(phrase)

    if trait_clauses:
        traits_sentence = trait_clauses[0][0].upper() + trait_clauses[0][1:]
        if len(trait_clauses) > 1:
            traits_sentence += "; " + "; ".join(trait_clauses[1:])
        sentence += f" {traits_sentence}."

    return sentence


def category_derivation_prompt(question: str) -> tuple[str, str]:
    system_prompt = (
        "You are designing the response scale for one survey question that will be asked "
        "to many simulated survey respondents, each answering independently. Read the "
        "question and produce a short, closed list of mutually exclusive answer categories "
        "that together cover the realistic range of honest positions someone could hold.\n\n"
        "Before choosing how many categories to use, first classify the shape of the "
        "question:\n"
        "- SPECTRUM: the question asks about a degree, intensity, or amount (e.g. how "
        "important, how likely, how satisfied). Use ordered categories spanning the range "
        "(e.g. Very / Somewhat / Not very / Not at all).\n"
        "- SEQUENCE / TRADE-OFF: the question asks which of several strategies, orderings, "
        "or approaches someone would take (e.g. do X before Y, or after). Enumerate the "
        "real strategies explicitly — do not collapse them into two options just because "
        "the question is phrased as 'A or B'.\n"
        "- GENUINELY BINARY: the question has no honest middle ground and no realistic third "
        "position (e.g. a plain yes/no fact or a strict either/or with no overlap possible).\n\n"
        "CRITICAL — tell real positions apart from escape hatches.\n"
        "- NEVER offer a non-committal catch-all: 'It depends', 'Depends on the context', "
        "'Maybe', 'Unsure', 'Neutral', 'No preference', 'Neither', 'Both are equally "
        "good'. These are refusals to answer. They are the safest choice for every "
        "respondent, so nearly all of them pick it and the survey returns no signal.\n"
        "- DO offer a distinct strategy that combines or reorders the named options — "
        "e.g. 'Build both at the same time', 'Alternate between them' — whenever that is "
        "a real approach someone would deliberately pursue and defend. That is a "
        "position, not a refusal, and for a trade-off question it is usually one of the "
        "genuine answers. Judge by intent: a deliberate strategy belongs; a way of "
        "dodging the choice does not.\n"
        "- For a SPECTRUM question, use an even number of points with no dead-centre "
        "neutral, so respondents must fall on one side.\n"
        "Respondents must lean; the survey is a forced choice.\n\n"
        "Default to 3-5 categories. Only use exactly 2 if the question is unambiguously "
        "GENUINELY BINARY per the definition above. "
        "LENGTH LIMIT — every category label must be 2-4 words and no more than 30 "
        "characters including spaces. This is a hard limit, not a guideline: these "
        "labels are displayed next to a percentage in a fixed-width UI column, and a "
        "longer label gets truncated and can crop the number itself. Prefer the short, "
        "punchy phrase over the precise one — trim qualifiers rather than exceed the "
        "limit. Word each category in the terms of the actual question — do not default "
        "to generic scale words like 'yes'/'no'/'agree'/'disagree' unless the question "
        "truly is a plain yes/no question. Categories must be mutually exclusive and "
        "collectively exhaustive enough that any honest respondent could pick exactly "
        'one. Return valid JSON only, matching the schema: {"categories": ["...", "..."]}.'
    )
    user_prompt = f"Survey question: {question}\nReturn the JSON category list."
    return system_prompt, user_prompt


def response_schema(categories: list[str]) -> dict[str, Any]:
    return {
        "type": "object",
        "properties": {
            "category": {"type": "string", "enum": categories},
            "reasoning": {"type": "string"},
        },
        "required": ["category", "reasoning"],
        "additionalProperties": False,
    }


def normalize_category(value: Any, categories: list[str]) -> str:
    text = str(value).strip().lower()
    for category in categories:
        if text == category.strip().lower():
            return category
    for category in categories:
        category_lower = category.strip().lower()
        if text in category_lower or category_lower in text:
            return category
    return ""


def normalize_reasoning(value: Any) -> str:
    return str(value).strip()


def strip_json_fence(text: str) -> str:
    """Strip a markdown code fence around JSON, if present.

    Some models (observed with Claude Haiku via OpenRouter) wrap JSON output
    in ```json ... ``` even when explicitly asked for JSON-only, unlike GPT
    models which return bare JSON under response_format=json_object.
    """
    stripped = text.strip()
    if stripped.startswith("```"):
        stripped = stripped.lstrip("`")
        if stripped.lower().startswith("json"):
            stripped = stripped[4:]
        stripped = stripped.strip()
        if stripped.endswith("```"):
            stripped = stripped[: -3]
        stripped = stripped.strip()
    return stripped


def compute_cost(model: str, input_tokens: int, output_tokens: int) -> float:
    rates = OPENAI_MODEL_RATES_PER_TOKEN.get(model)
    if rates is None:
        raise ValueError(
            f"No published rate on file for OpenAI model {model!r}; "
            f"known models: {sorted(OPENAI_MODEL_RATES_PER_TOKEN)}"
        )
    return input_tokens * rates["input"] + output_tokens * rates["output"]


def call_openai(
    client: OpenAI,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = 3,
    temperature: float | None = None,
) -> tuple[str, dict[str, Any]]:
    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            kwargs: dict[str, Any] = {}
            if temperature is not None:
                kwargs["temperature"] = temperature
            response = client.chat.completions.create(
                model=model,
                max_completion_tokens=16384,
                response_format={"type": "json_object"},
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt},
                ],
                **kwargs,
            )

            message = response.choices[0].message
            raw_text = strip_json_fence(message.content or "")
            usage_obj = response.usage
            input_tokens = usage_obj.prompt_tokens if usage_obj else 0
            output_tokens = usage_obj.completion_tokens if usage_obj else 0

            if not raw_text.strip():
                raise ValueError(
                    "Empty model content: finish_reason="
                    f"{response.choices[0].finish_reason}, usage={usage_obj}"
                )

            cost_usd = compute_cost(model, input_tokens, output_tokens)
            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": input_tokens + output_tokens,
                "cost_usd": round(cost_usd, 6),
            }
            return raw_text, usage
        except (OpenAIError, ValueError) as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(2 ** attempt)

    raise RuntimeError("Model call failed") from last_error


def call_openrouter(
    api_key: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    retries: int = 2,
) -> tuple[str, dict[str, Any]]:
    payload = {
        "model": model,
        "max_tokens": 16384,
        "response_format": {"type": "json_object"},
        # Ask OpenRouter to report actual per-call cost in usage, rather than
        # us computing it from a rate table (rates vary per model/provider).
        "usage": {"include": True},
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
    }

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            request = Request(
                "https://openrouter.ai/api/v1/chat/completions",
                data=json.dumps(payload).encode("utf-8"),
                headers={
                    "Authorization": f"Bearer {api_key}",
                    "Content-Type": "application/json",
                    "Accept": "application/json",
                    "HTTP-Referer": "http://localhost",
                    "X-Title": "survey8b",
                },
            )
            # Gemma summarization latency has been observed to range roughly
            # 10-228s in testing — 120s was clipping the slow end and forcing
            # a retry (and eventually a hard failure) on runs that were just
            # slow, not actually broken. 280s gives real headroom above the
            # observed max.
            with urlopen(request, timeout=280) as response:
                data = json.load(response)

            message = data["choices"][0]["message"]
            raw_text = strip_json_fence(message.get("content") or "")
            usage_obj = data.get("usage") or {}
            input_tokens = usage_obj.get("prompt_tokens", 0)
            output_tokens = usage_obj.get("completion_tokens", 0)

            if not raw_text.strip():
                raise ValueError(
                    "Empty model content: "
                    + json.dumps(
                        {
                            "message_keys": sorted(message.keys()),
                            "finish_reason": data["choices"][0].get("finish_reason"),
                            "usage": usage_obj,
                        },
                        ensure_ascii=False,
                    )
                )

            usage = {
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "total_tokens": usage_obj.get("total_tokens", input_tokens + output_tokens),
                "cost_usd": round(usage_obj.get("cost", 0.0) or 0.0, 6),
            }
            return raw_text, usage
        except (HTTPError, URLError, TimeoutError, ValueError) as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            # Free-tier models share an upstream rate-limit pool that can take
            # a while to clear, so back off further than the OpenAI path.
            time.sleep(min(30, 3 * (2 ** attempt)))

    raise RuntimeError("Model call failed") from last_error


CATEGORY_LABEL_MAX_CHARS = 30


def _shorten_category(label: str, max_chars: int = CATEGORY_LABEL_MAX_CHARS) -> str:
    """Last-resort trim to a word boundary, only used if the model still
    exceeds the limit after retries — the result card's legend column is
    fixed-width, so an oversized label crops the percentage next to it."""
    if len(label) <= max_chars:
        return label
    truncated = label[:max_chars].rsplit(" ", 1)[0].rstrip(",;:-")
    return truncated or label[:max_chars]


def derive_categories(call_fn: CallFn, question: str) -> tuple[list[str], dict[str, Any]]:
    system_prompt, user_prompt = category_derivation_prompt(question)

    last_error: Exception | None = None
    for _attempt in range(3):
        try:
            raw_text, usage = call_fn(system_prompt, user_prompt)
            parsed = json.loads(raw_text.strip())
            categories = [str(c).strip() for c in parsed.get("categories", []) if str(c).strip()]
            if len(categories) < 2:
                raise ValueError(f"Category derivation returned too few categories: {categories!r}")
            too_long = [c for c in categories if len(c) > CATEGORY_LABEL_MAX_CHARS]
            if too_long:
                raise ValueError(
                    f"Category label(s) exceed {CATEGORY_LABEL_MAX_CHARS} chars: {too_long!r}"
                )
            return categories, usage
        except (ValueError, json.JSONDecodeError) as exc:
            last_error = exc

    # The model didn't comply after 3 tries. Rather than fail the whole run
    # over a cosmetic length limit, use its last attempt's category list and
    # force-trim any oversized label — a guarantee instead of a hope.
    if "categories" in locals() and len(categories) >= 2:
        return [_shorten_category(c) for c in categories], usage
    raise ValueError(f"Category derivation failed after 3 attempts: {last_error}")


def build_response_prompt(persona_text: str, question: str, categories: list[str]) -> tuple[str, str]:
    category_list = " / ".join(categories)
    alternatives_note = ", ".join(categories)
    system_prompt = (
        "You are role-playing as the exact person described by the user, answering a "
        "survey question in character. Choose exactly one of these response categories, "
        f"worded exactly as given: {category_list}. Then, in 2-4 sentences and in your "
        "own voice as this persona, explain specifically why you chose that category "
        f"over the other options ({alternatives_note}) — what about your actual "
        "situation, priorities, or traits ruled the alternatives out for you. Return "
        "valid JSON only, matching the schema: "
        '{"category": "<one of the exact category labels above>", '
        '"reasoning": "<2-4 sentences>"}.'
    )
    user_prompt = f"{persona_text}\n\nSurvey question: {question}\n\nRespond with JSON matching the schema."
    return system_prompt, user_prompt


def call_model_for_persona(
    call_fn: CallFn,
    persona_text: str,
    question: str,
    categories: list[str],
) -> tuple[dict[str, Any], str, dict[str, Any]]:
    system_prompt, user_prompt = build_response_prompt(persona_text, question, categories)
    raw_text, usage = call_fn(system_prompt, user_prompt)

    stripped = raw_text.strip()
    parsed = json.loads(stripped)
    category = normalize_category(parsed.get("category"), categories)
    reasoning = normalize_reasoning(parsed.get("reasoning"))
    if not category:
        raise ValueError(
            "Invalid category field: "
            + json.dumps(
                {"raw_category": parsed.get("category"), "categories": categories, "raw_excerpt": stripped[:400]},
                ensure_ascii=False,
            )
        )
    if not reasoning:
        raise ValueError("Missing reasoning text")
    return {"category": category, "reasoning": reasoning}, raw_text, usage


def build_summary_tokens(distribution: list[dict[str, Any]], sample_size: int) -> dict[str, str]:
    """Rank-based placeholder tokens the model can reference instead of writing
    numbers itself. Keeping the vocabulary small and fixed (top/second only,
    regardless of how many categories exist) is what makes reliable model
    compliance realistic — see generate_summary()'s token-usage validation.
    """
    ranked = sorted(distribution, key=lambda d: d["pct"], reverse=True)
    tokens = {"sample_size": str(sample_size)}
    for rank_name, entry in zip(("top", "second"), ranked):
        tokens[f"{rank_name}_category"] = entry["category"]
        tokens[f"{rank_name}_pct"] = f"{entry['pct']:g}%"
    return tokens


def summary_generation_prompt(
    question: str,
    distribution: list[dict[str, Any]],
    reasoning_items: list[dict[str, str]],
    tokens: dict[str, str],
) -> tuple[str, str]:
    token_names = sorted(tokens.keys())
    token_list = ", ".join(f"{{{name}}}" for name in token_names)
    system_prompt = (
        "You are analyzing the results of a simulated survey. You'll be given the "
        "question, the closed set of response categories with their counts and "
        "percentages, and every individual respondent's category choice plus their "
        "own reasoning text. Write two things, both grounded in the actual reasoning "
        "text — not just the percentages: a 'key_takeaway' (10-20 words, one sentence, "
        "the single most actionable insight) and a 'summary' (around 100 words) that "
        "references specific real patterns, tensions, or recurring themes you see "
        "across the reasoning texts.\n\n"
        "Do not write out any specific number, percentage, or category name "
        "yourself anywhere a statistic belongs. Instead, insert one of these exact "
        f"placeholder tokens (spelled exactly as given, including the curly braces): "
        f"{token_list}. Each token will be substituted with the real value after you "
        "respond, so write natural sentences around them — e.g. \"{top_category} "
        "leads at {top_pct}\" is correct; writing the number or category name "
        "directly is not. You must use {top_pct} at least once, in either field. "
        "Do not invent any token name beyond the ones listed above.\n\n"
        'Return valid JSON only, matching the schema: {"key_takeaway": "...", "summary": "..."}.'
    )
    distribution_lines = "\n".join(f"- {d['category']}: {d['count']} ({d['pct']}%)" for d in distribution)
    reasoning_lines = "\n".join(
        f"{i}. [{item['category']}] {item['reasoning']}" for i, item in enumerate(reasoning_items, start=1)
    )
    user_prompt = (
        f"Survey question: {question}\n\n"
        f"Response distribution:\n{distribution_lines}\n\n"
        f"Individual respondent reasoning:\n{reasoning_lines}\n\n"
        "Respond with JSON matching the schema."
    )
    return system_prompt, user_prompt


_TOKEN_PATTERN = re.compile(r"\{([a-z_]+)\}")


def _apply_summary_tokens(text: str, tokens: dict[str, str]) -> str:
    used = set(_TOKEN_PATTERN.findall(text))
    unknown = used - set(tokens.keys())
    if unknown:
        raise ValueError(f"Response used unknown placeholder token(s): {sorted(unknown)}")
    for name, value in tokens.items():
        text = text.replace(f"{{{name}}}", value)
    return text


def generate_summary(
    call_fn: CallFn,
    question: str,
    distribution: list[dict[str, Any]],
    results: list[dict[str, Any]],
    sample_size: int,
) -> tuple[dict[str, str], dict[str, Any]]:
    reasoning_items = [
        {"category": item["response"]["category"], "reasoning": item["response"]["reasoning"]}
        for item in results
    ]
    tokens = build_summary_tokens(distribution, sample_size)
    system_prompt, user_prompt = summary_generation_prompt(question, distribution, reasoning_items, tokens)
    raw_text, usage = call_fn(system_prompt, user_prompt)
    parsed = json.loads(raw_text.strip())
    raw_key_takeaway = str(parsed.get("key_takeaway", "")).strip()
    raw_summary_text = str(parsed.get("summary", "")).strip()
    if not raw_key_takeaway or not raw_summary_text:
        raise ValueError(f"Incomplete summary response: {parsed!r}")
    if "{top_pct}" not in (raw_key_takeaway + " " + raw_summary_text):
        raise ValueError(f"Response never used required {{top_pct}} token: {parsed!r}")
    key_takeaway = _apply_summary_tokens(raw_key_takeaway, tokens)
    summary_text = _apply_summary_tokens(raw_summary_text, tokens)
    return {"key_takeaway": key_takeaway, "summary": summary_text}, usage


def summarise(
    results: list[dict[str, Any]],
    sampled_rows: list[dict[str, Any]],
    filters: dict[str, list[str]],
    categories: list[str],
    category_derivation_usage: dict[str, Any],
) -> dict[str, Any]:
    total = len(results)
    counts = {category: 0 for category in categories}
    for item in results:
        category = item["response"].get("category")
        if category in counts:
            counts[category] += 1
    distribution = sorted(
        (
            {
                "category": category,
                "count": counts[category],
                "pct": round((counts[category] / total * 100.0) if total else 0.0, 1),
            }
            for category in categories
        ),
        key=lambda d: d["pct"],
        reverse=True,
    )

    persona_input_tokens = sum(item["usage"]["input_tokens"] for item in results)
    persona_output_tokens = sum(item["usage"]["output_tokens"] for item in results)
    persona_cost = sum(item["usage"]["cost_usd"] for item in results)
    total_cost = persona_cost + category_derivation_usage["cost_usd"]

    return {
        "filters": filters,
        "categories": categories,
        "sample_size": total,
        "distribution": distribution,
        "sampled_persona_ids": [row["id"] for row in sampled_rows],
        "cost": {
            "category_derivation_cost_usd": category_derivation_usage["cost_usd"],
            "persona_calls_cost_usd": round(persona_cost, 6),
            "total_cost_usd": round(total_cost, 6),
            "total_input_tokens": category_derivation_usage["input_tokens"] + persona_input_tokens,
            "total_output_tokens": category_derivation_usage["output_tokens"] + persona_output_tokens,
        },
        "notes": (
            "Deterministic summary only: response distribution across the derived "
            "categories, plus sampled persona ids and exact cost. Use the per-persona "
            "JSON for the reasoning details."
        ),
    }


def main() -> None:
    args = parse_args()
    load_local_env(Path(".env"))

    db_url = os.getenv(args.db_url_env)
    if not db_url:
        raise RuntimeError(f"{args.db_url_env} is not set")
    needs_openai = args.category_provider == "openai" or args.persona_provider == "openai"
    needs_openrouter = args.category_provider == "openrouter" or args.persona_provider == "openrouter"

    client = None
    if needs_openai:
        openai_api_key = os.getenv(args.openai_key_env)
        if not openai_api_key:
            raise RuntimeError(f"{args.openai_key_env} is not set")
        client = OpenAI(api_key=openai_api_key)

    # The summarization call is always routed through OpenRouter to
    # google/gemma-4-31b-it, independent of --category-provider/--persona-provider,
    # so the OpenRouter key is required regardless.
    openrouter_api_key = os.getenv(args.openrouter_key_env)
    if not openrouter_api_key:
        raise RuntimeError(
            f"{args.openrouter_key_env} is not set (required for --summary-model, "
            "which is always called via OpenRouter, and for any --category-provider/"
            "--persona-provider set to openrouter)."
        )

    def build_call_fn(provider: str, model: str) -> CallFn:
        if provider == "openai":
            return functools.partial(call_openai, client, model)
        return functools.partial(call_openrouter, openrouter_api_key, model)

    category_call_fn: CallFn = build_call_fn(args.category_provider, args.category_model)
    persona_call_fn: CallFn = build_call_fn(args.persona_provider, args.persona_model)
    summary_call_fn: CallFn = functools.partial(call_openrouter, openrouter_api_key, args.summary_model)

    filters = collect_filters(args)
    count_sql, count_params = build_count_query(filters)
    sample_sql, sample_params = build_sample_query(filters, args.sample_size)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(count_sql, count_params)
            match_count = cur.fetchone()[0]
            if match_count < args.sample_size:
                raise RuntimeError(
                    f"Requested sample size {args.sample_size} exceeds the match count {match_count}."
                )

            cur.execute(sample_sql, sample_params)
            rows = cur.fetchall()

    if args.seed is not None:
        random.seed(args.seed)

    sampled_rows = [dict(zip(OUTPUT_COLUMNS, row)) for row in rows]
    if len(sampled_rows) > args.sample_size:
        sampled_rows = random.sample(sampled_rows, args.sample_size)

    category_derivation_start = time.perf_counter()
    categories, category_derivation_usage = derive_categories(category_call_fn, args.question)
    category_derivation_elapsed = time.perf_counter() - category_derivation_start
    category_derivation_usage["elapsed_seconds"] = round(category_derivation_elapsed, 3)
    running_total_cost = category_derivation_usage["cost_usd"]
    print(f"Derived response categories: {categories}")
    print(
        f"  category derivation | in={category_derivation_usage['input_tokens']} "
        f"out={category_derivation_usage['output_tokens']} "
        f"cost=${category_derivation_usage['cost_usd']:.6f} "
        f"time={category_derivation_elapsed:.2f}s "
        f"running total=${running_total_cost:.6f}"
    )

    results: list[dict[str, Any]] = []
    persona_calls_start = time.perf_counter()
    for index, row in enumerate(sampled_rows, start=1):
        persona_text = persona_sentence(row)
        # call_openrouter/call_openai already retry transport-level failures;
        # this retries schema-level failures (e.g. a weaker model returning
        # a malformed category/reasoning field) so one bad response doesn't
        # abort the whole batch.
        parse_error: Exception | None = None
        call_start = time.perf_counter()
        for parse_attempt in range(3):
            try:
                parsed, raw_text, usage = call_model_for_persona(
                    persona_call_fn, persona_text, args.question, categories
                )
                parse_error = None
                break
            except ValueError as exc:
                parse_error = exc
        call_elapsed = time.perf_counter() - call_start
        if parse_error is not None:
            print(f"{index}/{len(sampled_rows)} SKIPPED (malformed response 3x): {parse_error}")
            continue
        usage["elapsed_seconds"] = round(call_elapsed, 3)
        running_total_cost += usage["cost_usd"]
        results.append(
            {
                "index": index,
                "persona_id": row["id"],
                "persona_description": persona_text,
                "question": args.question,
                "model": args.persona_model,
                "response": parsed,
                "raw_response": raw_text,
                "usage": usage,
            }
        )
        print(
            f"{index}/{len(sampled_rows)} {parsed.get('category')} | "
            f"in={usage['input_tokens']} out={usage['output_tokens']} "
            f"cost=${usage['cost_usd']:.6f} time={call_elapsed:.2f}s "
            f"running total=${running_total_cost:.6f}"
        )
    persona_calls_elapsed = time.perf_counter() - persona_calls_start

    summary = summarise(results, sampled_rows, filters, categories, category_derivation_usage)

    # Persist the (expensive) category + persona work now, before the
    # summarization call, so a flaky summarization response doesn't discard
    # results that already cost real API credits to produce.
    Path(args.output).write_text(
        json.dumps(results, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )
    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    summary_call_start = time.perf_counter()
    summary_parse_error: Exception | None = None
    summary_text: dict[str, str] | None = None
    summary_usage: dict[str, Any] | None = None
    for summary_attempt in range(3):
        try:
            summary_text, summary_usage = generate_summary(
                summary_call_fn, args.question, summary["distribution"], results, len(results)
            )
            summary_parse_error = None
            break
        except ValueError as exc:
            summary_parse_error = exc
    summary_call_elapsed = time.perf_counter() - summary_call_start
    if summary_parse_error is not None:
        print(f"  summarization SKIPPED (malformed response 3x): {summary_parse_error}")
        summary["key_takeaway"] = None
        summary["summary"] = None
        summary["category_model"] = args.category_model
        summary["persona_model"] = args.persona_model
        summary["summary_model"] = args.summary_model
        summary["summary_usage"] = None
        summary["timing"] = {
            "category_derivation_seconds": round(category_derivation_elapsed, 3),
            "persona_calls_total_seconds": round(persona_calls_elapsed, 3),
            "persona_calls_avg_seconds": round(persona_calls_elapsed / len(results), 3) if results else 0.0,
            "summarization_seconds": round(summary_call_elapsed, 3),
            "total_seconds": round(
                category_derivation_elapsed + persona_calls_elapsed + summary_call_elapsed, 3
            ),
        }
        Path(args.summary_output).write_text(
            json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
            encoding="utf-8",
        )
        print(json.dumps(summary, indent=2, ensure_ascii=False))
        return
    summary_usage["elapsed_seconds"] = round(summary_call_elapsed, 3)
    running_total_cost += summary_usage["cost_usd"]
    print(
        f"  summarization ({args.summary_model}) | in={summary_usage['input_tokens']} "
        f"out={summary_usage['output_tokens']} cost=${summary_usage['cost_usd']:.6f} "
        f"time={summary_call_elapsed:.2f}s running total=${running_total_cost:.6f}"
    )
    print(f"  key_takeaway: {summary_text['key_takeaway']}")
    print(f"  summary: {summary_text['summary']}")

    summary["key_takeaway"] = summary_text["key_takeaway"]
    summary["summary"] = summary_text["summary"]
    summary["category_model"] = args.category_model
    summary["persona_model"] = args.persona_model
    summary["summary_model"] = args.summary_model
    summary["summary_usage"] = summary_usage
    summary["cost"]["summarization_cost_usd"] = summary_usage["cost_usd"]
    summary["cost"]["total_cost_usd"] = round(
        summary["cost"]["total_cost_usd"] + summary_usage["cost_usd"], 6
    )
    summary["cost"]["total_input_tokens"] += summary_usage["input_tokens"]
    summary["cost"]["total_output_tokens"] += summary_usage["output_tokens"]

    summary["timing"] = {
        "category_derivation_seconds": round(category_derivation_elapsed, 3),
        "persona_calls_total_seconds": round(persona_calls_elapsed, 3),
        "persona_calls_avg_seconds": round(persona_calls_elapsed / len(results), 3) if results else 0.0,
        "summarization_seconds": round(summary_call_elapsed, 3),
        "total_seconds": round(
            category_derivation_elapsed + persona_calls_elapsed + summary_call_elapsed, 3
        ),
    }

    Path(args.summary_output).write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n",
        encoding="utf-8",
    )

    print(json.dumps(summary, indent=2, ensure_ascii=False))


if __name__ == "__main__":
    main()
