#!/usr/bin/env python3
from __future__ import annotations

import argparse
import os
from pathlib import Path
from typing import Iterable

import psycopg


TABLE_NAME = "personas_matraix_subset"
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


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Survey8B Step 1: count exact persona matches with multi-select filters."
    )
    for column in FILTER_COLUMNS:
        parser.add_argument(
            f"--{column.replace('_', '-')}",
            nargs="+",
            required=False,
            help=f"One or more values for {column}; matched with IN (...).",
        )
    parser.add_argument(
        "--db-url-env",
        default="NEON_DATABASE_URL",
        help="Environment variable containing the Postgres connection string.",
    )
    return parser.parse_args()


def collect_filters(args: argparse.Namespace) -> dict[str, list[str]]:
    filters: dict[str, list[str]] = {}
    for column in FILTER_COLUMNS:
        value = getattr(args, column)
        if value:
            filters[column] = list(value)
    return filters


def main() -> None:
    args = parse_args()
    load_local_env(Path(".env"))
    db_url = os.getenv(args.db_url_env)
    if not db_url:
        raise RuntimeError(f"{args.db_url_env} is not set")

    filters = collect_filters(args)
    sql, params = build_count_query(filters)

    with psycopg.connect(db_url) as conn:
        with conn.cursor() as cur:
            cur.execute(sql, params)
            count = cur.fetchone()[0]

    print(count)


if __name__ == "__main__":
    main()
