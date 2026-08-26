#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable, Iterator

import pyarrow.parquet as pq
import psycopg


DEFAULT_ARCHIVE_ROOT = Path(
    "/Users/believe/Documents/zz. vibe vault/5. survey8b - archive/matrAIx"
)
DEFAULT_SCHEMA_PATH = DEFAULT_ARCHIVE_ROOT / "persona_codes.schema.json"
DEFAULT_DATA_DIR = DEFAULT_ARCHIVE_ROOT / "survey8b_matraix_full" / "data"
DEFAULT_TABLE = "personas_matraix_subset"
DEFAULT_BATCH_SIZE = 4096

TARGET_FIELD_IDS = [
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

TABLE_COLUMNS = [
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


@dataclass(frozen=True)
class FieldSpec:
    field_id: str
    label: str
    index: int
    values: list[str]


def load_local_env(path: Path) -> None:
    if not path.exists():
        return
    for raw_line in path.read_text(encoding="utf-8").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        if key and key not in os.environ:
            os.environ[key] = value


def normalize_missing(value: object) -> str | None:
    if value is None:
        return None
    if not isinstance(value, str):
        return str(value)
    stripped = value.strip()
    if stripped.lower() in {"", "none", "null"}:
        return None
    return stripped


def bit_is_set(bitmap: bytes | None, field_index: int) -> bool:
    if bitmap is None:
        return False
    byte = bitmap[field_index // 8]
    return bool(byte & (1 << (field_index % 8)))


def decode_nibble(attr_bytes: bytes, field_index: int, values: list[str]) -> str | None:
    byte = attr_bytes[field_index // 2]
    code = byte & 0x0F if field_index % 2 == 0 else byte >> 4
    if code >= len(values):
        return None
    return values[code]


def resolve_field_specs(schema_path: Path) -> list[FieldSpec]:
    with schema_path.open("r", encoding="utf-8") as f:
        schema = json.load(f)

    columns = schema["columns"]
    specs: list[FieldSpec] = []
    for field_id in TARGET_FIELD_IDS:
        for index, column in enumerate(columns):
            if column.get("id") == field_id:
                specs.append(
                    FieldSpec(
                        field_id=field_id,
                        label=column.get("label", field_id),
                        index=index,
                        values=list(column["values"]),
                    )
                )
                break
        else:
            raise KeyError(f"Field id not found in schema: {field_id}")
    return specs


def row_to_record(
    record_id: int,
    attr_bytes: bytes,
    null_bitmap: bytes | None,
    overrides: list[dict] | None,
    specs: list[FieldSpec],
) -> tuple[object, ...]:
    override_map: dict[int, str | None] = {}
    if overrides:
        for item in overrides:
            if not isinstance(item, dict):
                continue
            field_index = item.get("field_index")
            if field_index is None:
                continue
            override_map[int(field_index)] = normalize_missing(item.get("value"))

    decoded: list[object] = [record_id]
    for spec in specs:
        if spec.index in override_map:
            decoded.append(override_map[spec.index])
            continue

        if bit_is_set(null_bitmap, spec.index):
            decoded.append(None)
            continue

        value = decode_nibble(attr_bytes, spec.index, spec.values)
        decoded.append(value)

    return tuple(decoded)


def parquet_paths(data_dir: Path) -> list[Path]:
    paths = sorted(data_dir.glob("persona-1m-*.parquet"))
    if not paths:
        raise FileNotFoundError(f"No parquet shards found in {data_dir}")
    return paths


def create_table_sql(table_name: str) -> str:
    return f"""
        CREATE TABLE {table_name} (
            id BIGINT PRIMARY KEY,
            region TEXT,
            age_bracket TEXT,
            demo_household_income TEXT,
            role_function TEXT,
            seniority TEXT,
            pref_early_vs_late TEXT,
            val_social_status TEXT,
            economic_motivation TEXT,
            risk_tolerance TEXT,
            att_subscription_services TEXT
        )
    """


def create_indexes(cur: psycopg.Cursor, table_name: str) -> None:
    for column in TABLE_COLUMNS[1:]:
        idx_name = f"{table_name}_{column}_idx"
        cur.execute(f"CREATE INDEX {idx_name} ON {table_name} ({column})")


def batch_iter(iterable: Iterable[tuple[object, ...]], size: int) -> Iterator[list[tuple[object, ...]]]:
    batch: list[tuple[object, ...]] = []
    for item in iterable:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def iter_records(shards: list[Path], specs: list[FieldSpec], batch_size: int) -> Iterator[tuple[object, ...]]:
    selected_cols = [
        "source_row_index",
        "attributes",
        "null_bitmap",
        "attribute_overrides",
    ]
    total_rows = 0
    global_row_id = 0
    for shard_index, shard_path in enumerate(shards, start=1):
        parquet_file = pq.ParquetFile(shard_path)
        shard_rows = 0
        for batch in parquet_file.iter_batches(batch_size=batch_size, columns=selected_cols):
            row_ids = batch.column(0).to_pylist()
            attrs_list = batch.column(1).to_pylist()
            nulls_list = batch.column(2).to_pylist()
            overrides_list = batch.column(3).to_pylist()

            for row_id, attrs, nulls, overrides in zip(
                row_ids, attrs_list, nulls_list, overrides_list
            ):
                global_row_id += 1
                record_id = global_row_id
                record = row_to_record(record_id, attrs, nulls, overrides, specs)
                if record is not None:
                    total_rows += 1
                    shard_rows += 1
                    yield record
        print(f"[shard {shard_index}/{len(shards)}] loaded {shard_rows} complete rows from {shard_path.name}")
    print(f"[decode] total complete rows yielded: {total_rows}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Decode the MatrAIx Persona 1M archive and load a filtered subset into Neon."
    )
    parser.add_argument("--schema-path", type=Path, default=DEFAULT_SCHEMA_PATH)
    parser.add_argument("--data-dir", type=Path, default=DEFAULT_DATA_DIR)
    parser.add_argument("--table", default=DEFAULT_TABLE)
    parser.add_argument("--batch-size", type=int, default=DEFAULT_BATCH_SIZE)
    parser.add_argument(
        "--db-url-env",
        default="NEON_DATABASE_URL",
        help="Environment variable containing the Postgres connection string.",
    )
    parser.add_argument(
        "--replace",
        action="store_true",
        help="Drop and recreate the target table before loading.",
    )
    args = parser.parse_args()

    load_local_env(Path(".env"))
    db_url = os.getenv(args.db_url_env)
    if not db_url:
        raise RuntimeError(f"{args.db_url_env} is not set")

    if not args.schema_path.exists():
        raise FileNotFoundError(f"Schema file not found: {args.schema_path}")
    if not args.data_dir.exists():
        raise FileNotFoundError(f"Data directory not found: {args.data_dir}")

    specs = resolve_field_specs(args.schema_path)
    shards = parquet_paths(args.data_dir)

    print("[setup] field mapping:")
    for spec in specs:
        print(f"  - {spec.field_id:28s} -> schema index {spec.index:4d} ({spec.label})")
    print(f"[setup] data shards: {len(shards)}")

    with psycopg.connect(db_url) as conn:
        conn.execute("SET synchronous_commit TO off")
        with conn.cursor() as cur:
            if args.replace:
                cur.execute(f"DROP TABLE IF EXISTS {args.table}")
            else:
                cur.execute("SELECT to_regclass(%s)", (args.table,))
                existing = cur.fetchone()[0]
                if existing is not None:
                    raise RuntimeError(
                        f"Refusing to overwrite existing table {args.table!r}. "
                        "Choose a new table name or pass --replace to reload it."
                    )
            cur.execute(create_table_sql(args.table))
        conn.commit()

        inserted = 0
        with conn.cursor() as cur:
            copy_sql = f"COPY {args.table} ({', '.join(TABLE_COLUMNS)}) FROM STDIN"
            with cur.copy(copy_sql) as copy:
                for batch_num, batch in enumerate(batch_iter(iter_records(shards, specs, args.batch_size), args.batch_size), start=1):
                    for row in batch:
                        copy.write_row(row)
                    inserted += len(batch)
                    if batch_num % 10 == 0:
                        print(f"[copy] inserted {inserted} complete rows so far")
        conn.commit()

        with conn.cursor() as cur:
            create_indexes(cur, args.table)
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"ANALYZE {args.table}")
        conn.commit()

        with conn.cursor() as cur:
            cur.execute(f"SELECT COUNT(*) FROM {args.table}")
            row_count = cur.fetchone()[0]
            cur.execute(f"SELECT pg_total_relation_size('{args.table}')")
            total_bytes = cur.fetchone()[0]
            cur.execute(f"SELECT COUNT(*) FROM {args.table} WHERE region = %s", ("North America",))
            north_america_count = cur.fetchone()[0]
            null_rates = {}
            for column in TABLE_COLUMNS[1:]:
                cur.execute(
                    f"SELECT COUNT(*) FILTER (WHERE {column} IS NOT NULL), COUNT(*) FILTER (WHERE {column} IS NULL) FROM {args.table}"
                )
                populated, missing = cur.fetchone()
                null_rates[column] = (populated, missing)
            cur.execute(
                f"SELECT region, COUNT(*) FROM {args.table} GROUP BY region ORDER BY COUNT(*) DESC, region ASC"
            )
            region_breakdown = cur.fetchall()
            cur.execute(
                f"EXPLAIN (ANALYZE, BUFFERS) SELECT COUNT(*) FROM {args.table} WHERE region = %s",
                ("North America",),
            )
            explain_region = "\n".join(r[0] for r in cur.fetchall())
            cur.execute(
                f"EXPLAIN (ANALYZE, BUFFERS) SELECT COUNT(*) FROM {args.table} WHERE seniority = %s",
                ("Founder",),
            )
            explain_founder = "\n".join(r[0] for r in cur.fetchall())

    total_mb = total_bytes / (1024 * 1024)
    print("[result] final row count:", row_count)
    print(f"[result] table size: {total_bytes} bytes ({total_mb:.2f} MB)")
    print("[result] per-field populated/null counts:")
    for column in TABLE_COLUMNS[1:]:
        populated, missing = null_rates[column]
        populated_pct = (populated / row_count * 100) if row_count else 0.0
        missing_pct = (missing / row_count * 100) if row_count else 0.0
        print(f"  - {column}: populated={populated} ({populated_pct:.2f}%), null={missing} ({missing_pct:.2f}%)")
    print("[result] region='North America' count:", north_america_count)
    print("[result] region breakdown:")
    for region, count in region_breakdown:
        print(f"  - {region}\t{count}")
    print("[result] EXPLAIN for region='North America':")
    print(explain_region)
    print("[result] EXPLAIN for seniority='Founder':")
    print(explain_founder)


if __name__ == "__main__":
    main()
