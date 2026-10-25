"""Bronze layer: land raw API payloads untouched, plus provenance.

Each row = one raw record serialized as JSON text, so the bronze file never has to
change when the upstream schema does. Parsing happens in silver.
"""
from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd


def to_bronze_frame(records: list[dict[str, Any]], source: str, query: str, key_fn) -> pd.DataFrame:
    fetched_at = datetime.now(UTC).isoformat(timespec="seconds")
    rows = []
    for rec in records:
        raw = json.dumps(rec, sort_keys=True, separators=(",", ":"))
        rows.append(
            {
                "record_key": key_fn(rec),
                "source": source,
                "query": query,
                "fetched_at": fetched_at,
                "content_sha256": hashlib.sha256(raw.encode()).hexdigest(),
                "raw_json": raw,
            }
        )
    df = pd.DataFrame(rows, columns=["record_key", "source", "query", "fetched_at", "content_sha256", "raw_json"])
    # A study can match several drug queries; keep one copy per (key, content).
    return df.drop_duplicates(subset=["record_key", "content_sha256"]).reset_index(drop=True)


def write_bronze(df: pd.DataFrame, out_dir: Path, name: str) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    stamp = datetime.now(UTC).strftime("%Y%m%dT%H%M%SZ")
    path = out_dir / f"{name}_{stamp}.parquet"
    df.to_parquet(path, index=False)
    return path
