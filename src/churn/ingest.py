from __future__ import annotations

from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from churn.build_log import commit_for_path, log_build
from churn.config import get_engine

SUJET_DIR = Path(__file__).resolve().parents[2] / "sujet"

RAW_SOURCES = {
    "raw_plan_catalogue": SUJET_DIR / "catalogue_plans.csv",
    "raw_churn": SUJET_DIR / "churn_saas_complet.csv",
}


def _read_csv_as_text(csv_path: Path) -> pd.DataFrame:
    df = pd.read_csv(csv_path, dtype=str, keep_default_na=False, encoding="utf-8-sig")
    df.insert(0, "source_row_number", range(1, len(df) + 1))
    df["ingested_at"] = datetime.now(timezone.utc)
    return df


def ingest_raw_table(table_name: str, csv_path: Path, engine: Engine) -> int:
    df = _read_csv_as_text(csv_path)
    df.to_sql(table_name, engine, if_exists="replace", index=False)
    return len(df)


def ingest_bronze(engine: Engine | None = None) -> dict[str, int]:
    engine = engine or get_engine()
    stats = {
        table: ingest_raw_table(table, csv_path, engine)
        for table, csv_path in RAW_SOURCES.items()
    }
    print(log_build(engine, layer="bronze", stats=stats, provenance=commit_for_path("sujet")))
    return stats
