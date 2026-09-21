from __future__ import annotations

import json
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd
from sqlalchemy import text
from sqlalchemy.engine import Engine

TARGET_TABLE = "build_log"

_REPO_ROOT = Path(__file__).resolve().parents[2]

_CREATE_TABLE_SQL = text(
    f"""
    CREATE TABLE IF NOT EXISTS {TARGET_TABLE} (
        id BIGSERIAL PRIMARY KEY,
        layer TEXT NOT NULL,
        built_at TIMESTAMPTZ NOT NULL,
        commit_id TEXT NOT NULL,
        code_diff_stat TEXT,
        stats JSONB,
        provenance TEXT
    )
    """
)

_INSERT_SQL = text(
    f"""
    INSERT INTO {TARGET_TABLE} (layer, built_at, commit_id, code_diff_stat, stats, provenance)
    VALUES (:layer, :built_at, :commit_id, :code_diff_stat, CAST(:stats AS jsonb), :provenance)
    RETURNING id
    """
)

_DELETE_DIRTY_SQL = text(
    f"""
    DELETE FROM {TARGET_TABLE}
    WHERE layer = :layer AND code_diff_stat IS NOT NULL AND code_diff_stat <> ''
    """
)

_LAST_DIFF_SQL = text(
    f"""
    SELECT code_diff_stat FROM {TARGET_TABLE}
    WHERE layer = :layer
    ORDER BY built_at DESC
    LIMIT 1
    """
)


def _git(*args: str) -> str:
    result = subprocess.run(["git", *args], cwd=_REPO_ROOT, capture_output=True, text=True, check=True)
    return result.stdout.strip()


def commit_id() -> str:
    """Hash du commit HEAD du dépôt (état du code au moment du build)."""
    return _git("rev-parse", "HEAD")


def code_diff_stat(path: str = "src/") -> str:
    """État du code non commité sur `path` : diff des fichiers trackés + liste des fichiers non trackés.

    `git diff --stat` seul ignore les fichiers jamais ajoutés (`??`), donc un nouveau module
    fraîchement créé n'apparaîtrait pas dans l'état du code au moment du build.
    """
    diff = _git("diff", "--stat", "--", path)
    non_suivis = _git("ls-files", "--others", "--exclude-standard", "--", path)
    if non_suivis:
        diff = f"{diff}\nfichiers non suivis :\n{non_suivis}".strip()
    return diff


def commit_for_path(path: str) -> str:
    """Hash du dernier commit ayant modifié `path` (traçabilité d'une source de données versionnée)."""
    return _git("log", "-1", "--format=%H", "--", path)


def last_build(engine: Engine, layer: str) -> dict[str, object] | None:
    """Dernier build enregistré pour `layer` (id, built_at, commit_id) : base pour chaîner provenance et affichage."""
    try:
        log = pd.read_sql_table(TARGET_TABLE, engine)
    except ValueError:
        return None
    log = log[log["layer"] == layer]
    if log.empty:
        return None
    row = log.sort_values("built_at").iloc[-1]
    return {
        "id": int(row["id"]),
        "built_at": row["built_at"],
        "commit_id": row["commit_id"],
        "dirty": bool(row["code_diff_stat"]),
    }


def log_build(engine: Engine, layer: str, stats: dict[str, int | float], provenance: str | None) -> str:
    """Ajoute une ligne d'audit dans `build_log` (commit, diff non commité, stats, provenance), renvoie un message d'info."""
    diff = code_diff_stat()
    code = commit_id()
    with engine.begin() as conn:
        conn.execute(_CREATE_TABLE_SQL)
        row_id = conn.execute(
            _INSERT_SQL,
            {
                "layer": layer,
                "built_at": datetime.now(timezone.utc),
                "commit_id": code,
                "code_diff_stat": diff,
                "stats": json.dumps(stats),
                "provenance": provenance,
            },
        ).scalar_one()

    lignes = []
    if diff:
        lignes.append(
            f"⚠️ Warning : [{layer}] code non commité sous src/ : ce build n'est pas reproductible depuis git log, "
            "committez avant de figer ce résultat."
        )
    suffixe_dirty = "-dirty" if diff else ""
    lignes.append("")
    lignes.append(f"ingestion enregistrée dans `build_log` : id={row_id}, layer={layer}, commit={code[:8]}{suffixe_dirty}")
    return "\n".join(lignes)


def cleanup_dirty(engine: Engine, layer: str, tables: tuple[str, ...] = ()) -> dict[str, int | list[str]]:
    """Purge les entrées `build_log` dirty de `layer`. Si le *dernier* build était dirty, les données produites
    (`tables`) ne sont pas reproductibles depuis git non plus : elles sont supprimées avant la purge du log."""
    with engine.begin() as conn:
        dernier_diff = conn.execute(_LAST_DIFF_SQL, {"layer": layer}).scalar_one_or_none()
        tables_supprimees: list[str] = []
        if dernier_diff:
            for table in tables:
                conn.execute(text(f"DROP TABLE IF EXISTS {table}"))
                tables_supprimees.append(table)
        n_lignes = conn.execute(_DELETE_DIRTY_SQL, {"layer": layer}).rowcount
    return {"lignes_build_log_supprimees": n_lignes, "tables_supprimees": tables_supprimees}
