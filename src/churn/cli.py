from __future__ import annotations

import typer

from churn.build_log import cleanup_dirty
from churn.config import get_engine
from churn.ingest import RAW_SOURCES, ingest_bronze
from churn.silver import TARGET_TABLE, TARGET_TABLE_PLAN_CATALOGUE, build_silver, format_stats

app = typer.Typer(add_completion=False)

_LAYER_TABLES: dict[str, tuple[str, ...]] = {
    "bronze": tuple(RAW_SOURCES.keys()),
    "silver": (TARGET_TABLE, TARGET_TABLE_PLAN_CATALOGUE),
}


@app.callback()
def main() -> None:
    """CLI du pipeline churn-saas."""


@app.command("ingest-bronze")
def ingest_bronze_command() -> None:
    """Charge sujet/*.csv dans les tables raw_* (colonnes texte) de PostgreSQL."""
    for table, n in ingest_bronze().items():
        typer.echo(f"{table:<20}: {n} lignes")


@app.command("build-silver")
def build_silver_command() -> None:
    """Nettoie/type raw_churn et écrit silver_churn en base."""
    stats, log_message = build_silver()
    typer.echo(log_message)
    typer.echo(format_stats(stats))


@app.command("dirty-cleanup")
def dirty_cleanup_command() -> None:
    """Pour chaque couche (bronze, silver, ...), si son dernier build était dirty, supprime ses tables et purge build_log."""
    engine = get_engine()
    for layer, tables in _LAYER_TABLES.items():
        resultat = cleanup_dirty(engine, layer, tables=tables)
        typer.echo(f"[{layer}] entrées build_log dirty supprimées : {resultat['lignes_build_log_supprimees']}")
        if resultat["tables_supprimees"]:
            typer.echo(f"[{layer}] tables supprimées (dernier build dirty) : {', '.join(resultat['tables_supprimees'])}")
        else:
            typer.echo(f"[{layer}] dernier build propre : tables conservées")


if __name__ == "__main__":
    app()
