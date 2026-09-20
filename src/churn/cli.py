from __future__ import annotations

import typer

from churn.ingest import ingest_bronze

app = typer.Typer(add_completion=False)


@app.callback()
def main() -> None:
    """CLI du pipeline churn-saas."""


@app.command("ingest-bronze")
def ingest_bronze_command() -> None:
    """Charge sujet/*.csv dans les tables raw_* (colonnes texte) de PostgreSQL."""
    for table, n in ingest_bronze().items():
        typer.echo(f"{table:<20}: {n} lignes")


if __name__ == "__main__":
    app()
