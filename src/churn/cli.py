from __future__ import annotations

import typer

from churn.build_log import cleanup_dirty
from churn.config import get_engine
from churn.gold import TARGET_TABLE as GOLD_TARGET_TABLE
from churn.gold import build_gold
from churn.gold import format_stats as format_gold_stats
from churn.ingest import RAW_SOURCES, ingest_bronze
from churn.silver import TARGET_TABLE, TARGET_TABLE_PLAN_CATALOGUE, build_silver, format_stats
from churn.train import (
    charger_split,
    train_random_forest_baseline,
    train_random_forest_ordinal_baseline,
    train_reg_log_baseline,
    train_xgboost_baseline,
    tuner_random_forest,
    tuner_xgboost,
)

app = typer.Typer(add_completion=False)

_LAYER_TABLES: dict[str, tuple[str, ...]] = {
    "bronze": tuple(RAW_SOURCES.keys()),
    "silver": (TARGET_TABLE, TARGET_TABLE_PLAN_CATALOGUE),
    "gold": (GOLD_TARGET_TABLE,),
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


@app.command("build-gold")
def build_gold_command() -> None:
    """Sélectionne/enrichit les features de silver_churn et écrit gold_churn (table + parquet)."""
    stats, log_message = build_gold()
    typer.echo(log_message)
    typer.echo(format_gold_stats(stats))


@app.command("reg-log-baseline")
def reg_log_baseline_command() -> None:
    """Baseline regression logistique sur le gold (split 80/20 stratifie), tracking MLflow (experiment churn_saas)."""
    train, val, gold_sha = charger_split()
    _, log_message = train_reg_log_baseline(train, val, gold_sha)
    typer.echo(log_message)


@app.command("random-forest-baseline")
def random_forest_baseline_command() -> None:
    """Baseline Random Forest sur le gold (split 80/20 stratifie), tracking MLflow (experiment churn_saas)."""
    train, val, gold_sha = charger_split()
    _, log_message = train_random_forest_baseline(train, val, gold_sha)
    typer.echo(log_message)


@app.command("random-forest-ordinal-baseline")
def random_forest_ordinal_baseline_command() -> None:
    """Random Forest avec OrdinalEncoder sur commentaire_csm (au lieu d'OneHot), tracking MLflow (experiment churn_saas)."""
    train, val, gold_sha = charger_split()
    _, log_message = train_random_forest_ordinal_baseline(train, val, gold_sha)
    typer.echo(log_message)


@app.command("random-forest-tune")
def random_forest_tune_command(n_trials: int = 50) -> None:
    """Recherche d'hyperparametres Optuna pour Random Forest (encodage de commentaire_csm inclus
    dans la recherche), tracking MLflow (experiment churn_saas, un run imbrique par essai)."""
    train, val, gold_sha = charger_split()
    _, log_message = tuner_random_forest(train, val, gold_sha, n_trials=n_trials)
    typer.echo(log_message)


@app.command("xgboost-baseline")
def xgboost_baseline_command() -> None:
    """Baseline XGBoost sur le gold (split 80/20 stratifie), tracking MLflow (experiment churn_saas)."""
    train, val, gold_sha = charger_split()
    _, log_message = train_xgboost_baseline(train, val, gold_sha)
    typer.echo(log_message)


@app.command("xgboost-tune")
def xgboost_tune_command(n_trials: int = 50) -> None:
    """Recherche d'hyperparametres Optuna pour XGBoost, tracking MLflow (experiment churn_saas, un
    run imbrique par essai)."""
    train, val, gold_sha = charger_split()
    _, log_message = tuner_xgboost(train, val, gold_sha, n_trials=n_trials)
    typer.echo(log_message)


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
