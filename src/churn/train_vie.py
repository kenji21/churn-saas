"""Cible secondaire : regression de la valeur vie client (valeur_vie_client_eur). Jamais en entree du
modele churn ; ici c'est la cible, et churn n'est jamais en entree (inconnu au moment de la prediction).

Seule la mecanique commune est ici (donnees, entrainement + log MLflow, comparaison) : les modeles eux-memes
(pretraitement, Ridge, passage au log) sont construits dans les cellules du notebook, pour rester lisibles."""

from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
from sklearn.base import RegressorMixin
from sklearn.metrics import mean_absolute_error, r2_score, root_mean_squared_error

from churn.config import get_engine
from churn.train import MLFLOW_TRACKING_URI, charger_split

MLFLOW_EXPERIMENT_VIE = "churn_vie"
CIBLE_VIE = "valeur_vie_client_eur"

_METRIQUES = ("mae", "rmse", "r2", "r2_log")


def charger_split_vie(fichier_parquet: str | None = None) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Meme decoupe train / validation que le modele churn (charger_split : memes clients de validation pour
    les deux modeles), plus la valeur vie client, exclue du gold (fuite pour le churn) et jointe depuis le
    silver par client_id."""
    train, val, gold_sha = charger_split(fichier_parquet)
    valeur_vie = pd.read_sql_table("silver_churn", get_engine(), columns=["client_id", CIBLE_VIE])
    return train.merge(valeur_vie, on="client_id"), val.merge(valeur_vie, on="client_id"), gold_sha


def _metriques(y_val: pd.Series, prediction: np.ndarray) -> dict[str, float]:
    """MAE / RMSE / R2 en euros (demandes par l'enonce), plus le R2 sur log(1 + valeur) : la cible s'etale de
    300 a 2 000 000 euros, le R2 en euros est domine par les tres gros comptes."""
    return {
        "mae": float(mean_absolute_error(y_val, prediction)),
        "rmse": float(root_mean_squared_error(y_val, prediction)),
        "r2": float(r2_score(y_val, prediction)),
        "r2_log": float(r2_score(np.log1p(y_val), np.log1p(prediction))),
    }


def entrainer_vie(
    nom_run: str,
    modele: RegressorMixin,
    colonnes: list[str],
    train: pd.DataFrame,
    val: pd.DataFrame,
    gold_sha: str,
) -> tuple[dict[str, float | str], str]:
    """Entraine sur train[colonnes], evalue sur la validation, loggue le run dans l'experience MLflow churn_vie
    (params, metriques, modele, log.txt). Run nomme <nom_run>_<gold_sha>."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT_VIE)

    with mlflow.start_run(run_name=f"{nom_run}_{gold_sha}") as run:
        mlflow.set_tag("gold_commit", gold_sha)
        mlflow.log_param("gold_parquet", f"gold_churn_{gold_sha}.parquet")
        mlflow.log_param("colonnes", ",".join(colonnes))
        # Hyperparametres simples (alpha du Ridge, strategie du Dummy...), pas les objets imbriques
        mlflow.log_params({k: v for k, v in modele.get_params().items() if np.isscalar(v)})

        modele.fit(train[colonnes], train[CIBLE_VIE])
        prediction = np.clip(modele.predict(val[colonnes]), 0, None)  # une valeur vie n'est jamais negative
        metriques = _metriques(val[CIBLE_VIE], prediction)

        mlflow.log_metrics(metriques)
        mlflow.sklearn.log_model(modele, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

        log_message = "\n".join(
            [
                f"gold     : {gold_sha} (gold_churn_{gold_sha}.parquet), cible : {CIBLE_VIE}",
                f"MLflow   : run {nom_run}_{gold_sha} ({run.info.run_id}), experience {MLFLOW_EXPERIMENT_VIE}",
                f"colonnes : {json.dumps(colonnes)}",
                f"MAE      : {metriques['mae']:,.0f} euros",
                f"RMSE     : {metriques['rmse']:,.0f} euros",
                f"R2       : {metriques['r2']:.3f} (en euros) / {metriques['r2_log']:.3f} (en log)",
            ]
        )
        mlflow.log_text(log_message, "log.txt")

    return {"run_id": run.info.run_id, "gold_commit": gold_sha, **metriques}, log_message


def comparer_vie(gold_sha: str, runs: dict[str, str]) -> pd.DataFrame:
    """Tableau MAE / RMSE / R2 lu dans MLflow (dernier run de chaque prefixe pour ce gold), pas re-saisi."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    lignes = []
    for libelle, prefixe_run in runs.items():
        trouves = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT_VIE],
            filter_string=f"tags.mlflow.runName = '{prefixe_run}_{gold_sha}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if trouves.empty:
            continue
        run = trouves.iloc[0]
        lignes.append({"modèle": libelle, **{m: run[f"metrics.{m}"] for m in _METRIQUES}})

    tableau = pd.DataFrame(lignes).set_index("modèle")
    return tableau.rename(columns={"mae": "MAE (€)", "rmse": "RMSE (€)", "r2": "R² (€)", "r2_log": "R² (log)"})
