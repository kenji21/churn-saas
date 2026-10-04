"""Cartes des modeles exportes (model cards) : data/modele_champion_<sha>.md et data/modele_vie_<sha>.md, a cote
des .joblib, comme la carte du gold a cote du parquet. Generees depuis le .joblib (jamais reecrit) : metriques
recalculees sur la validation, provenance lue dans MLflow et git. Usage prevu et limites : texte fixe ci-dessous."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import joblib
import mlflow
import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

from churn.build_log import code_diff_stat, commit_id
from churn.gold import PARQUET_DIR
from churn.train import _COLONNES_EXCLUES, MLFLOW_TRACKING_URI, _meilleur_seuil_f1, charger_split
from churn.train_vie import CIBLE_VIE, _metriques, charger_split_vie

_USAGE_CHAMPION = [
    "Aide à la décision pour les CSM et le support : signaler les comptes à risque de résiliation à l'échéance, "
    "pour prioriser les appels et le traitement des tickets.",
    "Ne remplace pas la décision humaine : une alerte déclenche un contact, jamais une action automatique sur le "
    "compte (tarif, accès, contrat).",
]
_LIMITES_CHAMPION = [
    "Plafond de performance lié à l'information disponible (courbe d'apprentissage plate dès 2 000 lignes), pas au "
    "modèle : une partie du churn dépend de facteurs absents des données (budget, concurrence, interlocuteur).",
    "Données en photo à un instant t, sans historique : pas de tendance d'usage.",
    "Validation de 1 000 clients : des écarts de +/- 0,02 entre modèles ne sont pas significatifs.",
    "Imputations à 0 de nb_integrations et delai_reponse_support_h discutables (cf data review).",
    "Biais d'intervention en exploitation : un client alerté puis retenu par le CSM ne doit pas être compté comme "
    "une erreur du modèle (groupe témoin nécessaire pour mesurer la performance réelle).",
]
_USAGE_VIE = [
    "Estimer la valeur vie client d'un compte, pour enrichir l'analyse métier : combinée à la probabilité de churn, "
    "elle donne la valeur à risque (P(churn) x valeur vie) et aide à prioriser les comptes à contacter.",
    "Jamais en entrée du modèle churn.",
]
_LIMITES_VIE = [
    "Cible très étalée (300 à 2 000 000 €) : le modèle classe bien les comptes (R² en log élevé) mais l'erreur "
    "absolue reste forte sur les très gros comptes (R² en euros plus faible).",
    "Valeur vie fournie calculée après coup (elle diverge de MRR x ancienneté surtout chez les clients partis) : la "
    "cible elle-même est imparfaite.",
    "Mêmes limites de données que le modèle churn : photo à un instant t, imputations à 0 discutables.",
]
_EXCLUSIONS_VIE = {
    "churn": "inconnu au moment de la prédiction, et sans apport mesuré",
    "sante_compte_fin_periode": "calculée en fin de période, après coup",
    CIBLE_VIE: "la cible elle-même",
    "client_id": "identifiant, pas une feature",
}


def _puces(lignes: list[str]) -> list[str]:
    return [f"- {ligne}" for ligne in lignes]


def _donnees_entree(colonnes: list[str], exclusions: dict[str, str]) -> list[str]:
    return [
        f"- Entrées ({len(colonnes)} colonnes) : " + ", ".join(f"`{c}`" for c in colonnes),
        "- Exclues : " + " ; ".join(f"`{c}` ({raison})" for c, raison in exclusions.items()),
    ]


def _provenance(artefact: dict[str, Any], nom_joblib: str) -> list[str]:
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    run = mlflow.get_run(artefact["run_mlflow"])
    experience = mlflow.get_experiment(run.info.experiment_id).name
    date_run = pd.Timestamp(run.info.start_time, unit="ms", tz="UTC").strftime("%Y-%m-%d %H:%M UTC")
    etat_code = "code non commité sous src/ au moment de la carte" if code_diff_stat() else "code commité"
    return [
        "## Provenance",
        "",
        f"- Artefact : `data/{nom_joblib}`",
        f"- Gold d'entraînement : `data/{artefact['gold']}` "
        f"(carte `data/{artefact['gold'].replace('.parquet', '.md')}`)",
        f"- Run MLflow : `{run.info.run_name}` ({artefact['run_mlflow']}), expérience `{experience}`, {date_run}",
        f"- Commit du code à la génération de la carte : `{commit_id()}` ({etat_code})",
    ]


def carte_champion(nom_joblib: str) -> str:
    artefact = joblib.load(PARQUET_DIR / nom_joblib)
    train, val, _ = charger_split(artefact["gold"])
    y_val = val["churn"]
    proba = artefact["modele"].predict_proba(val[artefact["colonnes"]])[:, 1]
    alerte = proba >= artefact["seuil"]
    detectes = int((alerte & y_val).sum())
    estimateur = artefact["modele"].steps[-1][1]

    lignes = [
        f"# Carte du modèle churn `{nom_joblib}`",
        "",
        "## Usage prévu",
        "",
        *_puces(_USAGE_CHAMPION),
        "",
        "## Modèle",
        "",
        f"- {artefact['nom']} : `{type(estimateur).__name__}`, sortie = probabilité de churn",
        f"- Seuil de décision : {artefact['seuil']:.3f} (maximise le F1 sur la validation), "
        "alerte si probabilité >= seuil",
        "",
        "## Données",
        "",
        f"- Train : {len(train)} clients, validation : {len(val)} clients (découpe 80/20 stratifiée sur churn, "
        f"taux de churn {y_val.mean():.1%})",
        *_donnees_entree(artefact["colonnes"], _COLONNES_EXCLUES),
        "",
        "## Performance (validation)",
        "",
        "| Métrique | Valeur |",
        "|---|---|",
        f"| PR-AUC | {average_precision_score(y_val, proba):.3f} (hasard = {y_val.mean():.3f}) |",
        f"| AUC ROC | {roc_auc_score(y_val, proba):.3f} |",
        f"| F1 au seuil | {_meilleur_seuil_f1(y_val, proba)[1]:.3f} |",
        f"| Churners détectés (rappel) | {detectes}/{int(y_val.sum())} ({detectes / y_val.sum():.1%}) |",
        f"| Alertes justifiées (précision) | {detectes}/{int(alerte.sum())} ({detectes / alerte.sum():.1%}) |",
        f"| Fausses alertes | {int(alerte.sum()) - detectes} |",
        "",
        "## Limites",
        "",
        *_puces(_LIMITES_CHAMPION),
        "",
        *_provenance(artefact, nom_joblib),
    ]
    return "\n".join(lignes) + "\n"


def carte_vie(nom_joblib: str) -> str:
    artefact = joblib.load(PARQUET_DIR / nom_joblib)
    train, val, _ = charger_split_vie(artefact["gold"])
    y_val = val[CIBLE_VIE]
    prediction = np.clip(artefact["modele"].predict(val[artefact["colonnes"]]), 0, None)
    references = {
        "Prédire la moyenne du train": _metriques(y_val, np.full(len(val), train[CIBLE_VIE].mean())),
        "Prédire la médiane du train": _metriques(y_val, np.full(len(val), train[CIBLE_VIE].median())),
    }
    estimateur = artefact["modele"]
    cible_en_log = hasattr(estimateur, "regressor")  # TransformedTargetRegressor
    regresseur = (estimateur.regressor if cible_en_log else estimateur).steps[-1][1]

    def ligne_tableau(nom: str, m: dict[str, float]) -> str:
        return f"| {nom} | {m['mae']:,.0f} | {m['rmse']:,.0f} | {m['r2']:.3f} | {m['r2_log']:.3f} |"

    lignes = [
        f"# Carte du modèle valeur vie client `{nom_joblib}`",
        "",
        "## Usage prévu",
        "",
        *_puces(_USAGE_VIE),
        "",
        "## Modèle",
        "",
        f"- {artefact['nom']} : `{type(regresseur).__name__}` (alpha={regresseur.alpha}), "
        + ("entraîné sur log(1 + valeur vie), prédictions reconverties en euros" if cible_en_log else "cible en euros"),
        "- Sortie : valeur vie client estimée, en euros (bornée à 0)",
        "",
        "## Données",
        "",
        f"- Train : {len(train)} clients, validation : {len(val)} clients (même découpe que le modèle churn), "
        "valeur vie jointe depuis le silver",
        f"- Valeur vie (train) : médiane {train[CIBLE_VIE].median():,.0f} €, moyenne {train[CIBLE_VIE].mean():,.0f} €, "
        f"de {train[CIBLE_VIE].min():,.0f} à {train[CIBLE_VIE].max():,.0f} €",
        *_donnees_entree(artefact["colonnes"], _EXCLUSIONS_VIE),
        "",
        "## Performance (validation)",
        "",
        "| Modèle | MAE (€) | RMSE (€) | R² (€) | R² (log) |",
        "|---|---|---|---|---|",
        ligne_tableau(f"**{artefact['nom']}**", _metriques(y_val, prediction)),
        *(ligne_tableau(nom, m) for nom, m in references.items()),
        "",
        "## Limites",
        "",
        *_puces(_LIMITES_VIE),
        "",
        *_provenance(artefact, nom_joblib),
    ]
    return "\n".join(lignes) + "\n"


def ecrire_carte(nom_joblib: str) -> Path:
    """Ecrit data/<nom>.md a cote de data/<nom>.joblib (modele_champion_* ou modele_vie_*)."""
    carte = carte_vie(nom_joblib) if nom_joblib.startswith("modele_vie_") else carte_champion(nom_joblib)
    chemin = PARQUET_DIR / nom_joblib.replace(".joblib", ".md")
    chemin.write_text(carte, encoding="utf-8")
    return chemin
