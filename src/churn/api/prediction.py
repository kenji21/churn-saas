"""Passage d'un client brut (valide par le schema) aux features du gold, puis a la prediction --
avec les fonctions de churn.regles, celles-la memes qui ont fabrique le gold d'entrainement."""

from __future__ import annotations

from typing import Any

import pandas as pd

from churn import regles
from churn.api.schemas import Client, ClientVie, ErreurChamp, Prediction, PredictionVie


class EntreeInvalide(Exception):
    """Erreur sur une donnee valide en soi mais incoherente avec le catalogue des plans."""

    def __init__(self, erreurs: list[ErreurChamp]) -> None:
        super().__init__(erreurs)
        self.erreurs = erreurs


def _valider_contre_catalogue(client: Client, catalogue: pd.DataFrame) -> None:
    plans = catalogue.set_index("plan_normalized")
    if client.plan not in plans.index:
        raise EntreeInvalide([ErreurChamp(champ="plan", message=f"plan inconnu, plans du catalogue : {sorted(plans.index)}")])
    incluses = int(plans.loc[client.plan, "fonctionnalites_incluses"])
    if client.fonctionnalites_utilisees > incluses:
        raise EntreeInvalide(
            [
                ErreurChamp(
                    champ="fonctionnalites_utilisees",
                    message=f"ne peut pas depasser les {incluses} fonctionnalites du plan {client.plan}",
                )
            ]
        )


def preparer_features(client: Client, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Une ligne au format du gold (memes colonnes, memes regles d'imputation et de calcul)."""
    _valider_contre_catalogue(client, catalogue)

    df = pd.DataFrame([client.model_dump()]).rename(columns={"plan": "plan_normalized"})
    for colonne in regles.PLAGES:
        if colonne in df:
            df[colonne] = pd.to_numeric(df[colonne])  # None -> NaN, impute ensuite comme a l'entrainement
    df["fonctionnalites_total"] = df["plan_normalized"].map(
        catalogue.set_index("plan_normalized")["fonctionnalites_incluses"]
    )
    df = regles.completer_valeurs_absentes(df)
    return regles.ajouter_features_derivees(df, catalogue)


def predire(client: Client, artefact: dict[str, Any]) -> Prediction:
    features = preparer_features(client, artefact["catalogue"])
    proba = float(artefact["modele"].predict_proba(features[artefact["colonnes"]])[0, 1])
    return Prediction(
        client_id=client.client_id,
        probabilite_churn=round(proba, 4),
        seuil=round(artefact["seuil"], 4),
        alerte=proba >= artefact["seuil"],
        modele=artefact["nom"],
        gold=artefact["gold"],
    )


def predire_vie(client: ClientVie, artefact: dict[str, Any]) -> PredictionVie:
    """Memes features que le churn (meme preparation), plus le MRR brut : le passage au log du MRR et de la
    cible est dans le pipeline du modele exporte, qui renvoie directement des euros."""
    features = preparer_features(client, artefact["catalogue"])
    features["revenu_mensuel_recurrent_eur"] = pd.to_numeric(features["revenu_mensuel_recurrent_eur"])
    valeur = max(0.0, float(artefact["modele"].predict(features[artefact["colonnes"]])[0]))
    return PredictionVie(
        client_id=client.client_id,
        valeur_vie_client_eur=round(valeur, 2),
        modele=artefact["nom"],
        gold=artefact["gold"],
    )
