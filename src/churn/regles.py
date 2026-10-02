"""Regles de nettoyage, de validation et de fabrication des features, partagees entre le pipeline
(silver, gold) et l'API. Une seule definition : en exploitation, le modele recoit exactement les
memes transformations qu'a l'entrainement (sinon l'API derive en silence du gold)."""

from __future__ import annotations

import re

import pandas as pd

COMMENTAIRE_CSM_ABSENT = "n/a"
COMMENTAIRES_CSM_CONNUS = frozenset(
    {
        "Client insatisfait, risque de départ.",
        "Client très satisfait et actif.",
        "Compte dans la moyenne.",
        "Compte engagé, ambassadeur potentiel.",
        "Compte peu actif, relance nécessaire.",
        "Déploiement interne limité.",
        "Faible adoption des sièges.",
        "Multiples tickets ouverts, friction élevée.",
        "Mécontentement exprimé au support.",
        "RAS.",
        "Suivi standard.",
        "Support très sollicité récemment.",
        "Usage en forte baisse ce trimestre.",
    }
)

TAILLES_ENTREPRISE_CONNUES = frozenset({"TPE", "PME", "ETI", "GE"})

# Bornes (min, max) des colonnes numeriques, None = pas de borne de ce cote. sieges_souscrits >= 1 :
# taux_adoption_pct en est le denominateur.
PLAGES: dict[str, tuple[float | None, float | None]] = {
    "anciennete_mois": (0, None),
    "sieges_souscrits": (1, None),
    "utilisateurs_actifs": (0, None),
    "taux_adoption_pct": (0, 100),
    "connexions_30j": (0, None),
    "heures_usage_30j": (0, None),
    "fonctionnalites_utilisees": (0, None),
    "nb_integrations": (0, None),
    "derniere_connexion_jours": (0, None),
    "tickets_support_90j": (0, None),
    "delai_reponse_support_h": (0, None),
    "csat": (1, 5),
    "retards_paiement_12m": (0, None),
    "sante_compte_fin_periode": (0, 100),
}

# Valeurs retenues quand la donnee est absente (cf silver). A revoir : la data review montre que ces
# clients ressemblent a la moyenne, pas aux vrais 0 (nb_integrations = 0 churne a 53%, contre 24.5%
# pour les absents) -- a corriger ici, pour que silver et API changent ensemble.
NB_INTEGRATIONS_SI_ABSENT = 0
DELAI_REPONSE_SUPPORT_H_SI_ABSENT = 0.0


_SUFFIXES_NUMERIQUES = re.compile(r"[€%hH]")


def nombre_depuis_texte(series: pd.Series) -> pd.Series:
    """Texte du CSV source vers nombre : "60.0%" -> 60.0, "15,6 h" -> 15.6, "280.62 €" -> 280.62, "" -> NA."""
    nettoye = series.str.strip().str.replace(_SUFFIXES_NUMERIQUES, "", regex=True).str.strip()
    nettoye = nettoye.str.replace(",", ".", regex=False)
    return pd.to_numeric(nettoye.replace("", pd.NA), errors="raise")


def normaliser_plan(valeur: str) -> str:
    """ " Pro " -> "pro", "Grand Compte" -> "grand_compte" (cle de jointure avec le catalogue)."""
    return re.sub(r"\s+", "_", valeur.strip().lower())


def normaliser_taille_entreprise(valeur: str) -> str:
    return valeur.strip().upper()


def normaliser_commentaire_csm(valeur: str | None) -> str:
    return (valeur or "").strip() or COMMENTAIRE_CSM_ABSENT


def completer_valeurs_absentes(df: pd.DataFrame) -> pd.DataFrame:
    """taux_adoption_pct absent : recalcule depuis utilisateurs_actifs / sieges_souscrits (la valeur
    fournie y est egale, ecart moyen 0.02 pt) ; nb_integrations et delai_reponse_support_h absents :
    valeurs par defaut ci-dessus."""
    df = df.copy()
    taux_calcule = df["utilisateurs_actifs"].astype("float64") / df["sieges_souscrits"].astype("float64") * 100
    df["taux_adoption_pct"] = df["taux_adoption_pct"].fillna(taux_calcule)
    df["nb_integrations"] = df["nb_integrations"].fillna(NB_INTEGRATIONS_SI_ABSENT)
    df["delai_reponse_support_h"] = df["delai_reponse_support_h"].fillna(DELAI_REPONSE_SUPPORT_H_SI_ABSENT)
    return df


def taux_utilisation_fonctionnalites(df: pd.DataFrame) -> pd.Series:
    return (df["fonctionnalites_utilisees"] / df["fonctionnalites_total"] * 100).round(2)


def delai_support_x_non_adoption(df: pd.DataFrame) -> pd.Series:
    """Interaction proposee en cellule 80 du notebook : un delai support long pese plus lourd quand
    l'adoption est faible. (100 - adoption) pour que les deux termes aillent dans le sens du churn."""
    return (df["delai_reponse_support_h"] * (100 - df["taux_adoption_pct"])).round(2)


def ajouter_features_derivees(df: pd.DataFrame, catalogue: pd.DataFrame) -> pd.DataFrame:
    """Features du gold calculees a partir des colonnes du silver et du catalogue des plans
    (plan_normalized, support_dedie). df doit contenir fonctionnalites_total."""
    df = df.merge(catalogue[["plan_normalized", "support_dedie"]], on="plan_normalized", how="left")
    df = df.rename(columns={"support_dedie": "support_dedie_from_plan"})
    df["taux_utilisation_fonctionnalites_pct"] = taux_utilisation_fonctionnalites(df)
    df["delai_support_x_non_adoption"] = delai_support_x_non_adoption(df)
    return df
