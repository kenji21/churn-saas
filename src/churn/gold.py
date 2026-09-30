from __future__ import annotations

import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from churn.build_log import code_diff_stat, commit_id, last_build, log_build
from churn.config import get_engine

SOURCE_TABLE = "silver_churn"
SOURCE_TABLE_PLAN_CATALOGUE = "silver_plan_catalogue"
TARGET_TABLE = "gold_churn"

PARQUET_DIR = Path(__file__).resolve().parents[2] / "data"
PARQUET_SYMLINK = PARQUET_DIR / "gold_churn.parquet"

STATS_LABELS: dict[str, str] = {
    "lignes_avant": "Lignes dans silver_churn",
    "lignes_apres": "Lignes dans gold_churn",
    "colonnes_conservees": "Colonnes conservées (identifiant + features + cibles)",
    "colonnes_exclues": "Colonnes silver exclues du gold",
    "parquet_exporte": "Artefact parquet écrit (0 si code non commité)",
    "parquet_fichier": "Fichier parquet (nommé par commit, vide si non écrit)",
    "carte_fichier": "Carte du gold (.md, vide si non écrite)",
}

# Au-delà de ce nombre de valeurs distinctes, une colonne catégorielle est trop cardinale pour un
# comptage "NxValeur" lisible dans la carte (ex: client_id) -- on affiche juste son nombre de valeurs.
_SEUIL_CARDINALITE_CARTE = 20

# Features catégorielles retenues : lien avec le churn confirmé par chi²/Cramér's V dans le notebook
# (cellules 15, 25, 42, 45 pour plan_normalized/taille_entreprise/commentaire_csm/csat).
_CATEGORICAL_FEATURES = ("plan_normalized", "taille_entreprise", "commentaire_csm")

# Features numériques retenues : signal confirmé dans le notebook (boxplots/heatmaps par statut churn).
_NUMERIC_FEATURES = (
    "anciennete_mois",  # cellule 85 : plus le client est jeune, plus le risque de départ est élevé
    "taux_adoption_pct",  # cellule 73 : adoption faible -> churn
    "connexions_30j",  # cellule 60
    "heures_usage_30j",  # cellule 60
    "nb_integrations",  # cellule 90 : effet de "captivité" du client intégré
    "derniere_connexion_jours",  # cellule 61 : signal fort, clients partis inactifs avant la coupure
    "tickets_support_90j",  # cellule 52
    "delai_reponse_support_h",  # cellule 48
    "csat",  # cellule 46
    "retards_paiement_12m",  # cellule 79
    "sante_compte_fin_periode",  # cellule 79
)

_TARGET = "churn"
# Cible secondaire (valeur vie client) : "sortie" métier, jamais utilisée en entrée du modèle churn (cf notebook cellule 4).
_SECONDARY_TARGET = "revenu_mensuel_recurrent_eur"
_IDENTIFIER = "client_id"

# Colonnes silver écartées du gold (décisions à valider/challenger avec le métier avant la soutenance) :
# - pays, secteur : Cramér's V ~0 avec le churn (cellules 13, 26)
# - date_souscription_normalized, jour_souscription, plan (déjà hors silver) : redondants/bruit (cf README §4)
# - code_datacenter, groupe_experimentation, couleur_theme_interface (déjà hors silver) : aucun signal (cellule 58)
# - fonctionnalites_total, fonctionnalites_utilisees : remplacées par le ratio `taux_utilisation_fonctionnalites_pct`
#   (fonctionnalites_total est de toute façon déterminé 1:1 par plan_normalized, cf validation silver)
# - sieges_souscrits, utilisateurs_actifs : redondants avec taux_adoption_pct (leur ratio) et taille_entreprise/plan
#   (corrélation forte démontrée cellules 17/19/22)
# - valeur_vie_client_eur : risque de fuite -- la cellule 101 montre qu'elle diverge de revenu×ancienneté
#   précisément pour les clients partis, donc probablement influencée après-coup par le churn
#
# support_dedie_from_plan (feature engineering, jointure silver_plan_catalogue sur plan_normalized) :
# fonction déterministe de plan_normalized (business/enterprise -> Oui, starter/pro -> Non, cf notebook
# cellule 57), donc redondante une fois plan_normalized encodé -- conservée à la demande explicite du candidat.


def _taux_utilisation_fonctionnalites(df: pd.DataFrame) -> pd.Series:
    return (df["fonctionnalites_utilisees"] / df["fonctionnalites_total"] * 100).round(2)


def _resume_colonne(serie: pd.Series) -> str:
    """Résumé d'une colonne pour la carte gold : moyenne/min/max si numérique (hors booléen), sinon
    répartition "NxValeur" -- ou juste le nombre de valeurs distinctes au-delà de `_SEUIL_CARDINALITE_CARTE`
    (ex: identifiant à forte cardinalité, illisible en répartition complète)."""
    est_numerique = pd.api.types.is_numeric_dtype(serie) and not pd.api.types.is_bool_dtype(serie)
    if est_numerique:
        return f"moyenne={serie.mean():.2f}, min={serie.min()}, max={serie.max()}"
    effectifs = serie.value_counts(dropna=False)
    if len(effectifs) > _SEUIL_CARDINALITE_CARTE:
        return f"{serie.nunique(dropna=False)} valeurs distinctes"
    return ", ".join(f"{n}x {v}" for v, n in effectifs.items())


def _carte_gold(df: pd.DataFrame, nom_parquet: str, silver_build: dict[str, object]) -> str:
    """Carte du gold : commit de code (gold) + commit silver de base, listing des colonnes avec stats."""
    lignes = [
        f"# Carte du gold `{nom_parquet}`",
        "",
        f"- Lignes : {len(df)}",
        f"- Commit gold (code qui l'a généré) : `{commit_id()}`",
        f"- Commit silver (données de base) : `{silver_build['commit_id']}` "
        f"(build id={silver_build['id']}, built_at={silver_build['built_at']})",
        "",
        "## Colonnes",
        "",
    ]
    lignes += [f"- `{col}` : {_resume_colonne(df[col])}" for col in df.columns]
    return "\n".join(lignes) + "\n"


def _ouvrir_dans_vscode(path: Path) -> None:
    try:
        subprocess.run(["code", str(path)], check=False)
    except FileNotFoundError:
        pass


def export_parquet(df: pd.DataFrame, silver_build: dict[str, object]) -> tuple[str, str]:
    """Écrit data/gold_churn_<sha>.parquet + data/gold_churn_<sha>.md (carte du gold), nommés par
    commit, seulement si src/ est commité (garantit que l'artefact est reproductible depuis un commit
    git précis, cf `code_diff_stat`). Pointe `PARQUET_SYMLINK` (data/gold_churn.parquet) sur le parquet
    et ouvre la carte dans VS Code. Renvoie (nom_parquet, nom_carte), ou ("", "") si ignoré (code dirty)."""
    if code_diff_stat():
        return "", ""
    PARQUET_DIR.mkdir(parents=True, exist_ok=True)
    sha = commit_id()[:8]
    nom_fichier = f"gold_churn_{sha}.parquet"
    df.to_parquet(PARQUET_DIR / nom_fichier, index=False)

    if PARQUET_SYMLINK.is_symlink() or PARQUET_SYMLINK.exists():
        PARQUET_SYMLINK.unlink()
    PARQUET_SYMLINK.symlink_to(nom_fichier)

    nom_carte = f"gold_churn_{sha}.md"
    chemin_carte = PARQUET_DIR / nom_carte
    chemin_carte.write_text(_carte_gold(df, nom_fichier, silver_build), encoding="utf-8")
    _ouvrir_dans_vscode(chemin_carte)

    return nom_fichier, nom_carte


def label_stats(stats: dict[str, int | float | str]) -> dict[str, int | float | str]:
    """Remplace les clés techniques d'un dict de stats par leur libellé lisible (`STATS_LABELS`)."""
    return {STATS_LABELS.get(cle, cle): valeur for cle, valeur in stats.items()}


def format_stats(stats: dict[str, int | float | str]) -> str:
    """Formate un dict de stats en tableau aligné sur le libellé le plus long."""
    labels = label_stats(stats)
    largeur = max(len(cle) for cle in labels)
    return "\n".join(f"{cle:<{largeur}} : {valeur}" for cle, valeur in labels.items())


def build_gold(engine: Engine | None = None) -> tuple[dict[str, int | float | str], str]:
    """Charge silver_churn/silver_plan_catalogue, sélectionne/enrichit les features validées à l'exploration,
    écrit gold_churn (table PostgreSQL + artefact parquet).

    Refuse de construire le gold si le dernier build silver n'est pas reproductible (absent ou dirty) :
    le gold hériterait sinon d'une donnée de base qu'on ne peut pas retracer à un commit git précis."""
    engine = engine or get_engine()
    silver_build = last_build(engine, "silver")
    if silver_build is None:
        raise RuntimeError("Aucun build silver dans build_log -- lancez build-silver avant build-gold.")
    if silver_build["dirty"]:
        raise RuntimeError(
            f"Dernier build silver (id={silver_build['id']}, commit={silver_build['commit_id'][:8]}) "
            "non reproductible : code non commité sous src/ au moment du build. Committez, relancez "
            "build-silver depuis un état propre, puis build-gold."
        )

    silver = pd.read_sql_table(SOURCE_TABLE, engine)
    catalogue = pd.read_sql_table(SOURCE_TABLE_PLAN_CATALOGUE, engine)

    n_avant = len(silver)

    df = silver.merge(catalogue[["plan_normalized", "support_dedie"]], on="plan_normalized", how="left")
    df = df.rename(columns={"support_dedie": "support_dedie_from_plan"})
    df["taux_utilisation_fonctionnalites_pct"] = _taux_utilisation_fonctionnalites(df)

    colonnes = [
        _IDENTIFIER,
        *_CATEGORICAL_FEATURES,
        *_NUMERIC_FEATURES,
        "taux_utilisation_fonctionnalites_pct",
        "support_dedie_from_plan",
        _SECONDARY_TARGET,
        _TARGET,
    ]
    df = df[colonnes]

    colonnes_silver_exclues = sorted(set(silver.columns) - set(df.columns))

    df.to_sql(TARGET_TABLE, engine, if_exists="replace", index=False)
    parquet_fichier, carte_fichier = export_parquet(df, silver_build)

    stats = {
        "lignes_avant": n_avant,
        "lignes_apres": len(df),
        "colonnes_conservees": len(colonnes),
        "colonnes_exclues": len(colonnes_silver_exclues),
        "parquet_exporte": int(bool(parquet_fichier)),
        "parquet_fichier": parquet_fichier,
        "carte_fichier": carte_fichier,
    }
    log_message = log_build(engine, layer="gold", stats=stats, provenance=silver_build["commit_id"])
    log_message += (
        f"\nsilver de base : id={silver_build['id']}, built_at={silver_build['built_at']}, "
        f"commit={silver_build['commit_id'][:8]}"
    )
    if not parquet_fichier:
        log_message += (
            "\n⚠️ Warning : artefact parquet non écrit -- code non commité sous src/, "
            "l'artefact du dernier commit propre (s'il existe) reste inchangé."
        )
    return stats, log_message
