from __future__ import annotations

import re

import pandas as pd
from sqlalchemy.engine import Engine

from churn import regles
from churn.build_log import last_build, log_build
from churn.config import get_engine

SOURCE_TABLE = "raw_churn"
TARGET_TABLE = "silver_churn"
SOURCE_TABLE_PLAN_CATALOGUE = "raw_plan_catalogue"
TARGET_TABLE_PLAN_CATALOGUE = "silver_plan_catalogue"

STATS_LABELS: dict[str, str] = {
    "lignes_avant": "Lignes dans les données bronze (raw_churn)",
    "doublons_purs_valides": "Lignes dupliquées validées identiques sur toutes les colonnes avant dédoublonnage",
    "lignes_apres": "Lignes après dédoublonnage",
    "concordance_jour_souscription": "Concordance date_souscription_normalized / jour_souscription",
    "secteur_manquants_imputes": "secteur manquants imputés à 'n/a'",
    "taux_adoption_pct_imputes": "taux_adoption_pct manquants imputés depuis utilisateurs_actifs / sieges_souscrits",
    "taux_adoption_pct_valides": "Lignes validées sur la plage taux_adoption_pct [0, 100]",
    "nb_integrations_imputes": "nb_integrations manquants imputés à 0",
    "delai_reponse_support_h_imputes": "delai_reponse_support_h manquants imputés à 0",
    "delai_reponse_support_h_valides": "Lignes validées sur la plage delai_reponse_support_h >= 0",
    "fonctionnalites_utilisees_valides": "Lignes validées fonctionnalites_utilisees <= fonctionnalites_total",
    "fonctionnalites_total_valides": "Lignes validées fonctionnalites_total == catalogue",
    "plans_normalized_valides": "Plans distincts (plan_normalized) validés contre le catalogue",
    "csat_valides": "Lignes validées sur la plage csat [1, 5]",
    "sante_compte_valides": "Lignes validées sur la plage sante_compte_fin_periode [0, 100]",
    "commentaire_csm_manquants": "commentaire_csm manquants imputés à 'n/a'",
    "commentaire_csm_valides": "Lignes validées sur les 13 valeurs connues de commentaire_csm",
    "validation_occupation_sieges": "Taux de lignes avec utilisateurs_actifs <= sieges_souscrits",
    "catalogue_plans_lignes": "Lignes dans silver_plan_catalogue",
}

_TITLE_CASE_COLUMNS = ("secteur",)
_STRIP_ONLY_COLUMNS = ("pays",)
_DROP_COLUMNS = (
    "source_row_number",
    "ingested_at",
    "date_souscription",
    "jour_souscription",
    "plan",
    "couleur_theme_interface",
    "code_datacenter",
    "groupe_experimentation",
)

_CATALOGUE_INT_COLUMNS = (
    "prix_mensuel_par_siege_eur",
    "fonctionnalites_incluses",
    "sla_reponse_h",
    "quota_stockage_go",
)
_OUI_NON = {"Oui": True, "Non": False}

_INT_COLUMNS = (
    "anciennete_mois",
    "sieges_souscrits",
    "utilisateurs_actifs",
    "connexions_30j",
    "fonctionnalites_total",
    "fonctionnalites_utilisees",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "csat",
    "retards_paiement_12m",
    "sante_compte_fin_periode",
    "valeur_vie_client_eur",
)
_FLOAT_COLUMNS = ("taux_adoption_pct", "heures_usage_30j", "revenu_mensuel_recurrent_eur", "delai_reponse_support_h")

_JOURS_SEMAINE = ["lundi", "mardi", "mercredi", "jeudi", "vendredi", "samedi", "dimanche"]

_DATE_FORMATS: dict[re.Pattern[str], str] = {
    re.compile(r"^\d{4}-\d{2}-\d{2}$"): "%Y-%m-%d",
    re.compile(r"^\d{2}/\d{2}/\d{4}$"): "%d/%m/%Y",
    re.compile(r"^\d{2} [A-Za-z]{3} \d{4}$"): "%d %b %Y",
}


def _blank_to_na(series: pd.Series) -> pd.Series:
    return series.replace("", pd.NA)


def _snake_case(series: pd.Series) -> pd.Series:
    return _blank_to_na(series.str.strip()).map(regles.normaliser_plan, na_action="ignore")


def _to_bool_oui_non(series: pd.Series) -> pd.Series:
    cleaned = _blank_to_na(series.str.strip())
    valeurs = cleaned.dropna()
    inconnues = valeurs[~valeurs.isin(_OUI_NON)].unique()
    if len(inconnues):
        raise ValueError(f"Valeurs support_dedie hors Oui/Non : {inconnues.tolist()}")
    return cleaned.map(_OUI_NON).astype("boolean")


def _parse_dates(series: pd.Series) -> pd.Series:
    parsed = pd.Series(pd.NaT, index=series.index, dtype="datetime64[ns]")
    remaining = pd.Series(True, index=series.index)
    for pattern, fmt in _DATE_FORMATS.items():
        mask = remaining & series.str.match(pattern)
        parsed.loc[mask] = pd.to_datetime(series[mask], format=fmt)
        remaining &= ~mask
    if remaining.any():
        exemples = series[remaining].unique()[:5].tolist()
        raise ValueError(f"{int(remaining.sum())} dates de date_souscription hors motifs connus : {exemples}")
    return parsed.dt.date


def _concordance_jour_souscription(dates: pd.Series, jours: pd.Series) -> float:
    jours_attendus = dates.map(lambda d: _JOURS_SEMAINE[d.weekday()])
    return (jours_attendus == jours.str.strip().str.lower()).mean()


def _valider_plage(series: pd.Series, nom: str, minimum: int, maximum: int) -> None:
    valeurs = series.dropna()
    hors_plage = valeurs[(valeurs < minimum) | (valeurs > maximum)]
    if len(hors_plage):
        raise ValueError(f"{len(hors_plage)} valeurs de {nom} hors plage [{minimum}, {maximum}] : {hors_plage.unique().tolist()[:5]}")


def _valider_doublons_purs(raw: pd.DataFrame, colonnes_comparees: list[str]) -> int:
    """Vérifie que les `client_id` dupliqués le sont sur `colonnes_comparees`. Renvoie le nombre de lignes validées."""
    doublons = raw[raw.duplicated(subset="client_id", keep=False)]
    if doublons.empty:
        return 0
    variantes = doublons.groupby("client_id")[colonnes_comparees].nunique().gt(1).any(axis=1)
    client_ids_impurs = variantes[variantes].index.tolist()
    if client_ids_impurs:
        raise ValueError(f"client_id dupliqués avec des valeurs différentes (doublons non purs) : {client_ids_impurs[:5]}")
    return len(doublons)


def label_stats(stats: dict[str, int | float]) -> dict[str, int | float]:
    """Remplace les clés techniques d'un dict de stats par leur libellé lisible (`STATS_LABELS`)."""
    return {STATS_LABELS.get(cle, cle): valeur for cle, valeur in stats.items()}


def format_stats(stats: dict[str, int | float]) -> str:
    """Formate un dict de stats en tableau aligné sur le libellé le plus long."""
    labels = label_stats(stats)
    largeur = max(len(cle) for cle in labels)
    return "\n".join(f"{cle:<{largeur}} : {valeur}" for cle, valeur in labels.items())


def build_silver(engine: Engine | None = None) -> tuple[dict[str, int | float], str]:
    """Charge `raw_churn`/`raw_plan_catalogue`, nettoie/type/valide les données, écrit `silver_churn`/`silver_plan_catalogue`."""
    engine = engine or get_engine()
    raw = pd.read_sql_table(SOURCE_TABLE, engine)

    n_avant = len(raw)
    colonnes_comparees = [c for c in raw.columns if c not in ("source_row_number", "ingested_at")]
    n_doublons_purs_valides = _valider_doublons_purs(raw, colonnes_comparees)
    df = raw.drop_duplicates(subset=colonnes_comparees, keep="first").copy()

    for col in _TITLE_CASE_COLUMNS:
        df[col] = _blank_to_na(df[col].str.strip()).str.title()
    df["taille_entreprise"] = _blank_to_na(df["taille_entreprise"].str.strip()).map(
        regles.normaliser_taille_entreprise, na_action="ignore"
    )
    tailles_inconnues = set(df["taille_entreprise"].dropna()) - regles.TAILLES_ENTREPRISE_CONNUES
    if tailles_inconnues:
        raise ValueError(f"taille_entreprise hors des valeurs connues : {sorted(tailles_inconnues)}")
    for col in _STRIP_ONLY_COLUMNS:
        df[col] = _blank_to_na(df[col].str.strip())

    n_secteur_manquants = int(df["secteur"].isna().sum())
    df["secteur"] = df["secteur"].fillna("n/a")

    df["date_souscription_normalized"] = _parse_dates(df["date_souscription"])
    concordance_jour = _concordance_jour_souscription(df["date_souscription_normalized"], df["jour_souscription"])
    if concordance_jour < 1.0:
        raise ValueError(f"Concordance jour_souscription/date_souscription = {concordance_jour:.1%} (attendu 100%)")

    df["plan_normalized"] = _snake_case(df["plan"])
    catalogue = _build_silver_plan_catalogue(engine)

    plans_valides = set(df["plan_normalized"].dropna())
    plans_inconnus = plans_valides - set(catalogue["plan_normalized"])
    if plans_inconnus:
        raise ValueError(f"plan_normalized absents du catalogue : {sorted(plans_inconnus)}")

    for col in _FLOAT_COLUMNS:
        df[col] = regles.nombre_depuis_texte(df[col])
    for col in _INT_COLUMNS:
        df[col] = pd.to_numeric(_blank_to_na(df[col]), errors="raise").astype("Int64")
    df["churn"] = pd.to_numeric(_blank_to_na(df["churn"]), errors="raise").astype("Int64").astype("boolean")

    n_taux_adoption_impute = int(df["taux_adoption_pct"].isna().sum())
    n_nb_integrations_impute = int(df["nb_integrations"].isna().sum())
    n_delai_impute = int(df["delai_reponse_support_h"].isna().sum())
    df = regles.completer_valeurs_absentes(df)

    for col, (minimum, maximum) in regles.PLAGES.items():
        _valider_plage(
            df[col],
            col,
            minimum if minimum is not None else float("-inf"),
            maximum if maximum is not None else float("inf"),
        )
    n_taux_adoption_valides = len(df)

    fonctionnalites_invalides = df["fonctionnalites_utilisees"] > df["fonctionnalites_total"]
    if fonctionnalites_invalides.any():
        raise ValueError(
            f"{int(fonctionnalites_invalides.sum())} lignes avec fonctionnalites_utilisees > fonctionnalites_total"
        )
    n_fonctionnalites_utilisees_valides = len(df)

    fonctionnalites_attendues = df["plan_normalized"].map(
        catalogue.set_index("plan_normalized")["fonctionnalites_incluses"]
    )
    fonctionnalites_incoherentes = df["fonctionnalites_total"] != fonctionnalites_attendues
    if fonctionnalites_incoherentes.any():
        raise ValueError(
            f"{int(fonctionnalites_incoherentes.sum())} lignes avec fonctionnalites_total incohérent avec le catalogue"
        )
    n_fonctionnalites_total_valides = len(df)

    n_csat_valides = len(df)
    n_sante_compte_valides = len(df)

    utilisateurs_invalides = df["utilisateurs_actifs"] > df["sieges_souscrits"]
    if utilisateurs_invalides.any():
        raise ValueError(
            f"{int(utilisateurs_invalides.sum())} lignes avec utilisateurs_actifs > sieges_souscrits"
        )
    validation_occupation_sieges = (df["utilisateurs_actifs"] <= df["sieges_souscrits"]).mean()

    commentaire_csm_strip = _blank_to_na(df["commentaire_csm"].str.strip())
    n_commentaire_csm_manquants = int(commentaire_csm_strip.isna().sum())

    commentaires_renseignes = commentaire_csm_strip.dropna()
    commentaires_inconnus = commentaires_renseignes[
        ~commentaires_renseignes.isin(regles.COMMENTAIRES_CSM_CONNUS)
    ].unique()
    if len(commentaires_inconnus):
        raise ValueError(f"commentaire_csm hors des 13 valeurs connues : {commentaires_inconnus.tolist()[:5]}")
    n_commentaire_csm_valides = len(df)

    df["commentaire_csm"] = commentaire_csm_strip.fillna(regles.COMMENTAIRE_CSM_ABSENT)

    df = df.drop(columns=list(_DROP_COLUMNS), errors="ignore")

    df.to_sql(TARGET_TABLE, engine, if_exists="replace", index=False)

    stats = {
        "lignes_avant": n_avant,
        "doublons_purs_valides": n_doublons_purs_valides,
        "lignes_apres": len(df),
        "concordance_jour_souscription": round(float(concordance_jour), 4),
        "secteur_manquants_imputes": n_secteur_manquants,
        "taux_adoption_pct_imputes": n_taux_adoption_impute,
        "taux_adoption_pct_valides": n_taux_adoption_valides,
        "nb_integrations_imputes": n_nb_integrations_impute,
        "delai_reponse_support_h_imputes": n_delai_impute,
        "delai_reponse_support_h_valides": len(df),
        "fonctionnalites_utilisees_valides": n_fonctionnalites_utilisees_valides,
        "fonctionnalites_total_valides": n_fonctionnalites_total_valides,
        "plans_normalized_valides": len(plans_valides),
        "csat_valides": n_csat_valides,
        "sante_compte_valides": n_sante_compte_valides,
        "commentaire_csm_manquants": n_commentaire_csm_manquants,
        "commentaire_csm_valides": n_commentaire_csm_valides,
        "validation_occupation_sieges": round(float(validation_occupation_sieges), 4),
        "catalogue_plans_lignes": len(catalogue),
    }
    bronze = last_build(engine, "bronze")
    log_message = log_build(engine, layer="silver", stats=stats, provenance=bronze["commit_id"] if bronze else None)
    if bronze:
        suffixe_dirty = "-dirty" if bronze["dirty"] else ""
        log_message += (
            f"\nbronze de base : id={bronze['id']}, built_at={bronze['built_at']}, "
            f"commit={bronze['commit_id'][:8]}{suffixe_dirty}"
        )
    return stats, log_message


def _build_silver_plan_catalogue(engine: Engine) -> pd.DataFrame:
    """Charge `raw_plan_catalogue`, type/normalise les colonnes, écrit `silver_plan_catalogue`."""
    raw = pd.read_sql_table(SOURCE_TABLE_PLAN_CATALOGUE, engine)
    catalogue = raw.copy()

    catalogue["plan_normalized"] = _snake_case(catalogue["plan"])
    for col in _CATALOGUE_INT_COLUMNS:
        catalogue[col] = pd.to_numeric(_blank_to_na(catalogue[col]), errors="raise").astype("Int64")
    catalogue["support_dedie"] = _to_bool_oui_non(catalogue["support_dedie"])

    catalogue = catalogue.drop(columns=["source_row_number", "ingested_at", "plan"], errors="ignore")

    catalogue.to_sql(TARGET_TABLE_PLAN_CATALOGUE, engine, if_exists="replace", index=False)
    return catalogue
