from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBClassifier

from churn.gold import PARQUET_SYMLINK

MLFLOW_TRACKING_URI = "sqlite:///data/mlflow.db"
MLFLOW_EXPERIMENT = "churn_saas"

_TARGET = "churn"

# Features du gold donnees au modele.
FEATURES_CATEGORIELLES = ("plan_normalized", "taille_entreprise", "commentaire_csm")
FEATURES_NUMERIQUES = (
    "anciennete_mois",
    "taux_adoption_pct",
    "connexions_30j",
    "heures_usage_30j",
    "nb_integrations",
    "derniere_connexion_jours",
    "tickets_support_90j",
    "delai_reponse_support_h",
    "csat",
    "retards_paiement_12m",
    "taux_utilisation_fonctionnalites_pct",
    "support_dedie_from_plan",
)

# Colonnes du gold volontairement absentes de l'entrainement (cf notebook pour le detail) :
_COLONNES_EXCLUES: dict[str, str] = {
    "client_id": "identifiant, pas une feature",
    "sante_compte_fin_periode": (
        "fuite -- separe quasi parfaitement les classes (PR-AUC 0.999 avec, 0.746 sans)"
    ),
    "revenu_mensuel_recurrent_eur": "cible secondaire (valeur vie client), jamais en entree du modele churn",
    _TARGET: "la cible elle-meme",
}


def charger_split(random_state: int = 42, test_size: float = 0.2) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    """Charge le gold depuis le .parquet pinne par commit (immuable, contrairement a la table gold_churn
    reecrite a chaque build_gold), split train/validation stratifie sur churn."""
    gold_sha = PARQUET_SYMLINK.resolve().stem.removeprefix("gold_churn_")
    gold_churn = pd.read_parquet(PARQUET_SYMLINK)
    train, val = train_test_split(
        gold_churn, test_size=test_size, stratify=gold_churn[_TARGET], random_state=random_state
    )
    return train, val, gold_sha


def construire_pretraitement() -> ColumnTransformer:
    """OneHot sur les categorielles, imputation mediane + normalisation sur les numeriques
    (heures_usage_30j, csat, retards_paiement_12m ont des NaN reels dans le gold)."""
    return ColumnTransformer(
        [
            ("categorielles", OneHotEncoder(handle_unknown="ignore"), list(FEATURES_CATEGORIELLES)),
            (
                "numeriques",
                Pipeline([("imputation", SimpleImputer(strategy="median")), ("normalisation", StandardScaler())]),
                list(FEATURES_NUMERIQUES),
            ),
        ]
    )


def decrire_features() -> str:
    """Features donnees au modele vs colonnes du gold exclues de l'entrainement (et pourquoi) --
    pour comprendre l'input du gold sans rouvrir le notebook."""
    lignes = ["Features categorielles (OneHot) :"]
    lignes += [f"  - {f}" for f in FEATURES_CATEGORIELLES]
    lignes.append("Features numeriques (imputation mediane + normalisation) :")
    lignes += [f"  - {f}" for f in FEATURES_NUMERIQUES]
    lignes.append("Colonnes du gold exclues de l'entrainement :")
    lignes += [f"  - {col} : {raison}" for col, raison in _COLONNES_EXCLUES.items()]
    return "\n".join(lignes)


def _entrainer_et_logger(
    nom_run: str, estimateur: ClassifierMixin, gold_sha: str, train: pd.DataFrame, val: pd.DataFrame
) -> tuple[dict[str, float | str], str]:
    colonnes = list(FEATURES_CATEGORIELLES) + list(FEATURES_NUMERIQUES)
    X_train, y_train = train[colonnes], train[_TARGET]
    X_val, y_val = val[colonnes], val[_TARGET]

    modele = Pipeline([("pretraitement", construire_pretraitement()), ("estimateur", estimateur)])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=nom_run):
        params = modele.named_steps["estimateur"].get_params()

        mlflow.set_tag("gold_commit", gold_sha)
        mlflow.log_param("gold_parquet", PARQUET_SYMLINK.resolve().name)
        mlflow.log_params(params)

        modele.fit(X_train, y_train)

        proba_val = modele.predict_proba(X_val)[:, 1]
        pred_val = modele.predict(X_val)

        pr_auc = average_precision_score(y_val, proba_val)
        f1 = f1_score(y_val, pred_val)

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.sklearn.log_model(modele, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

        stats: dict[str, float | str] = {"gold_commit": gold_sha, "pr_auc": pr_auc, "f1": f1}
        log_message = "\n".join(
            [
                f"gold     : {gold_sha} ({PARQUET_SYMLINK.resolve().name})",
                "",
                decrire_features(),
                "",
                f"params   : {json.dumps(params, indent=2, default=str)}",
                f"PR-AUC   : {pr_auc:.3f}",
                f"F1-score : {f1:.3f}",
            ]
        )
        mlflow.log_text(log_message, "log.txt")

    return stats, log_message


def train_reg_log_baseline(train: pd.DataFrame, val: pd.DataFrame, gold_sha: str) -> tuple[dict[str, float | str], str]:
    """Baseline regression logistique : tous les hyperparametres pertinents figes explicitement
    (pas de class_weight -- cf notebook pour la discussion sur le desequilibre 28/72)."""
    estimateur = LogisticRegression(max_iter=1000, random_state=42, class_weight=None, solver="lbfgs", penalty="l2")
    return _entrainer_et_logger(f"logreg_baseline_{gold_sha}", estimateur, gold_sha, train, val)


def train_random_forest_baseline(
    train: pd.DataFrame, val: pd.DataFrame, gold_sha: str
) -> tuple[dict[str, float | str], str]:
    """Baseline Random Forest."""
    estimateur = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    return _entrainer_et_logger(f"random_forest_baseline_{gold_sha}", estimateur, gold_sha, train, val)


def train_xgboost_baseline(train: pd.DataFrame, val: pd.DataFrame, gold_sha: str) -> tuple[dict[str, float | str], str]:
    """Baseline XGBoost. scale_pos_weight calcule depuis le train (ratio non-churn/churn) plutot que fige,
    pour rester correct si le taux de churn du gold change."""
    scale_pos_weight = (~train[_TARGET]).sum() / train[_TARGET].sum()
    estimateur = XGBClassifier(
        n_estimators=300,
        max_depth=5,
        learning_rate=0.1,
        scale_pos_weight=scale_pos_weight,
        random_state=42,
    )
    return _entrainer_et_logger(f"xgboost_baseline_{gold_sha}", estimateur, gold_sha, train, val)
