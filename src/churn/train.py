from __future__ import annotations

import json

import mlflow
import mlflow.sklearn
import optuna
import pandas as pd
from sklearn.base import ClassifierMixin
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import average_precision_score, precision_recall_curve
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, OrdinalEncoder, StandardScaler
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


def construire_pretraitement_ordinal() -> ColumnTransformer:
    """Comme construire_pretraitement, mais commentaire_csm (13 categories) passe par un OrdinalEncoder
    plutot qu'un OneHot -- qui diluerait le tirage max_features des arbres (13 dummies pour une seule
    feature, contre 4 pour plan_normalized/taille_entreprise)."""
    return ColumnTransformer(
        [
            ("categorielles", OneHotEncoder(handle_unknown="ignore"), ["plan_normalized", "taille_entreprise"]),
            (
                "commentaire_csm",
                OrdinalEncoder(handle_unknown="use_encoded_value", unknown_value=-1),
                ["commentaire_csm"],
            ),
            (
                "numeriques",
                Pipeline([("imputation", SimpleImputer(strategy="median")), ("normalisation", StandardScaler())]),
                list(FEATURES_NUMERIQUES),
            ),
        ]
    )


def decrire_features(encodage_commentaire_csm: str = "OneHot") -> str:
    """Features donnees au modele vs colonnes du gold exclues de l'entrainement (et pourquoi) --
    pour comprendre l'input du gold sans rouvrir le notebook."""
    lignes = [
        "Features categorielles :",
        "  - plan_normalized (OneHot)",
        "  - taille_entreprise (OneHot)",
        f"  - commentaire_csm ({encodage_commentaire_csm})",
    ]
    lignes.append("Features numeriques (imputation mediane + normalisation) :")
    lignes += [f"  - {f}" for f in FEATURES_NUMERIQUES]
    lignes.append("Colonnes du gold exclues de l'entrainement :")
    lignes += [f"  - {col} : {raison}" for col, raison in _COLONNES_EXCLUES.items()]
    return "\n".join(lignes)


def _separer_X_y(df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
    colonnes = list(FEATURES_CATEGORIELLES) + list(FEATURES_NUMERIQUES)
    return df[colonnes], df[_TARGET]


def _meilleur_seuil_f1(y_val: pd.Series, proba_val) -> tuple[float, float]:
    """Seuil de decision qui maximise le F1-score sur la validation, plutot que le 0.5 par defaut de
    .predict(). Necessaire des qu'on utilise le modele pour une decision binaire actionnable (appeler
    le client / prioriser ses tickets), par opposition a un simple classement (role du PR-AUC) --
    sinon le F1 rapporte depend d'un seuil arbitraire qui n'a aucune raison d'etre le bon point de
    coupure, en particulier avec class_weight qui deplace la distribution des probabilites predites."""
    precision, recall, seuils = precision_recall_curve(y_val, proba_val)
    f1_scores = 2 * precision * recall / (precision + recall + 1e-12)
    meilleur_idx = f1_scores[:-1].argmax()
    return float(seuils[meilleur_idx]), float(f1_scores[meilleur_idx])


def _info_metier(y_val: pd.Series, proba_val, seuil: float) -> tuple[dict[str, float], str]:
    """Traduit precision/recall au seuil choisi en langage metier (clients signales, churners
    rattrapes, fausses alertes) -- ce que F1/PR-AUC seuls ne disent pas directement, cf discussion
    notebook sur la chute F1 0.650 -> 0.556 (recall qui s'ecroule sans que le F1 seul l'explicite)."""
    pred = proba_val >= seuil
    n_signales = int(pred.sum())
    n_churners = int(y_val.sum())
    n_detectes = int((pred & y_val.to_numpy()).sum())
    n_fausses_alertes = n_signales - n_detectes
    recall = n_detectes / n_churners if n_churners else 0.0
    precision = n_detectes / n_signales if n_signales else 0.0

    metriques = {
        "n_signales": n_signales,
        "n_detectes": n_detectes,
        "n_fausses_alertes": n_fausses_alertes,
        "recall": recall,
        "precision": precision,
    }
    lignes = [
        f"Decision metier (seuil = {seuil:.3f}, validation n={len(y_val)}) :",
        f"  {'Clients signales a risque':<38} : {n_signales}",
        f"  {'Vrais churners dans le val set':<38} : {n_churners}",
        f"  {'Vrais churners detectes (recall)':<38} : {n_detectes} ({recall:.1%})",
        f"  {'Fausses alertes (parmi signales)':<38} : {n_fausses_alertes} ({1 - precision:.1%})",
    ]
    return metriques, "\n".join(lignes)


def _decrire_recherche(study: optuna.Study, n_trials: int, nom_algo: str) -> str:
    """Espace de recherche Optuna, lu directement depuis les distributions du meilleur essai --
    jamais une description dupliquee a la main qui pourrait diverger du code au fil des changements."""
    lignes = [f"Recherche Optuna ({nom_algo}, {n_trials} essais, objectif = F1 au seuil optimal) :"]
    lignes += [f"  - {nom:<24} : {dist}" for nom, dist in study.best_trial.distributions.items()]
    return "\n".join(lignes)


def _entrainer_et_logger(
    nom_run: str,
    estimateur: ClassifierMixin,
    gold_sha: str,
    train: pd.DataFrame,
    val: pd.DataFrame,
    pretraitement: ColumnTransformer | None = None,
    encodage_commentaire_csm: str = "OneHot",
) -> tuple[dict[str, float | str], str]:
    X_train, y_train = _separer_X_y(train)
    X_val, y_val = _separer_X_y(val)

    modele = Pipeline([("pretraitement", pretraitement or construire_pretraitement()), ("estimateur", estimateur)])

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    with mlflow.start_run(run_name=nom_run):
        params = modele.named_steps["estimateur"].get_params()

        mlflow.set_tag("gold_commit", gold_sha)
        mlflow.log_param("gold_parquet", PARQUET_SYMLINK.resolve().name)
        mlflow.log_params(params)

        modele.fit(X_train, y_train)

        proba_val = modele.predict_proba(X_val)[:, 1]

        pr_auc = float(average_precision_score(y_val, proba_val))
        seuil, f1 = _meilleur_seuil_f1(y_val, proba_val)
        metier, tableau_metier = _info_metier(y_val, proba_val, seuil)

        mlflow.log_metric("pr_auc", pr_auc)
        mlflow.log_metric("f1", f1)
        mlflow.log_metric("seuil_f1", seuil)
        mlflow.log_metrics(metier)
        mlflow.sklearn.log_model(modele, "model", serialization_format=mlflow.sklearn.SERIALIZATION_FORMAT_PICKLE)

        stats: dict[str, float | str] = {
            "gold_commit": gold_sha,
            "pr_auc": pr_auc,
            "f1": f1,
            "seuil_f1": seuil,
            **metier,
        }
        log_message = "\n".join(
            [
                f"gold     : {gold_sha} ({PARQUET_SYMLINK.resolve().name})",
                "",
                decrire_features(encodage_commentaire_csm),
                "",
                f"params   : {json.dumps(params, indent=2, default=str)}",
                f"PR-AUC   : {pr_auc:.3f}",
                f"F1-score : {f1:.3f} (seuil decision = {seuil:.3f}, au lieu de 0.5 -- cf discussion "
                "F1 vs PR-AUC)",
                "",
                tableau_metier,
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


def train_random_forest_ordinal_baseline(
    train: pd.DataFrame, val: pd.DataFrame, gold_sha: str
) -> tuple[dict[str, float | str], str]:
    """Random Forest, OrdinalEncoder sur commentaire_csm (13 categories) au lieu d'OneHot -- le OneHot
    dilue le tirage max_features des arbres (13 dummies pour une seule feature, cf discussion notebook)."""
    estimateur = RandomForestClassifier(n_estimators=300, class_weight="balanced", n_jobs=-1, random_state=42)
    return _entrainer_et_logger(
        f"random_forest_ordinal_baseline_{gold_sha}",
        estimateur,
        gold_sha,
        train,
        val,
        pretraitement=construire_pretraitement_ordinal(),
        encodage_commentaire_csm="Ordinal",
    )


# Espace de recherche Optuna par defaut (Random Forest) -- declaratif : {nom: (type, *bornes)}.
# type="int"|"float"|"float_log" prend (low, high[, step]) ; type="categorical" prend (choix,).
# Une cellule notebook peut passer son propre dict a tuner_random_forest(espace_recherche=...) pour
# que l'espace recherche soit visible dans le code de la cellule plutot que fige ici.
ESPACE_RECHERCHE_RANDOM_FOREST: dict[str, tuple] = {
    "encodage_commentaire_csm": ("categorical", ["onehot", "ordinal"]),
    "n_estimators": ("int", 100, 500, 50),
    "max_depth": ("int", 3, 20),
    "min_samples_leaf": ("int", 1, 20),
    "max_features": ("categorical", ["sqrt", "log2", None]),
    "class_weight": ("categorical", ["balanced", None]),
}


def _suggerer_depuis_espace(trial: optuna.Trial, espace: dict[str, tuple]) -> dict[str, object]:
    """Traduit un espace de recherche declaratif (cf ESPACE_RECHERCHE_RANDOM_FOREST/XGBOOST) en
    appels trial.suggest_* -- l'espace reste une donnee simple (dict), inspectable et passable
    depuis une cellule notebook, plutot que des appels suggest_* eparpilles dans une fonction."""
    valeurs: dict[str, object] = {}
    for nom, spec in espace.items():
        type_spec, *bornes = spec
        if type_spec == "int":
            low, high, *step = bornes
            valeurs[nom] = trial.suggest_int(nom, low, high, step=step[0] if step else 1)
        elif type_spec == "float":
            low, high = bornes
            valeurs[nom] = trial.suggest_float(nom, low, high)
        elif type_spec == "float_log":
            low, high = bornes
            valeurs[nom] = trial.suggest_float(nom, low, high, log=True)
        elif type_spec == "categorical":
            (choix,) = bornes
            valeurs[nom] = trial.suggest_categorical(nom, choix)
        else:
            raise ValueError(f"type d'espace de recherche inconnu : {type_spec}")
    return valeurs


def tuner_random_forest(
    train: pd.DataFrame,
    val: pd.DataFrame,
    gold_sha: str,
    n_trials: int = 50,
    espace_recherche: dict[str, tuple] | None = None,
) -> tuple[dict[str, float | str], str]:
    """Recherche d'hyperparametres Optuna pour Random Forest. L'encodage de commentaire_csm
    (OneHot vs Ordinal, cf discussion notebook) est recherche comme un hyperparametre de plus,
    plutot que fige a l'avance -- Optuna n'a aucune notion de preprocessing, donc le choix de
    construire_pretraitement()/construire_pretraitement_ordinal() reste pilote ici, pas par Optuna.

    Optimise le PR-AUC (independant du seuil), pas le F1. Chercher le seuil qui maximise le F1 a
    l'interieur de la boucle Optuna reviendrait a optimiser deux choses a la fois (hyperparametres
    ET seuil) sur la meme validation (1000 lignes), a chaque essai -- le seuil finirait par
    sur-ajuster le bruit d'echantillonnage de la validation, repete sur n_trials essais. Le seuil
    n'est donc cherche qu'une seule fois, pour le seul modele final retenu, via
    _entrainer_et_logger -- jamais a l'interieur de la recherche elle-meme.

    Chaque essai est loggue en run MLflow imbrique (params + pr_auc, pas de modele -- seul le
    meilleur essai est reentrainage et loggue completement via _entrainer_et_logger)."""
    espace_recherche = espace_recherche or ESPACE_RECHERCHE_RANDOM_FOREST
    X_train, y_train = _separer_X_y(train)
    X_val, y_val = _separer_X_y(val)

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    def objectif(trial: optuna.Trial) -> float:
        suggestions = _suggerer_depuis_espace(trial, espace_recherche)
        encodage = suggestions.pop("encodage_commentaire_csm")
        hyperparams = suggestions
        pretraitement = construire_pretraitement_ordinal() if encodage == "ordinal" else construire_pretraitement()
        modele = Pipeline(
            [
                ("pretraitement", pretraitement),
                ("estimateur", RandomForestClassifier(random_state=42, n_jobs=-1, **hyperparams)),
            ]
        )
        modele.fit(X_train, y_train)
        proba_val = modele.predict_proba(X_val)[:, 1]
        pr_auc = float(average_precision_score(y_val, proba_val))

        with mlflow.start_run(run_name=f"rf_optuna_trial{trial.number:03d}_{gold_sha}", nested=True):
            mlflow.set_tag("gold_commit", gold_sha)
            mlflow.log_param("encodage_commentaire_csm", encodage)
            mlflow.log_params(hyperparams)
            mlflow.log_metric("pr_auc", pr_auc)

        return pr_auc

    with mlflow.start_run(run_name=f"rf_optuna_search_{gold_sha}"):
        mlflow.set_tag("gold_commit", gold_sha)
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objectif, n_trials=n_trials)

        meilleur_encodage = study.best_params["encodage_commentaire_csm"]
        meilleurs_hyperparams = {k: v for k, v in study.best_params.items() if k != "encodage_commentaire_csm"}

        mlflow.log_param("meilleur_encodage", meilleur_encodage)
        mlflow.log_params(meilleurs_hyperparams)
        mlflow.log_metric("best_pr_auc", study.best_value)

    description_recherche = _decrire_recherche(study, n_trials, "Random Forest")

    pretraitement = (
        construire_pretraitement_ordinal() if meilleur_encodage == "ordinal" else construire_pretraitement()
    )
    estimateur = RandomForestClassifier(random_state=42, n_jobs=-1, **meilleurs_hyperparams)
    stats, log_message = _entrainer_et_logger(
        f"random_forest_optuna_best_{gold_sha}",
        estimateur,
        gold_sha,
        train,
        val,
        pretraitement=pretraitement,
        encodage_commentaire_csm="OneHot" if meilleur_encodage == "onehot" else "Ordinal",
    )
    return stats, f"{description_recherche}\n\n{log_message}"


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


# cf ESPACE_RECHERCHE_RANDOM_FOREST pour le format. facteur_scale_pos_weight multiplie le ratio
# non-churn/churn du train (cf train_xgboost_baseline) -- pas fige a 1x ni laisse totalement libre.
ESPACE_RECHERCHE_XGBOOST: dict[str, tuple] = {
    "n_estimators": ("int", 100, 600, 50),
    "max_depth": ("int", 2, 10),
    "learning_rate": ("float_log", 0.01, 0.3),
    "subsample": ("float", 0.5, 1.0),
    "colsample_bytree": ("float", 0.5, 1.0),
    "min_child_weight": ("int", 1, 10),
    "reg_alpha": ("float_log", 1e-8, 10.0),
    "reg_lambda": ("float_log", 1e-8, 10.0),
    "facteur_scale_pos_weight": ("float", 0.5, 2.0),
}


def tuner_xgboost(
    train: pd.DataFrame,
    val: pd.DataFrame,
    gold_sha: str,
    n_trials: int = 50,
    espace_recherche: dict[str, tuple] | None = None,
) -> tuple[dict[str, float | str], str]:
    """Recherche d'hyperparametres Optuna pour XGBoost. Optimise le PR-AUC (independant du seuil),
    meme raisonnement que tuner_random_forest : chercher le seuil F1 a l'interieur de la boucle
    Optuna reviendrait a optimiser hyperparametres et seuil ensemble sur la meme validation, repete
    n_trials fois -- le seuil n'est donc cherche qu'une fois, pour le modele final, via
    _entrainer_et_logger. scale_pos_weight est recherche comme un multiplicateur du ratio
    non-churn/churn du train (pas fige a 1x ni laisse totalement libre), pour explorer autour de la
    valeur de reference de train_xgboost_baseline."""
    espace_recherche = espace_recherche or ESPACE_RECHERCHE_XGBOOST
    X_train, y_train = _separer_X_y(train)
    X_val, y_val = _separer_X_y(val)
    ratio_classes = (~train[_TARGET]).sum() / train[_TARGET].sum()

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(MLFLOW_EXPERIMENT)

    def objectif(trial: optuna.Trial) -> float:
        suggestions = _suggerer_depuis_espace(trial, espace_recherche)
        facteur_scale_pos_weight = float(suggestions.pop("facteur_scale_pos_weight"))
        hyperparams = suggestions
        scale_pos_weight = ratio_classes * facteur_scale_pos_weight

        modele = Pipeline(
            [
                ("pretraitement", construire_pretraitement()),
                ("estimateur", XGBClassifier(random_state=42, scale_pos_weight=scale_pos_weight, **hyperparams)),
            ]
        )
        modele.fit(X_train, y_train)
        proba_val = modele.predict_proba(X_val)[:, 1]
        pr_auc = float(average_precision_score(y_val, proba_val))

        with mlflow.start_run(run_name=f"xgb_optuna_trial{trial.number:03d}_{gold_sha}", nested=True):
            mlflow.set_tag("gold_commit", gold_sha)
            mlflow.log_params(hyperparams)
            mlflow.log_param("scale_pos_weight", scale_pos_weight)
            mlflow.log_metric("pr_auc", pr_auc)

        return pr_auc

    with mlflow.start_run(run_name=f"xgb_optuna_search_{gold_sha}"):
        mlflow.set_tag("gold_commit", gold_sha)
        mlflow.log_param("n_trials", n_trials)

        study = optuna.create_study(direction="maximize", sampler=optuna.samplers.TPESampler(seed=42))
        study.optimize(objectif, n_trials=n_trials)

        meilleurs_hyperparams = {k: v for k, v in study.best_params.items() if k != "facteur_scale_pos_weight"}
        meilleur_scale_pos_weight = ratio_classes * study.best_params["facteur_scale_pos_weight"]

        mlflow.log_params(meilleurs_hyperparams)
        mlflow.log_param("scale_pos_weight", meilleur_scale_pos_weight)
        mlflow.log_metric("best_pr_auc", study.best_value)

    description_recherche = _decrire_recherche(study, n_trials, "XGBoost")

    estimateur = XGBClassifier(random_state=42, scale_pos_weight=meilleur_scale_pos_weight, **meilleurs_hyperparams)
    stats, log_message = _entrainer_et_logger(f"xgboost_optuna_best_{gold_sha}", estimateur, gold_sha, train, val)
    return stats, f"{description_recherche}\n\n{log_message}"


_BASELINES_A_COMPARER: dict[str, str] = {
    "Regression logistique": "logreg_baseline",
    "Random Forest": "random_forest_baseline",
    "XGBoost": "xgboost_baseline",
}

_COLONNES_COMPARAISON = (
    "pr_auc",
    "f1",
    "seuil_f1",
    "recall",
    "precision",
    "n_detectes",
    "n_signales",
    "n_fausses_alertes",
)


def comparer_baselines(gold_sha: str, baselines: dict[str, str] = _BASELINES_A_COMPARER) -> pd.DataFrame:
    """Va chercher directement dans MLflow (pas de re-saisie a la main) le dernier run de chaque
    baseline pour ce gold, et construit un tableau comparatif PR-AUC / F1 / seuil / decision metier."""
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    lignes = []
    for label, prefixe_run in baselines.items():
        runs = mlflow.search_runs(
            experiment_names=[MLFLOW_EXPERIMENT],
            filter_string=f"tags.mlflow.runName = '{prefixe_run}_{gold_sha}'",
            order_by=["start_time DESC"],
            max_results=1,
        )
        if runs.empty:
            continue
        run = runs.iloc[0]
        ligne: dict[str, object] = {"modele": label}
        ligne.update({colonne: run.get(f"metrics.{colonne}") for colonne in _COLONNES_COMPARAISON})
        lignes.append(ligne)

    tableau = pd.DataFrame(lignes).set_index("modele")
    arrondis = {"pr_auc": 3, "f1": 3, "seuil_f1": 3, "recall": 3, "precision": 3}
    return tableau.round(arrondis)
