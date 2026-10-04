from __future__ import annotations

from collections.abc import Iterator
from typing import Any

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from churn.api.app import app
from churn.gold import PARQUET_DIR

# Donnees brutes du client CLI-001600 (jeu de validation), telles que saisies avant nettoyage :
# plan et taille dans une casse quelconque, commentaire absent.
CLIENT = {
    "client_id": "CLI-001600",
    "plan": " Pro ",
    "taille_entreprise": "tpe",
    "anciennete_mois": 5,
    "sieges_souscrits": 5,
    "utilisateurs_actifs": 3,
    "taux_adoption_pct": 60.0,
    "connexions_30j": 18,
    "heures_usage_30j": 7.6,
    "fonctionnalites_utilisees": 1,
    "nb_integrations": 1,
    "derniere_connexion_jours": 1,
    "tickets_support_90j": 4,
    "delai_reponse_support_h": 15.6,
    "csat": 1,
    "retards_paiement_12m": 3,
}


@pytest.fixture(scope="module")
def api() -> Iterator[TestClient]:
    with TestClient(app) as client:
        yield client


def _champs_en_erreur(reponse: Any) -> set[str]:
    assert reponse.status_code == 400, reponse.text
    return {erreur["champ"] for erreur in reponse.json()["erreurs"]}


def test_prediction_identique_au_modele_applique_a_la_ligne_du_gold(api: TestClient) -> None:
    """Garde-fou train/serve : depuis les donnees brutes, l'API doit reconstruire exactement les features
    du gold, donc donner la meme probabilite que le modele applique a la ligne du gold."""
    artefact = app.state.artefact
    gold = pd.read_parquet(PARQUET_DIR / artefact["gold"]).set_index("client_id")
    attendu = artefact["modele"].predict_proba(gold.loc[["CLI-001600"], artefact["colonnes"]])[0, 1]

    reponse = api.post("/api/prediction", json=CLIENT)

    assert reponse.status_code == 200, reponse.text
    corps = reponse.json()
    assert corps["probabilite_churn"] == pytest.approx(attendu, abs=1e-4)
    assert corps["alerte"] == bool(attendu >= artefact["seuil"])
    assert corps["client_id"] == "CLI-001600"


def test_champs_optionnels_absents_acceptes(api: TestClient) -> None:
    optionnels = ("taux_adoption_pct", "heures_usage_30j", "nb_integrations", "delai_reponse_support_h", "csat")
    client = {k: v for k, v in CLIENT.items() if k not in optionnels}
    assert api.post("/api/prediction", json=client).status_code == 200


@pytest.mark.parametrize(
    ("modification", "champ"),
    [
        ({"csat": 7}, "csat"),
        ({"taux_adoption_pct": 120}, "taux_adoption_pct"),
        ({"anciennete_mois": -1}, "anciennete_mois"),
        ({"taille_entreprise": "multinationale"}, "taille_entreprise"),
        ({"commentaire_csm": "texte libre"}, "commentaire_csm"),
        ({"utilisateurs_actifs": 6}, "utilisateurs_actifs"),
        ({"connexions_30j": "beaucoup"}, "connexions_30j"),
        ({"plan": "platinum"}, "plan"),
        ({"fonctionnalites_utilisees": 999}, "fonctionnalites_utilisees"),
        ({"champ_inconnu": 1}, "champ_inconnu"),
    ],
)
def test_entree_invalide_renvoie_400_avec_le_champ(api: TestClient, modification: dict, champ: str) -> None:
    assert _champs_en_erreur(api.post("/api/prediction", json={**CLIENT, **modification})) == {champ}


def test_champ_obligatoire_manquant(api: TestClient) -> None:
    client = {k: v for k, v in CLIENT.items() if k != "plan"}
    assert _champs_en_erreur(api.post("/api/prediction", json=client)) == {"plan"}


def test_plusieurs_erreurs_listees_ensemble(api: TestClient) -> None:
    reponse = api.post("/api/prediction", json={**CLIENT, "csat": 0, "taux_adoption_pct": -5, "taille_entreprise": "?"})
    assert _champs_en_erreur(reponse) == {"csat", "taux_adoption_pct", "taille_entreprise"}


def test_prediction_vie_identique_au_modele_applique_a_la_ligne_du_gold(api: TestClient) -> None:
    """Meme garde-fou que pour le churn : depuis les donnees brutes + le MRR, l'API retrouve la prediction du
    modele de valeur vie applique a la ligne du gold."""
    artefact = app.state.artefact_vie
    gold = pd.read_parquet(PARQUET_DIR / artefact["gold"]).set_index("client_id")
    attendu = max(0.0, artefact["modele"].predict(gold.loc[["CLI-001600"], artefact["colonnes"]])[0])

    reponse = api.post("/api/prediction_vie", json={**CLIENT, "revenu_mensuel_recurrent_eur": 120.45})

    assert reponse.status_code == 200, reponse.text
    assert reponse.json()["valeur_vie_client_eur"] == pytest.approx(attendu, abs=0.01)
    assert reponse.json()["client_id"] == "CLI-001600"


def test_modele_vie_decrit_le_modele_servi(api: TestClient) -> None:
    corps = api.get("/api/modele_vie").json()
    assert corps["modele"] == app.state.artefact_vie["nom"]
    assert corps["gold"] == app.state.artefact_vie["gold"]


def test_prediction_vie_mrr_absent_accepte(api: TestClient) -> None:
    assert api.post("/api/prediction_vie", json=CLIENT).status_code == 200


def test_prediction_vie_mrr_negatif_refuse(api: TestClient) -> None:
    reponse = api.post("/api/prediction_vie", json={**CLIENT, "revenu_mensuel_recurrent_eur": -1})
    assert _champs_en_erreur(reponse) == {"revenu_mensuel_recurrent_eur"}


def test_prediction_churn_refuse_le_mrr(api: TestClient) -> None:
    """Le MRR n'est pas une entree du modele churn : le contrat de /prediction reste strict."""
    reponse = api.post("/api/prediction", json={**CLIENT, "revenu_mensuel_recurrent_eur": 120.45})
    assert _champs_en_erreur(reponse) == {"revenu_mensuel_recurrent_eur"}
