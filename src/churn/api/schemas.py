"""Contrat d'entree/sortie de l'API. Les bornes et valeurs autorisees viennent de churn.regles, les
memes que celles appliquees par le silver : une donnee refusee a l'entrainement l'est aussi ici."""

from __future__ import annotations

from pydantic import BaseModel, ConfigDict, Field, ValidationInfo, field_validator

from churn import regles


def _bornes(colonne: str) -> dict[str, float | None]:
    minimum, maximum = regles.PLAGES[colonne]
    return {"ge": minimum, "le": maximum}


class Client(BaseModel):
    """Donnees brutes d'un client, telles qu'un CRM les connait (avant nettoyage et features du gold)."""

    model_config = ConfigDict(extra="forbid")

    client_id: str | None = None
    plan: str = Field(min_length=1, examples=["Pro"])
    taille_entreprise: str = Field(examples=["PME"])
    commentaire_csm: str | None = Field(default=None, validate_default=True, examples=["Suivi standard."])
    anciennete_mois: int = Field(**_bornes("anciennete_mois"))
    sieges_souscrits: int = Field(**_bornes("sieges_souscrits"))
    utilisateurs_actifs: int = Field(**_bornes("utilisateurs_actifs"))
    taux_adoption_pct: float | None = Field(
        default=None, **_bornes("taux_adoption_pct"), description="Absent : utilisateurs_actifs / sieges_souscrits"
    )
    connexions_30j: int = Field(**_bornes("connexions_30j"))
    heures_usage_30j: float | None = Field(default=None, **_bornes("heures_usage_30j"))
    fonctionnalites_utilisees: int = Field(**_bornes("fonctionnalites_utilisees"))
    nb_integrations: int | None = Field(default=None, **_bornes("nb_integrations"))
    derniere_connexion_jours: int = Field(**_bornes("derniere_connexion_jours"))
    tickets_support_90j: int = Field(**_bornes("tickets_support_90j"))
    delai_reponse_support_h: float | None = Field(default=None, **_bornes("delai_reponse_support_h"))
    csat: int | None = Field(default=None, **_bornes("csat"))
    retards_paiement_12m: int | None = Field(default=None, **_bornes("retards_paiement_12m"))

    @field_validator("plan")
    @classmethod
    def _normaliser_plan(cls, valeur: str) -> str:
        return regles.normaliser_plan(valeur)

    @field_validator("taille_entreprise")
    @classmethod
    def _valider_taille_entreprise(cls, valeur: str) -> str:
        taille = regles.normaliser_taille_entreprise(valeur)
        if taille not in regles.TAILLES_ENTREPRISE_CONNUES:
            raise ValueError(f"valeurs acceptees : {sorted(regles.TAILLES_ENTREPRISE_CONNUES)}")
        return taille

    @field_validator("commentaire_csm")
    @classmethod
    def _valider_commentaire_csm(cls, valeur: str | None) -> str:
        commentaire = regles.normaliser_commentaire_csm(valeur)
        if commentaire != regles.COMMENTAIRE_CSM_ABSENT and commentaire not in regles.COMMENTAIRES_CSM_CONNUS:
            raise ValueError(f"valeurs acceptees : {sorted(regles.COMMENTAIRES_CSM_CONNUS)} ou absent")
        return commentaire

    @field_validator("utilisateurs_actifs")
    @classmethod
    def _valider_utilisateurs_actifs(cls, valeur: int, info: ValidationInfo) -> int:
        sieges = info.data.get("sieges_souscrits")
        if sieges is not None and valeur > sieges:
            raise ValueError(f"ne peut pas depasser sieges_souscrits ({sieges})")
        return valeur


class Prediction(BaseModel):
    client_id: str | None
    probabilite_churn: float
    seuil: float
    alerte: bool
    modele: str
    gold: str


class ErreurChamp(BaseModel):
    champ: str
    message: str


class ErreurEntree(BaseModel):
    erreurs: list[ErreurChamp]
