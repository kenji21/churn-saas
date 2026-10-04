"""API de prediction, sous /api : POST /api/prediction (churn) et POST /api/prediction_vie (valeur vie client),
avec les donnees brutes d'un client ; GET /api/modele et /api/modele_vie decrivent les modeles servis.

Lancement : make api (ou uv run uvicorn churn.api.app:app). Modeles servis : CHURN_MODELE et
CHURN_MODELE_VIE s'ils sont definis, sinon les derniers data/modele_champion_*.joblib et
data/modele_vie_*.joblib exportes depuis le notebook (section 10)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import APIRouter, FastAPI, HTTPException, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.api.prediction import EntreeInvalide, predire, predire_vie
from churn.api.schemas import Client, ClientVie, ErreurChamp, ErreurEntree, Prediction, PredictionVie
from churn.gold import PARQUET_DIR


def chemin_modele() -> Path:
    if chemin := os.environ.get("CHURN_MODELE"):
        return Path(chemin)
    exports = sorted(PARQUET_DIR.glob("modele_champion_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not exports:
        raise RuntimeError("Aucun data/modele_champion_*.joblib : exporter le modele depuis le notebook (section 10).")
    return exports[-1]


def chemin_modele_vie() -> Path | None:
    """Optionnel : sans modele de valeur vie exporte, l'API sert le churn et /api/*_vie repondent 503."""
    if chemin := os.environ.get("CHURN_MODELE_VIE"):
        return Path(chemin)
    exports = sorted(PARQUET_DIR.glob("modele_vie_*.joblib"), key=lambda p: p.stat().st_mtime)
    return exports[-1] if exports else None


@asynccontextmanager
async def _cycle_de_vie(app: FastAPI) -> AsyncIterator[None]:
    app.state.artefact = joblib.load(chemin_modele())
    chemin_vie = chemin_modele_vie()
    app.state.artefact_vie = joblib.load(chemin_vie) if chemin_vie else None
    yield


app = FastAPI(title="Prediction du churn et de la valeur vie client", lifespan=_cycle_de_vie)
api = APIRouter(prefix="/api")


def _reponse_400(erreurs: list[ErreurChamp]) -> JSONResponse:
    return JSONResponse(status_code=400, content=ErreurEntree(erreurs=erreurs).model_dump())


@app.exception_handler(RequestValidationError)
async def _erreurs_de_validation(request: Request, exc: RequestValidationError) -> JSONResponse:
    """400 (au lieu du 422 par defaut de FastAPI), un element par champ en erreur."""
    erreurs = [
        ErreurChamp(
            champ=".".join(str(p) for p in erreur["loc"][1:]) or "corps de la requete",
            message=erreur["msg"].removeprefix("Value error, "),
        )
        for erreur in exc.errors()
    ]
    return _reponse_400(erreurs)


@app.exception_handler(EntreeInvalide)
async def _entree_invalide(request: Request, exc: EntreeInvalide) -> JSONResponse:
    return _reponse_400(exc.erreurs)


def _artefact_vie(request: Request) -> dict:
    artefact = request.app.state.artefact_vie
    if artefact is None:
        raise HTTPException(503, "Aucun data/modele_vie_*.joblib : exporter le modele depuis le notebook (section 10).")
    return artefact


_AUCUN_MODELE_VIE = {503: {"description": "Aucun modele de valeur vie exporte"}}


@api.get("/modele")
def modele(request: Request) -> dict[str, str | float]:
    artefact = request.app.state.artefact
    return {"modele": artefact["nom"], "gold": artefact["gold"], "seuil": artefact["seuil"], "run_mlflow": artefact["run_mlflow"]}


@api.post("/prediction", responses={400: {"model": ErreurEntree}})
def prediction(client: Client, request: Request) -> Prediction:
    return predire(client, request.app.state.artefact)


@api.get("/modele_vie", responses=_AUCUN_MODELE_VIE)
def modele_vie(request: Request) -> dict[str, str]:
    artefact = _artefact_vie(request)
    return {"modele": artefact["nom"], "gold": artefact["gold"], "run_mlflow": artefact["run_mlflow"]}


@api.post("/prediction_vie", responses={400: {"model": ErreurEntree}, **_AUCUN_MODELE_VIE})
def prediction_vie(client: ClientVie, request: Request) -> PredictionVie:
    return predire_vie(client, _artefact_vie(request))


app.include_router(api)
