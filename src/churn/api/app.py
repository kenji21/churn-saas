"""API de prediction du churn : POST /prediction avec les donnees brutes d'un client.

Lancement : make api (ou uv run uvicorn churn.api.app:app). Le modele servi est CHURN_MODELE s'il est
defini, sinon le dernier data/modele_champion_*.joblib exporte depuis le notebook (section 10)."""

from __future__ import annotations

import os
from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from pathlib import Path

import joblib
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import JSONResponse

from churn.api.prediction import EntreeInvalide, predire
from churn.api.schemas import Client, ErreurChamp, ErreurEntree, Prediction
from churn.gold import PARQUET_DIR


def chemin_modele() -> Path:
    if chemin := os.environ.get("CHURN_MODELE"):
        return Path(chemin)
    exports = sorted(PARQUET_DIR.glob("modele_champion_*.joblib"), key=lambda p: p.stat().st_mtime)
    if not exports:
        raise RuntimeError("Aucun data/modele_champion_*.joblib : exporter le modele depuis le notebook (section 10).")
    return exports[-1]


@asynccontextmanager
async def _cycle_de_vie(app: FastAPI) -> AsyncIterator[None]:
    app.state.artefact = joblib.load(chemin_modele())
    yield


app = FastAPI(title="Prediction du churn", lifespan=_cycle_de_vie)


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


@app.get("/modele")
def modele(request: Request) -> dict[str, str | float]:
    artefact = request.app.state.artefact
    return {"modele": artefact["nom"], "gold": artefact["gold"], "seuil": artefact["seuil"], "run_mlflow": artefact["run_mlflow"]}


@app.post("/prediction", responses={400: {"model": ErreurEntree}})
def prediction(client: Client, request: Request) -> Prediction:
    return predire(client, request.app.state.artefact)
