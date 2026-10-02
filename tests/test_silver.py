from __future__ import annotations

import pandas as pd
import pytest

from churn.regles import nombre_depuis_texte
from churn.silver import (
    _blank_to_na,
    _concordance_jour_souscription,
    _parse_dates,
    _snake_case,
    _to_bool_oui_non,
    _valider_plage,
)


def test_blank_to_na_convertit_uniquement_les_chaines_vides() -> None:
    result = _blank_to_na(pd.Series(["", "a", "", "b"]))
    assert result.isna().tolist() == [True, False, True, False]


@pytest.mark.parametrize(
    ("brut", "attendu"),
    [
        ("0,0", 0.0),
        ("33,3", 33.3),
        ("20.0%", 20.0),
        ("15.2", 15.2),
        ("280.62 €", 280.62),
        ("2000,67", 2000.67),
        ("14.6 h", 14.6),
        ("31,8 h", 31.8),
    ],
)
def test_nombre_depuis_texte_formats_reels_du_csv(brut: str, attendu: float) -> None:
    result = nombre_depuis_texte(pd.Series([brut]))
    assert result.iloc[0] == pytest.approx(attendu)


def test_nombre_depuis_texte_chaine_vide_devient_na() -> None:
    result = nombre_depuis_texte(pd.Series(["12.5", ""]))
    assert result.iloc[1] is pd.NA or pd.isna(result.iloc[1])


def test_parse_dates_reconnait_les_trois_motifs() -> None:
    series = pd.Series(["2024-02-06", "31/01/2024", "23 Dec 2024"])
    result = _parse_dates(series)
    assert list(result) == [
        pd.Timestamp("2024-02-06").date(),
        pd.Timestamp("2024-01-31").date(),
        pd.Timestamp("2024-12-23").date(),
    ]


def test_parse_dates_rejette_un_motif_inconnu() -> None:
    with pytest.raises(ValueError, match="hors motifs connus"):
        _parse_dates(pd.Series(["2024/02/06"]))


def test_concordance_jour_souscription_parfaite() -> None:
    dates = _parse_dates(pd.Series(["2024-02-06"]))  # 6 février 2024 est un mardi
    jours = pd.Series(["mardi"])
    assert _concordance_jour_souscription(dates, jours) == 1.0


def test_concordance_jour_souscription_detecte_incoherence() -> None:
    dates = _parse_dates(pd.Series(["2024-02-06", "2024-02-06"]))
    jours = pd.Series(["mardi", "lundi"])
    assert _concordance_jour_souscription(dates, jours) == 0.5


@pytest.mark.parametrize(
    ("brut", "attendu"),
    [
        (" Pro ", "pro"),
        ("STARTER", "starter"),
        ("Grand Compte", "grand_compte"),
        ("", None),
    ],
)
def test_snake_case(brut: str, attendu: str | None) -> None:
    result = _snake_case(pd.Series([brut]))
    if attendu is None:
        assert pd.isna(result.iloc[0])
    else:
        assert result.iloc[0] == attendu


def test_to_bool_oui_non_convertit_oui_non() -> None:
    result = _to_bool_oui_non(pd.Series(["Oui", "Non", ""]))
    assert result.iloc[0] == True  # noqa: E712 (BooleanArray scalar, pas un bool Python)
    assert result.iloc[1] == False  # noqa: E712
    assert pd.isna(result.iloc[2])


def test_to_bool_oui_non_rejette_valeur_inconnue() -> None:
    with pytest.raises(ValueError, match="hors Oui/Non"):
        _to_bool_oui_non(pd.Series(["Oui", "Peut-être"]))


def test_valider_plage_accepte_les_valeurs_dans_les_bornes() -> None:
    _valider_plage(pd.Series([1, 3, 5, None]), "csat", 1, 5)


def test_valider_plage_rejette_une_valeur_hors_bornes() -> None:
    with pytest.raises(ValueError, match=r"hors plage \[1, 5\]"):
        _valider_plage(pd.Series([1, 6]), "csat", 1, 5)
