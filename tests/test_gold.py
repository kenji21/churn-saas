from __future__ import annotations

import pandas as pd

from churn.gold import _taux_utilisation_fonctionnalites


def test_taux_utilisation_fonctionnalites_calcule_le_ratio_pourcent() -> None:
    df = pd.DataFrame({"fonctionnalites_utilisees": [5, 3], "fonctionnalites_total": [10, 4]})
    result = _taux_utilisation_fonctionnalites(df)
    assert result.tolist() == [50.0, 75.0]
