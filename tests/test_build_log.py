from __future__ import annotations

import re

from churn.build_log import code_diff_stat, commit_for_path, commit_id

_SHA_RE = re.compile(r"^[0-9a-f]{40}$")


def test_commit_id_renvoie_un_sha_git() -> None:
    assert _SHA_RE.match(commit_id())


def test_commit_for_path_renvoie_un_sha_git() -> None:
    assert _SHA_RE.match(commit_for_path("sujet"))


def test_code_diff_stat_ne_leve_pas_sur_un_chemin_sans_changement() -> None:
    assert code_diff_stat("README.md") == ""
