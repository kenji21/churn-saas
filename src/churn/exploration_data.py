from __future__ import annotations

import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import chi2_contingency

VALEUR_MANQUANTE = "Inconnu"


def churn_par_categorie(df: pd.DataFrame, colonne: str) -> tuple[pd.DataFrame, float, float]:
    """Taux de churn par valeur de `colonne` (NaN regroupés sous "Inconnu") + test d'association chi²."""
    categorie = df[colonne].astype("string").fillna(VALEUR_MANQUANTE)
    stats = (
        df.groupby(categorie)["churn"]
        .agg(taux_churn="mean", nb_clients="count")
        .sort_values("taux_churn", ascending=False)
    )
    stats["taux_churn_pct"] = (stats["taux_churn"] * 100).round(1)
    assert stats["nb_clients"].sum() == len(df), "somme des groupes != lignes du DataFrame"

    table_contingence = pd.crosstab(categorie, df["churn"])
    chi2, p_value, _, _ = chi2_contingency(table_contingence)
    n = table_contingence.to_numpy().sum()
    cramers_v = (chi2 / (n * (min(table_contingence.shape) - 1))) ** 0.5
    return stats, p_value, cramers_v


def graphe_churn_par_categorie(df: pd.DataFrame, colonne: str, titre: str) -> None:
    """Barh du taux de churn par `colonne`, ligne de moyenne globale, p-value/Cramér's V en titre."""
    stats, p_value, cramers_v = churn_par_categorie(df, colonne)
    taux_global = df["churn"].mean() * 100

    fig, ax = plt.subplots(figsize=(7, 3.5))
    couleurs = ["#999999" if c == VALEUR_MANQUANTE else "#3b6fa0" for c in stats.index]
    ax.barh(stats.index, stats["taux_churn_pct"], color=couleurs, height=0.6)
    ax.axvline(taux_global, color="#666666", linestyle="--", linewidth=1)
    for y, (pct, nb) in enumerate(zip(stats["taux_churn_pct"], stats["nb_clients"])):
        ax.text(pct + 0.3, y, f"{pct:.1f}% (n={nb})", va="center", fontsize=9)
    ax.set_xlabel("Taux de churn (%)")
    ax.set_title(f"{titre}\np-value={p_value:.3f}, Cramér's V={cramers_v:.3f}")
    ax.invert_yaxis()
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()


def graphe_boxplot_par_categorie(
    df: pd.DataFrame, colonne_categorielle: str, colonne_numerique: str, ordre: list[str], titre: str
) -> None:
    """Boxplot de `colonne_numerique` par valeur (ordonnée) de `colonne_categorielle`, médiane/n en annotation."""
    donnees = [df.loc[df[colonne_categorielle] == cat, colonne_numerique].dropna() for cat in ordre]

    fig, ax = plt.subplots(figsize=(7, 4))
    bp = ax.boxplot(
        donnees,
        tick_labels=ordre,
        patch_artist=True,
        widths=0.5,
        medianprops={"color": "#1f3a5f"},
        boxprops={"facecolor": "#3b6fa0", "edgecolor": "#1f3a5f"},
        whiskerprops={"color": "#1f3a5f"},
        capprops={"color": "#1f3a5f"},
        flierprops={"markeredgecolor": "#3b6fa0", "markersize": 3, "alpha": 0.5},
    )
    y_max = max((v.max() for v in donnees if len(v)), default=1.0)
    for i, valeurs in enumerate(donnees):
        cap_haut = bp["caps"][2 * i + 1].get_ydata()[0]
        fliers_haut = bp["fliers"][i].get_ydata()
        sommet = max(cap_haut, fliers_haut.max() if len(fliers_haut) else cap_haut)
        ax.text(
            i + 1,
            sommet + y_max * 0.03,
            f"med={valeurs.median():.0f}\nn={len(valeurs)}",
            ha="center",
            va="bottom",
            fontsize=8,
            color="#1f3a5f",
        )
    ax.set_ylim(top=y_max * 1.25)
    ax.set_ylabel(colonne_numerique)
    ax.set_title(titre)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()


def graphe_boxplot_groupe(
    df: pd.DataFrame,
    colonne_groupe: str,
    colonne_sous_groupe: str,
    colonne_numerique: str,
    ordre_groupe: list[str],
    ordre_sous_groupe: list[str],
    titre: str,
) -> None:
    """Boxplot groupé : `colonne_numerique` par `colonne_groupe` (axe X), une boîte par `colonne_sous_groupe`."""
    n_sous = len(ordre_sous_groupe)
    couleurs = ["#3b6fa0", "#c0392b", "#2e7d32", "#8e44ad"][:n_sous]

    fig, ax = plt.subplots(figsize=(2.4 * len(ordre_groupe) + 1, 4.5))
    positions_groupes = []
    for gi, groupe in enumerate(ordre_groupe):
        base = gi * (n_sous + 1)
        for si, sous in enumerate(ordre_sous_groupe):
            valeurs = df.loc[(df[colonne_groupe] == groupe) & (df[colonne_sous_groupe] == sous), colonne_numerique].dropna()
            ax.boxplot(
                [valeurs],
                positions=[base + si],
                widths=0.7,
                patch_artist=True,
                medianprops={"color": "#1f3a5f"},
                boxprops={"facecolor": couleurs[si], "edgecolor": "#1f3a5f"},
                whiskerprops={"color": "#1f3a5f"},
                capprops={"color": "#1f3a5f"},
                flierprops={"markeredgecolor": couleurs[si], "markersize": 3, "alpha": 0.4},
            )
        positions_groupes.append(base + (n_sous - 1) / 2)

    ax.set_xticks(positions_groupes)
    ax.set_xticklabels(ordre_groupe)
    ax.set_ylabel(colonne_numerique)
    ax.set_title(titre)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor=c, edgecolor="#1f3a5f") for c in couleurs]
    ax.legend(handles, ordre_sous_groupe, loc="upper left", frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    plt.tight_layout()
    plt.show()


def graphe_effectif_et_moyenne(
    df: pd.DataFrame, colonne_categorielle: str, colonne_numerique: str, ordre: list[str], titre: str
) -> None:
    """Deux barres côte à côte (échelles différentes) : effectif par catégorie, moyenne de `colonne_numerique`."""
    stats = (
        df.groupby(colonne_categorielle)
        .agg(nb_clients=(colonne_categorielle, "count"), moyenne=(colonne_numerique, "mean"))
        .reindex(ordre)
    )

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.5))
    axes[0].bar(stats.index, stats["nb_clients"], color="#3b6fa0")
    for i, v in enumerate(stats["nb_clients"]):
        axes[0].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    axes[0].set_title("Nombre de clients")
    axes[0].set_ylabel("nb clients")

    axes[1].bar(stats.index, stats["moyenne"], color="#3b6fa0")
    for i, v in enumerate(stats["moyenne"]):
        axes[1].text(i, v, f"{v:.0f}", ha="center", va="bottom", fontsize=9)
    axes[1].set_title(f"{colonne_numerique} (moyenne)")
    axes[1].set_ylabel(f"{colonne_numerique} (moyenne)")

    for ax in axes:
        ax.set_ylim(top=ax.get_ylim()[1] * 1.1)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    fig.suptitle(titre)
    plt.tight_layout()
    plt.show()


def graphe_heatmap_crosstab(
    df: pd.DataFrame, colonne_lignes: str, colonne_colonnes: str, ordre_lignes: list[str], ordre_colonnes: list[str], titre: str
) -> None:
    """Heatmap du croisement `colonne_lignes` x `colonne_colonnes` (% en ligne), p-value/Cramér's V en titre."""
    table = pd.crosstab(df[colonne_lignes], df[colonne_colonnes]).reindex(index=ordre_lignes, columns=ordre_colonnes)
    prop = table.div(table.sum(axis=1), axis=0) * 100

    chi2, p_value, _, _ = chi2_contingency(table)
    n = table.to_numpy().sum()
    cramers_v = (chi2 / (n * (min(table.shape) - 1))) ** 0.5

    beaucoup_de_lignes = len(ordre_lignes) > 6
    largeur = max(6, 1.4 * len(ordre_colonnes))
    hauteur = max(4.5, 0.55 * len(ordre_lignes))
    fig, ax = plt.subplots(figsize=(largeur, hauteur))
    im = ax.imshow(prop.to_numpy(), cmap="Blues", vmin=0, vmax=100)
    ax.set_xticks(range(len(ordre_colonnes)))
    ax.set_xticklabels(ordre_colonnes)
    ax.set_yticks(range(len(ordre_lignes)))
    ax.set_yticklabels(ordre_lignes)
    for i in range(len(ordre_lignes)):
        for j in range(len(ordre_colonnes)):
            valeur = prop.iloc[i, j]
            couleur = "white" if valeur > 55 else "#1f3a5f"
            label = f"{valeur:.0f}%" if beaucoup_de_lignes else f"{valeur:.0f}%\n(n={table.iloc[i, j]})"
            ax.text(j, i, label, ha="center", va="center", fontsize=8, color=couleur)
    ax.set_xlabel(colonne_colonnes)
    ax.set_ylabel(colonne_lignes)
    ax.set_title(f"{titre} (% en ligne)\np-value={p_value:.1e}, Cramér's V={cramers_v:.3f}")
    cbar = fig.colorbar(im, ax=ax, shrink=0.8)
    cbar.set_label("% de la ligne")
    plt.tight_layout()
    plt.show()


def _euros(valeur: float) -> str:
    return f"{valeur:,.0f} €".replace(",", " ")


def graphe_camembert_somme(df: pd.DataFrame, colonne_categorielle: str, colonne_valeur: str, ordre: list[str], titre: str) -> None:
    """Camembert de la somme de `colonne_valeur` par `colonne_categorielle`, ordre fixé, dégradé séquentiel (une seule teinte)."""
    sommes = df.groupby(colonne_categorielle)[colonne_valeur].sum().reindex(ordre)
    couleurs = [plt.cm.Blues(x) for x in [0.35, 0.55, 0.75, 0.95][: len(ordre)]]

    fig, ax = plt.subplots(figsize=(6, 6))
    _, _, autotexts = ax.pie(
        sommes,
        labels=[f"{cat}\n{_euros(v)}" for cat, v in sommes.items()],
        autopct="%1.0f%%",
        colors=couleurs,
        startangle=90,
        counterclock=False,
        wedgeprops={"edgecolor": "white", "linewidth": 2},
    )
    for t in autotexts:
        t.set_color("white")
        t.set_fontsize(10)
    ax.set_title(f"{titre}\ntotal {_euros(sommes.sum())}")
    plt.tight_layout()
    plt.show()
