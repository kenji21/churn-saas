# Carte du modèle valeur vie client `modele_vie_0764d46b.joblib`

## Usage prévu

- Estimer la valeur vie client d'un compte, pour enrichir l'analyse métier : combinée à la probabilité de churn, elle donne la valeur à risque (P(churn) x valeur vie) et aide à prioriser les comptes à contacter.
- Jamais en entrée du modèle churn.

## Modèle

- Ridge, cible en log + log(MRR) : `Ridge` (alpha=1.0), entraîné sur log(1 + valeur vie), prédictions reconverties en euros
- Sortie : valeur vie client estimée, en euros (bornée à 0)

## Données

- Train : 4000 clients, validation : 1000 clients (même découpe que le modèle churn), valeur vie jointe depuis le silver
- Valeur vie (train) : médiane 11,640 €, moyenne 76,173 €, de 300 à 2,000,000 €
- Entrées (17 colonnes) : `plan_normalized`, `taille_entreprise`, `commentaire_csm`, `anciennete_mois`, `taux_adoption_pct`, `connexions_30j`, `heures_usage_30j`, `nb_integrations`, `derniere_connexion_jours`, `tickets_support_90j`, `delai_reponse_support_h`, `csat`, `retards_paiement_12m`, `taux_utilisation_fonctionnalites_pct`, `delai_support_x_non_adoption`, `support_dedie_from_plan`, `revenu_mensuel_recurrent_eur`
- Exclues : `churn` (inconnu au moment de la prédiction, et sans apport mesuré) ; `sante_compte_fin_periode` (calculée en fin de période, après coup) ; `valeur_vie_client_eur` (la cible elle-même) ; `client_id` (identifiant, pas une feature)

## Performance (validation)

| Modèle | MAE (€) | RMSE (€) | R² (€) | R² (log) |
|---|---|---|---|---|
| **Ridge, cible en log + log(MRR)** | 35,419 | 129,071 | 0.594 | 0.862 |
| Prédire la moyenne du train | 97,046 | 202,659 | -0.001 | -0.981 |
| Prédire la médiane du train | 67,240 | 210,891 | -0.084 | -0.002 |

## Limites

- Cible très étalée (300 à 2 000 000 €) : le modèle classe bien les comptes (R² en log élevé) mais l'erreur absolue reste forte sur les très gros comptes (R² en euros plus faible).
- Valeur vie fournie calculée après coup (elle diverge de MRR x ancienneté surtout chez les clients partis) : la cible elle-même est imparfaite.
- Mêmes limites de données que le modèle churn : photo à un instant t, imputations à 0 discutables.

## Provenance

- Artefact : `data/modele_vie_0764d46b.joblib`
- Gold d'entraînement : `data/gold_churn_0764d46b.parquet` (carte `data/gold_churn_0764d46b.md`)
- Run MLflow : `vie_ridge_cible_log_mrr_0764d46b` (f0e8e20adfaf450cac6c291b44475a49), expérience `churn_vie`, 2026-10-04 07:00 UTC
- Commit du code à la génération de la carte : `90c39f7bc42860f6f89fe034c379a0d3d51c029a` (code commité)
