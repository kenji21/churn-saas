# Carte du modèle churn `modele_champion_0764d46b.joblib`

## Usage prévu

- Aide à la décision pour les CSM et le support : signaler les comptes à risque de résiliation à l'échéance, pour prioriser les appels et le traitement des tickets.
- Ne remplace pas la décision humaine : une alerte déclenche un contact, jamais une action automatique sur le compte (tarif, accès, contrat).

## Modèle

- Regression logistique (baseline) : `LogisticRegression`, sortie = probabilité de churn
- Seuil de décision : 0.279 (maximise le F1 sur la validation), alerte si probabilité >= seuil

## Données

- Train : 4000 clients, validation : 1000 clients (découpe 80/20 stratifiée sur churn, taux de churn 28.0%)
- Entrées (16 colonnes) : `plan_normalized`, `taille_entreprise`, `commentaire_csm`, `anciennete_mois`, `taux_adoption_pct`, `connexions_30j`, `heures_usage_30j`, `nb_integrations`, `derniere_connexion_jours`, `tickets_support_90j`, `delai_reponse_support_h`, `csat`, `retards_paiement_12m`, `taux_utilisation_fonctionnalites_pct`, `delai_support_x_non_adoption`, `support_dedie_from_plan`
- Exclues : `client_id` (identifiant, pas une feature) ; `sante_compte_fin_periode` (fuite -- separe quasi parfaitement les classes (PR-AUC 0.999 avec, 0.746 sans)) ; `revenu_mensuel_recurrent_eur` (revenu mensuel (MRR) : entree du modele valeur vie, pas du modele churn) ; `churn` (la cible elle-meme)

## Performance (validation)

| Métrique | Valeur |
|---|---|
| PR-AUC | 0.746 (hasard = 0.280) |
| AUC ROC | 0.873 |
| F1 au seuil | 0.697 |
| Churners détectés (rappel) | 229/280 (81.8%) |
| Alertes justifiées (précision) | 229/377 (60.7%) |
| Fausses alertes | 148 |

## Limites

- Plafond de performance lié à l'information disponible (courbe d'apprentissage plate dès 2 000 lignes), pas au modèle : une partie du churn dépend de facteurs absents des données (budget, concurrence, interlocuteur).
- Données en photo à un instant t, sans historique : pas de tendance d'usage.
- Validation de 1 000 clients : des écarts de +/- 0,02 entre modèles ne sont pas significatifs.
- Imputations à 0 de nb_integrations et delai_reponse_support_h discutables (cf data review).
- Biais d'intervention en exploitation : un client alerté puis retenu par le CSM ne doit pas être compté comme une erreur du modèle (groupe témoin nécessaire pour mesurer la performance réelle).

## Provenance

- Artefact : `data/modele_champion_0764d46b.joblib`
- Gold d'entraînement : `data/gold_churn_0764d46b.parquet` (carte `data/gold_churn_0764d46b.md`)
- Run MLflow : `logreg_baseline_0764d46b` (936024bd83714339bf6b93f07778d2e1), expérience `churn_saas`, 2026-10-02 13:05 UTC
- Commit du code à la génération de la carte : `90c39f7bc42860f6f89fe034c379a0d3d51c029a` (code commité)
