# Carte du gold `gold_churn_4ff8d380.parquet`

- Lignes : 5000
- Commit gold (code qui l'a généré) : `4ff8d380ae56241e05fdeec9028923ef4d67130d`
- Commit silver (données de base) : `4ff8d380ae56241e05fdeec9028923ef4d67130d` (build id=65, built_at=2026-09-30 14:43:35.764353+00:00)

## Colonnes

- `client_id` : 5000 valeurs distinctes
- `plan_normalized` : 1772x pro, 1367x starter, 1310x business, 551x enterprise
- `taille_entreprise` : 2068x PME, 1679x TPE, 893x ETI, 360x GE
- `commentaire_csm` : 2773x n/a, 339x RAS., 312x Suivi standard., 296x Compte dans la moyenne., 233x Client insatisfait, risque de départ., 224x Mécontentement exprimé au support., 212x Client très satisfait et actif., 195x Compte engagé, ambassadeur potentiel., 147x Faible adoption des sièges., 129x Déploiement interne limité., 44x Multiples tickets ouverts, friction élevée., 36x Support très sollicité récemment., 31x Usage en forte baisse ce trimestre., 29x Compte peu actif, relance nécessaire.
- `anciennete_mois` : moyenne=10.81, min=1, max=36
- `taux_adoption_pct` : moyenne=49.25, min=0.0, max=100.0
- `connexions_30j` : moyenne=23.87, min=0, max=156
- `heures_usage_30j` : moyenne=12.67, min=0.0, max=170.2
- `nb_integrations` : moyenne=2.07, min=0, max=16
- `derniere_connexion_jours` : moyenne=7.58, min=0, max=200
- `tickets_support_90j` : moyenne=2.54, min=0, max=17
- `delai_reponse_support_h` : moyenne=11.47, min=0.0, max=56.5
- `csat` : moyenne=3.45, min=1.0, max=5.0
- `retards_paiement_12m` : moyenne=0.74, min=0.0, max=7.0
- `sante_compte_fin_periode` : moyenne=46.52, min=0, max=100
- `taux_utilisation_fonctionnalites_pct` : moyenne=33.48, min=0.0, max=100.0
- `support_dedie_from_plan` : 3139x False, 1861x True
- `revenu_mensuel_recurrent_eur` : moyenne=3543.20, min=9.11, max=89402.85
- `churn` : 3600x False, 1400x True
