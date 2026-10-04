---
paths:
  - "**/*.ipynb"
---

# Notebooks Jupyter (.ipynb)

- Toujours relire le `.ipynb` concerné avec l'outil Read juste avant de le lire, l'éditer ou en discuter le contenu (cellules, `cell_id`, sorties) — même s'il a déjà été lu plus tôt dans la conversation. Le kernel Jupyter (exécution, cellule vide auto-ajoutée, `execution_count`) ou VS Code (édition manuelle, autosave) peuvent l'avoir modifié entre deux tours. Ne jamais se fier à un `cell_id` ou à un contenu mémorisé sans le revérifier contre l'état actuel du fichier sur disque.
- NotebookEdit écrit directement sur disque, pas via l'UI de VS Code — aucun outil disponible ici ne peut déclencher un "Enregistrer" natif de VS Code. Après une modification, si le notebook est ouvert dans l'éditeur, prévenir que VS Code peut afficher un état "modifié sur le disque" et qu'il faut recharger/enregistrer depuis l'UI (choisir la version disque, pas fusionner à la main sauf conflit réel constaté).
- Ne jamais éditer/ajouter/supprimer une cellule (markdown ou code) de `churn_saas_certification.ipynb` sans demande explicite de l'utilisateur — c'est sa narration personnelle pour la soutenance de certification, pas un document co-écrit. Si son contenu (tableau de décisions, etc.) devient incohérent avec du code changé par ailleurs, le signaler dans le chat et laisser l'utilisateur corriger lui-même, sans y toucher.
- Dans les réponses, désigner une cellule par son titre markdown, sa section ou une ligne distinctive de son code (ex. « sous le titre Baselines V2, section 8 »), jamais par son index : VS Code n'affiche pas les numéros de cellule, et ils se décalent dès que l'utilisateur en insère. Dans les scripts d'édition, repérer aussi la cellule par son contenu plutôt que par son index.
