# ZOIDBERG2.0 – Computer Aided Diagnosis

## Description

ZOIDBERG2.0 est un projet de diagnostic médical assisté par ordinateur utilisant l'apprentissage automatique pour détecter la pneumonie à partir d'images de radiographies. Ce projet a été réalisé dans le cadre du module T-DEV-810.

## Objectifs

- Utiliser des images de rayons X pour aider les médecins à diagnostiquer la pneumonie.
- Explorer et tester différents algorithmes de machine learning.
- Comparer différentes approches de validation (train/test, validation croisée).
- Appliquer des techniques d’optimisation, d’ingénierie des caractéristiques, et de réduction de dimension (PCA).
- Fournir une synthèse claire des résultats avec des visualisations pertinentes.

## Données

Trois jeux de données ont été fournis par les enseignants. Leur utilisation (entraînement, validation, test, tuning) est laissée à la discrétion des développeurs.

Lien vers les datasets : [Datasets Sharepoint](https://epitechfr.sharepoint.com/:f:/r/sites/TDEV810/Documents%20partages/datasets?csf=1&e=3ghePT)

## Méthodologie

- **Procédures mises en place** :
  - Train / Validation / Test split
  - Validation croisée
  - Comparaison avec une simple séparation train/test
- **Évaluation** :
  - Utilisation de plusieurs métriques dont la courbe ROC-AUC
  - Comparaison visuelle des résultats
- **Exploration** :
  - Différentes méthodes de classification
  - Réglage de paramètres
  - Réduction de dimension

## Livrables

- 📓 Un fichier de type Jupyter Notebook contenant :
  - Code, textes explicatifs, graphiques
  - Version HTML pour visualisation sans exécution
- 📄 Un document de synthèse (PDF) avec résultats et figures

## Bonus (facultatif)

- Réseaux de neurones
- Prédiction multi-classe : pas de pneumonie / pneumonie virale / pneumonie bactérienne
- Cartes auto-organisatrices pour la visualisation

## Recommandations

- Gérer efficacement les ressources (temps, espace, complexité)
- Éviter le surapprentissage / sous-apprentissage via une bonne gestion du biais et de la variance
- Bien choisir les métriques d’évaluation pour présenter des résultats compréhensibles

---

**Version** : 2.2  
**Projet T-DEV-810 – Epitech**
