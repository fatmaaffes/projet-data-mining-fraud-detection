# Détection de fraude e-commerce explicable

Projet académique de Data Mining — ENSI 2026–2027.

Cette application compare une approche supervisée et deux approches non supervisées pour détecter des transactions e-commerce suspectes, fixe un seuil d'alerte selon un coût métier et explique les alertes avec SHAP.

## Dataset

Fraudulent E-Commerce Transactions — dataset synthétique du catalogue du Challenge 2.

Source Kaggle : https://www.kaggle.com/datasets/shriyashjagtap/fraudulent-e-commerce-transactions

Fichier utilisé : Fraudulent_E-Commerce_Transaction_Data_2.csv, environ 23 634 transactions. La cible est Is Fraudulent, renommée is_fraud. Le fichier est chargé automatiquement depuis un miroir GitHub public afin de faciliter le déploiement Streamlit.

## Pipeline

1. EDA : taux de fraude, montants, temps et catégories.
2. Feature engineering sans fuite : heure cyclique, jour de la semaine, vélocité client sur 1 h et 24 h, montant moyen passé sur 30 jours, ratio montant courant / moyenne passée, temps depuis la transaction précédente et cohérence livraison / facturation.
3. Split chronologique : 60 % train, 20 % validation, 20 % test.
4. Modèles : Régression logistique, Isolation Forest et Local Outlier Factor.
5. Gestion du déséquilibre : comparaison de la régression logistique sans pondération et avec class_weight=balanced sur la validation.
6. Évaluation : PR-AUC, ROC-AUC, précision@k, rappel@k, F2, MCC et coût métier.
7. Le seuil est choisi sur validation en minimisant : montants des fraudes manquées + coût d'analyse × nombre d'alertes. Le seuil est ensuite appliqué au test sans réoptimisation.
8. XAI : SHAP beeswarm global, SHAP waterfall local et 3 à 5 codes motifs.
9. API FastAPI : /health, /predict et /explain.

## Lancer l'application

python -m venv .venv

Sous Windows :
.venv\Scripts\activate

Puis :
pip install -r requirements.txt
streamlit run app/app.py

## Lancer l'API

uvicorn api.main:app --reload

Swagger : http://127.0.0.1:8000/docs

## Architecture

app/app.py  
api/main.py  
src/data.py  
src/features.py  
src/modeling.py  
src/evaluation.py  
src/explainability.py  
requirements.txt  
.streamlit/config.toml

## Point méthodologique important

Validation et test conservent toujours la distribution réelle des fraudes. Les décisions de stratégie et de seuil sont prises sur la validation, jamais sur le test.
