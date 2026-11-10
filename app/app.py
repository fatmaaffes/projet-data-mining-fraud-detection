"""Streamlit application for explainable e-commerce fraud detection."""

from __future__ import annotations

import sys
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import plotly.express as px
import requests
import streamlit as st
from sklearn.metrics import precision_recall_curve

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from src.data import KAGGLE_SOURCE, fraud_summary, load_fraud_data, missing_summary
from src.evaluation import (
    business_cost,
    comparison_table,
    metrics_at_threshold,
    optimize_threshold,
    precision_recall_at_k,
)
from src.explainability import compute_shap, top_reason_codes
from src.features import build_features, chronological_split
from src.modeling import train_models


st.set_page_config(
    page_title="Détection de fraude explicable",
    layout="wide",
)

MODEL_COLORS = {
    "Régression logistique": "#7A3E48",
    "Isolation Forest": "#2F6F6D",
    "LOF": "#C97B63",
}

px.defaults.template = "plotly_white"
px.defaults.color_discrete_sequence = list(MODEL_COLORS.values())

st.markdown(
    """
<style>
.stApp {
    background: #FAF7F2;
    color: #1F2933;
}
.block-container {
    padding-top: 1.6rem;
}
[data-testid="stSidebar"] {
    background: #F1ECE5;
    border-right: 1px solid #DED6CD;
}
div[data-testid="stMetric"] {
    background: #FFFFFF;
    border: 1px solid #DED6CD;
    border-radius: 12px;
    padding: 0.75rem 0.9rem;
}
[data-testid="stExpander"] {
    background: #FFFFFF;
    border: 1px solid #DED6CD;
    border-radius: 10px;
}
h1, h2, h3, h4 {
    color: #55313A;
}
.stButton > button {
    background: #7A3E48;
    color: white;
    border: 1px solid #7A3E48;
    border-radius: 9px;
    font-weight: 600;
}
.stButton > button:hover {
    background: #63323A;
    border-color: #63323A;
    color: white;
}
[data-baseweb="tab-list"] {
    border-bottom: 1px solid #DED6CD;
}
button[role="tab"][aria-selected="true"] {
    color: #7A3E48 !important;
    border-bottom-color: #C97B63 !important;
}
</style>
""",
    unsafe_allow_html=True,
)


@st.cache_data(show_spinner=False)
def get_data():
    return load_fraud_data()


@st.cache_data(show_spinner=False)
def get_featured_data():
    return build_features(get_data())


@st.cache_data(show_spinner=False)
def get_split():
    return chronological_split(get_featured_data())


@st.cache_resource(show_spinner=False)
def get_models():
    return train_models(get_split())


def section_intro(goal: str, content: str, business: str) -> None:
    st.subheader("But du module")
    st.info(goal)
    st.subheader("Ce qu'on va traiter")
    st.write(content)
    st.subheader("Pourquoi c'est important pour la détection de fraude")
    st.success(business)


def interpretation(text: str) -> None:
    st.markdown("#### Interprétation")
    st.info(text)


df = get_data()
featured = get_featured_data()
split = get_split()
summary = fraud_summary(df)

st.title("Détection de fraude e-commerce explicable")
st.caption(
    "Data Mining · approche supervisée + détection d'anomalies · seuil métier · XAI"
)

st.success(
    f"Dataset chargé : {summary['transactions']:,} transactions · "
    f"{summary['frauds']:,} fraudes · taux de fraude {summary['fraud_rate']:.2%}"
)

module = st.sidebar.radio(
    "Navigation",
    [
        "1 · Données & EDA",
        "2 · Préparation & variables",
        "3 · Modélisation",
        "4 · Évaluation & seuil",
        "5 · Explainable AI",
    ],
)

st.sidebar.caption("Classe positive : is_fraud = 1")
st.sidebar.caption("Dataset : Fraudulent E-Commerce Transactions")


if module == "1 · Données & EDA":
    st.header("Module 1 — Données & EDA")
    section_intro(
        "Comprendre la rareté de la fraude, les montants et les comportements avant toute modélisation.",
        "Aperçu, qualité des données, taux de fraude, montants, évolution temporelle et catégories disponibles.",
        "En fraude, l'accuracy peut être trompeuse : il faut d'abord mesurer à quel point la classe frauduleuse est rare et comment elle se répartit.",
    )

    tab_overview, tab_target, tab_time, tab_categories = st.tabs(
        ["Aperçu", "Fraude & montants", "Temps", "Catégories"]
    )

    with tab_overview:
        st.subheader("Aperçu des transactions")
        st.dataframe(df.head(10), width="stretch")

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{summary['transactions']:,}")
        c2.metric("Fraudes", f"{summary['frauds']:,}")
        c3.metric("Taux de fraude", f"{summary['fraud_rate']:.2%}")
        c4.metric("Montant frauduleux", f"{summary['fraud_amount']:,.0f}")

        st.subheader("Valeurs manquantes")
        st.dataframe(
            missing_summary(df),
            width="stretch",
            hide_index=True,
        )
        interpretation(
            f"La fraude représente {summary['fraud_rate']:.2%} des transactions. "
            "La classe positive est minoritaire : une accuracy élevée ne suffit donc pas pour juger le détecteur."
        )

    with tab_target:
        class_counts = (
            df["is_fraud"]
            .value_counts()
            .rename_axis("classe")
            .reset_index(name="effectif")
        )
        class_counts["libelle"] = class_counts["classe"].map(
            {0: "Transaction normale", 1: "Fraude"}
        )
        fig = px.bar(
            class_counts,
            x="libelle",
            y="effectif",
            text="effectif",
            title="Répartition normale / fraude",
        )
        st.plotly_chart(fig, width="stretch")

        amount_box = df.copy()
        amount_box["classe"] = amount_box["is_fraud"].map(
            {0: "Normale", 1: "Fraude"}
        )
        fig_amount = px.box(
            amount_box,
            x="classe",
            y="transaction_amount",
            points=False,
            title="Montants selon la classe",
        )
        st.plotly_chart(fig_amount, width="stretch")

        grouped_amounts = (
            df.groupby("is_fraud")["transaction_amount"]
            .agg(["count", "mean", "median", "sum"])
            .rename(index={0: "Normales", 1: "Fraudes"})
        )
        st.dataframe(grouped_amounts, width="stretch")

    with tab_time:
        temp = df.copy()
        temp["hour"] = temp["transaction_date"].dt.hour
        hourly = (
            temp.groupby("hour")["is_fraud"]
            .agg(["mean", "count"])
            .reset_index()
        )
        hourly["fraud_rate_pct"] = 100 * hourly["mean"]

        fig_hour = px.line(
            hourly,
            x="hour",
            y="fraud_rate_pct",
            markers=True,
            title="Taux de fraude par heure",
        )
        fig_hour.update_layout(
            xaxis_title="Heure",
            yaxis_title="Taux de fraude (%)",
        )
        st.plotly_chart(fig_hour, width="stretch")

        daily = (
            temp.set_index("transaction_date")["is_fraud"]
            .resample("D")
            .agg(["mean", "count"])
            .reset_index()
        )
        daily["fraud_rate_pct"] = 100 * daily["mean"]
        fig_daily = px.line(
            daily,
            x="transaction_date",
            y="fraud_rate_pct",
            title="Évolution du taux de fraude dans le temps",
        )
        st.plotly_chart(fig_daily, width="stretch")

        interpretation(
            "La dimension temporelle est conservée jusqu'à la validation : les premières transactions servent à l'entraînement et les plus récentes à la validation puis au test."
        )

    with tab_categories:
        categorical_candidates = [
            column
            for column in [
                "payment_method",
                "product_category",
                "device_used",
                "customer_location",
            ]
            if column in df.columns
        ]
        category = st.selectbox(
            "Variable catégorielle",
            categorical_candidates,
        )
        grouped = (
            df.groupby(category)["is_fraud"]
            .agg(["mean", "count"])
            .reset_index()
            .sort_values("count", ascending=False)
            .head(20)
        )
        grouped["fraud_rate_pct"] = 100 * grouped["mean"]

        fig_cat = px.bar(
            grouped,
            x=category,
            y="fraud_rate_pct",
            hover_data=["count"],
            title=f"Taux de fraude selon {category}",
        )
        st.plotly_chart(fig_cat, width="stretch")


elif module == "2 · Préparation & variables":
    st.header("Module 2 — Préparation & variables")
    section_intro(
        "Transformer les transactions brutes en variables exploitables, sans utiliser d'information future.",
        "Feature Engineering, choix des variables, séparation chronologique et pipeline d'imputation, de standardisation et de One-Hot Encoding.",
        "Ces étapes rendent les données utilisables par les modèles et évitent d'apprendre sur des informations qui ne seraient pas disponibles au moment d'une transaction.",
    )

    st.subheader("1 · Feature Engineering — nouvelles variables")
    st.write(
        "**Feature Engineering** signifie créer des indicateurs utiles à partir des colonnes existantes. "
        "Ce n'est pas encore le One-Hot Encoding : celui-ci fait partie du prétraitement."
    )
    created_features = [column for column in featured.columns if column not in df.columns]
    initial_count = len(df.columns)
    engineered_count = len(featured.columns)

    metric_initial, metric_added, metric_total = st.columns(3)
    metric_initial.metric("Colonnes initiales", initial_count)
    metric_added.metric("Nouvelles colonnes", len(created_features))
    metric_total.metric("Colonnes après création", engineered_count)

    feature_explanations = {
        "hour": (
            "Extraire l'heure (0 à 23) à partir de transaction_date.",
            "Étudier les comportements selon l'heure de l'achat.",
        ),
        "day_of_week": (
            "Extraire le jour de la semaine (lundi = 0, dimanche = 6).",
            "Étudier les différences entre jours de la semaine.",
        ),
        "is_weekend": (
            "Mettre 1 pour samedi/dimanche, sinon 0.",
            "Distinguer les achats du week-end.",
        ),
        "hour_sin": (
            "Transformer l'heure par un sinus, selon un cycle de 24 h.",
            "Représenter la proximité entre 23 h et 0 h.",
        ),
        "hour_cos": (
            "Transformer l'heure par un cosinus, selon un cycle de 24 h.",
            "Compléter le sinus pour représenter correctement l'heure cyclique.",
        ),
        "shipping_billing_match": (
            "Mettre 1 si l'adresse de livraison égale l'adresse de facturation, sinon 0.",
            "Signaler une différence possible entre les deux adresses, sans conclure à une fraude.",
        ),
        "customer_tx_count_1h": (
            "Compter les achats précédents du client sur 1 h.",
            "Repérer une fréquence d'achats inhabituelle.",
        ),
        "customer_tx_count_24h": (
            "Compter les achats précédents du client sur 24 h.",
            "Repérer une activité élevée sur une journée.",
        ),
        "customer_amount_mean_30d_past": (
            "Calculer le montant moyen passé du client sur 30 jours.",
            "Définir son niveau habituel de dépenses.",
        ),
        "amount_vs_30d_mean": (
            "Diviser le montant actuel par sa moyenne passée sur 30 jours.",
            "Identifier un achat nettement supérieur au profil passé.",
        ),
        "hours_since_previous_tx": (
            "Calculer le nombre d'heures depuis l'achat précédent.",
            "Repérer des achats très rapprochés.",
        ),
    }
    feature_table = pd.DataFrame(
        [
            {
                "Variable créée": column,
                "Calcul effectué": feature_explanations[column][0],
                "Pourquoi c'est utile": feature_explanations[column][1],
            }
            for column in created_features
            if column in feature_explanations
        ]
    )
    st.dataframe(feature_table, width="stretch", hide_index=True)
    interpretation(
        f"Le dataset passe de {initial_count} à {engineered_count} colonnes : "
        f"{len(created_features)} nouvelles variables ont réellement été créées. "
        "Les indicateurs temporels décrivent le moment de l'achat ; la comparaison des adresses "
        "apporte un signal supplémentaire. Aucun de ces signaux ne prouve une fraude à lui seul."
    )

    behavioral_columns = [
        "customer_tx_count_1h",
        "customer_tx_count_24h",
        "customer_amount_mean_30d_past",
        "amount_vs_30d_mean",
        "hours_since_previous_tx",
    ]
    if all(column in featured.columns for column in behavioral_columns):
        st.info(
            "L'historique client est disponible : les variables comportementales ont été calculées "
            "uniquement avec les transactions antérieures de chaque client."
        )
    else:
        unique_ratio = df["customer_id"].nunique() / len(df)
        st.warning(
            f"Limite du dataset : {unique_ratio:.1%} des identifiants clients sont uniques. "
            "Aucune variable d'historique client (achats sur 1 h / 24 h, moyenne sur 30 jours, "
            "temps depuis le dernier achat) n'a été ajoutée au modèle. "
            "Il serait incorrect de présenter ces variables comme déjà calculées."
        )

    with st.expander("Voir les valeurs des variables créées (avant encodage)"):
        visible_created = [
            column for column in [
                "transaction_amount", "hour", "day_of_week", "is_weekend",
                "hour_sin", "hour_cos", "shipping_billing_match",
                "customer_tx_count_1h", "amount_vs_30d_mean",
            ]
            if column in featured.columns
        ]
        st.dataframe(featured[visible_created].head(8), width="stretch")
        st.caption(
            "Exemple : is_weekend = 1 indique un week-end ; shipping_billing_match = 0 "
            "indique des adresses différentes. hour_sin et hour_cos sont des nombres "
            "qui représentent l'heure sur un cycle de 24 heures."
        )

    st.subheader("2 · Sélection — quelles variables entrent dans le modèle ?")
    st.write(
        "Toutes les colonnes du dataset ne sont pas utilisées directement. Certaines servent à "
        "construire de nouvelles variables, d'autres sont des identifiants, et is_fraud est la cible à prédire."
    )
    exclusions = [
        (
            ["transaction_id", "customer_id"],
            "Identifiants : ils ne décrivent pas directement un comportement généralisable. "
            "customer_id sert seulement à reconstruire un historique lorsqu'il existe.",
        ),
        (
            ["transaction_date", "transaction_hour"],
            "La date sert à créer des variables temporelles et au découpage ; "
            "l'heure est recalculée à partir de transaction_date.",
        ),
        (
            ["shipping_address", "billing_address"],
            "Les adresses brutes sont écartées ; on conserve leur indicateur de correspondance.",
        ),
        (
            ["ip_address", "customer_location"],
            "Champs textuels ou à forte cardinalité non encodés dans ce modèle.",
        ),
        (
            ["is_fraud"],
            "Étiquette réelle 0/1 : c'est la valeur à prédire, jamais une entrée du modèle.",
        ),
    ]
    present_exclusions = []
    for columns, reason in exclusions:
        actual = [column for column in columns if column in featured.columns]
        if actual:
            present_exclusions.append(
                {"Colonnes écartées": ", ".join(actual), "Raison": reason}
            )
    st.dataframe(
        pd.DataFrame(present_exclusions),
        width="stretch",
        hide_index=True,
    )
    st.write("**Variables retenues pour l'apprentissage :**")
    selected_types = pd.DataFrame(
        [
            {
                "Type": "Numériques",
                "Nombre": len(split.numeric_features),
                "Variables": ", ".join(split.numeric_features) or "Aucune",
            },
            {
                "Type": "Catégorielles",
                "Nombre": len(split.categorical_features),
                "Variables": ", ".join(split.categorical_features) or "Aucune",
            },
        ]
    )
    st.dataframe(selected_types, width="stretch", hide_index=True)
    interpretation(
        f"Parmi les {engineered_count} colonnes disponibles après création, "
        f"{split.X_train.shape[1]} sont retenues comme entrées du modèle : "
        f"{len(split.numeric_features)} numériques et "
        f"{len(split.categorical_features)} catégorielles. "
        "La cible is_fraud reste séparée des entrées pour éviter toute fuite de réponse."
    )

    st.subheader("3 · Séparation chronologique — Train / Validation / Test")
    st.write(
        "On trie les transactions par date, puis on garde les plus anciennes pour apprendre "
        "et les plus récentes pour évaluer le modèle sur un futur simulé."
    )
    split_rows = []
    for label, part in [
        ("Train — 60 %", split.train_raw),
        ("Validation — 20 %", split.validation_raw),
        ("Test — 20 %", split.test_raw),
    ]:
        split_rows.append(
            {
                "Jeu": label,
                "Transactions": len(part),
                "Normales (0)": int(part["is_fraud"].eq(0).sum()),
                "Fraudes (1)": int(part["is_fraud"].eq(1).sum()),
                "Début": part["transaction_date"].min(),
                "Fin": part["transaction_date"].max(),
                "Taux fraude": part["is_fraud"].mean(),
            }
        )
    split_table = pd.DataFrame(split_rows)
    st.dataframe(
        split_table.style.format({"Taux fraude": "{:.2%}"}),
        width="stretch",
        hide_index=True,
    )
    interpretation(
        f"Le train contient {len(split.train_raw):,} transactions pour apprendre ; "
        f"la validation contient {len(split.validation_raw):,} transactions pour choisir "
        f"la stratégie et le seuil ; le test contient {len(split.test_raw):,} transactions "
        "pour mesurer la performance finale. On ne règle jamais le modèle sur le test."
    )

    st.markdown("#### Pourquoi ne pas utiliser `stratify=y` comme dans le projet 1 ?")
    st.write(
        "**Projet 1 — Credit Scoring :** le découpage Train / Test est aléatoire et "
        "utilise `stratify=y`. Cela conserve approximativement dans chaque ensemble "
        "la même proportion de bons et de mauvais payeurs que dans le dataset initial."
    )
    st.write(
        "**Projet 2 — Détection de fraude :** le découpage Train / Validation / Test "
        "respecte l'ordre des dates, **sans `stratify`**. "
        "Chaque ensemble contient des transactions normales (`is_fraud = 0`) "
        "et frauduleuses (`is_fraud = 1`), mais les proportions de fraude "
        "peuvent varier naturellement d'une période à l'autre."
    )
    interpretation(
        "On ne met pas toutes les fraudes d'un côté et toutes les transactions "
        "normales de l'autre : les colonnes « Normales (0) » et « Fraudes (1) » "
        "du tableau montrent la répartition réelle de chaque ensemble. "
        "Le tri chronologique simule le cas métier : apprendre sur le passé "
        "pour détecter des fraudes futures, sans utiliser de transactions plus récentes "
        "pour entraîner le modèle."
    )

    st.subheader("4 · Prétraitement — rendre les données utilisables")
    st.write(
        "Un ColumnTransformer applique automatiquement un traitement différent selon le type "
        "de variable. Les étapes sont apprises sur le train et réutilisées sans réajustement "
        "sur la validation et le test."
    )
    preprocessing_table = pd.DataFrame(
        [
            {
                "Variables": "Numériques",
                "Étape": "SimpleImputer — médiane",
                "Ce que cela fait": "Remplace une valeur manquante par la médiane calculée sur le train.",
                "Pourquoi": "Éviter de perdre une transaction si une mesure manque.",
            },
            {
                "Variables": "Numériques",
                "Étape": "StandardScaler",
                "Ce que cela fait": "Recentre et réduit chaque variable à partir du train.",
                "Pourquoi": "Éviter qu'une variable à grande échelle domine les autres.",
            },
            {
                "Variables": "Catégorielles",
                "Étape": "SimpleImputer — mode",
                "Ce que cela fait": "Remplace une catégorie manquante par la plus fréquente du train.",
                "Pourquoi": "Garder une catégorie exploitable même si une valeur manque.",
            },
            {
                "Variables": "Catégorielles",
                "Étape": "OneHotEncoder",
                "Ce que cela fait": "Crée une colonne 0/1 pour chaque modalité observée sur le train.",
                "Pourquoi": "Donner aux modèles des nombres sans inventer d'ordre entre les catégories.",
            },
        ]
    )
    st.dataframe(preprocessing_table, width="stretch", hide_index=True)
    missing_train = int(split.X_train.isna().sum().sum())
    if missing_train:
        st.caption(
            f"Dans les entrées du train, {missing_train} valeurs manquantes sont présentes "
            "et seront traitées par les imputations."
        )
    else:
        st.caption(
            "Dans les entrées du train, aucune valeur ne manque actuellement : "
            "les imputations sont prévues dans le pipeline, mais n'ont rien à remplacer ici."
        )
    interpretation(
        "Les données numériques sont mises sur une échelle comparable. Les catégories, "
        "comme le moyen de paiement ou le type d'appareil, deviennent des indicateurs 0/1. "
        "Le prétraitement est ajusté uniquement sur les transactions d'entraînement "
        "pour éviter la fuite d'informations du futur."
    )

    st.subheader("5 · Résultat final — avant / après transformation")
    from src.modeling import build_preprocessor

    demonstration_preprocessor = build_preprocessor(
        split.numeric_features, split.categorical_features
    )
    demonstration_preprocessor.fit(split.X_train)
    output_names = demonstration_preprocessor.get_feature_names_out().tolist()
    encoded_names = [
        name for name in output_names if name not in split.numeric_features
    ]
    before_after_a, before_after_b, before_after_c = st.columns(3)
    before_after_a.metric("Colonnes initiales", len(df.columns))
    before_after_b.metric("Variables sélectionnées", split.X_train.shape[1])
    before_after_c.metric("Variables après encodage", len(output_names))

    st.write(
        f"Au total, le modèle reçoit **{len(output_names)} variables numériques** "
        f"après imputation, standardisation et One-Hot Encoding, "
        f"dont **{len(encoded_names)} colonnes issues de l'encodage catégoriel**."
    )

    with st.expander("Voir les colonnes finales et un aperçu des données transformées"):
        st.write("**Noms des variables effectivement fournies aux modèles :**")
        st.dataframe(
            pd.DataFrame({"Variable après transformation": output_names}),
            width="stretch",
            hide_index=True,
        )
        transformed_sample = demonstration_preprocessor.transform(
            split.X_train.head(6)
        )
        transformed_preview = pd.DataFrame(
            transformed_sample, columns=output_names
        )
        preview_limit = min(18, len(output_names))
        st.write(
            f"**Six transactions du train après préparation** "
            f"(aperçu des {preview_limit} premières colonnes) :"
        )
        st.dataframe(
            transformed_preview.iloc[:, :preview_limit].round(3),
            width="stretch",
            hide_index=True,
        )
        st.caption(
            "Les valeurs standardisées peuvent être négatives ; "
            "les colonnes One-Hot sont des indicateurs 0/1. "
            "Ce sont les valeurs destinées aux algorithmes, pas les transactions brutes."
        )

    interpretation(
        f"Le parcours est maintenant vérifiable : {len(df.columns)} colonnes au départ, "
        f"{engineered_count} après Feature Engineering, "
        f"{split.X_train.shape[1]} variables retenues, puis "
        f"{len(output_names)} variables numériques en sortie du prétraitement. "
        "Les catégories sont remplacées par leurs colonnes 0/1. "
        "Les modèles entraînés utilisent ce même pipeline de préparation."
    )


elif module == "3 · Modélisation":
    st.header("Module 3 — Modélisation")
    section_intro(
        "Comparer une approche supervisée à deux approches de détection d'anomalies.",
        "Régression logistique, Isolation Forest et Local Outlier Factor (LOF).",
        "Le supervisé apprend les fraudes déjà connues ; les méthodes non supervisées recherchent les comportements inhabituels sans utiliser les labels pendant leur apprentissage.",
    )

    methods = pd.DataFrame(
        [
            ["Régression logistique", "Supervisée", "Apprend avec is_fraud.", "Probabilité de fraude."],
            ["Isolation Forest", "Non supervisée", "Une anomalie est plus facile à isoler par des coupures aléatoires.", "Score d'anomalie."],
            ["LOF", "Non supervisée", "Compare la densité locale d'une transaction à celle de ses voisins.", "Score d'anomalie local."],
        ],
        columns=["Méthode", "Famille", "Principe", "Sortie"],
    )
    st.dataframe(methods, width="stretch", hide_index=True)

    if st.button("Entraîner les 3 méthodes", type="primary"):
        with st.spinner("Entraînement en cours..."):
            results = get_models()
        st.session_state["fraud_results"] = results
        st.success("Entraînement terminé.")

    if "fraud_results" in st.session_state:
        results = st.session_state["fraud_results"]
        supervised = results["Régression logistique"]

        st.subheader("Stratégie contre le déséquilibre")
        st.write(
            "**Pourquoi comparer deux versions ?** Seulement 5,17 % environ des transactions "
            "sont frauduleuses : la classe fraude est minoritaire. On veut vérifier si "
            "lui donner davantage de poids aide vraiment le modèle."
        )
        st.markdown(
            "- **Sans pondération** : les transactions normales et frauduleuses "
            "ont le même poids individuel pendant l'entraînement.\n"
            "- **Avec `class_weight='balanced'`** : une transaction frauduleuse "
            "reçoit plus de poids, car elle appartient à la classe minoritaire. "
            "Cela oblige le modèle à davantage tenir compte des fraudes, "
            "mais ne garantit pas de meilleurs résultats."
        )

        st.markdown("#### Ce que nous avons fait, dans l'ordre")
        training_steps = pd.DataFrame(
            [
                {
                    "Étape": "1 · Entraînement (Train 60 %)",
                    "Action": "Entraîner deux régressions logistiques (sans / avec pondération), Isolation Forest et LOF. Les modèles n'apprennent que sur le train.",
                },
                {
                    "Étape": "2 · Choix de la régression (Validation 20 %)",
                    "Action": "Comparer les deux régressions avec la PR-AUC : garder celle qui classe le mieux les fraudes sur la validation, sans réentraîner.",
                },
                {
                    "Étape": "3 · Réglage du seuil (Validation 20 %, Module 4)",
                    "Action": "Tester plusieurs seuils d'alerte et choisir celui qui minimise le coût des fraudes manquées et des vérifications.",
                },
                {
                    "Étape": "4 · Évaluation finale (Test 20 %, Module 4)",
                    "Action": "Appliquer le modèle et le seuil déjà choisis aux transactions les plus récentes, sans nouveau réglage.",
                },
            ]
        )
        st.dataframe(training_steps, width="stretch", hide_index=True)
        st.info(
            "**À ne pas confondre :** entraîner = apprendre les paramètres du modèle sur le **Train** ; "
            "valider = comparer les solutions sur la **Validation** ; "
            "régler (ou « calibrer ») le seuil = choisir à partir de quel score on déclenche "
            "une alerte. Dans ce projet, nous ne faisons **pas** de calibration des probabilités "
            "(Platt / isotonic) : nous réglons seulement le **seuil de décision**."
        )
        st.caption(
            "La validation ne sert pas à réentraîner le modèle. Les trois méthodes seront "
            "évaluées sur le test dans le Module 4, avec leurs seuils fixés sur la validation."
        )

        # Les valeurs affichées proviennent du vrai entraînement, pas de chiffres fixes.
        chosen_strategy = supervised.strategy.split(
            " retenue sur validation", 1
        )[0]
        comparison_part = supervised.strategy.split("Comparaison : ", 1)
        strategy_rows = []
        if len(comparison_part) == 2:
            for entry in comparison_part[1].split(" ; "):
                if "=" in entry:
                    label, raw_score = entry.rsplit("=", 1)
                    try:
                        score = float(raw_score)
                    except ValueError:
                        continue
                    strategy_rows.append(
                        {
                            "Version": label,
                            "PR-AUC validation": score,
                            "Décision": "Retenue" if label == chosen_strategy else "Non retenue",
                        }
                    )

        st.markdown("#### Résultat de la comparaison")
        if strategy_rows:
            st.dataframe(
                pd.DataFrame(strategy_rows).style.format(
                    {"PR-AUC validation": "{:.4f}"}
                ),
                width="stretch",
                hide_index=True,
            )
            chosen_row = next(
                (row for row in strategy_rows if row["Décision"] == "Retenue"),
                None,
            )
            if chosen_row is not None:
                st.success(
                    f"**Version retenue : {chosen_strategy}**, car sa PR-AUC "
                    f"sur validation ({chosen_row['PR-AUC validation']:.4f}) "
                    "est la plus élevée des deux."
                )
            if len(strategy_rows) == 2:
                score_difference = abs(
                    strategy_rows[0]["PR-AUC validation"]
                    - strategy_rows[1]["PR-AUC validation"]
                )
                interpretation(
                    "Nous avons testé la pondération, sans la choisir automatiquement. "
                    "La PR-AUC mesure ici la qualité du classement des fraudes : "
                    "plus elle est élevée, meilleur est le résultat selon ce critère. "
                    f"L'écart observé est de {score_difference:.4f}, "
                    "donc un petit avantage sur cette validation, "
                    "pas une supériorité garantie dans tous les cas. "
                    "La PR-AUC ne représente pas l'accuracy."
                )
        else:
            st.write(supervised.strategy)
            interpretation(
                "Le choix se fait selon la meilleure PR-AUC sur la validation, "
                "sans réentraîner sur celle-ci. La PR-AUC n'est pas l'accuracy."
            )


        result_rows = [
            {
                "Méthode": name,
                "Type": "Supervisée" if result.supervised else "Non supervisée",
                "Stratégie": result.strategy,
            }
            for name, result in results.items()
        ]
        st.dataframe(
            pd.DataFrame(result_rows),
            width="stretch",
            hide_index=True,
        )


elif module == "4 · Évaluation & seuil":
    st.header("Module 4 — Évaluation & seuil")
    section_intro(
        "Évaluer correctement un événement rare et choisir un seuil d'alerte selon un coût métier.",
        "PR-AUC, ROC-AUC, précision@k, rappel@k, F2, MCC, courbes précision-rappel, coût et seuil.",
        "Une équipe fraude ne peut analyser qu'un nombre limité d'alertes, et manquer une fraude coûte son montant. Le seuil doit refléter cette réalité.",
    )

    if "fraud_results" not in st.session_state:
        st.warning("Entraînez d'abord les modèles dans le Module 3.")
    else:
        results = st.session_state["fraud_results"]

        st.markdown("### Comment le seuil d'alerte est-il choisi ?")
        st.write(
            "**Un score ne devient pas automatiquement une alerte.** "
            "On fixe un seuil : un score supérieur ou égal au seuil déclenche une vérification. "
            "Un seuil bas crée plus d'alertes ; un seuil haut risque de laisser passer plus de fraudes."
        )
        st.write(
            "**Sur la validation**, nous essayons plusieurs seuils et retenons celui qui "
            "minimise le coût métier. Ce réglage est parfois appelé « calibration du seuil », "
            "mais **ce n'est pas une calibration des probabilités**."
        )
        st.write(
            "**Coût total = montants des fraudes manquées + "
            "coût par alerte × nombre d'alertes.**"
        )
        st.write(
            "**Sur le test**, on applique sans modification le seuil choisi sur la validation. "
            "On observe ensuite la PR-AUC, le rappel, la précision, F2, MCC et le coût obtenu. "
            "Le test sert à mesurer la performance finale, pas à optimiser de nouveau."
        )

        c1, c2 = st.columns(2)
        with c1:
            alert_cost = st.slider(
                "Coût d'analyse d'une alerte",
                min_value=0.0,
                max_value=50.0,
                value=5.0,
                step=1.0,
            )
        with c2:
            max_k = min(500, len(split.y_test))
            k = st.slider(
                "Budget d'alertes k",
                min_value=10,
                max_value=max_k,
                value=min(100, max_k),
                step=10,
            )

        st.caption(
            "Le budget k représente le nombre de transactions que l'équipe peut examiner. "
            "Le seuil recommandé est choisi sur la validation, jamais sur le test."
        )

        table = comparison_table(
            results,
            split,
            alert_cost=alert_cost,
            k=k,
        )
        st.subheader("Décision issue de la validation — seuils et coûts")
        st.write(
            "Ce tableau présente les seuils calculés pour chaque méthode sur la validation. "
            "Ils ne sont pas recalculés à partir du test."
        )
        st.dataframe(
            table[["Méthode", "Seuil validation", "Coût validation"]].style.format(
                {"Seuil validation": "{:.4f}", "Coût validation": "{:,.2f}"}
            ),
            width="stretch",
            hide_index=True,
        )
        interpretation(
            "Pour chaque méthode, le seuil affiché est celui qui minimise le coût estimé "
            "sur la validation, compte tenu du coût par alerte choisi plus haut. "
            "Les seuils d'Isolation Forest et de LOF concernent des scores d'anomalie, "
            "pas des probabilités de fraude."
        )

        st.subheader("Résultats finaux — test")
        st.write(
            "Nous appliquons maintenant au test les modèles déjà entraînés et les seuils "
            "retenus sur validation. Les résultats ci-dessous mesurent ce qui se passe "
            "sur des transactions plus récentes."
        )
        st.dataframe(
            table.style.format(
                {
                    "PR-AUC test": "{:.4f}",
                    "ROC-AUC test": "{:.4f}",
                    f"Précision@{k}": "{:.4f}",
                    f"Rappel@{k}": "{:.4f}",
                    "F2 test": "{:.4f}",
                    "MCC test": "{:.4f}",
                    "Seuil validation": "{:.4f}",
                    "Coût validation": "{:,.2f}",
                    "Coût test": "{:,.2f}",
                }
            ),
            width="stretch",
        )

        st.subheader("Comprendre toutes les métriques du tableau")
        st.write(
            "Les scores ne mesurent pas tous la même chose. Certains évaluent le "
            "**classement des transactions**, d'autres les **alertes déclenchées "
            "au seuil choisi**, et d'autres encore le **coût métier**."
        )
        explanations = pd.DataFrame(
            [
                {
                    "Métrique": "PR-AUC test",
                    "Définition simple": "Résume la précision et le rappel lorsque le seuil varie. Ici calculée avec la précision moyenne (Average Precision).",
                    "Comment lire": "Plus elle est grande, mieux le modèle classe les fraudes rares. Elle ne représente pas l'accuracy.",
                },
                {
                    "Métrique": "ROC-AUC test",
                    "Définition simple": "Mesure si les fraudes reçoivent en général un score plus élevé que les transactions normales.",
                    "Comment lire": "0,5 ≈ classement aléatoire ; 1 = séparation parfaite. À compléter avec la PR-AUC pour les fraudes rares.",
                },
                {
                    "Métrique": f"Précision@{k}",
                    "Définition simple": f"Parmi les {k} transactions avec les scores les plus élevés, quelle proportion est réellement frauduleuse ?",
                    "Comment lire": f"Une précision de 0,46 signifie que 46 % de ces {k} transactions sont des fraudes (46 sur 100 si k = 100).",
                },
                {
                    "Métrique": f"Rappel@{k}",
                    "Définition simple": f"Parmi toutes les fraudes du test, quelle proportion figure dans les {k} transactions les plus suspectes ?",
                    "Comment lire": "Un rappel de 0,187 signifie que 18,7 % des fraudes sont retrouvées dans les premières alertes.",
                },
                {
                    "Métrique": "F2 test",
                    "Définition simple": "Combine précision et rappel au seuil choisi, mais donne plus d'importance au rappel que le F1.",
                    "Comment lire": "Plus proche de 1 = mieux, selon ce compromis. Utile quand manquer une fraude est coûteux.",
                },
                {
                    "Métrique": "MCC test",
                    "Définition simple": "Résume les quatre cas : vraies fraudes détectées, fausses alertes, fraudes manquées et normales reconnues.",
                    "Comment lire": "Entre -1 et 1 : 1 = parfait ; 0 ≈ aucune association ; -1 = prédictions inversées.",
                },
                {
                    "Métrique": "Seuil validation",
                    "Définition simple": "Score minimal à partir duquel une transaction déclenche une alerte, choisi sur la validation.",
                    "Comment lire": "Ce n'est pas une métrique de qualité. Pour LOF et Isolation Forest, ce n'est pas une probabilité.",
                },
                {
                    "Métrique": "Coût validation",
                    "Définition simple": "Coût estimé sur la validation au seuil choisi : fraudes manquées + vérifications d'alertes.",
                    "Comment lire": "Plus faible = mieux pour ce coût métier. C'est ce coût qui sert à régler le seuil.",
                },
                {
                    "Métrique": "Coût test",
                    "Définition simple": "Même calcul de coût, mais sur les transactions du test au seuil déjà choisi.",
                    "Comment lire": "Plus faible = mieux. Ce coût évalue le choix sur des données plus récentes.",
                },
                {
                    "Métrique": "Alertes test",
                    "Définition simple": "Nombre de transactions dont le score dépasse le seuil retenu.",
                    "Comment lire": "Plus d'alertes demandent plus de vérifications ; ce ne sont pas toutes des fraudes.",
                },
                {
                    "Métrique": "Précision au seuil",
                    "Définition simple": "Parmi toutes les alertes au seuil choisi, proportion de fraudes réelles.",
                    "Comment lire": "Une précision élevée signifie peu de fausses alertes parmi les alertes.",
                },
                {
                    "Métrique": "Rappel au seuil",
                    "Définition simple": "Parmi toutes les fraudes du test, proportion effectivement détectée au seuil choisi.",
                    "Comment lire": "Un rappel élevé signifie peu de fraudes manquées.",
                },
            ]
        )
        st.dataframe(explanations, width="stretch", hide_index=True)
        st.caption(
            "Précision@k et Rappel@k utilisent les k scores les plus élevés. "
            "F2, MCC, les coûts et le nombre d'alertes dépendent du seuil fixé sur "
            "la validation. Le curseur exploratoire plus bas illustre ces compromis, "
            "mais ne sert pas à rechoisir le seuil final sur le test."
        )

        best = table.iloc[0]
        best_precision = float(best[f"Précision@{k}"])
        best_recall = float(best[f"Rappel@{k}"])
        fraud_count_test = int(split.y_test.sum())
        k_effective = min(k, len(split.y_test))
        detected_in_top_k = int(round(best_precision * k_effective))
        interpretation(
            f"**Sur ce test**, {best['Méthode']} obtient la PR-AUC la plus élevée "
            f"({best['PR-AUC test']:.4f}), devant les autres méthodes. "
            f"Parmi les {k_effective} transactions prioritaires selon ses scores, "
            f"{detected_in_top_k} sont réellement frauduleuses "
            f"(précision@{k_effective} = {best_precision:.1%}). "
            f"Elles représentent {best_recall:.1%} des "
            f"{fraud_count_test} fraudes du test (rappel@{k_effective}). "
            f"Son F2 au seuil retenu est {best['F2 test']:.4f} et son MCC "
            f"est {best['MCC test']:.4f}. "
            "Ce sont des performances observées sur le test, "
            "et non une raison pour optimiser à nouveau le modèle sur ce test."
        )

        st.subheader("Courbes précision-rappel")
        pr_rows = []
        for name, result in results.items():
            precision, recall, _ = precision_recall_curve(
                split.y_test,
                result.test_scores,
            )
            pr_rows.append(
                pd.DataFrame(
                    {
                        "precision": precision,
                        "recall": recall,
                        "Méthode": name,
                    }
                )
            )
        pr_df = pd.concat(pr_rows, ignore_index=True)
        pr_fig = px.line(
            pr_df,
            x="recall",
            y="precision",
            color="Méthode",
            color_discrete_map=MODEL_COLORS,
            title="Courbes précision-rappel — test",
        )
        st.plotly_chart(pr_fig, width="stretch")

        st.subheader("Explorer le seuil d'une méthode")
        method = st.selectbox(
            "Méthode",
            list(results.keys()),
            key="eval_method",
        )
        result = results[method]

        recommended = optimize_threshold(
            split.y_validation,
            result.validation_scores,
            split.validation_raw["transaction_amount"],
            alert_cost,
        )
        scores = result.test_scores
        score_min = float(np.min(scores))
        score_max = float(np.max(scores))
        span = max(score_max - score_min, 1e-6)
        default_threshold = float(
            np.clip(recommended["threshold"], score_min, score_max)
        )

        threshold = st.slider(
            "Seuil exploratoire sur le test",
            min_value=score_min,
            max_value=score_max,
            value=default_threshold,
            step=span / 200,
            format="%.4f",
        )

        threshold_metrics = metrics_at_threshold(
            split.y_test,
            scores,
            threshold,
        )
        test_cost = business_cost(
            split.y_test,
            scores,
            split.test_raw["transaction_amount"],
            threshold,
            alert_cost,
        )
        p_at_k, r_at_k = precision_recall_at_k(
            split.y_test,
            scores,
            k,
        )

        m1, m2, m3, m4 = st.columns(4)
        m1.metric("Précision", f"{threshold_metrics['precision']:.3f}")
        m2.metric("Rappel", f"{threshold_metrics['recall']:.3f}")
        m3.metric("F2", f"{threshold_metrics['f2']:.3f}")
        m4.metric("MCC", f"{threshold_metrics['mcc']:.3f}")

        m5, m6, m7, m8 = st.columns(4)
        m5.metric("Alertes", threshold_metrics["alerts"])
        m6.metric("Coût test", f"{test_cost:,.2f}")
        m7.metric(f"Précision@{k}", f"{p_at_k:.3f}")
        m8.metric(f"Rappel@{k}", f"{r_at_k:.3f}")

        no_model_cost = float(
            split.test_raw.loc[
                split.y_test.eq(1),
                "transaction_amount",
            ].sum()
        )
        st.info(
            f"Seuil recommandé sur validation : {recommended['threshold']:.4f}. "
            f"Coût sans aucune alerte sur le test : {no_model_cost:,.2f}. "
            "Le curseur sert seulement à comprendre le compromis ; le seuil final reste celui choisi sur validation."
        )


elif module == "5 · Explainable AI":
    st.header("Module 5 — Explainable AI")
    section_intro(
        "Expliquer pourquoi le modèle supervisé signale certaines transactions.",
        "SHAP global, waterfall local et codes motifs des 3 à 5 facteurs les plus influents.",
        "Une alerte utile doit être compréhensible par l'analyste : il faut montrer les facteurs qui ont poussé le score vers la fraude.",
    )

    if "fraud_results" not in st.session_state:
        st.warning("Entraînez d'abord les modèles dans le Module 3.")
    else:
        supervised = st.session_state["fraud_results"]["Régression logistique"]
        model = supervised.estimator

        tab_global, tab_local = st.tabs(
            ["Explication globale", "Explication d'une alerte"]
        )

        with tab_global:
            st.write(
                "SHAP attribue à chaque variable une contribution à la prédiction. "
                "Le beeswarm montre quelles variables ont le plus d'effet sur plusieurs transactions et dans quel sens."
            )

            if st.button("Calculer SHAP global", type="primary"):
                sample = split.X_test.sample(
                    n=min(200, len(split.X_test)),
                    random_state=42,
                )
                with st.spinner("Calcul SHAP..."):
                    shap_values, feature_names = compute_shap(
                        model,
                        split.X_train,
                        sample,
                    )
                st.session_state["fraud_global_shap"] = (
                    shap_values,
                    feature_names,
                )

            if "fraud_global_shap" in st.session_state:
                import shap

                shap_values, feature_names = st.session_state[
                    "fraud_global_shap"
                ]
                plt.figure()
                shap.plots.beeswarm(
                    shap_values,
                    max_display=15,
                    show=False,
                )
                st.pyplot(plt.gcf(), clear_figure=True)

                mean_abs = np.abs(shap_values.values).mean(axis=0)
                top = (
                    pd.DataFrame(
                        {
                            "variable": feature_names,
                            "importance_SHAP": mean_abs,
                        }
                    )
                    .sort_values("importance_SHAP", ascending=False)
                    .head(10)
                )
                st.dataframe(top, width="stretch", hide_index=True)

        with tab_local:
            top_positions = np.argsort(
                supervised.test_scores
            )[::-1][: min(50, len(split.X_test))]

            selected_position = st.selectbox(
                "Choisir une transaction parmi les plus suspectes",
                options=[int(value) for value in top_positions],
                format_func=lambda pos: (
                    f"Position {pos} · score={supervised.test_scores[pos]:.3f} · "
                    f"réel={'fraude' if int(split.y_test.iloc[pos]) == 1 else 'normal'}"
                ),
            )

            row = split.X_test.iloc[[selected_position]]
            raw_row = split.test_raw.iloc[selected_position]
            probability = float(model.predict_proba(row)[0, 1])

            c1, c2, c3 = st.columns(3)
            c1.metric("Score fraude", f"{probability:.1%}")
            c2.metric(
                "Classe réelle",
                "Fraude" if int(split.y_test.iloc[selected_position]) == 1 else "Normale",
            )
            c3.metric(
                "Montant",
                f"{float(raw_row['transaction_amount']):,.2f}",
            )

            with st.expander("Voir la transaction"):
                raw_display = raw_row.astype(str).to_frame("Valeur")
                st.dataframe(
                    raw_display,
                    width="stretch",
                )

            if st.button("Expliquer cette alerte", type="primary"):
                with st.spinner("Calcul de l'explication locale..."):
                    local_values, feature_names = compute_shap(
                        model,
                        split.X_train,
                        row,
                    )
                st.session_state["fraud_local_shap"] = (
                    selected_position,
                    local_values,
                    feature_names,
                )

            local_state = st.session_state.get("fraud_local_shap")
            if local_state and local_state[0] == selected_position:
                import shap

                _, local_values, feature_names = local_state

                st.subheader("SHAP waterfall")
                plt.figure()
                shap.plots.waterfall(
                    local_values[0],
                    max_display=12,
                    show=False,
                )
                st.pyplot(plt.gcf(), clear_figure=True)

                st.subheader("Codes motifs")
                reasons = top_reason_codes(
                    local_values[0],
                    feature_names,
                    top_n=5,
                )
                reasons.insert(
                    0,
                    "code",
                    [f"M{i:02d}" for i in range(1, len(reasons) + 1)],
                )
                st.dataframe(
                    reasons,
                    width="stretch",
                    hide_index=True,
                )

                risk_up = reasons.loc[
                    reasons["contribution"] > 0,
                    "variable",
                ].tolist()
                if risk_up:
                    st.info(
                        "Principaux facteurs qui poussent cette transaction vers l'alerte : "
                        + ", ".join(risk_up[:3])
                        + "."
                    )

        st.warning(
            "Une explication SHAP décrit le comportement du modèle. "
            "Elle ne prouve pas qu'une variable cause réellement la fraude."
        )


st.sidebar.markdown("---")
st.sidebar.markdown(f"[Source dataset Kaggle]({KAGGLE_SOURCE})")
