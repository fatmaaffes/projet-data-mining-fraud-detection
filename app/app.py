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
        st.dataframe(df.head(10), use_container_width=True)

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Transactions", f"{summary['transactions']:,}")
        c2.metric("Fraudes", f"{summary['frauds']:,}")
        c3.metric("Taux de fraude", f"{summary['fraud_rate']:.2%}")
        c4.metric("Montant frauduleux", f"{summary['fraud_amount']:,.0f}")

        st.subheader("Valeurs manquantes")
        st.dataframe(
            missing_summary(df),
            use_container_width=True,
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
        st.plotly_chart(fig, use_container_width=True)

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
        st.plotly_chart(fig_amount, use_container_width=True)

        grouped_amounts = (
            df.groupby("is_fraud")["transaction_amount"]
            .agg(["count", "mean", "median", "sum"])
            .rename(index={0: "Normales", 1: "Fraudes"})
        )
        st.dataframe(grouped_amounts, use_container_width=True)

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
        st.plotly_chart(fig_hour, use_container_width=True)

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
        st.plotly_chart(fig_daily, use_container_width=True)

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
        st.plotly_chart(fig_cat, use_container_width=True)


elif module == "2 · Préparation & variables":
    st.header("Module 2 — Préparation & variables")
    section_intro(
        "Construire des variables temporelles et comportementales sans utiliser d'information future.",
        "Variables cycliques, vélocité client, écart au profil passé, temps depuis la transaction précédente et split chronologique 60/20/20.",
        "Une fraude est souvent anormale par rapport à l'historique du client ; ces variables décrivent ce comportement tout en évitant la fuite de données.",
    )

    st.subheader("1 · Variables construites")
    feature_rows = [
        ["hour_sin / hour_cos", "Codage cyclique de l'heure : 23 h est proche de 0 h."],
        ["day_of_week", "Jour de la semaine de 0 à 6."],
        ["is_weekend", "1 si samedi ou dimanche."],
        ["shipping_billing_match", "1 si les adresses de livraison et facturation sont identiques."],
    ]

    behavioral_columns = [
        "customer_tx_count_1h",
        "customer_tx_count_24h",
        "customer_amount_mean_30d_past",
        "amount_vs_30d_mean",
        "hours_since_previous_tx",
    ]

    if all(column in featured.columns for column in behavioral_columns):
        feature_rows.extend(
            [
                ["customer_tx_count_1h", "Nombre de transactions précédentes du même client dans la dernière heure."],
                ["customer_tx_count_24h", "Nombre de transactions précédentes du même client dans les dernières 24 h."],
                ["customer_amount_mean_30d_past", "Montant moyen des transactions passées du client sur 30 jours."],
                ["amount_vs_30d_mean", "Montant actuel rapporté au montant moyen passé du client."],
                ["hours_since_previous_tx", "Temps depuis la transaction précédente du même client."],
            ]
        )
        st.info(
            "Les variables comportementales sont calculées avant d'ajouter la transaction courante à l'historique. "
            "Elles n'utilisent donc que le passé du client."
        )
    else:
        unique_ratio = df["customer_id"].nunique() / len(df)
        st.warning(
            f"Limite du dataset : {unique_ratio:.1%} des Customer ID sont uniques. "
            "Il n'y a donc pas assez d'historique répété par client pour construire honnêtement "
            "les variables de vélocité 1 h / 24 h et de moyenne passée. Elles ne sont pas utilisées par le modèle."
        )

    feature_dictionary = pd.DataFrame(
        feature_rows,
        columns=["Variable", "Explication humaine"],
    )
    st.dataframe(feature_dictionary, use_container_width=True, hide_index=True)

    st.subheader("2 · Séparation chronologique")
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
                "Début": part["transaction_date"].min(),
                "Fin": part["transaction_date"].max(),
                "Taux fraude": part["is_fraud"].mean(),
            }
        )
    split_table = pd.DataFrame(split_rows)
    st.dataframe(
        split_table.style.format({"Taux fraude": "{:.2%}"}),
        use_container_width=True,
    )

    st.write(
        "Le modèle apprend sur les transactions les plus anciennes, choisit sa stratégie et son seuil sur la validation, "
        "puis le test représente la période la plus récente jamais utilisée pour prendre une décision de modélisation."
    )

    st.subheader("3 · Prétraitement")
    st.write(
        "Les variables numériques sont imputées par la médiane puis standardisées. "
        "Les variables catégorielles sont imputées par la modalité la plus fréquente puis encodées en One-Hot. "
        "Le prétraitement est ajusté uniquement sur le train."
    )

    with st.expander("Voir un échantillon des variables préparées"):
        display_cols = [
            "transaction_date",
            "transaction_amount",
            "is_fraud",
            "hour",
            "day_of_week",
            "customer_tx_count_1h",
            "customer_tx_count_24h",
            "amount_vs_30d_mean",
            "hours_since_previous_tx",
        ]
        display_cols = [
            col for col in display_cols if col in featured.columns
        ]
        st.dataframe(
            featured[display_cols].head(20),
            use_container_width=True,
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
    st.dataframe(methods, use_container_width=True, hide_index=True)

    if st.button("Entraîner les 3 méthodes", type="primary"):
        with st.spinner("Entraînement en cours..."):
            results = get_models()
        st.session_state["fraud_results"] = results
        st.success("Entraînement terminé.")

    if "fraud_results" in st.session_state:
        results = st.session_state["fraud_results"]
        supervised = results["Régression logistique"]

        st.subheader("Stratégie contre le déséquilibre")
        st.write(supervised.strategy)
        st.caption(
            "La pondération des classes est comparée à l'absence de pondération sur la validation. "
            "La stratégie n'est pas choisie par principe mais par performance PR-AUC."
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
            use_container_width=True,
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
        st.subheader("Comparaison finale — test")
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
            use_container_width=True,
        )

        best = table.iloc[0]
        interpretation(
            f"Selon la PR-AUC sur le test, la meilleure méthode est {best['Méthode']} "
            f"avec {best['PR-AUC test']:.4f}. Le seuil affiché a été déterminé sur la validation puis appliqué au test."
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
        st.plotly_chart(pr_fig, use_container_width=True)

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
                st.dataframe(top, use_container_width=True, hide_index=True)

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
                st.dataframe(
                    raw_row.to_frame("Valeur"),
                    use_container_width=True,
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
                    use_container_width=True,
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
