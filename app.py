import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

from data.loader       import get_connection, load_from_db
from features.engineer import add_features, prepare_Xy, time_split
from models.train      import build_models, train_all
from models.tune       import tune_all
from models.evaluate   import evaluate_all

from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix

# ─── Page config ──────────────────────────────────────────────────────────────

st.set_page_config(
    page_title  = "AAPL Classifier",
    page_icon   = "📈",
    layout      = "wide",
    initial_sidebar_state = "expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────

st.markdown("""
<style>
    [data-testid="stMetricValue"] { font-size: 2rem; font-weight: 700; }
    .block-container { padding-top: 2rem; }
    .stTabs [data-baseweb="tab"] { font-size: 0.95rem; font-weight: 500; }
</style>
""", unsafe_allow_html=True)

# ─── Sidebar ──────────────────────────────────────────────────────────────────

with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    tune_mode = st.toggle("Hyperparameter Tuning", value=False,
                          help="ON = RandomizedSearchCV + TimeSeriesSplit (slower). OFF = fixed params (fast).")

    n_iter = 20
    if tune_mode:
        n_iter = st.slider("Tuning iterations (n_iter)", 10, 80, 20,
                           help="More iterations = better params but slower. Each iteration × 5 folds.")
        st.caption(f"Total fits: {n_iter * 5} per model")

    st.divider()
    test_size = st.slider("Test set size", 0.10, 0.40, 0.20, 0.05,
                          help="Fraction of data held out chronologically for testing.")

    st.divider()
    run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

# ─── Header ───────────────────────────────────────────────────────────────────

st.title("📈 AAPL Stock Direction Classifier")
st.caption("Predicts next-day price direction (Up / Down) using Random Forest and XGBoost")
st.divider()

# ─── Pipeline ─────────────────────────────────────────────────────────────────

if run_btn:

    # 1. Load
    with st.status("Loading data from MySQL …", expanded=True) as status:
        conn = get_connection()
        df_raw = load_from_db(conn)
        conn.close()
        status.update(label=f"✅ Loaded {len(df_raw):,} rows  "
                            f"({df_raw.index[0].date()} → {df_raw.index[-1].date()})",
                      state="complete")

    # 2. Features
    with st.status("Engineering features …", expanded=False) as status:
        df_feat    = add_features(df_raw.copy())
        X, y, df_feat = prepare_Xy(df_feat)
        X_train, X_test, y_train, y_test = time_split(X, y, test_size=test_size)
        status.update(label=f"✅ {X.shape[1]} features  |  "
                            f"Train: {len(X_train):,}  Test: {len(X_test):,}",
                      state="complete")

    # 3. Train / Tune
    label = "Tuning models (this may take a few minutes) …" if tune_mode else "Training models …"
    with st.status(label, expanded=False) as status:
        if tune_mode:
            models = tune_all(X_train, y_train, n_iter=n_iter)
        else:
            models = build_models()
            models = train_all(models, X_train, y_train)
        status.update(label="✅ Models ready", state="complete")

    # 4. Metrics
    results = {}
    for name, model in models.items():
        preds = model.predict(X_test)
        proba = model.predict_proba(X_test)[:, 1]
        cm    = confusion_matrix(y_test, preds)
        results[name] = {
            "preds": preds,
            "proba": proba,
            "cm":    cm,
            "acc":   accuracy_score(y_test, preds),
            "auc":   roc_auc_score(y_test, proba),
            "model": model,
        }

    # ── Tomorrow's prediction ─────────────────────────────────────────────────
    latest = df_feat[X.columns].dropna().iloc[[-1]]
    last_date = df_feat.index[-1].date()

    st.subheader("🔮 Tomorrow's Prediction")
    pred_cols = st.columns(len(models))
    for i, (name, r) in enumerate(results.items()):
        prob = r["model"].predict_proba(latest)[0, 1]
        direction = "⬆ UP" if prob >= 0.5 else "⬇ DOWN"
        conf_color = "green" if prob >= 0.5 else "red"
        with pred_cols[i]:
            st.metric(
                label    = name,
                value    = direction,
                delta    = f"Confidence: {prob:.1%}",
                delta_color = "normal" if prob >= 0.5 else "inverse",
            )
    st.caption(f"Based on data up to {last_date}. Not financial advice.")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab1, tab2, tab3, tab4 = st.tabs(["📊 Metrics", "🔢 Confusion Matrix",
                                       "📉 Probability Distribution", "🌲 Feature Importance"])

    # ── Tab 1: Metrics ────────────────────────────────────────────────────────
    with tab1:
        st.subheader("Model Performance")
        metric_cols = st.columns(len(models))
        for i, (name, r) in enumerate(results.items()):
            with metric_cols[i]:
                st.markdown(f"**{name}**")
                st.metric("Accuracy",  f"{r['acc']:.4f}")
                st.metric("ROC-AUC",   f"{r['auc']:.4f}")
                st.divider()

                # Classification report as dataframe
                from sklearn.metrics import classification_report
                report = classification_report(y_test, r["preds"],
                                               target_names=["Down", "Up"],
                                               output_dict=True)
                report_df = pd.DataFrame(report).T.round(3)
                st.dataframe(report_df, use_container_width=True)

        # Side-by-side bar chart comparison
        st.subheader("Accuracy vs ROC-AUC")
        fig_compare = go.Figure()
        names = list(results.keys())
        fig_compare.add_trace(go.Bar(
            name="Accuracy", x=names,
            y=[results[n]["acc"] for n in names],
            marker_color="#4C72B0"
        ))
        fig_compare.add_trace(go.Bar(
            name="ROC-AUC", x=names,
            y=[results[n]["auc"] for n in names],
            marker_color="#55A868"
        ))
        fig_compare.update_layout(
            barmode="group", yaxis_range=[0, 1],
            height=350, margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=1.1)
        )
        st.plotly_chart(fig_compare, use_container_width=True)

    # ── Tab 2: Confusion Matrix ───────────────────────────────────────────────
    with tab2:
        st.subheader("Confusion Matrix")
        cm_cols = st.columns(len(models))
        for i, (name, r) in enumerate(results.items()):
            with cm_cols[i]:
                cm = r["cm"]
                fig_cm = px.imshow(
                    cm,
                    labels=dict(x="Predicted", y="Actual", color="Count"),
                    x=["Down", "Up"], y=["Down", "Up"],
                    text_auto=True,
                    color_continuous_scale="Blues",
                    title=name,
                )
                fig_cm.update_layout(height=350, margin=dict(t=50, b=20))
                st.plotly_chart(fig_cm, use_container_width=True)

                # Key stats below matrix
                tn, fp, fn, tp = cm.ravel()
                c1, c2 = st.columns(2)
                c1.metric("True Up (TP)",   tp)
                c2.metric("True Down (TN)", tn)
                c1.metric("False Up (FP)",  fp)
                c2.metric("False Down (FN)",fn)

    # ── Tab 3: Probability Distribution ──────────────────────────────────────
    with tab3:
        st.subheader("Predicted Probability Distribution")
        prob_cols = st.columns(len(models))
        for i, (name, r) in enumerate(results.items()):
            with prob_cols[i]:
                fig_hist = go.Figure()
                fig_hist.add_trace(go.Histogram(
                    x=r["proba"], nbinsx=30,
                    name=name, marker_color="#4C72B0",
                    opacity=0.8
                ))
                fig_hist.add_vline(x=0.5, line_dash="dash",
                                   line_color="red",
                                   annotation_text="threshold=0.5")
                fig_hist.update_layout(
                    title=name,
                    xaxis_title="Predicted probability (Up)",
                    yaxis_title="Count",
                    height=350, margin=dict(t=50, b=20),
                    showlegend=False,
                )
                st.plotly_chart(fig_hist, use_container_width=True)

    # ── Tab 4: Feature Importance ─────────────────────────────────────────────
    with tab4:
        st.subheader("Top 15 Feature Importances")
        fi_cols = st.columns(len(models))
        for i, (name, r) in enumerate(results.items()):
            with fi_cols[i]:
                clf   = r["model"].named_steps["clf"]
                imps  = clf.feature_importances_
                feat_df = pd.DataFrame({
                    "feature":    list(X.columns),
                    "importance": imps,
                }).sort_values("importance", ascending=False).head(15)

                fig_fi = px.bar(
                    feat_df.sort_values("importance"),
                    x="importance", y="feature",
                    orientation="h",
                    title=name,
                    color="importance",
                    color_continuous_scale="Teal",
                )
                fig_fi.update_layout(
                    height=450, margin=dict(t=50, b=20),
                    showlegend=False,
                    coloraxis_showscale=False,
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    # ── Price chart with train/test split ─────────────────────────────────────
    st.divider()
    st.subheader("📅 Train / Test Split on Price History")

    split_idx  = int(len(df_feat) * (1 - test_size))
    train_df   = df_raw.iloc[:split_idx]
    test_df    = df_raw.iloc[split_idx:]

    fig_price = go.Figure()
    fig_price.add_trace(go.Scatter(
        x=train_df.index, y=train_df["close"],
        name="Train", line=dict(color="#4C72B0", width=1.5)
    ))
    fig_price.add_trace(go.Scatter(
        x=test_df.index, y=test_df["close"],
        name="Test", line=dict(color="#DD8452", width=1.5)
    ))
    fig_price.update_layout(
        height=350, margin=dict(t=20, b=20),
        xaxis_title="Date", yaxis_title="Close Price (USD)",
        legend=dict(orientation="h", y=1.1),
        hovermode="x unified",
    )
    st.plotly_chart(fig_price, use_container_width=True)

else:
    # ── Landing state ─────────────────────────────────────────────────────────
    st.info("Configure settings in the sidebar and click **🚀 Run Pipeline** to start.", icon="👈")

    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("### 🗄️ Data")
        st.markdown("Pulls directly from your MySQL `stock_data` table. No CSV needed.")
    with col2:
        st.markdown("### ⚙️ Models")
        st.markdown("Random Forest + XGBoost with optional hyperparameter tuning via `TimeSeriesSplit`.")
    with col3:
        st.markdown("### 📊 Results")
        st.markdown("Accuracy, ROC-AUC, confusion matrix, feature importance, and tomorrow's prediction.")