import warnings
warnings.filterwarnings("ignore")

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, roc_auc_score, confusion_matrix
from datetime import datetime
from pathlib import Path

from data.loader import load_from_db, get_connection
from data.validator import data_validation
from features.engineer import add_features, prepare_Xy, time_split
from models.train import train_all, save_models
from models.tune import tune_all, build_base_models
from models.evaluate import evaluate_all, predict_tomorrow
from config import TEST_SIZE, RANDOM_STATE, TICKER, FEATURE_COLS

# ── Page Config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title=f"{TICKER} · ML Pipeline",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Global Style ──────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Space+Mono:wght@400;700&family=DM+Sans:wght@300;400;500;600&display=swap');

html, body, [class*="css"] { font-family: 'DM Sans', sans-serif; }
h1, h2, h3 { font-family: 'Space Mono', monospace !important; letter-spacing: -0.5px; }

section[data-testid="stSidebar"] { background: #0a0a12; border-right: 1px solid #1e1e30; }

.stat-card { background: #0f0f1a; border: 1px solid #1e1e30; border-radius: 12px; padding: 20px 24px; margin-bottom: 8px; }
.stat-card .label { font-size: 11px; color: #555577; text-transform: uppercase; letter-spacing: 1.5px; font-family: 'Space Mono', monospace; }
.stat-card .value { font-size: 28px; font-weight: 600; color: #e0e0ff; margin-top: 4px; }

.check-row { display: flex; align-items: center; gap: 10px; padding: 10px 16px; border-radius: 8px; margin-bottom: 6px; font-family: 'Space Mono', monospace; font-size: 13px; }
.check-pass { background: #0d1f14; border: 1px solid #1a3d26; color: #55A868; }
.check-fail { background: #1f0d0d; border: 1px solid #3d1a1a; color: #C44E52; }
.check-warn { background: #1f1a0d; border: 1px solid #3d3020; color: #CCB974; }

.pred-card { border-radius: 14px; padding: 28px; text-align: center; border: 1px solid #1e1e30; }
.pred-up   { background: linear-gradient(135deg, #0d1f14 0%, #0a1510 100%); border-color: #1a3d26; }
.pred-down { background: linear-gradient(135deg, #1f0d0d 0%, #150a0a 100%); border-color: #3d1a1a; }
.pred-direction { font-size: 42px; font-family: 'Space Mono', monospace; font-weight: 700; }
.pred-up   .pred-direction { color: #55A868; }
.pred-down .pred-direction { color: #C44E52; }
.pred-conf { font-size: 14px; color: #888; margin-top: 8px; font-family: 'Space Mono', monospace; }
.pred-name { font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #555577; margin-bottom: 12px; }

button[data-baseweb="tab"] { font-family: 'Space Mono', monospace !important; font-size: 12px !important; }
div[data-testid="stMetricValue"] { font-family: 'Space Mono', monospace; }

div.stButton > button[kind="primary"] {
    background: linear-gradient(135deg, #4C72B0 0%, #2a4a8a 100%);
    border: none; border-radius: 10px;
    font-family: 'Space Mono', monospace; font-size: 13px;
    letter-spacing: 0.5px; padding: 12px; transition: all 0.2s;
}
div.stButton > button[kind="primary"]:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(76, 114, 176, 0.4);
}
.pipeline-header { font-family: 'Space Mono', monospace; font-size: 11px; text-transform: uppercase; letter-spacing: 2px; color: #555577; }
</style>
""", unsafe_allow_html=True)

# ── Session State ─────────────────────────────────────────────────────────────
defaults = {
    "df": None, "df_feat": None,
    "X_train": None, "X_test": None,
    "y_train": None, "y_test": None,
    "trained_models": None, "eval_results": None,
    "validation_results": None, "pipeline_done": False,
    "last_run": None, "baseline_acc": None,
}
for k, v in defaults.items():
    st.session_state.setdefault(k, v)

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    st.markdown(f"""
    <div style='padding: 8px 0 20px'>
        <div style='font-family: Space Mono, monospace; font-size: 10px;
                    text-transform: uppercase; letter-spacing: 2px; color: #555577;'>Ticker</div>
        <div style='font-family: Space Mono, monospace; font-size: 32px;
                    font-weight: 700; color: #e0e0ff; margin-top: 4px;'>{TICKER}</div>
    </div>
    """, unsafe_allow_html=True)

    st.divider()
    tune = st.toggle("Hyperparameter Tuning", value=False)
    if tune:
        st.caption("⚠️ ~5–15 min. 500+ model fits.")
        # FIX: n_iter was declared but never actually passed to tune_all in the original
        n_iter = st.slider("Tuning Iterations", 10, 80, 40)
    else:
        n_iter = 40  # default, unused when tune=False

    st.divider()
    run_btn = st.button("▶ Run Pipeline", type="primary", use_container_width=True)

    if st.session_state.pipeline_done:
        if st.button("💾 Save Models", use_container_width=True):
            save_models(st.session_state.trained_models)
            st.success("Saved.")

    st.divider()
    if st.session_state.last_run:
        st.caption(f"Last run: {st.session_state.last_run}")
    else:
        st.caption("Not yet run.")

    st.markdown("""
    <div style='position: fixed; bottom: 24px; font-size: 10px; color: #333355; font-family: Space Mono, monospace;'>
        sklearn · xgboost · yfinance · streamlit
    </div>
    """, unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(f"""
<div style='margin-bottom: 4px'>
    <span class='pipeline-header'>Price Direction Classification</span>
</div>
<h1 style='margin: 0; font-size: 36px;'>{TICKER} ML Pipeline</h1>
""", unsafe_allow_html=True)
st.markdown("<br>", unsafe_allow_html=True)

# ── Tabs ──────────────────────────────────────────────────────────────────────
tab_data, tab_val, tab_feat, tab_models, tab_pred = st.tabs([
    " 📊 Data ", " ✅ Validation ", " 🔧 Features ", " 🤖 Models ", " 🔮 Prediction "
])

# ── Pipeline Runner ───────────────────────────────────────────────────────────
if run_btn:
    conn = get_connection()
    try:
        with st.status("Running pipeline...", expanded=True) as status:

            # Step 1 — Load
            st.write("📥 Loading data from database...")
            df = load_from_db(conn)
            if df.empty:
                status.update(label="❌ No data in DB", state="error")
                st.stop()
            st.session_state.df = df
            st.write(f"✅ {df.shape[0]:,} rows loaded | {df.index.min().date()} → {df.index.max().date()}")

            # Step 2 — Validate
            st.write("🔍 Validating OHLCV data...")
            val = data_validation(df)
            st.session_state.validation_results = val

            errors = []
            if not val["ohlc_clean"]:
                errors.append(f"OHLC violations: {val['ohlc_violations']}")
            if val["has_nulls"]:
                errors.append(f"Null values: {val['missing_values']}")
            if val["duplicate_dates"] > 0:
                errors.append(f"{val['duplicate_dates']} duplicate dates")
            if errors:
                status.update(label="❌ Validation failed", state="error")
                for e in errors:
                    st.error(e)
                conn.close()
                st.stop()
            st.write("✅ Validation passed — data is clean")

            # Step 3 — Features
            st.write("🔧 Engineering features...")
            df_feat        = add_features(df)
            X, y, df_feat  = prepare_Xy(df_feat)
            X_train, X_test, y_train, y_test = time_split(X, y, test_size=TEST_SIZE)
            st.session_state.df_feat  = df_feat
            st.session_state.X_train  = X_train
            st.session_state.X_test   = X_test
            st.session_state.y_train  = y_train
            st.session_state.y_test   = y_test

            # Baseline accuracy (always predict UP)
            baseline_acc = float(y_test.mean())
            st.session_state.baseline_acc = baseline_acc
            st.write(f"✅ {X_train.shape[1]} features | Train: {len(X_train)} Test: {len(X_test)} | Baseline: {baseline_acc:.2%}")

            # Step 4 — Train / Tune
            if tune:
                st.write(f"⚙️ Tuning ({n_iter} iterations × 5 folds)...")
                # FIX: pass n_iter — original forgot this argument
                trained_models = tune_all(X_train, y_train, n_iter=n_iter)
            else:
                st.write("🤖 Training models with default params...")
                base       = build_base_models()
                models_only = {name: model for name, (model, _) in base.items()}
                trained_models = train_all(models_only, X_train, y_train)

            st.session_state.trained_models = trained_models
            st.write(f"✅ {len(trained_models)} models trained")

            # Step 5 — Evaluate
            st.write("📊 Evaluating on test set...")
            eval_results = evaluate_all(trained_models, X_test, y_test)
            st.session_state.eval_results  = eval_results
            st.session_state.pipeline_done = True
            st.session_state.last_run      = datetime.now().strftime("%Y-%m-%d %H:%M")
            status.update(label="✅ Pipeline complete", state="complete")

    except Exception as e:
        st.error(f"Pipeline crashed: {e}")
        try:
            conn.rollback()
        except Exception:
            pass
        raise
    finally:
        pass

# ── Tab: Data ─────────────────────────────────────────────────────────────────
with tab_data:
    if st.session_state.df is not None:
        df = st.session_state.df
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",  f"{df.shape[0]:,}")
        c2.metric("Date From",   str(df.index.min().date()))
        c3.metric("Date To",     str(df.index.max().date()))
        c4.metric("Years",       f"{(df.index.max() - df.index.min()).days / 365:.1f}")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([3, 1])
        with col_l:
            st.markdown("**Close Price History**")
            st.line_chart(df["close"], color="#4C72B0", height=280)
        with col_r:
            st.markdown("**Volume**")
            st.bar_chart(df["volume"], color="#2a3a5a", height=280)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Recent Data (last 50 rows)**")
        st.dataframe(df.tail(50).style.format(precision=2), use_container_width=True, height=280)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Run the pipeline from the sidebar to load data.")

# ── Tab: Validation ───────────────────────────────────────────────────────────
with tab_val:
    if st.session_state.validation_results is not None:
        r = st.session_state.validation_results
        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Total Rows",    f"{r['row_count']:,}")
        c2.metric("Null Columns",  len(r["missing_values"]) if r["has_nulls"] else 0)
        c3.metric("Duplicate Dates", r["duplicate_dates"])
        c4.metric("Price Jumps",   r["suspicious_price_jumps"])

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Checks**")
        checks = [
            ("OHLC Logic",     r["ohlc_clean"],               "High ≥ Low, Open & Close within range"),
            ("No Null Values", not r["has_nulls"],             f"{r['missing_values'] or 'None'}"),
            ("No Duplicates",  r["duplicate_dates"] == 0,      f"{r['duplicate_dates']} duplicate dates"),
            ("No Neg Volume",  r["negative_volume"] == 0,      f"{r['negative_volume']} rows"),
        ]
        for label, passed, detail in checks:
            cls  = "check-pass" if passed else "check-fail"
            icon = "✓" if passed else "✗"
            st.markdown(f"""
            <div class='check-row {cls}'>
                <span style='font-size:16px'>{icon}</span>
                <span style='flex:1'>{label}</span>
                <span style='opacity:0.6; font-size:11px'>{detail}</span>
            </div>
            """, unsafe_allow_html=True)

        if r["suspicious_price_jumps"] > 0:
            st.markdown(f"""
            <div class='check-row check-warn' style='margin-top:8px'>
                ⚠ {r['suspicious_price_jumps']} suspicious price jumps &nbsp;·&nbsp;
                <span style='font-size:11px'>{r.get('suspicious_dates', [])}</span>
            </div>
            """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown(f"**Date range:** `{r['date_from']}` → `{r['date_to']}`")
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Run the pipeline to see validation results.")

# ── Tab: Features ─────────────────────────────────────────────────────────────
with tab_feat:
    if st.session_state.X_train is not None:
        X_train = st.session_state.X_train
        X_test  = st.session_state.X_test
        y_train = st.session_state.y_train

        c1, c2, c3, c4 = st.columns(4)
        c1.metric("Features",        X_train.shape[1])
        c2.metric("Train Rows",      f"{len(X_train):,}")
        c3.metric("Test Rows",       f"{len(X_test):,}")
        c4.metric("Up Days (train)", f"{y_train.mean() * 100:.1f}%")

        st.markdown("<br>", unsafe_allow_html=True)
        col_l, col_r = st.columns([1, 2])
        with col_l:
            st.markdown("**Target Distribution**")
            dist = y_train.value_counts().rename({0: "⬇ Down", 1: "⬆ Up"})
            st.bar_chart(dist, color="#4C72B0", height=240)
        with col_r:
            st.markdown("**Feature Statistics**")
            st.dataframe(X_train.describe().T.style.format(precision=4), use_container_width=True, height=280)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Run the pipeline to see feature data.")

# ── Chart helpers ─────────────────────────────────────────────────────────────
def make_cm_fig(cm):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#0a0a12")
    ax.set_facecolor("#0a0a12")
    ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Down", "Up"], color="#aaa")
    ax.set_yticklabels(["Down", "Up"], color="#aaa")
    ax.set_xlabel("Predicted", color="#666"); ax.set_ylabel("Actual", color="#666")
    ax.tick_params(colors="#666")
    for spine in ax.spines.values(): spine.set_edgecolor("#1e1e30")
    for r in range(2):
        for c in range(2):
            ax.text(c, r, cm[r, c], ha="center", va="center",
                    color="white" if cm[r, c] > cm.max() / 2 else "#aaa",
                    fontsize=15, fontweight="bold")
    plt.tight_layout(pad=0.5)
    return fig

def make_prob_fig(proba_pos):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#0a0a12")
    ax.set_facecolor("#0a0a12")
    ax.hist(proba_pos, bins=30, color="#4C72B0", edgecolor="#0a0a12", alpha=0.85)
    ax.axvline(0.5, color="#C44E52", linestyle="--", linewidth=1.5, label="0.5 threshold")
    ax.set_xlabel("P(Up)", color="#666"); ax.set_ylabel("Count", color="#666")
    ax.tick_params(colors="#555")
    ax.legend(fontsize=9, labelcolor="#aaa", facecolor="#111")
    for spine in ax.spines.values(): spine.set_edgecolor("#1e1e30")
    plt.tight_layout(pad=0.5)
    return fig

def make_imp_fig(model, name):
    fig, ax = plt.subplots(figsize=(3.5, 3))
    fig.patch.set_facecolor("#0a0a12")
    ax.set_facecolor("#0a0a12")
    clf  = model.named_steps["clf"]
    imps = clf.feature_importances_
    n    = min(15, len(FEATURE_COLS))
    idx  = np.argsort(imps)[-n:]
    ax.barh(np.array(FEATURE_COLS)[idx], imps[idx], color="#55A868", alpha=0.85)
    ax.set_xlabel("Importance", color="#666")
    ax.tick_params(colors="#555", labelsize=7)
    for spine in ax.spines.values(): spine.set_edgecolor("#1e1e30")
    plt.tight_layout(pad=0.5)
    return fig

# ── Tab: Models ───────────────────────────────────────────────────────────────
with tab_models:
    if st.session_state.eval_results is not None:
        eval_results   = st.session_state.eval_results
        trained_models = st.session_state.trained_models
        y_test         = st.session_state.y_test
        baseline_acc   = st.session_state.baseline_acc

        # Show baseline for reference
        st.markdown(f"**Baseline (always predict UP):** `{baseline_acc:.4f}` — your models must beat this to add value.")
        st.markdown("<br>", unsafe_allow_html=True)

        for name, (preds, proba, cm) in eval_results.items():
            acc = accuracy_score(y_test, preds)
            auc = roc_auc_score(y_test, proba[:, 1])

            # Beat baseline indicator
            beat = "✅ beats baseline" if acc > baseline_acc else "⚠️ below baseline"
            st.markdown(f"#### {name} — {beat}")

            c1, c2, c3, c4 = st.columns(4)
            c1.metric("Accuracy",  f"{acc:.4f}", delta=f"{acc - baseline_acc:+.4f} vs baseline")
            c2.metric("ROC-AUC",   f"{auc:.4f}")
            c3.metric("Test Rows", f"{len(preds):,}")
            c4.metric("Baseline",  f"{baseline_acc:.4f}")

            col_cm, col_prob, col_imp = st.columns(3)

            with col_cm:
                st.markdown("<div style='font-size:12px;color:#555577;font-family:Space Mono,monospace'>CONFUSION MATRIX</div>", unsafe_allow_html=True)
                fig = make_cm_fig(cm)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)   # FIX: was plt.close() — close the specific figure

            with col_prob:
                st.markdown("<div style='font-size:12px;color:#555577;font-family:Space Mono,monospace'>PROB DISTRIBUTION</div>", unsafe_allow_html=True)
                fig = make_prob_fig(proba[:, 1])
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            with col_imp:
                st.markdown("<div style='font-size:12px;color:#555577;font-family:Space Mono,monospace'>FEATURE IMPORTANCE</div>", unsafe_allow_html=True)
                fig = make_imp_fig(trained_models[name], name)
                st.pyplot(fig, use_container_width=True)
                plt.close(fig)

            st.divider()
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Run the pipeline to see model results.")

# ── Tab: Prediction ───────────────────────────────────────────────────────────
with tab_pred:
    if st.session_state.trained_models is not None:
        df_feat        = st.session_state.df_feat
        trained_models = st.session_state.trained_models
        latest         = df_feat[FEATURE_COLS].dropna().iloc[[-1]]
        as_of_date     = df_feat.index[-1].date()

        st.markdown(f"""
        <div style='margin-bottom: 24px'>
            <div class='pipeline-header'>Tomorrow's forecast</div>
            <div style='font-size: 13px; color: #555577; margin-top: 4px; font-family: Space Mono, monospace;'>
                Based on data as of {as_of_date}
            </div>
        </div>
        """, unsafe_allow_html=True)

        # FIX: compute probabilities ONCE per model, store in dict — original called predict_proba
        # twice per model (once for cards, once for vote counting), doubling inference cost
        model_probs = {name: model.predict_proba(latest)[0, 1] for name, model in trained_models.items()}

        cols = st.columns(len(trained_models))
        for col, (name, prob) in zip(cols, model_probs.items()):
            is_up     = prob >= 0.5
            direction = "⬆ UP" if is_up else "⬇ DOWN"
            cls       = "pred-up" if is_up else "pred-down"
            with col:
                st.markdown(f"""
                <div class='pred-card {cls}'>
                    <div class='pred-name'>{name}</div>
                    <div class='pred-direction'>{direction}</div>
                    <div class='pred-conf'>confidence {prob:.1%}</div>
                </div>
                """, unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        st.markdown("**Model Agreement**")
        votes_up   = sum(1 for p in model_probs.values() if p >= 0.5)
        votes_down = len(model_probs) - votes_up
        consensus  = "⬆ UP" if votes_up > votes_down else ("⬇ DOWN" if votes_down > votes_up else "— SPLIT")

        c1, c2, c3 = st.columns(3)
        c1.metric("Votes UP",   votes_up)
        c2.metric("Votes DOWN", votes_down)
        c3.metric("Consensus",  consensus)
    else:
        st.markdown("<br><br>", unsafe_allow_html=True)
        st.info("Run the pipeline to see tomorrow's prediction.")