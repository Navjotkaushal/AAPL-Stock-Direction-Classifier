"""
AAPL Stock Direction Classifier — Demo App
Self-contained: fetches live data from yfinance, no MySQL required.
Drop this file into your repo root and run: streamlit run app_demo.py
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).resolve().parent))

import streamlit as st
import pandas as pd
import numpy as np
import yfinance as yf
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
from sklearn.ensemble import RandomForestClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (
    accuracy_score, roc_auc_score, confusion_matrix,
    classification_report, roc_curve
)
from xgboost import XGBClassifier
import warnings
warnings.filterwarnings("ignore")

# ─── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="AAPL Direction Classifier",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─── Styling ──────────────────────────────────────────────────────────────────
st.markdown("""
<style>
  /* terminal-style header for prediction card */
  .pred-card {
      background: #0d1117;
      border: 1px solid #30363d;
      border-radius: 8px;
      padding: 1.2rem 1.5rem;
      font-family: 'Courier New', monospace;
  }
  .pred-up   { color: #3fb950; font-size: 2rem; font-weight: 700; }
  .pred-down { color: #f85149; font-size: 2rem; font-weight: 700; }
  .pred-label { color: #8b949e; font-size: 0.8rem; letter-spacing: 0.08em; text-transform: uppercase; }
  .pred-conf  { color: #d2a679; font-size: 1rem; }

  [data-testid="stMetricValue"] { font-size: 1.8rem; font-weight: 700; }
  .block-container { padding-top: 1.8rem; }
  .stTabs [data-baseweb="tab"] { font-size: 0.9rem; font-weight: 500; }

  /* sidebar note */
  .sidebar-note {
      background: #161b22;
      border-left: 3px solid #388bfd;
      padding: 0.6rem 0.8rem;
      border-radius: 4px;
      font-size: 0.82rem;
      color: #8b949e;
  }
</style>
""", unsafe_allow_html=True)

# ─── Feature engineering (mirrors features/engineer.py) ──────────────────────
def engineer_features(df: pd.DataFrame) -> pd.DataFrame:
    d = df.copy()
    c = d["Close"]

    # Returns
    for n in [1, 3, 5, 10]:
        d[f"ret_{n}d"] = c.pct_change(n)

    # SMA ratios
    for n in [5, 10, 20, 50]:
        d[f"sma_{n}_ratio"] = c / c.rolling(n).mean() - 1

    # MACD
    ema12 = c.ewm(span=12).mean()
    ema26 = c.ewm(span=26).mean()
    d["macd"]        = ema12 - ema26
    d["macd_signal"] = d["macd"].ewm(span=9).mean()
    d["macd_hist"]   = d["macd"] - d["macd_signal"]

    # RSI-14
    delta  = c.diff()
    gain   = delta.clip(lower=0).rolling(14).mean()
    loss   = (-delta.clip(upper=0)).rolling(14).mean()
    rs     = gain / (loss + 1e-9)
    d["rsi14"] = 100 - 100 / (1 + rs)

    # Bollinger Bands
    ma20   = c.rolling(20).mean()
    std20  = c.rolling(20).std()
    upper  = ma20 + 2 * std20
    lower  = ma20 - 2 * std20
    d["bb_width"] = (upper - lower) / (ma20 + 1e-9)
    d["bb_pct"]   = (c - lower) / (upper - lower + 1e-9)

    # ATR
    h, l, p = df["High"], df["Low"], df["Close"].shift(1)
    tr = pd.concat([h - l, (h - p).abs(), (l - p).abs()], axis=1).max(axis=1)
    d["atr_ratio"] = tr.rolling(14).mean() / (c + 1e-9)

    # Volume
    vol = df["Volume"]
    d["vol_spike"]  = vol / (vol.rolling(20).mean() + 1e-9)
    d["vol_chg"]    = vol.pct_change()

    # Candle structure
    d["body"]        = (df["Close"] - df["Open"]).abs() / (df["Open"] + 1e-9)
    d["upper_shadow"] = (df["High"] - df[["Close", "Open"]].max(axis=1)) / (df["Open"] + 1e-9)
    d["lower_shadow"] = (df[["Close", "Open"]].min(axis=1) - df["Low"]) / (df["Open"] + 1e-9)

    # Target: 1 if tomorrow closes higher
    d["target"] = (c.shift(-1) > c).astype(int)

    return d


def get_feature_cols(df: pd.DataFrame) -> list:
    drop = {"Open", "High", "Low", "Close", "Adj Close", "Volume", "target"}
    return [c for c in df.columns if c not in drop]


def time_split(X, y, test_size=0.20):
    n        = len(X)
    split    = int(n * (1 - test_size))
    return X.iloc[:split], X.iloc[split:], y.iloc[:split], y.iloc[split:]


def build_models():
    return {
        "Random Forest": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", RandomForestClassifier(
                n_estimators=300, max_depth=6,
                min_samples_leaf=20, max_features="sqrt",
                random_state=42, n_jobs=-1
            ))
        ]),
        "XGBoost": Pipeline([
            ("scaler", StandardScaler()),
            ("clf", XGBClassifier(
                n_estimators=300, max_depth=4,
                learning_rate=0.05, subsample=0.8,
                colsample_bytree=0.8, reg_alpha=0.1,
                reg_lambda=1.0, use_label_encoder=False,
                eval_metric="logloss", random_state=42,
                verbosity=0
            ))
        ]),
    }

# ─── Sidebar ──────────────────────────────────────────────────────────────────
with st.sidebar:
    st.title("⚙️ Settings")
    st.divider()

    ticker = st.selectbox(
        "Ticker",
        ["AAPL", "MSFT", "GOOGL", "AMZN", "TSLA"],
        index=0,
        help="AAPL is the primary focus of this project."
    )

    start_date = st.date_input("Data start", value=pd.Timestamp("2015-01-01"))

    test_size = st.slider(
        "Test set fraction",
        min_value=0.10, max_value=0.35,
        value=0.20, step=0.05,
        help="Data is split chronologically — no leakage."
    )

    conf_threshold = st.slider(
        "Confidence threshold",
        min_value=0.45, max_value=0.70,
        value=0.55, step=0.01,
        help="Only predict when model is at least this confident."
    )

    st.divider()
    st.markdown('<div class="sidebar-note">No MySQL needed — data fetched live from Yahoo Finance.</div>', unsafe_allow_html=True)
    st.divider()

    run_btn = st.button("🚀 Run Pipeline", use_container_width=True, type="primary")

# ─── Header ───────────────────────────────────────────────────────────────────
st.title("📈 AAPL Stock Direction Classifier")
st.caption(
    "End-to-end ML pipeline · Random Forest + XGBoost · "
    "20+ technical indicators · Next-day Up/Down prediction"
)
st.divider()

# ─── Main pipeline ────────────────────────────────────────────────────────────
if run_btn:

    # 1 ── Fetch data
    with st.status(f"Fetching {ticker} data from Yahoo Finance …", expanded=True) as status:
        raw = yf.download(ticker, start=str(start_date), auto_adjust=True, progress=False)
        if raw.empty:
            st.error("No data returned. Check ticker or date range.")
            st.stop()
        # Flatten MultiIndex if present
        if isinstance(raw.columns, pd.MultiIndex):
            raw.columns = raw.columns.get_level_values(0)
        raw = raw[["Open", "High", "Low", "Close", "Volume"]].dropna()
        status.update(
            label=f"✅ {len(raw):,} rows · {raw.index[0].date()} → {raw.index[-1].date()}",
            state="complete"
        )

    # 2 ── Feature engineering
    with st.status("Engineering features …", expanded=False) as status:
        feat_df   = engineer_features(raw)
        feat_cols = get_feature_cols(feat_df)
        clean_df  = feat_df[feat_cols + ["target", "Close"]].dropna()
        X         = clean_df[feat_cols]
        y         = clean_df["target"]
        X_train, X_test, y_train, y_test = time_split(X, y, test_size)
        status.update(
            label=f"✅ {len(feat_cols)} features | Train: {len(X_train):,} · Test: {len(X_test):,}",
            state="complete"
        )

    # 3 ── Train
    with st.status("Training Random Forest + XGBoost …", expanded=False) as status:
        models = build_models()
        for name, pipe in models.items():
            pipe.fit(X_train, y_train)
        status.update(label="✅ Models trained", state="complete")

    # 4 ── Evaluate
    results = {}
    for name, pipe in models.items():
        preds = pipe.predict(X_test)
        proba = pipe.predict_proba(X_test)[:, 1]
        cm    = confusion_matrix(y_test, preds)
        results[name] = {
            "pipe":  pipe,
            "preds": preds,
            "proba": proba,
            "cm":    cm,
            "acc":   accuracy_score(y_test, preds),
            "auc":   roc_auc_score(y_test, proba),
        }

    # ── Tomorrow's prediction ──────────────────────────────────────────────────
    last_row = X.iloc[[-1]]
    last_date = X.index[-1].date()

    st.subheader("🔮 Tomorrow's Prediction")
    pred_cols = st.columns(len(models))

    for i, (name, r) in enumerate(results.items()):
        prob      = r["pipe"].predict_proba(last_row)[0, 1]
        confident = prob >= conf_threshold or prob <= (1 - conf_threshold)
        direction = "UP ▲" if prob >= 0.5 else "DOWN ▼"
        css_cls   = "pred-up" if prob >= 0.5 else "pred-down"
        conf_pct  = max(prob, 1 - prob)

        with pred_cols[i]:
            st.markdown(f"""
            <div class="pred-card">
                <div class="pred-label">{name}</div>
                <div class="{css_cls}">{direction}</div>
                <div class="pred-conf">confidence: {conf_pct:.1%}</div>
                <div class="pred-label" style="margin-top:0.5rem;">
                    {'✓ above threshold' if confident else '⚠ below threshold — skip trade'}
                </div>
            </div>
            """, unsafe_allow_html=True)

    st.caption(f"Based on data through {last_date} · Threshold: {conf_threshold:.0%} · Not financial advice.")
    st.divider()

    # ── Tabs ──────────────────────────────────────────────────────────────────
    tab_metrics, tab_roc, tab_cm, tab_dist, tab_fi, tab_price, tab_data = st.tabs([
        "📊 Metrics",
        "📉 ROC Curve",
        "🔢 Confusion Matrix",
        "📦 Probability Distribution",
        "🌲 Feature Importance",
        "🕯 Price History",
        "🔍 Recent Indicators",
    ])

    # ── Tab 1: Metrics ─────────────────────────────────────────────────────────
    with tab_metrics:
        st.subheader("Model performance")

        m_cols = st.columns(len(models))
        for i, (name, r) in enumerate(m_cols):
            pass  # avoid overwriting

        for i, (name, r) in enumerate(results.items()):
            with m_cols[i]:
                st.markdown(f"**{name}**")
                c1, c2 = st.columns(2)
                c1.metric("Accuracy", f"{r['acc']:.4f}")
                c2.metric("ROC-AUC", f"{r['auc']:.4f}")
                rep = classification_report(
                    y_test, r["preds"],
                    target_names=["Down", "Up"],
                    output_dict=True
                )
                rep_df = pd.DataFrame(rep).T.round(3)
                st.dataframe(rep_df, use_container_width=True)

        st.divider()
        st.subheader("Accuracy vs ROC-AUC")
        names = list(results.keys())
        fig_bar = go.Figure()
        fig_bar.add_trace(go.Bar(
            name="Accuracy", x=names,
            y=[results[n]["acc"] for n in names],
            marker_color="#388bfd"
        ))
        fig_bar.add_trace(go.Bar(
            name="ROC-AUC", x=names,
            y=[results[n]["auc"] for n in names],
            marker_color="#3fb950"
        ))
        fig_bar.add_hline(y=0.5, line_dash="dot", line_color="red",
                          annotation_text="random baseline")
        fig_bar.update_layout(
            barmode="group", yaxis_range=[0, 1],
            height=320, margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=1.08),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_bar, use_container_width=True)

        # Confidence-threshold analysis
        st.subheader("Accuracy at different confidence thresholds")
        thresh_rows = []
        for name, r in results.items():
            for t in np.arange(0.45, 0.76, 0.05):
                mask = (r["proba"] >= t) | (r["proba"] <= 1 - t)
                if mask.sum() == 0:
                    continue
                acc_t   = accuracy_score(y_test[mask], r["preds"][mask])
                covered = mask.mean()
                thresh_rows.append({"Model": name, "Threshold": round(t, 2), "Accuracy": acc_t, "Coverage": covered})

        thresh_df = pd.DataFrame(thresh_rows)
        fig_thresh = px.line(
            thresh_df, x="Threshold", y="Accuracy",
            color="Model", markers=True,
            labels={"Accuracy": "Accuracy on confident predictions"},
            color_discrete_map={"Random Forest": "#388bfd", "XGBoost": "#3fb950"}
        )
        fig_thresh.add_hline(y=0.5, line_dash="dot", line_color="red",
                             annotation_text="random baseline")
        fig_thresh.update_layout(
            height=320, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_thresh, use_container_width=True)
        st.caption("Higher threshold = fewer trades but higher accuracy on those taken.")

    # ── Tab 2: ROC Curve ───────────────────────────────────────────────────────
    with tab_roc:
        st.subheader("ROC curve (test set)")
        fig_roc = go.Figure()
        fig_roc.add_trace(go.Scatter(
            x=[0, 1], y=[0, 1],
            mode="lines", line=dict(dash="dot", color="gray"),
            name="Random baseline"
        ))
        colors = {"Random Forest": "#388bfd", "XGBoost": "#3fb950"}
        for name, r in results.items():
            fpr, tpr, _ = roc_curve(y_test, r["proba"])
            fig_roc.add_trace(go.Scatter(
                x=fpr, y=tpr,
                mode="lines",
                name=f"{name} (AUC={r['auc']:.3f})",
                line=dict(color=colors[name], width=2)
            ))
        fig_roc.update_layout(
            xaxis_title="False positive rate",
            yaxis_title="True positive rate",
            height=420, margin=dict(t=20, b=20),
            legend=dict(orientation="h", y=-0.2),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_roc, use_container_width=True)

    # ── Tab 3: Confusion Matrix ────────────────────────────────────────────────
    with tab_cm:
        st.subheader("Confusion matrix")
        cm_cols = st.columns(len(results))
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
                fig_cm.update_layout(height=340, margin=dict(t=50, b=20),
                                     paper_bgcolor="rgba(0,0,0,0)")
                st.plotly_chart(fig_cm, use_container_width=True)

                tn, fp, fn, tp = cm.ravel()
                a, b = st.columns(2)
                a.metric("True UP (TP)", int(tp))
                b.metric("True DOWN (TN)", int(tn))
                a.metric("False UP (FP)", int(fp))
                b.metric("False DOWN (FN)", int(fn))
                precision_up = tp / (tp + fp + 1e-9)
                recall_up    = tp / (tp + fn + 1e-9)
                st.caption(f"UP precision: {precision_up:.1%} · UP recall: {recall_up:.1%}")

    # ── Tab 4: Probability Distribution ───────────────────────────────────────
    with tab_dist:
        st.subheader("Predicted probability distribution")
        dist_cols = st.columns(len(results))
        for i, (name, r) in enumerate(results.items()):
            with dist_cols[i]:
                fig_h = go.Figure()
                fig_h.add_trace(go.Histogram(
                    x=r["proba"], nbinsx=30,
                    marker_color="#388bfd", opacity=0.8,
                    name="Predicted P(Up)"
                ))
                fig_h.add_vline(x=0.5, line_dash="dash", line_color="gray",
                                annotation_text="0.5")
                fig_h.add_vline(x=conf_threshold, line_dash="dot", line_color="#d2a679",
                                annotation_text=f"threshold {conf_threshold:.2f}")
                fig_h.add_vline(x=1 - conf_threshold, line_dash="dot", line_color="#d2a679")
                fig_h.update_layout(
                    title=name,
                    xaxis_title="P(next day UP)",
                    yaxis_title="Count",
                    height=340, margin=dict(t=50, b=20),
                    showlegend=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_h, use_container_width=True)

                # Calibration summary
                buckets = pd.cut(r["proba"], bins=[0, 0.3, 0.4, 0.5, 0.6, 0.7, 1.0])
                cal = (
                    pd.DataFrame({"prob": r["proba"], "actual": y_test.values, "bucket": buckets})
                    .groupby("bucket", observed=True)["actual"]
                    .agg(["mean", "count"])
                    .rename(columns={"mean": "Actual UP rate", "count": "N"})
                )
                cal.index = cal.index.astype(str)
                st.dataframe(cal.round(3), use_container_width=True)

    # ── Tab 5: Feature Importance ──────────────────────────────────────────────
    with tab_fi:
        st.subheader("Top 15 feature importances")
        fi_cols = st.columns(len(results))
        for i, (name, r) in enumerate(results.items()):
            with fi_cols[i]:
                clf   = r["pipe"].named_steps["clf"]
                imps  = clf.feature_importances_
                fi_df = (
                    pd.DataFrame({"feature": feat_cols, "importance": imps})
                    .sort_values("importance", ascending=False)
                    .head(15)
                )
                fig_fi = px.bar(
                    fi_df.sort_values("importance"),
                    x="importance", y="feature",
                    orientation="h",
                    title=name,
                    color="importance",
                    color_continuous_scale="Teal",
                )
                fig_fi.update_layout(
                    height=460, margin=dict(t=50, b=20),
                    showlegend=False,
                    coloraxis_showscale=False,
                    paper_bgcolor="rgba(0,0,0,0)",
                    plot_bgcolor="rgba(0,0,0,0)",
                )
                st.plotly_chart(fig_fi, use_container_width=True)

    # ── Tab 6: Price History ───────────────────────────────────────────────────
    with tab_price:
        st.subheader(f"{ticker} price history with train/test split")
        split_idx = int(len(clean_df) * (1 - test_size))
        train_price = clean_df["Close"].iloc[:split_idx]
        test_price  = clean_df["Close"].iloc[split_idx:]

        fig_p = go.Figure()
        fig_p.add_trace(go.Scatter(
            x=train_price.index, y=train_price,
            name="Train", line=dict(color="#388bfd", width=1.5)
        ))
        fig_p.add_trace(go.Scatter(
            x=test_price.index, y=test_price,
            name="Test", line=dict(color="#f0883e", width=1.5)
        ))
        fig_p.update_layout(
            height=380, margin=dict(t=20, b=20),
            xaxis_title="Date", yaxis_title="Close price (USD)",
            legend=dict(orientation="h", y=1.1),
            hovermode="x unified",
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_p, use_container_width=True)

        # Candlestick — last 90 days
        st.subheader("Last 90 trading days (candlestick)")
        recent = raw.iloc[-90:]
        fig_candle = go.Figure(go.Candlestick(
            x=recent.index,
            open=recent["Open"], high=recent["High"],
            low=recent["Low"],  close=recent["Close"],
            increasing_line_color="#3fb950",
            decreasing_line_color="#f85149",
        ))
        fig_candle.update_layout(
            xaxis_rangeslider_visible=False,
            height=380, margin=dict(t=20, b=20),
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
        )
        st.plotly_chart(fig_candle, use_container_width=True)

    # ── Tab 7: Recent indicator values ────────────────────────────────────────
    with tab_data:
        st.subheader("Latest 30 rows — engineered features")
        display_cols = [
            "rsi14", "macd", "macd_signal", "bb_width", "bb_pct",
            "atr_ratio", "vol_spike", "sma_5_ratio", "sma_20_ratio",
            "ret_1d", "ret_5d", "body", "upper_shadow", "lower_shadow"
        ]
        latest_df = feat_df[display_cols].tail(30).round(4)
        st.dataframe(latest_df[::-1], use_container_width=True, height=460)

        st.divider()
        st.subheader("Signal summary (last row)")
        last = latest_df.iloc[-1]
        s1, s2, s3 = st.columns(3)
        s1.metric("RSI-14",       f"{last['rsi14']:.1f}",
                  delta="overbought" if last['rsi14'] > 70 else ("oversold" if last['rsi14'] < 30 else "neutral"))
        s2.metric("MACD hist",    f"{last['macd_hist']:.4f}",
                  delta="bullish" if last['macd_hist'] > 0 else "bearish")
        s3.metric("BB %B",        f"{last['bb_pct']:.2f}",
                  delta="near upper band" if last['bb_pct'] > 0.8 else (
                      "near lower band" if last['bb_pct'] < 0.2 else "mid-band"))

else:
    # ── Landing state ──────────────────────────────────────────────────────────
    st.info("Configure settings in the sidebar, then click **🚀 Run Pipeline**.", icon="👈")

    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.markdown("### 📡 Live data")
        st.markdown("Fetches AAPL history from Yahoo Finance — no MySQL required in this demo.")
    with col2:
        st.markdown("### ⚙️ 20+ features")
        st.markdown("RSI, MACD, Bollinger Bands, ATR, SMA ratios, volume spikes, candle structure.")
    with col3:
        st.markdown("### 🤖 Two models")
        st.markdown("Random Forest + XGBoost trained with chronological train/test split.")
    with col4:
        st.markdown("### 🎯 Thresholded pred")
        st.markdown("Only predict when model confidence clears your chosen threshold.")

    st.divider()
    st.markdown("#### Pipeline flow")
    st.code("""
Yahoo Finance → feature engineering (20+ indicators)
    → chronological train/test split
    → Random Forest + XGBoost training
    → Accuracy · ROC-AUC · Confusion matrix · Feature importance
    → Tomorrow's UP/DOWN prediction with confidence score
    """, language="text")