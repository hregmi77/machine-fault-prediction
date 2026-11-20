"""Interactive dashboard: explore vibration signals, features, model results, and fleet risk.

Run:
    streamlit run app/dashboard.py
"""
import json
import sys
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

sys.path.append(str(Path(__file__).resolve().parents[1]))
from src.config import BPFI, BPFO, BSF, CLASSES, FS, MODEL_DIR, PROCESSED_DIR, WINDOW  # noqa: E402
from src.data import load_recordings  # noqa: E402
from src.features import FEATURE_COLS, envelope_spectrum, spectrum  # noqa: E402

st.set_page_config(page_title="Machine Fault Prediction", page_icon="⚙️", layout="wide")

COLORS = {"normal": "#2a9d8f", "inner_race": "#e76f51", "ball": "#e9c46a", "outer_race": "#264653"}
FAULT_LINES = {"BPFI (inner)": BPFI, "BPFO (outer)": BPFO, "BSF (ball)": BSF}


@st.cache_data
def get_recordings():
    return {r.rec_id: r for r in load_recordings()}


@st.cache_data
def get_features():
    path = PROCESSED_DIR / "features.csv"
    return pd.read_csv(path) if path.exists() else None


@st.cache_resource
def get_model():
    path = MODEL_DIR / "model.joblib"
    return joblib.load(path) if path.exists() else None


def get_metrics():
    path = MODEL_DIR / "metrics.json"
    return json.loads(path.read_text()) if path.exists() else None


recs = get_recordings()
df = get_features()
bundle = get_model()
metrics = get_metrics()

st.title("⚙️ Machine Fault Prediction from Vibration Data")
source = next(iter(recs.values())).source
st.caption(
    f"Data source: **{'CWRU Bearing Data Center' if source == 'cwru' else 'synthetic (run src.download_cwru for real data)'}** · "
    f"{len(recs)} recordings · sampled at {FS:,} Hz"
)

if df is None or bundle is None:
    st.warning("No trained model yet. Run `python -m src.train` first, then refresh.")
    st.stop()

tab_signal, tab_feat, tab_model, tab_fleet = st.tabs(
    ["📈 Signal Explorer", "🔬 Features", "🤖 Model", "🏭 Fleet Health"])

# ---------------------------------------------------------------- Signal explorer
with tab_signal:
    c1, c2 = st.columns([1, 3])
    with c1:
        rec_id = st.selectbox("Recording", list(recs),
                              format_func=lambda k: f"{k} · {recs[k].label} · {recs[k].severity} in")
        rec = recs[rec_id]
        max_start = len(rec.signal) - WINDOW
        start = st.slider("Window start (s)", 0.0, max_start / FS, 0.0, step=0.05)
        seg = rec.signal[int(start * FS): int(start * FS) + WINDOW]
        st.metric("Label", rec.label)
        st.metric("RMS", f"{np.sqrt(np.mean(seg ** 2)):.3f} g")
        k = ((seg - seg.mean()) ** 4).mean() / seg.var() ** 2
        st.metric("Kurtosis", f"{k:.2f}", help="≈3 for healthy noise; impacts push it higher")
    with c2:
        t = np.arange(len(seg)) / FS * 1000
        fig = px.line(x=t, y=seg, labels={"x": "Time (ms)", "y": "Acceleration (g)"},
                      title="Time waveform")
        fig.update_traces(line_color=COLORS[rec.label])
        st.plotly_chart(fig, use_container_width=True)

        f, m = spectrum(seg)
        fig = px.line(x=f, y=m, labels={"x": "Frequency (Hz)", "y": "Amplitude"},
                      title="FFT spectrum")
        st.plotly_chart(fig, use_container_width=True)

        ef, em = envelope_spectrum(rec.signal[: FS * 2])
        keep = ef < 500
        fig = go.Figure(go.Scatter(x=ef[keep], y=em[keep], mode="lines", name="envelope"))
        for name, f0 in FAULT_LINES.items():
            fig.add_vline(x=f0, line_dash="dash", annotation_text=name)
        fig.update_layout(title="Envelope spectrum (2 s): peaks at a defect frequency reveal the failing part",
                          xaxis_title="Frequency (Hz)", yaxis_title="Amplitude")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Features
with tab_feat:
    c1, c2 = st.columns(2)
    with c1:
        feat = st.selectbox("Feature", FEATURE_COLS, index=FEATURE_COLS.index("kurtosis"))
        fig = px.box(df, x="label", y=feat, color="label", color_discrete_map=COLORS,
                     category_orders={"label": CLASSES}, points=False,
                     title=f"{feat} by condition")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        fx = st.selectbox("X axis", FEATURE_COLS, index=FEATURE_COLS.index("env_bpfi"))
        fy = st.selectbox("Y axis", FEATURE_COLS, index=FEATURE_COLS.index("env_bpfo"))
        fig = px.scatter(df, x=fx, y=fy, color="label", color_discrete_map=COLORS,
                         category_orders={"label": CLASSES}, opacity=0.6,
                         title="How well two features separate the classes")
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Model
with tab_model:
    best = metrics["best_model"]
    cols = st.columns(len(metrics["models"]) + 1)
    cols[0].metric("Best model", best.replace("_", " ").title())
    for col, (name, m) in zip(cols[1:], metrics["models"].items()):
        col.metric(name.replace("_", " ").title(), f"{m['accuracy']:.1%} acc",
                   f"macro F1 {m['macro_f1']:.2f}", delta_color="off")
    st.caption(f"Time-based split: first 70% of each recording for training "
               f"({metrics['n_train']} windows), last 30% for testing ({metrics['n_test']} windows).")

    c1, c2 = st.columns(2)
    with c1:
        cm = np.array(metrics["confusion_matrix"])
        fig = px.imshow(cm, x=CLASSES, y=CLASSES, text_auto=True, color_continuous_scale="Blues",
                        labels={"x": "Predicted", "y": "Actual"}, title="Confusion matrix (test set)")
        st.plotly_chart(fig, use_container_width=True)
    with c2:
        imp = pd.Series(metrics["feature_importance"]).head(12).sort_values()
        fig = px.bar(imp, orientation="h", title="Top features (random forest importance)",
                     labels={"value": "Importance", "index": ""})
        fig.update_layout(showlegend=False)
        st.plotly_chart(fig, use_container_width=True)

# ---------------------------------------------------------------- Fleet health
with tab_fleet:
    st.markdown(
        "Each recording plays the role of one machine on a production line. The model scores "
        "the latest unseen data from each machine, and machines are ranked so technicians "
        "fix the riskiest ones first.")
    model = bundle["model"]
    latest = df[df.position >= 0.7]
    proba = model.predict_proba(latest[FEATURE_COLS])
    classes = list(model.classes_)
    latest = latest.assign(fault_prob=1 - proba[:, classes.index("normal")],
                           predicted=np.array(classes)[proba.argmax(1)])

    fleet = (latest.groupby("rec_id")
             .agg(actual=("label", "first"),
                  predicted=("predicted", lambda s: s.mode().iat[0]),
                  fault_risk=("fault_prob", "mean"),
                  rms=("rms", "mean"), kurtosis=("kurtosis", "mean"))
             .sort_values("fault_risk", ascending=False).reset_index())
    fleet.insert(0, "machine", [f"Station {i + 1:02d}" for i in range(len(fleet))])
    fleet["priority"] = pd.cut(fleet.fault_risk, [-0.01, 0.3, 0.7, 1.01],
                               labels=["🟢 Monitor", "🟡 Plan repair", "🔴 Fix now"])

    c1, c2, c3 = st.columns(3)
    c1.metric("Machines", len(fleet))
    c2.metric("Fix now", int((fleet.priority == "🔴 Fix now").sum()))
    c3.metric("Diagnosis matches label", f"{(fleet.actual == fleet.predicted).mean():.0%}")

    fig = px.bar(fleet, x="machine", y="fault_risk", color="predicted", color_discrete_map=COLORS,
                 category_orders={"machine": list(fleet.machine)},
                 title="Fault risk by machine (ranked)", labels={"fault_risk": "Fault risk"})
    fig.update_yaxes(range=[0, 1])
    st.plotly_chart(fig, use_container_width=True)
    st.dataframe(fleet.style.format({"fault_risk": "{:.2f}", "rms": "{:.3f}", "kurtosis": "{:.2f}"}),
                 use_container_width=True, hide_index=True)
