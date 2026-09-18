# ==================== HERMIS AI — INDUSTRIAL PREDICTIVE MAINTENANCE PLATFORM ====================
# v2.0 — Restructured from single-page demo into a multi-section industrial monitoring platform.
#
# ARCHITECTURE:
#   Fleet Overview      -> at-a-glance status across all machines
#   Machine Deep Dive   -> per-machine RUL, ranked failure modes, live trends
#   Signal Analysis      -> kurtosis trending + envelope spectrum (early fault isolation)
#   Alerts & Audit Log   -> full history, filterable, exportable
#   Reports              -> fleet health scorecard, downloadable summary
#
# DATA SOURCES:
#   Reads factory_audit_log.csv if present (populated by hermis_subscriber.py).
#   Falls back to a clearly-labeled DEMO DATA generator if no real log exists yet,
#   so the platform is always runnable and reviewable -- but never silently
#   pretends demo data is live plant data.

import streamlit as st
import pandas as pd
import numpy as np
import os
from datetime import datetime, timedelta

st.set_page_config(
    page_title="HERMIS AI | Industrial Reliability Platform",
    layout="wide",
    page_icon="⚙️",
    initial_sidebar_state="expanded"
)

# ---------------------------------------------------------------------------
# DESIGN TOKENS
# ---------------------------------------------------------------------------
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

    :root {
        --bg: #14181C;
        --panel: #1C2126;
        --panel-raised: #21262C;
        --border: #2C333A;
        --text: #E8EAED;
        --text-muted: #8B95A1;
        --accent-data: #5EA8ED;
        --accent-good: #4CAF7D;
        --accent-warn: #E8A93E;
        --accent-critical: #FF6B35;
    }

    .stApp { background-color: var(--bg); font-family: 'Inter', sans-serif; }
    [data-testid="stSidebar"] { background-color: var(--panel); border-right: 1px solid var(--border); }
    [data-testid="stSidebar"] * { font-family: 'Inter', sans-serif; }

    h1, h2, h3, h4 { color: var(--text) !important; font-family: 'Inter', sans-serif; font-weight: 600; }
    p, span, label, div { color: var(--text); }

    /* Monospace for all numeric/data readouts -- instrumentation convention */
    [data-testid="stMetricValue"] {
        font-family: 'JetBrains Mono', monospace;
        font-size: 26px;
        font-weight: 700;
        color: var(--accent-data);
    }
    [data-testid="stMetricLabel"] { color: var(--text-muted) !important; font-size: 12px; letter-spacing: 0.02em; }

    /* Flat panels, hairline borders -- not SaaS rounded cards */
    .hermis-panel {
        background: var(--panel);
        border: 1px solid var(--border);
        border-radius: 3px;
        padding: 18px 20px;
        margin-bottom: 12px;
    }
    .hermis-panel-title {
        font-size: 13px;
        color: var(--text-muted);
        margin-bottom: 10px;
        font-weight: 500;
    }

    .status-dot { display: inline-block; width: 9px; height: 9px; border-radius: 50%; margin-right: 8px; }
    .status-good { background: var(--accent-good); }
    .status-warn { background: var(--accent-warn); }
    .status-critical { background: var(--accent-critical); }

    .hermis-badge {
        display: inline-block; padding: 2px 9px; border-radius: 2px;
        font-family: 'JetBrains Mono', monospace; font-size: 11px; font-weight: 600;
        letter-spacing: 0.03em;
    }
    .badge-good { background: rgba(76,175,125,0.15); color: var(--accent-good); border: 1px solid rgba(76,175,125,0.3); }
    .badge-warn { background: rgba(232,169,62,0.15); color: var(--accent-warn); border: 1px solid rgba(232,169,62,0.3); }
    .badge-critical { background: rgba(255,107,53,0.15); color: var(--accent-critical); border: 1px solid rgba(255,107,53,0.3); }
    .badge-demo { background: rgba(139,149,161,0.15); color: var(--text-muted); border: 1px solid rgba(139,149,161,0.3); }

    hr { border-color: var(--border); }
    [data-testid="stDataFrame"] { font-family: 'JetBrains Mono', monospace; }

    [data-testid="stSidebar"] .stRadio > label { display: none; }
    [data-testid="stSidebar"] [role="radiogroup"] label {
        padding: 8px 12px; border-radius: 3px; margin-bottom: 2px;
    }
</style>
""", unsafe_allow_html=True)


# ---------------------------------------------------------------------------
# DATA LAYER — real log if present, clearly-labeled demo data otherwise
# ---------------------------------------------------------------------------
LOG_FILE = "factory_audit_log.csv"
MACHINES = ["M-01", "M-02", "M-03", "M-04"]


@st.cache_data(ttl=5)
def load_real_log():
    if os.path.exists(LOG_FILE):
        df = pd.read_csv(LOG_FILE)
        if len(df) > 0:
            return df
    return None


def generate_demo_data(n_points=200):
    """
    Clearly-labeled synthetic data for when no real audit log exists yet.
    Exists so the platform is reviewable/demoable without requiring a live
    MQTT stream + trained model to be running -- never presented as real
    plant data anywhere in the UI (see the DEMO DATA badge wherever this is used).
    """
    np.random.seed(7)
    rows = []
    start = datetime.now() - timedelta(minutes=n_points)
    for m in MACHINES:
        base_vib = np.random.uniform(0.3, 0.45)
        base_temp = np.random.uniform(50, 60)
        degrade = np.random.random() < 0.4
        for i in range(n_points):
            drift = (i / n_points) * np.random.uniform(0.3, 0.8) if degrade else 0
            vib = base_vib + drift + np.random.normal(0, 0.03)
            temp = base_temp + drift * 15 + np.random.normal(0, 0.8)
            anomaly_score = min(1.0, max(0.0, drift + np.random.normal(0, 0.05)))
            prediction = "ANOMALY" if anomaly_score > 0.55 else "NORMAL"
            rows.append({
                "timestamp": (start + timedelta(minutes=i)).strftime("%Y-%m-%d %H:%M:%S"),
                "machine_id": m, "vibration": round(vib, 3), "temperature": round(temp, 2),
                "prediction": prediction, "anomaly_score": round(anomaly_score, 3),
                "rul_minutes": int(max(5, 500 - drift * 600)) if degrade else None,
            })
    return pd.DataFrame(rows)


real_data = load_real_log()
using_demo = real_data is None
df_all = real_data if not using_demo else generate_demo_data()


def status_for_machine(mdf):
    if len(mdf) == 0:
        return "good", "NO DATA"
    latest = mdf.iloc[-1]
    score = latest.get("anomaly_score", 0) or 0
    if latest.get("prediction") == "ANOMALY" and score > 0.7:
        return "critical", "CRITICAL"
    elif latest.get("prediction") == "ANOMALY":
        return "warn", "WARNING"
    return "good", "HEALTHY"


# ---------------------------------------------------------------------------
# SIDEBAR NAVIGATION
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown("## ⚙️ HERMIS AI")
    st.markdown(
        f"<span class='hermis-badge {'badge-demo' if using_demo else 'badge-good'}'>"
        f"{'DEMO DATA' if using_demo else 'LIVE DATA'}</span>",
        unsafe_allow_html=True
    )
    st.markdown("---")
    page = st.radio(
        "Navigation",
        ["📊 Fleet Overview", "🔍 Machine Deep Dive", "📈 Signal Analysis", "🔔 Alerts & Audit Log", "📄 Reports"],
        label_visibility="collapsed"
    )
    st.markdown("---")
    st.caption("Industrial Reliability Platform")
    st.caption("v2.0")


# ---------------------------------------------------------------------------
# PAGE: FLEET OVERVIEW
# ---------------------------------------------------------------------------
if page == "📊 Fleet Overview":
    st.markdown("# Fleet Overview")
    st.caption("Real-time health status across all monitored assets")

    machine_ids = sorted(df_all["machine_id"].unique())
    cols = st.columns(len(machine_ids))
    for i, mid in enumerate(machine_ids):
        mdf = df_all[df_all["machine_id"] == mid]
        status, label = status_for_machine(mdf)
        latest = mdf.iloc[-1] if len(mdf) > 0 else None
        with cols[i]:
            st.markdown(f"""
            <div class="hermis-panel">
                <div class="hermis-panel-title">{mid}</div>
                <div style="margin-bottom:10px;">
                    <span class="status-dot status-{status}"></span>
                    <span class="hermis-badge badge-{status}">{label}</span>
                </div>
            </div>
            """, unsafe_allow_html=True)
            if latest is not None:
                st.metric("Vibration", f"{latest['vibration']:.3f} g")
                st.metric("Temperature", f"{latest['temperature']:.1f} °C")

    st.markdown("---")
    st.markdown("### Fleet Vibration Trend")
    pivot = df_all.pivot_table(index="timestamp", columns="machine_id", values="vibration", aggfunc="last")
    st.line_chart(pivot, height=320)

    critical_n = sum(1 for mid in machine_ids if status_for_machine(df_all[df_all["machine_id"] == mid])[0] == "critical")
    warn_n = sum(1 for mid in machine_ids if status_for_machine(df_all[df_all["machine_id"] == mid])[0] == "warn")
    st.markdown("---")
    c1, c2, c3 = st.columns(3)
    c1.metric("Total Assets", len(machine_ids))
    c2.metric("Warnings", warn_n)
    c3.metric("Critical", critical_n)


# ---------------------------------------------------------------------------
# PAGE: MACHINE DEEP DIVE
# ---------------------------------------------------------------------------
elif page == "🔍 Machine Deep Dive":
    st.markdown("# Machine Deep Dive")
    machine_ids = sorted(df_all["machine_id"].unique())
    selected = st.selectbox("Select asset", machine_ids)
    mdf = df_all[df_all["machine_id"] == selected].copy()

    status, label = status_for_machine(mdf)
    latest = mdf.iloc[-1]

    c1, c2, c3, c4 = st.columns(4)
    c1.markdown(f"<span class='hermis-badge badge-{status}'>{label}</span>", unsafe_allow_html=True)
    c2.metric("Vibration", f"{latest['vibration']:.3f} g")
    c3.metric("Temperature", f"{latest['temperature']:.1f} °C")
    rul_val = latest.get("rul_minutes")
    c4.metric("Est. RUL", f"{int(rul_val)} min" if pd.notna(rul_val) else "N/A")

    st.markdown("---")
    tcol1, tcol2 = st.columns(2)
    with tcol1:
        st.markdown("### Vibration Trend")
        st.line_chart(mdf.set_index("timestamp")["vibration"], height=280)
    with tcol2:
        st.markdown("### Temperature Trend")
        st.line_chart(mdf.set_index("timestamp")["temperature"], height=280)

    st.markdown("---")
    st.markdown("### Ranked Likely Failure Mode")
    st.caption("Multi-factor scoring across current trend signals, normalized to probability")
    try:
        from ranked_failure_classifier import classify_failure_ranked
        classify_df = mdf.copy()
        if "temperature_c" not in classify_df.columns:
            classify_df["temperature_c"] = classify_df["temperature"]
        if "power_watts" not in classify_df.columns:
            classify_df["power_watts"] = 0.0
        ranked = classify_failure_ranked(classify_df.tail(20))
        for fmode, prob in ranked:
            st.markdown(f"**{fmode}**")
            st.progress(min(1.0, prob / 100))
            st.caption(f"{prob:.0f}%")
    except ImportError:
        st.info("ranked_failure_classifier.py not found in this directory — place it alongside this file to enable ranked failure mode scoring.")


# ---------------------------------------------------------------------------
# PAGE: SIGNAL ANALYSIS
# ---------------------------------------------------------------------------
elif page == "📈 Signal Analysis":
    st.markdown("# Signal Analysis")
    st.caption("Envelope analysis and kurtosis trending — isolates early-stage fault impulses from background noise")

    st.markdown("""
    <div class="hermis-panel">
    <div class="hermis-panel-title">ABOUT THESE TECHNIQUES</div>
    Kurtosis trending flags impulsive shock patterns in the vibration signal that
    RMS-based monitoring misses in early stages — a healthy bearing sits near
    kurtosis ≈ 3; developing faults push it well above that, often before RMS
    shows any visible change. Envelope analysis isolates the repeating impact
    rate of a bearing fault, hidden inside the high-frequency structural
    resonance it excites, via bandpass filtering → Hilbert transform → FFT.
    </div>
    """, unsafe_allow_html=True)

    data_dir = st.text_input("Path to raw accelerometer files (e.g. Bearing1_1/acc_*.csv)", value="")

    if data_dir:
        try:
            import glob
            from envelope_kurtosis_analysis import (
                compute_kurtosis_trend, find_resonance_band,
                envelope_analysis, find_top_envelope_peaks
            )

            files = sorted(glob.glob(data_dir))
            st.caption(f"Found {len(files)} files")

            if files:
                if st.button("Run Signal Analysis"):
                    with st.spinner("Computing kurtosis trend across full run-to-failure sequence..."):
                        kurt_trend = compute_kurtosis_trend(files)

                    st.markdown("### Kurtosis Trend")
                    kdf = pd.DataFrame({"kurtosis": kurt_trend, "gaussian_baseline": [3] * len(kurt_trend)})
                    st.line_chart(kdf, height=320)

                    latest_raw = pd.read_csv(files[-1], header=None).iloc[:, 4].values
                    low, high = find_resonance_band(latest_raw)
                    st.caption(f"Auto-suggested resonance band: {low:.0f}–{high:.0f} Hz — verify against a manual spectrum review before trusting blindly")

                    env_freqs, env_spec = envelope_analysis(latest_raw, low_hz=low, high_hz=high)
                    st.markdown("### Envelope Spectrum (Latest Snapshot)")
                    edf = pd.DataFrame({"amplitude": env_spec[:2000]}, index=env_freqs[:2000])
                    st.line_chart(edf, height=280)

                    peaks = find_top_envelope_peaks(env_freqs, env_spec)
                    st.markdown("### Top Candidate Fault Frequencies")
                    st.caption("Compare against theoretical BPFO/BPFI/BSF/FTF for your bearing geometry to confirm a genuine fault match")
                    peak_df = pd.DataFrame(peaks, columns=["Frequency (Hz)", "Amplitude"])
                    st.dataframe(peak_df, use_container_width=True)
        except ImportError:
            st.error("envelope_kurtosis_analysis.py not found in this directory — place it alongside this file.")
    else:
        st.info("Enter a path to raw accelerometer CSV files above to run signal analysis.")


# ---------------------------------------------------------------------------
# PAGE: ALERTS & AUDIT LOG
# ---------------------------------------------------------------------------
elif page == "🔔 Alerts & Audit Log":
    st.markdown("# Alerts & Audit Log")

    fcol1, fcol2, fcol3 = st.columns(3)
    with fcol1:
        machine_filter = st.multiselect("Machine", sorted(df_all["machine_id"].unique()), default=None)
    with fcol2:
        status_filter = st.multiselect("Status", df_all["prediction"].unique().tolist(), default=None)
    with fcol3:
        n_rows = st.slider("Rows to show", 10, 200, 50)

    filtered = df_all.copy()
    if machine_filter:
        filtered = filtered[filtered["machine_id"].isin(machine_filter)]
    if status_filter:
        filtered = filtered[filtered["prediction"].isin(status_filter)]

    st.dataframe(filtered.tail(n_rows).sort_values("timestamp", ascending=False), use_container_width=True, height=450)

    csv = filtered.to_csv(index=False)
    st.download_button("Export filtered log (CSV)", csv, "hermis_audit_export.csv", "text/csv")


# ---------------------------------------------------------------------------
# PAGE: REPORTS
# ---------------------------------------------------------------------------
elif page == "📄 Reports":
    st.markdown("# Fleet Health Report")
    st.caption(f"Generated {datetime.now().strftime('%Y-%m-%d %H:%M')}")

    machine_ids = sorted(df_all["machine_id"].unique())
    summary_rows = []
    for mid in machine_ids:
        mdf = df_all[df_all["machine_id"] == mid]
        status, label = status_for_machine(mdf)
        anomaly_count = len(mdf[mdf["prediction"] == "ANOMALY"])
        summary_rows.append({
            "Machine": mid, "Status": label,
            "Anomaly Events": anomaly_count,
            "Avg Vibration": round(mdf["vibration"].mean(), 3),
            "Avg Temperature": round(mdf["temperature"].mean(), 2),
        })
    summary_df = pd.DataFrame(summary_rows)

    st.markdown("### Fleet Summary")
    st.dataframe(summary_df, use_container_width=True)

    st.markdown("### Anomaly Events by Asset")
    st.bar_chart(summary_df.set_index("Machine")["Anomaly Events"], height=280)

    st.markdown("---")
    report_csv = summary_df.to_csv(index=False)
    st.download_button("Download Fleet Report (CSV)", report_csv, "hermis_fleet_report.csv", "text/csv")

    if using_demo:
        st.warning("This report is generated from DEMO DATA. Connect hermis_subscriber.py to a live MQTT stream to generate reports from real plant data.")
