# HERMIS AI — Industrial Predictive Maintenance Platform

A predictive maintenance dashboard for rotating industrial equipment, covering
anomaly detection, RUL (Remaining Useful Life) forecasting, ranked failure-mode
classification, and vibration signal analysis (envelope analysis + kurtosis
trending for early-stage fault isolation).

**Live demo:** [add your deployed Streamlit Cloud URL here once deployed]

## What this is

This is a working demonstration platform built to explore predictive
maintenance techniques hands-on — anomaly detection, failure-mode ranking,
and classical vibration diagnostics (envelope analysis, kurtosis trending)
applied to real bearing degradation data (FEMTO/PRONOSTIA run-to-failure
dataset).

**The deployed dashboard runs on demo data by default.** It is clearly
labeled as such in the UI (see the `DEMO DATA` badge). This is intentional —
see "Architecture & honest scope" below for why, and what a real deployment
would additionally require.

## Features

- **Fleet Overview** — real-time status across monitored assets
- **Machine Deep Dive** — per-machine RUL estimate, ranked likely failure
  mode (multi-factor probability scoring, not a single rule-based guess)
- **Signal Analysis** — envelope analysis and kurtosis trending against raw
  accelerometer data, isolating early-stage impulsive fault signatures that
  RMS-based monitoring misses
- **Alerts & Audit Log** — filterable history, CSV export
- **Reports** — fleet health scorecard, downloadable summary

## Architecture & honest scope

This repo contains two parts that are **not currently wired together in the
deployed version**, and it's worth being upfront about that:

1. **`hermis_platform.py`** — the dashboard. This is what's deployed to
   Streamlit Community Cloud. It reads `factory_audit_log.csv` if present,
   or generates clearly-labeled demo data otherwise.

2. **`hermis_subscriber.py`** — a separate script that subscribes to a live
   MQTT sensor stream, runs readings through a trained LSTM model for
   anomaly detection, estimates RUL, calls the ranked failure classifier,
   and writes results to `factory_audit_log.csv`.

**Streamlit Community Cloud cannot run `hermis_subscriber.py` as a
persistent background process alongside the dashboard.** For this to run
on genuinely live data 24/7, the subscriber needs to run continuously
somewhere else (a small always-on VM, a service like Railway/Render, or
your own machine) and write to a database the deployed dashboard can read
remotely — a local CSV won't be shared between the two environments.

That's a real next step, not yet built. Right now, this repo demonstrates
the full technique set and a working UI on synthetic-but-realistic demo
data, with the live-data pipeline built and tested separately.

## Techniques used

- **Isolation Forest** — anomaly detection on power/temperature/vibration
- **Linear regression extrapolation** — naive RUL estimation to threshold
- **LSTM** — vibration sequence forecasting (trained on FEMTO bearing
  degradation data)
- **Multi-factor ranked failure classification** — scores multiple failure
  mode candidates against current trend signals, normalized to probability
  (not a single rule-based label)
- **Kurtosis trending** — flags impulsive shock patterns that predate
  visible RMS changes
- **Envelope analysis** — bandpass filter → Hilbert transform → FFT, to
  isolate bearing fault repetition rate from the structural resonance it
  excites

## Running locally

```bash
pip install -r requirements.txt
streamlit run hermis_platform.py
```

To run the full live pipeline (requires a trained model + MQTT broker):

```bash
pip install -r requirements-subscriber.txt
python hermis_subscriber.py    # run this first, in its own terminal
```

## Dataset

Signal analysis is built against the [FEMTO/PRONOSTIA bearing degradation
dataset](https://github.com/wkzs111/phm-ieee-2012-data-challenge-dataset) —
a standard public run-to-failure benchmark for bearing condition monitoring
research.

## Author

Built by Ujjwal Hiredesai — Mechanical Engineering graduate, exploring
predictive maintenance and IIoT.
