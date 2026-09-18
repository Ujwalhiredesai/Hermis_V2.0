# ==================== HERMIS SUBSCRIBER / INFERENCE BRIDGE ====================
# This is the missing middle piece: it subscribes to the MQTT stream your
# publisher script sends, runs each reading through your trained LSTM model
# to check for anomalies, estimates RUL, ranks likely failure modes, and logs
# EVERY reading (not just anomalies) so the Streamlit dashboard has continuous
# data to chart.
#
# Run this in a SEPARATE process/terminal from your publisher script.
# It needs to be running BEFORE (or at the same time as) the publisher,
# so it doesn't miss messages.

import json
import csv
import os
from datetime import datetime
from collections import deque

import numpy as np
import joblib
import tensorflow as tf
import paho.mqtt.client as mqtt

# ---- Import your ranked failure classifier ----
# (paste ranked_failure_classifier.py in the same folder, or import it)
from ranked_failure_classifier import classify_failure_ranked
import pandas as pd

# ---------------- Configuration ----------------
BROKER = "broker.hivemq.com"
PORT = 1883
TOPIC = "factory/machine_01/sensors"
LOG_FILE = "factory_audit_log.csv"
WINDOW_SIZE = 50          # must match the window size the LSTM was trained on
VIBRATION_ANOMALY_THRESHOLD = 2.5   # z-score-like threshold on prediction error -> tune this using your validation data, this is a placeholder starting point

# ---------------- Load trained model + scaler ----------------
print("Loading trained LSTM model and scaler...")
model = tf.keras.models.load_model("hermis_lstm_v1.keras")
scaler = joblib.load("hermis_scaler.pkl")
print("Model and scaler loaded.")

# Rolling window of the last WINDOW_SIZE scaled vibration readings,
# needed because the LSTM predicts the NEXT reading from the last 50.
vibration_window = deque(maxlen=WINDOW_SIZE)

# Rolling history of raw readings, used to build a small dataframe for the
# ranked failure-mode classifier (which needs a "recent trend", not just one point).
recent_readings = deque(maxlen=30)

# ---------------- Logging setup ----------------
if not os.path.exists(LOG_FILE):
    with open(LOG_FILE, mode='w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            "timestamp", "machine_id", "vibration", "temperature",
            "prediction", "anomaly_score", "rul_minutes", "top_failure_mode", "top_failure_prob"
        ])


def log_event(timestamp, machine_id, v, t, prediction, anomaly_score, rul_minutes, top_mode, top_prob):
    with open(LOG_FILE, mode='a', newline='') as f:
        writer = csv.writer(f)
        writer.writerow([
            timestamp, machine_id, f"{v:.3f}", f"{t:.2f}",
            prediction, f"{anomaly_score:.3f}" if anomaly_score is not None else "",
            rul_minutes if rul_minutes is not None else "",
            top_mode, f"{top_prob:.1f}" if top_prob is not None else ""
        ])


def estimate_rul(current_vibration, predicted_next_vibration, failure_threshold=1.2):
    """
    Very simple RUL estimate: if vibration is rising, project forward using the
    per-step rate of change implied by (predicted_next - current). This is a
    naive linear projection, same honesty level as the RUL logic in the main
    dashboard -- not a validated survival model, just a directional estimate.
    """
    step_change = predicted_next_vibration - current_vibration
    if step_change <= 0 or current_vibration >= failure_threshold:
        return None
    steps_remaining = (failure_threshold - current_vibration) / step_change
    return max(0, int(steps_remaining))  # in "readings" -- convert to time using your stream interval


def on_connect(client, userdata, flags, rc):
    if rc == 0:
        print(f"Connected to broker. Subscribing to {TOPIC}...")
        client.subscribe(TOPIC)
    else:
        print(f"Connection failed with code {rc}")


def on_message(client, userdata, msg):
    try:
        payload = json.loads(msg.payload.decode())
        timestamp = payload["timestamp"]
        v = float(payload["vibration_rms"])
        t = float(payload["temperature"])
    except (json.JSONDecodeError, KeyError, ValueError) as e:
        print(f"Skipping malformed message: {e}")
        return

    # ---- 1. Scale and add to rolling window for LSTM ----
    v_scaled = scaler.transform(np.array([[v]]))[0][0]
    vibration_window.append(v_scaled)

    prediction_label = "COLLECTING"
    anomaly_score = None
    rul_minutes = None
    top_mode, top_prob = "N/A", None

    # ---- 2. Run LSTM inference once we have enough history ----
    if len(vibration_window) == WINDOW_SIZE:
        window_array = np.array(vibration_window).reshape(1, WINDOW_SIZE, 1)
        predicted_scaled = model.predict(window_array, verbose=0)[0][0]

        # Compare prediction to actual next reading's error to flag anomalies.
        # NOTE: this treats "how wrong was the LSTM's last prediction" as the
        # anomaly signal -- prediction error spikes when behavior deviates from
        # the learned normal pattern. anomaly_score here is that raw error,
        # not a calibrated probability -- be accurate about that if asked.
        error = abs(v_scaled - predicted_scaled)
        anomaly_score = float(error)
        prediction_label = "ANOMALY" if error > VIBRATION_ANOMALY_THRESHOLD else "NORMAL"

        # Inverse-transform the prediction back to real units for RUL estimation
        predicted_next_real = scaler.inverse_transform([[predicted_scaled]])[0][0]
        rul_minutes = estimate_rul(v, predicted_next_real)

    # ---- 3. Ranked failure mode classification ----
    recent_readings.append({"power_watts": 0.0, "temperature_c": t, "vibration": v})
    # NOTE: power_watts is not in this dataset (bearing dataset has no power
    # channel), so it's held at 0.0 here -- the classifier will just treat
    # power-related failure modes as always low-probability for this stream.
    # If you add a power sensor later, wire its real value in here instead.
    if len(recent_readings) >= 20:
        recent_df = pd.DataFrame(list(recent_readings))
        ranked = classify_failure_ranked(recent_df, anomaly_score=min(anomaly_score, 1.0) if anomaly_score else None)
        top_mode, top_prob = ranked[0]

    # ---- 4. Log EVERY reading (not just anomalies) so dashboard has a full trend ----
    log_event(timestamp, "M-01", v, t, prediction_label, anomaly_score, rul_minutes, top_mode, top_prob)

    icon = "🚨" if prediction_label == "ANOMALY" else ("⏳" if prediction_label == "COLLECTING" else "✅")
    print(f"{icon} {timestamp} | vib={v:.3f} temp={t:.1f} | {prediction_label} | top mode: {top_mode} ({top_prob if top_prob else '--'}%)")


# ---------------- Run ----------------
client = mqtt.Client(client_id="Virtual_Machine_01_Subscriber")
client.on_connect = on_connect
client.on_message = on_message

print(f"Connecting to broker at {BROKER}:{PORT}...")
client.connect(BROKER, PORT)
print("Listening for sensor data... (Ctrl+C to stop)")
client.loop_forever()
