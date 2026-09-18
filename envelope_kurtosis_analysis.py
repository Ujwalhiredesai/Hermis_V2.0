# ==================== ENVELOPE ANALYSIS + KURTOSIS TRENDING ====================
# Isolates early-stage bearing fault impulses from environmental/background noise.
#
# WHY THIS MATTERS (and what it adds over RMS/Isolation Forest):
# RMS and simple anomaly detection only notice a problem once the OVERALL
# ENERGY of the signal has risen enough to be visible in aggregate. Early-stage
# bearing faults don't raise overall energy much -- they add short, sharp
# impulsive shocks buried inside otherwise-normal-looking vibration. Two
# techniques specifically target that:
#
#   1. KURTOSIS TRENDING: a healthy (Gaussian-like) vibration signal has
#      kurtosis ~3. Impulsive faults push kurtosis well above 3, often
#      WEEKS before RMS shows any visible rise. Trending kurtosis over time
#      is a standard early-warning indicator in condition monitoring.
#
#   2. ENVELOPE ANALYSIS: bearing defects excite a high-frequency structural
#      resonance every time the damaged surface makes contact. That repeating
#      impact rate is invisible in the raw time signal (buried under the
#      resonance), but becomes clearly visible as a peak in the envelope
#      spectrum -- computed via bandpass filter -> Hilbert transform ->
#      FFT of the envelope.
#
# Both are standard, explainable, peer-reviewed techniques (not black-box ML)
# -- this is exactly the kind of thing that holds up under technical
# questioning, because the physics behind each step can be explained plainly.

import numpy as np
import pandas as pd
import glob
from scipy import signal
from scipy.stats import kurtosis
import matplotlib.pyplot as plt


# ---------------- Configuration ----------------
SAMPLE_RATE_HZ = 25600     # FEMTO/PRONOSTIA dataset acquisition rate
                            # (confirm against your dataset's documentation --
                            # this is the commonly cited rate for this dataset,
                            # verify before quoting it publicly)

# Bandpass range around the structural resonance the bearing impacts excite.
# This is dataset/rig-specific -- a broad starting range that usually
# captures bearing resonances on small test rigs. Should be tuned by looking
# at the raw signal's frequency spectrum first (see find_resonance_band below)
# rather than trusted blindly.
DEFAULT_BANDPASS_LOW = 2000   # Hz
DEFAULT_BANDPASS_HIGH = 6000  # Hz


# ---------------- Kurtosis Trending ----------------
def compute_kurtosis_trend(file_list, vibration_col_index=4):
    """
    Computes kurtosis for each raw accelerometer file in sequence.
    Returns a list of kurtosis values, one per file (i.e. per time snapshot).

    A rising trend, especially crossing well above ~3-4, is the early
    indicator this function exists to surface.
    """
    kurt_values = []
    for f in file_list:
        df = pd.read_csv(f, header=None)
        vib = df.iloc[:, vibration_col_index].values
        k = kurtosis(vib, fisher=False)  # fisher=False -> normal distribution baseline is 3, not 0
        kurt_values.append(k)
    return kurt_values


def plot_kurtosis_trend(kurt_values, rms_values=None, title="Kurtosis Trend (Early Fault Indicator)"):
    fig, ax1 = plt.subplots(figsize=(12, 6))
    ax1.plot(kurt_values, color='orange', label='Kurtosis')
    ax1.axhline(y=3, color='gray', linestyle='--', alpha=0.6, label='Gaussian baseline (kurtosis=3)')
    ax1.set_xlabel('Time (File Index)')
    ax1.set_ylabel('Kurtosis', color='orange')
    ax1.tick_params(axis='y', labelcolor='orange')
    ax1.legend(loc='upper left')

    if rms_values is not None:
        ax2 = ax1.twinx()
        ax2.plot(rms_values, color='cyan', alpha=0.5, label='RMS')
        ax2.set_ylabel('RMS (g)', color='cyan')
        ax2.tick_params(axis='y', labelcolor='cyan')
        ax2.legend(loc='upper right')

    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


# ---------------- Envelope Analysis ----------------
def find_resonance_band(raw_signal, sample_rate=SAMPLE_RATE_HZ, n_bands=8):
    """
    Helper to SUGGEST a bandpass range instead of guessing blindly.
    Splits the spectrum into bands and returns the band with the most energy
    as a starting candidate for the bandpass filter -- a rough but honest
    way to pick the resonance region rather than hardcoding a number with
    no justification.
    """
    freqs, psd = signal.welch(raw_signal, fs=sample_rate, nperseg=1024)
    band_edges = np.linspace(500, sample_rate / 2, n_bands + 1)
    band_energies = []
    for i in range(n_bands):
        mask = (freqs >= band_edges[i]) & (freqs < band_edges[i + 1])
        band_energies.append(np.sum(psd[mask]))
    best_band = np.argmax(band_energies)
    return band_edges[best_band], band_edges[best_band + 1]


def envelope_analysis(raw_signal, sample_rate=SAMPLE_RATE_HZ,
                       low_hz=DEFAULT_BANDPASS_LOW, high_hz=DEFAULT_BANDPASS_HIGH):
    """
    Standard envelope analysis pipeline:
      1. Bandpass filter around the structural resonance
      2. Hilbert transform to extract the envelope (amplitude modulation)
      3. FFT of the envelope to reveal the repeating fault impact rate

    Returns (envelope_freqs, envelope_spectrum) for plotting/peak-finding.
    """
    nyquist = sample_rate / 2
    low = low_hz / nyquist
    high = min(high_hz / nyquist, 0.99)  # keep below Nyquist limit

    b, a = signal.butter(4, [low, high], btype='band')
    filtered = signal.filtfilt(b, a, raw_signal)

    envelope = np.abs(signal.hilbert(filtered))
    envelope = envelope - np.mean(envelope)  # remove DC offset before FFT

    n = len(envelope)
    envelope_fft = np.abs(np.fft.rfft(envelope)) / n
    envelope_freqs = np.fft.rfftfreq(n, d=1 / sample_rate)

    return envelope_freqs, envelope_fft


def plot_envelope_spectrum(envelope_freqs, envelope_spectrum, max_freq=500, title="Envelope Spectrum"):
    """
    max_freq caps the x-axis to the range where bearing fault frequencies
    (BPFO, BPFI, BSF, FTF) typically fall -- usually well under 500 Hz,
    even though the resonance itself is in the kHz range.
    """
    mask = envelope_freqs <= max_freq
    plt.figure(figsize=(12, 5))
    plt.plot(envelope_freqs[mask], envelope_spectrum[mask], color='magenta')
    plt.xlabel('Frequency (Hz)')
    plt.ylabel('Envelope Amplitude')
    plt.title(title)
    plt.grid(True, alpha=0.3)
    plt.show()


def find_top_envelope_peaks(envelope_freqs, envelope_spectrum, max_freq=500, n_peaks=5, min_freq=1):
    """
    Returns the top N peaks in the envelope spectrum below max_freq --
    candidate fault repetition frequencies. These need to be compared against
    the THEORETICAL bearing fault frequencies (BPFO/BPFI/BSF/FTF) for your
    specific bearing geometry and shaft speed to confirm a genuine fault
    match rather than coincidental noise -- this function surfaces candidates,
    it does not confirm a diagnosis by itself.
    """
    mask = (envelope_freqs >= min_freq) & (envelope_freqs <= max_freq)
    freqs = envelope_freqs[mask]
    spec = envelope_spectrum[mask]

    peak_indices, _ = signal.find_peaks(spec, distance=5)
    peak_freqs = freqs[peak_indices]
    peak_amps = spec[peak_indices]

    top_order = np.argsort(peak_amps)[::-1][:n_peaks]
    return [(round(peak_freqs[i], 2), round(peak_amps[i], 5)) for i in top_order]


# ==================== EXAMPLE USAGE ====================
if __name__ == "__main__":
    vib_files = sorted(glob.glob("/content/dataset/Learning_set/Bearing1_1/acc_*.csv"))
    print(f"Found {len(vib_files)} vibration files.")

    # --- 1. Kurtosis trending across the full run-to-failure sequence ---
    print("Computing kurtosis trend (this scans every file, may take a moment)...")
    kurt_trend = compute_kurtosis_trend(vib_files)
    plot_kurtosis_trend(kurt_trend)

    # --- 2. Envelope analysis on the LATEST file (closest to failure) ---
    # Compare this to an envelope analysis run on an EARLY file too --
    # a healthy bearing's envelope spectrum should look flat/noisy, while
    # a degraded one should show a clear peak at the fault frequency.
    latest_file = vib_files[-1]
    df_latest = pd.read_csv(latest_file, header=None)
    raw_signal = df_latest.iloc[:, 4].values

    # Suggest a resonance band from the data itself rather than guessing
    suggested_low, suggested_high = find_resonance_band(raw_signal)
    print(f"Suggested resonance band from data: {suggested_low:.0f}-{suggested_high:.0f} Hz")

    env_freqs, env_spectrum = envelope_analysis(raw_signal, low_hz=suggested_low, high_hz=suggested_high)
    plot_envelope_spectrum(env_freqs, env_spectrum)

    top_peaks = find_top_envelope_peaks(env_freqs, env_spectrum)
    print("Top envelope spectrum peaks (Hz, amplitude):")
    for freq, amp in top_peaks:
        print(f"  {freq} Hz -> amplitude {amp}")
    print("\nCompare these frequencies against BPFO/BPFI/BSF/FTF calculated")
    print("from your bearing's geometry and shaft RPM to confirm a fault match.")
