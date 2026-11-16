import wfdb
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import butter, sosfiltfilt, find_peaks

# Load Record 100
record = wfdb.rdrecord('100', pn_dir='mitdb')
ecg = record.p_signal[:, 0]
fs = record.fs

# Select 30 seconds
duration = 30
N = int(duration * fs)

signal = ecg[:N]
time = np.arange(N) / fs

# Band-pass filtering
sos = butter(
    4,
    [0.5, 40],
    btype='bandpass',
    fs=fs,
    output='sos'
)

filtered_ecg = sosfiltfilt(sos, signal)
# Calculate maximum amplitude
max_amplitude = np.max(filtered_ecg)

# Choose threshold
threshold_ratio = 0.50
threshold = threshold_ratio * max_amplitude

# Detect peaks
detected_peaks, properties = find_peaks(
    filtered_ecg,
    height=threshold,
    distance=int(0.20 * fs),
    prominence=0.1
)

print("Threshold:", threshold)
print("Detected beats:", len(detected_peaks))
threshold_ratios = [0.80, 0.50, 0.20]

for ratio in threshold_ratios:

    threshold = ratio * np.max(filtered_ecg)

    peaks, properties = find_peaks(
        filtered_ecg,
        height=threshold,
        distance=int(0.20 * fs),
        prominence=0.1
    )

    print(
        "Threshold ratio:", ratio,
        "Threshold:", threshold,
        "Detected beats:", len(peaks)
    )
fig, axes = plt.subplots(3, 1, figsize=(14, 10))

for ax, ratio in zip(axes, threshold_ratios):

    threshold = ratio * np.max(filtered_ecg)

    peaks, _ = find_peaks(
        filtered_ecg,
        height=threshold,
        distance=int(0.20 * fs),
        prominence=0.1
    )

    ax.plot(time, filtered_ecg, label='Filtered ECG')
    ax.scatter(
        peaks / fs,
        filtered_ecg[peaks],
        color='red',
        label='Detected R-peaks'
    )

    ax.set_title(f'Threshold ratio = {ratio}')
    ax.set_xlabel('Time (seconds)')
    ax.set_ylabel('Amplitude (mV)')
    ax.legend()
    ax.grid()

plt.tight_layout()
plt.show()
# Load reference annotations
ann = wfdb.rdann('100', 'atr', pn_dir='mitdb')

beat_symbols = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
    'V', 'r', 'F', 'e', 'j', 'n', 'E', '/',
    'f', 'Q'
}

reference_peaks = np.array([
    sample
    for sample, symbol in zip(ann.sample, ann.symbol)
    if sample < N and symbol in beat_symbols
])

print("Reference beats:", len(reference_peaks))
def evaluate_detector(detected, reference, fs, tolerance=0.15):

    tolerance_samples = int(tolerance * fs)

    matched_reference = set()
    matched_detected = set()

    for i, peak in enumerate(detected):

        differences = np.abs(reference - peak)

        candidates = np.where(
            differences <= tolerance_samples
        )[0]

        for j in candidates:
            if j not in matched_reference:
                matched_reference.add(j)
                matched_detected.add(i)
                break

    TP = len(matched_reference)
    FP = len(detected) - TP
    FN = len(reference) - TP

    sensitivity = TP / (TP + FN) * 100 if TP + FN else 0

    precision = TP / (TP + FP) * 100 if TP + FP else 0

    return TP, FP, FN, sensitivity, precision
for ratio in threshold_ratios:

    threshold = ratio * np.max(filtered_ecg)

    peaks, _ = find_peaks(
        filtered_ecg,
        height=threshold,
        distance=int(0.20 * fs),
        prominence=0.1
    )

    TP, FP, FN, sensitivity, precision = evaluate_detector(
        peaks,
        reference_peaks,
        fs
    )

    print("\nThreshold ratio:", ratio)
    print("True Positives:", TP)
    print("False Positives:", FP)
    print("False Negatives:", FN)
    print("Sensitivity:", sensitivity)
    print("Precision:", precision)