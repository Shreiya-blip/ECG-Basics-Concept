import wfdb
import numpy as np
import matplotlib.pyplot as plt

# Load ECG record
record = wfdb.rdrecord('100', pn_dir='mitdb')
ecg = record.p_signal[:, 0]
fs = record.fs

# Select first 30 seconds
duration = 30
N = int(duration * fs)

signal = ecg[:N]
time = np.arange(N) / fs

# Load reference annotations
ann = wfdb.rdann('100', 'atr', pn_dir='mitdb')

# Select beat annotations within first 30 seconds
beat_symbols = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
    'V', 'r', 'F', 'e', 'j', 'n', 'E', '/',
    'f', 'Q'
}

r_peaks = np.array([
    sample
    for sample, symbol in zip(ann.sample, ann.symbol)
    if sample < N and symbol in beat_symbols
])

# Calculate RR intervals
rr_samples = np.diff(r_peaks)
rr_intervals = rr_samples / fs

# Calculate instantaneous heart rate
heart_rate = 60 / rr_intervals

print("Number of annotated beats:", len(r_peaks))
print("RR intervals (seconds):")
print(rr_intervals)

print("Heart rates (BPM):")
print(heart_rate)

print("Average RR interval:", np.mean(rr_intervals))
print("Average heart rate:", 60 / np.mean(rr_intervals))
plt.figure(figsize=(14, 5))

plt.plot(time, signal, label='Original ECG')

plt.scatter(
    r_peaks / fs,
    signal[r_peaks],
    color='red',
    label='Reference R-peaks',
    s=20
)

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.title('ECG Signal with Reference R-peak Annotations')
plt.legend()
plt.grid()
plt.show()
plt.figure(figsize=(12, 4))

plt.plot(
    np.arange(1, len(rr_intervals) + 1),
    rr_intervals,
    marker='o'
)

plt.xlabel('Beat interval number')
plt.ylabel('RR interval (seconds)')
plt.title('RR Interval Variation')
plt.grid()
plt.show()
plt.figure(figsize=(12, 4))

plt.plot(
    np.arange(1, len(heart_rate) + 1),
    heart_rate,
    marker='o'
)

plt.xlabel('Beat interval number')
plt.ylabel('Heart rate (BPM)')
plt.title('Instantaneous Heart Rate')
plt.grid()
plt.show()
