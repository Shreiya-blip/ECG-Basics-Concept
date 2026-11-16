import wfdb
import numpy as np
import matplotlib.pyplot as plt

from scipy.signal import iirnotch, filtfilt

# ==================================
# 1. LOAD MIT-BIH RECORD 100
# ==================================

record = wfdb.rdrecord('100', pn_dir='mitdb')

ecg = record.p_signal[:, 0]

fs = record.fs

# Select first 30 seconds
duration = 30

N = int(fs * duration)

signal = ecg[:N]

time = np.arange(N) / fs

# ==================================
# 2. DESIGN NOTCH FILTER
# ==================================

f0 = 60
Q = 30

b, a = iirnotch(f0, Q, fs)

print("Sampling Frequency:", fs)
print("Notch Frequency:", f0)
print("Quality Factor:", Q)

print("\nNumerator Coefficients:")
print(b)

print("\nDenominator Coefficients:")
print(a)

# ==================================
# 3. APPLY NOTCH FILTER
# ==================================

filtered_ecg = filtfilt(b, a, signal)

# ==================================
# 4. PLOT ECG SIGNAL
# ==================================

plt.figure(figsize=(14, 8))

plt.subplot(2, 1, 1)
plt.plot(time, signal)
plt.title("Original ECG Signal")
plt.ylabel("Amplitude (mV)")
plt.grid()

plt.subplot(2, 1, 2)
plt.plot(time, filtered_ecg)
plt.title("ECG After 60 Hz Notch Filtering")
plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")
plt.grid()

plt.tight_layout()
plt.show()
# verification using fft
# FFT of original signal
original_fft = np.abs(
    np.fft.rfft(signal - np.mean(signal))
) / N

# FFT of filtered signal
filtered_fft = np.abs(
    np.fft.rfft(filtered_ecg - np.mean(filtered_ecg))
) / N

frequencies = np.fft.rfftfreq(N, d=1/fs)

plt.figure(figsize=(12, 5))

plt.plot(
    frequencies,
    original_fft,
    label="Original ECG"
)

plt.plot(
    frequencies,
    filtered_fft,
    label="After Notch Filtering"
)

plt.xlim(45, 75)

plt.xlabel("Frequency (Hz)")
plt.ylabel("Magnitude")

plt.title("60 Hz Notch Filter – Frequency Spectrum")

plt.legend()
plt.grid()

plt.show()
#fft magnitude calculation
# fft magnitude calculation
from scipy.fft import rfft, rfftfreq
import numpy as np

# FFT of original signal
N = len(signal)
freq = rfftfreq(N, 1/fs)

fft_original = np.abs(rfft(signal)) / N

# FFT of notch-filtered signal
fft_filtered = np.abs(rfft(filtered_ecg)) / N

# Find the frequency bin closest to 60 Hz
index_60 = np.argmin(np.abs(freq - 60))

magnitude_before = fft_original[index_60]
magnitude_after = fft_filtered[index_60]

print("Frequency:", freq[index_60], "Hz")
print("Magnitude before:", magnitude_before)
print("Magnitude after:", magnitude_after)