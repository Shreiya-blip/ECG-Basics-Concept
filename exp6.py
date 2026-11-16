import wfdb
import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import butter, sosfiltfilt
from scipy.fft import rfft, rfftfreq

# Load MIT-BIH Record 100
record = wfdb.rdrecord('100', pn_dir='mitdb')
ecg = record.p_signal[:, 0]
fs = record.fs

# Select first 30 seconds
duration = 30
N = int(fs * duration)

signal = ecg[:N]
time = np.arange(N) / fs

# Low-pass filter parameters
cutoff = 40
order = 4
nyquist = fs / 2

# Normalized cutoff frequency
Wn = cutoff / nyquist

# Design Butterworth low-pass filter
sos = butter(
    order,
    Wn,
    btype='lowpass',
    output='sos'
)

# Apply filter
filtered_ecg = sosfiltfilt(sos, signal)

print("Sampling frequency:", fs)
print("Cutoff frequency:", cutoff)
print("Filter order:", order)
print("Normalized cutoff:", Wn)
print("Filter coefficients:")
print(sos)
plt.figure(figsize=(14, 5))

plt.plot(time, signal, label='Original ECG')
plt.plot(time, filtered_ecg, label='Low-pass filtered ECG')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.title('Original ECG vs Low-pass Filtered ECG')
plt.legend()
plt.grid()

plt.show()
freq = rfftfreq(N, 1/fs)

fft_original = np.abs(rfft(signal)) / N
fft_filtered = np.abs(rfft(filtered_ecg)) / N

plt.figure(figsize=(14, 5))

plt.plot(freq, fft_original, label='Original ECG')
plt.plot(freq, fft_filtered, label='Filtered ECG')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Comparison Before and After Low-pass Filtering')
plt.xlim(0, 100)
plt.legend()
plt.grid()

plt.show()
index_60 = np.argmin(np.abs(freq - 60))

magnitude_before = fft_original[index_60]
magnitude_after = fft_filtered[index_60]

attenuation = 20 * np.log10(
    magnitude_before / magnitude_after
)

print("Frequency:", freq[index_60], "Hz")
print("Magnitude before:", magnitude_before)
print("Magnitude after:", magnitude_after)
print("Attenuation:", attenuation, "dB")