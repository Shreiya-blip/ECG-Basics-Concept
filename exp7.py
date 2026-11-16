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

# Band-pass filter parameters
low_cutoff = 0.5
high_cutoff = 40
order = 4

nyquist = fs / 2

low_normalized = low_cutoff / nyquist
high_normalized = high_cutoff / nyquist

# Design Butterworth band-pass filter
sos = butter(
    order,
    [low_normalized, high_normalized],
    btype='bandpass',
    output='sos'
)

# Apply filter
filtered_ecg = sosfiltfilt(sos, signal)

print("Sampling frequency:", fs)
print("Lower cutoff:", low_cutoff)
print("Upper cutoff:", high_cutoff)
print("Filter order:", order)
print("Normalized cutoff frequencies:",
      low_normalized, high_normalized)
print("Filter coefficients:")
print(sos)
plt.figure(figsize=(14, 5))

plt.plot(time, signal, label='Original ECG')
plt.plot(time, filtered_ecg, label='Band-pass filtered ECG')

plt.xlabel('Time (seconds)')
plt.ylabel('Amplitude (mV)')
plt.title('ECG Before and After Band-pass Filtering')
plt.legend()
plt.grid()
plt.show()
freq = rfftfreq(N, 1/fs)

fft_original = np.abs(rfft(signal)) / N
fft_filtered = np.abs(rfft(filtered_ecg)) / N

plt.figure(figsize=(14, 5))

plt.plot(freq, fft_original, label='Original ECG')
plt.plot(freq, fft_filtered, label='Band-pass filtered ECG')

plt.xlabel('Frequency (Hz)')
plt.ylabel('Magnitude')
plt.title('FFT Before and After Band-pass Filtering')
plt.xlim(0, 100)
plt.legend()
plt.grid()
plt.show()