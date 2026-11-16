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