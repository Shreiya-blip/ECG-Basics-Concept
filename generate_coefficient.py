import numpy as np
from scipy.signal import butter

fs = 360
lowcut = 0.5
highcut = 40
order = 4

sos = butter(
    order,
    [lowcut, highcut],
    btype='bandpass',
    fs=fs,
    output='sos'
)

np.savetxt("sos_coefficients.txt", sos, fmt="%.17g")

print("SOS coefficients generated:")
print(sos)