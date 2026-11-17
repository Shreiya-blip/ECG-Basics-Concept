import wfdb
import numpy as np
import os
print(os.path.abspath("ecg_input.txt"))
record = wfdb.rdrecord('100', pn_dir='mitdb')

ecg = record.p_signal[:, 0]

fs = record.fs
N = int(fs * 30)

signal = ecg[:N]

np.savetxt(
    "ecg_input.txt",
    signal,
    fmt="%.17g"
)

print("Sampling frequency:", fs)
print("Number of samples:", len(signal))
print("ECG input file generated.")