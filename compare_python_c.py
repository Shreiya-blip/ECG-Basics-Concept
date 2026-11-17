import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import sosfilt

# --------------------------------------------------
# STEP 1: LOAD ORIGINAL ECG DATASET
# --------------------------------------------------

ecg = np.loadtxt("ecg_input.txt")

fs = 360
N = len(ecg)

time = np.arange(N) / fs

print("ECG DATASET INFORMATION")
print("-----------------------")
print("Sampling frequency:", fs, "Hz")
print("Number of samples:", N)
print("Duration:", N / fs, "seconds")


# --------------------------------------------------
# STEP 2: LOAD SOS FILTER COEFFICIENTS
# --------------------------------------------------

sos = np.loadtxt("sos_coefficients.txt")

print("\nIIR FILTER INFORMATION")
print("----------------------")
print("Filter type: Butterworth Bandpass")
print("Lower cutoff frequency: 0.5 Hz")
print("Upper cutoff frequency: 40 Hz")
print("Prototype order: 4")
print("Number of SOS sections:", len(sos))


# --------------------------------------------------
# STEP 3: APPLY IIR FILTER IN PYTHON
# --------------------------------------------------

python_output = sosfilt(sos, ecg)

print("\nPython IIR filtering completed.")


# --------------------------------------------------
# STEP 4: LOAD C FILTER OUTPUT
# --------------------------------------------------

c_output = np.loadtxt("c_bandpass_output.txt")

print("C filter output loaded.")


# --------------------------------------------------
# STEP 5: CHECK SAMPLE COUNTS
# --------------------------------------------------

if len(python_output) != len(c_output):
    raise ValueError("Python and C output lengths do not match!")

print("\nBoth implementations contain", len(c_output), "samples.")


# --------------------------------------------------
# STEP 6: CALCULATE ERROR
# --------------------------------------------------

error = python_output - c_output

absolute_error = np.abs(error)

MAE = np.mean(absolute_error)

MSE = np.mean(error ** 2)

RMSE = np.sqrt(MSE)

MAX_ERROR = np.max(absolute_error)

print("\nPYTHON VS C COMPARISON")
print("======================")

print("Mean Absolute Error (MAE):", MAE)

print("Mean Squared Error (MSE):", MSE)

print("Root Mean Squared Error (RMSE):", RMSE)

print("Maximum Absolute Error:", MAX_ERROR)


# --------------------------------------------------
# STEP 7: DISPLAY FIRST 10 OUTPUT SAMPLES
# --------------------------------------------------

print("\nFIRST 10 OUTPUT SAMPLES")
print("=======================")

print("Sample\tPython Output\tC Output\tAbsolute Error")

for i in range(10):

    print(
        f"{i}\t"
        f"{python_output[i]:.12f}\t"
        f"{c_output[i]:.12f}\t"
        f"{absolute_error[i]:.12e}"
    )


# --------------------------------------------------
# STEP 8: PLOT ORIGINAL ECG SIGNAL
# --------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(time, ecg)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")

plt.title("Original ECG Signal - MIT-BIH Record 100")

plt.grid()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# STEP 9: PLOT PYTHON AND C FILTER OUTPUTS
# --------------------------------------------------

plt.figure(figsize=(12, 5))

plt.plot(
    time,
    python_output,
    label="Python IIR Output",
    linewidth=1.5
)

plt.plot(
    time,
    c_output,
    "--",
    label="C IIR Output",
    linewidth=1
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")

plt.title("Comparison of Python and C IIR Bandpass Filter")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# STEP 10: PLOT ABSOLUTE ERROR
# --------------------------------------------------

plt.figure(figsize=(12, 4))

plt.plot(time, absolute_error)

plt.xlabel("Time (seconds)")
plt.ylabel("Absolute Error")

plt.title("Absolute Error Between Python and C Outputs")

plt.grid()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# STEP 11: PLOT OUTPUT OVERLAP FOR FIRST 5 SECONDS
# --------------------------------------------------

samples_5s = min(5 * fs, N)

plt.figure(figsize=(12, 5))

plt.plot(
    time[:samples_5s],
    python_output[:samples_5s],
    label="Python Output"
)

plt.plot(
    time[:samples_5s],
    c_output[:samples_5s],
    "--",
    label="C Output"
)

plt.xlabel("Time (seconds)")
plt.ylabel("Amplitude (mV)")

plt.title("Detailed Python vs C Output - First 5 Seconds")

plt.legend()
plt.grid()
plt.tight_layout()
plt.show()


# --------------------------------------------------
# STEP 12: FINAL RESULT
# --------------------------------------------------

print("\nFINAL RESULT")
print("============")

if np.allclose(python_output, c_output, rtol=1e-9, atol=1e-11):

    print("SUCCESS: Python and C outputs match within tolerance.")

else:

    print("CHECK: Differences exist between Python and C outputs.")

print("\nComparison completed successfully.")