import numpy as np
import wfdb
import os
# --------------------------------------------------
# 1. PARAMETERS
# --------------------------------------------------

fs = 360
duration = 30
tolerance_ms = 150

tolerance_samples = int(fs * tolerance_ms / 1000)

# --------------------------------------------------
# 2. LOAD DETECTED R-PEAKS
# --------------------------------------------------

script_folder = os.path.dirname(os.path.abspath(__file__))

detected_file = os.path.join(
    script_folder,
    "detected_peaks_30_60.txt"
)

detected_peaks = np.loadtxt(detected_file, dtype=int)
detected_peaks = np.atleast_1d(detected_peaks)

print("Detected peaks file:", detected_file)
print("Number of detected peaks:", len(detected_peaks))

detected_peaks = np.atleast_1d(detected_peaks)

print("\n===================================")
print("DETECTED R-PEAK INFORMATION")
print("===================================")

print("Total detected peaks:", len(detected_peaks))
print("Detected peaks:", detected_peaks)


# --------------------------------------------------
# 3. LOAD MIT-BIH REFERENCE ANNOTATIONS
# --------------------------------------------------

record = wfdb.rdrecord("100", pn_dir="mitdb")

annotation = wfdb.rdann(
    "100",
    "atr",
    pn_dir="mitdb"
)

beat_symbols = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
    'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'
}

# --------------------------------------------------
# REFERENCE PEAKS: 30–60 SECONDS
# --------------------------------------------------

start_sec = 30
end_sec = 60

start_sample = int(start_sec * fs)
end_sample = int(end_sec * fs)

reference_peaks = np.array([
    sample - start_sample
    for sample, symbol in zip(annotation.sample, annotation.symbol)
    if symbol in beat_symbols
    and start_sample <= sample < end_sample
])

print("Number of reference peaks:", len(reference_peaks))

print("\n===================================")
print("REFERENCE R-PEAK INFORMATION")
print("===================================")

print("Total reference peaks:", len(reference_peaks))
print("Reference peaks:", reference_peaks)


# --------------------------------------------------
# 4. VERIFICATION 1: R-PEAK DETECTION
# --------------------------------------------------

matched_reference = set()
matched_detected = set()

for i, detected in enumerate(detected_peaks):

    differences = np.abs(reference_peaks - detected)

    valid_indices = np.where(
        (differences <= tolerance_samples) &
        (~np.isin(np.arange(len(reference_peaks)),
                  list(matched_reference)))
    )[0]

    if len(valid_indices) > 0:

        best_index = valid_indices[
            np.argmin(differences[valid_indices])
        ]

        matched_reference.add(best_index)
        matched_detected.add(i)


TP = len(matched_detected)

FP = len(detected_peaks) - TP

FN = len(reference_peaks) - TP

sensitivity = TP / (TP + FN) if (TP + FN) else 0

PPV = TP / (TP + FP) if (TP + FP) else 0

F1_score = (
    2 * sensitivity * PPV / (sensitivity + PPV)
    if (sensitivity + PPV) else 0
)

print("\n===================================")
print("VERIFICATION 1: R-PEAK DETECTION")
print("===================================")

print("Tolerance:", tolerance_ms, "ms")
print("True Positives (TP):", TP)
print("False Positives (FP):", FP)
print("False Negatives (FN):", FN)

print("Sensitivity:", sensitivity * 100, "%")
print("Positive Predictive Value:", PPV * 100, "%")
print("F1 Score:", F1_score * 100, "%")


# --------------------------------------------------
# 5. VERIFICATION 2: RR INTERVAL COMPARISON
# --------------------------------------------------

detected_rr = np.diff(detected_peaks) / fs

reference_rr = np.diff(reference_peaks) / fs

print("\n===================================")
print("VERIFICATION 2: RR INTERVAL")
print("===================================")

print("Detected RR count:", len(detected_rr))
print("Reference RR count:", len(reference_rr))

if len(detected_rr) == len(reference_rr):

    rr_error = np.abs(detected_rr - reference_rr)

    print("\nDetected RR intervals:")
    print(detected_rr)

    print("\nReference RR intervals:")
    print(reference_rr)

    print("\nAbsolute RR errors:")
    print(rr_error)

    print("Mean RR error:", np.mean(rr_error), "seconds")
    print("Maximum RR error:", np.max(rr_error), "seconds")

else:
    print(
        "\nRR arrays have different lengths. "
        "Match the reference and detected beats before "
        "performing a one-to-one RR comparison."
    )


# --------------------------------------------------
# 6. HEART RATE COMPARISON
# --------------------------------------------------

detected_hr = 60 / detected_rr

reference_hr = 60 / reference_rr

print("\n===================================")
print("HEART RATE COMPARISON")
print("===================================")

print("Detected average HR:",
      np.mean(detected_hr), "BPM")

print("Reference average HR:",
      np.mean(reference_hr), "BPM")

if len(detected_hr) == len(reference_hr):

    hr_error = np.abs(detected_hr - reference_hr)

    print("\nDetected heart rates:")
    print(detected_hr)

    print("\nReference heart rates:")
    print(reference_hr)

    print("\nAbsolute HR errors:")
    print(hr_error)

    print("Mean HR error:", np.mean(hr_error), "BPM")

    print("Maximum HR error:", np.max(hr_error), "BPM")

print("\n===================================")
print("VERIFICATION COMPLETED")
print("===================================")