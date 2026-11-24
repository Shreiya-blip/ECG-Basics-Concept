import wfdb
import numpy as np

fs = 360

# Load original MIT-BIH annotations
annotation = wfdb.rdann('100', 'atr', pn_dir='mitdb')

# Extract annotation sample positions
reference_peaks = annotation.sample
symbols = annotation.symbol

# Define 30–60 second segment boundaries
start_sample = 30 * fs
end_sample = 60 * fs

# Select valid heartbeat annotations
beat_symbols = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
    'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'
}

# Extract reference peaks for Segment 2
reference_peaks_30_60 = []

for peak, symbol in zip(reference_peaks, symbols):

    if start_sample <= peak < end_sample and symbol in beat_symbols:

        # Convert global index to local segment index
        local_peak = peak - start_sample

        reference_peaks_30_60.append(local_peak)

reference_peaks_30_60 = np.array(reference_peaks_30_60)

print("Reference beat count:", len(reference_peaks_30_60))
print("Reference peak indices:", reference_peaks_30_60)

# Save reference peaks
np.savetxt(
    "reference_peaks_30_60.txt",
    reference_peaks_30_60,
    fmt="%d"
)

print("Reference peaks saved successfully.")