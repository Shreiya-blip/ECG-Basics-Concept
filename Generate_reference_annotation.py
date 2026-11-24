import wfdb
import numpy as np

record_id = '100'

annotation = wfdb.rdann(
    record_id,
    'atr',
    pn_dir='mitdb'
)

# MIT-BIH beat annotation symbols
beat_symbols = {
    'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
    'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'
}

reference_peaks = [
    sample
    for sample, symbol in zip(annotation.sample, annotation.symbol)
    if symbol in beat_symbols and sample < 10800
]

np.savetxt(
    "reference_peaks.txt",
    reference_peaks,
    fmt="%d"
)

print("Reference beat count:", len(reference_peaks))
print("Reference peak indices:", reference_peaks)