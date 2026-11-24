"""
Pan-Tompkins Python vs C verification and error-analysis script.

Before running:
1. Edit the four PEAK FILE paths in CONFIG below to match your actual filenames.
2. Keep this script, the peak files, and your MIT-BIH access setup in the same project
   folder, or use full paths.
3. Install dependencies:  pip install numpy wfdb

Expected peak-file format:
- One 0-based sample index per line.
- Segment 1 indices are local to 0-30 s.
- Segment 2 indices are local to 30-60 s.
"""

from pathlib import Path
import csv
import numpy as np

# ---------------------------- CONFIGURATION ----------------------------

FS = 360
RECORD = "100"
DATABASE = "mitdb"
TOLERANCE_MS = 150
TOLERANCE_SAMPLES = int(round(FS * TOLERANCE_MS / 1000))  # 54 samples
SEGMENT_LENGTH = 30 * FS
TOTAL_SAMPLES = 60 * FS

# CHANGE THESE FOUR FILENAMES to your actual saved peak-file names.
# Use only peak-index files here, not amplitude or RR/HR files.
CONFIG = {
    "0-30 s": {
        "offset": 0,
        "python_peaks": "detected_peaks.txt",
        "c_peaks": "detected_peaks_c.txt",
    },
    "30-60 s": {
        "offset": SEGMENT_LENGTH,
        "python_peaks": "detected_peaks_30_60.txt",
        "c_peaks": "detected_peaks_c_30_60.txt",
    },
}

OUTPUT_DIR = Path("pan_tompkin_verification_results")
OUTPUT_DIR.mkdir(exist_ok=True)

# MIT-BIH beat annotation symbols. Non-beat annotations are excluded.
BEAT_SYMBOLS = {
    "N", "L", "R", "B", "A", "a", "J", "S",
    "V", "r", "F", "e", "j", "n", "E", "/", "f", "Q"
}


# ---------------------------- FILE HELPERS ----------------------------

def load_peak_file(filename):
    path = Path(filename)
    if not path.is_file():
        raise FileNotFoundError(
            f"Cannot find peak file: {path.resolve()}\n"
            "Edit the CONFIG filenames at the top of this script."
        )

    values = np.loadtxt(path, dtype=float)
    values = np.atleast_1d(values).astype(int)
    values = np.sort(values)

    if values.size == 0:
        raise ValueError(f"No peak indices found in {path}")
    if np.any(values < 0):
        raise ValueError(f"Negative sample index found in {path}")
    if np.any(np.diff(values) <= 0):
        raise ValueError(
            f"Peak indices in {path} must be unique and strictly increasing."
        )
    return values


def load_reference_annotations():
    try:
        import wfdb
    except ImportError as exc:
        raise ImportError(
            "WFDB is required to read MIT-BIH annotations. Install it with:\n"
            "    pip install wfdb"
        ) from exc

    annotation = wfdb.rdann(RECORD, "atr", pn_dir=DATABASE)
    symbols = np.asarray(annotation.symbol)
    samples = np.asarray(annotation.sample, dtype=int)

    keep = np.array([symbol in BEAT_SYMBOLS for symbol in symbols])
    return samples[keep]


# ---------------------------- MATCHING / METRICS ----------------------------

def match_peaks(reference, detected, tolerance):
    """
    Chronological, one-to-one matching.
    Returns pairs of (reference_index, detected_index), plus unmatched indices.
    """
    reference = np.asarray(reference, dtype=int)
    detected = np.asarray(detected, dtype=int)

    i = j = 0
    pairs = []
    unmatched_ref = []
    unmatched_det = []

    while i < len(reference) and j < len(detected):
        difference = detected[j] - reference[i]

        if abs(difference) <= tolerance:
            pairs.append((int(reference[i]), int(detected[j])))
            i += 1
            j += 1
        elif detected[j] < reference[i] - tolerance:
            unmatched_det.append(int(detected[j]))
            j += 1
        else:
            unmatched_ref.append(int(reference[i]))
            i += 1

    unmatched_ref.extend(int(x) for x in reference[i:])
    unmatched_det.extend(int(x) for x in detected[j:])

    return pairs, unmatched_ref, unmatched_det


def detection_metrics(reference, detected, tolerance):
    pairs, fn_samples, fp_samples = match_peaks(reference, detected, tolerance)
    tp = len(pairs)
    fn = len(fn_samples)
    fp = len(fp_samples)

    sensitivity = tp / (tp + fn) if (tp + fn) else 0.0
    ppv = tp / (tp + fp) if (tp + fp) else 0.0
    f1 = (2 * sensitivity * ppv / (sensitivity + ppv)
          if (sensitivity + ppv) else 0.0)

    return {
        "TP": tp, "FP": fp, "FN": fn,
        "Sensitivity_percent": 100 * sensitivity,
        "PPV_percent": 100 * ppv,
        "F1_percent": 100 * f1,
        "matched_pairs": pairs,
        "FP_samples": fp_samples,
        "FN_samples": fn_samples,
    }


def consecutive_interval_errors(pairs, fs):
    """
    Compare intervals only where two adjacent reference beats were both matched.
    This avoids comparing unrelated RR intervals when a beat is missed/extra.
    """
    if len(pairs) < 2:
        return np.array([]), np.array([]), np.array([])

    pairs = sorted(pairs, key=lambda pair: pair[0])
    ref_intervals = []
    det_intervals = []
    for (ref_a, det_a), (ref_b, det_b) in zip(pairs[:-1], pairs[1:]):
        ref_rr = (ref_b - ref_a) / fs
        det_rr = (det_b - det_a) / fs
        if ref_rr > 0 and det_rr > 0:
            ref_intervals.append(ref_rr)
            det_intervals.append(det_rr)

    ref_rr = np.asarray(ref_intervals, dtype=float)
    det_rr = np.asarray(det_intervals, dtype=float)
    rr_error = det_rr - ref_rr

    ref_hr = 60.0 / ref_rr if len(ref_rr) else np.array([])
    det_hr = 60.0 / det_rr if len(det_rr) else np.array([])
    hr_error = det_hr - ref_hr

    return rr_error, hr_error, det_rr


def error_summary(errors):
    errors = np.asarray(errors, dtype=float)
    if errors.size == 0:
        return {"count": 0, "MAE": float("nan"),
                "RMSE": float("nan"), "MaxAbs": float("nan"),
                "Bias": float("nan")}
    return {
        "count": int(errors.size),
        "MAE": float(np.mean(np.abs(errors))),
        "RMSE": float(np.sqrt(np.mean(errors ** 2))),
        "MaxAbs": float(np.max(np.abs(errors))),
        "Bias": float(np.mean(errors)),
    }


def write_csv(filename, rows):
    if not rows:
        return
    path = OUTPUT_DIR / filename
    with path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


# ---------------------------- MAIN VERIFICATION ----------------------------

def main():
    print("=" * 72)
    print("PAN-TOMPKINS PYTHON vs C VERIFICATION")
    print(f"Sampling frequency: {FS} Hz")
    print(f"MIT-BIH record: {RECORD}, database: {DATABASE}")
    print(f"Reference matching tolerance: ±{TOLERANCE_MS} ms "
          f"({TOLERANCE_SAMPLES} samples)")
    print("=" * 72)

    reference_all = load_reference_annotations()
    segment_results = []
    combined_python = []
    combined_c = []
    combined_reference = []

    for segment_name, settings in CONFIG.items():
        offset = settings["offset"]
        py_local = load_peak_file(settings["python_peaks"])
        c_local = load_peak_file(settings["c_peaks"])

        # Ensure segment-local peak indices are within a 30-second segment.
        for label, peaks in (("Python", py_local), ("C", c_local)):
            if np.any(peaks >= SEGMENT_LENGTH):
                raise ValueError(
                    f"{label} peaks for {segment_name} include indices >= "
                    f"{SEGMENT_LENGTH}. These files must contain segment-local "
                    "indices, not full-record/global indices."
                )

        # Reference annotation samples are global in Record 100.
        ref_global = reference_all[
            (reference_all >= offset) &
            (reference_all < offset + SEGMENT_LENGTH)
        ]
        ref_local = ref_global - offset

        py_vs_ref = detection_metrics(ref_local, py_local, TOLERANCE_SAMPLES)
        c_vs_ref = detection_metrics(ref_local, c_local, TOLERANCE_SAMPLES)
        c_vs_py = detection_metrics(py_local, c_local, TOLERANCE_SAMPLES)

        # Peak-location errors for paired Python/C detections.
        py_c_pairs = c_vs_py["matched_pairs"]
        py_c_location_errors = np.array(
            [det - ref for ref, det in py_c_pairs], dtype=float
        )
        py_c_location_summary = error_summary(py_c_location_errors)

        # RR/HR comparison between C and Python, based on matched consecutive beats.
        rr_error_c_py, hr_error_c_py, _ = consecutive_interval_errors(
            py_c_pairs, FS
        )

        # RR/HR error against reference for both implementations.
        py_rr_error, py_hr_error, _ = consecutive_interval_errors(
            py_vs_ref["matched_pairs"], FS
        )
        c_rr_error, c_hr_error, _ = consecutive_interval_errors(
            c_vs_ref["matched_pairs"], FS
        )

        result = {
            "segment": segment_name,
            "reference_beats": len(ref_local),
            "python_beats": len(py_local),
            "c_beats": len(c_local),
            "python_c_matched": len(py_c_pairs),
            "python_c_exact_matches": int(np.sum(py_c_location_errors == 0)),
            "python_c_exact_match_percent_of_python": (
                100.0 * np.sum(py_c_location_errors == 0) / len(py_local)
                if len(py_local) else 0.0
            ),
            "python_vs_reference_TP": py_vs_ref["TP"],
            "python_vs_reference_FP": py_vs_ref["FP"],
            "python_vs_reference_FN": py_vs_ref["FN"],
            "python_sensitivity_percent": py_vs_ref["Sensitivity_percent"],
            "python_PPV_percent": py_vs_ref["PPV_percent"],
            "python_F1_percent": py_vs_ref["F1_percent"],
            "c_vs_reference_TP": c_vs_ref["TP"],
            "c_vs_reference_FP": c_vs_ref["FP"],
            "c_vs_reference_FN": c_vs_ref["FN"],
            "c_sensitivity_percent": c_vs_ref["Sensitivity_percent"],
            "c_PPV_percent": c_vs_ref["PPV_percent"],
            "c_F1_percent": c_vs_ref["F1_percent"],
            "python_c_peak_location_MAE_samples": py_c_location_summary["MAE"],
            "python_c_peak_location_RMSE_samples": py_c_location_summary["RMSE"],
            "python_c_peak_location_max_abs_samples": py_c_location_summary["MaxAbs"],
            "python_c_peak_location_MAE_ms": (
                py_c_location_summary["MAE"] * 1000 / FS
                if py_c_location_summary["count"] else float("nan")
            ),
            "python_c_RR_MAE_ms": (
                error_summary(rr_error_c_py)["MAE"] * 1000
                if len(rr_error_c_py) else float("nan")
            ),
            "python_c_RR_max_abs_ms": (
                error_summary(rr_error_c_py)["MaxAbs"] * 1000
                if len(rr_error_c_py) else float("nan")
            ),
            "python_c_HR_MAE_BPM": error_summary(hr_error_c_py)["MAE"],
            "python_c_HR_max_abs_BPM": error_summary(hr_error_c_py)["MaxAbs"],
            "python_vs_reference_RR_MAE_ms": (
                error_summary(py_rr_error)["MAE"] * 1000
                if len(py_rr_error) else float("nan")
            ),
            "python_vs_reference_HR_MAE_BPM": error_summary(py_hr_error)["MAE"],
            "c_vs_reference_RR_MAE_ms": (
                error_summary(c_rr_error)["MAE"] * 1000
                if len(c_rr_error) else float("nan")
            ),
            "c_vs_reference_HR_MAE_BPM": error_summary(c_hr_error)["MAE"],
        }
        segment_results.append(result)

        # Save local detailed peak comparisons.
        peak_rows = []
        for ref_peak, c_peak in py_c_pairs:
            peak_rows.append({
                "python_peak_local": ref_peak,
                "c_peak_local": c_peak,
                "C_minus_Python_samples": c_peak - ref_peak,
                "C_minus_Python_ms": (c_peak - ref_peak) * 1000 / FS,
            })
        write_csv(
            f"peak_comparison_{segment_name.replace('-', '_').replace(' ', '')}.csv",
            peak_rows
        )

        # Build global-index arrays for the combined 60-second report.
        combined_python.extend((py_local + offset).tolist())
        combined_c.extend((c_local + offset).tolist())
        combined_reference.extend(ref_global.tolist())

        print(f"\n--- {segment_name} ---")
        print(f"Reference / Python / C beats: "
              f"{len(ref_local)} / {len(py_local)} / {len(c_local)}")
        print(f"Python vs reference: TP={py_vs_ref['TP']}, "
              f"FP={py_vs_ref['FP']}, FN={py_vs_ref['FN']}, "
              f"Sensitivity={py_vs_ref['Sensitivity_percent']:.2f}%, "
              f"PPV={py_vs_ref['PPV_percent']:.2f}%, "
              f"F1={py_vs_ref['F1_percent']:.2f}%")
        print(f"C vs reference:      TP={c_vs_ref['TP']}, "
              f"FP={c_vs_ref['FP']}, FN={c_vs_ref['FN']}, "
              f"Sensitivity={c_vs_ref['Sensitivity_percent']:.2f}%, "
              f"PPV={c_vs_ref['PPV_percent']:.2f}%, "
              f"F1={c_vs_ref['F1_percent']:.2f}%")
        print(f"Python vs C matched peaks: {len(py_c_pairs)}")
        print(f"Exact Python-C peak matches: "
              f"{result['python_c_exact_matches']} "
              f"({result['python_c_exact_match_percent_of_python']:.2f}%)")
        print(f"Python-C peak-location MAE: "
              f"{result['python_c_peak_location_MAE_samples']:.4f} samples "
              f"({result['python_c_peak_location_MAE_ms']:.4f} ms)")
        print(f"Python-C peak-location maximum absolute error: "
              f"{result['python_c_peak_location_max_abs_samples']} samples")
        print(f"Python-C RR MAE: {result['python_c_RR_MAE_ms']:.4f} ms")
        print(f"Python-C HR MAE: {result['python_c_HR_MAE_BPM']:.6f} BPM")

    # Combined 60-second metrics. Add segment offsets before matching.
    combined_python = np.asarray(sorted(combined_python), dtype=int)
    combined_c = np.asarray(sorted(combined_c), dtype=int)
    combined_reference = np.asarray(sorted(combined_reference), dtype=int)

    combined_py_ref = detection_metrics(
        combined_reference, combined_python, TOLERANCE_SAMPLES
    )
    combined_c_ref = detection_metrics(
        combined_reference, combined_c, TOLERANCE_SAMPLES
    )
    combined_c_py = detection_metrics(
        combined_python, combined_c, TOLERANCE_SAMPLES
    )

    combined_location_errors = np.array(
        [det - ref for ref, det in combined_c_py["matched_pairs"]],
        dtype=float
    )
    combined_rr_error, combined_hr_error, _ = consecutive_interval_errors(
        combined_c_py["matched_pairs"], FS
    )

    combined_row = {
        "segment": "Combined 0-60 s",
        "reference_beats": len(combined_reference),
        "python_beats": len(combined_python),
        "c_beats": len(combined_c),
        "python_c_matched": len(combined_c_py["matched_pairs"]),
        "python_c_exact_matches": int(np.sum(combined_location_errors == 0)),
        "python_c_exact_match_percent_of_python": (
            100.0 * np.sum(combined_location_errors == 0) / len(combined_python)
            if len(combined_python) else 0.0
        ),
        "python_vs_reference_TP": combined_py_ref["TP"],
        "python_vs_reference_FP": combined_py_ref["FP"],
        "python_vs_reference_FN": combined_py_ref["FN"],
        "python_sensitivity_percent": combined_py_ref["Sensitivity_percent"],
        "python_PPV_percent": combined_py_ref["PPV_percent"],
        "python_F1_percent": combined_py_ref["F1_percent"],
        "c_vs_reference_TP": combined_c_ref["TP"],
        "c_vs_reference_FP": combined_c_ref["FP"],
        "c_vs_reference_FN": combined_c_ref["FN"],
        "c_sensitivity_percent": combined_c_ref["Sensitivity_percent"],
        "c_PPV_percent": combined_c_ref["PPV_percent"],
        "c_F1_percent": combined_c_ref["F1_percent"],
        "python_c_peak_location_MAE_samples": error_summary(combined_location_errors)["MAE"],
        "python_c_peak_location_RMSE_samples": error_summary(combined_location_errors)["RMSE"],
        "python_c_peak_location_max_abs_samples": error_summary(combined_location_errors)["MaxAbs"],
        "python_c_peak_location_MAE_ms": (
            error_summary(combined_location_errors)["MAE"] * 1000 / FS
            if len(combined_location_errors) else float("nan")
        ),
        "python_c_RR_MAE_ms": (
            error_summary(combined_rr_error)["MAE"] * 1000
            if len(combined_rr_error) else float("nan")
        ),
        "python_c_RR_max_abs_ms": (
            error_summary(combined_rr_error)["MaxAbs"] * 1000
            if len(combined_rr_error) else float("nan")
        ),
        "python_c_HR_MAE_BPM": error_summary(combined_hr_error)["MAE"],
        "python_c_HR_max_abs_BPM": error_summary(combined_hr_error)["MaxAbs"],
        "python_vs_reference_RR_MAE_ms": "",
        "python_vs_reference_HR_MAE_BPM": "",
        "c_vs_reference_RR_MAE_ms": "",
        "c_vs_reference_HR_MAE_BPM": "",
    }

    all_rows = segment_results + [combined_row]
    write_csv("verification_summary.csv", all_rows)

    # Text report with key combined metrics and important limitations.
    report_path = OUTPUT_DIR / "verification_report.txt"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("PAN-TOMPKINS PYTHON-C VERIFICATION REPORT\n")
        f.write("=" * 55 + "\n")
        f.write(f"Record: {RECORD}, database: {DATABASE}\n")
        f.write(f"Sampling frequency: {FS} Hz\n")
        f.write(f"Reference tolerance: +/- {TOLERANCE_MS} ms "
                f"({TOLERANCE_SAMPLES} samples)\n\n")
        for row in all_rows:
            f.write(f"{row['segment']}\n")
            f.write("-" * 35 + "\n")
            for key, value in row.items():
                if key != "segment":
                    f.write(f"{key}: {value}\n")
            f.write("\n")
        f.write("Interpretation notes:\n")
        f.write("- Sensitivity, PPV and F1 are calculated against MIT-BIH beat annotations.\n")
        f.write("- Python-C agreement measures implementation consistency, not clinical accuracy.\n")
        f.write("- Peak timing is compared with one-to-one matching within the selected tolerance.\n")
        f.write("- RR/HR errors use consecutive matched beats only.\n")
        f.write("- Results cover only the selected 60 seconds of Record 100.\n")
        f.write("- This is offline software validation, not real-time sensor or hardware validation.\n")

    print("\n" + "=" * 72)
    print("COMBINED 0-60 SECOND RESULTS")
    print(f"Reference beats: {len(combined_reference)}")
    print(f"Python beats:    {len(combined_python)}")
    print(f"C beats:         {len(combined_c)}")
    print(f"Python vs reference: TP={combined_py_ref['TP']}, "
          f"FP={combined_py_ref['FP']}, FN={combined_py_ref['FN']}, "
          f"Sensitivity={combined_py_ref['Sensitivity_percent']:.2f}%, "
          f"PPV={combined_py_ref['PPV_percent']:.2f}%, "
          f"F1={combined_py_ref['F1_percent']:.2f}%")
    print(f"C vs reference:      TP={combined_c_ref['TP']}, "
          f"FP={combined_c_ref['FP']}, FN={combined_c_ref['FN']}, "
          f"Sensitivity={combined_c_ref['Sensitivity_percent']:.2f}%, "
          f"PPV={combined_c_ref['PPV_percent']:.2f}%, "
          f"F1={combined_c_ref['F1_percent']:.2f}%")
    print(f"Python-C exact peak matches: "
          f"{int(np.sum(combined_location_errors == 0))}/{len(combined_python)}")
    print(f"Python-C peak-location MAE: "
          f"{combined_row['python_c_peak_location_MAE_ms']:.4f} ms")
    print(f"Python-C RR MAE: {combined_row['python_c_RR_MAE_ms']:.4f} ms")
    print(f"Python-C HR MAE: {combined_row['python_c_HR_MAE_BPM']:.6f} BPM")
    print(f"\nSaved results in: {OUTPUT_DIR.resolve()}")
    print("  - verification_summary.csv")
    print("  - verification_report.txt")
    print("  - peak_comparison_*.csv")


if __name__ == "__main__":
    main()
