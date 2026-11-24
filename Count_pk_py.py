import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, find_peaks
import wfdb


def pan_tompkins(ecg, fs):

    ecg = np.asarray(ecg, dtype=float).flatten()

    if len(ecg) < 2 * fs:
        raise ValueError("ECG signal must contain at least 2 seconds of data.")

    # --------------------------------------------------
    # 1. BANDPASS FILTERING
    # --------------------------------------------------

    if fs == 200:

        # Low-pass filter
        b_lp = np.array(
            [1, 0, 0, 0, 0, 0, -2, 0, 0, 0, 0, 0, 1]
        )

        a_lp = np.array([1, -2, 1])

        ecg_l = filtfilt(b_lp, a_lp, ecg)

        ecg_l = ecg_l / (np.max(np.abs(ecg_l)) + 1e-12)

        # High-pass filter
        b_hp = np.zeros(33)

        b_hp[0] = -1
        b_hp[16] = 32
        b_hp[32] = 1

        a_hp = np.array([1, 1])

        ecg_h = filtfilt(b_hp, a_hp, ecg_l)

        ecg_h = ecg_h / (np.max(np.abs(ecg_h)) + 1e-12)

    else:

        # 5-15 Hz Butterworth bandpass filter
        low = 5
        high = 15

        b, a = butter(
            3,
            [low, high],
            btype='bandpass',
            fs=fs
        )

        ecg_h = filtfilt(b, a, ecg)

        ecg_h = ecg_h / (np.max(np.abs(ecg_h)) + 1e-12)

    # --------------------------------------------------
    # 2. DERIVATIVE FILTER
    # --------------------------------------------------

    derivative_coeff = np.array([1, 2, 0, -2, -1]) * fs / 8

    ecg_d = filtfilt(derivative_coeff, [1], ecg_h)

    ecg_d = ecg_d / (np.max(np.abs(ecg_d)) + 1e-12)

    # --------------------------------------------------
    # 3. SQUARING
    # --------------------------------------------------

    ecg_s = ecg_d ** 2

    # --------------------------------------------------
    # 4. MOVING WINDOW INTEGRATION
    # --------------------------------------------------

    window_size = round(0.150 * fs)

    window = np.ones(window_size) / window_size

    ecg_m = np.convolve(ecg_s, window, mode='same')

    # --------------------------------------------------
    # 5. FIDUCIAL MARKS / CANDIDATE PEAKS
    # --------------------------------------------------

    min_distance = round(0.200 * fs)

    peaks, properties = find_peaks(
        ecg_m,
        distance=min_distance
    )

    peak_values = ecg_m[peaks]

    # --------------------------------------------------
    # 6. INITIAL THRESHOLDING
    # --------------------------------------------------

    training_length = min(2 * fs, len(ecg_m))

    THR_SIG = np.max(ecg_m[:training_length]) / 3

    THR_NOISE = np.mean(ecg_m[:training_length]) / 2

    SIG_LEV = THR_SIG
    NOISE_LEV = THR_NOISE

    THR_SIG1 = np.max(ecg_h[:training_length]) / 3

    THR_NOISE1 = np.mean(ecg_h[:training_length]) / 2

    SIG_LEV1 = THR_SIG1
    NOISE_LEV1 = THR_NOISE1

    # --------------------------------------------------
    # 7. ADAPTIVE THRESHOLDING AND QRS DETECTION
    # --------------------------------------------------

    qrs_indices = []
    qrs_amplitudes = []

    noise_candidates = []

    rr_intervals = []

    previous_slope = None

    refractory = round(0.200 * fs)

    twave_window = round(0.360 * fs)

    integration_window = round(0.150 * fs)

    # Search around each integrated-signal candidate to locate the
    # corresponding positive R-wave in the bandpass-filtered ECG.
    # At 360 Hz, 75 ms is approximately 27 samples on either side.
    search_radius = round(0.075 * fs)

    def locate_r_peak(candidate_index):
        start = max(0, candidate_index - search_radius)
        end = min(len(ecg_h), candidate_index + search_radius + 1)

        segment = ecg_h[start:end]
        if len(segment) == 0:
            return None, None

        local_index = int(np.argmax(segment))
        r_index = start + local_index
        r_amplitude = ecg_h[r_index]
        return r_index, r_amplitude

    for peak, value in zip(peaks, peak_values):

        # Map the integrated-signal candidate to the nearby R-wave.
        r_index, y_i = locate_r_peak(peak)
        if r_index is None:
            continue

        # --------------------------------------------------
        # SEARCH-BACK
        # --------------------------------------------------

        if len(qrs_indices) >= 2:

            mean_rr = np.mean(rr_intervals[-8:]) if rr_intervals else 0

            if mean_rr > 0:

                current_gap = peak - qrs_indices[-1]

                if current_gap >= 1.66 * mean_rr:

                    eligible = [
                        item for item in noise_candidates
                        if item[0] > qrs_indices[-1] + refractory
                        and item[0] < peak - refractory
                    ]

                    if eligible:

                        best_candidate = max(
                            eligible,
                            key=lambda item: item[1]
                        )

                        sb_index, sb_value = best_candidate

                        if sb_value > THR_NOISE:

                            sb_r_index, sb_r_amplitude = locate_r_peak(sb_index)

                            if sb_r_index is not None and (
                                not qrs_indices
                                or sb_r_index - qrs_indices[-1] >= refractory
                            ):

                                    qrs_indices.append(sb_r_index)

                                    qrs_amplitudes.append(sb_r_amplitude)

                                    SIG_LEV = (
                                        0.25 * sb_value
                                        + 0.75 * SIG_LEV
                                    )

                                    SIG_LEV1 = (
                                        0.25 * sb_r_amplitude
                                        + 0.75 * SIG_LEV1
                                    )

                                    if len(qrs_indices) >= 2:

                                        rr_intervals.append(
                                            qrs_indices[-1]
                                            - qrs_indices[-2]
                                        )

                                    noise_candidates.remove(best_candidate)

        # --------------------------------------------------
        # QRS CANDIDATE CLASSIFICATION
        # --------------------------------------------------

        if value >= THR_SIG:

            skip = False

            # T-wave discrimination
            if qrs_indices:

                gap = r_index - qrs_indices[-1]

                if refractory < gap <= twave_window:

                    current_start = max(
                        0, peak - round(0.075 * fs)
                    )

                    current_slope = np.mean(
                        np.diff(ecg_m[current_start:peak + 1])
                    )

                    previous_peak = qrs_indices[-1]

                    previous_start = max(
                        0, previous_peak - round(0.075 * fs)
                    )

                    previous_slope = np.mean(
                        np.diff(
                            ecg_m[
                                previous_start:previous_peak + 1
                            ]
                        )
                    )

                    if abs(current_slope) <= 0.5 * abs(previous_slope):

                        skip = True

            if not skip:

                # Refractory-period check
                if (
                    not qrs_indices
                    or r_index - qrs_indices[-1] >= refractory
                ):

                    # Bandpass signal threshold check
                    if y_i >= THR_SIG1:

                        qrs_indices.append(r_index)

                        qrs_amplitudes.append(y_i)

                        if len(qrs_indices) >= 2:

                            rr_intervals.append(
                                qrs_indices[-1] - qrs_indices[-2]
                            )

                        SIG_LEV1 = (
                            0.125 * y_i
                            + 0.875 * SIG_LEV1
                        )

                SIG_LEV = (
                    0.125 * value
                    + 0.875 * SIG_LEV
                )

        elif value >= THR_NOISE:

            NOISE_LEV = (
                0.125 * value
                + 0.875 * NOISE_LEV
            )

            NOISE_LEV1 = (
                0.125 * y_i
                + 0.875 * NOISE_LEV1
            )

            noise_candidates.append((peak, value))

        else:

            NOISE_LEV = (
                0.125 * value
                + 0.875 * NOISE_LEV
            )

            NOISE_LEV1 = (
                0.125 * y_i
                + 0.875 * NOISE_LEV1
            )

        # --------------------------------------------------
        # UPDATE ADAPTIVE THRESHOLDS
        # --------------------------------------------------

        THR_SIG = (
            NOISE_LEV
            + 0.25 * abs(SIG_LEV - NOISE_LEV)
        )

        THR_NOISE = 0.5 * THR_SIG

        THR_SIG1 = (
            NOISE_LEV1
            + 0.25 * abs(SIG_LEV1 - NOISE_LEV1)
        )

        THR_NOISE1 = 0.5 * THR_SIG1

    # --------------------------------------------------
    # 8. HEART RATE AND RR INTERVAL
    # --------------------------------------------------

    qrs_indices = np.asarray(qrs_indices, dtype=int)

    qrs_amplitudes = np.asarray(qrs_amplitudes)

    if len(qrs_indices) >= 2:

        rr_samples = np.diff(qrs_indices)

        rr_seconds = rr_samples / fs

        heart_rate = 60 / rr_seconds

    else:

        rr_samples = np.array([])
        rr_seconds = np.array([])
        heart_rate = np.array([])

    return (
        ecg_h,
        ecg_d,
        ecg_s,
        ecg_m,
        qrs_indices,
        qrs_amplitudes,
        rr_seconds,
        heart_rate
    )

# ======================================================
# MAIN PROGRAM: MIT-BIH 3-RECORD VALIDATION
# ======================================================

if __name__ == "__main__":

    # Select three MIT-BIH records
    RECORDS = ['100', '101', '102']

    # Detection tolerance
    TOLERANCE_MS = 150

    # Standard MIT-BIH beat annotation symbols
    BEAT_SYMBOLS = {
        'N', 'L', 'R', 'B', 'A', 'a', 'J', 'S',
        'V', 'r', 'F', 'e', 'j', 'n', 'E', '/', 'f', 'Q'
    }

    # --------------------------------------------------
    # FUNCTION: COMPARE REFERENCE AND DETECTED PEAKS
    # --------------------------------------------------

    def compare_peaks(reference, detected, tolerance):

        reference = np.sort(np.asarray(reference, dtype=int))
        detected = np.sort(np.asarray(detected, dtype=int))

        i = 0
        j = 0

        TP = 0
        FN = 0
        FP = 0

        matched_pairs = []
        missed = []
        extra = []

        while i < len(reference) and j < len(detected):

            difference = detected[j] - reference[i]

            if abs(difference) <= tolerance:

                TP += 1

                matched_pairs.append(
                    (reference[i], detected[j], difference)
                )

                i += 1
                j += 1

            elif detected[j] < reference[i] - tolerance:

                FP += 1
                extra.append(detected[j])

                j += 1

            else:

                FN += 1
                missed.append(reference[i])

                i += 1

        # Count remaining unmatched reference peaks
        while i < len(reference):

            FN += 1
            missed.append(reference[i])

            i += 1

        # Count remaining unmatched detected peaks
        while j < len(detected):

            FP += 1
            extra.append(detected[j])

            j += 1

        return TP, FN, FP, matched_pairs, missed, extra

    # --------------------------------------------------
    # PROCESS THREE RECORDS
    # --------------------------------------------------

    all_results = []

    for record_name in RECORDS:

        print("\n======================================")
        print("PROCESSING RECORD:", record_name)
        print("======================================")

        # 1. Load ECG record
        record = wfdb.rdrecord(
            record_name,
            pn_dir='mitdb'
        )

        ecg = record.p_signal[:, 0]
        fs = record.fs

        print("Sampling frequency:", fs, "Hz")
        print("Number of samples:", len(ecg))
        print("Duration:", len(ecg) / fs, "seconds")

        # 2. Load MIT-BIH reference annotations
        annotation = wfdb.rdann(
            record_name,
            'atr',
            pn_dir='mitdb'
        )

        reference_peaks = np.array([
            sample
            for sample, symbol in zip(
                annotation.sample,
                annotation.symbol
            )
            if symbol in BEAT_SYMBOLS
        ], dtype=int)

        print("Reference beats:", len(reference_peaks))

        # 3. Run Pan-Tompkins algorithm
        (
            filtered,
            derivative,
            squared,
            integrated,
            detected_peaks,
            amplitudes,
            rr_intervals,
            heart_rate
        ) = pan_tompkins(ecg, fs)

        detected_peaks = np.sort(
            np.asarray(detected_peaks, dtype=int)
        )

        print("Detected beats:", len(detected_peaks))

        # 4. Calculate tolerance in samples
        tolerance = round(fs * TOLERANCE_MS / 1000)

        print("Tolerance:", tolerance, "samples")
        print("Tolerance:", TOLERANCE_MS, "ms")

        # 5. Compare detected peaks with reference peaks
        (
            TP,
            FN,
            FP,
            matched_pairs,
            missed,
            extra
        ) = compare_peaks(
            reference_peaks,
            detected_peaks,
            tolerance
        )

        # --------------------------------------------------
        # 6. VISUAL ERROR ANALYSIS FOR RECORD 101
        # --------------------------------------------------

        if record_name == '101':

            print("\nGenerating Record 101 error plots...")

            window_centers = [175.875, 320.1, 856.3]
            window_seconds = 2

            for center in window_centers:

                start = int((center - window_seconds) * fs)
                end = int((center + window_seconds) * fs)

                # Generate time axis
                time = np.arange(start, end) / fs

                # Reference peaks within this window
                ref_in = reference_peaks[
                    (reference_peaks >= start) &
                    (reference_peaks < end)
                ]

                # Detected peaks within this window
                det_in = detected_peaks[
                    (detected_peaks >= start) &
                    (detected_peaks < end)
                ]

                # Plot ECG signal
                plt.figure(figsize=(12, 5))

                plt.plot(
                    time,
                    ecg[start:end],
                    label='ECG Signal'
                )

                # Mark MIT-BIH reference peaks
                plt.scatter(
                    ref_in / fs,
                    ecg[ref_in],
                    marker='x',
                    s=80,
                    label='MIT-BIH Reference Peaks'
                )

                # Mark Pan-Tompkins detected peaks
                plt.scatter(
                    det_in / fs,
                    ecg[det_in],
                    marker='o',
                    facecolors='none',
                    s=100,
                    label='Pan-Tompkins Detected Peaks'
                )

                plt.title(
                    f'Record 101 - Peak Comparison around {center}s'
                )

                plt.xlabel('Time (seconds)')
                plt.ylabel('ECG Amplitude')

                plt.legend()
                plt.grid()
                plt.tight_layout()

                # Save plot
                plt.savefig(
                    f'record101_error_{center}.png'
                )

                plt.show()
                plt.close()

        # --------------------------------------------------
        # 7. DISPLAY MISSED AND EXTRA PEAKS
        # --------------------------------------------------

        print("\n----- PEAK ERROR ANALYSIS -----")

        print("Missed peak locations:", missed)
        print("Extra peak locations:", extra)

        print("\nMissed peak times (seconds):")
        print(np.array(missed) / fs)

        print("\nExtra peak times (seconds):")
        print(np.array(extra) / fs)

        # --------------------------------------------------
        # 8. CALCULATE PERFORMANCE METRICS
        # --------------------------------------------------

        sensitivity = (
            TP / (TP + FN) * 100
            if (TP + FN) > 0 else 0
        )

        ppv = (
            TP / (TP + FP) * 100
            if (TP + FP) > 0 else 0
        )

        f1_score = (
            2 * sensitivity * ppv / (sensitivity + ppv)
            if (sensitivity + ppv) > 0 else 0
        )

        # --------------------------------------------------
        # 9. DISPLAY VALIDATION RESULTS
        # --------------------------------------------------

        print("\n----- VALIDATION RESULTS -----")

        print("Correct beats (TP):", TP)
        print("Missed beats (FN):", FN)
        print("Extra beats (FP):", FP)

        print("Sensitivity:", round(sensitivity, 2), "%")
        print("Positive Predictivity:", round(ppv, 2), "%")
        print("F1 Score:", round(f1_score, 2), "%")

        # --------------------------------------------------
        # 10. SAVE DETECTED PEAKS
        # --------------------------------------------------

        np.savetxt(
            f"detected_peaks_{record_name}.txt",
            detected_peaks,
            fmt="%d"
        )

        # --------------------------------------------------
        # 11. SAVE MATCHED PEAKS
        # --------------------------------------------------

        np.savetxt(
            f"matched_peaks_{record_name}.txt",
            np.asarray(matched_pairs, dtype=int).reshape(-1, 3),
            fmt="%d",
            header="Reference_Peak Detected_Peak Difference"
        )

        # --------------------------------------------------
        # 12. SAVE MISSED PEAKS
        # --------------------------------------------------

        np.savetxt(
            f"missed_peaks_{record_name}.txt",
            missed,
            fmt="%d"
        )

        # --------------------------------------------------
        # 13. SAVE EXTRA PEAKS
        # --------------------------------------------------

        np.savetxt(
            f"extra_peaks_{record_name}.txt",
            extra,
            fmt="%d"
        )

        # --------------------------------------------------
        # 14. STORE SUMMARY
        # --------------------------------------------------

        all_results.append([
            record_name,
            fs,
            len(reference_peaks),
            len(detected_peaks),
            TP,
            FN,
            FP,
            round(sensitivity, 2),
            round(ppv, 2),
            round(f1_score, 2)
        ])

    # ==================================================
    # FINAL SUMMARY FOR ALL THREE RECORDS
    # ==================================================

    print("\n======================================")
    print("FINAL THREE-RECORD VALIDATION SUMMARY")
    print("======================================")

    header = [
        "Record",
        "Fs",
        "Reference",
        "Detected",
        "Correct",
        "Missed",
        "Extra",
        "Sensitivity",
        "PPV",
        "F1 Score"
    ]

    print(
        "{:<10} {:<6} {:<10} {:<10} {:<10} {:<10} "
        "{:<10} {:<12} {:<10} {:<10}".format(*header)
    )

    for result in all_results:

        print(
            "{:<10} {:<6} {:<10} {:<10} {:<10} {:<10} "
            "{:<10} {:<12} {:<10} {:<10}".format(*result)
        )

    # Save final summary as CSV
    np.savetxt(
        "three_record_validation_summary.csv",
        np.array(all_results, dtype=object),
        delimiter=",",
        fmt="%s",
        header="Record,Fs,Reference,Detected,Correct,Missed,Extra,Sensitivity,PPV,F1_Score",
        comments=""
    )

    print("\nFinal summary saved successfully.")
    print("All three records processed successfully.")