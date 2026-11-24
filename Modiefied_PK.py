import numpy as np
import matplotlib.pyplot as plt
import os
from scipy.signal import butter, filtfilt, find_peaks



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
# MAIN PROGRAM
# ======================================================

if __name__ == "__main__":

    # Load your ECG text file
    # --------------------------------------------------
# LOAD ECG DATA: 30–60 SECONDS
# --------------------------------------------------

    script_folder = os.path.dirname(os.path.abspath(__file__))

    input_file = os.path.join(
    script_folder,
    "ecg_input_30_60.txt"
)

    ecg = np.loadtxt(input_file)


    # Enter the actual sampling frequency of your dataset
    fs = 360
# Verify input data
    print("Input file:", input_file)
    print("File exists:", os.path.exists(input_file))
    print("Sampling frequency:", fs)
    print("Number of samples:", len(ecg))
    print("First 10 ECG samples:", ecg[:10])

    (
        filtered,
        derivative,
        squared,
        integrated,
        r_peaks,
        amplitudes,
        rr_intervals,
        heart_rate
    ) = pan_tompkins(ecg, fs)
    # --------------------------------------------------
    # SAVE DETECTED R-PEAK INDICES
    # --------------------------------------------------

    np.savetxt(
        "detected_peaks_30_60.txt",
        r_peaks,
        fmt="%d"
    )

    print("Detected R-peaks saved successfully.")

    # --------------------------------------------------
    # DISPLAY RESULTS
    # --------------------------------------------------

    time = np.arange(len(ecg)) / fs

    plt.figure(figsize=(14, 10))

    plt.subplot(4, 1, 1)
    plt.plot(time, ecg)
    plt.title("Original ECG Signal")
    plt.xlabel("Time (seconds)")

    plt.subplot(4, 1, 2)
    plt.plot(time, filtered)
    plt.title("Bandpass Filtered ECG")

    plt.subplot(4, 1, 3)
    plt.plot(time, integrated)
    plt.title("Moving Window Integrated Signal")

    plt.subplot(4, 1, 4)
    plt.plot(time, ecg, label="Original ECG")

    plt.scatter(
        r_peaks / fs,
        ecg[r_peaks],
        color='red',
        label="Detected R-peaks"
    )

    plt.title("Pan-Tompkins R-Peak Detection")
    plt.xlabel("Time (seconds)")
    plt.legend()

    plt.tight_layout()
    plt.show()

    # --------------------------------------------------
    # PRINT RESULTS
    # --------------------------------------------------

    print("\nDetected R-peaks:", len(r_peaks))

    print("\nR-peak sample indices:")
    print(r_peaks)

    print("\nRR intervals (seconds):")
    print(rr_intervals)

    print("\nHeart rate (BPM):")
    print(heart_rate)

    if len(heart_rate):
        print("\nAverage heart rate:", np.mean(heart_rate), "BPM")
 