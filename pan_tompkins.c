/*
 * Pan-Tompkins-style ECG R-peak detector in C
 * Based on the MATLAB function supplied by the user.
 *
 * Input:  one ECG sample per line in ecg_input.txt
 * Output: detected_peaks_c.txt, detected_amplitudes_c.txt, rr_hr_c.txt
 *
 * Sampling frequency: 360 Hz
 * Bandpass: 5-15 Hz, 3rd-order Butterworth prototype (6th-order bandpass)
 * MWI window: 150 ms (54 samples)
 * Minimum candidate spacing: 200 ms (72 samples)
 *
 * Build:
 *   gcc -O2 pan_tompkins.c -o pan_tompkins -lm
 *
 * Run:
 *   ./pan_tompkins
 *   ./pan_tompkins ecg_input_30_60.txt
 *
 * Note:
 * This is an offline floating-point C port of the main processing and
 * decision stages. The SOS forward/backward filter below approximates
 * MATLAB filtfilt; edge handling and peak selection can cause small
 * differences from MATLAB/Python. Compare the output against your
 * reference annotations before using it in embedded hardware.
 */

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>

#define FS              360.0
#define MAX_SAMPLES     20000
#define MAX_PEAKS       2000
#define MWI_LEN         54
#define REFRACTORY      72       /* 200 ms at 360 Hz */
#define TWAVE_LIMIT     130      /* about 360 ms */
#define LOCALIZE_WIN    54       /* 150 ms */
#define MATCH_UNUSED    0

typedef struct {
    double b0, b1, b2, a1, a2;
} Biquad;

/* SOS coefficients for butter(3, [5 15], btype='bandpass', fs=360).
   Each section uses: y[n] = b0*x[n]+b1*x[n-1]+b2*x[n-2]
                         -a1*y[n-1]-a2*y[n-2]. */
static const Biquad sos[3] = {
    {0.000562289281,  0.001124578562,  0.000562289281,
     -1.81807791,     0.839099631},
    {1.0,             0.0,            -1.0,
     -1.82315065,     0.880921959},
    {1.0,            -2.0,             1.0,
     -1.94554052,     0.953800959}
};

static int read_ecg(const char *filename, double *x, int max_n)
{
    FILE *fp = fopen(filename, "r");
    int n = 0;

    if (!fp) {
        perror("Could not open ECG input file");
        return -1;
    }

    while (n < max_n && fscanf(fp, "%lf", &x[n]) == 1)
        n++;

    fclose(fp);
    return n;
}

static double max_abs(const double *x, int n)
{
    double m = 0.0;
    for (int i = 0; i < n; i++) {
        double a = fabs(x[i]);
        if (a > m) m = a;
    }
    return m;
}

static void normalize_max_abs(double *x, int n)
{
    double m = max_abs(x, n);
    if (m < 1e-12) return;
    for (int i = 0; i < n; i++) x[i] /= m;
}

/* One pass through the three second-order sections. */
static void sos_filter_pass(const double *input, double *output, int n)
{
    double *a = (double *)malloc((size_t)n * sizeof(double));
    double *b = (double *)malloc((size_t)n * sizeof(double));
    if (!a || !b) {
        fprintf(stderr, "Memory allocation failed in filter.\n");
        free(a); free(b);
        exit(1);
    }

    memcpy(a, input, (size_t)n * sizeof(double));

    for (int s = 0; s < 3; s++) {
        double x1 = 0.0, x2 = 0.0, y1 = 0.0, y2 = 0.0;
        for (int i = 0; i < n; i++) {
            double x0 = a[i];
            double y0 = sos[s].b0*x0 + sos[s].b1*x1 + sos[s].b2*x2
                        - sos[s].a1*y1 - sos[s].a2*y2;
            b[i] = y0;
            x2 = x1; x1 = x0;
            y2 = y1; y1 = y0;
        }
        memcpy(a, b, (size_t)n * sizeof(double));
    }

    memcpy(output, a, (size_t)n * sizeof(double));
    free(a);
    free(b);
}

/* Approximate zero-phase filtering: forward filter, reverse, filter, reverse. */
static void bandpass_filtfilt(const double *input, double *output, int n)
{
    double *temp = (double *)malloc((size_t)n * sizeof(double));
    double *rev  = (double *)malloc((size_t)n * sizeof(double));
    double *fwd  = (double *)malloc((size_t)n * sizeof(double));

    if (!temp || !rev || !fwd) {
        fprintf(stderr, "Memory allocation failed in bandpass filter.\n");
        free(temp); free(rev); free(fwd);
        exit(1);
    }

    sos_filter_pass(input, temp, n);

    for (int i = 0; i < n; i++)
        rev[i] = temp[n - 1 - i];

    sos_filter_pass(rev, fwd, n);

    for (int i = 0; i < n; i++)
        output[i] = fwd[n - 1 - i];

    normalize_max_abs(output, n);

    free(temp);
    free(rev);
    free(fwd);
}

/* Five-point derivative from the supplied MATLAB implementation. */
static void derivative_stage(const double *x, double *y, int n)
{
    for (int i = 0; i < n; i++) y[i] = 0.0;

    for (int i = 2; i < n - 2; i++) {
        y[i] = (-x[i-2] - 2.0*x[i-1] + 2.0*x[i+1] + x[i+2]) / 8.0;
    }
    normalize_max_abs(y, n);
}

static void square_stage(const double *x, double *y, int n)
{
    for (int i = 0; i < n; i++)
        y[i] = x[i] * x[i];
}

/* Centered 150 ms moving average, retaining exactly n output samples. */
static void moving_window_integration(const double *x, double *y, int n)
{
    int half = MWI_LEN / 2;

    for (int i = 0; i < n; i++) {
        double sum = 0.0;
        int count = 0;
        int start = i - half;
        int end = start + MWI_LEN;

        for (int j = start; j < end; j++) {
            if (j >= 0 && j < n) {
                sum += x[j];
                count++;
            }
        }
        y[i] = (count > 0) ? sum / (double)MWI_LEN : 0.0;
    }
}

/* Find local maxima, then retain the largest candidate within each
   200-ms neighborhood (similar to findpeaks MinPeakDistance). */
static int find_candidates(const double *x, int n, int *locs, double *pks)
{
    int *all_locs = (int *)malloc((size_t)n * sizeof(int));
    double *all_pks = (double *)malloc((size_t)n * sizeof(double));
    int count = 0;

    if (!all_locs || !all_pks) {
        fprintf(stderr, "Memory allocation failed in peak finder.\n");
        free(all_locs); free(all_pks);
        exit(1);
    }

    for (int i = 1; i < n - 1; i++) {
        if (x[i] > x[i-1] && x[i] >= x[i+1]) {
            all_locs[count] = i;
            all_pks[count] = x[i];
            count++;
        }
    }

    /* Greedy highest-first non-maximum suppression. */
    int *used = (int *)calloc((size_t)count, sizeof(int));
    int selected = 0;

    for (int k = 0; k < count; k++) {
        int best = -1;
        for (int j = 0; j < count; j++) {
            if (!used[j] && (best < 0 || all_pks[j] > all_pks[best]))
                best = j;
        }
        if (best < 0) break;

        used[best] = 1;
        int too_close = 0;
        for (int j = 0; j < selected; j++) {
            if (abs(all_locs[best] - locs[j]) < REFRACTORY) {
                too_close = 1;
                break;
            }
        }

        if (!too_close && selected < MAX_PEAKS) {
            locs[selected] = all_locs[best];
            pks[selected] = all_pks[best];
            selected++;
        }
    }

    /* Sort selected peaks by sample index. */
    for (int i = 0; i < selected - 1; i++) {
        for (int j = i + 1; j < selected; j++) {
            if (locs[j] < locs[i]) {
                int ti = locs[i]; locs[i] = locs[j]; locs[j] = ti;
                double tp = pks[i]; pks[i] = pks[j]; pks[j] = tp;
            }
        }
    }

    free(all_locs);
    free(all_pks);
    free(used);
    return selected;
}

static double mean_value(const double *x, int n)
{
    if (n <= 0) return 0.0;
    double sum = 0.0;
    for (int i = 0; i < n; i++) sum += x[i];
    return sum / (double)n;
}

static int local_max_index(const double *x, int start, int end)
{
    int best = start;
    for (int i = start + 1; i <= end; i++)
        if (x[i] > x[best]) best = i;
    return best;
}

static double mean_slope(const double *x, int peak)
{
    int start = peak - (int)round(0.075 * FS);
    if (start < 0) start = 0;
    if (peak <= start) return 0.0;

    double sum = 0.0;
    int count = 0;
    for (int i = start; i < peak; i++) {
        sum += x[i+1] - x[i];
        count++;
    }
    return count ? sum / count : 0.0;
}

static int already_detected(const int *peaks, int n, int index)
{
    for (int i = 0; i < n; i++)
        if (peaks[i] == index) return 1;
    return 0;
}

static int detect_qrs(const double *mwi, const double *bandpass, int n,
                      int *r_peaks, double *r_amplitudes)
{
    int cand_locs[MAX_PEAKS];
    double cand_pks[MAX_PEAKS];
    int candidate_count = find_candidates(mwi, n, cand_locs, cand_pks);

    if (candidate_count == 0) return 0;

    int train_n = (int)(2.0 * FS);
    if (train_n > n) train_n = n;

    double max_mwi = 0.0, max_bp = 0.0;
    for (int i = 0; i < train_n; i++) {
        if (mwi[i] > max_mwi) max_mwi = mwi[i];
        if (bandpass[i] > max_bp) max_bp = bandpass[i];
    }

    double THR_SIG = max_mwi / 3.0;
    double THR_NOISE = mean_value(mwi, train_n) / 2.0;
    double SIG_LEV = THR_SIG;
    double NOISE_LEV = THR_NOISE;

    double THR_SIG1 = max_bp / 3.0;
    double THR_NOISE1 = mean_value(bandpass, train_n) / 2.0;
    double SIG_LEV1 = THR_SIG1;
    double NOISE_LEV1 = THR_NOISE1;

    int qrs_mwi[MAX_PEAKS];
    int qrs_count = 0;
    int raw_count = 0;
    double rr_history[8];
    int rr_count = 0;
    double mean_rr = 0.0;

    for (int c = 0; c < candidate_count; c++) {
        int loc = cand_locs[c];
        double peak = cand_pks[c];

        int search_start = loc - LOCALIZE_WIN;
        if (search_start < 0) search_start = 0;
        int bp_loc = local_max_index(bandpass, search_start, loc);
        double bp_amp = bandpass[bp_loc];

        /* Search-back: if the gap is too long, accept the strongest
           intermediate MWI candidate above the noise threshold. */
        if (qrs_count > 0 && rr_count > 0 && mean_rr > 0.0) {
            int gap = loc - qrs_mwi[qrs_count - 1];
            if (gap >= (int)round(1.66 * mean_rr)) {
                int sb_start = qrs_mwi[qrs_count - 1] + REFRACTORY;
                int sb_end = loc - REFRACTORY;
                int sb_best = -1;
                if (sb_start < 0) sb_start = 0;
                if (sb_end >= n) sb_end = n - 1;

                for (int j = 0; j < candidate_count; j++) {
                    int p = cand_locs[j];
                    if (p >= sb_start && p <= sb_end &&
                        cand_pks[j] > THR_NOISE &&
                        (sb_best < 0 || cand_pks[j] > cand_pks[sb_best]))
                        sb_best = j;
                }

                if (sb_best >= 0 && qrs_count < MAX_PEAKS) {
                    int sb_loc = cand_locs[sb_best];
                    int sb_s = sb_loc - LOCALIZE_WIN;
                    if (sb_s < 0) sb_s = 0;
                    int sb_bp = local_max_index(bandpass, sb_s, sb_loc);
                    if (bandpass[sb_bp] > THR_NOISE1 &&
                        !already_detected(r_peaks, raw_count, sb_bp)) {
                        qrs_mwi[qrs_count++] = sb_loc;
                        r_peaks[raw_count] = sb_bp;
                        r_amplitudes[raw_count++] = bandpass[sb_bp];
                        SIG_LEV = 0.25*cand_pks[sb_best] + 0.75*SIG_LEV;
                        SIG_LEV1 = 0.25*bandpass[sb_bp] + 0.75*SIG_LEV1;
                    }
                }
            }
        }

        int accept = (peak >= THR_SIG);

        /* T-wave discrimination for candidates within about 360 ms. */
        if (accept && qrs_count >= 1) {
            int gap = loc - qrs_mwi[qrs_count - 1];
            if (gap > REFRACTORY && gap <= TWAVE_LIMIT) {
                double slope_now = fabs(mean_slope(mwi, loc));
                double slope_prev = fabs(mean_slope(mwi, qrs_mwi[qrs_count - 1]));
                if (slope_now <= 0.5 * slope_prev)
                    accept = 0;
            }
        }

        /* Refractory period. */
        if (accept && qrs_count > 0 &&
            loc - qrs_mwi[qrs_count - 1] < REFRACTORY)
            accept = 0;

        if (accept) {
            if (qrs_count < MAX_PEAKS)
                qrs_mwi[qrs_count++] = loc;

            if (bp_amp >= THR_SIG1 && raw_count < MAX_PEAKS &&
                !already_detected(r_peaks, raw_count, bp_loc)) {
                r_peaks[raw_count] = bp_loc;
                r_amplitudes[raw_count] = bp_amp;
                raw_count++;
                SIG_LEV1 = 0.125*bp_amp + 0.875*SIG_LEV1;
            }

            SIG_LEV = 0.125*peak + 0.875*SIG_LEV;
        } else {
            NOISE_LEV = 0.125*peak + 0.875*NOISE_LEV;
            NOISE_LEV1 = 0.125*bp_amp + 0.875*NOISE_LEV1;
        }

        THR_SIG = NOISE_LEV + 0.25*fabs(SIG_LEV - NOISE_LEV);
        THR_NOISE = 0.5 * THR_SIG;
        THR_SIG1 = NOISE_LEV1 + 0.25*fabs(SIG_LEV1 - NOISE_LEV1);
        THR_NOISE1 = 0.5 * THR_SIG1;

        /* Update recent RR history from accepted MWI peaks. */
        if (qrs_count >= 2) {
            int rr = qrs_mwi[qrs_count - 1] - qrs_mwi[qrs_count - 2];
            if (rr > 0) {
                if (rr_count < 8) rr_history[rr_count++] = (double)rr;
                else {
                    for (int k = 0; k < 7; k++) rr_history[k] = rr_history[k+1];
                    rr_history[7] = (double)rr;
                }
                mean_rr = mean_value(rr_history, rr_count);
            }
        }
    }

    /* Sort localized raw R peaks chronologically. */
    for (int i = 0; i < raw_count - 1; i++) {
        for (int j = i + 1; j < raw_count; j++) {
            if (r_peaks[j] < r_peaks[i]) {
                int ti = r_peaks[i]; r_peaks[i] = r_peaks[j]; r_peaks[j] = ti;
                double ta = r_amplitudes[i];
                r_amplitudes[i] = r_amplitudes[j];
                r_amplitudes[j] = ta;
            }
        }
    }

    /* Remove any duplicate or too-close localized positions. */
    int write = 0;
    for (int i = 0; i < raw_count; i++) {
        if (write == 0 || r_peaks[i] - r_peaks[write-1] >= REFRACTORY) {
            r_peaks[write] = r_peaks[i];
            r_amplitudes[write] = r_amplitudes[i];
            write++;
        }
    }

    return write;
}

int main(int argc, char **argv)
{
    const char *input_file = (argc > 1) ? argv[1] : "ecg_input.txt";

    double *ecg = (double *)calloc(MAX_SAMPLES, sizeof(double));
    double *bandpass = (double *)calloc(MAX_SAMPLES, sizeof(double));
    double *deriv = (double *)calloc(MAX_SAMPLES, sizeof(double));
    double *squared = (double *)calloc(MAX_SAMPLES, sizeof(double));
    double *mwi = (double *)calloc(MAX_SAMPLES, sizeof(double));

    int *r_peaks = (int *)calloc(MAX_PEAKS, sizeof(int));
    double *r_amps = (double *)calloc(MAX_PEAKS, sizeof(double));

    if (!ecg || !bandpass || !deriv || !squared || !mwi ||
        !r_peaks || !r_amps) {
        fprintf(stderr, "Memory allocation failed.\n");
        return 1;
    }

    int n = read_ecg(input_file, ecg, MAX_SAMPLES);
    if (n <= 0) {
        fprintf(stderr, "No ECG samples loaded.\n");
        return 1;
    }

    printf("Input file: %s\n", input_file);
    printf("Sampling frequency: %.0f Hz\n", FS);
    printf("ECG samples loaded: %d\n", n);

    bandpass_filtfilt(ecg, bandpass, n);
    derivative_stage(bandpass, deriv, n);
    square_stage(deriv, squared, n);
    moving_window_integration(squared, mwi, n);

    int peak_count = detect_qrs(mwi, bandpass, n, r_peaks, r_amps);

    FILE *fp_peaks = fopen("detected_peaks_c_30_60.txt", "w");
    FILE *fp_amps = fopen("detected_amplitudes_c_30_60.txt", "w");
    FILE *fp_rr = fopen("rr_hr_c_30_60.txt", "w");

    if (!fp_peaks || !fp_amps || !fp_rr) {
        perror("Could not create output file");
        return 1;
    }

    fprintf(fp_rr, "RR_interval_seconds,Heart_rate_BPM\n");

    printf("\nDetected R-peaks: %d\n", peak_count);
    printf("R-peak sample indices (0-based):\n");

    for (int i = 0; i < peak_count; i++) {
        printf("%d\n", r_peaks[i]);
        fprintf(fp_peaks, "%d\n", r_peaks[i]);
        fprintf(fp_amps, "%.10f\n", r_amps[i]);

        if (i > 0) {
            double rr = (double)(r_peaks[i] - r_peaks[i-1]) / FS;
            double hr = (rr > 0.0) ? 60.0 / rr : 0.0;
            fprintf(fp_rr, "%.8f,%.8f\n", rr, hr);
        }
    }

    fclose(fp_peaks);
    fclose(fp_amps);
    fclose(fp_rr);

    printf("\nSaved detected_peaks_c_30_60.txt\n");
    printf("Saved detected_amplitudes_c_30_60.txt\n");
    printf("Saved rr_hr_c_30_60.txt\n");

    free(ecg); free(bandpass); free(deriv); free(squared); free(mwi);
    free(r_peaks); free(r_amps);
    return 0;
}

