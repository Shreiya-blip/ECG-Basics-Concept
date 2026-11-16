# ECG Signal Processing – 

## Overview

This section focuses on ECG signal preprocessing and cardiac feature extraction using the MIT-BIH Arrhythmia Database.

The experiments use **Record 100**, with a sampling frequency of 360 Hz and the MLII ECG channel. The first 30 seconds of the recording (10,800 samples) are used for the filtering experiments to maintain consistency during analysis.

The main objectives are to investigate noise-removal techniques, improve ECG signal quality, and extract important cardiac parameters such as RR intervals and heart rate.

---

##  Power-Line Interference Removal Using an IIR Notch Filter

### Objective
To design and implement an IIR notch filter to attenuate power-line interference at 60 Hz in the ECG signal and evaluate its attenuation performance before and after filtering.

### Methodology
- Load the ECG signal from MIT-BIH Record 100.
- Select the first 30 seconds of the MLII channel.
- Design a second-order IIR notch filter using a 60 Hz notch frequency.
- Set the quality factor (Q) to 30.
- Apply the notch filter using forward-backward filtering.
- Compare the original and filtered signals in the time and frequency domains.
- Calculate attenuation at 60 Hz using FFT magnitudes.

### Mathematical Concepts
- Sampling interval calculation.
- Nyquist frequency.
- Normalized notch frequency.
- Quality factor and bandwidth.
- FFT frequency resolution.
- Attenuation in decibels (dB).

### Expected Outcome
Reduction of the 60 Hz frequency component while maintaining the overall ECG waveform.

---

##  High-Frequency Noise Reduction Using a Butterworth Low-Pass Filter

### Objective
To design and implement a fourth-order Butterworth low-pass filter with a cutoff frequency of 40 Hz to attenuate high-frequency noise in the ECG signal.

### Methodology
- Load the ECG signal from MIT-BIH Record 100.
- Select the first 30 seconds of the recording.
- Calculate the Nyquist frequency and normalized cutoff frequency.
- Design a fourth-order Butterworth low-pass filter.
- Apply the filter using forward-backward filtering.
- Compare the original and filtered ECG signals.
- Perform FFT analysis to observe the reduction of frequency components above the cutoff frequency.
- Calculate attenuation at 60 Hz.

### Mathematical Concepts
- Nyquist frequency.
- Normalized cutoff frequency.
- Butterworth filter design.
- FFT magnitude analysis.
- Attenuation calculation.

### Expected Outcome
Reduction of high-frequency components while retaining the important ECG waveform features, including the QRS complex.

---

## ECG Signal Enhancement Using a Butterworth Band-Pass Filter

### Objective
To design and implement a fourth-order Butterworth band-pass filter with cutoff frequencies of 0.5 Hz and 40 Hz to reduce baseline wander and high-frequency noise in the ECG signal.

### Methodology
- Load the ECG signal from MIT-BIH Record 100.
- Select the first 30 seconds of the recording.
- Set the lower cutoff frequency to 0.5 Hz and upper cutoff frequency to 40 Hz.
- Calculate the normalized cutoff frequencies.
- Design a Butterworth band-pass filter.
- Apply the filter using forward-backward filtering.
- Compare the original and filtered ECG signals in the time domain.
- Analyse the frequency spectrum using FFT.
- Examine the preservation of P waves, QRS complexes and T waves.

### Mathematical Concepts
- Nyquist frequency.
- Normalized lower and upper cutoff frequencies.
- Bandwidth calculation.
- Centre frequency calculation.
- Butterworth band-pass filter design.
- Frequency-domain analysis.

### Expected Outcome
Reduction of low-frequency baseline variations and high-frequency noise while preserving important ECG waveform characteristics.

---

##  RR Interval and Heart Rate Calculation Using MIT-BIH Annotations

### Objective
To identify consecutive R-peak locations using MIT-BIH reference annotations, calculate RR intervals and instantaneous heart rate, and analyse variations in cardiac rhythm.

### Methodology
- Load MIT-BIH Record 100 and its corresponding annotation file.
- Select the first 30 seconds of ECG data.
- Extract annotated heartbeat locations.
- Calculate the difference between consecutive R-peak sample indices.
- Convert the sample differences into time intervals.
- Calculate instantaneous heart rate in beats per minute (BPM).
- Calculate the average RR interval and average heart rate.
- Plot RR interval and heart-rate variations.

### Mathematical Concepts

**Sampling Interval:**

Ts = 1 / fs

**RR Interval:**

RR = (R(i+1) - R(i)) / fs

**Instantaneous Heart Rate:**

HR = 60 / RR

**Average RR Interval:**

RR(avg) = Sum of RR intervals / Number of RR intervals

**Average Heart Rate:**

HR(avg) = 60 / RR(avg)

### Expected Outcome
Extraction of RR intervals and heart-rate values from annotated ECG beats, providing important features for subsequent cardiac rhythm analysis.

**Note:** This experiment uses reference annotations to calculate RR intervals. It does not implement automatic R-peak detection. Automatic detection and verification are addressed separately.

---

## Technologies and Libraries Used

- Python
- NumPy – Numerical computations and array processing.
- SciPy – Digital filter design and signal processing.
- Matplotlib – ECG waveform and frequency-spectrum visualization.
- WFDB – Loading MIT-BIH ECG records and reference annotations.

## Dataset

**MIT-BIH Arrhythmia Database**

- Record: 100
- Channel: MLII (first ECG channel)
- Sampling Frequency: 360 Hz
- Selected Duration: 30 seconds
- Number of Samples: 10,800

## Overall ECG Processing Workflow

1. Power-line interference removal using a notch filter.
2. High-frequency noise reduction using a low-pass filter.
3. Baseline wander and high-frequency noise reduction using a band-pass filter.
4. RR interval and heart-rate extraction using reference annotations.

These experiments establish the signal-processing foundation for an ECG monitoring pipeline, supporting subsequent R-peak detection, cardiac rhythm classification and embedded ECG processing.
## Beat Detection Failure Analysis
Investigated the effect of different detection threshold values on ECG beat detection. Analysed how unsuitable thresholds can cause missed beats and false detections. Adjusted the threshold and evaluated detection performance using true positives (TP), false positives (FP), false negatives (FN), sensitivity and precision.
