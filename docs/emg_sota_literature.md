# EMG SOTA Literature Notes

This note separates reproducible tool targets from headline numbers that depend on easier within-subject splits.

## Gesture / Prosthetic Control

- Common benchmarks: NinaPro DB1-DB5, CapgMyo, BioPatRec, UCI Gesture.
- Strong reported families: temporal convolutional networks (TCN), multi-stream CNN, CNN-LSTM-attention, transformer/ConvTransformer, deformable CNN, few-shot/domain adaptation.
- NinaPro DB1 numbers are highly split-dependent. Literature surveys report around 82-90% for DB1 under intra-subject or easier protocols, e.g. CNN image encodings around 82.5%, CNN-LSTM-attention around 86.36%, TCN around 89.76%, and HVPN around 88.4% intra-subject. These are not directly comparable to strict subject-held-out transfer.
- Practical tool choice: report both strict subject-held-out zero-calibration and calibrated-user held-out-repetition. For deployment, user calibration/adaptation is the realistic target.

## Whole-body / Physical Action Recognition

- Public EMG action benchmarks are smaller than hand-gesture benchmarks. The UCI Physical Action dataset has only four subjects and 20 classes.
- SOTA-style model families are CNN/TCN/LSTM over raw multi-channel windows, but subject-held-out performance can remain low because action execution and electrode placement vary strongly.
- Practical tool choice: keep 20-class action recognition as a coarse research baseline and keep binary normal/aggressive as a more stable dataset-specific screen.

## Movement Intent / Lower Limb / Gait

- Lower-limb intent recognition literature uses CNN, LSTM, TCN, attention CNN, residual learning, and sensor fusion with IMU/FSR/kinematics.
- Datasets: SIAT-LLMD, GEDS, Camargo-style locomotion datasets, BASAN/JJ-style lower-limb datasets.
- Practical tool choice: for GEDS, compare feature ensembles against raw-sequence TCN for walking speed and stance/swing phase under subject-held-out GroupKFold.

## Muscle Fatigue

- Classical SOTA remains strong: RMS/MAV, median frequency, mean power frequency, spectral entropy, wavelet features, fractal/complexity features with SVM/RF/XGBoost.
- Deep methods increasingly use CNN/TCN/LSTM on raw windows or time-frequency images, but public fatigue labels are often protocol proxies rather than subjective/physiological ground truth.
- Practical tool choice: add CWT/spectrogram-CNN or 1D TCN only when label quality is clear; otherwise keep fatigue as protocol-level screening.

## Neuromuscular Abnormality

- Needle EMG disease classification papers use CWT/spectrogram CNN, attention-CNN, CNN-GRU, and conventional ML. Recent large clinical datasets report modest real-world performance, e.g. CWT + two-layer CNN around 62% accuracy on a 608-participant clinical database.
- Public EMGDB is small for robust deep learning. High smoke-test scores on snippets are not evidence of clinical performance.
- Practical tool choice: keep explicit research-only disclaimer unless a larger subject-level clinical dataset is available.

## Implementation Priority

1. NinaPro DB1 TCN/ResNet under calibrated-user and strict subject-held-out splits.
2. GEDS raw-sequence TCN for gait speed and stance/swing phase.
3. UCI Physical Action raw-sequence TCN for 20-class action recognition.
4. Fatigue spectrogram/TCN only as protocol-label benchmark.
5. Neuromuscular spectrogram CNN only if more subject-level clinical data is added.

## Sources

- Gesture recognition survey: https://pmc.ncbi.nlm.nih.gov/articles/PMC8107289/
- EMG pattern recognition and deep learning challenges: https://www.mdpi.com/2504-2289/2/3/21
- HVPN NinaPro comparison: https://pmc.ncbi.nlm.nih.gov/articles/PMC8413066/
- User-independent adaptive EMG recognition: https://www.frontiersin.org/articles/10.3389/fnins.2022.847180/full
- Deep learning taxonomy for EMG signals: https://www.sciencedirect.com/science/article/pii/S0925231220319020
- Lower-limb movement dataset context: https://www.nature.com/articles/s41597-023-02263-3
- Neuromuscular AI clinical database example: https://pubmed.ncbi.nlm.nih.gov/41316805/


## Local SOTA-Family Attempts

- NinaPro DB1 TinyTCN, calibrated held-out repetitions: window top-1 0.340/top-5 0.642; worse than augmented ExtraTrees window top-1 0.585/top-5 0.821.
- NinaPro DB1 TinyTCN, strict subject-held-out: window top-1 0.148/top-5 0.412; better than feature zero-calibration top-1 0.107, but still far from intra-subject literature numbers.
- NinaPro DB1 multi-stream CNN with channel attention, 1s windows, calibrated held-out repetitions: window top-1 0.541/top-5 0.828; repetition/trial voting top-1 0.732/top-5 0.937. This is the strongest local continuous-decision benchmark so far.
- NinaPro DB1 multi-stream CNN with channel attention, strict subject-held-out: window top-1 0.209/top-5 0.500; repetition/trial voting top-1 0.274/top-5 0.593. This is the strongest local zero-calibration deep benchmark so far.
- UCI Physical Action raw-window TCN, leave-one-subject-out: accuracy 0.147/macro-F1 0.151; worse than the feature ensemble accuracy 0.269/macro-F1 0.266.


- GEDS gait speed raw-sequence TCN, subject split: accuracy 0.718/macro-F1 0.724; below feature ensemble accuracy 0.789/macro-F1 0.788.
- GEDS gait phase raw-sequence TCN, subject split: accuracy 0.972/macro-F1 0.972/AUROC 0.996; near parity but slightly below feature ensemble accuracy 0.974/macro-F1 0.972/AUROC 0.997.

Decision: keep `EMG_classify_prosthetic_gesture` defaulting to the feature ensemble for fast calibrated window-level inference, and expose the multi-stream CNN as `backend="multistream_cnn"` for SOTA-family continuous-decision inference. Do not replace `EMG_classify_action` with raw TCN.
