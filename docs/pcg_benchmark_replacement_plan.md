# PCG Benchmark Replacement Plan

Current PCG wrappers are explicit feature/proxy tools. They should be replaced or calibrated against public labeled benchmarks before being presented as model-backed outputs.

## Segmentation, HR, Rhythm, S3/S4

- Primary benchmark: Springer/Schmidt HSMM heart-sound segmentation labels.
- Practical source: Hugging Face `alvgaona/springer-sounds`, derived from Springer's PhysioNet implementation data, with 792 PCG rows sampled at 1000 Hz and timestep labels for S1, systole, S2, and diastole.
- Replacement target: train/evaluate S1/S2 state segmentation and use state intervals for `PCG_estimate_heart_rate` and `PCG_assess_rhythm_irregularity`.
- Gap: this benchmark labels S1/systole/S2/diastole, not true pathological S3/S4. `PCG_detect_s3_s4_proxy` remains low-confidence until a labeled extra-heart-sound dataset is found.

## Murmur, Pediatric Structural Abnormality, CHD Proxy

- Primary benchmark already integrated: CirCor DigiScope / PhysioNet 2022.
- Useful labels: murmur present/absent/unknown, outcome, auscultation location, and pediatric multi-site recordings.
- Replacement target: keep `PCG_screen_murmur_patient_multisite` as the model-backed patient-level path; use CirCor outcome as a pediatric structural-abnormality proxy only, not true CHD diagnosis.
- Gap: CHD-specific labels are not the same as CirCor outcome. A recent pediatric CHD dataset reports 3004 recordings from 751 subjects, but public availability must be verified before integration.

## Valve Disease Subtype

- Strong candidate benchmark: BMD-HS, a BUET multi-disease heart-sound dataset with patient-level labels for AS, AR, MR, MS, multi-disease, and normal, and up to eight PCG recordings per patient.
- Secondary candidate: Yaseen/Khan VHD dataset, 1000 recordings across normal, AS, MS, MR, and MVP classes.
- Replacement target: train a multi-label or multi-class PCG valve classifier to replace `PCG_screen_valve_disease_proxy` candidates with calibrated AS/AR/MR/MS probabilities.
- Risk: Yaseen-style datasets are often preprocessed/short-cycle clips from heterogeneous sources, so BMD-HS is preferred for patient-level validation.

## Next Implementation Order

1. Add Springer segmentation benchmark loader and evaluate current S1/S2 proxy.
2. Train a small segmentation model or wrap Springer HSMM-style states for `PCG_segment_s1_s2_proxy` replacement.
3. Add BMD-HS downloader/manifest builder and train valve multi-label classifier for AS/AR/MR/MS/N.
4. Reuse CirCor patient multi-site pipeline for congenital/structural abnormality proxy with outcome labels, while clearly distinguishing it from CHD diagnosis.
5. Keep S3/S4 as proxy until a labeled extra-heart-sound benchmark is verified.

## Sources

- Springer sounds dataset: https://huggingface.co/datasets/alvgaona/springer-sounds
- CirCor DigiScope dataset: https://physionet.org/content/circor-heart-sound/
- BMD-HS PyHealth loader / GitHub pointer: https://pyhealth.readthedocs.io/en/latest/api/datasets/pyhealth.datasets.BMDHSDataset.html
- Yaseen VHD dataset discussion: https://www.sciencedirect.com/science/article/pii/S0020025521001298
