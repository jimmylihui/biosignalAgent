# BioSignalAgent Existing-Work Source Catalog

This catalog maps each current BioSignalAgent task to existing algorithms, libraries, datasets, and wrapper priorities. It follows the TxAgent-style pattern: use trusted existing work as tools, then evaluate tool selection and tool execution with structured traces.

## Priority Key

| Priority | Meaning |
| ---: | --- |
| 1 | Implement next; directly improves current weak labeled benchmarks. |
| 2 | Near-term baseline improvement with available public labels. |
| 3 | Useful once a stronger dataset/evaluator is attached. |
| 4 | Later expansion; keep scaffolded but do not optimize first. |
| 5 | Backlog. |

## P1 Wrappers To Implement Next

| Modality | Task | Existing Work | Candidate Datasets | Next Wrapper |
| --- | --- | --- | --- | --- |
| ECG | AF and rhythm classification | PhysioNet/CinC 2017 single-lead AF challenge, RR irregularity plus P-wave/f-wave feature methods, CNN, DenseNet, CNN-BiLSTM rhythm classifiers | PhysioNet/CinC 2017, MIT-BIH AFDB, MIT-BIH Arrhythmia rhythm aux notes | Add feature-based AF/rhythm classifier baseline over the existing ECG rhythm benchmark. |
| ECG | R-peak and QRS detection | Pan-Tompkins QRS detector, Hamilton-style QRS detectors, NeuroKit2 ECG cleaning/peak detection | MIT-BIH Arrhythmia Database, QT Database, Noise Stress Test Database | Keep Pan-Tompkins as default, add detector comparison and annotation-tolerance evaluator. |
| EEG+EOG+EMG | Sleep staging | YASA automatic sleep staging, DeepSleepNet, U-Sleep | Sleep-EDF, ISRUC, UCDDB | Add YASA wrapper and PSG manifest channels for EEG/EOG/EMG, then compare against current bandpower baseline. |
| PCG | Normal/abnormal and murmur classification | PhysioNet/CinC 2016 sample logistic-regression classifier, MFCC plus CNN, feature and deep-learning ensembles | PhysioNet/CinC 2016, CirCor DigiScope / PhysioNet 2022 | Feature+logistic-regression baseline is implemented; next expand records and add Springer/HSMM or MFCC features. |
| PCG | S1/S2 segmentation | Springer logistic-regression HSMM segmentation, Schmidt HMM-style segmentation, PhysioToolkit HSS package | PhysioNet/CinC 2016, PASCAL Heart Sound Challenge | Replace alternating-peak proxy with Springer HSMM annotations or wrapper. |
| RESP+SpO2+PSG | Sleep apnea and hypopnea detection | AASM airflow reduction rules, SpO2 desaturation lag fusion, PSG event detection | UCDDB, SHHS, MESA | Improve UCDDB PSG respiratory-event recall with Flow envelope, hypopnea, desaturation timing, and fusion thresholds. |

## Full Catalog

| Priority | Modality | Task | Current Tools | Candidate Libraries | Candidate Datasets | Key Sources |
| ---: | --- | --- | --- | --- | --- | --- |
| 1 | ECG | AF and rhythm classification | `ECG_screen_arrhythmia` | wfdb, scikit-learn, pytorch | PhysioNet/CinC 2017, MIT-BIH AFDB, MIT-BIH Arrhythmia rhythm aux notes | [1](https://physionet.org/challenge/2017/), [2](https://physionet.org/content/afdb/), [3](https://pmc.ncbi.nlm.nih.gov/articles/PMC8052181/) |
| 1 | ECG | R-peak and QRS detection | `ECG_detect_r_peaks` | neurokit2, wfdb, biosppy | MIT-BIH Arrhythmia Database, QT Database, Noise Stress Test Database | [1](https://physionet.org/content/mitdb/), [2](https://wfdb.readthedocs.io/), [3](https://neuropsychology.github.io/NeuroKit/) |
| 1 | EEG+EOG+EMG | Sleep staging | `EEG_compute_bandpower`, `EEG_estimate_sleep_stage_features` | yasa, mne, pytorch | Sleep-EDF, ISRUC, UCDDB, MESA | [1](https://yasa-sleep.org/generated/yasa.SleepStaging.html), [2](https://arxiv.org/abs/1703.04046), [3](https://arxiv.org/abs/2111.08446) |
| 1 | PCG | Normal/abnormal and murmur classification | `PCG_screen_murmur_proxy`, `PCG_detect_heart_sounds`, `PCG_extract_murmur_features` | scipy, scikit-learn, pytorch | PhysioNet/CinC 2016, CirCor DigiScope / PhysioNet 2022 | [1](https://physionet.org/content/challenge-2016/), [2](https://physionet.org/content/circor-heart-sound/), [3](https://arxiv.org/abs/1707.04642) |
| 1 | PCG | S1/S2 segmentation | `PCG_segment_s1_s2_proxy` | PhysioToolkit HSS, matlab/octave wrappers, python ports if available | PhysioNet/CinC 2016, PASCAL Heart Sound Challenge | [1](https://archive.physionet.org/physiotools/hss/), [2](https://physionet.org/content/challenge-2016/) |
| 1 | RESP+SpO2+PSG | Sleep apnea and hypopnea detection | `RESP_detect_apnea`, `RESP_detect_hypopnea`, `SpO2_detect_desaturation`, `Session_screen_sleep_apnea_multimodal` | yasa, mne, scipy | UCDDB, SHHS, MESA, Apnea-ECG | [1](https://physionet.org/content/ucddb/), [2](https://sleepdata.org/datasets/shhs), [3](https://physionet.org/content/apnea-ecg/) |
| 2 | ECG | Beat classification | `ECG_detect_r_peaks`, `ECG_screen_arrhythmia` | wfdb, scikit-learn, pytorch | MIT-BIH Arrhythmia Database, MIT-BIH Supraventricular Arrhythmia Database, INCART | [1](https://physionet.org/content/mitdb/), [2](https://pubmed.ncbi.nlm.nih.gov/41395626/) |
| 2 | PPG | Pulse peak detection | `PPG_detect_peaks` | heartpy, pyPPG, neurokit2 | PPG-beats datasets, MIMIC PERform, BIDMC PPG and Respiration | [1](https://ppg-beats.readthedocs.io/en/latest/datasets/summary/), [2](https://github.com/paulvangentcom/heartrate_analysis_python), [3](https://pyppg.readthedocs.io/) |
| 2 | RESP | Respiratory rate and pattern | `RESP_estimate_rate`, `RESP_screen_rate_pattern` | neurokit2, scipy | BIDMC, UCDDB, MIMIC waveform | [1](https://physionet.org/content/bidmc/), [2](https://physionet.org/content/ucddb/) |
| 2 | SpO2 | Desaturation and hypoxemia burden | `SpO2_summarize`, `SpO2_detect_desaturation`, `SpO2_assess_hypoxemia_burden` | scipy | UCDDB, SHHS, MESA | [1](https://physionet.org/content/ucddb/), [2](https://sleepdata.org/datasets/mesa) |
| 3 | ABP | Pressure events and hemodynamics | `ABP_detect_pulses`, `ABP_screen_pressure_events`, `ABP_compute_hemodynamics` | wfdb, scipy, scikit-learn | MIMIC waveform, BIDMC, UCI cuffless BP datasets | [1](https://physionet.org/content/mimic3wdb/), [2](https://physionet.org/content/bidmc/) |
| 3 | BCG | J-peak, heart rate, and respiration | `BCG_detect_j_peaks`, `BCG_estimate_respiration` | scipy, pywavelets | CEBSDB, bed-based BCG datasets, chair/mattress BCG datasets | [1](https://pubmed.ncbi.nlm.nih.gov/25312966/), [2](https://arxiv.org/abs/1807.00951), [3](https://arxiv.org/abs/1809.03174) |
| 3 | ECG | ECG morphology intervals | `ECG_measure_morphology_intervals` | neurokit2, wfdb | QT Database, MIT-BIH Arrhythmia Database | [1](https://physionet.org/content/qtdb/), [2](https://neuropsychology.github.io/NeuroKit/) |
| 3 | EDA | Tonic/phasic decomposition and SCR events | `EDA_summarize`, `EDA_detect_arousal_events` | neurokit2, pyEDA, cvxEDA | WESAD, AMIGOS, DEAP | [1](https://www.mdpi.com/1424-8220/22/22/8886), [2](https://www.sciencedirect.com/science/article/pii/S1877050921006438) |
| 3 | EDA+ECG+ACC | Stress/arousal classification | `EDA_detect_arousal_events`, `EDA_screen_stress_proxy`, `ECG_compute_hrv`, `ACC_summarize_activity` | scikit-learn, neurokit2 | WESAD | [1](https://pmc.ncbi.nlm.nih.gov/articles/PMC12987280/), [2](https://www.mdpi.com/1424-8220/22/22/8886) |
| 3 | PPG | PPG AF and pulse irregularity | `PPG_assess_perfusion_variability`, `PPG_screen_pulse_irregularity` | scikit-learn, pytorch | MIMIC PERform AF, PPG AF datasets, simultaneous ECG/PPG subsets | [1](https://www.nature.com/articles/s41746-019-0207-9), [2](https://pmc.ncbi.nlm.nih.gov/articles/PMC9632370/), [3](https://ppg-beats.readthedocs.io/en/latest/datasets/summary/) |
| 3 | PPG | Respiration from PPG | `PPG_estimate_respiration_modulation` | scipy, neurokit2 | BIDMC PPG and Respiration, MIMIC waveform, Capnobase | [1](https://www.mdpi.com/1424-8220/22/6/2079/htm), [2](https://www.ncbi.nlm.nih.gov/books/NBK543644/) |
| 3 | PPG+ECG | Pulse arrival/transit timing and BP proxy | `Session_compute_ecg_ppg_pulse_arrival` | scipy, scikit-learn | BIDMC, MIMIC waveform | [1](https://pmc.ncbi.nlm.nih.gov/articles/PMC6912608/), [2](https://www.mdpi.com/1424-8220/20/19/5699) |
| 3 | SCG | J-peak, heart rate, respiration, and cardiac timing | `SCG_detect_j_peaks`, `SCG_estimate_respiration` | scipy, pywavelets | CEBSDB, OpenSCG-style datasets, SCG+ECG lab datasets | [1](https://pubmed.ncbi.nlm.nih.gov/25312966/), [2](https://www.mdpi.com/1424-8220/22/23/9565), [3](https://arxiv.org/abs/1803.10346) |
| 4 | ACC | Activity recognition and actigraphy sleep/wake | `ACC_summarize_activity`, `ACC_estimate_sleep_wake`, `ACC_detect_activity_bouts` | scikit-learn, tsfresh | UCI HAR, WISDM, MESA actigraphy/PSG, SHHS actigraphy | [1](https://archive.ics.uci.edu/dataset/240/human+activity+recognition+using+smartphones), [2](https://sleepdata.org/datasets/mesa), [3](https://sleepdata.org/datasets/shhs) |
| 4 | ACC | Fall and impact detection | `ACC_detect_fall_proxy` | scikit-learn, tsfresh | UniMiB SHAR, SisFall, MobiFall, UP-Fall | [1](https://pmc.ncbi.nlm.nih.gov/articles/PMC5539544/), [2](https://www.sciencedirect.com/science/article/abs/pii/S1350453315001575) |
| 4 | EEG | EEG artifact detection | `EEG_detect_artifact_proxy`, `Signal_detect_artifacts` | mne, autoreject, yasa | EEG artifact datasets, Sleep PSG with EOG/EMG context | [1](https://mne.tools/stable/), [2](https://autoreject.github.io/) |
| 4 | EEG | Seizure-like event detection | `EEG_screen_seizure_like_activity` | mne, scipy, pytorch | CHB-MIT, TUH EEG Seizure Corpus | [1](https://physionet.org/content/chbmit/), [2](https://isip.piconepress.com/projects/tuh_eeg/) |
| 4 | EMG | Activation, bursts, and onset detection | `EMG_summarize_activation`, `EMG_detect_bursts` | scipy, neurokit2 | EMGDB, Ninapro, gesture/fatigue datasets | [1](https://pmc.ncbi.nlm.nih.gov/articles/PMC10059683/), [2](https://ninapro.hevs.ch/) |
| 4 | EMG | Muscle fatigue | `EMG_estimate_fatigue` | scipy, pywavelets | Ninapro, fatigue protocol datasets, EMGDB for abnormality not fatigue | [1](https://www.mdpi.com/2076-3417/9/15/2952), [2](https://www.frontiersin.org/articles/10.3389/fnsys.2022.893275/full) |

## Implementation Order

1. PCG scale-up: expand the current feature+logistic-regression baseline beyond the 10-record smoke split, then add Springer HSMM/S1-S2 or MFCC features.
2. PSG v2: YASA sleep staging wrapper plus UCDDB/Sleep-EDF evaluation, and stronger Flow/SpO2 respiratory-event fusion.
3. ECG v2: feature-based AF/rhythm classifier and AAMI beat classifier over MIT-BIH/CinC 2017 style labels.
4. PPG v2: expand the new MIMIC PERform AF benchmark, then compare HeartPy/pyPPG/PPG-beats detectors and artifact filtering.
5. EDA/ACC/EMG expansion: WESAD stress, fall datasets, and EMG gesture/fatigue benchmarks.
