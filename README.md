# EEG Motor Movement and Imagery Classification

This repository contains an end-to-end EEG processing and machine-learning pipeline for the PhysioNet **EEG Motor Movement/Imagery Dataset**. It loads EDF recordings, standardizes and preprocesses the signals, creates task epochs, extracts frequency and time-frequency features, trains several classifiers, and evaluates both within-subject and cross-subject performance.

The project is intended for a reproducible research workflow rather than a real-time BCI application. The saved CSV files and PNG figures in this repository are example outputs from the implemented pipeline.

## Pipeline

```mermaid
flowchart LR
    A[PhysioNet EDF recordings] --> B[Load and identify subject/run]
    B --> C[Rename channels and set montage]
    C --> D[CAR and 8-30 Hz band-pass filter]
    D --> E[Map annotations and create 0-4 s epochs]
    E --> F[Reject noisy epochs]
    F --> G{Representation}
    G --> H[Welch band power\nmu 8-12 Hz, beta 13-30 Hz]
    G --> I[Morlet time-frequency\n4 time bins]
    G --> J[Raw epochs for CSP]
    H --> K[Scaler + classifier]
    I --> K
    J --> L[CSP + LDA]
    K --> M[5-fold per-subject CV]
    K --> N[GroupKFold cross-subject CV]
    L --> M
    L --> N
    M --> O[Accuracy, F1, ROC-AUC]
    N --> O
    O --> P[CSV results and PNG plots]
```

## What is implemented

- **Data loading:** reads PhysioNet EDF+ recordings and organizes them by subject and run.
- **Run interpretation:** identifies baseline, unilateral, and bilateral runs, and distinguishes execution from imagery.
- **Preprocessing:** channel-name cleanup, standard 10-10/10-20 montage, common average reference, optional notch filtering, 8-30 Hz band-pass filtering, optional ICA, and amplitude-based epoch rejection.
- **Epoching:** creates 4-second epochs from task annotations using consistent event codes across runs and subjects.
- **Feature extraction:**
  - Welch power in the mu (8-12 Hz) and beta (13-30 Hz) bands.
  - Morlet wavelet power with four temporal bins per epoch.
  - Raw epochs for CSP.
- **Models:** LDA, linear SVM, RBF SVM, Random Forest, Histogram Gradient Boosting, CSP + LDA, and an optional EEGNet implementation when PyTorch is available. The committed result grid includes all of these classifiers across the supported tasks.
- **Evaluation:** stratified 5-fold cross-validation within subjects and group-aware cross-validation across subjects.
- **Outputs:** cached features, experiment grids in CSV format, confusion matrices, PSD comparisons, and per-subject accuracy plots.

## Dataset and run structure

The code follows the run protocol used by the PhysioNet dataset:

| Run type | Run IDs | Meaning |
| --- | --- | --- |
| Baseline, eyes open | 1 | Continuous baseline |
| Baseline, eyes closed | 2 | Continuous baseline |
| Unilateral | 3, 4, 7, 8, 11, 12 | Left/right fist events |
| Bilateral | 5, 6, 9, 10, 13, 14 | Both fists/both feet events |

The code maps the original annotations as follows:

| Run family | Rest | Class 1 | Class 2 |
| --- | --- | --- | --- |
| Unilateral | `T0 -> rest` | `T1 -> left_fist` | `T2 -> right_fist` |
| Bilateral | `T0 -> rest` | `T1 -> both_fists` | `T2 -> both_feet` |

The local `files/` directory is ignored by Git. Download the dataset from [PhysioNet](https://physionet.org/content/eegmmidb/1.0.0/) and place the EDF files under `src/files/` using the dataset's subject directories, for example `src/files/S001/S001R01.edf`.

## Classification tasks

The available binary tasks are defined in `src/značajke.py`:

| Task | Label 0 | Label 1 |
| --- | --- | --- |
| `left_right_fist` | Left fist | Right fist |
| `fists_feet` | Both fists | Both feet |
| `execution_vs_imagery` | Real execution | Imagined execution |
| `rest_vs_task` | Rest | Any motor task |

## Project layout

```text
.
├── README.md
├── .gitignore
├── z_rad/                              # local venv for the current working copy (optional)
├── src/
│   ├── obrada.py                       # Loading, annotation mapping, epoching
│   ├── pretprocesiranje.py             # Preprocessing and diagnostic plots
│   ├── značajke.py                     # Tasks and feature extraction
│   ├── modeli.py                       # ML pipelines and CV helpers
│   ├── evaluacija.py                   # Evaluation grid and result export
│   ├── dedupe.py                       # Utility for duplicate data handling
│   ├── files/                          # PhysioNet EDF data (not tracked by Git)
│   ├── processed/                      # cached processed subject data
│   ├── cache/                          # cached feature matrices for repeated runs
│   ├── results/                        # CSV grids and generated plots
│   ├── example_epoch.png               # Example epoch plot
│   ├── psd_comparison.png              # PSD before/after preprocessing
│   └── __pycache__/                    # local Python cache
└── .venv/                              # optional fresh environment created by the user
```


## Installation

Use Python 3.10 or newer and install the scientific Python dependencies:

```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install mne numpy scipy pandas matplotlib scikit-learn
```

If you want to run the optional EEGNet comparison model as well, install PyTorch in the same environment:

```powershell
python -m pip install torch
```

The repository also contains a local virtual environment in some working copies (for example `z_rad/` or a user-created `.venv/`). These directories are ignored by Git and are not required for normal use; creating a fresh environment is recommended.

## Running the pipeline

Run commands from the repository root. Because the modules use local imports and relative data paths, execute scripts from `src/`:

```powershell
Set-Location .\src
python .\obrada.py
```

The loader example applies common average reference, an 8-30 Hz band-pass filter, and 150 microvolt peak-to-peak epoch rejection. It verifies subject `S001` and can save processed FIF files when `save_subject_epochs(subjects)` is enabled in the script.

To generate preprocessing diagnostic figures:

```powershell
Set-Location .\src
python .\pretprocesiranje.py
```

To run the example model evaluation and the configured experiment grid:

```powershell
Set-Location .\src
python .\evaluacija.py
```

Results are written to `src/results/`. Feature computations are cached in `src/cache/` so interrupted grid experiments can resume without recomputing completed combinations.

## Evaluation methodology

Two evaluation scenarios are provided:

1. **Per-subject:** a stratified 5-fold split is performed independently for each subject. This measures how well a model can work after subject-specific data are available.
2. **Cross-subject:** subjects are kept intact as groups using `GroupKFold` or leave-one-subject-out evaluation. A subject must never appear in both training and test data. This is the more realistic estimate of generalization to unseen people.

All learned transformations, including scaling, PCA, and CSP, are placed inside scikit-learn pipelines so they are fitted within each training fold. Feature caches are keyed by task, representation, channel selection, modality, subject IDs, and the minimum epoch threshold.

The default grid excludes subjects with fewer than 30 usable epochs and removes individual epochs containing non-finite feature values. These are methodological choices and should be reported alongside any newly generated results.

## Recorded results

The committed CSVs in `src/results/` contain a full evaluation grid for both scenarios and all implemented tasks. Each task was evaluated with the following classifiers: `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, and `csp_lda`.

### Evaluation coverage by scenario and task

| Scenario | Task | Tested classifiers | Best-performing configuration | Accuracy | F1 | ROC-AUC | n |
| --- | --- | --- | --- | ---: | ---: | ---: | ---: |
| Per-subject | `left_right_fist` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Raw, motor, `csp_lda` | 0.668925 | 0.651037 | 0.717191 | 57 |
| Per-subject | `fists_feet` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Raw, all, `csp_lda` | 0.735367 | 0.734150 | 0.774070 | 56 |
| Per-subject | `execution_vs_imagery` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Band power, all, `svm_linear` | 0.794717 | 0.784755 | 0.855377 | 64 |
| Per-subject | `rest_vs_task` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Band power, all, `svm_linear` | 0.793096 | 0.800346 | 0.857445 | 74 |
| Cross-subject | `left_right_fist` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Time-frequency, all, `svm_linear` | 0.641655 | 0.611649 | 0.689140 | 10 |
| Cross-subject | `fists_feet` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Raw, all, `csp_lda` | 0.621192 | 0.590618 | 0.694545 | 8 |
| Cross-subject | `execution_vs_imagery` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Band power, all, `random_forest` | 0.574283 | 0.572693 | 0.611586 | 10 |
| Cross-subject | `rest_vs_task` | `lda`, `svm_linear`, `svm_rbf`, `random_forest`, `gradient_boosting`, `csp_lda` | Time-frequency, all, `gradient_boosting` | 0.652535 | 0.669484 | 0.715282 | 10 |

This matrix makes the actual evaluation intent explicit: every classifier was applied across every task in both the per-subject and cross-subject scenarios; the table only highlights the best result for each task/scenario pair, while the full data remain in the CSV files.

The difference between per-subject and cross-subject performance is expected: EEG signals vary substantially between people, and cross-subject evaluation prevents the model from relying on subject-specific patterns.

## Figures

### PSD before and after preprocessing

The figure compares the power spectral density of a recording before and after the configured preprocessing steps.

![PSD comparison before and after preprocessing](src/psd_comparison.png)

### Example epoch

This figure shows the time course of selected EEG channels from one 4-second epoch.

![Example EEG epoch](src/example_epoch.png)

### Confusion matrix

The confusion matrix below shows the `rest_vs_task` classification performance for the best-performing per-subject band-power SVM-linear configuration.

![Confusion matrix for rest_vs_task SVM-linear](src/results/confusion_matrix_rest_vs_task_svm_linear.png)

### Per-subject accuracy distribution

The boxplot shows the distribution of per-subject accuracy for the `rest_vs_task` task using LDA and band-power features.

![Per-subject accuracy boxplot](src/results/boxplot_rest_vs_task_lda.png)

## Reproducibility notes

- The default random seed is `42`.
- The sampling rate specified by the dataset is 160 Hz.
- Epochs span 0 to 4 seconds after each task event.
- Normalization statistics must be estimated from training data only; fitting them on the complete dataset would leak information from the test folds.
- CSP is fitted inside the cross-validation pipeline for the same reason.
- The CSV files preserve the exact aggregate outputs generated by the current implementation. Re-running experiments after changing dependencies, preprocessing settings, or the input data may produce different values.

## Limitations and next steps

- The current repository does not include automated unit tests or a pinned dependency lockfile.
- ICA is fitted and returned for inspection, but automatic component selection is not enabled because the dataset does not provide dedicated EOG channels.
- The committed result grid covers all four implemented tasks (`left_right_fist`, `fists_feet`, `execution_vs_imagery`, and `rest_vs_task`), but not every possible classifier-feature-channel combination is exhaustively reported in the figures.
- Future work could add nested hyperparameter tuning, confidence intervals, explicit subject metadata, and a fully automated figure/report generation command.

## License and data use

This repository contains research code. Consult the [PhysioNet dataset page](https://physionet.org/content/eegmmidb/1.0.0/) for the dataset's citation, license, and data-use conditions. Cite the original dataset and the relevant methods when reusing the data or results.
