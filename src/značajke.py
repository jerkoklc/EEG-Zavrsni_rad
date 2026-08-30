"""Izdvajanje značajki iz EEG epoha i definicije klasifikacijskih zadataka.

Implementirane reprezentacije signala:
  - band-power (5.1, stavka 2)                -> extract_band_power_features
  - vremensko-frekvencijske / wavelet (5.1, stavka 3) -> extract_time_frequency_features
  - sirove epohe za CNN/EEGNet (5.1, stavka 4, opcionalno) -> prepare_raw_epoch_array

CSP je supervizirana transformacija, pa se nalazi u sklearn pipelineu u
modeli.py i fita se unutar svakog train folda.
"""

import numpy as np
import mne
from scipy.signal import welch

from obrada import CODE_MAP

BANDS = {'mu': (8.0, 12.0), 'beta': (13.0, 30.0)}


# Band power značajke.

def extract_band_power_features(epochs, bands=None, picks=None, nperseg=None):
    """Računa prosječnu Welch snagu po kanalu i frekvencijskom pojasu.

    Vraća: X oblika (n_epoha, n_kanala * n_pojaseva) i ista imena značajki
    redom po pojasima i kanalima.
    """
    bands = bands or BANDS
    data = epochs.get_data(picks=picks)  # (n_epoha, n_kanala, n_uzoraka)
    ch_names = picks if picks is not None else epochs.ch_names
    sfreq = epochs.info['sfreq']
    nperseg = nperseg or min(256, data.shape[-1])

    freqs, psd = welch(data, fs=sfreq, nperseg=nperseg, axis=-1)
    # Oblik je (epoha, kanal, frekvencija).

    feature_blocks = []
    feature_names = []
    for band_name, (fmin, fmax) in bands.items():
        mask = (freqs >= fmin) & (freqs <= fmax)
        band_power = psd[..., mask].mean(axis=-1)  # (n_epoha, n_kanala)
        feature_blocks.append(band_power)
        feature_names.extend([f'{ch}_{band_name}' for ch in ch_names])

    X = np.concatenate(feature_blocks, axis=1)
    return X, feature_names


# Vremensko-frekvencijske značajke.

def extract_time_frequency_features(epochs, bands=None, n_time_bins=4,
                                     picks=None, n_freqs_per_band=3):
    """Računa Morlet snagu po kanalu, pojasu i vremenskom segmentu.

    Vraća X oblika (n_epoha, n_kanala * n_pojaseva * n_time_bins) i imena
    značajki koja sadrže kanal, pojas i indeks vremenskog segmenta.
    """
    bands = bands or BANDS
    epochs = epochs.copy()
    if picks is not None:
        epochs.pick(picks)
    ch_names = epochs.ch_names

    freqs = np.concatenate([
        np.linspace(fmin, fmax, n_freqs_per_band) for fmin, fmax in bands.values()
    ])
    n_cycles = freqs / 2.0  # standardna heuristika: širi vremenski prozor za niže frekvencije

    tfr = mne.time_frequency.tfr_morlet(
        epochs, freqs=freqs, n_cycles=n_cycles, return_itc=False,
        average=False, verbose=False,
    )  # EpochsTFR, data: (n_epoha, n_kanala, n_freqs_total, n_times)

    data = tfr.data
    n_times = data.shape[-1]
    time_bins = np.array_split(np.arange(n_times), n_time_bins)

    feature_blocks = []
    feature_names = []
    freq_offset = 0
    for band_name in bands:
        band_idx = slice(freq_offset, freq_offset + n_freqs_per_band)
        freq_offset += n_freqs_per_band
        band_power = data[:, :, band_idx, :].mean(axis=2)  # (n_epoha, n_kanala, n_times)

        for bin_idx, idx in enumerate(time_bins):
            bin_power = band_power[:, :, idx].mean(axis=-1)  # (n_epoha, n_kanala)
            feature_blocks.append(bin_power)
            feature_names.extend([f'{ch}_{band_name}_t{bin_idx}' for ch in ch_names])

    X = np.concatenate(feature_blocks, axis=1)
    return X, feature_names


# Oblik sirovih epoha za EEGNet.

def prepare_raw_epoch_array(epochs, picks=None):
    """Sirove (filtrirane) epohe kao numpy niz oblika
    (n_epoha, 1, n_kanala, n_uzoraka) - format koji EEGNet/CNN implementacije
    u models.py očekuju (dodatna dimenzija = 'kanal' konvolucijske mreže,
    ne EEG kanal)."""
    data = epochs.get_data(picks=picks)  # (n_epoha, n_kanala, n_uzoraka)
    return data[:, np.newaxis, :, :]


# Definicije zadataka klasifikacije.

TASK_NAMES = ('left_right_fist', 'fists_feet', 'execution_vs_imagery', 'rest_vs_task')


def select_epochs_for_task(subject_runs, task_name, modality=None):
    """Spaja odgovarajuće epohe jednog ispitanika i vraća binarne labele.

    task_name može biti:
      'left_right_fist'      - lijeva (0) vs. desna (1) šaka; samo unilateralni runovi
      'fists_feet'           - obje šake (0) vs. obje noge (1); samo bilateralni runovi
      'execution_vs_imagery' - izvedba (0) vs. predodžba (1); T1/T2 evente iz oba tipa runa
      'rest_vs_task'         - rest/T0 (0) vs. bilo koji task/T1,T2 (1); svi zadatni runovi

    modality: None (zadano, koristi sve runove) | 'execution' | 'imagery' -
    opcionalni filter za odvajanje stvarne izvedbe i predodžbe.
    """
    if task_name not in TASK_NAMES:
        raise ValueError(f"Nepoznat task_name: {task_name}. Očekivano jedno od {TASK_NAMES}.")

    epoch_list = []
    label_list = []

    for run_id, data in subject_runs.items():
        if data['epochs'] is None:
            continue  # baseline runovi (R01/R02) nisu epohirani, preskoči

        if modality is not None and data['modality'] != modality:
            continue

        epochs = data['epochs']
        codes = epochs.events[:, 2]

        if task_name == 'left_right_fist':
            if data['run_type'] != 'unilateral':
                continue
            mask = np.isin(codes, [CODE_MAP['left_fist'], CODE_MAP['right_fist']])
        elif task_name == 'fists_feet':
            if data['run_type'] != 'bilateral':
                continue
            mask = np.isin(codes, [CODE_MAP['both_fists'], CODE_MAP['both_feet']])
        elif task_name == 'execution_vs_imagery':
            if data['run_type'] not in ('unilateral', 'bilateral'):
                continue
            mask = codes != CODE_MAP['rest']  # uspoređujemo samo stvarne task evente
        else:  # rest_vs_task
            if data['run_type'] not in ('unilateral', 'bilateral'):
                continue
            mask = np.ones_like(codes, dtype=bool)  # zadrži i rest i task epohe

        if not mask.any():
            continue

        selected = epochs[mask]
        selected_codes = selected.events[:, 2]

        if task_name == 'left_right_fist':
            labels = (selected_codes == CODE_MAP['right_fist']).astype(int)
        elif task_name == 'fists_feet':
            labels = (selected_codes == CODE_MAP['both_feet']).astype(int)
        elif task_name == 'execution_vs_imagery':
            labels = np.full(len(selected), 1 if data['modality'] == 'imagery' else 0)
        else:  # rest_vs_task
            labels = (selected_codes != CODE_MAP['rest']).astype(int)

        epoch_list.append(selected)
        label_list.append(labels)

    if not epoch_list:
        raise ValueError(
            f"Nema epoha za zadatak '{task_name}' (modality={modality}) - "
            f"provjeri jesu li runovi ispravno tipizirani u obrada.py."
        )

    all_epochs = mne.concatenate_epochs(epoch_list)
    y = np.concatenate(label_list)
    return all_epochs, y