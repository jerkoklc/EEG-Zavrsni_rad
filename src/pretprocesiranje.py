"""Konfigurabilni pipeline za pretprocesiranje PhysioNet EEG podataka.

Dokumentacija skupa podataka propisuje samo snimanje prema 10-10 sustavu
elektroda pri 160 Hz - ne propisuje referencu, pojas filtriranja niti metodu
uklanjanja artefakata. Svi parametri ispod su stoga metodološki izbori
(uobičajeni u MI/BCI literaturi), ne specifikacija dataseta, i moraju biti
obrazloženi u radu.

Koraci se mogu pojedinačno uključiti ili isključiti kroz PreprocessConfig,
što olakšava usporedbu različitih postavki.

Tijek:
    raw (iz loading.py)  --preprocess_raw-->  filtrirani/re-referencirani raw
                          --epohiranje (loading.py, NAKON preprocess_raw)-->
    epochs                --postprocess_epochs-->  epohe bez artefakata

preprocess_raw() se poziva prije epohiranja, a postprocess_epochs() nakon toga.
Normalizacija se fit-a samo na trening podacima.
"""

import os
from dataclasses import dataclass, field
from typing import List, Optional

import matplotlib
matplotlib.use('Agg')  # spremanje na disk bez potrebe za GUI okruženjem
import matplotlib.pyplot as plt
import mne
import numpy as np

# Nazivi kanala u EDF-u imaju dopunjene točke, pa ih prvo čistimo.
# Zapisi se prije ostale obrade usklađuju na 160 Hz.
TARGET_SFREQ = 160.0

MOTOR_CORTEX_CHANNELS = [
    'FC5', 'FC3', 'FC1', 'FCZ', 'FC2', 'FC4', 'FC6',
    'C5', 'C3', 'C1', 'CZ', 'C2', 'C4', 'C6',
    'CP5', 'CP3', 'CP1', 'CPZ', 'CP2', 'CP4', 'CP6',
]


@dataclass
class PreprocessConfig:
    """Postavke koje se koriste u obradi kontinuiranog zapisa i epoha.

    Zadane vrijednosti predstavljaju konfiguraciju korištenu u glavnim
    primjerima: CAR, pojas 8-30 Hz, svi kanali i z-score normalizacija.
    """
    # 1. re-referenciranje
    apply_car: bool = True

    # 2. filtriranje
    apply_bandpass: bool = True
    bandpass_low: float = 8.0    # Hz, donja granica (mu ritam počinje ~8Hz)
    bandpass_high: float = 30.0  # Hz, gornja granica (kraj beta ritma)
    apply_notch: bool = False
    notch_freqs: List[float] = field(default_factory=lambda: [50.0])

    # 3. uklanjanje artefakata
    artifact_method: Optional[str] = None  # None | 'ica' | 'threshold'
    ica_n_components: int = 20
    ica_random_state: int = 42
    amplitude_reject_uv: float = 150.0  # peak-to-peak prag u mikrovoltima

    # 4. odabir kanala
    channel_selection: str = 'all'  # 'all' | 'motor'

    # 5. normalizacija
    normalize: bool = True
    normalize_method: str = 'zscore'  # 'zscore' | 'minmax'


# Priprema kanala.

def clean_channel_names(raw):
    """Čisti nazive kanala i postavlja standardnu EEG montažu.

    Radi nad kopijom ulaza, pa pozivatelj zadržava originalni Raw objekt.
    """
    raw = raw.copy()
    raw.rename_channels(lambda ch: ch.strip('.').upper())
    montage = mne.channels.make_standard_montage('standard_1005')
    raw.set_montage(montage, on_missing='ignore')
    return raw


# Obrada kontinuiranog Raw zapisa, prije epohiranja.

def preprocess_raw(raw, config: PreprocessConfig, verbose=False):
    """Primjenjuje re-referenciranje, filtriranje, ICA i odabir kanala na
    kontinuirani (neepohirani) zapis. Vraća (raw, ica) - novi (kopirani) Raw
    objekt i fitanu ICA instancu (ili None ako ICA nije korištena/primijenjena).
    Originalni raw ostaje nepromijenjen. Resampliranje se radi prije filtera
    kako bi svi ispitanici kasnije imali isti broj uzoraka po epohi.

    NAPOMENA: ica se namjerno vraća kao zaseban objekt, a NE sprema u
    raw.info - MNE-ov Info objekt prihvaća samo unaprijed poznata polja pa
    spremanje proizvoljnog ključa (npr. 'temp_ica') tamo nije pouzdano."""
    raw = clean_channel_names(raw)
    ica = None

    # Neki zapisi su na 128 Hz. Bez resampliranja ne mogu se spojiti s
    # zapisima na 160 Hz jer epohe nemaju isti broj uzoraka.
    if abs(raw.info['sfreq'] - TARGET_SFREQ) > 1e-6:
        if verbose:
            print(f"Resampliram s {raw.info['sfreq']} Hz na {TARGET_SFREQ} Hz.")
        raw.resample(TARGET_SFREQ, verbose=verbose)

    # 1. common average reference
    if config.apply_car:
        raw.set_eeg_reference('average', projection=False, verbose=verbose)

    # 2. filtriranje
    if config.apply_bandpass:
        raw.filter(
            l_freq=config.bandpass_low, h_freq=config.bandpass_high,
            fir_design='firwin', verbose=verbose,
        )
    if config.apply_notch:
        raw.notch_filter(freqs=config.notch_freqs, verbose=verbose)

    # 3. ICA se samo fita; threshold odbacivanje radi se kasnije na epohama.
    if config.artifact_method == 'ica':
        ica = mne.preprocessing.ICA(
            n_components=config.ica_n_components,
            random_state=config.ica_random_state,
            max_iter='auto', verbose=verbose,
        )
        ica.fit(raw, verbose=verbose)
        # Dataset nema EOG kanale, zato se komponente za isključivanje
        # ne biraju automatski. ICA se vraća pozivatelju za pregled.

    # 4. odabir kanala
    if config.channel_selection == 'motor':
        available = [ch for ch in MOTOR_CORTEX_CHANNELS if ch in raw.ch_names]
        missing = set(MOTOR_CORTEX_CHANNELS) - set(available)
        if missing:
            print(f"Upozorenje: nedostaju motorički kanali u zapisu: {missing}")
        raw.pick(available)

    return raw, ica


# Obrada epoha nakon epohiranja.

def postprocess_epochs(epochs, config: PreprocessConfig):
    """Odbacuje epohe koje prelaze zadani amplitudni prag."""
    epochs = epochs.copy()

    if config.artifact_method == 'threshold':
        n_before = len(epochs)
        reject_criteria = dict(eeg=config.amplitude_reject_uv * 1e-6)  # V
        epochs.drop_bad(reject=reject_criteria, verbose=False)
        n_after = len(epochs)
        print(f"Amplitudno odbacivanje: {n_before - n_after}/{n_before} epoha odbačeno "
              f"(prag {config.amplitude_reject_uv} uV)")

    return epochs


def normalize_epochs(X_train, X_test=None, method='zscore'):
    """Normalizira po kanalima koristeći statistiku samo iz trening skupa.

    X_train, X_test: np.ndarray oblika (n_epoha, n_kanala, n_uzoraka).
    Statistika se računa preko epoha i vremena, odvojeno za svaki kanal.
    Vraća normalizirane X_train (i X_test ako je zadan) koristeći statistiku
    (mean/std ili min/max) izračunatu SAMO iz X_train.

    Eksplicitni train/test argumenti pomažu da se statistika slučajno ne
    izračuna iz test podataka.
    """
    if method == 'zscore':
        mean = X_train.mean(axis=(0, 2), keepdims=True)
        std = X_train.std(axis=(0, 2), keepdims=True) + 1e-12
        X_train_norm = (X_train - mean) / std
        X_test_norm = (X_test - mean) / std if X_test is not None else None
    elif method == 'minmax':
        ch_min = X_train.min(axis=(0, 2), keepdims=True)
        ch_max = X_train.max(axis=(0, 2), keepdims=True)
        denom = (ch_max - ch_min) + 1e-12
        X_train_norm = (X_train - ch_min) / denom
        X_test_norm = (X_test - ch_min) / denom if X_test is not None else None
    else:
        raise ValueError(f"Nepoznata metoda normalizacije: {method}")

    return X_train_norm, X_test_norm


# Pomoćna funkcija za brzu provjeru jednog zapisa.

def run_preprocessing_pipeline(raw, epochs, config: PreprocessConfig):
    """Prikladno za brzo isprobavanje jedne konfiguracije: primjenjuje
    preprocess_raw pa postprocess_epochs. Za stvarni eksperiment epohe iz
    loading.py treba napraviti NAKON preprocess_raw (ne prije), jer se ovdje
    epochs objekt već epohirao od originalnog raw-a pa filtriranje na raw-u
    poslije neće djelovati na već postojeće epohe - koristi ovu funkciju
    samo za vizualnu/dijagnostičku provjeru jednog zapisa."""
    raw_processed, ica = preprocess_raw(raw, config)
    epochs_processed = postprocess_epochs(epochs, config)
    return raw_processed, epochs_processed, ica


# Vizualne provjere.

def plot_psd_comparison(raw_before, raw_after, out_path='psd_comparison.png',
                         fmax=60.0):
    """Sprema PSD graf prije/poslije filtriranja radi vizualne provjere."""
    fig, axes = plt.subplots(1, 2, figsize=(12, 4), sharey=True)

    raw_before.compute_psd(fmax=fmax, verbose=False).plot(
        axes=axes[0], show=False, average=True)
    axes[0].set_title('Prije pretprocesiranja')

    raw_after.compute_psd(fmax=fmax, verbose=False).plot(
        axes=axes[1], show=False, average=True)
    axes[1].set_title('Poslije pretprocesiranja')

    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"PSD usporedba spremljena u {out_path}")


def plot_example_epoch(epochs, index=0, channels=('C3', 'CZ', 'C4'),
                        out_path='example_epoch.png'):
    """Sprema graf jedne epohe za odabrane kanale radi vizualne provjere."""
    epochs = epochs.copy()
    epochs.rename_channels(lambda ch: ch.strip('.').upper())

    available = [ch for ch in channels if ch in epochs.ch_names]
    if not available:
        print(f"Nijedan od traženih kanala {channels} nije prisutan; "
              f"dostupni kanali: {epochs.ch_names[:10]}...")
        return

    data = epochs.get_data(picks=available)[index]  # (n_kanala, n_uzoraka)
    times = epochs.times

    fig, ax = plt.subplots(figsize=(10, 4))
    for ch_name, ch_data in zip(available, data):
        ax.plot(times, ch_data * 1e6, label=ch_name)  # V -> uV

    label = epochs.events[index, 2]
    ax.set_title(f'Primjer epohe #{index} (event kod={label})')
    ax.set_xlabel('Vrijeme (s)')
    ax.set_ylabel('Amplituda (uV)')
    ax.legend()
    fig.tight_layout()
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Primjer epohe spremljen u {out_path}")


if __name__ == '__main__':
    # Primjer korištenja - poziva obrada.py, ali BEZ preprocess_config (da
    # dobijemo sirov raw za usporedbu), pa preprocesiranje radimo ovdje ručno
    # radi PSD grafa "prije/poslije".
    from obrada import loading_files

    subjects = loading_files(preprocess_config=None)
    subject_data = subjects.get(1)
    if subject_data is None:
        raise SystemExit("Ispitanik S001 nije pronađen - provjeri putanju do 'files/'.")

    # uzmi jedan unilateralni run kao primjer
    unilateral_run = next(
        (d for d in subject_data.values() if d['run_type'] == 'unilateral'), None
    )
    if unilateral_run is None:
        raise SystemExit("Nije pronađen unilateralni run za S001.")

    config_all = PreprocessConfig(channel_selection='all')

    raw_before = unilateral_run['raw']
    raw_after, ica = preprocess_raw(raw_before, config_all)

    plot_psd_comparison(raw_before, raw_after)

    epochs_after = postprocess_epochs(unilateral_run['epochs'], config_all)
    plot_example_epoch(epochs_after, index=0)

    print("\nPretprocesiranje dovršeno za primjer runa ispitanika S001.")