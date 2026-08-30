"""Učitavanje, mapiranje anotacija, pretprocesiranje i epohiranje PhysioNet EEG-a.

Redoslijed po runu:
    1. čitanje EDF+ (raw)
    2. remap_annotations       - T0/T1/T2 -> semantičke labele, ovisno o run_type
    3. preprocess_raw          - CAR, filter, ICA i odabir kanala
    4. epohiranje               - nakon obrade kontinuiranog zapisa
    5. postprocess_epochs      - odbacivanje epoha s velikom amplitudom

Pretprocesiranje se radi prije epohiranja jer
CAR i filtriranje rade ispravnije na kontinuiranom zapisu nego na već izrezanim
epohama (rubni efekti filtra na kratkim segmentima).

Struktura runova po ispitaniku (S001..S109), 14 runova svaki:
    R01           -> baseline, eyes open   (1 min, bez T1/T2 eventa)
    R02           -> baseline, eyes closed (1 min, bez T1/T2 eventa)
    R03,04,07,08,11,12 -> unilateralni zadaci (šaka): T1=lijeva šaka, T2=desna šaka
    R05,06,09,10,13,14 -> bilateralni zadaci (šaka/noga): T1=obje šake, T2=obje noge
"""

import os
import re
import mne
import numpy as np
import pandas as pd

from pretprocesiranje import PreprocessConfig, preprocess_raw, postprocess_epochs

path = os.path.join(os.getcwd(), 'files')

# Vrste runova iz protokola dataseta.

BASELINE_OPEN_RUNS = {1}
BASELINE_CLOSED_RUNS = {2}
UNILATERAL_RUNS = {3, 4, 7, 8, 11, 12}
BILATERAL_RUNS = {5, 6, 9, 10, 13, 14}

# Isti raspored runova razlikuje stvarnu izvedbu od zamišljenog pokreta.
EXECUTION_RUNS = {3, 5, 7, 9, 11, 13}
IMAGERY_RUNS = {4, 6, 8, 10, 12, 14}

EPOCH_TMIN = 0.0
# Svaka epoha obuhvaća prve četiri sekunde nakon događaja.
EPOCH_TMAX = 4.0

# T1 i T2 ne znače isto u unilateralnim i bilateralnim runovima.

LABEL_MAP_UNILATERAL = {'T0': 'rest', 'T1': 'left_fist', 'T2': 'right_fist'}
LABEL_MAP_BILATERAL = {'T0': 'rest', 'T1': 'both_fists', 'T2': 'both_feet'}

# Kodovi moraju ostati isti kroz sve zapise da bi se epohe mogle spojiti.
CODE_MAP = {'rest': 0, 'left_fist': 1, 'right_fist': 2, 'both_fists': 3, 'both_feet': 4}


def parse_run_info(filename):
    """Iz imena EDF datoteke vraća broj ispitanika i broj runa."""
    match = re.match(r'S(\d{3})R(\d{2})\.edf$', filename)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def get_run_type(run_id):
    """Vraća grupu runa kojoj pripada zadani broj."""
    if run_id in BASELINE_OPEN_RUNS:
        return 'baseline_open'
    if run_id in BASELINE_CLOSED_RUNS:
        return 'baseline_closed'
    if run_id in UNILATERAL_RUNS:
        return 'unilateral'
    if run_id in BILATERAL_RUNS:
        return 'bilateral'
    return 'unknown'


def get_modality(run_id):
    """Vraća ``execution``, ``imagery`` ili ``None`` za baseline runove."""
    if run_id in EXECUTION_RUNS:
        return 'execution'
    if run_id in IMAGERY_RUNS:
        return 'imagery'
    return None


def remap_annotations(annotations, run_type):
    """Pretvara T0/T1/T2 u labele koje ovise o vrsti runa.

    Baseline zapisi koriste samo ``rest``. Ako nema anotacija, funkcija vraća
    originalni objekt kako bi se izbjegla nepotrebna izrada praznih anotacija.
    """
    if annotations is None or len(annotations) == 0:
        return annotations

    if run_type == 'unilateral':
        label_map = LABEL_MAP_UNILATERAL
    elif run_type == 'bilateral':
        label_map = LABEL_MAP_BILATERAL
    else:
        label_map = {'T0': 'rest'}

    mapped_descriptions = [label_map.get(desc, desc) for desc in annotations.description]
    return mne.Annotations(
        onset=annotations.onset,
        duration=annotations.duration,
        description=mapped_descriptions,
        orig_time=annotations.orig_time,
    )


def make_epochs(raw, run_type):
    """Stvara epohe za task runove; baseline runovi vraćaju ``(None, None)``."""
    if run_type not in ('unilateral', 'bilateral'):
        return None, None

    events, event_id_map = mne.events_from_annotations(raw, event_id=CODE_MAP, verbose=False)
    if len(events) == 0:
        return None, None

    epochs = mne.Epochs(
        raw, events, event_id=event_id_map,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False,
    )
    return epochs, event_id_map


def loading_files(preprocess_config: PreprocessConfig = None, use_processed_cache=True,
                   processed_dir='processed'):
    """Učitava EDF zapise, po potrebi ih obrađuje i slaže podatke po ispitaniku.

    Ako je ``preprocess_config=None``, obrada se preskače. To je korisno za
    brzu provjeru učitavanja i anotacija bez čekanja na filtriranje ili ICA.

    Uz zadani cache prvo se provjerava postoji li obrađena verzija u
    ``processed_dir``
    (spremljena ranijim pozivom save_subject_epochs). Ako postoji, učitava
    se izravno (mne.read_epochs, bez ponovnog CAR/filter/ICA/epohiranja) -
    to je bio glavni razlog sporosti pri svakom novom pokretanju evaluacije,
    jer se cijeli EDF->epohe pipeline prije ponavljao od nule za svih 109
    ispitanika čak i kad se ništa u pretprocesiranju nije promijenilo.
    Ispitanici KOJIH NEMA u cacheu se obrađuju klasičnim putem i automatski
    spremaju u cache za sljedeći put. Postavi na False ako namjerno mijenjaš
    preprocess_config i želiš da se SVI ispitanici obrade iznova (cache ne
    zna razlikovati različite konfiguracije - vidi napomenu ispod)."""
    subjects = {}
    cached_subject_ids = set()
    subject_ids_to_process = None

    if use_processed_cache:
        if not _cache_config_matches(preprocess_config, processed_dir):
            print(f"UPOZORENJE: konfiguracija pretprocesiranja se razlikuje od one "
                  f"kojom je {processed_dir}/ prethodno napunjen (ili cache još ne "
                  f"postoji) - ignoriram cache, obrađujem sve ispitanike iznova i "
                  f"prepisujem cache novom konfiguracijom.")
        else:
            cached_subjects, missing = load_processed_subjects(processed_dir=processed_dir)
            subjects.update(cached_subjects)
            cached_subject_ids = set(cached_subjects)
            if cached_subjects:
                print(f"Učitano {len(cached_subjects)} ispitanika iz processed cache-a "
                      f"({processed_dir}/) - preskočeno ponovno pretprocesiranje.")
            subject_ids_to_process = set(missing) if missing or cached_subjects else None
                    # Prazan cache znači da treba proći sve pronađene zapise.

    for root, _, files in os.walk(path):
        for file in sorted(files):
            if not file.endswith('.edf'):
                continue

            subject_id, run_id = parse_run_info(file)
            if subject_id is None:
                print(f"Preskačem neprepoznati naziv datoteke: {file}")
                continue

            if subject_id in cached_subject_ids:
                continue  # već je učitan iz cachea

            if subject_ids_to_process is not None and subject_id not in subject_ids_to_process:
                continue  # obrađuju se samo ispitanici koji nedostaju u cacheu

            run_type = get_run_type(run_id)
            edf_path = os.path.join(root, file)

            try:
                raw = mne.io.read_raw_edf(edf_path, preload=True, encoding='latin1')
                raw.set_annotations(remap_annotations(raw.annotations, run_type))

                ica = None
                if preprocess_config is not None:
                    raw, ica = preprocess_raw(raw, preprocess_config)

                epochs, event_id_map = make_epochs(raw, run_type)

                if epochs is not None and preprocess_config is not None:
                    epochs = postprocess_epochs(epochs, preprocess_config)

                subjects.setdefault(subject_id, {})[run_id] = {
                    'run_type': run_type,
                    'modality': get_modality(run_id),
                    'raw': raw,
                    'epochs': epochs,
                    'event_id_map': event_id_map,
                    'ica': ica,
                }
            except Exception as e:
                print(f"Greška pri učitavanju {edf_path}: {e}")

    if use_processed_cache:
        newly_processed = {sid: subjects[sid] for sid in subjects
                            if subject_ids_to_process is None or sid in subject_ids_to_process}
        if newly_processed:
            save_subject_epochs(newly_processed, out_dir=processed_dir)
            _save_cache_config(preprocess_config, processed_dir)
            print(f"Spremljeno {len(newly_processed)} novoobrađenih ispitanika u {processed_dir}/ "
                  f"za buduća pokretanja.")

    return subjects


# Promijeni verziju kad se promijeni logika obrade koja utječe na cache.
# Verzija 2 uključuje usklađivanje zapisa na 160 Hz.
PIPELINE_VERSION = 2


def _cache_config_matches(preprocess_config, processed_dir):
    """Provjerava odgovaraju li konfiguracija i verzija pipelinea cacheu."""
    import json
    from dataclasses import asdict

    config_path = os.path.join(processed_dir, '_config.json')
    current = {
        'preprocess_config': asdict(preprocess_config) if preprocess_config is not None else None,
        'pipeline_version': PIPELINE_VERSION,
    }

    if not os.path.exists(config_path):
        # Cache još ne postoji - "podudara se" u smislu da nema ničeg za
        # sukobiti; loading_files će ionako obraditi sve od nule.
        return True

    with open(config_path) as fh:
        saved = json.load(fh)

    return saved == current


def _save_cache_config(preprocess_config, processed_dir):
    import json
    from dataclasses import asdict

    os.makedirs(processed_dir, exist_ok=True)
    current = {
        'preprocess_config': asdict(preprocess_config) if preprocess_config is not None else None,
        'pipeline_version': PIPELINE_VERSION,
    }
    with open(os.path.join(processed_dir, '_config.json'), 'w') as fh:
        json.dump(current, fh, indent=2)


def save_subject_epochs(subjects, out_dir='processed'):
    """Sprema task epohe i baseline Raw zapise u FIF datoteke.

    VAŽNO: prije spajanja (concatenate_epochs) svaka run-ova epoha dobiva
    metapodatke (run_id, run_type, modality) kroz epochs.metadata - MNE ih
    sprema/učitava zajedno s .fif datotekom. Bez ovoga bi se nakon spajanja
    izgubila informacija o tome koji dio epoha dolazi iz kojeg tipa runa, a
    load_processed_subjects() je treba da rekonstruira strukturu identičnu
    onoj koju vraća loading_files() (po run_id, ne jedna spojena hrpa)."""
    os.makedirs(out_dir, exist_ok=True)

    for subject_id, runs in subjects.items():
        subj_dir = os.path.join(out_dir, f'S{subject_id:03d}')
        os.makedirs(subj_dir, exist_ok=True)

        task_epochs_list = []
        for run_id, data in runs.items():
            if data['run_type'] in ('baseline_open', 'baseline_closed'):
                data['raw'].save(
                    os.path.join(subj_dir, f'{data["run_type"]}_raw.fif'),
                    overwrite=True,
                )
            elif data['epochs'] is not None and len(data['epochs']) > 0:
                epochs = data['epochs'].copy()
                epochs.metadata = pd.DataFrame({
                    'run_id': run_id,
                    'run_type': data['run_type'],
                    'modality': data['modality'],
                }, index=range(len(epochs)))
                task_epochs_list.append(epochs)

        if task_epochs_list:
            all_epochs = mne.concatenate_epochs(task_epochs_list)
            all_epochs.save(
                os.path.join(subj_dir, 'task_epochs-epo.fif'),
                overwrite=True,
            )


def load_processed_subjects(subject_ids=None, processed_dir='processed'):
    """Učitava obrađene epohe iz processed/ i obnavlja strukturu po runovima.
    pozivom save_subject_epochs) i rekonstruira strukturu IDENTIČNU onoj iz
    loading_files() - dict subject_id -> {run_id: {'run_type', 'modality',
    'epochs', 'raw': None}} - koristeći epochs.metadata da razdvoji spojene
    epohe natrag po run_id.

    Vraća (subjects, missing_subject_ids) - missing_subject_ids su oni za
    koje processed/ ne sadrži datoteku (treba ih obraditi klasičnim putem
    preko loading_files() i spremiti pomoću save_subject_epochs()).

    NAPOMENA: 'raw' je None za sve runove pri učitavanju iz cachea (sirovi
    kontinuirani zapis se ne sprema za task runove) - to je dovoljno za
    značajke.select_epochs_for_task() koja koristi samo 'epochs'/'run_type'/
    'modality', ali NIJE dovoljno za dijagnostiku u verify_single_subject()
    koja ispisuje raw.info (fs, broj kanala) - ta dijagnostika i dalje
    zahtijeva klasično učitavanje preko loading_files()."""
    if subject_ids is None:
        if not os.path.isdir(processed_dir):
            return {}, []
        subject_ids = sorted(
            int(d[1:]) for d in os.listdir(processed_dir)
            if d.startswith('S') and d[1:].isdigit()
        )

    subjects = {}
    missing = []

    for subject_id in subject_ids:
        subj_dir = os.path.join(processed_dir, f'S{subject_id:03d}')
        task_path = os.path.join(subj_dir, 'task_epochs-epo.fif')

        if not os.path.exists(task_path):
            missing.append(subject_id)
            continue

        all_epochs = mne.read_epochs(task_path, preload=True, verbose=False)
        if all_epochs.metadata is None:
            print(f"[S{subject_id:03d}] processed datoteka nema metapodatke "
                  f"(spremljena prije ove izmjene?) - obrađujem iznova.")
            missing.append(subject_id)
            continue

        runs = {}
        for run_id in sorted(all_epochs.metadata['run_id'].unique()):
            mask = (all_epochs.metadata['run_id'] == run_id).to_numpy()
            sub_epochs = all_epochs[mask]
            row = all_epochs.metadata[mask].iloc[0]
            runs[int(run_id)] = {
                'run_type': row['run_type'],
                'modality': row['modality'] if pd.notna(row['modality']) else None,
                'raw': None,
                'epochs': sub_epochs,
                'event_id_map': sub_epochs.event_id,
                'ica': None,
            }

        # baseline runovi (ako su spremljeni) - učitaj kao raw, bez epoha
        for run_id, run_type in ((1, 'baseline_open'), (2, 'baseline_closed')):
            raw_path = os.path.join(subj_dir, f'{run_type}_raw.fif')
            if os.path.exists(raw_path):
                runs[run_id] = {
                    'run_type': run_type,
                    'modality': None,
                    'raw': mne.io.read_raw_fif(raw_path, preload=False, verbose=False),
                    'epochs': None,
                    'event_id_map': None,
                    'ica': None,
                }

        subjects[subject_id] = runs

    return subjects, missing


# --- provjera za JEDNOG ispitanika --------------------------------------

def verify_single_subject(subjects, subject_id=1):
    """Ispisuje dijagnostiku za jednog ispitanika: broj runova, tip svakog runa,
    broj kanala/frekvenciju uzorkovanja, broj i raspodjelu labela po epohama."""
    if subject_id not in subjects:
        print(f"Ispitanik S{subject_id:03d} nije pronađen u učitanim podacima.")
        return

    print(f"\n=== Provjera za ispitanika S{subject_id:03d} ===")
    runs = subjects[subject_id]
    print(f"Broj pronađenih runova: {len(runs)} (očekivano: 14)")

    for run_id in sorted(runs):
        data = runs[run_id]
        raw = data['raw']
        print(f"\n--- Run {run_id:02d} ({data['run_type']}, modality={data['modality']}) ---")
        if raw is not None:
            print(f"  Trajanje: {raw.times[-1]:.1f} s, fs: {raw.info['sfreq']} Hz, "
                  f"broj kanala: {len(raw.ch_names)}")
            print(f"  Prvih 5 kanala: {raw.ch_names[:5]}")
        elif data['epochs'] is not None:
            print(f"  (raw nije dostupan - podaci učitani iz processed cachea; "
                  f"fs: {data['epochs'].info['sfreq']} Hz, broj kanala: {len(data['epochs'].ch_names)})")
        else:
            print("  (raw nije dostupan - podaci učitani iz processed cachea, bez baseline raw datoteke)")

        if data['epochs'] is not None:
            epochs = data['epochs']
            print(f"  Broj epoha: {len(epochs)}")
            counts = {
                label: int(np.sum(epochs.events[:, 2] == code))
                for label, code in data['event_id_map'].items()
            }
            print(f"  Raspodjela labela: {counts}")
        else:
            print("  (baseline run - nije epohiran, koristi se kao kontinuirani zapis)")

    type_counts = {}
    for data in runs.values():
        type_counts[data['run_type']] = type_counts.get(data['run_type'], 0) + 1
    print(f"\nBroj runova po tipu: {type_counts}")
    print("Očekivano: {'baseline_open': 1, 'baseline_closed': 1, 'unilateral': 6, 'bilateral': 6}")


if __name__ == '__main__':
    # Konfiguracija koja se koristi za sve eksperimente u ovom pokretanju.
    # Mijenjaj ovdje (ili instanciraj drugi PreprocessConfig) za analizu
    # utjecaja pojedinih koraka - to je i svrha zasebnih on/off zastavica.
    config = PreprocessConfig(
        apply_car=True,
        apply_bandpass=True,
        bandpass_low=8.0,
        bandpass_high=30.0,
        apply_notch=False,
        artifact_method='threshold',
        amplitude_reject_uv=150.0,
        channel_selection='all',
    )

    subjects = loading_files(preprocess_config=config)
    verify_single_subject(subjects, subject_id=1)
    save_subject_epochs(subjects)
    print("\nGotovo. Pretprocesirane epohe i baseline zapisi spremljeni u ./processed/")