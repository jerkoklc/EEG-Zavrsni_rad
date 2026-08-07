import os
import re
import mne 
import numpy as np
from pretprocesiranje import PreprocessConfig, preprocess_raw, postprocess_epochs

path = os.path.join(os.getcwd(), 'files')


BASELINE_OPEN_RUNS = {1}
BASELINE_CLOSED_RUNS = {2}
UNILATERAL_RUNS = {3, 4, 7, 8, 11, 12}
BILATERAL_RUNS = {5, 6, 9, 10, 13, 14}



#zbog PhysioNet dokumentacje, protokol se ponavlja 3 puta u ciklusu od 4 runa
# Task1 (real unilateral) -> Task2 (imagined unilateral) -> Task3 (real bilateral)  -> Task4 (imagined bilateral), tri puta zaredom.
EXECUTION_RUNS = {3, 4, 7, 8, 11, 12}
IMAGERY_RUNS = {5, 6, 9, 10, 13, 14}

EPOCH_TMIN = 0.0
EPOCH_TMAX = 4.0 #pretpostavka svaki zadatak u skupu traje 4 sekunde

LABEL_MAP_UNILATERAL = {'T0': 'rest', 'T1': 'left_fist', 'T2': 'right_fist'}
LABEL_MAP_BILATERAL = {'T0': 'rest', 'T1': 'both_fists', 'T2': 'both_feet'}


# fiksni event_id -> ISTI numerički kod za isti event kroz sve runove/ispitanike
# i zato kad se koristi mne.concatenate_epochs, epohe se mogu spojiti jer svi T1/T2 imaju isti numerički kod inače abecedno events_from_annotations bi dodijelio različite numeričke kodove za T1/T2 ovisno o abecednom redoslijedu opisa u anotacijama
CODE_MAP = {'rest': 0, 'left_fist': 1, 'right_fist': 2, 'both_fists': 3, 'both_feet': 4}


def parse_run_info(filename): #pretvara na način da npr. S001R02 -> subject_id=1, run_id=2  
    match = re.match(r'S(\d{3})R(\d{2})\.edf$', filename)
    if not match:
        return None, None
    return int(match.group(1)), int(match.group(2))


def get_run_type(run_id):
    
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
    #'execution' | 'imagery' | None (baseline runovi nemaju modalitet)
    if run_id in EXECUTION_RUNS:
        return 'execution'
    if run_id in IMAGERY_RUNS:
        return 'imagery'
    return None



def remap_annotations(annotations, run_type):  #preimenuje T0/T1/T2 u opisne nazive ovisno o tipu runa
        if annotations is None or len(annotations) == 0:
            return annotations
        
        if run_type == 'unilateral':
            label_map = LABEL_MAP_UNILATERAL
        elif run_type == 'bilateral':
           label_map = LABEL_MAP_BILATERAL
        else:
        # baseline runovi obično nemaju T1/T2, samo eventualno T0 preko cijelog runa
            label_map = {'T0': 'rest'}
        
        mapped_descriptions = [label_map.get(desc, desc) for desc in annotations.description]
        
        return mne.Annotations(
            onset=annotations.onset,
            duration=annotations.duration,
            description=mapped_descriptions,
            orig_time=annotations.orig_time,
        )
   
def make_epochs(raw, run_type): #baseline se koriste kao kontinuirani zapis, a T1 i T2 evente epohira za zadatne runove
    if run_type not in ('unilateral', 'bilateral'):
        return None, None
    
    # Use the same global event_id mapping for all runs so epochs can be concatenated
    events, event_id_map = mne.events_from_annotations(raw, event_id=CODE_MAP, verbose=False)
    if len(events) == 0:
        return None, None
    
    epochs = mne.Epochs(
        raw, events, event_id=event_id_map,
        tmin=EPOCH_TMIN, tmax=EPOCH_TMAX,
        baseline=None, preload=True, verbose=False
    )
    return epochs, event_id_map


def loading_files(preprocess_config: PreprocessConfig = None):
    """Učitava sve EDF+ zapise, mapira anotacije, PRETPROCESIRA (ako je
    preprocess_config zadan) i epohira zadatne runove. Vraća organiziranu
    strukturu po ispitaniku.
 
    preprocess_config=None -> preprocesiranje se preskače (sirovi raw ide
    direktno u epohiranje) - korisno za brzu provjeru učitavanja/anotacija
    bez čekanja na filtriranje/ICA. Za stvarne eksperimente uvijek proslijedi
    konkretan PreprocessConfig.
    """
    subjects = {}
 
    for root, _, files in os.walk(path):
        for file in sorted(files):
            if not file.endswith('.edf'):
                continue
 
            subject_id, run_id = parse_run_info(file)
            if subject_id is None:
                print(f"Preskačem neprepoznati naziv datoteke: {file}")
                continue
 
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
                    'raw': raw,
                    'modality': get_modality(run_id),
                    'epochs': epochs,
                    'event_id_map': event_id_map,
                    'ica': ica,
                }
            except Exception as e:
                print(f"Greška pri učitavanju {edf_path}: {e}")
 
    return subjects


def save_subject_epochs(subjects, out_dir='processed'):
    #sprema epohe unilateral i bilateral runova po ispitaniku u .fif a baseline runove kao sirove .fif zapise
    
    os.makedirs(out_dir, exist_ok=True)
    
    for subject_id, runs in subjects.items():
        subj_dir = os.path.join(out_dir, f'S{subject_id:03d}')
        os.makedirs(subj_dir, exist_ok=True)
        
        task_epochs_list = []
        for run_id, data in runs.items():
            if data['run_type'] in ('baseline_open', 'baseline_closed'):
                data['raw'].save(
                    os.path.join(subj_dir, f'{data["run_type"]}_raw.fif'), 
                    overwrite=True
                )
            elif data['epochs'] is not None and len(data['epochs']) > 0:
                task_epochs_list.append(data['epochs'])
                
        if task_epochs_list:
            #valid_epochs = [epochs for epochs in task_epochs_list if len(epochs) > 0]
            #if not valid_epochs:
                #print(f"Nema preostalih važećih epoha za ispitanika S{subject_id:03d} nakon postprocesiranja.")
                #continue
            all_epochs = mne.concatenate_epochs(task_epochs_list)
            all_epochs.save(
                os.path.join(subj_dir, 'task_epochs-epo.fif'),
                overwrite=True
            )
            
"""služi za brzu provjeru učitanih podataka i raspodjele labela po epohama za jednog ispitanika 
    tako što se ne mora čekati cijeli proces preprocesiranja i epohiranja svih ispitanika, 
    a može se koristiti i za provjeru da li su svi runovi učitani, broj kanala/ferkvenciju uzorkovanja i da li su anotacije ispravno mapirane 

"""
def verify_single_subject(subjects, subject_id=1):

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
        print(f"  Trajanje: {raw.times[-1]:.1f} s, fs: {raw.info['sfreq']} Hz, "
              f"broj kanala: {len(raw.ch_names)}")
        print(f"  Prvih 5 kanala: {raw.ch_names[:5]}")
 
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
    #save_subject_epochs(subjects)
    print("\nGotovo. Pretprocesirane epohe i baseline zapisi spremljeni u ./processed/")
