"""Evaluacija po ispitaniku, između ispitanika i kroz experiment grid.

Kod cross-subject evaluacije cijeli ispitanik ostaje u jednom foldu. Značajke
se računaju jednom po kombinaciji i zatim dijele između klasifikatora, jer je
wavelet ekstrakcija dosta sporija od samog fitanja modela.
"""

import os
import itertools
import hashlib
import pickle

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from sklearn.model_selection import (
    StratifiedKFold, GroupKFold, LeaveOneGroupOut, cross_validate, cross_val_predict,
)
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, roc_auc_score

from pretprocesiranje import MOTOR_CORTEX_CHANNELS
from značajke import extract_band_power_features, extract_time_frequency_features, select_epochs_for_task
from modeli import build_feature_based_pipelines, build_csp_lda_pipeline, build_eegnet_pipeline, TORCH_AVAILABLE
from značajke import prepare_raw_epoch_array


FEATURE_EXTRACTORS = {
    'band_power': extract_band_power_features,
    'time_frequency': extract_time_frequency_features,
}

CACHE_DIR = 'cache'


def _picks_for(channel_selection):
    return MOTOR_CORTEX_CHANNELS if channel_selection == 'motor' else None


def _get_classifier_pipeline(classifier_name, random_state=42):
    """Vraća pipeline za traženi naziv klasifikatora.
    | 'csp_lda' | 'eegnet'. csp_lda i eegnet rade nad sirovim epohama
    (različitog oblika - vidi build_features_cache), ostali nad značajkama."""
    if classifier_name == 'csp_lda':
        return build_csp_lda_pipeline()
    if classifier_name == 'eegnet':
        return build_eegnet_pipeline(random_state=random_state)
    pipelines = build_feature_based_pipelines(random_state=random_state)
    if classifier_name not in pipelines:
        raise ValueError(f"Nepoznat classifier_name: {classifier_name}")
    return pipelines[classifier_name]


# Keširanje značajki.

# Povećaj verziju ako promjena u ekstrakciji čini postojeći cache nevažećim.
# Verzija 3 je uvedena nakon promjene učitavanja runova po ispitaniku.
CACHE_VERSION = 3


def _disk_cache_key(task_name, feature_type, channel_selection, modality, subject_ids,
                     min_epochs_per_subject=30):
    raw = (f"v{CACHE_VERSION}|{task_name}|{feature_type}|{channel_selection}|{modality}|"
           f"{min_epochs_per_subject}|{sorted(subject_ids)}")
    return hashlib.md5(raw.encode()).hexdigest()


def _load_disk_cache(key):
    path = os.path.join(CACHE_DIR, f'{key}.pkl')
    if not os.path.exists(path):
        return None

    try:
        with open(path, 'rb') as fh:
            cached = pickle.load(fh)
    except (EOFError, OSError, pickle.UnpicklingError, AttributeError, ValueError) as exc:
        print(f"  [cache invalid] {path}: {type(exc).__name__}; rebuilding")
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    if not cached:
        print(f"  [cache empty] {path}; rebuilding")
        try:
            os.remove(path)
        except OSError:
            pass
        return None

    return cached


def _save_disk_cache(key, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    path = os.path.join(CACHE_DIR, f'{key}.pkl')
    temp_path = f'{path}.tmp'
    with open(temp_path, 'wb') as fh:
        pickle.dump(obj, fh)
        fh.flush()
        os.fsync(fh.fileno())
    os.replace(temp_path, path)


def build_features_cache(all_subjects, task_name, feature_type, channel_selection,
                          modality=None, use_disk_cache=True, verbose=True,
                          min_epochs_per_subject=30):
    """Računa X i y po ispitaniku za zadanu (task, feature_type,
    channel_selection) kombinaciju. Vraća dict subject_id -> (X, y).

    feature_type može biti ``band_power``, ``time_frequency``, ``raw`` ili
    ``eegnet_raw``. Sirovi oblici se koriste za CSP+LDA odnosno EEGNet.

    min_epochs_per_subject: ispitanici s MANJE epoha se PRESKAČU (ne ulaze u
    cache). Razlog: 5-fold CV na npr. n=11 epoha daje test foldove od svega
    2-3 uzorka - metrika je tada praktički šum (vidi primjere u logu: acc=0.100
    na n=11, auc=nan na n=12/26 jer fold sadrži samo jednu klasu). Ti isti
    ispitanici, ako uđu u cross-subject pool, povećavaju šansu da neki
    GroupKFold fold dobije skoro jednorodnu raspodjelu klasa - što je
    najvjerojatniji uzrok SVC.fit() pada s "number of classes has to be
    greater than one" ili sličnom greškom. Zadano 30 je gruba procjena
    (barem ~15 po klasi za binarni zadatak); prilagodi po potrebi, ali
    obrazloži izbor u radu (metodološka odluka, ne proizvoljna)."""
    subject_ids = list(all_subjects.keys())
    key = _disk_cache_key(task_name, feature_type, channel_selection, modality, subject_ids,
                           min_epochs_per_subject)

    if use_disk_cache:
        cached = _load_disk_cache(key)
        if cached is not None:
            if verbose:
                print(f"  [cache hit] {task_name}/{feature_type}/{channel_selection}")
            return cached

    picks = _picks_for(channel_selection)
    cache = {}
    n_excluded_small = 0
    n_excluded_nan = 0

    for subject_id, subject_runs in all_subjects.items():
        try:
            epochs, y = select_epochs_for_task(subject_runs, task_name, modality=modality)
        except ValueError:
            continue
        if len(np.unique(y)) < 2:
            continue
        if len(y) < min_epochs_per_subject:
            n_excluded_small += 1
            continue

        if feature_type == 'raw':
            epochs_picked = epochs.copy().pick(picks) if picks else epochs
            X = epochs_picked.get_data()
        elif feature_type == 'eegnet_raw':
            # 4D (n_epoha, 1, n_kanala, n_uzoraka) - EEGNet očekuje dodatnu
            # "kanal" dimenziju za konvoluciju, za razliku od CSP-ovog 3D ulaza.
            X = prepare_raw_epoch_array(epochs, picks=picks)
        else:
            extractor = FEATURE_EXTRACTORS[feature_type]
            X, _ = extractor(epochs, picks=picks)

        # Izbacujemo samo epohe s NaN/Inf vrijednostima.
        finite_mask = np.isfinite(X.reshape(len(X), -1)).all(axis=1)
        if not finite_mask.all():
            n_bad = (~finite_mask).sum()
            n_excluded_nan += n_bad
            X, y = X[finite_mask], y[finite_mask]
            if len(np.unique(y)) < 2 or len(y) < min_epochs_per_subject:
                continue  # nakon čišćenja NaN-ova više ne zadovoljava uvjete

        cache[subject_id] = (X, y)

    if verbose:
        if n_excluded_small:
            print(f"  Isključeno {n_excluded_small} ispitanika s manje od "
                  f"{min_epochs_per_subject} epoha.")
        if n_excluded_nan:
            print(f"  Uklonjeno {n_excluded_nan} epoha s NaN/Inf vrijednostima "
                  f"(zadržani ispitanici i dalje u cache-u ako imaju dovoljno "
                  f"preostalih epoha).")

    if use_disk_cache:
        _save_disk_cache(key, cache)

    return cache


# Per-subject scenarij.

def evaluate_per_subject(features_cache, classifier_name, cv_splits=5,
                          random_state=42, verbose=True, label=''):
    """Za svakog ispitanika u features_cache zasebno radi K-fold CV unutar tog
    ispitanika. features_cache: dict subject_id -> (X, y), iz
    build_features_cache() - izračunat jednom i ponovno iskorišten za sve
    klasifikatore koji dijele istu reprezentaciju.

    Vraća pandas.DataFrame s jednim retkom po uspješno evaluiranom ispitaniku.
    Ispitanici s premalo uzoraka ili greškom pri fitanju se preskaču.
    """
    rows = []

    for subject_id, (X, y) in features_cache.items():
        if len(y) < cv_splits * 2:
            if verbose:
                print(f"[S{subject_id:03d}] premalo podataka za {cv_splits}-fold CV (n={len(y)}), preskačem")
            continue

        pipeline = _get_classifier_pipeline(classifier_name, random_state)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
        try:
            scores = cross_validate(
                pipeline, X, y, cv=cv,
                scoring={'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'},
                error_score='raise',
            )
        except Exception as e:
            values, counts = np.unique(y, return_counts=True)
            print(f"[S{subject_id:03d}] GREŠKA pri fitanju '{classifier_name}' "
                  f"(n={len(y)}, raspodjela klasa={dict(zip(values.tolist(), counts.tolist()))}): "
                  f"{type(e).__name__}: {e} - preskačem ovog ispitanika.")
            continue
        rows.append({
            'subject_id': subject_id,
            'accuracy': scores['test_accuracy'].mean(),
            'f1': scores['test_f1'].mean(),
            'roc_auc': scores['test_roc_auc'].mean(),
            'n_epochs': len(y),
        })
        if verbose:
            print(f"[S{subject_id:03d}] acc={rows[-1]['accuracy']:.3f} "
                  f"f1={rows[-1]['f1']:.3f} auc={rows[-1]['roc_auc']:.3f} (n={len(y)})")

    df = pd.DataFrame(rows)
    if verbose and len(df) > 0:
        print(f"\nPer-subject sažetak ({label}, {classifier_name}):")
        print(f"  Accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f} "
              f"(n={len(df)} ispitanika)")
        print(f"  F1:       {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
        print(f"  ROC-AUC:  {df['roc_auc'].mean():.3f} ± {df['roc_auc'].std():.3f}")
    return df


def plot_per_subject_boxplot(df, metric='accuracy', title='', out_path='per_subject_boxplot.png'):
    """Box-plot distribucije metrike kroz ispitanike."""
    if len(df) == 0:
        print("Prazan DataFrame - nema što obraditi")
        return

    fig, ax = plt.subplots(figsize=(4, 5))
    ax.boxplot(df[metric], vert=True, tick_labels=[metric])
    ax.scatter(np.ones(len(df)) + np.random.uniform(-0.04, 0.04, len(df)),
               df[metric], alpha=0.5, s=15, color='gray')
    ax.set_ylabel(metric)
    ax.set_title(title or f'Distribucija {metric} kroz ispitanike (n={len(df)})')
    fig.tight_layout()
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Box-plot spremljen u {out_path}")


# Cross-subject scenarij.

def _pool_features_cache(features_cache):
    """Spaja (X, y) svih ispitanika iz features_cache u jedan X, y, groups
    niz (groups = subject_id) - potrebno za GroupKFold/LeaveOneGroupOut.

    VAŽNO: iterira po SORTIRANIM subject_id, ne po redoslijedu dict-a.
    features_cache dolazi iz build_features_cache(), koji iterira
    all_subjects.items() - a taj redoslijed ovisi o tome kojim su redom
    os.walk() pronašao EDF datoteke na disku, što NIJE zajamčeno stabilno
    između pokretanja (može ovisiti o datotečnom sustavu/OS-u). GroupKFold
    ne miješa podatke (nema shuffle), nego dijeli grupe redoslijedom kojim
    se prvi put pojave - pa nestabilan ulazni redoslijed daje RAZLIČITU
    podjelu foldova (i time blago drugačije rezultate) iz pokretanja u
    pokretanje, iako je random_state fiksan. Sortiranje ovdje to uklanja."""
    X_list, y_list, groups_list = [], [], []
    for subject_id in sorted(features_cache.keys()):
        X, y = features_cache[subject_id]
        X_list.append(X)
        y_list.append(y)
        groups_list.append(np.full(len(y), subject_id))

    if not X_list:
        raise ValueError("features_cache je prazan - nema podataka niti za jednog ispitanika.")

    return (np.concatenate(X_list, axis=0),
            np.concatenate(y_list, axis=0),
            np.concatenate(groups_list, axis=0))


def evaluate_cross_subject(features_cache, classifier_name, cv_strategy='group_kfold',
                            n_splits=10, random_state=42, verbose=True, label=''):
    """Trenira na nekim ispitanicima, a testira na drugim ispitanicima.

    cv_strategy:
      'group_kfold'           - podijeli ispitanike u n_splits grupa (leave-N-subjects-out)
      'leave_one_subject_out' - svaki ispitanik jednom test skup

    Vraća pandas.DataFrame s jednim retkom po uspješnom foldu, uključujući
    ispitanike koji su u tom foldu bili izdvojeni za testiranje.
    """
    X, y, groups = _pool_features_cache(features_cache)
    if verbose:
        print(f"Spojeno {len(features_cache)} ispitanika, ukupno {len(y)} epoha.")

    if cv_strategy == 'leave_one_subject_out':
        cv = LeaveOneGroupOut()
    elif cv_strategy == 'group_kfold':
        n_groups = len(np.unique(groups))
        if n_groups < 2:
            raise ValueError(f"Za group_kfold treba barem 2 ispitanika, pronađen {n_groups}.")
        n_splits = min(n_splits, n_groups)
        cv = GroupKFold(n_splits=n_splits)
    else:
        raise ValueError(f"Nepoznat cv_strategy: {cv_strategy}")

    rows = []
    n_failed_folds = 0
    for fold_idx, (train_idx, test_idx) in enumerate(cv.split(X, y, groups=groups)):
        held_out = sorted(set(groups[test_idx].tolist()))
        pipeline_fold = _get_classifier_pipeline(classifier_name, random_state)

        try:
            pipeline_fold.fit(X[train_idx], y[train_idx])
            y_pred = pipeline_fold.predict(X[test_idx])
            y_proba = (pipeline_fold.predict_proba(X[test_idx])[:, 1]
                       if hasattr(pipeline_fold, 'predict_proba') else None)
        except Exception as e:
            train_values, train_counts = np.unique(y[train_idx], return_counts=True)
            n_failed_folds += 1
            print(f"[fold {fold_idx}] GREŠKA pri fitanju '{classifier_name}' "
                  f"(train raspodjela klasa={dict(zip(train_values.tolist(), train_counts.tolist()))}, "
                  f"n_train={len(train_idx)}): {type(e).__name__}: {e} - preskačem ovaj fold.")
            continue

        acc = accuracy_score(y[test_idx], y_pred)
        f1 = f1_score(y[test_idx], y_pred, zero_division=0)
        try:
            auc = roc_auc_score(y[test_idx], y_proba) if y_proba is not None else np.nan
        except ValueError:
            auc = np.nan  # test fold ima samo jednu klasu - AUC nedefiniran

        rows.append({
            'fold': fold_idx, 'held_out_subjects': held_out,
            'accuracy': acc, 'f1': f1, 'roc_auc': auc, 'n_test': len(test_idx),
        })
        if verbose:
            print(f"[fold {fold_idx}] test ispitanici={held_out[:5]}"
                  f"{'...' if len(held_out) > 5 else ''} acc={acc:.3f} f1={f1:.3f} auc={auc:.3f}")

    if n_failed_folds:
        print(f"  ({n_failed_folds} fold(ova) preskočeno zbog greške pri fitanju - "
              f"vidi poruke iznad; ako ih je puno, smanji broj n_splits ili povećaj "
              f"min_epochs_per_subject u build_features_cache.)")

    df = pd.DataFrame(rows)
    if len(df) == 0:
        raise ValueError(
            f"Svi foldovi su propali za '{classifier_name}' ({label}) - "
            f"provjeri greške ispisane iznad."
        )
    if verbose and len(df) > 0:
        print(f"\nCross-subject sažetak ({label}, {classifier_name}, strategija={cv_strategy}):")
        print(f"  Accuracy: {df['accuracy'].mean():.3f} ± {df['accuracy'].std():.3f}")
        print(f"  F1:       {df['f1'].mean():.3f} ± {df['f1'].std():.3f}")
        print(f"  ROC-AUC:  {df['roc_auc'].mean():.3f} ± {df['roc_auc'].std():.3f}")
        print("  (očekivano NIŽE od per-subject rezultata - EEG jako varira među osobama)")
    return df


# Konfuzijske matrice.

def plot_confusion_matrix(cm, class_names=('klasa 0', 'klasa 1'), title='',
                           out_path='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Expected')
    ax.set_ylabel('Real')
    ax.set_title(title)

    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, str(cm[i, j]), ha='center', va='center',
                     color='white' if cm[i, j] > cm.max() / 2 else 'black')

    fig.colorbar(im, ax=ax, fraction=0.046)
    fig.tight_layout()
    if os.path.dirname(out_path):
        os.makedirs(os.path.dirname(out_path), exist_ok=True)
    fig.savefig(out_path, dpi=150)
    plt.close(fig)
    print(f"Konfuzijska matrica spremljena u {out_path}")


def confusion_matrix_for_subject(features_cache, subject_id, classifier_name,
                                  cv_splits=5, random_state=42):
    """Računa out-of-fold konfuzijsku matricu za jednog ispitanika."""
    X, y = features_cache[subject_id]
    pipeline = _get_classifier_pipeline(classifier_name, random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    return confusion_matrix(y, y_pred, labels=[0, 1])


def confusion_matrix_for_all_subjects(features_cache, classifier_name,
                                      cv_splits=5, random_state=42,
                                      verbose=True):
    """Računa jednu konfuzijsku matricu iz svih per-subject predikcija.

    Svaki ispitanik se dijeli i predviđa zasebno, isto kao u
    ``evaluate_per_subject``. Tek nakon toga se spajaju stvarne i predviđene
    labele. Na taj način matrica predstavlja sve ispitanike, a ne samo jednog
    od njih, i nijedna epoha nije predviđena modelom koji ju je vidio u treningu.

    Ispitanik mora imati barem dvije klase i dovoljno uzoraka za zadani broj
    foldova. Ako neki ispitanik ne zadovoljava uvjet, preskače se kao i u
    ``evaluate_per_subject``. Funkcija vraća fiksnu 2x2 matricu s redoslijedom
    klasa 0, 1.
    """
    y_true_all = []
    y_pred_all = []
    used_subjects = []

    for subject_id in sorted(features_cache):
        X, y = features_cache[subject_id]
        values, counts = np.unique(y, return_counts=True)
        if len(values) < 2 or counts.min() < cv_splits:
            if verbose:
                print(f"[S{subject_id:03d}] preskačem konfuzijsku matricu "
                      f"(premalo uzoraka po klasi za {cv_splits}-fold CV).")
            continue

        pipeline = _get_classifier_pipeline(classifier_name, random_state)
        cv = StratifiedKFold(n_splits=cv_splits, shuffle=True,
                             random_state=random_state)
        try:
            y_pred = cross_val_predict(pipeline, X, y, cv=cv)
        except Exception as exc:
            if verbose:
                print(f"[S{subject_id:03d}] greška pri izračunu konfuzijske "
                      f"matrice: {type(exc).__name__}: {exc}")
            continue

        y_true_all.append(y)
        y_pred_all.append(y_pred)
        used_subjects.append(subject_id)

    if not y_true_all:
        raise ValueError("Nema ispitanika s dovoljno podataka za konfuzijsku matricu.")

    if verbose:
        print(f"Pooled konfuzijska matrica: {len(used_subjects)} ispitanika, "
              f"{sum(len(y) for y in y_true_all)} epoha.")

    return confusion_matrix(
        np.concatenate(y_true_all),
        np.concatenate(y_pred_all),
        labels=[0, 1],
    )


# Grid analiza: zadatak, reprezentacija, kanali i klasifikator.

def run_experiment_grid(all_subjects, tasks, feature_types, channel_selections,
                         classifier_names, scenario='per_subject',
                         out_csv='results/experiment_grid.csv',
                         cv_strategy='group_kfold', random_state=42,
                         use_disk_cache=True, verbose=True):
    """Pokreće tražene kombinacije i rezultate zapisuje u CSV.
    rezultate zapisuje u CSV, redak po redak.

    Vanjska petlja ide po (task, feature_type, channel_selection), pa se
    značajke računaju jednom po kombinaciji i dijele između klasifikatora.
    CSP+LDA se računa
    zasebno (koristi 'raw' reprezentaciju, ne band_power/time_frequency).

    scenario: 'per_subject' | 'cross_subject'"""
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)

    # Na kraju provjeravamo jesu li sve tražene kombinacije zapisane.
    expected_combos = set()
    for task in tasks:
        for c in classifier_names:
            if c == 'csp_lda':
                for ch in channel_selections:
                    expected_combos.add((task, 'raw', ch, c))
            elif c == 'eegnet':
                for ch in channel_selections:
                    expected_combos.add((task, 'eegnet_raw', ch, c))
            else:
                for ft, ch in itertools.product(feature_types, channel_selections):
                    expected_combos.add((task, ft, ch, c))

    write_header = not os.path.exists(out_csv)

    if not write_header:
        # Upozori ako se u isti CSV pokušava dodati drugi scenarij.
        try:
            existing = pd.read_csv(out_csv)
            existing_scenarios = set(existing['scenario'].unique()) if len(existing) else set()
        except Exception:
            existing_scenarios = set()
        if existing_scenarios and scenario not in existing_scenarios:
            print(f"UPOZORENJE: {out_csv} već sadrži retke sa scenario={sorted(existing_scenarios)}, "
                  f"a sada dopisuješ scenario='{scenario}'. Ako ovo nije namjerno (npr. htio si "
                  f"odvojene fajlove za per_subject i cross_subject), prekini i promijeni out_csv "
                  f"prije nastavka - inače će se u istom fajlu pomiješati dva različita scenarija.")

    # Ako CSV već postoji, preskačemo kombinacije koje su već zapisane.
    completed_combos = set()
    if not write_header:
        try:
            existing = pd.read_csv(out_csv)
            existing = existing[existing['scenario'] == scenario]
            completed_combos = set(
                zip(existing['task'], existing['feature_type'],
                    existing['channel_selection'], existing['classifier'])
            )
            if completed_combos and verbose:
                print(f"Resume: {len(completed_combos)} kombinacija već postoji u {out_csv}, preskačem ih.")
        except Exception:
            completed_combos = set()

    fieldnames = ['scenario', 'task', 'feature_type', 'channel_selection',
                  'classifier', 'accuracy_mean', 'accuracy_std', 'f1_mean',
                  'f1_std', 'roc_auc_mean', 'roc_auc_std', 'n']

    feature_classifiers = [c for c in classifier_names if c not in ('csp_lda', 'eegnet')]
    use_csp = 'csp_lda' in classifier_names
    use_eegnet = 'eegnet' in classifier_names
    if use_eegnet and not TORCH_AVAILABLE:
        print("UPOZORENJE: 'eegnet' je u classifier_names, ali PyTorch nije "
              "instaliran - preskačem EEGNet kombinacije u ovom gridu.")
        use_eegnet = False

    import csv as csv_module
    with open(out_csv, 'a', newline='') as fh:
        writer = csv_module.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for task in tasks:
            # Feature-based klasifikatori dijele isti cache značajki.
            if feature_classifiers:
                for feature_type, channel_selection in itertools.product(feature_types, channel_selections):
                    label = f"{task}/{feature_type}/{channel_selection}"

                    remaining_classifiers = [
                        c for c in feature_classifiers
                        if (task, feature_type, channel_selection, c) not in completed_combos
                    ]
                    if not remaining_classifiers:
                        if verbose:
                            print(f"\n>>> Preskačem (sve već gotovo): {label}")
                        continue

                    if verbose:
                        print(f"\n>>> Računam značajke: {label}")
                    cache = build_features_cache(
                        all_subjects, task, feature_type, channel_selection,
                        use_disk_cache=use_disk_cache, verbose=verbose,
                    )
                    if not cache:
                        print(f"  preskočeno (nema podataka): {label}")
                        continue

                    for classifier_name in remaining_classifiers:
                        if verbose:
                            print(f"  --- klasifikator: {classifier_name} ---")
                        try:
                            if scenario == 'per_subject':
                                df = evaluate_per_subject(cache, classifier_name,
                                                           random_state=random_state, verbose=False, label=label)
                            else:
                                df = evaluate_cross_subject(cache, classifier_name, cv_strategy=cv_strategy,
                                                             random_state=random_state, verbose=False, label=label)
                        except ValueError as e:
                            print(f"  preskočeno: {e}")
                            continue

                        row = {
                            'scenario': scenario, 'task': task, 'feature_type': feature_type,
                            'channel_selection': channel_selection, 'classifier': classifier_name,
                            'accuracy_mean': df['accuracy'].mean(), 'accuracy_std': df['accuracy'].std(),
                            'f1_mean': df['f1'].mean(), 'f1_std': df['f1'].std(),
                            'roc_auc_mean': df['roc_auc'].mean(), 'roc_auc_std': df['roc_auc'].std(),
                            'n': len(df),
                        }
                        writer.writerow(row)
                        fh.flush()
                        if verbose:
                            print(f"  acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} "
                                  f"f1={row['f1_mean']:.3f} auc={row['roc_auc_mean']:.3f}")

            # CSP+LDA radi nad sirovim epohama.
            if use_csp:
                for channel_selection in channel_selections:
                    label = f"{task}/raw/{channel_selection}"

                    if (task, 'raw', channel_selection, 'csp_lda') in completed_combos:
                        if verbose:
                            print(f"\n>>> Preskačem (već gotovo): {label}")
                        continue

                    if verbose:
                        print(f"\n>>> Računam sirove epohe (CSP): {label}")
                    cache = build_features_cache(
                        all_subjects, task, 'raw', channel_selection,
                        use_disk_cache=use_disk_cache, verbose=verbose,
                    )
                    if not cache:
                        print(f"  preskočeno (nema podataka): {label}")
                        continue

                    try:
                        if scenario == 'per_subject':
                            df = evaluate_per_subject(cache, 'csp_lda',
                                                       random_state=random_state, verbose=False, label=label)
                        else:
                            df = evaluate_cross_subject(cache, 'csp_lda', cv_strategy=cv_strategy,
                                                         random_state=random_state, verbose=False, label=label)
                    except ValueError as e:
                        print(f"  preskočeno: {e}")
                        continue

                    row = {
                        'scenario': scenario, 'task': task, 'feature_type': 'raw',
                        'channel_selection': channel_selection, 'classifier': 'csp_lda',
                        'accuracy_mean': df['accuracy'].mean(), 'accuracy_std': df['accuracy'].std(),
                        'f1_mean': df['f1'].mean(), 'f1_std': df['f1'].std(),
                        'roc_auc_mean': df['roc_auc'].mean(), 'roc_auc_std': df['roc_auc'].std(),
                        'n': len(df),
                    }
                    writer.writerow(row)
                    fh.flush()
                    if verbose:
                        print(f"  acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} "
                              f"f1={row['f1_mean']:.3f} auc={row['roc_auc_mean']:.3f}")

            # EEGNet koristi sirove epohe s dodatnom dimenzijom.
            if use_eegnet:
                for channel_selection in channel_selections:
                    label = f"{task}/eegnet_raw/{channel_selection}"

                    if (task, 'eegnet_raw', channel_selection, 'eegnet') in completed_combos:
                        if verbose:
                            print(f"\n>>> Preskačem (već gotovo): {label}")
                        continue

                    if verbose:
                        print(f"\n>>> Računam sirove epohe (EEGNet): {label}")
                    cache = build_features_cache(
                        all_subjects, task, 'eegnet_raw', channel_selection,
                        use_disk_cache=use_disk_cache, verbose=verbose,
                    )
                    if not cache:
                        print(f"  preskočeno (nema podataka): {label}")
                        continue

                    try:
                        if scenario == 'per_subject':
                            df = evaluate_per_subject(cache, 'eegnet',
                                                       random_state=random_state, verbose=False, label=label)
                        else:
                            df = evaluate_cross_subject(cache, 'eegnet', cv_strategy=cv_strategy,
                                                         random_state=random_state, verbose=False, label=label)
                    except ValueError as e:
                        print(f"  preskočeno: {e}")
                        continue

                    row = {
                        'scenario': scenario, 'task': task, 'feature_type': 'eegnet_raw',
                        'channel_selection': channel_selection, 'classifier': 'eegnet',
                        'accuracy_mean': df['accuracy'].mean(), 'accuracy_std': df['accuracy'].std(),
                        'f1_mean': df['f1'].mean(), 'f1_std': df['f1'].std(),
                        'roc_auc_mean': df['roc_auc'].mean(), 'roc_auc_std': df['roc_auc'].std(),
                        'n': len(df),
                    }
                    writer.writerow(row)
                    fh.flush()
                    if verbose:
                        print(f"  acc={row['accuracy_mean']:.3f}±{row['accuracy_std']:.3f} "
                              f"f1={row['f1_mean']:.3f} auc={row['roc_auc_mean']:.3f}")

    print(f"\nGrid rezultati spremljeni u {out_csv}")

    final_df = pd.read_csv(out_csv)
    actual_combos = set(zip(
        final_df.loc[final_df['scenario'] == scenario, 'task'],
        final_df.loc[final_df['scenario'] == scenario, 'feature_type'],
        final_df.loc[final_df['scenario'] == scenario, 'channel_selection'],
        final_df.loc[final_df['scenario'] == scenario, 'classifier'],
    ))
    missing_combos = expected_combos - actual_combos

    print(f"Očekivano kombinacija: {len(expected_combos)}, stvarno u CSV-u "
          f"(scenario={scenario}): {len(actual_combos & expected_combos)}")
    if missing_combos:
        print(f"NEDOSTAJE {len(missing_combos)} traženih kombinacija (nisu "
              f"uspješno izračunate : "
              f"'nema podataka' ili grešku pri fitanju):")
        for combo in sorted(missing_combos):
            print(f"  - {combo}")
    else:
        print("Sve tražene kombinacije su prisutne u CSV-u.")

    return final_df


if __name__ == '__main__':
    from obrada import loading_files
    from pretprocesiranje import PreprocessConfig

    config = PreprocessConfig(
        apply_car=True, apply_bandpass=True, bandpass_low=8.0, bandpass_high=30.0,
        artifact_method='threshold', amplitude_reject_uv=150.0, channel_selection='all',
    )

    all_subjects = loading_files(preprocess_config=config)

    # Brzi per-subject primjer.
    cache = build_features_cache(all_subjects, 'rest_vs_task', 'band_power', 'all')
    df_per_subject = evaluate_per_subject(cache, 'lda', label='rest_vs_task/band_power/all')
    plot_per_subject_boxplot(df_per_subject, metric='accuracy',
                              title='Per-subject accuracy: rest vs. task (LDA, band-power)',
                              out_path='results/boxplot_rest_vs_task_lda.png')

    # Cross-subject primjer koristi isti cache.
    df_cross_subject = evaluate_cross_subject(cache, 'lda', cv_strategy='group_kfold', n_splits=10,
                                               label='rest_vs_task/band_power/all')

    # Jedna pooled per-subject matrica za najbolju rest_vs_task konfiguraciju.
    # Predikcije se rade odvojeno po ispitaniku pa se tek onda spajaju.
    cm_per_subject = confusion_matrix_for_all_subjects(
        cache, 'svm_linear', cv_splits=5,
    )
    plot_confusion_matrix(
        cm_per_subject,
        class_names=('rest', 'task'),
        title='Per-subject pooled: rest vs. task (linear SVM, band-power)',
        out_path='results/confusion_matrix_rest_vs_task_svm_linear.png',
    )

    # Grid uključuje sve definirane zadatke i klasifikatore. EEGNet se
    # preskače ako PyTorch nije instaliran. Scenariji imaju odvojene CSV-ove.
    ALL_TASKS = ['left_right_fist', 'fists_feet', 'rest_vs_task', 'execution_vs_imagery']
    ALL_CLASSIFIERS = ['lda', 'svm_linear', 'svm_rbf', 'random_forest',
                        'gradient_boosting', 'csp_lda', 'eegnet']

    run_experiment_grid(
        all_subjects,
        tasks=ALL_TASKS,
        feature_types=['band_power', 'time_frequency'],
        channel_selections=['all', 'motor'],
        classifier_names=ALL_CLASSIFIERS,
        scenario='per_subject',
        out_csv='results/experiment_grid_per_subject.csv',
    )

    run_experiment_grid(
        all_subjects,
        tasks=ALL_TASKS,
        feature_types=['band_power', 'time_frequency'],
        channel_selections=['all', 'motor'],
        classifier_names=ALL_CLASSIFIERS,
        scenario='cross_subject',
        cv_strategy='group_kfold',
        out_csv='results/experiment_grid_cross_subject.csv',
    )