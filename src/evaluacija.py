"""Evaluacija modela (per-subject scenarij, cross-subject scenarij, konfuzijske matrice
    i sustavna analiza utjecaja (kanali * reprezentacija * klasifikator) s CSV logiranjem i box-plotovima)

    BITNO!!!
    Kod cross-subject scenarija ispitanik ne smije nikad istovremeneno biti u train i test skupu
    (ako nije tako implementirano model djelomično uči osobne karakteristike tog ispitanik što umjetno napuhava rezultate)
    zato se ovdje koristi GroupFold/LeaveOneGroupOut s groups=subject_id, a ne običan K-fold
    cross-validation kao što je implementirano u per-subject scenariju

    VAŽNO O PERFORMANSAMA (popravljeno u ovoj verziji):
    Značajke (band-power/wavelet) i epohiranje po zadatku se sada računaju SAMO JEDNOM po
    kombinaciji (task, feature_type, channel_selection) i keširaju u memoriji (i opcionalno na
    disku), a zatim se nad ISTIM podacima isprobaju svi klasifikatori. Prijašnja verzija je
    vanjskom petljom išla po (task, classifier), pa se ista (skupa) ekstrakcija značajki - 
    pogotovo wavelet, tfr_morlet - ponavljala jednom PO SVAKOM klasifikatoru, za svakog
    ispitanika. To je uzrok sporosti/prekida (KeyboardInterrupt), ne greška u rezultatima.
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
from modeli import build_feature_based_pipelines, build_csp_lda_pipeline


FEATURE_EXTRACTORS = {
    'band_power': extract_band_power_features,
    'time_frequency': extract_time_frequency_features,
}

CACHE_DIR = 'cache'


def _picks_for(channel_selection):
    return MOTOR_CORTEX_CHANNELS if channel_selection == 'motor' else None


def _get_classifier_pipeline(classifier_name, random_state=42):
    """'lda' | 'svm_linear' | 'svm_rbf' | 'random_forest' | 'gradient_boosting'
    | 'csp_lda'. csp_lda radi nad sirovim epohama, ostali nad značajkama."""
    if classifier_name == 'csp_lda':
        return build_csp_lda_pipeline()
    pipelines = build_feature_based_pipelines(random_state=random_state)
    if classifier_name not in pipelines:
        raise ValueError(f"Nepoznat classifier_name: {classifier_name}")
    return pipelines[classifier_name]


# ============================================================================
# Keširanje značajki - JEDNOM po (task, feature_type, channel_selection),
# ponovno iskorišteno za sve klasifikatore koji rade nad tom reprezentacijom.
# ============================================================================

def _disk_cache_key(task_name, feature_type, channel_selection, modality, subject_ids,
                     min_epochs_per_subject=30):
    raw = (f"{task_name}|{feature_type}|{channel_selection}|{modality}|"
           f"{min_epochs_per_subject}|{sorted(subject_ids)}")
    return hashlib.md5(raw.encode()).hexdigest()


def _load_disk_cache(key):
    path = os.path.join(CACHE_DIR, f'{key}.pkl')
    if os.path.exists(path):
        with open(path, 'rb') as fh:
            return pickle.load(fh)
    return None


def _save_disk_cache(key, obj):
    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(os.path.join(CACHE_DIR, f'{key}.pkl'), 'wb') as fh:
        pickle.dump(obj, fh)


def build_features_cache(all_subjects, task_name, feature_type, channel_selection,
                          modality=None, use_disk_cache=True, verbose=True,
                          min_epochs_per_subject=30):
    """Računa X, y JEDNOM po ispitaniku za zadanu (task, feature_type,
    channel_selection) kombinaciju. Vraća dict subject_id -> (X, y).

    feature_type: 'band_power' | 'time_frequency' | 'raw' (raw = sirove epohe
    za CSP+LDA, bez izdvajanja značajki).

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
        else:
            extractor = FEATURE_EXTRACTORS[feature_type]
            X, _ = extractor(epochs, picks=picks)

        # NaN/Inf provjera - odbaci SAMO zahvaćene epohe (retke), ne cijelog
        # ispitanika, jer se problem obično tiče par pojedinačnih epoha
        # (npr. rubni efekt filtra na kratkom segmentu), ne cijelog zapisa.
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


# ============================================================================
# 6.1 Per-subject scenarij
# ============================================================================

def evaluate_per_subject(features_cache, classifier_name, cv_splits=5,
                          random_state=42, verbose=True, label=''):
    """Za svakog ispitanika u features_cache ZASEBNO: K-fold CV unutar tog
    ispitanika. features_cache: dict subject_id -> (X, y), iz
    build_features_cache() - JEDNOM izračunat i ponovno iskorišten za sve
    klasifikatore koji dijele istu reprezentaciju.

    Vraća pandas.DataFrame, jedan red po ispitaniku
    (subject_id, accuracy, f1, roc_auc, n_epoha)."""
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


# ============================================================================
# 6.2 Cross-subject scenarij
# ============================================================================

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
    """Model treniran na skupu ispitanika, testiran na POTPUNO DRUGIM
    ispitanicima (nikad isti subject_id u train i test).

    cv_strategy:
      'group_kfold'           - podijeli ispitanike u n_splits grupa (leave-N-subjects-out)
      'leave_one_subject_out' - svaki ispitanik jednom test skup

    Vraća pandas.DataFrame, jedan red po foldu."""
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


# ============================================================================
# 6.3 Konfuzijske matrice
# ============================================================================

def plot_confusion_matrix(cm, class_names=('klasa 0', 'klasa 1'), title='',
                           out_path='confusion_matrix.png'):
    fig, ax = plt.subplots(figsize=(4, 4))
    im = ax.imshow(cm, cmap='Blues')
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel('Predviđeno')
    ax.set_ylabel('Stvarno')
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
    """Konfuzijska matrica za JEDNOG ispitanika, iz out-of-fold predikcija
    (koristi već izračunate značajke iz features_cache)."""
    X, y = features_cache[subject_id]
    pipeline = _get_classifier_pipeline(classifier_name, random_state)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv)
    return confusion_matrix(y, y_pred)


# ============================================================================
# 6.4 Sustavna analiza utjecaja (grid) - kanali x reprezentacija x klasifikator
# ============================================================================

def run_experiment_grid(all_subjects, tasks, feature_types, channel_selections,
                         classifier_names, scenario,
                         out_csv='results/experiment_grid.csv',
                         cv_strategy='group_kfold', random_state=42,
                         use_disk_cache=True, verbose=True):
    """Sustavno provodi SVE kombinacije (task x feature_type x channel_selection
    x classifier) i sprema rezultate u CSV, redak po redak.

    KLJUČNA RAZLIKA od prijašnje verzije: vanjska petlja ide po
    (task, feature_type, channel_selection) - značajke se računaju JEDNOM po
    toj kombinaciji (build_features_cache), a klasifikatori se isprobavaju
    kao unutarnja petlja NAD ISTIM keširanim podacima. CSP+LDA se računa
    zasebno (koristi 'raw' reprezentaciju, ne band_power/time_frequency).

    scenario: 'per_subject' | 'cross_subject'"""
    os.makedirs(os.path.dirname(out_csv) or '.', exist_ok=True)
    write_header = not os.path.exists(out_csv)

    if not write_header:
        # Zaštita od miješanja scenarija u istom CSV-u kroz uzastopna
        # pokretanja (npr. slučajno pokretanje scenario='per_subject' pa
        # scenario='cross_subject' na isti out_csv put) - upravo ovo je
        # uzrokovalo duplicirane retke s identičnim brojkama pod pogrešnom
        # oznakom scenarija u prethodnom rezultatu.
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

    # RESUME: učitaj kombinacije koje su VEĆ izračunate i spremljene u
    # out_csv, da se izbjegne upravo ono što se dogodilo u prethodnom
    # rezultatu - isti (task, feature_type, channel_selection, classifier)
    # izračunat dvaput (jer je skripta prekinuta pa ponovno pokrenuta bez
    # provjere što je već gotovo), s blago drugačijim brojkama zbog
    # nestabilnog redoslijeda ispitanika između pokretanja.
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

    feature_classifiers = [c for c in classifier_names if c != 'csp_lda']
    use_csp = 'csp_lda' in classifier_names

    import csv as csv_module
    with open(out_csv, 'a', newline='') as fh:
        writer = csv_module.DictWriter(fh, fieldnames=fieldnames)
        if write_header:
            writer.writeheader()

        for task in tasks:
            # --- feature-based klasifikatori: značajke računamo JEDNOM po (feature_type, channel_selection) ---
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

            # --- CSP+LDA: 'raw' reprezentacija, računa se JEDNOM po channel_selection ---
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

    print(f"\nGrid rezultati spremljeni u {out_csv}")
    return pd.read_csv(out_csv)


if __name__ == '__main__':
    from obrada import loading_files
    from pretprocesiranje import PreprocessConfig

    config = PreprocessConfig(
        apply_car=True, apply_bandpass=True, bandpass_low=8.0, bandpass_high=30.0,
        artifact_method='threshold', amplitude_reject_uv=150.0, channel_selection='all',
    )

    all_subjects = loading_files(preprocess_config=config)

    # --- per-subject scenarij za jedan zadatak/klasifikator ---
    cache = build_features_cache(all_subjects, 'rest_vs_task', 'band_power', 'all')
    df_per_subject = evaluate_per_subject(cache, 'lda', label='rest_vs_task/band_power/all')
    plot_per_subject_boxplot(df_per_subject, metric='accuracy',
                              title='Per-subject accuracy: rest vs. task (LDA, band-power)',
                              out_path='results/boxplot_rest_vs_task_lda.png')

    # --- cross-subject scenarij (leave-N-subjects-out), ISTI cache, bez ponovnog računanja ---
    df_cross_subject = evaluate_cross_subject(cache, 'lda', cv_strategy='group_kfold', n_splits=10,
                                               label='rest_vs_task/band_power/all')

    # --- sustavna grid analiza (preporuka: prvo testiraj na par ispitanika prije punog skupa) ---
    run_experiment_grid(
        all_subjects,
        tasks=['left_right_fist', 'rest_vs_task'],
        feature_types=['band_power', 'time_frequency'],
        channel_selections=['all', 'motor'],
        classifier_names=['lda', 'svm_linear', 'random_forest', 'csp_lda'],
        scenario='cross_subject',
        out_csv='results/experiment_grid_cross_subject.csv',
    )