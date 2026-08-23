"""Klasifikacijski modeli i evaluacija
    Dvije skupine pipelinea:
        Feature_based koji radi nad predobrađenom 2D matricom iz pretprocesiranja.py 
        (band-power ili vremensko-frekvencijske značajke): LDA, SVM, Random Forest, Gradient Boosting
    
    CSP + LDA radi izravno nad sirovim epoha nizom (n_epoha, n_kanala, n_uzoraka),
    zato što CSP mora biti fitan unutar cross-validation folda da bi evaluacija bila korektna 
    inače ako je CSP fitan nad cijelim skupom prije podjele nastaje curenje podataka

"""


import numpy as np
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis as LDA
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier, HistGradientBoostingClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.model_selection import cross_validate, cross_val_predict, StratifiedKFold, train_test_split
from sklearn.metrics import confusion_matrix
from mne.decoding import CSP


def print_class_balance(y, label=''):
    #dijagnostički ispis raspodjele klasa (pokretanje omogućuje da klasifikator stane točno na baseline accuracy)
    
    values, counts = np.unique(y, return_counts=True)
    total = len(y)
    dist = {int(v): f'{c} ({c/total:.1%})' for v, c in zip(values, counts)}
    print(f"Raspodjela klasa{f' ({label})' if label else ''}: {dist}, "
          f"baseline accuracy (majority class) = {counts.max()/total:.3f}")

    
#baseline i napredni pipeline za precomputed značajke

def build_feature_based_pipelines(random_state=42, pca_variance=0.95):
    #vraća dict naziv sklearn Pipelena, služi se 2D (n_epoha i n_znacajki) 
    #iz značajke.py filea funkcije extract_band_power ili extract_time_frequency_features

    #RF i HistGradientBoosting dobivaju PCA kroak prije klasifikatora
    #PCA je fitan unutar pipelinea dakle koristi se samo na train foldu u cross-validationu

    # sanitize random_state: allow int, None, or np.random.RandomState
    if isinstance(random_state, np.random.RandomState) or random_state is None:
        rs = random_state
    else:
        try:
            rs = int(random_state)
        except Exception:
            rs = 42

    return{        
        'lda': Pipeline(
            [
                ('scaler', StandardScaler()),
                ('clf', LDA()), 
            ]
        ),
        'svm_linear': Pipeline(
            [
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='linear', probability=True, random_state=rs)),
            ]
            
        ),
        'svm_rbf': Pipeline(
            [
                ('scaler', StandardScaler()),
                ('clf', SVC(kernel='rbf', probability=True, random_state=rs)),
            ]
        ),
        'random_forest': Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=pca_variance, random_state=rs)),
                ('clf', RandomForestClassifier(
                    n_estimators=300, max_depth=8, min_samples_leaf=3,
                    class_weight='balanced', random_state=rs)),
            ]
        ),
       # HistGradientBoostingClassifier se koristi zato što je brzi i otporan na koreliranje značajke bez ručnog podešavanja dubine
       # i podržava class_weight izravno
        'gradient_boosting': Pipeline(
            [
                ('scaler', StandardScaler()),
                ('pca', PCA(n_components=pca_variance, random_state=rs)),
                ('clf', HistGradientBoostingClassifier(max_depth= 6, class_weight='balanced',
                    random_state=rs)),
            ]
        ),
        
    }
    
def build_csp_lda_pipeline(n_components=4):
        """CSP + LDA metode korištene za obradu singlal se 
        često koriste u kombinaciji na temlju BCI literature
        Funkcija radi izravno nad sirovim epoha nizom (n_epoha, kanala i uzoraka)"""
        return Pipeline(
            [
                ('csp', CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
                ('clf', LDA()),
            ]
        )

# evaluacija (accuracy, F1, ROC-AUC)
def  evaluate_pipeline(pipeline, X, y, cv_splits=5, random_state=42, n_jobs=1):
    """per-subject scenarij (k-fold cross-validation)
        X može biti 2D (feature-based pipeline) ili 3D (CSP+LDA nad sirovim
        epohama) - sklearn cross_validate radi jednako u oba slučaja jer samo
        indeksira prvu os.
     
        Svi zadaci iz features.TASK_NAMES su binarni pa su accuracy/F1/ROC-AUC
        izravno primjenjivi bez one-vs-rest proširenja"""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'}
    
    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)
    
    return{
        'accuracy_mean': float(scores['test_accuracy'].mean()),
        'accuracy_std': float(scores['test_accuracy'].std()),
        'f1_mean': float(scores['test_f1'].mean()),
        'f1_std': float(scores['test_f1'].std()),
        'roc_auc_mean': float(scores['test_roc_auc'].mean()),
        'roc_auc_std': float(scores['test_roc_auc'].std()),
        'n_folds': cv_splits,
        'n_samples': len(y),
    }
    
def get_confusion_matrix(pipeline, X, y, cv_splits=5, random_state=42, n_jobs=1):
    #poštena procjena grešaka modela (svaka epoha se predviđa modelom koji ju nije vidio u treningu)
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=n_jobs)
    return confusion_matrix(y, y_pred)
 


def evaluate_all_pipelines(X_features, X_raw_epochs, y, cv_splits=5, random_state=42):
    """Prikladna funkcija za brzu usporedbu SVIH pipelinea (5.2) na jednom
    zadatku: feature-based pipeline-i dobivaju X_features (2D), CSP+LDA
    dobiva X_raw_epochs (3D, iz epochs.get_data()). Vraća dict naziv -> rezultati."""
    results = {}
 
    for name, pipe in build_feature_based_pipelines(random_state).items():
        results[name] = evaluate_pipeline(pipe, X_features, y, cv_splits, random_state)
 
    results['csp_lda'] = evaluate_pipeline(
        build_csp_lda_pipeline(), X_raw_epochs, y, cv_splits, random_state)
 
    return results




def print_results_table(results, task_name=''):
    header = f"{'Model':<20}{'Accuracy':>12}{'F1':>12}{'ROC-AUC':>12}"
    print(f"\n=== Rezultati: {task_name} ===")
    print(header)
    print('-' * len(header))
    for name, r in results.items():
        print(f"{name:<20}{r['accuracy_mean']:>7.3f}±{r['accuracy_std']:.2f}"
              f"{r['f1_mean']:>7.3f}±{r['f1_std']:.2f}"
              f"{r['roc_auc_mean']:>7.3f}±{r['roc_auc_std']:.2f}")
        
#demo prvi rezultat klasifikacija nad jednim ispitanikom

if __name__ == '__main__':
    from obrada import loading_files
    from pretprocesiranje import PreprocessConfig
    from značajke import (
        extract_band_power_features, select_epochs_for_task, TASK_NAMES,
    )

    config = PreprocessConfig(
        apply_car=True, apply_bandpass=True, bandpass_low=8.0, bandpass_high=30.0,
        artifact_method='treshold', amplitude_reject_uv=150.0, channel_selection='all',
    )
    
    subjects = loading_files(preprocess_config=config)
    subject_id = 1
    if subject_id not in subjects:
        raise SystemExit(f"Ispitanik S{subject_id:03d} nije pronađen - provjeri putanju do 'files/'.")
    
    subject_runs = subjects[subject_id]
    
    for task_name in TASK_NAMES:
        try:
            epochs, y = select_epochs_for_task(subject_runs, task_name)
        except ValueError as e:
            print(f"Preskačem '{task_name}': {e}")
            continue
        
    X_features, _ = extract_band_power_features(epochs)
    X_raw = epochs.get_data() #za CSP + LDA
    
    results = evaluate_all_pipelines(X_features, X_raw, y, cv_splits=5)
    print_results_table(results, task_name=f"{task_name} (S{subject_id:03d}, n={len(y)})")
