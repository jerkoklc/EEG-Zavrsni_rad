"""Sklearn pipelinei i pomoćne funkcije za EEG klasifikaciju.

Klasični modeli rade nad već izračunatim značajkama, a CSP+LDA nad sirovim
epohama. CSP je u pipelineu kako bi se fit-ao samo na trening dijelu folda.

Opcionalno: pojednostavljena EEGNet (PyTorch) arhitektura za usporedbu s
klasičnim pristupima u raspravi rada - aktivna samo ako je torch instaliran.
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
from sklearn.base import BaseEstimator, ClassifierMixin
from mne.decoding import CSP


def print_class_balance(y, label=''):
    """Ispisuje broj uzoraka po klasi i accuracy većinske klase."""
    values, counts = np.unique(y, return_counts=True)
    total = len(y)
    dist = {int(v): f'{c} ({c/total:.1%})' for v, c in zip(values, counts)}
    print(f"Raspodjela klasa{f' ({label})' if label else ''}: {dist}, "
          f"baseline accuracy (majority class) = {counts.max()/total:.3f}")


# Pipelinei nad 2D značajkama.

def build_feature_based_pipelines(random_state=42, pca_variance=0.95):
    """Vraća pipeline za svaki klasični klasifikator.
    (n_epoha, n_znacajki) iz features.extract_band_power_features ili
    features.extract_time_frequency_features.

    RF i GradientBoosting imaju PCA korak prije klasifikatora:
    Band-power i vremensko-frekvencijske značajke su dosta korelirane preko
    kanala (64 kanala x nekoliko pojaseva), a stablo-bazirani modeli s
    zadanim hiperparametrima na takvim, visokodimenzionalnim, koreliranim
    značajkama (uz svega stotinjak-dvjestotinjak epoha po ispitaniku) često
    često počnu predviđati samo većinsku klasu. PCA je fitan unutar pipelinea
    (dakle samo na
    train foldu u cross-validationu)."""
    return {
        'lda': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', LDA()),
        ]),
        'svm_linear': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='linear', probability=True, random_state=random_state)),
        ]),
        'svm_rbf': Pipeline([
            ('scaler', StandardScaler()),
            ('clf', SVC(kernel='rbf', probability=True, random_state=random_state)),
        ]),
        'random_forest': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_variance, random_state=random_state)),
            ('clf', RandomForestClassifier(
                n_estimators=300, max_depth=8, min_samples_leaf=3,
                class_weight='balanced', random_state=random_state)),
        ]),
        # HistGradientBoostingClassifier (sklearn >=1.0) zamjenjuje
        # GradientBoostingClassifier: podržava class_weight izravno, brži je
        # i otporniji na korelirane značajke bez ručnog podešavanja dubine.
        'gradient_boosting': Pipeline([
            ('scaler', StandardScaler()),
            ('pca', PCA(n_components=pca_variance, random_state=random_state)),
            ('clf', HistGradientBoostingClassifier(
                max_depth=6, class_weight='balanced', random_state=random_state)),
        ]),
    }


def build_csp_lda_pipeline(n_components=4):
    """Stvara CSP+LDA pipeline za sirove epohe.

    CSP koristi labele tijekom fitanja, zato mora ostati unutar pipelinea i
    validacijskog folda. Ulaz ima oblik (epoha, kanal, uzorak).
    """
    return Pipeline([
        ('csp', CSP(n_components=n_components, reg=None, log=True, norm_trace=False)),
        ('clf', LDA()),
    ])


# Osnovne evaluacijske funkcije.

def evaluate_pipeline(pipeline, X, y, cv_splits=5, random_state=42, n_jobs=1):
    """Pokreće stratificiranu K-fold validaciju i vraća glavne metrike.
    X može biti 2D (feature-based pipeline) ili 3D (CSP+LDA nad sirovim
    epohama) - sklearn cross_validate radi jednako u oba slučaja jer samo
    indeksira prvu os.

    Svi zadaci su binarni, pa se accuracy, F1 i ROC-AUC računaju izravno.
    Vraća srednju vrijednost i standardnu devijaciju svake metrike."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    scoring = {'accuracy': 'accuracy', 'f1': 'f1', 'roc_auc': 'roc_auc'}

    scores = cross_validate(pipeline, X, y, cv=cv, scoring=scoring, n_jobs=n_jobs)

    return {
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
    """Vraća konfuzijsku matricu iz out-of-fold predikcija."""
    cv = StratifiedKFold(n_splits=cv_splits, shuffle=True, random_state=random_state)
    y_pred = cross_val_predict(pipeline, X, y, cv=cv, n_jobs=n_jobs)
    return confusion_matrix(y, y_pred)


def evaluate_all_pipelines(X_features, X_raw_epochs, y, cv_splits=5, random_state=42):
    """Uspoređuje sve pipelinee na jednom zadatku.
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


# --- opcionalno: pojednostavljena EEGNet arhitektura (PyTorch) ------------

try:
    import torch
    import torch.nn as nn
    TORCH_AVAILABLE = True
except ImportError:
    TORCH_AVAILABLE = False


if TORCH_AVAILABLE:

    class EEGNet(nn.Module):
        """Pojednostavljena EEGNet arhitektura (prema Lawhern i sur., 2018).
        za klasifikaciju izravno iz filtriranih EEG epoha, bez ručnog
        izdvajanja značajki. Ulaz: (batch, 1, n_kanala, n_uzoraka).

        Ovo je pomoćna implementacija za usporedbu s klasičnim modelima."""

        def __init__(self, n_channels, n_times, n_classes=2,
                     F1=8, D=2, F2=16, kernel_length=64, dropout=0.5):
            super().__init__()
            self.block1 = nn.Sequential(
                nn.Conv2d(1, F1, (1, kernel_length), padding=(0, kernel_length // 2), bias=False),
                nn.BatchNorm2d(F1),
                nn.Conv2d(F1, F1 * D, (n_channels, 1), groups=F1, bias=False),  # depthwise (prostorni filtri)
                nn.BatchNorm2d(F1 * D),
                nn.ELU(),
                nn.AvgPool2d((1, 4)),
                nn.Dropout(dropout),
            )
            self.block2 = nn.Sequential(
                nn.Conv2d(F1 * D, F1 * D, (1, 16), padding=(0, 8), groups=F1 * D, bias=False),  # separable: depthwise
                nn.Conv2d(F1 * D, F2, 1, bias=False),  # separable: pointwise
                nn.BatchNorm2d(F2),
                nn.ELU(),
                nn.AvgPool2d((1, 8)),
                nn.Dropout(dropout),
            )
            with torch.no_grad():
                dummy = torch.zeros(1, 1, n_channels, n_times)
                out_dim = self.block2(self.block1(dummy)).flatten(1).shape[1]
            self.classify = nn.Linear(out_dim, n_classes)

        def forward(self, x):
            x = self.block1(x)
            x = self.block2(x)
            x = x.flatten(1)
            return self.classify(x)


    def train_eegnet(X, y, n_epochs=30, batch_size=32, lr=1e-3, device='cpu',
                      val_split=0.2, random_state=42):
        """Trenira EEGNet uz jednostavnu train/validation podjelu.
        X: (n_epoha, 1, n_kanala, n_uzoraka) - iz features.prepare_raw_epoch_array
        y: (n_epoha,)
        Vraća (model, history); history sadrži train/val loss i val accuracy po epohi.

        NAPOMENA: koristi jednostavnu train/val podjelu (ne cross-validation)
        radi brzine - za konačne rezultate uskladiti s istim CV protokolom
        koji se koristi za ostale pipeline-e (evaluate_pipeline)."""
        X_train, X_val, y_train, y_val = train_test_split(
            X, y, test_size=val_split, stratify=y, random_state=random_state)

        n_channels, n_times = X.shape[2], X.shape[3]
        n_classes = len(np.unique(y))
        model = EEGNet(n_channels=n_channels, n_times=n_times, n_classes=n_classes).to(device)
        optimizer = torch.optim.Adam(model.parameters(), lr=lr)
        criterion = nn.CrossEntropyLoss()

        X_train_t = torch.tensor(X_train, dtype=torch.float32).to(device)
        y_train_t = torch.tensor(y_train, dtype=torch.long).to(device)
        X_val_t = torch.tensor(X_val, dtype=torch.float32).to(device)
        y_val_t = torch.tensor(y_val, dtype=torch.long).to(device)

        history = {'train_loss': [], 'val_loss': [], 'val_acc': []}
        n_samples = X_train_t.shape[0]

        for epoch in range(n_epochs):
            model.train()
            perm = torch.randperm(n_samples)
            epoch_loss = 0.0
            for i in range(0, n_samples, batch_size):
                idx = perm[i:i + batch_size]
                xb, yb = X_train_t[idx], y_train_t[idx]
                optimizer.zero_grad()
                out = model(xb)
                loss = criterion(out, yb)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item() * len(idx)
            epoch_loss /= n_samples

            model.eval()
            with torch.no_grad():
                val_out = model(X_val_t)
                val_loss = criterion(val_out, y_val_t).item()
                val_acc = (val_out.argmax(1) == y_val_t).float().mean().item()

            history['train_loss'].append(epoch_loss)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

        return model, history


    class EEGNetClassifier(BaseEstimator, ClassifierMixin):
        """sklearn-kompatibilan wrapper oko EEGNet - omogućuje da se EEGNet
        koristi kroz ISTI cross_validate/GroupKFold kod kao ostali
        klasifikatori (evaluate_per_subject/evaluate_cross_subject u
        evaluacija.py), umjesto da ostane izolirana petlja za treniranje.

        Prima X oblika (n_epoha, 1, n_kanala, n_uzoraka) - vidi
        značajke.prepare_raw_epoch_array. Tretira se u evaluacija.py na isti
        način kao CSP+LDA: kao klasifikator koji radi nad sirovom (ne
        precomputed feature) reprezentacijom.

        NAPOMENA o troškovima: treniranje neuronske mreže unutar SVAKOG CV
        folda (5x po ispitaniku za per-subject, ili 10x/broj-ispitanika za
        cross-subject) je znatno sporije od klasičnih klasifikatora - očekuj
        da grid s EEGNet-om traje puno duže. Testiraj prvo na par ispitanika/
        jednom zadatku prije punog grida."""

        def __init__(self, n_epochs=30, batch_size=32, lr=1e-3, dropout=0.5,
                     F1=8, D=2, F2=16, kernel_length=64, random_state=42, device='cpu'):
            self.n_epochs = n_epochs
            self.batch_size = batch_size
            self.lr = lr
            self.dropout = dropout
            self.F1 = F1
            self.D = D
            self.F2 = F2
            self.kernel_length = kernel_length
            self.random_state = random_state
            self.device = device

        def fit(self, X, y):
            torch.manual_seed(self.random_state)
            self.classes_ = np.unique(y)
            y_idx = np.searchsorted(self.classes_, y)

            n_channels, n_times = X.shape[2], X.shape[3]
            self.model_ = EEGNet(
                n_channels=n_channels, n_times=n_times, n_classes=len(self.classes_),
                F1=self.F1, D=self.D, F2=self.F2, kernel_length=self.kernel_length,
                dropout=self.dropout,
            ).to(self.device)

            optimizer = torch.optim.Adam(self.model_.parameters(), lr=self.lr)
            criterion = nn.CrossEntropyLoss()

            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            y_t = torch.tensor(y_idx, dtype=torch.long).to(self.device)
            n = len(X_t)

            self.model_.train()
            for _ in range(self.n_epochs):
                perm = torch.randperm(n)
                for i in range(0, n, self.batch_size):
                    idx = perm[i:i + self.batch_size]
                    optimizer.zero_grad()
                    loss = criterion(self.model_(X_t[idx]), y_t[idx])
                    loss.backward()
                    optimizer.step()

            return self

        def predict_proba(self, X):
            self.model_.eval()
            X_t = torch.tensor(X, dtype=torch.float32).to(self.device)
            with torch.no_grad():
                proba = torch.softmax(self.model_(X_t), dim=1).cpu().numpy()
            return proba

        def predict(self, X):
            idx = self.predict_proba(X).argmax(axis=1)
            return self.classes_[idx]


def build_eegnet_pipeline(**kwargs):
    """Vraća EEGNetClassifier - nema smisla stavljati u sklearn Pipeline sa
    StandardScaler ispred, jer EEGNet ima vlastite BatchNorm slojeve koji
    rade istu normalizacijsku ulogu unutar mreže."""
    if not TORCH_AVAILABLE:
        raise RuntimeError(
            "PyTorch nije instaliran - EEGNet klasifikator nije dostupan. "
            "Instaliraj s 'pip install torch' ako ga želiš koristiti."
        )
    return EEGNetClassifier(**kwargs)


# --- demo: prvi rezultati klasifikacije za jednog ispitanika --------------

if __name__ == '__main__':
    from obrada import loading_files
    from pretprocesiranje import PreprocessConfig
    from značajke import (
        extract_band_power_features, select_epochs_for_task, TASK_NAMES,
    )

    config = PreprocessConfig(
        apply_car=True, apply_bandpass=True, bandpass_low=8.0, bandpass_high=30.0,
        artifact_method='threshold', amplitude_reject_uv=150.0, channel_selection='all',
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
        X_raw = epochs.get_data()  # za CSP+LDA

        results = evaluate_all_pipelines(X_features, X_raw, y, cv_splits=5)
        print_results_table(results, task_name=f"{task_name} (S{subject_id:03d}, n={len(y)})")