import json
import numpy as np
import matplotlib
import umap
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.manifold import TSNE
from sklearn.linear_model import LogisticRegression
from sklearn.svm import LinearSVC
from sklearn.neural_network import MLPClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split

# ── Config ────────────────────────────────────────────────────────────────────
EMBEDDINGS_PATH = Path("/home/jade/Documents/INRIA/retreat/embeddings/cbramod_embeddings/embeddings_npatients2993_nseconds60.0.npy")
METADATA_PATH   = Path("/home/jade/Documents/INRIA/retreat/embeddings/cbramod_embeddings/metadata_2993_nseconds60.0.json")
#EMBEDDINGS_PATH = Path("/data/parietal/store2/data/tuh_eeg_abnormal/embeddings/cbramod_embeddings/embeddings_npatients2993_nseconds60.0.npy")
#METADATA_PATH   = Path("/data/parietal/store2/data/tuh_eeg_abnormal/embeddings/cbramod_embeddings/metadata_npatients2993_nseconds60.0.json")
OUTPUT_DIR      = Path("./output")
N_WINDOWS       = 15  
RANDOM_STATE    = 42
# ─────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
# 1. Load data
print("Loading embeddings and metadata…")
raw = np.load(EMBEDDINGS_PATH)
with open(METADATA_PATH) as f:
    metadata = json.load(f)
 
if raw.ndim == 2:
    n_patients = len(metadata)
    total_rows, emb_dim = raw.shape
    if total_rows % n_patients != 0:
        raise ValueError(f"raw rows ({total_rows}) not divisible by n_patients ({n_patients}).")
    n_windows = total_rows // n_patients
    embeddings = raw.reshape(n_patients, n_windows, emb_dim)
elif raw.ndim == 3:
    embeddings = raw
    n_patients, n_windows, emb_dim = embeddings.shape
    if n_patients != len(metadata):
        raise ValueError(f"embeddings first dim ({n_patients}) != metadata entries ({len(metadata)}).")
else:
    raise ValueError(f"Unexpected embeddings shape: {raw.shape}")
 
labels = np.array([int(m["pathological"]) for m in metadata])   # (n_patients,)
 
print(f"  patients={n_patients}, windows/patient={n_windows}, dim={emb_dim}")
print(f"  pathological={labels.sum()}  normal={n_patients - labels.sum()}")
 
# 2. Expand labels and patient index to window level
#    X_win : (n_patients * n_windows, emb_dim)
#    y_win : (n_patients * n_windows,)  — same label repeated for all windows of a patient
#    patient_ids : which patient each window belongs to
X_win       = embeddings.reshape(-1, emb_dim)                          # flatten windows
y_win       = np.repeat(labels, n_windows)                             # repeat label per window
patient_ids = np.repeat(np.arange(n_patients), n_windows)             # track patient index
 
# 3. Train / eval split — split at PATIENT level to avoid leakage, then expand to windows
train_patients, eval_patients = train_test_split(
    np.arange(n_patients),
    test_size=0.2, random_state=RANDOM_STATE,
    stratify=labels
)
 
train_mask = np.isin(patient_ids, train_patients)
eval_mask  = np.isin(patient_ids, eval_patients)
 
X_tr, y_tr = X_win[train_mask], y_win[train_mask]
X_ev, y_ev = X_win[eval_mask],  y_win[eval_mask]
patient_ids_ev = patient_ids[eval_mask]
 
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_ev_sc = scaler.transform(X_ev)
 
print(f"\n  Train windows={len(y_tr)}  Eval windows={len(y_ev)}")
 
# 4. Train classifiers on windows, predict per window, aggregate by majority vote per patient
clfs = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "LinearSVC":          LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
    "MLP":                MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500, random_state=RANDOM_STATE),
}
 
eval_patient_labels = labels[eval_patients]   # true label per eval patient
 
print()
for name, clf in clfs.items():
    # Train on windows
    clf.fit(X_tr_sc, y_tr)
 
    # Predict on eval windows
    win_preds = clf.predict(X_ev_sc)           # (n_eval_windows,)
 
    # Aggregate: majority vote per patient
    patient_preds = []
    for pid in eval_patients:
        mask       = patient_ids_ev == pid
        votes      = win_preds[mask]           # predictions for all windows of this patient
        majority   = int(votes.sum() > len(votes) / 2)   # 1 if >50% windows predicted pathological
        patient_preds.append(majority)
    patient_preds = np.array(patient_preds)
 
    acc = accuracy_score(eval_patient_labels, patient_preds)
 
    # For AUROC use soft scores averaged over windows
    if hasattr(clf, "predict_proba"):
        win_probs = clf.predict_proba(X_ev_sc)[:, 1]
    else:
        score     = clf.decision_function(X_ev_sc)
        win_probs = (score - score.min()) / (score.max() - score.min() + 1e-9)
 
    patient_probs = np.array([
        win_probs[patient_ids_ev == pid].mean() for pid in eval_patients
    ])
    auc = roc_auc_score(eval_patient_labels, patient_probs)
 
    print(f"  {name:<22}  patient acc={acc:.3f}  patient auc={auc:.3f}")
 
# 5. t-SNE — one point per patient (mean embedding for visualisation only)
print("\nRunning t-SNE (this may take a minute)…")
X_mean = embeddings.mean(axis=1)               # (n_patients, emb_dim) — only for viz
X_sc   = StandardScaler().fit_transform(X_mean)
n_pca  = min(50, emb_dim, n_patients - 1)
X_pca  = PCA(n_components=n_pca, random_state=RANDOM_STATE).fit_transform(X_sc)
 
perplexity = min(30, max(5, n_patients // 100))
coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
              learning_rate="auto", random_state=RANDOM_STATE, n_jobs=-1).fit_transform(X_pca)
 
fig, ax = plt.subplots(figsize=(10, 8))
for cls, color, label in [(0, "#1f77b4", "normal"), (1, "#d62728", "pathological")]:
    mask = labels == cls
    ax.scatter(coords[mask, 0], coords[mask, 1],
               s=8, alpha=0.6, c=color, label=label, edgecolors="none")
ax.set_title(f"t-SNE of patient embeddings  (n={n_patients})", fontsize=13)
ax.set_xlabel("t-SNE 1")
ax.set_ylabel("t-SNE 2")
ax.legend(frameon=False, markerscale=2)
ax.grid(alpha=0.2, linestyle=":")
fig.tight_layout()
out_png = OUTPUT_DIR / "tsne_patients.png"
fig.savefig(out_png, dpi=200)
plt.close(fig)
print(f"t-SNE plot saved to {out_png}")
# Also plot the UMAP 
umap_reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
umap_coords = umap_reducer.fit_transform(X_sc)
fig, ax = plt.subplots(figsize=(10, 8))
for cls, color, label in [(0, "#1f77b4", "normal"), (1, "#d62728", "pathological")]:
    mask = labels == cls
    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               s=8, alpha=0.6, c=color, label=label, edgecolors="none")
ax.set_title(f"UMAP of patient embeddings  (n={n_patients})", fontsize=13)
ax.set_xlabel("UMAP 1")
ax.set_ylabel("UMAP 2")
ax.legend(frameon=False, markerscale=2)
ax.grid(alpha=0.2, linestyle=":")
fig.tight_layout()
out_png = OUTPUT_DIR / "umap_patients.png"
fig.savefig(out_png, dpi=200)
plt.close(fig)
print(f"UMAP plot saved to {out_png}")
