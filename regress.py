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
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

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
    n_windows  = total_rows // n_patients
    embeddings = raw.reshape(n_patients, n_windows, emb_dim)
elif raw.ndim == 3:
    embeddings = raw
    n_patients, n_windows, emb_dim = embeddings.shape
    if n_patients != len(metadata):
        raise ValueError(f"embeddings first dim ({n_patients}) != metadata entries ({len(metadata)}).")
else:
    raise ValueError(f"Unexpected embeddings shape: {raw.shape}")
 
labels      = np.array([int(m["pathological"]) for m in metadata])
X_win       = embeddings.reshape(-1, emb_dim)
y_win       = np.repeat(labels, n_windows)
patient_ids = np.repeat(np.arange(n_patients), n_windows)
 
print(f"  patients={n_patients}, windows/patient={n_windows}, dim={emb_dim}")
print(f"  pathological={labels.sum()}  normal={n_patients - labels.sum()}")
 
# 2. Patient-level train/eval split (no leakage)
train_patients, eval_patients = train_test_split(
    np.arange(n_patients), test_size=0.2,
    random_state=RANDOM_STATE, stratify=labels
)
 
train_mask     = np.isin(patient_ids, train_patients)
eval_mask      = np.isin(patient_ids, eval_patients)
X_tr, y_tr     = X_win[train_mask], y_win[train_mask]
X_ev, y_ev     = X_win[eval_mask],  y_win[eval_mask]
patient_ids_ev = patient_ids[eval_mask]
 
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_ev_sc = scaler.transform(X_ev)
 
eval_patient_labels = labels[eval_patients]
 
print(f"\n  Train windows={len(y_tr)}  Eval windows={len(y_ev)}")
 
 
def find_optimal_threshold(fpr, tpr, thresholds):
    """Youden's J statistic: maximises TPR - FPR (= sensitivity + specificity - 1)."""
    j_scores = tpr - fpr
    best_idx = np.argmax(j_scores)
    return thresholds[best_idx], tpr[best_idx], fpr[best_idx]
 
 
def patient_scores_and_preds(win_probs, patient_ids_ev, eval_patients, threshold):
    """Average window probabilities per patient, then threshold."""
    patient_probs = np.array([
        win_probs[patient_ids_ev == pid].mean() for pid in eval_patients
    ])
    patient_preds = (patient_probs >= threshold).astype(int)
    return patient_probs, patient_preds
 
 
# 3. Train classifiers, compute ROC curves, find optimal threshold
clfs = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "LinearSVC":          LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
    "MLP":                MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                        random_state=RANDOM_STATE),
}
 
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
 
print()
results = {}
for name, clf in clfs.items():
    clf.fit(X_tr_sc, y_tr)
 
    if hasattr(clf, "predict_proba"):
        win_probs = clf.predict_proba(X_ev_sc)[:, 1]
    else:
        score     = clf.decision_function(X_ev_sc)
        win_probs = (score - score.min()) / (score.max() - score.min() + 1e-9)
 
    # Patient-level scores (averaged over windows)
    patient_probs = np.array([
        win_probs[patient_ids_ev == pid].mean() for pid in eval_patients
    ])
 
    # ROC curve on patient-level scores
    fpr, tpr, thresholds = roc_curve(eval_patient_labels, patient_probs)
    auc = roc_auc_score(eval_patient_labels, patient_probs)
 
    # Optimal threshold via Youden's J
    best_thresh, best_tpr, best_fpr = find_optimal_threshold(fpr, tpr, thresholds)
 
    # Accuracy at fixed 0.5 threshold
    _, preds_50  = patient_scores_and_preds(win_probs, patient_ids_ev, eval_patients, 0.50)
    acc_50       = accuracy_score(eval_patient_labels, preds_50)
 
    # Accuracy at optimal threshold
    _, preds_opt = patient_scores_and_preds(win_probs, patient_ids_ev, eval_patients, best_thresh)
    acc_opt      = accuracy_score(eval_patient_labels, preds_opt)
 
    results[name] = dict(auc=auc, acc_50=acc_50, acc_opt=acc_opt, best_thresh=best_thresh)
 
    print(f"  {name:<22}  AUC={auc:.3f}  "
          f"acc@0.50={acc_50:.3f}  "
          f"acc@Youden({best_thresh:.2f})={acc_opt:.3f}")
 
    # Plot ROC
    ax_roc.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.3f})")
    ax_roc.scatter([best_fpr], [best_tpr], marker="x", s=80, zorder=5)
 
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title("ROC curves — patient-level (window avg score)")
ax_roc.legend(frameon=False, fontsize=9)
ax_roc.grid(alpha=0.2, linestyle=":")
fig_roc.tight_layout()
roc_path = OUTPUT_DIR / "roc_curves.png"
fig_roc.savefig(roc_path, dpi=200)
plt.close(fig_roc)
print(f"\n  ROC curves saved to {roc_path}")
print("  (✕ markers show the Youden-optimal operating point per model)")
 
 
# 4. t-SNE
print("\nRunning t-SNE (this may take a minute)…")
X_mean = embeddings.mean(axis=1)
X_sc   = StandardScaler().fit_transform(X_mean)
n_pca  = min(50, emb_dim, n_patients - 1)
X_pca  = PCA(n_components=n_pca, random_state=RANDOM_STATE).fit_transform(X_sc)
 
perplexity = min(30, max(5, n_patients // 100))
coords = TSNE(n_components=2, perplexity=perplexity, init="pca",
              learning_rate="auto", random_state=RANDOM_STATE, n_jobs=-1).fit_transform(X_pca)
 
fig, ax = plt.subplots(figsize=(10, 8))
for cls, color, lbl in [(0, "#1f77b4", "normal"), (1, "#d62728", "pathological")]:
    mask = labels == cls
    ax.scatter(coords[mask, 0], coords[mask, 1],
               s=8, alpha=0.6, c=color, label=lbl, edgecolors="none")
ax.set_title(f"t-SNE of patient embeddings  (n={n_patients})", fontsize=13)
ax.set_xlabel("t-SNE 1"); ax.set_ylabel("t-SNE 2")
ax.legend(frameon=False, markerscale=2)
ax.grid(alpha=0.2, linestyle=":")
fig.tight_layout()
tsne_path = OUTPUT_DIR / "tsne_patients.png"
fig.savefig(tsne_path, dpi=200)
plt.close(fig)
print(f"  t-SNE plot saved to {tsne_path}")

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
