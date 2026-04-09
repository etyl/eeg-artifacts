import sys
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
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, roc_auc_score, roc_curve

# # ── Config ────────────────────────────────────────────────────────────────────
# EMBEDDINGS_PATH = Path("/home/jade/Documents/INRIA/retreat/embeddings/cbramod_embeddings/embeddings_npatients2993_nseconds60.0.npy")
# METADATA_PATH   = Path("/home/jade/Documents/INRIA/retreat/embeddings/cbramod_embeddings/metadata_2993_nseconds60.0.json")
# #EMBEDDINGS_PATH = Path("/data/parietal/store2/data/tuh_eeg_abnormal/embeddings/cbramod_embeddings/embeddings_npatients2993_nseconds60.0.npy")
# #METADATA_PATH   = Path("/data/parietal/store2/data/tuh_eeg_abnormal/embeddings/cbramod_embeddings/metadata_npatients2993_nseconds60.0.json")
# OUTPUT_DIR      = Path("./output")
# N_WINDOWS       = 15  
# RANDOM_STATE    = 42
# # ─────────────────────────────────────────────────────────────────────────────

# OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
BASE = Path("~/Documents/INRIA/retreat/embeddings").expanduser()
 
MODEL_PATHS = {
    "cbramod": {
        "embeddings": BASE / "cbramod_embeddings"/ "embeddings_npatients2993_nseconds60.0.npy",
        "metadata":   BASE / "cbramod_embeddings" / "metadata_2993_nseconds60.0.json",
    },
    "cbramod_30": {
        "embeddings": BASE / "cbramod_embeddings" / "embeddings_npatients2993_nseconds30.0.npy",
        "metadata":   BASE / "cbramod_embeddings" / "metadata_2993_nseconds30.0.json",
    },
    "reve": {
        "embeddings": BASE / "reve_embeddings" / "embeddings.npy",
        "metadata":   BASE / "reve_embeddings" / "metadata.json",
    },
}
 
RANDOM_STATE = 42
FONT_SIZE = 18
# ── Parse argument ────────────────────────────────────────────────────────────
if len(sys.argv) < 2 or sys.argv[1].lower() not in MODEL_PATHS:
    print(f"Usage: python {sys.argv[0]} <model>")
    print(f"  Available models: {', '.join(MODEL_PATHS)}")
    sys.exit(1)
 
model_name     = sys.argv[1].lower()
embeddings_path = MODEL_PATHS[model_name]["embeddings"]
metadata_path   = MODEL_PATHS[model_name]["metadata"]
output_dir      = Path(f"./output_{model_name}")
output_dir.mkdir(parents=True, exist_ok=True)
 
print(f"Model       : {model_name}")
print(f"Embeddings  : {embeddings_path}")
print(f"Metadata    : {metadata_path}")
print(f"Output dir  : {output_dir}")
 
# ── 1. Load ───────────────────────────────────────────────────────────────────
print("\nLoading embeddings and metadata…")
raw = np.load(embeddings_path)
with open(metadata_path) as f:
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
        raise ValueError(f"embeddings dim 0 ({n_patients}) != metadata entries ({len(metadata)}).")
else:
    raise ValueError(f"Unexpected embeddings shape: {raw.shape}")
 
labels       = np.array([int(m["pathological"]) for m in metadata])
is_train     = np.array([bool(m["train"])        for m in metadata])
 
print(f"  patients={n_patients}, windows/patient={n_windows}, dim={emb_dim}")
print(f"  train={is_train.sum()}  eval={(~is_train).sum()}")
print(f"  pathological={labels.sum()}  normal={n_patients - labels.sum()}")
 
# ── 2. Split using official train/eval flag from metadata ─────────────────────
train_patients = np.where( is_train)[0]
eval_patients  = np.where(~is_train)[0]
 
patient_ids = np.repeat(np.arange(n_patients), n_windows)
X_win       = embeddings.reshape(-1, emb_dim)
y_win       = np.repeat(labels, n_windows)
 
train_mask     = np.isin(patient_ids, train_patients)
eval_mask      = np.isin(patient_ids, eval_patients)
X_tr, y_tr     = X_win[train_mask], y_win[train_mask]
X_ev           = X_win[eval_mask]
patient_ids_ev = patient_ids[eval_mask]
 
eval_patient_labels = labels[eval_patients]
 
scaler  = StandardScaler()
X_tr_sc = scaler.fit_transform(X_tr)
X_ev_sc = scaler.transform(X_ev)
 
print(f"\n  Train windows={len(y_tr)}  Eval windows={len(X_ev)}")
 
# ── Helpers ───────────────────────────────────────────────────────────────────
def patient_avg_probs(win_probs, patient_ids_ev, eval_patients):
    return np.array([
        win_probs[patient_ids_ev == pid].mean() for pid in eval_patients
    ])
 
def youden_threshold(fpr, tpr, thresholds):
    idx = np.argmax(tpr - fpr)
    return thresholds[idx], tpr[idx], fpr[idx]
 
# ── 3. Classifiers + ROC curves ───────────────────────────────────────────────
clfs = {
    "LogisticRegression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "LinearSVC":          LinearSVC(max_iter=2000, random_state=RANDOM_STATE),
    "Random Forest":    RandomForestClassifier(n_estimators=100, random_state=RANDOM_STATE),
    "MLP":                MLPClassifier(hidden_layer_sizes=(128, 64), max_iter=500,
                                        random_state=RANDOM_STATE),
}
 
fig_roc, ax_roc = plt.subplots(figsize=(7, 6))
ax_roc.plot([0, 1], [0, 1], "k--", lw=1, label="Random")
 
print()
for name, clf in clfs.items():
    clf.fit(X_tr_sc, y_tr)
 
    if hasattr(clf, "predict_proba"):
        win_probs = clf.predict_proba(X_ev_sc)[:, 1]
    else:
        score     = clf.decision_function(X_ev_sc)
        win_probs = (score - score.min()) / (score.max() - score.min() + 1e-9)
 
    pat_probs                  = patient_avg_probs(win_probs, patient_ids_ev, eval_patients)
    fpr, tpr, thresholds       = roc_curve(eval_patient_labels, pat_probs)
    auc                        = roc_auc_score(eval_patient_labels, pat_probs)
    best_thresh, b_tpr, b_fpr  = youden_threshold(fpr, tpr, thresholds)
 
    acc_50  = accuracy_score(eval_patient_labels, (pat_probs >= 0.50).astype(int))
    acc_opt = accuracy_score(eval_patient_labels, (pat_probs >= best_thresh).astype(int))
 
    print(f"  {name:<22}  AUC={auc:.3f}  "
          f"acc@0.50={acc_50:.3f}  "
          f"acc@Youden({best_thresh:.2f})={acc_opt:.3f}")
 
    ax_roc.plot(fpr, tpr, lw=2, label=f"{name}  (AUC={auc:.3f})")
    ax_roc.scatter([b_fpr], [b_tpr], marker="x", s=80, zorder=5)
 
ax_roc.set_xlabel("False Positive Rate")
ax_roc.set_ylabel("True Positive Rate")
ax_roc.set_title(f"ROC curves — {model_name} — patient-level")
ax_roc.legend(frameon=False, fontsize=9)
ax_roc.grid(alpha=0.2, linestyle=":")
fig_roc.tight_layout()
roc_path = output_dir / "roc_curves.png"
fig_roc.savefig(roc_path, dpi=200)
plt.close(fig_roc)
print(f"\n  ROC curves saved to {roc_path}")
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
               s=13, alpha=0.6, c=color, label=lbl, edgecolors="none")
ax.set_title(f"t-SNE — {model_name} — patient embeddings (n={n_patients})", fontsize=FONT_SIZE)
ax.set_xlabel("t-SNE 1", fontsize=FONT_SIZE)
ax.set_ylabel("t-SNE 2", fontsize=FONT_SIZE)
ax.legend(frameon=False, markerscale=2)
ax.grid(alpha=0.2, linestyle=":")
fig.tight_layout()
tsne_path = output_dir / f"tsne_patients_{model_name}.png"
fig.savefig(tsne_path, dpi=200)
plt.close(fig)
print(f"  t-SNE saved to {tsne_path}")

# Also plot the UMAP 
umap_reducer = umap.UMAP(n_components=2, random_state=RANDOM_STATE)
umap_coords = umap_reducer.fit_transform(X_sc)
fig, ax = plt.subplots(figsize=(10, 8))
for cls, color, label in [(0, "#1f77b4", "normal"), (1, "#d62728", "pathological")]:
    mask = labels == cls
    ax.scatter(umap_coords[mask, 0], umap_coords[mask, 1],
               s=13, alpha=0.6, c=color, label=label, edgecolors="none")
ax.set_title(f"UMAP — {model_name} — patient embeddings (n={n_patients})", fontsize=FONT_SIZE)
ax.set_xlabel("UMAP 1", fontsize=FONT_SIZE)
ax.set_ylabel("UMAP 2", fontsize=FONT_SIZE)
ax.legend(frameon=False, markerscale=2)
ax.grid(alpha=0.2, linestyle=":")
fig.tight_layout()
out_png = output_dir / f"umap_patients_{model_name}.png"
fig.savefig(out_png, dpi=200)
plt.close(fig)
print(f"UMAP plot saved to {out_png}")
