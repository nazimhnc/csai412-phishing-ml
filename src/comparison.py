"""
Model Comparison & Evaluation Script
=====================================
CSAI412 Machine Learning Group Project — Phishing Website Detection

Loads ALL trained models from data/*.pkl, evaluates them on the same test set,
creates comparison tables, visualisations, and performs statistical significance
tests between top-performing models.

Binary classification: Phishing (0) vs Legitimate (1)
Additional metric: ROC-AUC (since this is binary classification)

Outputs:
    data/comparison_results.csv
    figures/comparison_accuracy.png
    figures/comparison_metrics.png
    figures/comparison_radar.png
    figures/comparison_confusion_matrices.png
"""

import os
import sys
import time
import glob
import warnings
import numpy as np
import pandas as pd
import joblib
import matplotlib
matplotlib.use("Agg")  # non-interactive backend
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
    classification_report,
)
from sklearn.cluster import KMeans
from scipy import stats

warnings.filterwarnings("ignore")

# ── paths ────────────────────────────────────────────────────────────────
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
SRC_DIR = os.path.join(PROJECT_DIR, "src")

os.makedirs(FIGURES_DIR, exist_ok=True)

# Add src to path so we can import data_loader
sys.path.insert(0, SRC_DIR)
from data_loader import get_train_test, load_data, CLASS_NAMES

# ── model registry ───────────────────────────────────────────────────────
# Maps a human-readable name to the expected .pkl filename (without .pkl extension).
# The actual files use the pattern <stem>_model.pkl (e.g., knn_model.pkl).
MODEL_REGISTRY = {
    "Logistic Regression": "logistic_regression_model",
    "K-Nearest Neighbours": "knn_model",
    "SVM (Linear)": "svm_linear_model",
    "SVM (RBF)": "svm_rbf_model",
    "Decision Tree": "decision_tree_model",
    "Multi-Layer Perceptron": "mlp_model",
    "K-Means Clustering": "kmeans_model",
}

# Stem names (without _model suffix) for matching training-time files
MODEL_STEMS = {
    "Logistic Regression": "logistic_regression",
    "K-Nearest Neighbours": "knn",
    "SVM (Linear)": "svm_linear",
    "SVM (RBF)": "svm_rbf",
    "Decision Tree": "decision_tree",
    "Multi-Layer Perceptron": "mlp",
    "K-Means Clustering": "kmeans",
}

REQUIRED_SUPERVISED_FILES = [
    "logistic_regression_model", "knn_model", "svm_linear_model", "svm_rbf_model",
    "decision_tree_model", "mlp_model",
]


# ── helper: wait for model files ─────────────────────────────────────────
def wait_for_models(timeout_seconds=600, poll_interval=20):
    """Block until at least 6 .pkl files (the supervised ones) exist."""
    start = time.time()
    while True:
        pkl_files = glob.glob(os.path.join(DATA_DIR, "*.pkl"))
        stems = {os.path.splitext(os.path.basename(f))[0] for f in pkl_files}
        present = [m for m in REQUIRED_SUPERVISED_FILES if m in stems]
        print(
            f"[Comparison] {len(present)}/{len(REQUIRED_SUPERVISED_FILES)} supervised "
            f"models found  ({', '.join(present) or 'none yet'})  "
            f"[elapsed {time.time()-start:.0f}s]"
        )
        if len(present) >= len(REQUIRED_SUPERVISED_FILES):
            # Also check for kmeans (optional but nice)
            if "kmeans_model" in stems:
                print("[Comparison] K-Means model also found. All 7 models ready.")
            else:
                print("[Comparison] K-Means not yet found — will proceed without it if needed.")
            return
        elapsed = time.time() - start
        if elapsed > timeout_seconds:
            if len(present) >= 4:
                print(f"[Comparison] Timeout after {timeout_seconds}s but {len(present)} models available. Proceeding.")
                return
            raise TimeoutError(
                f"Only {len(present)} models found after {timeout_seconds}s. "
                "Cannot produce a meaningful comparison."
            )
        time.sleep(poll_interval)


# ── load models ──────────────────────────────────────────────────────────
def load_all_models():
    """Return dict {display_name: model} for every .pkl file found."""
    models = {}
    for display_name, file_stem in MODEL_REGISTRY.items():
        path = os.path.join(DATA_DIR, f"{file_stem}.pkl")
        if os.path.exists(path):
            models[display_name] = joblib.load(path)
            print(f"  Loaded {display_name} from {path}")
        else:
            print(f"  [SKIP] {display_name} — file not found: {path}")
    return models


# ── load training-time metadata ──────────────────────────────────────────
def load_training_times():
    """Try to load training times saved by model scripts (JSON or CSV)."""
    times = {}
    # Check for a shared training_times.json
    json_path = os.path.join(DATA_DIR, "training_times.json")
    if os.path.exists(json_path):
        import json
        with open(json_path) as f:
            times = json.load(f)
        return times

    # Fallback: look for individual <model>_time.txt files
    for stem in MODEL_STEMS.values():
        time_path = os.path.join(DATA_DIR, f"{stem}_time.txt")
        if os.path.exists(time_path):
            with open(time_path) as f:
                try:
                    times[stem] = float(f.read().strip())
                except ValueError:
                    pass

    # Also check per-model metadata pkl
    for stem in MODEL_STEMS.values():
        meta_path = os.path.join(DATA_DIR, f"{stem}_metadata.pkl")
        if os.path.exists(meta_path):
            meta = joblib.load(meta_path)
            if isinstance(meta, dict) and "training_time" in meta:
                times[stem] = meta["training_time"]

    return times


# ── evaluate supervised models ───────────────────────────────────────────
def evaluate_supervised(models, X_test, y_test, training_times):
    """Evaluate all supervised models and return a comparison DataFrame.

    Since this is binary classification, we also compute ROC-AUC where possible.
    """
    rows = []
    predictions = {}
    for name, model in models.items():
        if "K-Means" in name:
            continue  # handled separately
        y_pred = model.predict(X_test)
        predictions[name] = y_pred
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
        rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
        f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        # ROC-AUC (binary classification)
        try:
            if hasattr(model, "predict_proba"):
                y_prob = model.predict_proba(X_test)[:, 1]
                auc = roc_auc_score(y_test, y_prob)
            elif hasattr(model, "decision_function"):
                y_scores = model.decision_function(X_test)
                auc = roc_auc_score(y_test, y_scores)
            else:
                auc = roc_auc_score(y_test, y_pred)
        except Exception:
            auc = None

        stem = MODEL_STEMS.get(name, "")
        t_time = training_times.get(stem, training_times.get(name, None))

        rows.append({
            "Model": name,
            "Accuracy": round(acc, 4),
            "Precision (weighted)": round(prec, 4),
            "Recall (weighted)": round(rec, 4),
            "F1 (weighted)": round(f1, 4),
            "ROC-AUC": round(auc, 4) if auc is not None else "N/A",
            "Training Time (s)": round(t_time, 3) if t_time is not None else "N/A",
        })

    df = pd.DataFrame(rows).sort_values("Accuracy", ascending=False).reset_index(drop=True)
    return df, predictions


# ── evaluate K-Means (unsupervised) ─────────────────────────────────────
def evaluate_kmeans(models, X_test, y_test, training_times):
    """Map K-Means cluster labels to the majority true class, then score."""
    if "K-Means Clustering" not in models:
        return None, None

    km = models["K-Means Clustering"]
    cluster_labels = km.predict(X_test)

    # Majority-vote label mapping
    label_map = {}
    for c in np.unique(cluster_labels):
        mask = cluster_labels == c
        if mask.sum() > 0:
            label_map[c] = int(stats.mode(y_test[mask], keepdims=True).mode[0])
    mapped_preds = np.array([label_map.get(c, -1) for c in cluster_labels])

    acc = accuracy_score(y_test, mapped_preds)
    prec = precision_score(y_test, mapped_preds, average="weighted", zero_division=0)
    rec = recall_score(y_test, mapped_preds, average="weighted", zero_division=0)
    f1 = f1_score(y_test, mapped_preds, average="weighted", zero_division=0)

    stem = "kmeans"
    t_time = training_times.get(stem, training_times.get("K-Means Clustering", None))

    row = {
        "Model": "K-Means Clustering*",
        "Accuracy": round(acc, 4),
        "Precision (weighted)": round(prec, 4),
        "Recall (weighted)": round(rec, 4),
        "F1 (weighted)": round(f1, 4),
        "ROC-AUC": "N/A",
        "Training Time (s)": round(t_time, 3) if t_time is not None else "N/A",
    }
    return row, mapped_preds


# ── visualisation 1: accuracy bar chart ──────────────────────────────────
def plot_accuracy_bars(df_all, save_path):
    fig, ax = plt.subplots(figsize=(10, 6))
    colors = sns.color_palette("viridis", n_colors=len(df_all))
    bars = ax.barh(df_all["Model"], df_all["Accuracy"], color=colors, edgecolor="white")
    for bar, val in zip(bars, df_all["Accuracy"]):
        ax.text(val + 0.003, bar.get_y() + bar.get_height() / 2,
                f"{val:.3f}", va="center", fontsize=10, fontweight="bold")
    ax.set_xlabel("Accuracy", fontsize=12)
    ax.set_title("Model Comparison — Accuracy on Test Set\n(Phishing Website Detection)",
                 fontsize=14, fontweight="bold")
    ax.set_xlim(0, min(1.0, df_all["Accuracy"].max() + 0.08))
    ax.invert_yaxis()
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── visualisation 2: grouped bar chart (precision/recall/F1) ─────────────
def plot_grouped_metrics(df_all, save_path):
    models = df_all["Model"].tolist()
    metrics = ["Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
    x = np.arange(len(models))
    width = 0.25

    fig, ax = plt.subplots(figsize=(12, 6))
    palette = ["#4C72B0", "#55A868", "#C44E52"]
    for i, (metric, color) in enumerate(zip(metrics, palette)):
        vals = df_all[metric].tolist()
        ax.bar(x + i * width, vals, width, label=metric.split(" ")[0], color=color)

    ax.set_xticks(x + width)
    ax.set_xticklabels(models, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Score", fontsize=12)
    ax.set_title("Precision / Recall / F1 by Model — Phishing Detection",
                 fontsize=14, fontweight="bold")
    ax.legend(fontsize=10)
    ax.set_ylim(0, 1.05)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── visualisation 3: radar / spider chart ────────────────────────────────
def plot_radar(df_all, save_path):
    metrics = ["Accuracy", "Precision (weighted)", "Recall (weighted)", "F1 (weighted)"]
    labels = ["Accuracy", "Precision", "Recall", "F1"]
    num_vars = len(metrics)
    angles = np.linspace(0, 2 * np.pi, num_vars, endpoint=False).tolist()
    angles += angles[:1]  # close the polygon

    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    palette = sns.color_palette("husl", n_colors=len(df_all))

    for idx, row in df_all.iterrows():
        values = [row[m] for m in metrics]
        values += values[:1]
        ax.plot(angles, values, "o-", linewidth=2, label=row["Model"], color=palette[idx])
        ax.fill(angles, values, alpha=0.08, color=palette[idx])

    ax.set_thetagrids(np.degrees(angles[:-1]), labels, fontsize=11)
    ax.set_ylim(0, 1.0)
    ax.set_title("Multi-Metric Radar Comparison — Phishing Detection",
                 fontsize=14, fontweight="bold", y=1.08)
    ax.legend(loc="upper right", bbox_to_anchor=(1.35, 1.12), fontsize=8)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── visualisation 4: side-by-side confusion matrices ─────────────────────
def plot_confusion_matrices(all_predictions, y_test, save_path):
    """Plot confusion matrices for all models in a grid."""
    n = len(all_predictions)
    if n == 0:
        print("  [SKIP] No predictions to plot confusion matrices.")
        return
    cols = min(3, n)
    rows_grid = int(np.ceil(n / cols))

    classes = sorted(np.unique(y_test))
    class_labels = [CLASS_NAMES[c] for c in classes]

    fig, axes = plt.subplots(rows_grid, cols, figsize=(6 * cols, 5 * rows_grid))
    if rows_grid == 1 and cols == 1:
        axes = np.array([[axes]])
    elif rows_grid == 1:
        axes = axes[np.newaxis, :]
    elif cols == 1:
        axes = axes[:, np.newaxis]

    for idx, (name, y_pred) in enumerate(all_predictions.items()):
        r, c = divmod(idx, cols)
        cm = confusion_matrix(y_test, y_pred, labels=classes)
        sns.heatmap(
            cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=class_labels, yticklabels=class_labels,
            ax=axes[r][c], cbar=False,
        )
        axes[r][c].set_title(name, fontsize=11, fontweight="bold")
        axes[r][c].set_xlabel("Predicted")
        axes[r][c].set_ylabel("Actual")

    # Hide empty subplots
    for idx in range(n, rows_grid * cols):
        r, c = divmod(idx, cols)
        axes[r][c].axis("off")

    fig.suptitle("Confusion Matrices — All Models (Phishing Detection)",
                 fontsize=15, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved: {save_path}")


# ── statistical significance (McNemar's test) ───────────────────────────
def mcnemar_test(y_test, pred_a, pred_b, name_a, name_b):
    """
    Perform McNemar's test between two classifiers.
    Returns (chi2, p_value).
    """
    correct_a = (pred_a == y_test)
    correct_b = (pred_b == y_test)
    # Contingency: b01 = A wrong & B right, b10 = A right & B wrong
    b01 = np.sum(~correct_a & correct_b)
    b10 = np.sum(correct_a & ~correct_b)
    # McNemar's chi-squared (with continuity correction)
    if b01 + b10 == 0:
        return 0.0, 1.0
    chi2 = (abs(b01 - b10) - 1) ** 2 / (b01 + b10)
    p_value = 1 - stats.chi2.cdf(chi2, df=1)
    return chi2, p_value


def run_significance_tests(predictions, y_test):
    """Run pairwise McNemar's tests between the top models."""
    names = list(predictions.keys())
    if len(names) < 2:
        print("  [SKIP] Fewer than 2 models — no significance test possible.")
        return

    # Sort by accuracy descending
    accs = {n: accuracy_score(y_test, predictions[n]) for n in names}
    ranked = sorted(accs, key=accs.get, reverse=True)

    print("\n" + "=" * 65)
    print("Statistical Significance — McNemar's Test (top pairs)")
    print("=" * 65)
    tested = set()
    for i in range(min(4, len(ranked))):
        for j in range(i + 1, min(4, len(ranked))):
            a, b = ranked[i], ranked[j]
            key = tuple(sorted([a, b]))
            if key in tested:
                continue
            tested.add(key)
            chi2, pval = mcnemar_test(y_test, predictions[a], predictions[b], a, b)
            sig = "YES (p < 0.05)" if pval < 0.05 else "NO  (p >= 0.05)"
            print(f"  {a:30s} vs {b:30s}  chi2={chi2:8.3f}  p={pval:.4f}  Significant: {sig}")


# ── main ─────────────────────────────────────────────────────────────────
def main():
    print("=" * 65)
    print("CSAI412 — Phishing Website Detection — Model Comparison")
    print("=" * 65)

    # 1. Wait for model files
    print("\n[1/7] Waiting for trained model files...")
    wait_for_models(timeout_seconds=600, poll_interval=20)

    # 2. Load test data (same split every model used)
    print("\n[2/7] Loading test data...")
    data = get_train_test()
    X_test = data["X_test"]
    y_test = data["y_test"]

    # 3. Load all models
    print("\n[3/7] Loading trained models...")
    models = load_all_models()
    if not models:
        print("[ERROR] No models loaded. Exiting.")
        sys.exit(1)

    training_times = load_training_times()
    print(f"  Training times found for: {list(training_times.keys()) or 'none'}")

    # 4. Evaluate supervised models
    print("\n[4/7] Evaluating supervised models...")
    df_supervised, predictions = evaluate_supervised(models, X_test, y_test, training_times)
    print("\nSupervised Model Results:")
    print(df_supervised.to_string(index=False))

    # 5. Evaluate K-Means
    km_row, km_preds = evaluate_kmeans(models, X_test, y_test, training_times)
    all_predictions = dict(predictions)
    if km_row is not None:
        df_km = pd.DataFrame([km_row])
        df_all = pd.concat([df_supervised, df_km], ignore_index=True)
        all_predictions["K-Means Clustering*"] = km_preds
        print("\n(K-Means evaluated via majority-vote label mapping)")
    else:
        df_all = df_supervised.copy()

    # 6. Save comparison CSV
    csv_path = os.path.join(DATA_DIR, "comparison_results.csv")
    df_all.to_csv(csv_path, index=False)
    print(f"\n  Saved comparison results to: {csv_path}")

    # 7. Create visualisations
    print("\n[5/7] Creating comparison visualisations...")
    plot_accuracy_bars(df_all, os.path.join(FIGURES_DIR, "comparison_accuracy.png"))
    plot_grouped_metrics(df_all, os.path.join(FIGURES_DIR, "comparison_metrics.png"))
    plot_radar(df_all, os.path.join(FIGURES_DIR, "comparison_radar.png"))
    plot_confusion_matrices(all_predictions, y_test, os.path.join(FIGURES_DIR, "comparison_confusion_matrices.png"))

    # 8. Statistical significance
    print("\n[6/7] Running statistical significance tests...")
    run_significance_tests(predictions, y_test)

    # 9. Comprehensive summary
    print("\n[7/7] Comprehensive Comparison Summary")
    print("=" * 65)
    print(df_all.to_string(index=False))
    print("\n--- Ranking (by Accuracy, descending) ---")
    for i, row in df_all.iterrows():
        auc_str = f"  AUC={row['ROC-AUC']}" if row['ROC-AUC'] != "N/A" else ""
        print(f"  #{i+1}  {row['Model']:35s}  Accuracy={row['Accuracy']:.4f}  F1={row['F1 (weighted)']:.4f}{auc_str}")

    best = df_all.iloc[0]
    print(f"\n  BEST MODEL: {best['Model']}  (Accuracy={best['Accuracy']:.4f}, F1={best['F1 (weighted)']:.4f})")
    if km_row:
        print(f"\n  * K-Means results use majority-vote label mapping and are not directly")
        print(f"    comparable to supervised models. Included for reference only.")

    print("\n" + "=" * 65)
    print("Comparison complete. Outputs:")
    print(f"  CSV:   {csv_path}")
    print(f"  Plots: {FIGURES_DIR}/comparison_*.png")
    print("=" * 65)


if __name__ == "__main__":
    main()
