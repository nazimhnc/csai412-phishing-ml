"""
SVM with Linear Kernel -- Phishing Website Detection
=====================================================
CSAI412 Machine Learning Group Project

Model: Support Vector Machine with LINEAR kernel (SVC(kernel='linear'))
Dataset: UCI Phishing Websites (11,055 samples, 30 features, binary classification)
Target:  Result -- Phishing (0) vs Legitimate (1) after encoding

Hyperparameter Tuning: GridSearchCV over C values
Evaluation: Accuracy, Precision, Recall, F1 (weighted), confusion matrix, cross-validation
Visualizations: Confusion matrix heatmap, decision boundary (PCA 2D)

COMMENTARY -- Why Linear SVM for Phishing Website Detection?
--------------------------------------------------------------
Linear SVM fits a maximum-margin hyperplane that separates classes with the widest
possible gap. For phishing website detection, whose 30 URL/website features are
encoded as ternary values {-1, 0, 1}, a linear decision boundary is a natural
choice because:

  * Maximum margin classifier concept: Among all hyperplanes that correctly separate
    training points, SVM picks the one whose distance to the nearest training point
    (the "margin") is maximized. This acts as a structural regularizer and often
    leads to better generalization than simpler linear models.

  * Ternary feature space: The features are already discrete (-1, 0, 1), meaning
    the data lies on a discrete lattice in 30-dimensional space. A linear boundary
    can often effectively partition this space because phishing indicators tend to
    combine additively -- more suspicious features generally mean more likely phishing.

  STRENGTHS:
    - Effective in high-dimensional spaces (30 features is well within SVM sweet spot)
    - Memory-efficient: the decision function depends only on support vectors, not
      the full training set
    - Well-understood regularization via the C parameter
    - Fast prediction (dot product with weight vector)
    - Weight vector provides feature importance (like Logistic Regression)

  WEAKNESSES:
    - The linear assumption limits expressiveness: some phishing patterns may
      involve non-linear feature interactions
    - Training is O(n^2) to O(n^3) -- slower than tree-based methods on large sets
    - No probabilistic output by default (must enable probability=True, which is
      even slower due to Platt scaling)
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.svm import SVC
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)
from sklearn.decomposition import PCA

# ---------------------------------------------------------------------------
# Path setup -- allow running from project root or from src/models/
# ---------------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from data_loader import get_train_test

FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

warnings.filterwarnings("ignore")

# Class labels for display
CLASS_NAMES = ["Phishing", "Legitimate"]

# ===========================================================================
# 1. Load & prepare data
# ===========================================================================
print("=" * 70)
print("SVM with LINEAR Kernel -- Phishing Website Detection")
print("=" * 70)

data = get_train_test()
X_train = data["X_train"]
X_test = data["X_test"]
y_train = data["y_train"]
y_test = data["y_test"]
feature_names = data["feature_names"]

classes = sorted(np.unique(np.concatenate([y_train, y_test])))
print(f"\nClasses: {[CLASS_NAMES[c] for c in classes]}")
print(f"Training samples: {X_train.shape[0]}, Test samples: {X_test.shape[0]}")
print(f"Features: {X_train.shape[1]}")

# ===========================================================================
# 2. Hyperparameter tuning with GridSearchCV
# ===========================================================================
print("\n" + "-" * 70)
print("Hyperparameter Tuning -- GridSearchCV (5-fold CV)")
print("-" * 70)

param_grid = {
    "C": [0.001, 0.01, 0.1, 1, 10],
}

svm_linear = SVC(kernel="linear", random_state=42)

grid_search = GridSearchCV(
    estimator=svm_linear,
    param_grid=param_grid,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
    verbose=1,
    refit=True,
)

start_time = time.time()
grid_search.fit(X_train, y_train)
grid_time = time.time() - start_time

print(f"\nGrid search completed in {grid_time:.1f} seconds")
print(f"Best parameters: {grid_search.best_params_}")
print(f"Best CV F1 (weighted): {grid_search.best_score_:.4f}")

# Show all CV results
print("\nAll GridSearchCV results:")
print(f"  {'C':<10} {'Mean F1':<12} {'Std F1':<12}")
print(f"  {'-'*34}")
cv_results = grid_search.cv_results_
for i in range(len(cv_results["params"])):
    c_val = cv_results["params"][i]["C"]
    mean_f1 = cv_results["mean_test_score"][i]
    std_f1 = cv_results["std_test_score"][i]
    marker = " <-- best" if cv_results["params"][i] == grid_search.best_params_ else ""
    print(f"  {c_val:<10} {mean_f1:<12.4f} {std_f1:<12.4f}{marker}")

best_model = grid_search.best_estimator_

# ===========================================================================
# 3. Evaluation on test set
# ===========================================================================
print("\n" + "-" * 70)
print("Test Set Evaluation")
print("-" * 70)

y_pred = best_model.predict(X_test)

accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

print(f"\nAccuracy:           {accuracy:.4f}")
print(f"Precision (weighted): {precision:.4f}")
print(f"Recall (weighted):    {recall:.4f}")
print(f"F1 Score (weighted):  {f1:.4f}")

print(f"\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0))

# ===========================================================================
# 4. Confusion matrix heatmap
# ===========================================================================
print("-" * 70)
print("Generating confusion matrix heatmap...")
print("-" * 70)

cm = confusion_matrix(y_test, y_pred, labels=classes)

fig, ax = plt.subplots(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt="d",
    cmap="Blues",
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    ax=ax,
    linewidths=0.5,
    linecolor="gray",
)
ax.set_xlabel("Predicted Class", fontsize=13)
ax.set_ylabel("True Class", fontsize=13)
ax.set_title(
    f"SVM (Linear Kernel) -- Confusion Matrix\n"
    f"Phishing Detection | Accuracy: {accuracy:.4f} | F1: {f1:.4f} | C={grid_search.best_params_['C']}",
    fontsize=14,
    fontweight="bold",
)
plt.tight_layout()

cm_path = os.path.join(FIGURES_DIR, "svm_linear_confusion_matrix.png")
fig.savefig(cm_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved confusion matrix to: {cm_path}")

# ===========================================================================
# 5. Support vector analysis
# ===========================================================================
print("\n" + "-" * 70)
print("Support Vector Analysis")
print("-" * 70)

n_support = best_model.n_support_
support_classes = best_model.classes_
total_support = sum(n_support)
total_train = X_train.shape[0]

print(f"\nTotal support vectors: {total_support} / {total_train} "
      f"({total_support / total_train * 100:.1f}% of training data)")
print(f"\nSupport vectors per class:")
print(f"  {'Class':<15} {'Count':<10} {'% of SV':<10} {'% of Class Train':<18}")
print(f"  {'-'*53}")

# Count training samples per class for per-class percentages
train_class_counts = pd.Series(y_train).value_counts().sort_index()
for cls, count in zip(support_classes, n_support):
    pct_sv = count / total_support * 100
    cls_total = train_class_counts.get(cls, 1)
    pct_cls = count / cls_total * 100
    print(f"  {CLASS_NAMES[cls]:<15} {count:<10} {pct_sv:<10.1f} {pct_cls:<18.1f}")

print(f"""
COMMENTARY -- Support Vector Analysis:
  A {'high' if total_support / total_train > 0.3 else 'moderate'} percentage of training points becoming support vectors ({total_support / total_train * 100:.1f}%)
  indicates {'significant' if total_support / total_train > 0.3 else 'moderate'} class overlap in the feature space. For phishing detection,
  this reflects how similar some phishing sites are to legitimate ones in terms
  of their URL/website feature profiles. A linear boundary {'struggles to cleanly' if total_support / total_train > 0.3 else 'reasonably'} separate{'s' if total_support / total_train <= 0.3 else ''} them,
  so {'many' if total_support / total_train > 0.3 else 'some'} points end up on or inside the margin.
""")

# ===========================================================================
# 6. Cross-validation (5-fold) with best model
# ===========================================================================
print("-" * 70)
print("5-Fold Cross-Validation (full training set, best parameters)")
print("-" * 70)

cv_scores = cross_val_score(
    SVC(kernel="linear", C=grid_search.best_params_["C"], random_state=42),
    X_train,
    y_train,
    cv=5,
    scoring="f1_weighted",
    n_jobs=-1,
)

print(f"\nCV F1 scores:    {cv_scores}")
print(f"Mean CV F1:      {cv_scores.mean():.4f}")
print(f"Std CV F1:       {cv_scores.std():.4f}")
print(f"95% CI:          [{cv_scores.mean() - 1.96 * cv_scores.std():.4f}, "
      f"{cv_scores.mean() + 1.96 * cv_scores.std():.4f}]")

# Also compute cross-val accuracy
cv_acc = cross_val_score(
    SVC(kernel="linear", C=grid_search.best_params_["C"], random_state=42),
    X_train,
    y_train,
    cv=5,
    scoring="accuracy",
    n_jobs=-1,
)
print(f"\nCV Accuracy:     {cv_acc}")
print(f"Mean CV Acc:     {cv_acc.mean():.4f}")
print(f"Std CV Acc:      {cv_acc.std():.4f}")

# ===========================================================================
# 7. Decision boundary visualization (PCA 2D)
# ===========================================================================
print("\n" + "-" * 70)
print("Generating decision boundary visualization (PCA 2D)...")
print("-" * 70)

# Reduce to 2D with PCA
pca = PCA(n_components=2, random_state=42)
X_train_2d = pca.fit_transform(X_train)
X_test_2d = pca.transform(X_test)

print(f"PCA explained variance ratio: {pca.explained_variance_ratio_}")
print(f"Total variance explained: {sum(pca.explained_variance_ratio_):.4f}")

# Train a new linear SVM on 2D data for visualization
svm_2d = SVC(kernel="linear", C=grid_search.best_params_["C"], random_state=42)
svm_2d.fit(X_train_2d, y_train)

# Create mesh grid for decision boundary
h = 0.1  # Step size
x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

# Predict on mesh
Z = svm_2d.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)

# Plot
fig, axes = plt.subplots(1, 2, figsize=(18, 7))

cmap_bg = plt.cm.get_cmap("RdYlGn", len(classes))
colors = ["#d32f2f", "#388e3c"]

# Left: decision boundary with training data
ax = axes[0]
ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg, levels=np.arange(min(classes) - 0.5, max(classes) + 1.5, 1))
for i, cls in enumerate(classes):
    mask = y_train == cls
    ax.scatter(X_train_2d[mask, 0], X_train_2d[mask, 1],
               c=colors[i], label=CLASS_NAMES[cls], edgecolors="k", s=12, alpha=0.6)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
ax.set_title("Training Data -- Linear SVM Decision Boundary", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

# Right: test data on decision boundary
ax = axes[1]
ax.contourf(xx, yy, Z, alpha=0.3, cmap=cmap_bg, levels=np.arange(min(classes) - 0.5, max(classes) + 1.5, 1))
for i, cls in enumerate(classes):
    mask = y_test == cls
    ax.scatter(X_test_2d[mask, 0], X_test_2d[mask, 1],
               c=colors[i], label=CLASS_NAMES[cls], edgecolors="k", s=12, alpha=0.6)
ax.set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)", fontsize=12)
ax.set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)", fontsize=12)
ax.set_title("Test Data -- Linear SVM Decision Boundary", fontsize=13, fontweight="bold")
ax.legend(fontsize=10)

fig.suptitle(
    f"SVM (Linear Kernel) Decision Boundary in PCA Space\n"
    f"Phishing Detection | C={grid_search.best_params_['C']} | 2D Accuracy: {svm_2d.score(X_test_2d, y_test):.4f}",
    fontsize=15, fontweight="bold", y=1.02,
)
plt.tight_layout()

db_path = os.path.join(FIGURES_DIR, "svm_linear_decision_boundary.png")
fig.savefig(db_path, dpi=150, bbox_inches="tight")
plt.close(fig)
print(f"Saved decision boundary to: {db_path}")

# ===========================================================================
# 8. Save model
# ===========================================================================
print("\n" + "-" * 70)
print("Saving model...")
print("-" * 70)

model_path = os.path.join(DATA_DIR, "svm_linear_model.pkl")
joblib.dump(best_model, model_path)
print(f"Model saved to: {model_path}")

# ===========================================================================
# 9. Final summary
# ===========================================================================
print("\n" + "=" * 70)
print("SVM LINEAR KERNEL -- FINAL SUMMARY")
print("=" * 70)
print(f"""
  Best C:                {grid_search.best_params_['C']}
  Test Accuracy:         {accuracy:.4f}
  Test F1 (weighted):    {f1:.4f}
  Test Precision (w):    {precision:.4f}
  Test Recall (w):       {recall:.4f}
  CV F1 (mean +/- std):  {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}
  CV Acc (mean +/- std): {cv_acc.mean():.4f} +/- {cv_acc.std():.4f}
  Total support vectors: {total_support} / {total_train} ({total_support/total_train*100:.1f}%)
  Grid search time:      {grid_time:.1f}s

  COMMENTARY -- Linear SVM for Phishing Website Detection:
    The linear SVM provides a solid classifier for phishing detection. With
    ternary-encoded features ({'{-1, 0, 1}'}), the maximum-margin hyperplane
    effectively combines these binary/ternary indicators into a phishing score.
    The weight vector can be interpreted similarly to Logistic Regression
    coefficients, revealing which URL and website features are most indicative
    of phishing.

    The support vector percentage ({total_support/total_train*100:.1f}%) reflects the
    degree of overlap between phishing and legitimate sites in feature space.
    A non-linear kernel (RBF) may capture subtler feature interactions and
    improve performance by modeling non-linear combinations of URL properties.

  Artifacts saved:
    - Model:              {model_path}
    - Confusion matrix:   {cm_path}
    - Decision boundary:  {db_path}
""")
print("=" * 70)
print("SVM Linear Kernel -- DONE")
print("=" * 70)
