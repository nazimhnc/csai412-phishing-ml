"""
Decision Tree Classifier for Phishing Website Detection
=========================================================
CSAI412 Machine Learning Group Project

Model: sklearn.tree.DecisionTreeClassifier
Tuning: GridSearchCV with 5-fold stratified CV
Evaluation: Accuracy, Precision, Recall, F1, Confusion Matrix,
            Feature Importance, Tree Visualization, Overfitting Analysis

Dataset: UCI Phishing Websites (11,055 samples, 30 features, binary classification)
Target:  Result -- Phishing (0) vs Legitimate (1) after encoding
"""

import os
import sys
import time
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.tree import DecisionTreeClassifier, plot_tree, export_text
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# ---------------------------------------------------------------------------
# Data loading -- import from shared data_loader
# ---------------------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import get_train_test  # noqa: E402

# Paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)

warnings.filterwarnings("ignore", category=FutureWarning)

# Class labels for display
CLASS_NAMES = ["Phishing", "Legitimate"]

# ===========================================================================
# 0. COMMENTARY - Why Decision Trees?
# ===========================================================================
INTRO_COMMENTARY = """
================================================================================
   DECISION TREE CLASSIFIER FOR PHISHING WEBSITE DETECTION
================================================================================

WHY DECISION TREES ARE APPROPRIATE FOR PHISHING DETECTION
--------------------------------------------------------------
Decision Trees are a natural fit for the phishing website dataset for several
compelling reasons:

  1. INTERPRETABILITY: Unlike black-box models, Decision Trees produce
     human-readable rules. A security analyst can understand exactly why a
     website was flagged as phishing -- e.g., "if having_IP_Address = -1 and
     URL_Length = -1 and SSLfinal_State = -1, then Phishing". This is
     critical in cybersecurity where explainability is required.

  2. NO FEATURE SCALING REQUIRED: Decision Trees split on individual feature
     thresholds, so the raw scale of features does not matter. Since our
     features are already ternary ({-1, 0, 1}), this is a natural fit --
     trees will learn meaningful thresholds directly on these values.

  3. NATURAL FIT FOR BINARY/TERNARY FEATURES: The phishing features are
     encoded as {-1, 0, 1} representing suspicious, neutral, and legitimate.
     Decision Trees split on these values directly, creating rules like
     "if feature > -0.5 then ..." which naturally corresponds to "if feature
     is 0 or 1 (neutral or legitimate)". This makes the tree structure
     highly interpretable.

  4. FEATURE IMPORTANCE: Trees provide a built-in measure of feature
     importance based on impurity reduction, giving direct insight into
     which URL/website properties most effectively distinguish phishing
     from legitimate sites.

  5. NON-LINEAR DECISION BOUNDARIES: Phishing detection likely depends on
     combinations of features -- a site might look legitimate in most
     aspects but suspicious in a critical combination. Trees capture
     these naturally through hierarchical splits.

KNOWN WEAKNESSES (addressed through pruning):
  - Prone to overfitting (unpruned trees memorize training noise)
  - High variance (small data changes can produce very different trees)
  - Greedy splitting may miss globally optimal partitions
  - With binary/ternary features, splits are limited to {-1, 0, 1} thresholds

We will demonstrate these issues and show how pruning (via max_depth,
min_samples_split, min_samples_leaf) mitigates overfitting.
================================================================================
"""


def print_section(title):
    """Print a formatted section header."""
    width = 72
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


# ===========================================================================
# 1. LOAD DATA (no scaling -- trees don't need it)
# ===========================================================================
def load_phishing_data():
    """Load phishing data WITHOUT scaling (trees are scale-invariant)."""
    print_section("DATA LOADING (scale=False -- trees are scale-invariant)")
    data = get_train_test(scale=False)
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print(f"\n  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")
    print(f"  Feature names    : {feature_names}")
    print(f"  Classes          : {sorted(np.unique(y_train))} ({CLASS_NAMES})")

    return X_train, X_test, y_train, y_test, feature_names


# ===========================================================================
# 2. UNPRUNED (DEFAULT) DECISION TREE
# ===========================================================================
def train_unpruned_tree(X_train, y_train, X_test, y_test):
    """Train a fully-grown (unpruned) decision tree."""
    print_section("UNPRUNED DECISION TREE (default parameters)")

    dt_unpruned = DecisionTreeClassifier(random_state=42)
    dt_unpruned.fit(X_train, y_train)

    train_acc = accuracy_score(y_train, dt_unpruned.predict(X_train))
    test_acc = accuracy_score(y_test, dt_unpruned.predict(X_test))

    print(f"\n  Tree depth        : {dt_unpruned.get_depth()}")
    print(f"  Number of leaves  : {dt_unpruned.get_n_leaves()}")
    print(f"  Training accuracy : {train_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Overfit gap       : {train_acc - test_acc:.4f}")

    print("\n  OBSERVATION: The unpruned tree achieves near-perfect training")
    print("  accuracy but lower test accuracy, demonstrating classic overfitting.")
    print("  The tree memorizes the training data, including noise, rather than")
    print("  learning generalizable phishing detection patterns.")

    return dt_unpruned, train_acc, test_acc


# ===========================================================================
# 3. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ===========================================================================
def tune_hyperparameters(X_train, y_train):
    """Run GridSearchCV to find optimal pruning parameters."""
    print_section("HYPERPARAMETER TUNING (GridSearchCV, 5-fold stratified CV)")

    param_grid = {
        "max_depth": [None, 3, 5, 7, 10, 15, 20],
        "min_samples_split": [2, 5, 10, 20],
        "min_samples_leaf": [1, 2, 5, 10],
        "criterion": ["gini", "entropy"],
    }

    total_combos = 1
    for v in param_grid.values():
        total_combos *= len(v)
    print(f"\n  Parameter grid: {total_combos} combinations x 5 folds = {total_combos * 5} fits")
    print(f"  Searching...")

    grid_search = GridSearchCV(
        DecisionTreeClassifier(random_state=42),
        param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )

    start_time = time.time()
    grid_search.fit(X_train, y_train)
    elapsed = time.time() - start_time

    print(f"  Search completed in {elapsed:.1f} seconds")
    print(f"\n  Best parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param:<25s}: {value}")
    print(f"\n  Best CV accuracy: {grid_search.best_score_:.4f}")

    # Show top 10 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    print(f"\n  Top 10 parameter combinations:")
    print(f"  {'Rank':<6} {'Mean Test Acc':<15} {'Std':<10} {'Params'}")
    print(f"  {'-'*80}")
    for i, row in results_df.head(10).iterrows():
        params_str = ", ".join(f"{k.split('__')[-1]}={v}" for k, v in row["params"].items())
        print(f"  {row['rank_test_score']:<6.0f} {row['mean_test_score']:<15.4f} {row['std_test_score']:<10.4f} {params_str}")

    # Gini vs Entropy comparison
    print_section("GINI vs ENTROPY COMPARISON")
    gini_mask = results_df["param_criterion"] == "gini"
    entropy_mask = results_df["param_criterion"] == "entropy"
    gini_mean = results_df.loc[gini_mask, "mean_test_score"].mean()
    entropy_mean = results_df.loc[entropy_mask, "mean_test_score"].mean()
    gini_best = results_df.loc[gini_mask, "mean_test_score"].max()
    entropy_best = results_df.loc[entropy_mask, "mean_test_score"].max()

    print(f"\n  {'Criterion':<12} {'Mean CV Acc':<15} {'Best CV Acc':<15}")
    print(f"  {'-'*42}")
    print(f"  {'Gini':<12} {gini_mean:<15.4f} {gini_best:<15.4f}")
    print(f"  {'Entropy':<12} {entropy_mean:<15.4f} {entropy_best:<15.4f}")
    print(f"\n  ANALYSIS: For phishing detection with ternary features, Gini and")
    print(f"  Entropy typically produce very similar results. The ternary feature")
    print(f"  space provides clear split points at -0.5 and 0.5, making the choice")
    print(f"  of impurity criterion less impactful than the pruning parameters.")

    return grid_search


# ===========================================================================
# 4. TRAIN PRUNED (BEST) TREE
# ===========================================================================
def train_pruned_tree(grid_search, X_train, y_train, X_test, y_test):
    """Extract the best model and evaluate on the test set."""
    print_section("PRUNED DECISION TREE (best parameters from GridSearchCV)")

    best_tree = grid_search.best_estimator_

    train_acc = accuracy_score(y_train, best_tree.predict(X_train))
    test_acc = accuracy_score(y_test, best_tree.predict(X_test))

    print(f"\n  Best parameters   : {grid_search.best_params_}")
    print(f"  Tree depth        : {best_tree.get_depth()}")
    print(f"  Number of leaves  : {best_tree.get_n_leaves()}")
    print(f"  Training accuracy : {train_acc:.4f}")
    print(f"  Test accuracy     : {test_acc:.4f}")
    print(f"  Overfit gap       : {train_acc - test_acc:.4f}")

    return best_tree, train_acc, test_acc


# ===========================================================================
# 5. EVALUATION METRICS
# ===========================================================================
def evaluate_model(model, X_test, y_test, model_name="Decision Tree"):
    """Compute and print full evaluation metrics."""
    print_section(f"EVALUATION METRICS -- {model_name}")

    y_pred = model.predict(X_test)
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Accuracy  (overall)  : {acc:.4f}")
    print(f"  Precision (weighted) : {prec:.4f}")
    print(f"  Recall    (weighted) : {rec:.4f}")
    print(f"  F1-score  (weighted) : {f1:.4f}")

    print(f"\n  Full Classification Report:")
    print(classification_report(y_test, y_pred,
                                target_names=CLASS_NAMES, zero_division=0))

    return y_pred, classes, {"accuracy": acc, "precision": prec, "recall": rec, "f1": f1}


# ===========================================================================
# 6. CONFUSION MATRIX HEATMAP
# ===========================================================================
def plot_confusion_matrix(y_test, y_pred, classes):
    """Generate and save confusion matrix heatmap."""
    print_section("CONFUSION MATRIX")

    cm = confusion_matrix(y_test, y_pred, labels=classes)

    fig, axes = plt.subplots(1, 2, figsize=(16, 6))

    # Raw counts
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[0],
    )
    axes[0].set_xlabel("Predicted Class", fontsize=12)
    axes[0].set_ylabel("Actual Class", fontsize=12)
    axes[0].set_title("Confusion Matrix (Counts)", fontsize=14)

    # Normalized (per row)
    cm_norm = cm.astype("float") / cm.sum(axis=1, keepdims=True)
    cm_norm = np.nan_to_num(cm_norm)
    sns.heatmap(
        cm_norm,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=CLASS_NAMES,
        yticklabels=CLASS_NAMES,
        ax=axes[1],
    )
    axes[1].set_xlabel("Predicted Class", fontsize=12)
    axes[1].set_ylabel("Actual Class", fontsize=12)
    axes[1].set_title("Confusion Matrix (Normalized)", fontsize=14)

    plt.suptitle("Decision Tree -- Phishing Detection Confusion Matrix", fontsize=16, y=1.02)
    plt.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "decision_tree_confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to: {save_path}")

    # Print notable observations
    print(f"\n  Key observations from confusion matrix:")
    for i, actual in enumerate(classes):
        row_total = cm[i].sum()
        if row_total == 0:
            continue
        correct = cm[i, i]
        pct = correct / row_total * 100
        wrong = row_total - correct
        if wrong > 0:
            other_idx = 1 - i
            print(f"    - {CLASS_NAMES[actual]}: {pct:.1f}% correct ({correct}/{row_total}), "
                  f"{wrong} misclassified as {CLASS_NAMES[classes[other_idx]]}")


# ===========================================================================
# 7. CROSS-VALIDATION (5-fold)
# ===========================================================================
def run_cross_validation(model, X_train, y_train):
    """Run 5-fold stratified cross-validation on the best model."""
    print_section("5-FOLD CROSS-VALIDATION (on training data)")

    scoring_metrics = ["accuracy", "precision_weighted", "recall_weighted", "f1_weighted"]
    results = {}

    for metric in scoring_metrics:
        scores = cross_val_score(model, X_train, y_train, cv=5, scoring=metric, n_jobs=-1)
        metric_short = metric.replace("_weighted", "")
        results[metric_short] = scores
        print(f"\n  {metric_short.capitalize():<12}:")
        print(f"    Fold scores : {', '.join(f'{s:.4f}' for s in scores)}")
        print(f"    Mean        : {scores.mean():.4f}")
        print(f"    Std         : {scores.std():.4f}")
        print(f"    95% CI      : [{scores.mean() - 1.96*scores.std():.4f}, {scores.mean() + 1.96*scores.std():.4f}]")

    print(f"\n  INTERPRETATION: The cross-validation provides a robust estimate")
    print(f"  of generalization performance. Low standard deviation across folds")
    print(f"  indicates stable performance; high std suggests sensitivity to the")
    print(f"  particular data split.")

    return results


# ===========================================================================
# 8. FEATURE IMPORTANCE
# ===========================================================================
def plot_feature_importance(model, feature_names):
    """Generate and save feature importance bar chart."""
    print_section("FEATURE IMPORTANCE ANALYSIS")

    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    print(f"\n  {'Rank':<6} {'Feature':<35} {'Importance':<12}")
    print(f"  {'-'*53}")
    for rank, idx in enumerate(indices, 1):
        bar = "#" * int(importances[idx] * 50)
        print(f"  {rank:<6} {feature_names[idx]:<35} {importances[idx]:<12.4f} {bar}")

    # Plot
    fig, ax = plt.subplots(figsize=(12, 9))
    sorted_features = [feature_names[i] for i in indices]
    sorted_importances = importances[indices]

    colors = plt.cm.viridis(np.linspace(0.3, 0.9, len(sorted_features)))
    bars = ax.barh(range(len(sorted_features)), sorted_importances, color=colors)

    ax.set_yticks(range(len(sorted_features)))
    ax.set_yticklabels(sorted_features, fontsize=10)
    ax.set_xlabel("Feature Importance (Gini / Entropy Reduction)", fontsize=12)
    ax.set_title("Decision Tree -- Feature Importance for Phishing Detection", fontsize=14)
    ax.invert_yaxis()

    # Add value labels on bars
    for bar, val in zip(bars, sorted_importances):
        if val > 0.005:
            ax.text(
                bar.get_width() + 0.003,
                bar.get_y() + bar.get_height() / 2,
                f"{val:.4f}",
                va="center",
                fontsize=9,
            )

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "decision_tree_feature_importance.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved to: {save_path}")

    # Commentary on top features
    top3 = [(feature_names[indices[i]], importances[indices[i]]) for i in range(min(3, len(indices)))]
    print(f"\n  TOP FEATURES COMMENTARY:")
    print(f"  The three most important features for detecting phishing websites are:")
    for i, (feat, imp) in enumerate(top3, 1):
        print(f"    {i}. {feat} (importance = {imp:.4f})")
    print(f"\n  These features represent the URL/website properties that most")
    print(f"  effectively separate phishing from legitimate websites. The tree")
    print(f"  uses these features at the top levels where splits create the")
    print(f"  largest information gain / Gini reduction.")

    return importances, indices


# ===========================================================================
# 9. TREE VISUALIZATION
# ===========================================================================
def plot_tree_structure(model, feature_names, class_names):
    """Visualize the decision tree (limited depth for readability)."""
    print_section("TREE STRUCTURE VISUALIZATION (max_depth=3 for readability)")

    fig, ax = plt.subplots(figsize=(28, 14))
    plot_tree(
        model,
        max_depth=3,
        feature_names=feature_names,
        class_names=CLASS_NAMES,
        filled=True,
        rounded=True,
        fontsize=8,
        proportion=True,
        impurity=True,
        ax=ax,
    )
    ax.set_title(
        "Decision Tree Structure (showing top 3 levels)\n"
        "Each node shows: split condition, impurity, sample proportion, predicted class",
        fontsize=14,
    )

    save_path = os.path.join(FIGURES_DIR, "decision_tree_structure.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to: {save_path}")

    # Also print text representation of top levels
    print(f"\n  Text representation of the tree (top 3 levels):")
    tree_text = export_text(model, feature_names=feature_names, max_depth=3)
    for line in tree_text.split("\n")[:30]:
        print(f"    {line}")
    if len(tree_text.split("\n")) > 30:
        print(f"    ... (truncated for readability)")

    print(f"\n  INTERPRETABILITY ADVANTAGE:")
    print(f"  Unlike Logistic Regression (coefficient interpretation), KNN")
    print(f"  (no explicit model), or SVM (complex hyperplanes), a Decision")
    print(f"  Tree produces human-readable IF-THEN rules. A security analyst")
    print(f"  can follow the tree to understand why a website was flagged as")
    print(f"  phishing, and which URL/website properties drive the distinction.")
    print(f"  This is critical for cybersecurity applications where")
    print(f"  explainability is required for incident response and policy making.")


# ===========================================================================
# 10. OVERFITTING ANALYSIS
# ===========================================================================
def plot_overfitting_analysis(X_train, y_train, X_test, y_test):
    """Plot train vs test accuracy as a function of max_depth."""
    print_section("OVERFITTING ANALYSIS (train vs test accuracy by max_depth)")

    depths = list(range(1, 31))
    train_accs = []
    test_accs = []

    for d in depths:
        dt = DecisionTreeClassifier(max_depth=d, random_state=42)
        dt.fit(X_train, y_train)
        train_accs.append(accuracy_score(y_train, dt.predict(X_train)))
        test_accs.append(accuracy_score(y_test, dt.predict(X_test)))

    # Also add unlimited depth
    dt_full = DecisionTreeClassifier(max_depth=None, random_state=42)
    dt_full.fit(X_train, y_train)
    full_train_acc = accuracy_score(y_train, dt_full.predict(X_train))
    full_test_acc = accuracy_score(y_test, dt_full.predict(X_test))

    # Find optimal depth
    best_depth = depths[np.argmax(test_accs)]
    best_test_acc = max(test_accs)

    fig, ax = plt.subplots(figsize=(12, 7))
    ax.plot(depths, train_accs, "b-o", label="Training Accuracy", markersize=5, linewidth=2)
    ax.plot(depths, test_accs, "r-s", label="Test Accuracy", markersize=5, linewidth=2)
    ax.axvline(x=best_depth, color="green", linestyle="--", alpha=0.7, label=f"Best depth = {best_depth}")
    ax.axhline(y=full_test_acc, color="gray", linestyle=":", alpha=0.5, label=f"Unlimited depth test acc = {full_test_acc:.4f}")

    # Shade overfitting region
    ax.fill_between(depths, train_accs, test_accs, alpha=0.15, color="red", label="Overfitting gap")

    ax.set_xlabel("Max Depth", fontsize=12)
    ax.set_ylabel("Accuracy", fontsize=12)
    ax.set_title("Decision Tree -- Overfitting Analysis\n(Phishing Detection: Train vs Test Accuracy by Max Depth)", fontsize=14)
    ax.legend(fontsize=10, loc="center right")
    ax.set_xticks(depths[::2])
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(FIGURES_DIR, "decision_tree_overfitting.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"  Saved to: {save_path}")

    # Print analysis
    print(f"\n  {'Depth':<8} {'Train Acc':<12} {'Test Acc':<12} {'Gap':<10}")
    print(f"  {'-'*42}")
    for d, tr, te in zip(depths, train_accs, test_accs):
        marker = " <-- best test" if d == best_depth else ""
        print(f"  {d:<8} {tr:<12.4f} {te:<12.4f} {tr-te:<10.4f}{marker}")
    print(f"  {'None':<8} {full_train_acc:<12.4f} {full_test_acc:<12.4f} {full_train_acc-full_test_acc:<10.4f} (unlimited)")

    print(f"\n  OVERFITTING DISCUSSION:")
    print(f"  - At depth 1-3: UNDERFITTING -- both train and test accuracy are lower.")
    print(f"    The tree is too simple to capture the complexity of phishing patterns.")
    print(f"  - At depth {best_depth}: OPTIMAL -- best generalization (test acc = {best_test_acc:.4f}).")
    print(f"    The tree captures meaningful patterns without memorizing noise.")
    print(f"  - At depth 20+: OVERFITTING -- training accuracy approaches 1.0 but")
    print(f"    test accuracy plateaus or decreases. The tree memorizes training")
    print(f"    noise, creating brittle rules that don't generalize.")
    print(f"  - Unlimited depth: train acc = {full_train_acc:.4f}, test acc = {full_test_acc:.4f}")
    print(f"    (gap = {full_train_acc - full_test_acc:.4f}) -- clear overfitting.")
    print(f"\n  This demonstrates why pruning is essential for Decision Trees.")

    return depths, train_accs, test_accs, best_depth


# ===========================================================================
# 11. SAVE MODEL
# ===========================================================================
def save_model(model, grid_search):
    """Save the best model to disk."""
    print_section("SAVING MODEL")

    model_path = os.path.join(DATA_DIR, "decision_tree_model.pkl")
    joblib.dump(model, model_path)
    print(f"  Model saved to: {model_path}")
    print(f"  Best parameters: {grid_search.best_params_}")

    # Also save grid search results for later comparison
    results_path = os.path.join(DATA_DIR, "decision_tree_grid_results.pkl")
    joblib.dump(grid_search, results_path)
    print(f"  GridSearchCV results saved to: {results_path}")


# ===========================================================================
# 12. FINAL COMMENTARY & MODEL COMPARISON
# ===========================================================================
def print_final_commentary(unpruned_acc, pruned_acc, metrics, cv_results):
    """Print final analysis and comparison commentary."""
    print_section("FINAL COMMENTARY AND MODEL COMPARISON")

    print(f"""
  UNPRUNED vs PRUNED PERFORMANCE
  --------------------------------
  Unpruned test accuracy : {unpruned_acc['test']:.4f}  (train: {unpruned_acc['train']:.4f}, gap: {unpruned_acc['train']-unpruned_acc['test']:.4f})
  Pruned test accuracy   : {pruned_acc['test']:.4f}  (train: {pruned_acc['train']:.4f}, gap: {pruned_acc['train']-pruned_acc['test']:.4f})
  Improvement            : {pruned_acc['test'] - unpruned_acc['test']:+.4f}

  Pruning reduces the overfit gap while maintaining (or improving) test
  accuracy. The pruned tree is simpler, faster, and more reliable on
  unseen phishing data.

  STRENGTHS OF DECISION TREES FOR PHISHING DETECTION
  -----------------------------------------------------
  + Fully interpretable -- produces human-readable security rules
  + No feature scaling needed (scale-invariant splits on ternary features)
  + Natural fit for binary/ternary features (clear split thresholds)
  + Handles non-linear relationships via hierarchical partitioning
  + Built-in feature importance via impurity reduction
  + Fast training and prediction

  WEAKNESSES OF DECISION TREES
  -------------------------------
  - Prone to overfitting without careful pruning
  - High variance: small changes in data can produce very different trees
  - Greedy algorithm may miss globally optimal splits
  - Axis-aligned splits: can't capture diagonal decision boundaries efficiently
  - Individual trees are weaker learners compared to ensembles

  COMPARISON WITH OTHER MODELS (for project report)
  ---------------------------------------------------
  | Model                | Key Advantage             | Key Disadvantage           |
  |----------------------|---------------------------|----------------------------|
  | Logistic Regression  | Linear baseline, fast     | Assumes linear boundaries  |
  | KNN                  | Non-parametric, simple    | Slow prediction, no model  |
  | SVM                  | Strong with kernels       | Slow training, black-box   |
  | Decision Tree        | Interpretable rules       | Overfits, high variance    |

  Decision Trees serve as the foundation for more powerful ensemble methods
  (Random Forest, Gradient Boosting) which address the variance issue by
  combining many trees. Even alone, a well-pruned tree provides competitive
  accuracy with unmatched interpretability -- particularly valuable in
  cybersecurity where explaining WHY a site is flagged as phishing is
  critical for incident response.

  CROSS-VALIDATION STABILITY
  ----------------------------
  5-fold CV accuracy: {cv_results['accuracy'].mean():.4f} +/- {cv_results['accuracy'].std():.4f}
  This {'shows stable' if cv_results['accuracy'].std() < 0.02 else 'indicates some variability in'} performance across folds.
""")


# ===========================================================================
# MAIN
# ===========================================================================
def main():
    print(INTRO_COMMENTARY)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_phishing_data()
    classes = sorted(np.unique(np.concatenate([y_train, y_test])))

    # 2. Unpruned tree
    dt_unpruned, unpruned_train_acc, unpruned_test_acc = train_unpruned_tree(
        X_train, y_train, X_test, y_test
    )

    # 3. Hyperparameter tuning
    grid_search = tune_hyperparameters(X_train, y_train)

    # 4. Pruned (best) tree
    best_tree, pruned_train_acc, pruned_test_acc = train_pruned_tree(
        grid_search, X_train, y_train, X_test, y_test
    )

    # 5. Evaluation metrics (on best tree)
    y_pred, pred_classes, metrics = evaluate_model(best_tree, X_test, y_test, "Pruned Decision Tree")

    # 6. Confusion matrix
    plot_confusion_matrix(y_test, y_pred, classes)

    # 7. Cross-validation
    cv_results = run_cross_validation(best_tree, X_train, y_train)

    # 8. Feature importance
    plot_feature_importance(best_tree, feature_names)

    # 9. Tree visualization
    plot_tree_structure(best_tree, feature_names, classes)

    # 10. Overfitting analysis
    plot_overfitting_analysis(X_train, y_train, X_test, y_test)

    # 11. Save model
    save_model(best_tree, grid_search)

    # 12. Final commentary
    print_final_commentary(
        {"train": unpruned_train_acc, "test": unpruned_test_acc},
        {"train": pruned_train_acc, "test": pruned_test_acc},
        metrics,
        cv_results,
    )

    # Also evaluate unpruned tree for comparison
    print_section("UNPRUNED TREE EVALUATION (for comparison)")
    evaluate_model(dt_unpruned, X_test, y_test, "Unpruned Decision Tree")

    print_section("ALL DONE")
    print(f"  Figures saved in: {FIGURES_DIR}/")
    print(f"    - decision_tree_confusion_matrix.png")
    print(f"    - decision_tree_feature_importance.png")
    print(f"    - decision_tree_structure.png")
    print(f"    - decision_tree_overfitting.png")
    print(f"  Model saved in: {DATA_DIR}/decision_tree_model.pkl")


if __name__ == "__main__":
    main()
