"""
K-Nearest Neighbours (KNN) Classifier for Phishing Website Detection
=====================================================================
CSAI412 Machine Learning Group Project

Model:   K-Nearest Neighbours (scikit-learn KNeighborsClassifier)
Dataset: UCI Phishing Websites (11,055 samples, 30 features)
Target:  Result -- Phishing (0) vs Legitimate (1) after encoding

This script performs:
  1. Hyperparameter tuning via GridSearchCV
  2. k vs accuracy curve
  3. Full evaluation (accuracy, precision, recall, F1, confusion matrix)
  4. 5-fold cross-validation
  5. PCA-based decision boundary visualization
  6. Model persistence with joblib
  7. Detailed commentary on model suitability
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving figures
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.neighbors import KNeighborsClassifier
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

# -- Data Import ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import get_train_test  # noqa: E402

# -- Paths ---------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Suppress convergence/future warnings for cleaner output
warnings.filterwarnings("ignore")

# Reproducibility
RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# Class labels for display
CLASS_NAMES = ["Phishing", "Legitimate"]


# =============================================================================
#  HELPER FUNCTIONS
# =============================================================================

def print_header(title: str) -> None:
    """Print a formatted section header."""
    width = 70
    print("\n" + "=" * width)
    print(f"  {title}")
    print("=" * width)


def print_subheader(title: str) -> None:
    """Print a formatted sub-section header."""
    print(f"\n--- {title} ---")


# =============================================================================
#  1. LOAD DATA
# =============================================================================

def load_data():
    """Load and return the train/test split from data_loader."""
    print_header("1. LOADING PHISHING WEBSITE DATASET")
    data = get_train_test()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print(f"\n  Training samples : {X_train.shape[0]}")
    print(f"  Test samples     : {X_test.shape[0]}")
    print(f"  Features         : {X_train.shape[1]}")
    print(f"  Feature names    : {feature_names}")
    print(f"  Target classes   : {sorted(np.unique(y_train))}")

    return X_train, X_test, y_train, y_test, feature_names


# =============================================================================
#  2. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# =============================================================================

def tune_hyperparameters(X_train, y_train):
    """Run GridSearchCV to find optimal KNN hyperparameters."""
    print_header("2. HYPERPARAMETER TUNING (GridSearchCV)")

    param_grid = {
        "n_neighbors": [1, 3, 5, 7, 9, 11, 15, 21],
        "weights": ["uniform", "distance"],
        "metric": ["euclidean", "manhattan"],
    }

    total_combos = (
        len(param_grid["n_neighbors"])
        * len(param_grid["weights"])
        * len(param_grid["metric"])
    )
    print(f"\n  Parameter grid:")
    print(f"    n_neighbors : {param_grid['n_neighbors']}")
    print(f"    weights     : {param_grid['weights']}")
    print(f"    metric      : {param_grid['metric']}")
    print(f"    Total combinations : {total_combos}")
    print(f"    CV folds           : 5")
    print(f"    Total fits         : {total_combos * 5}")
    print(f"\n  Running GridSearchCV... (this may take a minute)")

    knn = KNeighborsClassifier()
    grid_search = GridSearchCV(
        estimator=knn,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=0,
        return_train_score=True,
    )
    grid_search.fit(X_train, y_train)

    print(f"\n  Best parameters found:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param:<15}: {value}")
    print(f"\n  Best cross-validation accuracy: {grid_search.best_score_:.4f}")

    # Show top 10 parameter combinations
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    print(f"\n  Top 10 parameter combinations:")
    print(f"  {'Rank':<6} {'k':<5} {'Weight':<10} {'Metric':<12} {'Mean CV Acc':<14} {'Std':<8}")
    print(f"  {'-'*55}")
    for _, row in results_df.head(10).iterrows():
        print(
            f"  {int(row['rank_test_score']):<6} "
            f"{row['param_n_neighbors']:<5} "
            f"{row['param_weights']:<10} "
            f"{row['param_metric']:<12} "
            f"{row['mean_test_score']:<14.4f} "
            f"{row['std_test_score']:<8.4f}"
        )

    return grid_search, grid_search.best_estimator_


# =============================================================================
#  3. K VS ACCURACY CURVE
# =============================================================================

def plot_k_vs_accuracy(X_train, y_train, X_test, y_test, best_params):
    """Plot how accuracy changes with different values of k."""
    print_header("3. K VS ACCURACY CURVE")

    k_values = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 25, 31]
    train_accuracies = []
    test_accuracies = []

    best_weight = best_params.get("weights", "distance")
    best_metric = best_params.get("metric", "euclidean")

    print(f"\n  Using weights='{best_weight}', metric='{best_metric}' (from best params)")
    print(f"\n  {'k':<6} {'Train Acc':<12} {'Test Acc':<12}")
    print(f"  {'-'*30}")

    for k in k_values:
        knn = KNeighborsClassifier(
            n_neighbors=k, weights=best_weight, metric=best_metric
        )
        knn.fit(X_train, y_train)
        train_acc = accuracy_score(y_train, knn.predict(X_train))
        test_acc = accuracy_score(y_test, knn.predict(X_test))
        train_accuracies.append(train_acc)
        test_accuracies.append(test_acc)
        print(f"  {k:<6} {train_acc:<12.4f} {test_acc:<12.4f}")

    # Plot
    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(k_values, train_accuracies, "o-", label="Training Accuracy", color="#2196F3", linewidth=2, markersize=7)
    ax.plot(k_values, test_accuracies, "s-", label="Test Accuracy", color="#FF5722", linewidth=2, markersize=7)

    # Mark the best k from grid search
    best_k = best_params.get("n_neighbors", 5)
    if best_k in k_values:
        idx = k_values.index(best_k)
        ax.axvline(x=best_k, color="green", linestyle="--", alpha=0.7, label=f"Best k={best_k}")
        ax.scatter([best_k], [test_accuracies[idx]], color="green", s=150, zorder=5, edgecolors="black", linewidth=2)

    ax.set_xlabel("Number of Neighbors (k)", fontsize=13)
    ax.set_ylabel("Accuracy", fontsize=13)
    ax.set_title("KNN: k vs Accuracy (Phishing Detection)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=11)
    ax.set_xticks(k_values)
    ax.grid(True, alpha=0.3)
    fig.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "knn_k_vs_accuracy.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved k vs accuracy plot to: {save_path}")

    return k_values, train_accuracies, test_accuracies


# =============================================================================
#  4. EVALUATION ON TEST SET
# =============================================================================

def evaluate_model(best_model, X_train, X_test, y_train, y_test):
    """Evaluate the best KNN model on the test set."""
    print_header("4. MODEL EVALUATION ON TEST SET")

    y_pred = best_model.predict(X_test)

    # Core metrics (weighted for consistency with comparison script)
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    rec = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Best model parameters:")
    print(f"    n_neighbors : {best_model.n_neighbors}")
    print(f"    weights     : {best_model.weights}")
    print(f"    metric      : {best_model.metric}")

    print(f"\n  Test Set Metrics (weighted averages):")
    print(f"    Accuracy  : {acc:.4f}  ({acc*100:.2f}%)")
    print(f"    Precision : {prec:.4f}")
    print(f"    Recall    : {rec:.4f}")
    print(f"    F1-Score  : {f1:.4f}")

    # Training accuracy for comparison
    y_train_pred = best_model.predict(X_train)
    train_acc = accuracy_score(y_train, y_train_pred)
    print(f"\n    Training Accuracy : {train_acc:.4f}  ({train_acc*100:.2f}%)")
    overfit_gap = train_acc - acc
    print(f"    Overfit Gap       : {overfit_gap:.4f}  (train - test)")

    # Full classification report
    print_subheader("Full Classification Report")
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    report = classification_report(y_test, y_pred,
                                   target_names=[CLASS_NAMES[c] for c in classes],
                                   zero_division=0)
    print(report)

    # Confusion Matrix
    print_subheader("Confusion Matrix")
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"\n  Classes: {[CLASS_NAMES[c] for c in classes]}")
    print(f"\n  {'':<15}", end="")
    for c in classes:
        print(f"Pred {CLASS_NAMES[c]:<12}", end=" ")
    print()
    for i, c in enumerate(classes):
        print(f"  Actual {CLASS_NAMES[c]:<7}", end=" ")
        for j in range(len(classes)):
            print(f"{cm[i][j]:<17}", end=" ")
        print()

    # Confusion matrix heatmap
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
    ax.set_ylabel("Actual Class", fontsize=13)
    ax.set_title("KNN Confusion Matrix -- Phishing Website Detection", fontsize=15, fontweight="bold")
    fig.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "knn_confusion_matrix.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved confusion matrix heatmap to: {save_path}")

    return acc, prec, rec, f1, y_pred


# =============================================================================
#  5. CROSS-VALIDATION (5-FOLD)
# =============================================================================

def run_cross_validation(best_model, X_train, y_train):
    """Run 5-fold cross-validation and report results."""
    print_header("5. 5-FOLD CROSS-VALIDATION")

    cv_scores = cross_val_score(
        best_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1
    )

    print(f"\n  Model: KNeighborsClassifier(n_neighbors={best_model.n_neighbors}, "
          f"weights='{best_model.weights}', metric='{best_model.metric}')")
    print(f"\n  Fold-by-fold accuracy:")
    for i, score in enumerate(cv_scores, 1):
        bar = "#" * int(score * 50)
        print(f"    Fold {i}: {score:.4f}  |{bar}|")
    print(f"\n  Mean CV Accuracy : {cv_scores.mean():.4f}")
    print(f"  Std CV Accuracy  : {cv_scores.std():.4f}")
    print(f"  95% CI           : [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
          f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

    return cv_scores


# =============================================================================
#  6. DECISION BOUNDARY VISUALIZATION (PCA 2D)
# =============================================================================

def plot_decision_boundary(best_model, X_train, y_train, X_test, y_test):
    """Visualize decision boundaries using PCA-reduced 2D features."""
    print_header("6. DECISION BOUNDARY VISUALIZATION (PCA-2D)")

    # Reduce to 2D with PCA
    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    X_train_2d = pca.fit_transform(X_train)
    X_test_2d = pca.transform(X_test)

    explained = pca.explained_variance_ratio_
    print(f"\n  PCA variance explained: PC1={explained[0]:.4f}, PC2={explained[1]:.4f}")
    print(f"  Total variance explained: {sum(explained):.4f} ({sum(explained)*100:.1f}%)")

    # Fit a KNN on the 2D data for boundary visualization
    knn_2d = KNeighborsClassifier(
        n_neighbors=best_model.n_neighbors,
        weights=best_model.weights,
        metric=best_model.metric,
    )
    knn_2d.fit(X_train_2d, y_train)

    acc_2d = accuracy_score(y_test, knn_2d.predict(X_test_2d))
    print(f"  Accuracy on 2D-projected test data: {acc_2d:.4f} (lower than full-dimensional is expected)")

    # Create mesh grid
    h = 0.3  # Step size -- coarser for speed
    x_min, x_max = X_train_2d[:, 0].min() - 1, X_train_2d[:, 0].max() + 1
    y_min, y_max = X_train_2d[:, 1].min() - 1, X_train_2d[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h), np.arange(y_min, y_max, h))

    print(f"  Generating decision boundary mesh ({xx.shape[0]}x{xx.shape[1]} grid)...")
    Z = knn_2d.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = Z.reshape(xx.shape)

    # Plot
    classes = sorted(np.unique(y_train))
    n_classes = len(classes)
    colors = ["#d32f2f", "#388e3c"]  # Red for phishing, green for legitimate

    fig, ax = plt.subplots(figsize=(10, 8))

    # Decision regions
    cmap_bg = plt.cm.get_cmap("RdYlGn", n_classes)
    ax.contourf(xx, yy, Z, alpha=0.25, cmap=cmap_bg, levels=np.arange(min(classes) - 0.5, max(classes) + 1.5, 1))
    ax.contour(xx, yy, Z, colors="gray", linewidths=0.5, alpha=0.5)

    # Plot test points
    for i, cls in enumerate(classes):
        mask = y_test == cls
        ax.scatter(
            X_test_2d[mask, 0],
            X_test_2d[mask, 1],
            c=colors[i],
            label=CLASS_NAMES[cls],
            edgecolors="black",
            linewidth=0.5,
            s=30,
            alpha=0.75,
        )

    ax.set_xlabel(f"PC1 ({explained[0]*100:.1f}% variance)", fontsize=13)
    ax.set_ylabel(f"PC2 ({explained[1]*100:.1f}% variance)", fontsize=13)
    ax.set_title(
        f"KNN Decision Boundary (k={best_model.n_neighbors}, PCA 2D)\n"
        f"Phishing Detection | 2D Test Accuracy: {acc_2d:.4f}",
        fontsize=14,
        fontweight="bold",
    )
    ax.legend(fontsize=11, loc="best", markerscale=1.5)
    ax.grid(True, alpha=0.2)
    fig.tight_layout()

    save_path = os.path.join(FIGURES_DIR, "knn_decision_boundary.png")
    fig.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n  Saved decision boundary plot to: {save_path}")


# =============================================================================
#  7. SAVE MODEL
# =============================================================================

def save_model(best_model):
    """Save the best KNN model to disk using joblib."""
    print_header("7. SAVING BEST MODEL")

    model_path = os.path.join(DATA_DIR, "knn_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  Saved best KNN model to: {model_path}")
    print(f"  Model: KNeighborsClassifier(n_neighbors={best_model.n_neighbors}, "
          f"weights='{best_model.weights}', metric='{best_model.metric}')")

    # Verify by reloading
    loaded = joblib.load(model_path)
    print(f"  Verification: model reloaded successfully (type={type(loaded).__name__})")

    return model_path


# =============================================================================
#  8. COMMENTARY -- WHY KNN FOR PHISHING DETECTION?
# =============================================================================

def print_commentary(acc, cv_scores):
    """Print detailed commentary on model choice, strengths, and weaknesses."""
    print_header("8. COMMENTARY -- KNN FOR PHISHING WEBSITE DETECTION")

    commentary = """
  WHY KNN IS APPROPRIATE FOR THIS DATASET
  ----------------------------------------
  K-Nearest Neighbours is a reasonable choice for phishing website detection
  for several reasons:

  1) Non-parametric and instance-based: KNN makes NO assumptions about the
     underlying data distribution. Phishing detection involves 30 URL and
     website features encoded as ternary values {-1, 0, 1}. The decision
     boundary between phishing and legitimate sites may be complex and
     non-linear, and KNN naturally captures these patterns by looking at
     the actual neighborhood of each data point in feature space.

  2) What "instance-based" means here: KNN stores the entire training set
     and classifies new websites by finding the k most similar websites in
     the training data and voting on the class label. For phishing detection,
     this is intuitive: a new website's classification is predicted by
     finding websites with similar URL/page characteristics and seeing how
     they were classified.

  3) Binary classification naturally: The phishing target has 2 classes
     (Phishing vs Legitimate). KNN handles binary classification natively
     without any modification.


  THE CURSE OF DIMENSIONALITY WITH 30 FEATURES
  -----------------------------------------------
  The dataset has 30 features, which is moderate-to-high. The "curse of
  dimensionality" states that as dimensionality increases, data becomes
  sparse -- distances between points converge, making nearest-neighbor
  lookups less meaningful.

  With 30 features and ~11,055 samples, we are in a regime where:
  - The curse is present and may impact performance. 11,055 samples in 30
    dimensions is less dense than ideal for nearest-neighbor methods.
  - However, the features are TERNARY ({-1, 0, 1}), which limits the
    effective volume of the feature space compared to continuous features.
    This partially mitigates the curse of dimensionality.
  - Feature scaling (StandardScaler) is less critical here than with
    continuous features, since all features are already on the same scale
    ({-1, 0, 1}). However, scaling can still help distance-based methods
    if the data loader applies it.
  - Not all 30 features are equally informative for phishing detection. KNN
    weights all features equally in the distance computation, making it
    sensitive to irrelevant or noisy features.


  DISTANCE METRICS: EUCLIDEAN VS MANHATTAN
  ------------------------------------------
  We tested both Euclidean (L2) and Manhattan (L1) distance metrics:
  - Euclidean: sqrt(sum((x_i - y_i)^2)) -- penalizes large differences more
    heavily. With ternary features, differences are at most 2 per dimension.
  - Manhattan: sum(|x_i - y_i|) -- more robust to outliers and often more
    discriminative in higher dimensions. For ternary features, Manhattan
    distance effectively counts how many features differ and by how much.
  - For phishing detection with 30 ternary features, Manhattan distance may
    be slightly preferable as it treats each feature dimension independently.


  STRENGTHS OF KNN
  -------------------
  + Simple to understand and implement -- no complex training procedure
  + No training phase (lazy learning) -- new training data is added trivially
  + Naturally handles binary classification
  + Adapts to complex, non-linear decision boundaries
  + No distributional assumptions (non-parametric)
  + Works well when decision boundary is irregular


  WEAKNESSES OF KNN
  --------------------
  - Slow prediction: must compute distances to ALL training samples for
    each prediction. With ~8,800 training samples and 30 features, each
    prediction requires ~264,000 distance calculations.
  - Memory-intensive: stores entire training set in memory
  - Sensitive to irrelevant features: all features contribute equally to
    distance, so noisy features degrade performance
  - Curse of dimensionality: 30 dimensions may cause distance metrics to
    lose discriminative power
  - No interpretability: unlike Logistic Regression (which gives feature
    coefficients) or Decision Trees (which give split rules), KNN provides
    no insight into WHY a prediction was made"""

    print(commentary)

    print(f"""

  COMPARISON WITH LOGISTIC REGRESSION BASELINE
  -----------------------------------------------
  KNN vs Logistic Regression for phishing website detection:

  | Aspect                  | KNN                              | Logistic Regression           |
  |-------------------------|----------------------------------|-------------------------------|
  | Decision boundary       | Non-linear, flexible             | Linear (hyperplane)           |
  | Training                | None (lazy learning)             | Iterative optimization        |
  | Prediction speed        | Slow (distance to all samples)   | Fast (matrix multiply)        |
  | Interpretability        | None (black box)                 | Feature coefficients          |
  | Binary classification   | Native                           | Native (sigmoid)              |
  | Feature scaling         | Beneficial                       | Recommended                   |
  | Typical accuracy (here) | ~{acc*100:.1f}%                        | Competitive                   |

  KNN achieves ~{acc*100:.1f}% accuracy on this dataset. The 5-fold CV accuracy
  of {cv_scores.mean()*100:.1f}% +/- {cv_scores.std()*100:.1f}% confirms the model generalizes reasonably.

  Key takeaway: KNN's non-linear boundary may capture interactions between
  URL features that Logistic Regression misses. However, with 30 ternary
  features, the improvement over linear models may be modest since the
  features are already highly informative individually.""")


# =============================================================================
#  MAIN EXECUTION
# =============================================================================

def main():
    """Run the complete KNN pipeline."""
    print("\n" + "#" * 70)
    print("#" + " " * 68 + "#")
    print("#   K-NEAREST NEIGHBOURS (KNN) -- PHISHING WEBSITE DETECTION" + " " * 8 + "#")
    print("#   CSAI412 Machine Learning Group Project" + " " * 26 + "#")
    print("#" + " " * 68 + "#")
    print("#" * 70)

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_data()

    # 2. Hyperparameter tuning
    grid_search, best_model = tune_hyperparameters(X_train, y_train)

    # 3. K vs accuracy curve
    plot_k_vs_accuracy(X_train, y_train, X_test, y_test, grid_search.best_params_)

    # 4. Evaluation on test set
    acc, prec, rec, f1, y_pred = evaluate_model(best_model, X_train, X_test, y_train, y_test)

    # 5. Cross-validation
    cv_scores = run_cross_validation(best_model, X_train, y_train)

    # 6. Decision boundary visualization
    plot_decision_boundary(best_model, X_train, y_train, X_test, y_test)

    # 7. Save model
    model_path = save_model(best_model)

    # 8. Commentary
    print_commentary(acc, cv_scores)

    # Final summary
    print_header("SUMMARY")
    print(f"""
  Model            : K-Nearest Neighbours (KNN)
  Best k           : {best_model.n_neighbors}
  Best weights     : {best_model.weights}
  Best metric      : {best_model.metric}
  Test Accuracy    : {acc:.4f} ({acc*100:.2f}%)
  Test F1 (wt.)    : {f1:.4f}
  CV Accuracy      : {cv_scores.mean():.4f} +/- {cv_scores.std():.4f}
  Model saved to   : {model_path}

  Figures saved:
    - figures/knn_k_vs_accuracy.png
    - figures/knn_confusion_matrix.png
    - figures/knn_decision_boundary.png
""")

    print("=" * 70)
    print("  KNN PIPELINE COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
