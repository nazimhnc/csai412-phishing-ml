"""
Logistic Regression -- Baseline Classifier for Phishing Website Detection
==========================================================================
CSAI412 Machine Learning Group Project

Model: Logistic Regression (binary, sigmoid)
Role:  BASELINE classifier -- all other models are compared against this.
Dataset: UCI Phishing Websites (11,055 samples, 30 features, binary classification)
Target:  Result -- Phishing (0) vs Legitimate (1) after encoding

Hyperparameter tuning via GridSearchCV, 5-fold cross-validation,
full evaluation metrics, confusion matrix heatmap, ROC curves,
and feature importance (coefficient) analysis.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend -- safe for headless servers
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
    roc_curve,
    auc,
)
from sklearn.preprocessing import label_binarize
import joblib

# -- Data loader import --------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import get_train_test

# -- Paths ---------------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Suppress convergence warnings during grid search (we set max_iter high enough)
warnings.filterwarnings("ignore", category=UserWarning)

# Class labels for display
CLASS_NAMES = ["Phishing", "Legitimate"]


# ======================================================================
# 1. COMMENTARY -- WHY LOGISTIC REGRESSION AS BASELINE
# ======================================================================
def print_commentary():
    """Print detailed commentary on why Logistic Regression is our baseline."""
    commentary = """
+========================================================================+
|          COMMENTARY -- LOGISTIC REGRESSION AS BASELINE                  |
+========================================================================+

1. WHY LOGISTIC REGRESSION IS APPROPRIATE AS A BASELINE
   -----------------------------------------------------
   Logistic Regression is the standard first-choice baseline for classification
   tasks because it is simple, well-understood, and provides a principled
   probabilistic framework. For our phishing website detection task (binary
   classification, 30 URL/website features, ~11,055 samples), it gives us a
   clear "floor" of performance: any more complex model (SVM, Decision Tree,
   MLP, KNN) should beat this baseline to justify its added complexity.

   As a linear model, it also tells us HOW MUCH of the phishing signal is
   linearly separable -- if Logistic Regression performs well, complex models
   may not be needed. If it struggles, that is strong evidence that non-linear
   decision boundaries are required.

2. LINEAR DECISION BOUNDARIES & PHISHING FEATURES
   ------------------------------------------------
   Logistic Regression assumes that classes can be separated by a linear
   (hyperplane) boundary in feature space. Phishing website features are
   encoded as ternary values {-1, 0, 1} representing suspicious (-1),
   neutral (0), and legitimate (1) characteristics of URLs and web pages.

   These features capture properties like:
   - URL-based: having_IP_Address, URL_Length, Shortining_Service,
     having_At_Symbol, double_slash_redirecting, Prefix_Suffix, etc.
   - Domain-based: Domain_registeration_length, Favicon, port, HTTPS_token
   - HTML/JS-based: Redirect, on_mouseover, RightClick, popUpWidnow
   - Page-based: Google_Index, Links_pointing_to_page, Statistical_report

   Since these features are already encoded as discrete ordinal values,
   a linear model can effectively combine them as a weighted sum to produce
   a phishing score. Many of these features have direct, additive effects
   on phishing likelihood, making linear combination a natural fit.

3. BINARY CLASSIFICATION -- SIGMOID FUNCTION
   -------------------------------------------
   For binary classification (Phishing vs Legitimate), Logistic Regression
   uses the sigmoid function:

       P(y = 1 | x) = 1 / (1 + exp(-w^T x))

   This produces a probability between 0 and 1, directly interpretable as
   the likelihood that a website is legitimate. The decision threshold is
   typically 0.5, but can be adjusted based on the relative cost of false
   positives (blocking legitimate sites) vs false negatives (missing phishing
   sites).

4. STRENGTHS
   ----------
   - Interpretable coefficients: Each feature gets a coefficient telling us
     exactly how each URL/website property influences the phishing prediction.
     This is invaluable for understanding which features are most indicative
     of phishing attacks.
   - Fast training: Even with GridSearchCV over multiple hyperparameter combos,
     training completes in seconds. This makes it ideal for rapid iteration.
   - Probabilistic outputs: predict_proba() gives calibrated class
     probabilities, enabling ROC analysis and confidence thresholding.
   - Regularisation: The C parameter controls L2 regularisation strength,
     preventing overfitting on our moderate-sized dataset.

5. WEAKNESSES
   -----------
   - Limited to linear boundaries: Cannot model feature interactions or
     non-linear thresholds without feature engineering.
   - May miss complex phishing patterns: Sophisticated phishing sites may
     combine features in non-linear ways that a linear model cannot capture.
   - Assumes feature independence in the linear sense: Does not naturally
     capture correlations between URL features and page-based features.

6. PERFORMANCE ANALYSIS (see metrics below)
   -----------------------------------------
   After running the model, examine:
   - Overall accuracy: Gives the proportion of correct predictions.
   - Per-class precision/recall/F1: Reveals if the model is better at
     detecting phishing or confirming legitimate sites.
   - Confusion matrix: Shows false positive (legitimate flagged as phishing)
     and false negative (phishing missed) rates.
   - ROC-AUC: Measures discriminative ability across all thresholds.
"""
    print(commentary)


# ======================================================================
# 2. LOAD DATA
# ======================================================================
def load_dataset():
    """Load the phishing website dataset using the shared data_loader."""
    print("=" * 72)
    print("  LOGISTIC REGRESSION -- BASELINE CLASSIFIER")
    print("  Phishing Website Detection Dataset")
    print("=" * 72)

    data = get_train_test()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print(f"\n[LR] Training samples: {X_train.shape[0]}")
    print(f"[LR] Test samples:     {X_test.shape[0]}")
    print(f"[LR] Features:         {X_train.shape[1]}")
    print(f"[LR] Classes:          {sorted(np.unique(y_train))}")

    return X_train, X_test, y_train, y_test, feature_names


# ======================================================================
# 3. HYPERPARAMETER TUNING WITH GRIDSEARCHCV
# ======================================================================
def tune_hyperparameters(X_train, y_train):
    """Run GridSearchCV to find the best Logistic Regression hyperparameters."""
    print("\n" + "=" * 72)
    print("  HYPERPARAMETER TUNING (GridSearchCV, 5-fold CV)")
    print("=" * 72)

    param_grid = {
        "C": [0.001, 0.01, 0.1, 1, 10, 100],
        "solver": ["lbfgs", "newton-cg"],
    }

    lr = LogisticRegression(max_iter=1000, random_state=42)

    grid_search = GridSearchCV(
        estimator=lr,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        return_train_score=True,
    )

    print("\n[LR] Grid search parameter space:")
    print(f"     C values:       {param_grid['C']}")
    print(f"     Solvers:        {param_grid['solver']}")
    total_fits = len(param_grid["C"]) * len(param_grid["solver"]) * 5  # 5-fold
    print(f"     Total fits:     {total_fits}")
    print()

    grid_search.fit(X_train, y_train)

    print(f"\n[LR] Best parameters:      {grid_search.best_params_}")
    print(f"[LR] Best CV accuracy:     {grid_search.best_score_:.4f}")

    # Show all results
    results_df = pd.DataFrame(grid_search.cv_results_)
    results_df = results_df.sort_values("rank_test_score")
    print("\n[LR] All GridSearchCV results (sorted by rank):")
    print(f"{'Rank':<6} {'C':<10} {'Solver':<12} {'Mean CV Acc':<14} {'Std':<10} {'Mean Train Acc':<16}")
    print("-" * 68)
    for _, row in results_df.iterrows():
        print(
            f"{int(row['rank_test_score']):<6} "
            f"{row['param_C']:<10} "
            f"{row['param_solver']:<12} "
            f"{row['mean_test_score']:<14.4f} "
            f"{row['std_test_score']:<10.4f} "
            f"{row['mean_train_score']:<16.4f}"
        )

    return grid_search.best_estimator_, grid_search


# ======================================================================
# 4. CROSS-VALIDATION (5-FOLD) ON BEST MODEL
# ======================================================================
def cross_validate_model(best_model, X_train, y_train):
    """Run 5-fold cross-validation on the best model and report mean/std accuracy."""
    print("\n" + "=" * 72)
    print("  5-FOLD CROSS-VALIDATION (Best Model)")
    print("=" * 72)

    cv_scores = cross_val_score(best_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

    print(f"\n[LR] Fold accuracies: {[f'{s:.4f}' for s in cv_scores]}")
    print(f"[LR] Mean accuracy:   {cv_scores.mean():.4f}")
    print(f"[LR] Std deviation:   {cv_scores.std():.4f}")
    print(f"[LR] 95% CI:          [{cv_scores.mean() - 1.96*cv_scores.std():.4f}, "
          f"{cv_scores.mean() + 1.96*cv_scores.std():.4f}]")

    return cv_scores


# ======================================================================
# 5. TEST SET EVALUATION
# ======================================================================
def evaluate_on_test_set(best_model, X_test, y_test):
    """Evaluate the best model on the held-out test set."""
    print("\n" + "=" * 72)
    print("  TEST SET EVALUATION")
    print("=" * 72)

    y_pred = best_model.predict(X_test)

    # Core metrics (weighted for consistency with comparison script)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n[LR] Test Set Metrics (weighted averages):")
    print(f"     Accuracy:   {accuracy:.4f}")
    print(f"     Precision:  {precision:.4f}")
    print(f"     Recall:     {recall:.4f}")
    print(f"     F1-Score:   {f1:.4f}")

    # Full classification report
    print(f"\n[LR] Full Classification Report:")
    print("-" * 72)
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))
    report = classification_report(y_test, y_pred, labels=classes,
                                   target_names=CLASS_NAMES, zero_division=0)
    print(report)

    # Confusion matrix
    cm = confusion_matrix(y_test, y_pred, labels=classes)
    print(f"[LR] Confusion Matrix (rows=true, cols=predicted):")
    cm_df = pd.DataFrame(cm,
                         index=[f"True {n}" for n in CLASS_NAMES],
                         columns=[f"Pred {n}" for n in CLASS_NAMES])
    print(cm_df)

    return y_pred, accuracy, precision, recall, f1, cm, classes


# ======================================================================
# 6. CONFUSION MATRIX HEATMAP
# ======================================================================
def plot_confusion_matrix(cm, classes):
    """Save a confusion matrix heatmap to figures/."""
    print("\n" + "=" * 72)
    print("  CONFUSION MATRIX HEATMAP")
    print("=" * 72)

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
    ax.set_title("Logistic Regression -- Confusion Matrix\n(Phishing Website Detection)", fontsize=14, fontweight="bold")
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "logistic_regression_confusion_matrix.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"[LR] Saved confusion matrix heatmap to: {path}")


# ======================================================================
# 7. ROC CURVES
# ======================================================================
def plot_roc_curves(best_model, X_test, y_test, classes):
    """Plot ROC curve for binary phishing detection."""
    print("\n" + "=" * 72)
    print("  ROC CURVE (Binary Classification)")
    print("=" * 72)

    # Get probability predictions
    y_prob = best_model.predict_proba(X_test)

    # For binary classification, we can compute a single ROC curve
    # Use probability of the positive class (Legitimate = 1)
    # But also show per-class for consistency with comparison script
    n_classes = len(classes)

    if n_classes == 2:
        # Binary ROC curve
        fpr, tpr, _ = roc_curve(y_test, y_prob[:, 1])
        roc_auc_val = auc(fpr, tpr)
        print(f"  ROC-AUC: {roc_auc_val:.4f}")

        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(fpr, tpr, color="darkorange", lw=2,
                label=f"ROC curve (AUC = {roc_auc_val:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(
            f"Logistic Regression -- ROC Curve\n"
            f"Phishing Detection | AUC = {roc_auc_val:.3f}",
            fontsize=14,
            fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=11)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

        # Also compute per-class AUC for the return value
        roc_auc = {}
        y_test_bin = label_binarize(y_test, classes=classes)
        if y_test_bin.ndim == 1 or y_test_bin.shape[1] == 1:
            # Binary case: label_binarize returns (n,1), expand to (n,2)
            if y_test_bin.ndim == 1:
                y_test_bin = y_test_bin.reshape(-1, 1)
            y_test_bin = np.column_stack([1 - y_test_bin, y_test_bin])
        for i, c in enumerate(classes):
            fpr_c, tpr_c, _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[c] = auc(fpr_c, tpr_c)
            print(f"  Class {CLASS_NAMES[i]} ({c}): AUC = {roc_auc[c]:.4f}")

        macro_auc = roc_auc_val  # For binary, same as the single AUC

    else:
        # Multi-class fallback (shouldn't happen for this dataset)
        y_test_bin = label_binarize(y_test, classes=classes)
        fpr = {}
        tpr = {}
        roc_auc = {}
        for i, c in enumerate(classes):
            fpr[c], tpr[c], _ = roc_curve(y_test_bin[:, i], y_prob[:, i])
            roc_auc[c] = auc(fpr[c], tpr[c])
            print(f"  Class {c}: AUC = {roc_auc[c]:.4f}")
        macro_auc = np.mean(list(roc_auc.values()))

        fig, ax = plt.subplots(figsize=(8, 6))
        for i, c in enumerate(classes):
            ax.plot(fpr[c], tpr[c], lw=2,
                    label=f"Class {c} (AUC = {roc_auc[c]:.3f})")
        ax.plot([0, 1], [0, 1], "k--", lw=1, label="Random (AUC = 0.500)")
        ax.set_xlim([-0.02, 1.02])
        ax.set_ylim([-0.02, 1.02])
        ax.set_xlabel("False Positive Rate", fontsize=13)
        ax.set_ylabel("True Positive Rate", fontsize=13)
        ax.set_title(
            f"Logistic Regression -- ROC Curves\n"
            f"Macro-Average AUC = {macro_auc:.3f}",
            fontsize=14, fontweight="bold",
        )
        ax.legend(loc="lower right", fontsize=10)
        ax.grid(True, alpha=0.3)
        plt.tight_layout()

    print(f"\n  Macro-average AUC: {macro_auc:.4f}")

    path = os.path.join(FIGURES_DIR, "logistic_regression_roc.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[LR] Saved ROC curve to: {path}")

    return roc_auc, macro_auc


# ======================================================================
# 8. FEATURE IMPORTANCE (COEFFICIENTS)
# ======================================================================
def analyse_feature_importance(best_model, feature_names, classes):
    """Analyse and visualise Logistic Regression coefficients."""
    print("\n" + "=" * 72)
    print("  FEATURE IMPORTANCE (Logistic Regression Coefficients)")
    print("=" * 72)

    coefs = best_model.coef_  # shape: (1, n_features) for binary
    # For binary classification, coef_ has shape (1, n_features)
    abs_coef = np.abs(coefs[0])

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Coefficient": coefs[0],
        "|Coefficient|": abs_coef,
    }).sort_values("|Coefficient|", ascending=False)

    print("\n[LR] Feature importance (absolute coefficient, binary classification):")
    print("-" * 60)
    for _, row in importance_df.iterrows():
        bar = "#" * int(row["|Coefficient|"] * 20 / importance_df["|Coefficient|"].max())
        direction = "+" if row["Coefficient"] > 0 else "-"
        print(f"  {direction} {row['Feature']:<35} {row['Coefficient']:+.4f}  {bar}")

    print(f"\n[LR] Positive coefficient = pushes toward Legitimate (class 1)")
    print(f"[LR] Negative coefficient = pushes toward Phishing (class 0)")

    # Plot feature importance bar chart
    fig, ax = plt.subplots(figsize=(10, 8))
    importance_sorted = importance_df.sort_values("|Coefficient|", ascending=True)
    colors = ["#d32f2f" if c < 0 else "#388e3c" for c in importance_sorted["Coefficient"]]
    ax.barh(
        importance_sorted["Feature"],
        importance_sorted["Coefficient"],
        color=colors,
        edgecolor="gray",
    )
    ax.set_xlabel("Coefficient Value", fontsize=12)
    ax.set_title("Logistic Regression -- Feature Coefficients\n(Phishing Detection: - = Phishing, + = Legitimate)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.grid(True, axis="x", alpha=0.3)
    plt.tight_layout()

    path = os.path.join(FIGURES_DIR, "logistic_regression_feature_importance.png")
    fig.savefig(path, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"\n[LR] Saved feature importance plot to: {path}")

    return importance_df


# ======================================================================
# 9. SAVE MODEL
# ======================================================================
def save_model(best_model):
    """Save the best Logistic Regression model to disk."""
    print("\n" + "=" * 72)
    print("  SAVE MODEL")
    print("=" * 72)

    model_path = os.path.join(DATA_DIR, "logistic_regression_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"[LR] Best model saved to: {model_path}")
    print(f"[LR] Model parameters: {best_model.get_params()}")

    return model_path


# ======================================================================
# 10. PERFORMANCE ANALYSIS COMMENTARY
# ======================================================================
def print_performance_analysis(accuracy, precision, recall, f1, cv_scores, roc_auc, macro_auc, cm, classes):
    """Print detailed performance analysis commentary."""
    analysis = f"""
+========================================================================+
|          PERFORMANCE ANALYSIS -- LOGISTIC REGRESSION                    |
+========================================================================+

TEST SET RESULTS:
  Accuracy:  {accuracy:.4f}  |  Precision: {precision:.4f}  |  Recall: {recall:.4f}  |  F1: {f1:.4f}

CROSS-VALIDATION:
  Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})

ROC-AUC: {macro_auc:.4f}

INTERPRETATION:
  - The test accuracy of {accuracy:.1%} establishes our BASELINE. All subsequent
    models (SVM, Decision Tree, KNN, MLP) must beat this to justify their
    added complexity.

  - Weighted precision ({precision:.4f}) and recall ({recall:.4f}) are close to each
    other, indicating the model is not systematically biasing toward
    precision or recall for either class.

  - The cross-validation mean ({cv_scores.mean():.4f}) is close to the test accuracy
    ({accuracy:.4f}), suggesting the model generalises consistently and is not
    overfitting. The low standard deviation ({cv_scores.std():.4f}) confirms
    stability across folds.
"""

    # Analyse confusion matrix
    total_samples = cm.sum()
    diagonal = np.diag(cm).sum()

    analysis += f"""
CONFUSION MATRIX ANALYSIS:
  - Correct predictions: {diagonal}/{total_samples} ({diagonal/total_samples:.1%})
  - False Positives (legitimate flagged as phishing): {cm[1, 0] if cm.shape[0] > 1 else 'N/A'}
  - False Negatives (phishing missed as legitimate): {cm[0, 1] if cm.shape[0] > 1 else 'N/A'}

  In phishing detection, FALSE NEGATIVES are more dangerous than false
  positives -- missing a phishing site exposes users to credential theft,
  while blocking a legitimate site is merely inconvenient. A model with
  higher recall on the Phishing class is generally preferred.

ROC-AUC ANALYSIS:
  - AUC of {macro_auc:.4f} indicates {'strong' if macro_auc > 0.85 else 'moderate' if macro_auc > 0.75 else 'weak'} discriminative ability.
  - An AUC close to 1.0 means the model can effectively rank phishing
    sites as more suspicious than legitimate ones across all thresholds.

CONCLUSION:
  Logistic Regression provides a {'solid' if accuracy > 0.85 else 'modest'} baseline at {accuracy:.1%} accuracy.
  {'This is a strong baseline -- the discrete ternary features (-1, 0, 1) lend themselves well to linear combination, meaning the phishing signal is substantially linearly separable.' if accuracy > 0.85 else 'There is room for improvement. Non-linear models may capture feature interactions that linear combination misses.'}
  The interpretable coefficients reveal which URL/website properties most
  influence phishing predictions, providing valuable security insight
  regardless of the accuracy level.
"""
    print(analysis)


# ======================================================================
# MAIN EXECUTION
# ======================================================================
def main():
    """Run the complete Logistic Regression baseline pipeline."""

    # Commentary (printed first for the report)
    print_commentary()

    # 1. Load data
    X_train, X_test, y_train, y_test, feature_names = load_dataset()

    # 2. Hyperparameter tuning
    best_model, grid_search = tune_hyperparameters(X_train, y_train)

    # 3. Cross-validation on best model
    cv_scores = cross_validate_model(best_model, X_train, y_train)

    # 4. Test set evaluation
    y_pred, accuracy, precision, recall, f1, cm, classes = evaluate_on_test_set(
        best_model, X_test, y_test
    )

    # 5. Confusion matrix heatmap
    plot_confusion_matrix(cm, classes)

    # 6. ROC curves
    roc_auc, macro_auc = plot_roc_curves(best_model, X_test, y_test, classes)

    # 7. Feature importance
    importance_df = analyse_feature_importance(best_model, feature_names, classes)

    # 8. Save model
    model_path = save_model(best_model)

    # 9. Performance analysis
    print_performance_analysis(
        accuracy, precision, recall, f1, cv_scores, roc_auc, macro_auc, cm, classes
    )

    # Final summary
    print("=" * 72)
    print("  LOGISTIC REGRESSION BASELINE -- COMPLETE")
    print("=" * 72)
    print(f"  Best params:     {grid_search.best_params_}")
    print(f"  Test accuracy:   {accuracy:.4f}")
    print(f"  CV accuracy:     {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
    print(f"  ROC-AUC:         {macro_auc:.4f}")
    print(f"  Model saved:     {model_path}")
    print(f"  Figures saved:   {FIGURES_DIR}/")
    print("=" * 72)


if __name__ == "__main__":
    main()
