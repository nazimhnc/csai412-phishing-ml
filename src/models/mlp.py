"""
Multi-Layer Perceptron (MLP) Classifier for Phishing Website Detection
=======================================================================
CSAI412 Machine Learning Group Project

Model: MLPClassifier (sklearn.neural_network)
Dataset: UCI Phishing Websites (11,055 samples, 30 features)
Task: Binary classification (Phishing vs Legitimate)

This module implements a fully-connected feedforward neural network
(Multi-Layer Perceptron) with hyperparameter tuning via GridSearchCV,
comprehensive evaluation, architecture comparison, and convergence analysis.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.neural_network import MLPClassifier
from sklearn.model_selection import GridSearchCV, cross_val_score
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

# Suppress convergence warnings during grid search (we use early_stopping)
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", message=".*ConvergenceWarning.*")
from sklearn.exceptions import ConvergenceWarning
warnings.filterwarnings("ignore", category=ConvergenceWarning)

# -- Data Import ---------------------------------------------------------------
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from data_loader import get_train_test

# -- Project Paths -------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Class labels for display
CLASS_NAMES = ["Phishing", "Legitimate"]


# ==============================================================================
# COMMENTARY -- Why MLP for Phishing Website Detection?
# ==============================================================================
def print_commentary():
    """Print detailed commentary on MLP for phishing detection."""
    commentary = """
+==============================================================================+
|          MULTI-LAYER PERCEPTRON (MLP) -- MODEL COMMENTARY                    |
+==============================================================================+

1. WHY MLP IS APPROPRIATE FOR PHISHING WEBSITE DETECTION
   -------------------------------------------------------
   Phishing detection involves classifying websites based on 30 URL and
   website features encoded as ternary values {-1, 0, 1}. A Multi-Layer
   Perceptron is well-suited because:

   - It can model ARBITRARY non-linear decision boundaries between phishing
     and legitimate websites, unlike linear classifiers (Logistic Regression).
   - The 30 input features likely interact in non-trivial ways -- e.g., a
     website with a suspicious URL AND manipulated browser features is far
     more likely to be phishing than one with either characteristic alone.
     MLPs naturally capture such feature interactions through hidden layers.
   - With 11,055 samples and 30 features, the dataset is large enough to
     train a moderately-sized neural network without severe overfitting.

2. UNIVERSAL APPROXIMATION THEOREM
   ---------------------------------
   The Universal Approximation Theorem (Cybenko, 1989; Hornik, 1991) proves
   that a feedforward network with a single hidden layer containing a finite
   number of neurons can approximate ANY continuous function on compact
   subsets of R^n to arbitrary accuracy. This means:

   - Our MLP can theoretically learn the true mapping from URL/website
     features to phishing classification -- no matter how complex.
   - In practice, deeper architectures (multiple hidden layers) often learn
     this mapping more EFFICIENTLY, requiring fewer total parameters.
   - However, the theorem is EXISTENTIAL -- it guarantees a solution exists
     but does not guarantee that gradient descent will find it.

3. ARCHITECTURE CHOICES -- DEPTH vs. WIDTH TRADEOFFS
   ---------------------------------------------------
   We test 5 architectures:
   - (50,)          -- Shallow & narrow: fast to train, limited capacity
   - (100,)         -- Shallow & wide: more capacity, still single-layer
   - (100, 50)      -- Two layers with tapering: progressive abstraction
   - (128, 64)      -- Two layers, powers of 2: gradual compression
   - (128, 64, 32)  -- Three layers: deepest model, hierarchical feature
                        extraction

   For phishing detection with ternary features, moderate depth (2 layers)
   often suffices because the features are already highly informative
   individually, and the MLP needs mainly to learn which COMBINATIONS
   of suspicious indicators are most predictive.

4. ACTIVATION FUNCTIONS -- ReLU vs. TANH
   ----------------------------------------
   - ReLU: f(x) = max(0, x) -- fast, sparse activation, good for deeper nets
   - Tanh: f(x) = (e^x - e^-x) / (e^x + e^-x) -- zero-centered [-1, 1]

   For ternary features ({-1, 0, 1}), tanh may align well since the input
   values are already in the [-1, 1] range. ReLU may benefit from sparsity.

5. REGULARIZATION -- ALPHA (L2 PENALTY)
   ---------------------------------------
   Alpha controls L2 regularization strength:
   - alpha = 0.0001 (weak): allows complex patterns, risk of overfitting
   - alpha = 0.001 (moderate): balanced bias-variance tradeoff
   - alpha = 0.01 (strong): constrains weights, may underfit

   Combined with early_stopping=True, regularization prevents memorizing
   training noise.

6. FEATURE SCALING -- LESS CRITICAL FOR TERNARY FEATURES
   -------------------------------------------------------
   While neural networks generally require feature scaling, our features
   are already on the same scale ({-1, 0, 1}). Scaling is still applied
   by the data loader for consistency, but the impact is minimal compared
   to datasets with heterogeneous feature ranges.

7. STRENGTHS OF MLP
   ------------------
   - Captures complex non-linear patterns and feature interactions
   - Flexible architecture -- can be scaled for the problem
   - Works well with numerical features
   - No assumptions about data distribution
   - Handles binary classification naturally via sigmoid output

8. WEAKNESSES OF MLP
   -------------------
   - BLACK BOX -- difficult to interpret WHY a site is flagged as phishing
     (unlike Decision Trees with explicit rules)
   - Requires more data and careful tuning than simpler models
   - Sensitive to hyperparameters (architecture, learning rate, alpha)
   - Training is computationally expensive
   - No built-in feature importance (requires post-hoc methods like SHAP)
   - Can converge to local minima
"""
    print(commentary)


# ==============================================================================
# TRAINING & EVALUATION
# ==============================================================================
def run_mlp():
    """Main function: train, tune, evaluate, and visualize the MLP model."""

    print("=" * 78)
    print("  MULTI-LAYER PERCEPTRON (MLP) CLASSIFIER -- PHISHING WEBSITE DETECTION")
    print("=" * 78)

    # -- Print Commentary ------------------------------------------------------
    print_commentary()

    # -- Load Data -------------------------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 1: LOADING & PREPROCESSING DATA")
    print("=" * 78)

    data = get_train_test()
    X_train = data["X_train"]
    X_test = data["X_test"]
    y_train = data["y_train"]
    y_test = data["y_test"]
    feature_names = data["feature_names"]

    print(f"\n  Training samples:  {X_train.shape[0]}")
    print(f"  Test samples:      {X_test.shape[0]}")
    print(f"  Number of features: {X_train.shape[1]}")
    print(f"  Classes:           {sorted(np.unique(y_train))} ({CLASS_NAMES})")
    print(f"  Feature names:     {feature_names}")

    # -- Hyperparameter Tuning with GridSearchCV -------------------------------
    print("\n" + "=" * 78)
    print("  STEP 2: HYPERPARAMETER TUNING (GridSearchCV)")
    print("=" * 78)

    param_grid = {
        "hidden_layer_sizes": [(50,), (100,), (100, 50), (128, 64), (128, 64, 32)],
        "activation": ["relu", "tanh"],
        "alpha": [0.0001, 0.001, 0.01],
        "learning_rate": ["constant", "adaptive"],
    }

    total_combos = 1
    for key, values in param_grid.items():
        total_combos *= len(values)
    print(f"\n  Parameter grid:")
    for key, values in param_grid.items():
        print(f"    {key}: {values}")
    print(f"\n  Total hyperparameter combinations: {total_combos}")
    print(f"  Cross-validation folds: 5")
    print(f"  Total fits: {total_combos * 5}")
    print(f"\n  Fixed parameters:")
    print(f"    max_iter: 500")
    print(f"    early_stopping: True")
    print(f"    random_state: 42")
    print(f"\n  Running GridSearchCV... (this may take several minutes)")

    base_mlp = MLPClassifier(
        max_iter=500,
        early_stopping=True,
        random_state=42,
        validation_fraction=0.1,
    )

    grid_search = GridSearchCV(
        estimator=base_mlp,
        param_grid=param_grid,
        cv=5,
        scoring="accuracy",
        n_jobs=-1,
        verbose=1,
        refit=True,
    )

    grid_search.fit(X_train, y_train)

    # -- Best Parameters -------------------------------------------------------
    print(f"\n  GridSearchCV complete!")
    print(f"\n  Best Parameters:")
    for param, value in grid_search.best_params_.items():
        print(f"    {param}: {value}")
    print(f"\n  Best Cross-Validation Accuracy: {grid_search.best_score_:.4f}")

    best_model = grid_search.best_estimator_

    # -- Evaluation on Test Set ------------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 3: EVALUATION ON TEST SET")
    print("=" * 78)

    y_pred = best_model.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
    recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
    f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

    print(f"\n  Test Set Performance:")
    print(f"  {'_' * 40}")
    print(f"  Accuracy:           {accuracy:.4f}  ({accuracy * 100:.2f}%)")
    print(f"  Precision (weighted): {precision:.4f}")
    print(f"  Recall (weighted):    {recall:.4f}")
    print(f"  F1-Score (weighted):  {f1:.4f}")

    print(f"\n  Full Classification Report:")
    print(f"  {'_' * 60}")
    report = classification_report(y_test, y_pred, target_names=CLASS_NAMES, zero_division=0)
    for line in report.split("\n"):
        print(f"  {line}")

    # -- Confusion Matrix Heatmap ----------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 4: CONFUSION MATRIX")
    print("=" * 78)

    cm = confusion_matrix(y_test, y_pred)
    classes = sorted(np.unique(np.concatenate([y_test, y_pred])))

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
    ax.set_xlabel("Predicted Class", fontsize=13, fontweight="bold")
    ax.set_ylabel("Actual Class", fontsize=13, fontweight="bold")
    ax.set_title("MLP Classifier -- Confusion Matrix\n(Phishing Website Detection)", fontsize=15, fontweight="bold")
    plt.tight_layout()
    cm_path = os.path.join(FIGURES_DIR, "mlp_confusion_matrix.png")
    plt.savefig(cm_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n  Confusion matrix saved to: {cm_path}")

    # Print confusion matrix as text too
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    cm_df = pd.DataFrame(cm, index=CLASS_NAMES, columns=CLASS_NAMES)
    print(f"\n{cm_df.to_string()}")

    # -- Cross-Validation (5-Fold) with Best Model -----------------------------
    print("\n" + "=" * 78)
    print("  STEP 5: 5-FOLD CROSS-VALIDATION (Best Model)")
    print("=" * 78)

    best_params = grid_search.best_params_.copy()
    cv_model = MLPClassifier(
        **best_params,
        max_iter=500,
        early_stopping=True,
        random_state=42,
        validation_fraction=0.1,
    )

    cv_scores = cross_val_score(cv_model, X_train, y_train, cv=5, scoring="accuracy", n_jobs=-1)

    print(f"\n  5-Fold Cross-Validation Results:")
    print(f"  {'_' * 40}")
    for i, score in enumerate(cv_scores, 1):
        print(f"    Fold {i}: {score:.4f}")
    print(f"  {'_' * 40}")
    print(f"    Mean Accuracy:  {cv_scores.mean():.4f}")
    print(f"    Std Deviation:  {cv_scores.std():.4f}")
    print(f"    95% CI:         [{cv_scores.mean() - 1.96 * cv_scores.std():.4f}, "
          f"{cv_scores.mean() + 1.96 * cv_scores.std():.4f}]")

    # -- Training Loss Curve ---------------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 6: TRAINING LOSS CURVE (Convergence Analysis)")
    print("=" * 78)

    # Re-train the best model to get the loss_curve_ attribute
    loss_model = MLPClassifier(
        **best_params,
        max_iter=500,
        early_stopping=True,
        random_state=42,
        validation_fraction=0.1,
    )
    loss_model.fit(X_train, y_train)

    loss_curve = loss_model.loss_curve_

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(range(1, len(loss_curve) + 1), loss_curve, color="royalblue", linewidth=2, label="Training Loss")
    ax.set_xlabel("Epoch (Iteration)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Loss", fontsize=13, fontweight="bold")
    ax.set_title("MLP Training Loss Curve\n(Phishing Website Detection)", fontsize=15, fontweight="bold")
    ax.legend(fontsize=12)
    ax.grid(True, alpha=0.3)

    # Mark convergence point
    min_loss_idx = np.argmin(loss_curve)
    ax.axvline(x=min_loss_idx + 1, color="red", linestyle="--", alpha=0.6, label=f"Min loss at epoch {min_loss_idx + 1}")
    ax.scatter([min_loss_idx + 1], [loss_curve[min_loss_idx]], color="red", s=80, zorder=5)
    ax.legend(fontsize=12)

    plt.tight_layout()
    loss_path = os.path.join(FIGURES_DIR, "mlp_loss_curve.png")
    plt.savefig(loss_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Loss curve saved to: {loss_path}")
    print(f"\n  Convergence Analysis:")
    print(f"    Total epochs trained:    {len(loss_curve)}")
    print(f"    Initial loss:            {loss_curve[0]:.4f}")
    print(f"    Final loss:              {loss_curve[-1]:.4f}")
    print(f"    Minimum loss:            {min(loss_curve):.4f} (epoch {min_loss_idx + 1})")
    print(f"    Loss reduction:          {loss_curve[0] - loss_curve[-1]:.4f} ({(1 - loss_curve[-1] / loss_curve[0]) * 100:.1f}%)")

    if len(loss_curve) < 500:
        print(f"    Early stopping triggered at epoch {len(loss_curve)} (max_iter=500)")
        print(f"    -> The model converged before reaching the maximum iterations.")
        print(f"    -> Early stopping prevented overfitting by monitoring validation loss.")
    else:
        print(f"    Model trained for all 500 epochs (no early stopping triggered).")
        print(f"    -> Consider increasing max_iter or adjusting learning rate.")

    # -- Architecture Comparison -----------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 7: ARCHITECTURE COMPARISON")
    print("=" * 78)

    architectures = {
        "(50,)": (50,),
        "(100,)": (100,),
        "(100, 50)": (100, 50),
        "(128, 64)": (128, 64),
        "(128, 64, 32)": (128, 64, 32),
    }

    arch_results = {}
    print(f"\n  Training {len(architectures)} architectures with best activation/alpha/lr...")
    print(f"  (Using: activation={best_params['activation']}, alpha={best_params['alpha']}, "
          f"learning_rate={best_params['learning_rate']})")
    print()

    for name, layers in architectures.items():
        arch_model = MLPClassifier(
            hidden_layer_sizes=layers,
            activation=best_params["activation"],
            alpha=best_params["alpha"],
            learning_rate=best_params["learning_rate"],
            max_iter=500,
            early_stopping=True,
            random_state=42,
            validation_fraction=0.1,
        )
        arch_model.fit(X_train, y_train)
        train_acc = arch_model.score(X_train, y_train)
        test_acc = arch_model.score(X_test, y_test)
        n_params = sum(
            coef.size for coef in arch_model.coefs_
        ) + sum(
            bias.size for bias in arch_model.intercepts_
        )
        arch_results[name] = {
            "train_acc": train_acc,
            "test_acc": test_acc,
            "n_params": n_params,
            "n_epochs": len(arch_model.loss_curve_),
        }
        print(f"    {name:<16}  Train: {train_acc:.4f}  Test: {test_acc:.4f}  "
              f"Params: {n_params:>6,}  Epochs: {arch_model.loss_curve_.__len__()}")

    # Architecture comparison bar chart
    arch_names = list(arch_results.keys())
    train_accs = [arch_results[n]["train_acc"] for n in arch_names]
    test_accs = [arch_results[n]["test_acc"] for n in arch_names]

    fig, ax = plt.subplots(figsize=(12, 7))
    x = np.arange(len(arch_names))
    width = 0.35

    bars_train = ax.bar(x - width / 2, train_accs, width, label="Train Accuracy",
                        color="steelblue", edgecolor="black", linewidth=0.5)
    bars_test = ax.bar(x + width / 2, test_accs, width, label="Test Accuracy",
                       color="coral", edgecolor="black", linewidth=0.5)

    # Add value labels on bars
    for bar in bars_train:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)
    for bar in bars_test:
        height = bar.get_height()
        ax.annotate(f"{height:.3f}", xy=(bar.get_x() + bar.get_width() / 2, height),
                    xytext=(0, 4), textcoords="offset points", ha="center", va="bottom", fontsize=9)

    ax.set_xlabel("Architecture (Hidden Layer Sizes)", fontsize=13, fontweight="bold")
    ax.set_ylabel("Accuracy", fontsize=13, fontweight="bold")
    ax.set_title("MLP Architecture Comparison -- Phishing Website Detection",
                 fontsize=15, fontweight="bold")
    ax.set_xticks(x)
    ax.set_xticklabels(arch_names, fontsize=11)
    ax.legend(fontsize=12)
    ax.grid(axis="y", alpha=0.3)

    # Set y-axis to show differences clearly
    all_accs = train_accs + test_accs
    y_min = max(0, min(all_accs) - 0.05)
    y_max = min(1.0, max(all_accs) + 0.03)
    ax.set_ylim(y_min, y_max)

    plt.tight_layout()
    arch_path = os.path.join(FIGURES_DIR, "mlp_architecture_comparison.png")
    plt.savefig(arch_path, dpi=150, bbox_inches="tight")
    plt.close()

    print(f"\n  Architecture comparison chart saved to: {arch_path}")

    # Identify overfitting
    print(f"\n  Overfitting Analysis:")
    for name in arch_names:
        gap = arch_results[name]["train_acc"] - arch_results[name]["test_acc"]
        status = "OK" if gap < 0.05 else "MILD OVERFIT" if gap < 0.10 else "OVERFIT"
        print(f"    {name:<16}  Train-Test Gap: {gap:.4f}  [{status}]")

    # -- Save Best Model -------------------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 8: SAVING BEST MODEL")
    print("=" * 78)

    model_path = os.path.join(DATA_DIR, "mlp_model.pkl")
    joblib.dump(best_model, model_path)
    print(f"\n  Best model saved to: {model_path}")
    print(f"  Model architecture:  {best_model.hidden_layer_sizes}")
    print(f"  Activation function: {best_model.activation}")
    print(f"  Alpha (L2 penalty):  {best_model.alpha}")
    print(f"  Learning rate:       {best_model.learning_rate}")
    print(f"  Number of layers:    {best_model.n_layers_}")
    print(f"  Output layer neurons: {best_model.n_outputs_}")

    total_params = sum(c.size for c in best_model.coefs_) + sum(b.size for b in best_model.intercepts_)
    print(f"  Total parameters:    {total_params:,}")

    layer_info = []
    layer_info.append(f"    Input:  {X_train.shape[1]} neurons")
    for i, (coef, bias) in enumerate(zip(best_model.coefs_, best_model.intercepts_)):
        layer_info.append(f"    {'Hidden' if i < len(best_model.coefs_) - 1 else 'Output'} {i+1}: "
                          f"{coef.shape[1]} neurons  (weights: {coef.size}, biases: {bias.size})")
    print(f"\n  Network Architecture:")
    for info in layer_info:
        print(f"  {info}")

    # -- Comparison with Other Models ------------------------------------------
    print("\n" + "=" * 78)
    print("  STEP 9: COMPARISON WITH OTHER MODELS")
    print("=" * 78)

    print(f"""
  MLP vs. Other Classifiers (Phishing Detection Context):
  {'_' * 65}

  1. MLP vs. LOGISTIC REGRESSION:
     - MLP captures non-linear patterns that LR misses
     - LR is interpretable (coefficients show feature importance); MLP is opaque
     - For phishing with ternary features, LR may be surprisingly competitive
       since the features are designed as individual phishing indicators

  2. MLP vs. DECISION TREE:
     - Both handle non-linearity, but MLP creates smoother decision boundaries
     - Decision Trees produce human-readable security rules; MLP is a black box
     - For cybersecurity, Decision Tree interpretability is often preferred

  3. MLP vs. SVM:
     - Both handle non-linearity well (SVM via kernels, MLP via hidden layers)
     - SVM with RBF kernel is often comparable to MLP on this dataset size
     - MLP scales better to very large datasets (SGD training)

  4. MLP vs. KNN:
     - KNN is simple but slow at prediction; MLP is fast (forward pass)
     - KNN suffers more from curse of dimensionality with 30 features
     - MLP can learn compact representations; KNN relies on raw feature space

  KEY TAKEAWAY:
  MLP is among the most powerful models for this task when properly tuned.
  However, for phishing detection where explainability matters for security
  operations, Decision Trees or Logistic Regression may be preferred despite
  potentially lower accuracy. The ternary feature encoding means that simpler
  models can often capture most of the classification signal.
""")

    # -- Final Summary ---------------------------------------------------------
    print("=" * 78)
    print("  FINAL SUMMARY -- MLP CLASSIFIER")
    print("=" * 78)
    print(f"""
  Best Hyperparameters:
    Hidden Layers:   {best_model.hidden_layer_sizes}
    Activation:      {best_model.activation}
    Alpha (L2):      {best_model.alpha}
    Learning Rate:   {best_model.learning_rate}

  Test Set Performance:
    Accuracy:        {accuracy:.4f} ({accuracy * 100:.2f}%)
    Precision:       {precision:.4f}
    Recall:          {recall:.4f}
    F1-Score:        {f1:.4f}

  Cross-Validation:
    Mean Accuracy:   {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})

  Convergence:
    Epochs trained:  {len(loss_curve)}
    Final loss:      {loss_curve[-1]:.4f}

  Files Generated:
    Model:           {model_path}
    Confusion Matrix: {cm_path}
    Loss Curve:      {loss_path}
    Architecture:    {arch_path}
""")
    print("=" * 78)
    print("  MLP CLASSIFIER ANALYSIS COMPLETE")
    print("=" * 78)

    return {
        "model": best_model,
        "accuracy": accuracy,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "cv_scores": cv_scores,
        "best_params": grid_search.best_params_,
        "grid_search": grid_search,
        "confusion_matrix": cm,
        "loss_curve": loss_curve,
        "architecture_results": arch_results,
    }


if __name__ == "__main__":
    results = run_mlp()
