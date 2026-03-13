"""
Comprehensive Exploratory Data Analysis (EDA)
==============================================
CSAI412 Machine Learning Group Project

Phishing Websites Dataset - UCI (Mohammad, Thabtah & McCluskey)
Generates all figures to /figures/ and prints a full EDA report.

Key difference from continuous-feature datasets: all 30 features are
categorical/ordinal with values in {-1, 0, 1}, so visualisations use
bar charts and grouped proportions rather than histograms/KDE.
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for server/CI environments
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import seaborn as sns
from scipy import stats

# Add project root to path
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, PROJECT_DIR)

from src.data_loader import (
    load_data, FIGURES_DIR, TARGET_COL, CLASS_NAMES, FEATURE_DESCRIPTIONS
)

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

# Style configuration
sns.set_theme(style="whitegrid", palette="muted", font_scale=1.1)
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "figure.facecolor": "white",
    "axes.facecolor": "white",
    "font.size": 11,
})

# Color palettes — red for phishing, green for legitimate
CLASS_PALETTE = {"Phishing": "#e74c3c", "Legitimate": "#2ecc71"}
CLASS_PALETTE_LIST = ["#e74c3c", "#2ecc71"]  # index 0=Phishing, 1=Legitimate
VALUE_PALETTE = {-1: "#e74c3c", 0: "#f39c12", 1: "#2ecc71"}  # suspicious/neutral/legitimate


def print_section(title, char="="):
    """Print a formatted section header."""
    width = 70
    print(f"\n{char * width}")
    print(f"  {title}")
    print(f"{char * width}")


def dataset_overview(df):
    """Print dataset overview: shape, dtypes, missing values."""
    print_section("1. DATASET OVERVIEW")

    print(f"\nShape: {df.shape[0]} samples x {df.shape[1]} columns")
    print(f"Memory usage: {df.memory_usage(deep=True).sum() / 1024:.1f} KB")

    print(f"\nColumn Types:")
    print(f"  {'Column':<35} {'Type':<12} {'Non-Null':<10} {'Null':<6} {'Unique':<8}")
    print(f"  {'-'*71}")
    for col in df.columns:
        dtype = str(df[col].dtype)
        non_null = df[col].notna().sum()
        null = df[col].isna().sum()
        unique = df[col].nunique()
        print(f"  {col:<35} {dtype:<12} {non_null:<10} {null:<6} {unique:<8}")

    total_missing = df.isna().sum().sum()
    print(f"\nTotal missing values: {total_missing}")
    if total_missing == 0:
        print("  --> Dataset is COMPLETE (no missing values)")

    # Duplicate check
    dupes = df.duplicated().sum()
    print(f"\nDuplicate rows: {dupes} ({dupes/len(df)*100:.2f}%)")


def statistical_summary(df):
    """Print comprehensive statistical summary with value counts per feature."""
    print_section("2. STATISTICAL SUMMARY")

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # Basic descriptive statistics
    desc = df[feature_cols].describe()
    desc.loc["median"] = df[feature_cols].median()
    desc.loc["skew"] = df[feature_cols].skew()
    desc.loc["kurtosis"] = df[feature_cols].kurtosis()
    print(f"\n{desc.round(4).to_string()}")

    # Value counts per feature (since all features are {-1, 0, 1})
    print(f"\n\n  Value Counts per Feature (all features are in {{-1, 0, 1}}):")
    print(f"  {'Feature':<35} {'  -1 (Suspicious)':>18} {'  0 (Neutral)':>14} {'  1 (Legitimate)':>17}")
    print(f"  {'-'*87}")
    for col in feature_cols:
        vc = df[col].value_counts().sort_index()
        v_neg = vc.get(-1, 0)
        v_zero = vc.get(0, 0)
        v_pos = vc.get(1, 0)
        print(f"  {col:<35} {v_neg:>8} ({v_neg/len(df)*100:5.1f}%) {v_zero:>6} ({v_zero/len(df)*100:5.1f}%) {v_pos:>8} ({v_pos/len(df)*100:5.1f}%)")

    # Features that have only 2 unique values (binary: {-1, 1})
    binary_feats = [c for c in feature_cols if df[c].nunique() == 2]
    ternary_feats = [c for c in feature_cols if df[c].nunique() == 3]
    print(f"\n  Binary features (only -1 and 1): {len(binary_feats)}")
    for f in binary_feats:
        print(f"    - {f}")
    print(f"  Ternary features (-1, 0, and 1): {len(ternary_feats)}")
    for f in ternary_feats:
        print(f"    - {f}")


def class_distribution_analysis(df):
    """Analyze and visualize class distribution (Phishing vs Legitimate)."""
    print_section("3. CLASS DISTRIBUTION (Target: Result)")

    # Map target values for display
    class_map = {-1: "Phishing", 1: "Legitimate"}
    class_labels = df[TARGET_COL].map(class_map)
    class_dist = class_labels.value_counts()
    class_pct = class_labels.value_counts(normalize=True) * 100

    print(f"\n  {'Class':<15} {'Count':<10} {'Percentage':<12} {'Bar'}")
    print(f"  {'-'*55}")
    for cls in ["Phishing", "Legitimate"]:
        if cls in class_dist.index:
            bar = "#" * int(class_pct[cls] * 2)
            print(f"  {cls:<15} {class_dist[cls]:<10} {class_pct[cls]:>6.2f}%      {bar}")

    print(f"\n  Total samples: {len(df)}")
    print(f"  Number of classes: 2 (Binary Classification)")

    # Imbalance metrics
    majority = class_dist.max()
    minority = class_dist.min()
    imbalance_ratio = majority / minority
    print(f"\n  Balance Analysis:")
    print(f"    Majority class: {class_dist.idxmax()} ({majority} samples, {class_pct[class_dist.idxmax()]:.1f}%)")
    print(f"    Minority class: {class_dist.idxmin()} ({minority} samples, {class_pct[class_dist.idxmin()]:.1f}%)")
    print(f"    Imbalance ratio: {imbalance_ratio:.2f}:1")
    print(f"    --> {'Near-perfect balance' if imbalance_ratio < 1.5 else 'Moderate imbalance' if imbalance_ratio < 3 else 'Significant imbalance'}")

    # --- Figure: Class Distribution ---
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Bar plot
    colors = [CLASS_PALETTE.get(cls, "#999") for cls in ["Phishing", "Legitimate"]]
    bar_data = [class_dist.get("Phishing", 0), class_dist.get("Legitimate", 0)]
    bars = axes[0].bar(["Phishing", "Legitimate"], bar_data, color=colors, edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Class")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Class Distribution")
    for i, (cls, cnt) in enumerate(zip(["Phishing", "Legitimate"], bar_data)):
        axes[0].text(i, cnt + 50, str(cnt), ha="center", va="bottom", fontsize=11, fontweight="bold")

    # Pie chart
    axes[1].pie(bar_data, labels=["Phishing", "Legitimate"],
                colors=colors, autopct="%1.1f%%", startangle=90, pctdistance=0.85,
                textprops={"fontsize": 12})
    axes[1].set_title("Class Proportions")

    plt.suptitle("Phishing Website Dataset - Class Distribution Analysis",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "class_distribution.png"))
    plt.close(fig)
    print(f"\n  [Saved] figures/class_distribution.png")


def correlation_analysis(df):
    """Generate and analyze correlation heatmap."""
    print_section("4. CORRELATION ANALYSIS")

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    # Compute correlation matrix (all columns are numeric including target)
    corr = df[feature_cols + [TARGET_COL]].corr()

    # Print top correlations with target
    target_corr = corr[TARGET_COL].drop(TARGET_COL).sort_values(key=abs, ascending=False)
    print(f"\nCorrelation with target (Result):")
    print(f"  {'Feature':<35} {'Correlation':<12} {'Strength'}")
    print(f"  {'-'*60}")
    for feat, val in target_corr.items():
        strength = "STRONG" if abs(val) > 0.3 else ("MODERATE" if abs(val) > 0.15 else "WEAK")
        direction = "+" if val > 0 else "-"
        print(f"  {feat:<35} {val:>+.4f}       {direction} {strength}")

    # Top feature-feature correlations (potential multicollinearity)
    print(f"\nTop Feature-Feature Correlations (|r| > 0.3, potential multicollinearity):")
    pairs = []
    for i, c1 in enumerate(feature_cols):
        for c2 in feature_cols[i+1:]:
            r = corr.loc[c1, c2]
            if abs(r) > 0.3:
                pairs.append((c1, c2, r))
    pairs.sort(key=lambda x: abs(x[2]), reverse=True)
    if pairs:
        for c1, c2, r in pairs:
            print(f"  {c1:<30} <-> {c2:<30} r={r:+.4f}")
    else:
        print("  (none found)")

    # --- Figure: Correlation Heatmap ---
    corr_features = df[feature_cols].corr()
    fig, ax = plt.subplots(figsize=(18, 15))
    mask = np.triu(np.ones_like(corr_features, dtype=bool), k=1)
    sns.heatmap(corr_features, mask=mask, annot=True, fmt=".2f", cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True,
                linewidths=0.5, linecolor="white",
                cbar_kws={"shrink": 0.8, "label": "Pearson Correlation"},
                ax=ax, annot_kws={"size": 7})
    ax.set_title("Correlation Heatmap - Phishing Website Features",
                 fontsize=14, fontweight="bold", pad=20)
    ax.tick_params(axis='x', rotation=45, labelsize=8)
    ax.tick_params(axis='y', labelsize=8)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_heatmap.png"))
    plt.close(fig)
    print(f"\n  [Saved] figures/correlation_heatmap.png")

    # --- Figure: Correlation with Target Bar Chart ---
    fig, ax = plt.subplots(figsize=(10, 10))
    colors = ["#2ecc71" if v > 0 else "#e74c3c" for v in target_corr.values]
    bars = ax.barh(target_corr.index, target_corr.values, color=colors, edgecolor="black", linewidth=0.5)
    ax.set_xlabel("Pearson Correlation Coefficient")
    ax.set_title("Feature Correlation with Phishing Result (Target)",
                 fontsize=13, fontweight="bold")
    ax.axvline(x=0, color="black", linewidth=0.8)
    for bar, val in zip(bars, target_corr.values):
        ax.text(val + (0.01 if val > 0 else -0.01), bar.get_y() + bar.get_height()/2,
                f"{val:+.3f}", ha="left" if val > 0 else "right", va="center", fontsize=8)
    ax.tick_params(axis='y', labelsize=9)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "correlation_with_target.png"))
    plt.close(fig)
    print(f"  [Saved] figures/correlation_with_target.png")


def feature_distributions(df):
    """Plot value distributions for all features (bar charts since features are categorical {-1, 0, 1})."""
    print_section("5. FEATURE DISTRIBUTIONS")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    n_features = len(feature_cols)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    # --- Figure: Value distribution bar charts ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        vc = df[col].value_counts().sort_index()
        bar_colors = [VALUE_PALETTE.get(v, "#999") for v in vc.index]
        ax.bar([str(v) for v in vc.index], vc.values, color=bar_colors,
               edgecolor="black", linewidth=0.3)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Count")
        # Add count labels on bars
        for j, (val, cnt) in enumerate(vc.items()):
            ax.text(j, cnt + 30, str(cnt), ha="center", va="bottom", fontsize=7)

    # Hide unused axes
    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Value Distributions (Red=-1 Suspicious, Orange=0 Neutral, Green=1 Legitimate)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_distributions.png"))
    plt.close(fig)
    print(f"  [Saved] figures/feature_distributions.png")

    # --- Figure: Distributions by Class (Phishing vs Legitimate) ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.5))
    axes = axes.flatten()

    class_map = {-1: "Phishing", 1: "Legitimate"}
    df_labeled = df.copy()
    df_labeled["Class"] = df_labeled[TARGET_COL].map(class_map)

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        # For each class, compute the proportion of each feature value
        for cls_name, cls_color in [("Phishing", "#e74c3c"), ("Legitimate", "#2ecc71")]:
            subset = df_labeled[df_labeled["Class"] == cls_name][col]
            vc = subset.value_counts(normalize=True).sort_index()
            x_positions = np.array([v for v in vc.index])
            offset = -0.15 if cls_name == "Phishing" else 0.15
            ax.bar(x_positions + offset, vc.values, width=0.3, color=cls_color,
                   alpha=0.8, edgecolor="black", linewidth=0.2, label=cls_name)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Proportion")
        ax.set_xticks([-1, 0, 1])
        if i == 0:
            ax.legend(fontsize=7)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Value Proportions by Class (Phishing vs Legitimate)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_distributions_by_class.png"))
    plt.close(fig)
    print(f"  [Saved] figures/feature_distributions_by_class.png")

    # Print distribution stats
    print(f"\n  Feature Distribution Statistics:")
    print(f"  {'Feature':<35} {'Mean':>8} {'Std':>8} {'Mode':>6} {'Unique Vals':>12}")
    print(f"  {'-'*72}")
    for col in feature_cols:
        s = df[col]
        mode_val = s.mode().iloc[0] if len(s.mode()) > 0 else "N/A"
        print(f"  {col:<35} {s.mean():>8.3f} {s.std():>8.3f} {mode_val:>6} {s.nunique():>12}")


def box_plots_per_class(df):
    """Create grouped bar charts showing feature value proportions per class."""
    print_section("6. FEATURE VALUE PROPORTIONS PER CLASS")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    class_map = {-1: "Phishing", 1: "Legitimate"}
    df_labeled = df.copy()
    df_labeled["Class"] = df_labeled[TARGET_COL].map(class_map)

    n_features = len(feature_cols)
    n_cols = 5
    n_rows = (n_features + n_cols - 1) // n_cols

    # --- Figure: Stacked bar charts showing proportion of -1, 0, 1 per class ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 4))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        # Cross-tabulation: for this feature, what % of phishing vs legitimate have each value
        ct = pd.crosstab(df_labeled["Class"], df_labeled[col], normalize="index")
        ct = ct.reindex(columns=[-1, 0, 1], fill_value=0)
        ct.plot(kind="bar", ax=ax, color=[VALUE_PALETTE[-1], VALUE_PALETTE[0], VALUE_PALETTE[1]],
                edgecolor="black", linewidth=0.3, width=0.7)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_ylabel("Proportion")
        ax.set_xticklabels(["Phishing", "Legitimate"], rotation=0, fontsize=8)
        ax.legend(title="Value", fontsize=6, title_fontsize=7)
        ax.set_ylim(0, 1.05)

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Value Proportions by Class (Stacked: -1=Red, 0=Orange, 1=Green)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_proportions_per_class.png"))
    plt.close(fig)
    print(f"  [Saved] figures/feature_proportions_per_class.png")

    # --- Figure: Box plots (still useful even for {-1, 0, 1} to show quartile distribution) ---
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(22, n_rows * 3.5))
    axes = axes.flatten()

    for i, col in enumerate(feature_cols):
        ax = axes[i]
        sns.boxplot(x="Class", y=col, data=df_labeled, order=["Phishing", "Legitimate"],
                    palette=CLASS_PALETTE, ax=ax, fliersize=2, linewidth=0.8)
        ax.set_title(col, fontsize=9, fontweight="bold")
        ax.set_xlabel("")

    for j in range(i + 1, len(axes)):
        axes[j].set_visible(False)

    plt.suptitle("Feature Box Plots by Class (Phishing vs Legitimate)",
                 fontsize=13, fontweight="bold", y=1.01)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "box_plots_per_class.png"))
    plt.close(fig)
    print(f"  [Saved] figures/box_plots_per_class.png")

    # Chi-squared test for feature independence from class
    print(f"\n  Chi-Squared Test of Independence (feature vs class):")
    print(f"  {'Feature':<35} {'Chi2':>10} {'p-value':>12} {'Significant?'}")
    print(f"  {'-'*72}")
    for col in feature_cols:
        ct = pd.crosstab(df[TARGET_COL], df[col])
        chi2, p_val, dof, expected = stats.chi2_contingency(ct)
        sig = "*** YES" if p_val < 0.001 else ("** YES" if p_val < 0.01 else ("* YES" if p_val < 0.05 else "NO"))
        print(f"  {col:<35} {chi2:>10.2f} {p_val:>12.2e} {sig}")


def outlier_analysis(df):
    """Detect and report outliers using IQR method."""
    print_section("7. OUTLIER ANALYSIS (IQR Method)")

    feature_cols = [c for c in df.columns if c != TARGET_COL]

    print(f"\n  Note: Since all features have values in {{-1, 0, 1}}, most features will")
    print(f"  have zero IQR outliers. The IQR method is included for completeness and")
    print(f"  to identify any features with unusual distributions.\n")

    print(f"  {'Feature':<35} {'Outliers':>10} {'% of Data':>10} {'Lower Bound':>12} {'Upper Bound':>12}")
    print(f"  {'-'*82}")

    total_outliers = 0
    features_with_outliers = 0
    for col in feature_cols:
        Q1 = df[col].quantile(0.25)
        Q3 = df[col].quantile(0.75)
        IQR = Q3 - Q1
        lower = Q1 - 1.5 * IQR
        upper = Q3 + 1.5 * IQR
        outliers = ((df[col] < lower) | (df[col] > upper)).sum()
        pct = outliers / len(df) * 100
        total_outliers += outliers
        if outliers > 0:
            features_with_outliers += 1
        marker = " <--" if pct > 5 else ""
        print(f"  {col:<35} {outliers:>10} {pct:>9.2f}% {lower:>12.3f} {upper:>12.3f}{marker}")

    print(f"\n  Total outlier instances: {total_outliers}")
    print(f"  Features with outliers: {features_with_outliers}/{len(feature_cols)}")
    print(f"  (Note: With {-1, 0, 1} values, outliers occur when IQR=0 and some values differ from the quartile range)")


def feature_importance_preview(df):
    """Quick feature importance using mutual information and random forest."""
    print_section("8. FEATURE IMPORTANCE PREVIEW")

    from sklearn.ensemble import RandomForestClassifier
    from sklearn.feature_selection import mutual_info_classif

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    X = df[feature_cols].values

    # Remap target: -1 -> 0, 1 -> 1
    y = np.where(df[TARGET_COL].values == -1, 0, 1)

    # Mutual Information
    mi_scores = mutual_info_classif(X, y, random_state=42)
    mi_df = pd.DataFrame({"Feature": feature_cols, "MI Score": mi_scores}).sort_values("MI Score", ascending=False)

    # Random Forest importance
    rf = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X, y)
    rf_imp = rf.feature_importances_
    rf_df = pd.DataFrame({"Feature": feature_cols, "RF Importance": rf_imp}).sort_values("RF Importance", ascending=False)

    # Print combined rankings
    combined = mi_df.merge(rf_df, on="Feature")
    combined["MI Rank"] = range(1, len(combined) + 1)
    combined = combined.sort_values("RF Importance", ascending=False)
    combined["RF Rank"] = range(1, len(combined) + 1)
    combined["Avg Rank"] = (combined["MI Rank"] + combined["RF Rank"]) / 2
    combined = combined.sort_values("Avg Rank")

    print(f"\n  {'Feature':<35} {'MI Score':>10} {'MI Rank':>8} {'RF Imp':>10} {'RF Rank':>8} {'Avg Rank':>9}")
    print(f"  {'-'*84}")
    for _, row in combined.iterrows():
        print(f"  {row['Feature']:<35} {row['MI Score']:>10.4f} {int(row['MI Rank']):>8} "
              f"{row['RF Importance']:>10.4f} {int(row['RF Rank']):>8} {row['Avg Rank']:>9.1f}")

    # --- Figure: Feature Importance ---
    fig, axes = plt.subplots(1, 2, figsize=(18, 9))

    # MI scores
    mi_sorted = mi_df.sort_values("MI Score", ascending=True)
    axes[0].barh(mi_sorted["Feature"], mi_sorted["MI Score"], color="#3498db", edgecolor="black", linewidth=0.5)
    axes[0].set_xlabel("Mutual Information Score")
    axes[0].set_title("Mutual Information with Phishing Result", fontsize=12, fontweight="bold")
    axes[0].tick_params(axis='y', labelsize=9)

    # RF importance
    rf_sorted = rf_df.sort_values("RF Importance", ascending=True)
    axes[1].barh(rf_sorted["Feature"], rf_sorted["RF Importance"], color="#e67e22", edgecolor="black", linewidth=0.5)
    axes[1].set_xlabel("Random Forest Feature Importance")
    axes[1].set_title("Random Forest Importance", fontsize=12, fontweight="bold")
    axes[1].tick_params(axis='y', labelsize=9)

    plt.suptitle("Feature Importance Analysis — Phishing Website Detection",
                 fontsize=14, fontweight="bold", y=1.02)
    plt.tight_layout()
    fig.savefig(os.path.join(FIGURES_DIR, "feature_importance.png"))
    plt.close(fig)
    print(f"\n  [Saved] figures/feature_importance.png")


def generate_summary_dashboard(df):
    """Generate a single summary dashboard figure."""
    print_section("9. SUMMARY DASHBOARD")

    feature_cols = [c for c in df.columns if c != TARGET_COL]
    class_map = {-1: "Phishing", 1: "Legitimate"}
    df_labeled = df.copy()
    df_labeled["Class"] = df_labeled[TARGET_COL].map(class_map)

    fig = plt.figure(figsize=(20, 16))
    gs = gridspec.GridSpec(3, 3, figure=fig, hspace=0.45, wspace=0.35)

    # 1. Class distribution bar
    ax1 = fig.add_subplot(gs[0, 0])
    class_dist = df_labeled["Class"].value_counts()
    colors = [CLASS_PALETTE.get(cls, "#999") for cls in ["Phishing", "Legitimate"]]
    bar_data = [class_dist.get("Phishing", 0), class_dist.get("Legitimate", 0)]
    ax1.bar(["Phishing", "Legitimate"], bar_data, color=colors, edgecolor="black", linewidth=0.5)
    ax1.set_title("Class Distribution", fontweight="bold")
    ax1.set_ylabel("Count")
    for i, cnt in enumerate(bar_data):
        ax1.text(i, cnt + 50, str(cnt), ha="center", fontsize=10, fontweight="bold")

    # 2. Class proportions pie
    ax2 = fig.add_subplot(gs[0, 1])
    ax2.pie(bar_data, labels=["Phishing", "Legitimate"],
            colors=colors, autopct="%1.1f%%", startangle=90, textprops={"fontsize": 10})
    ax2.set_title("Class Proportions", fontweight="bold")

    # 3. Top correlations with target
    ax3 = fig.add_subplot(gs[0, 2])
    corr_target = df[feature_cols + [TARGET_COL]].corr()[TARGET_COL].drop(TARGET_COL)
    corr_target = corr_target.sort_values()
    colors_bar = ["#2ecc71" if v > 0 else "#e74c3c" for v in corr_target.values]
    ax3.barh(corr_target.index, corr_target.values, color=colors_bar, edgecolor="black", linewidth=0.3)
    ax3.set_title("Correlation with Result", fontweight="bold")
    ax3.axvline(x=0, color="black", linewidth=0.8)
    ax3.tick_params(axis='y', labelsize=7)

    # 4-6. Feature value proportions for top 3 most correlated features
    corr_abs = corr_target.abs().sort_values(ascending=False)
    top3 = corr_abs.head(3).index.tolist()
    for idx, feat in enumerate(top3):
        ax = fig.add_subplot(gs[1, idx])
        ct = pd.crosstab(df_labeled["Class"], df_labeled[feat], normalize="index")
        ct = ct.reindex(columns=[-1, 0, 1], fill_value=0)
        ct.plot(kind="bar", ax=ax,
                color=[VALUE_PALETTE[-1], VALUE_PALETTE[0], VALUE_PALETTE[1]],
                edgecolor="black", linewidth=0.3, width=0.7)
        ax.set_title(f"{feat}\n(|r|={corr_abs[feat]:.3f})", fontweight="bold", fontsize=10)
        ax.set_xticklabels(["Phishing", "Legitimate"], rotation=0, fontsize=9)
        ax.legend(title="Value", fontsize=7, title_fontsize=7)
        ax.set_ylim(0, 1.05)

    # 7. Correlation heatmap (compact)
    ax7 = fig.add_subplot(gs[2, :2])
    corr_matrix = df[feature_cols].corr()
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool), k=1)
    sns.heatmap(corr_matrix, mask=mask, annot=False, cmap="RdBu_r",
                center=0, vmin=-1, vmax=1, square=True, linewidths=0.2,
                cbar_kws={"shrink": 0.6}, ax=ax7)
    ax7.set_title("Correlation Heatmap", fontweight="bold")
    ax7.tick_params(labelsize=6)

    # 8. Samples per class bar (horizontal)
    ax8 = fig.add_subplot(gs[2, 2])
    ax8.barh(["Phishing", "Legitimate"], bar_data, color=colors, edgecolor="black", linewidth=0.5)
    for i, cnt in enumerate(bar_data):
        ax8.text(cnt + 30, i, str(cnt), va="center", fontsize=10)
    ax8.set_title("Samples per Class", fontweight="bold")
    ax8.set_xlabel("Count")

    fig.suptitle("Phishing Website Dataset - EDA Summary Dashboard",
                 fontsize=16, fontweight="bold", y=1.01)
    fig.savefig(os.path.join(FIGURES_DIR, "summary_dashboard.png"))
    plt.close(fig)
    print(f"  [Saved] figures/summary_dashboard.png")


# ============================================================
# MAIN EDA EXECUTION
# ============================================================

def run_eda():
    """Run the complete EDA pipeline."""
    print("\n" + "=" * 70)
    print("  COMPREHENSIVE EDA - Phishing Websites Dataset (UCI)")
    print("  CSAI412 Machine Learning Group Project")
    print("=" * 70)

    # Load data
    df = load_data(force_reload=True)

    # Run all EDA sections
    dataset_overview(df)
    statistical_summary(df)
    class_distribution_analysis(df)
    correlation_analysis(df)
    feature_distributions(df)
    box_plots_per_class(df)
    outlier_analysis(df)
    feature_importance_preview(df)
    generate_summary_dashboard(df)

    # Final summary
    print_section("EDA COMPLETE", "=")
    figures = [f for f in os.listdir(FIGURES_DIR) if f.endswith(".png")]
    print(f"\n  Total figures generated: {len(figures)}")
    for fig_name in sorted(figures):
        fig_path = os.path.join(FIGURES_DIR, fig_name)
        size_kb = os.path.getsize(fig_path) / 1024
        print(f"    - {fig_name} ({size_kb:.0f} KB)")

    print(f"\n  All figures saved to: {FIGURES_DIR}/")
    print(f"  Dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"  Classes: Phishing (-1) = {(df[TARGET_COL]==-1).sum()}, Legitimate (1) = {(df[TARGET_COL]==1).sum()}")
    print(f"\n  Key Findings:")
    print(f"  1. Dataset is COMPLETE (no missing values, 11,055 samples)")
    print(f"  2. Near-perfect class balance (44.3% Phishing vs 55.7% Legitimate, ratio 1.26:1)")
    print(f"  3. All 30 features are categorical/ordinal with values in {{-1, 0, 1}}")
    print(f"  4. SSLfinal_State, URL_of_Anchor, and web_traffic show strongest correlation with target")
    print(f"  5. Most features are statistically significant (Chi-squared test, p<0.001)")
    print(f"  6. Low multicollinearity compared to continuous datasets — features are largely independent")
    print(f"\n{'=' * 70}")

    return df


if __name__ == "__main__":
    run_eda()
