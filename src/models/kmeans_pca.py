"""
K-Means Clustering & PCA Analysis -- Phishing Website Dataset
================================================================
CSAI412 Machine Learning Group Project

This module performs UNSUPERVISED analysis on the Phishing Website dataset:
  Part 1: Principal Component Analysis (PCA) -- dimensionality reduction & visualization
  Part 2: K-Means Clustering -- discovering natural groupings & comparing to true labels

Key question: Do phishing and legitimate websites naturally cluster into
              distinct groups based on their URL/website features, even without
              using the class labels?
"""

import os
import sys
import warnings
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # non-interactive backend for saving figures
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import seaborn as sns
from mpl_toolkits.mplot3d import Axes3D  # noqa: F401  (needed for 3D projection)

from sklearn.decomposition import PCA
from sklearn.cluster import KMeans
from sklearn.metrics import (
    silhouette_score,
    silhouette_samples,
    adjusted_rand_score,
    normalized_mutual_info_score,
    homogeneity_completeness_v_measure,
)
import joblib

# -- Project paths -------------------------------------------------------------
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
sys.path.insert(0, os.path.join(PROJECT_DIR, "src"))

from data_loader import get_train_test

FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
os.makedirs(FIGURES_DIR, exist_ok=True)
os.makedirs(DATA_DIR, exist_ok=True)

# Suppress minor warnings for cleaner output
warnings.filterwarnings("ignore", category=FutureWarning)

# Class labels for display
CLASS_NAMES = ["Phishing", "Legitimate"]

# -- Plotting defaults ---------------------------------------------------------
plt.rcParams.update({
    "figure.dpi": 150,
    "savefig.dpi": 150,
    "savefig.bbox": "tight",
    "font.size": 11,
    "axes.titlesize": 13,
    "axes.labelsize": 11,
})

# ==============================================================================
#  PART 1 -- PCA ANALYSIS
# ==============================================================================

def run_pca_analysis(X: np.ndarray, y: np.ndarray, feature_names: list) -> PCA:
    """Full PCA analysis: variance plots, scatter plots, biplots, loading analysis."""

    print("\n" + "=" * 70)
    print("  PART 1 -- PRINCIPAL COMPONENT ANALYSIS (PCA)")
    print("=" * 70)

    n_features = X.shape[1]
    pca_full = PCA(n_components=n_features, random_state=42)
    X_pca_full = pca_full.fit_transform(X)

    evr = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(evr)

    # -- 1. Explained variance ratio bar chart ---------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    bars = ax.bar(range(1, n_features + 1), evr * 100, color="#4C72B0", edgecolor="white")
    ax.set_xlabel("Principal Component")
    ax.set_ylabel("Explained Variance (%)")
    ax.set_title("PCA -- Explained Variance Ratio per Component\n(Phishing Website Features)")
    ax.set_xticks(range(1, n_features + 1))
    ax.tick_params(axis='x', labelsize=8)
    for bar, v in zip(bars, evr):
        if v > 0.03:
            ax.text(bar.get_x() + bar.get_width() / 2, bar.get_height() + 0.3,
                    f"{v*100:.1f}%", ha="center", va="bottom", fontsize=7)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_explained_variance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[PCA] Saved explained variance plot -> {path}")

    # -- 2. Cumulative explained variance --------------------------------------
    fig, ax = plt.subplots(figsize=(12, 5))
    ax.plot(range(1, n_features + 1), cumulative * 100, "o-", color="#DD8452", linewidth=2, markersize=7)
    ax.axhline(y=95, color="red", linestyle="--", alpha=0.7, label="95% threshold")
    n_95 = int(np.argmax(cumulative >= 0.95)) + 1
    ax.axvline(x=n_95, color="green", linestyle="--", alpha=0.7, label=f"{n_95} components for 95%")
    ax.fill_between(range(1, n_features + 1), cumulative * 100, alpha=0.15, color="#DD8452")
    ax.set_xlabel("Number of Principal Components")
    ax.set_ylabel("Cumulative Explained Variance (%)")
    ax.set_title("PCA -- Cumulative Explained Variance\n(Phishing Website Features)")
    ax.set_xticks(range(1, n_features + 1))
    ax.tick_params(axis='x', labelsize=8)
    ax.legend(loc="lower right")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_cumulative_variance.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[PCA] Saved cumulative variance plot -> {path}")

    print(f"\n[PCA] Explained variance per component:")
    for i, (v, c) in enumerate(zip(evr, cumulative)):
        marker = "  <-- 95% reached" if i + 1 == n_95 else ""
        print(f"  PC{i+1:>2}: {v*100:6.2f}%  (cumulative: {c*100:6.2f}%){marker}")
    print(f"\n[PCA] Components needed for 95% variance: {n_95} out of {n_features}")

    # -- 3. 2D PCA scatter plot colored by true labels -------------------------
    unique_classes = sorted(np.unique(y))
    colors = ["#d32f2f", "#388e3c"]  # Red for phishing, green for legitimate

    fig, ax = plt.subplots(figsize=(10, 8))
    for idx, cls in enumerate(unique_classes):
        mask = y == cls
        ax.scatter(X_pca_full[mask, 0], X_pca_full[mask, 1],
                   c=colors[idx], label=CLASS_NAMES[cls], alpha=0.4, s=20, edgecolors="none")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    ax.set_title("PCA -- 2D Projection Colored by True Label\n(Phishing Website Detection)")
    ax.legend(title="Class", markerscale=2, fontsize=10)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_2d_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[PCA] Saved 2D scatter plot -> {path}")

    # -- 4. 3D PCA scatter plot ------------------------------------------------
    fig = plt.figure(figsize=(12, 9))
    ax = fig.add_subplot(111, projection="3d")
    for idx, cls in enumerate(unique_classes):
        mask = y == cls
        ax.scatter(X_pca_full[mask, 0], X_pca_full[mask, 1], X_pca_full[mask, 2],
                   c=colors[idx], label=CLASS_NAMES[cls], alpha=0.3, s=15, edgecolors="none")
    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    ax.set_zlabel(f"PC3 ({evr[2]*100:.1f}%)")
    ax.set_title("PCA -- 3D Projection Colored by True Label\n(Phishing Website Detection)")
    ax.legend(title="Class", fontsize=9, markerscale=2, loc="upper left")
    ax.view_init(elev=25, azim=135)
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_3d_scatter.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[PCA] Saved 3D scatter plot -> {path}")

    # -- 5. PCA component loading analysis -------------------------------------
    print(f"\n[PCA] Component Loadings (top features per PC):")
    loadings = pca_full.components_  # shape: (n_components, n_features)
    n_pcs_to_show = min(5, n_features)
    loading_df = pd.DataFrame(
        loadings[:n_pcs_to_show].T,
        index=feature_names,
        columns=[f"PC{i+1}" for i in range(n_pcs_to_show)],
    )
    for pc_idx in range(n_pcs_to_show):
        pc_name = f"PC{pc_idx+1}"
        abs_loadings = np.abs(loading_df[pc_name])
        top_features = abs_loadings.sort_values(ascending=False).head(5)
        print(f"\n  {pc_name} ({evr[pc_idx]*100:.1f}% variance) -- Top contributing features:")
        for feat, val in top_features.items():
            sign = "+" if loading_df.loc[feat, pc_name] > 0 else "-"
            print(f"    {sign} {feat:<35s}  |loading| = {val:.4f}")

    # -- 6. Loading heatmap for first 5 PCs ------------------------------------
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(loading_df, annot=True, fmt=".2f", cmap="RdBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Loading"},
                annot_kws={"size": 7})
    ax.set_title("PCA Component Loadings (PC1-PC5)\n(Phishing Website Features)")
    ax.set_ylabel("Feature")
    ax.set_xlabel("Principal Component")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_loadings_heatmap.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[PCA] Saved loadings heatmap -> {path}")

    # -- 7. Biplot (scores + loadings) -----------------------------------------
    fig, ax = plt.subplots(figsize=(12, 9))

    # Scores (data points in PC1-PC2 space)
    for idx, cls in enumerate(unique_classes):
        mask = y == cls
        ax.scatter(X_pca_full[mask, 0], X_pca_full[mask, 1],
                   c=colors[idx], label=CLASS_NAMES[cls], alpha=0.2, s=10, edgecolors="none")

    # Feature loading arrows
    score_scale = max(np.abs(X_pca_full[:, :2]).max(), 1)
    arrow_scale = score_scale * 0.8
    for i, feat in enumerate(feature_names):
        ax.annotate(
            "", xy=(loadings[0, i] * arrow_scale, loadings[1, i] * arrow_scale), xytext=(0, 0),
            arrowprops=dict(arrowstyle="->", color="red", lw=1.5),
        )
        ax.text(loadings[0, i] * arrow_scale * 1.08,
                loadings[1, i] * arrow_scale * 1.08,
                feat, color="red", fontsize=6, ha="center", va="center",
                fontweight="bold",
                bbox=dict(boxstyle="round,pad=0.15", fc="white", ec="none", alpha=0.7))

    ax.set_xlabel(f"PC1 ({evr[0]*100:.1f}% variance)")
    ax.set_ylabel(f"PC2 ({evr[1]*100:.1f}% variance)")
    ax.set_title("PCA Biplot -- Data Scores + Feature Loadings\n(Phishing Website Detection)")
    ax.legend(title="Class", markerscale=2, fontsize=9, loc="upper right")
    ax.axhline(0, color="gray", linewidth=0.5, linestyle="--")
    ax.axvline(0, color="gray", linewidth=0.5, linestyle="--")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "pca_biplot.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[PCA] Saved biplot -> {path}")

    # -- Commentary ------------------------------------------------------------
    print("\n" + "-" * 70)
    print("  PCA COMMENTARY")
    print("-" * 70)
    print(f"""
  * The first two principal components capture {cumulative[1]*100:.1f}% of total variance.
    This means a 2D projection preserves roughly {cumulative[1]*100:.0f}% of the information
    in the original {n_features}-dimensional feature space.

  * {n_95} components are needed to retain 95% of the variance. With 30 ternary
    features, this indicates moderate redundancy among the URL/website
    indicators -- some features capture overlapping aspects of phishing
    behavior (e.g., URL_Length and Shortining_Service may correlate).

  * The 2D scatter plot shows the degree of OVERLAP between phishing and
    legitimate websites. Clear separation suggests that URL/website features
    effectively distinguish the two classes; overlap indicates that some
    phishing sites mimic legitimate characteristics (or vice versa).

  * The biplot reveals which URL/website properties drive the main axes of
    variation. Features pointing in similar directions are correlated;
    features pointing in opposite directions are anti-correlated.

  * For phishing detection, the PCA analysis helps understand:
    - Which features contain the most discriminative information
    - How much of the phishing signal is redundant across features
    - The fundamental separability of phishing vs legitimate sites
""")

    return pca_full


# ==============================================================================
#  PART 2 -- K-MEANS CLUSTERING
# ==============================================================================

def run_kmeans_analysis(X: np.ndarray, y: np.ndarray, feature_names: list, pca: PCA):
    """Full K-Means analysis: elbow, silhouette, clustering, comparison with true labels."""

    print("\n" + "=" * 70)
    print("  PART 2 -- K-MEANS CLUSTERING")
    print("=" * 70)
    print("\n  NOTE: K-Means is UNSUPERVISED -- no class labels are used during")
    print("  clustering. Labels are only used AFTER clustering to evaluate how")
    print("  well the discovered clusters correspond to phishing/legitimate.\n")

    k_range = range(2, 12)
    inertias = []
    silhouette_scores_list = []

    print("[K-Means] Running K-Means for k = 2..11 ...")
    for k in k_range:
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        km.fit(X)
        inertias.append(km.inertia_)
        sil = silhouette_score(X, km.labels_)
        silhouette_scores_list.append(sil)
        print(f"  k={k:>2d}  |  Inertia: {km.inertia_:>12.1f}  |  Silhouette: {sil:.4f}")

    # -- 1. Elbow method plot --------------------------------------------------
    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), inertias, "o-", color="#4C72B0", linewidth=2, markersize=7)
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Inertia (Within-Cluster Sum of Squares)")
    ax.set_title("K-Means -- Elbow Method\n(Phishing Website Features)")
    ax.set_xticks(list(k_range))

    # Compute second derivative to find elbow automatically
    diffs_1 = np.diff(inertias)
    diffs_2 = np.diff(diffs_1)
    elbow_k = list(k_range)[np.argmax(diffs_2) + 2]
    ax.axvline(x=elbow_k, color="red", linestyle="--", alpha=0.7, label=f"Elbow ~ k={elbow_k}")
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "kmeans_elbow.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n[K-Means] Saved elbow plot -> {path}")
    print(f"[K-Means] Estimated elbow at k={elbow_k}")

    # -- 2. Silhouette score plot ----------------------------------------------
    best_k_sil = list(k_range)[np.argmax(silhouette_scores_list)]
    best_sil = max(silhouette_scores_list)

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(list(k_range), silhouette_scores_list, "s-", color="#55A868", linewidth=2, markersize=7)
    ax.axvline(x=best_k_sil, color="red", linestyle="--", alpha=0.7,
               label=f"Best k={best_k_sil} (sil={best_sil:.4f})")
    ax.set_xlabel("Number of Clusters (k)")
    ax.set_ylabel("Silhouette Score")
    ax.set_title("K-Means -- Silhouette Score vs. k\n(Phishing Website Features)")
    ax.set_xticks(list(k_range))
    ax.legend()
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "kmeans_silhouette.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[K-Means] Saved silhouette plot -> {path}")
    print(f"[K-Means] Best silhouette score: k={best_k_sil} (score={best_sil:.4f})")

    # -- 3. Silhouette diagram for selected k values ---------------------------
    n_true_classes = len(np.unique(y))
    selected_ks = sorted(set([best_k_sil, elbow_k, n_true_classes]))
    fig, axes = plt.subplots(1, len(selected_ks), figsize=(6 * len(selected_ks), 6))
    if len(selected_ks) == 1:
        axes = [axes]

    for ax_idx, k in enumerate(selected_ks):
        ax = axes[ax_idx]
        km = KMeans(n_clusters=k, init="k-means++", n_init=10, max_iter=300, random_state=42)
        labels = km.fit_predict(X)
        sil_avg = silhouette_score(X, labels)
        sample_sils = silhouette_samples(X, labels)

        y_lower = 10
        cmap_sil = matplotlib.colormaps.get_cmap("tab10")
        for i in range(k):
            ith_sils = sample_sils[labels == i]
            ith_sils.sort()
            size_i = ith_sils.shape[0]
            y_upper = y_lower + size_i
            color = cmap_sil(i / max(k - 1, 1))
            ax.fill_betweenx(np.arange(y_lower, y_upper), 0, ith_sils, facecolor=color, alpha=0.7)
            ax.text(-0.05, y_lower + 0.5 * size_i, str(i), fontsize=8)
            y_lower = y_upper + 10

        ax.axvline(x=sil_avg, color="red", linestyle="--", linewidth=1.5)
        ax.set_title(f"k={k}  (avg sil={sil_avg:.3f})")
        ax.set_xlabel("Silhouette Coefficient")
        ax.set_ylabel("Cluster Label (sorted)")
        ax.set_yticks([])

    fig.suptitle("K-Means -- Silhouette Diagrams (Phishing Data)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "kmeans_silhouette_diagrams.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[K-Means] Saved silhouette diagrams -> {path}")

    # -- 4. Choose the best k --------------------------------------------------
    best_k = best_k_sil
    print(f"\n[K-Means] Using best k={best_k} (silhouette-optimal) for primary analysis")
    print(f"[K-Means] Also analyzing k={n_true_classes} (matching {n_true_classes} true classes: {CLASS_NAMES})")

    # -- 5. Final clustering with best k ---------------------------------------
    km_best = KMeans(n_clusters=best_k, init="k-means++", n_init=20, max_iter=500, random_state=42)
    labels_best = km_best.fit_predict(X)

    # Also run with k=2 (matching true number of classes)
    km_2 = KMeans(n_clusters=n_true_classes, init="k-means++", n_init=20, max_iter=500, random_state=42)
    labels_2 = km_2.fit_predict(X)

    # -- 6. Cluster visualization in PCA-reduced 2D space ----------------------
    X_pca_2d = pca.transform(X)[:, :2]
    evr = pca.explained_variance_ratio_

    fig, axes = plt.subplots(1, 3, figsize=(20, 6))

    colors_true = ["#d32f2f", "#388e3c"]

    # (a) True labels
    unique_classes = sorted(np.unique(y))
    for idx, cls in enumerate(unique_classes):
        mask = y == cls
        axes[0].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                        c=colors_true[idx], label=CLASS_NAMES[cls], alpha=0.4, s=12, edgecolors="none")
    axes[0].set_title("True Labels (Phishing / Legitimate)")
    axes[0].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    axes[0].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    axes[0].legend(title="Class", fontsize=9, markerscale=2)

    # (b) Best k clusters
    cmap_clust = matplotlib.colormaps.get_cmap("tab10")
    for c in range(best_k):
        mask = labels_best == c
        axes[1].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                        c=[cmap_clust(c)], label=f"C{c}", alpha=0.4, s=12, edgecolors="none")
    # Mark cluster centers
    centers_2d = pca.transform(km_best.cluster_centers_)[:, :2]
    axes[1].scatter(centers_2d[:, 0], centers_2d[:, 1], c="black", marker="X", s=120,
                    edgecolors="white", linewidths=1.5, label="Centers", zorder=5)
    axes[1].set_title(f"K-Means (k={best_k}, best silhouette)")
    axes[1].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    axes[1].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    axes[1].legend(title="Cluster", fontsize=9, markerscale=2)

    # (c) k=2 clusters (matching true classes)
    for c in range(n_true_classes):
        mask = labels_2 == c
        axes[2].scatter(X_pca_2d[mask, 0], X_pca_2d[mask, 1],
                        c=[cmap_clust(c)], label=f"C{c}", alpha=0.4, s=12, edgecolors="none")
    centers_2_2d = pca.transform(km_2.cluster_centers_)[:, :2]
    axes[2].scatter(centers_2_2d[:, 0], centers_2_2d[:, 1], c="black", marker="X", s=120,
                    edgecolors="white", linewidths=1.5, label="Centers", zorder=5)
    axes[2].set_title(f"K-Means (k={n_true_classes}, matching true classes)")
    axes[2].set_xlabel(f"PC1 ({evr[0]*100:.1f}%)")
    axes[2].set_ylabel(f"PC2 ({evr[1]*100:.1f}%)")
    axes[2].legend(title="Cluster", fontsize=9, markerscale=2)

    fig.suptitle("Cluster Visualization in PCA-Reduced 2D Space\n(Phishing Website Detection)", fontsize=14, fontweight="bold")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "kmeans_clusters_2d.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"[K-Means] Saved cluster visualization -> {path}")

    # -- 7. Compare clusters vs true labels ------------------------------------
    print("\n" + "-" * 70)
    print("  CLUSTER vs. TRUE LABEL COMPARISON")
    print("-" * 70)

    for name, labels_pred, km_model, k_val in [
        (f"Best k={best_k}", labels_best, km_best, best_k),
        (f"k={n_true_classes} (matching classes)", labels_2, km_2, n_true_classes),
    ]:
        print(f"\n  -- {name} --")

        # Cross-tabulation
        ct = pd.crosstab(
            pd.Series(labels_pred, name="Cluster"),
            pd.Series(y, name="True Class"),
        )
        # Rename columns for readability
        ct.columns = [CLASS_NAMES[c] for c in ct.columns]
        print(f"\n  Cross-tabulation (Cluster x True Class):")
        print(ct.to_string().replace("\n", "\n  "))

        # External clustering metrics
        ari = adjusted_rand_score(y, labels_pred)
        nmi = normalized_mutual_info_score(y, labels_pred)
        homo, comp, v_measure = homogeneity_completeness_v_measure(y, labels_pred)
        sil = silhouette_score(X, labels_pred)

        print(f"\n  Clustering Quality Metrics:")
        print(f"    Adjusted Rand Index (ARI):          {ari:.4f}  (1.0 = perfect, 0 = random)")
        print(f"    Normalized Mutual Information (NMI): {nmi:.4f}  (1.0 = perfect agreement)")
        print(f"    Homogeneity:                         {homo:.4f}  (each cluster contains one class)")
        print(f"    Completeness:                        {comp:.4f}  (each class in one cluster)")
        print(f"    V-measure:                           {v_measure:.4f}  (harmonic mean of H & C)")
        print(f"    Silhouette Score:                    {sil:.4f}  (cluster cohesion & separation)")

    # -- 8. Cluster center analysis --------------------------------------------
    print("\n" + "-" * 70)
    print("  CLUSTER CENTER ANALYSIS (Best k)")
    print("-" * 70)

    centers_df = pd.DataFrame(km_best.cluster_centers_, columns=feature_names)
    centers_df.index.name = "Cluster"
    print(f"\n  Cluster centers (feature values):")
    print(centers_df.round(3).to_string().replace("\n", "\n  "))

    # Heatmap of cluster centers
    fig, ax = plt.subplots(figsize=(14, max(5, best_k * 0.8)))
    sns.heatmap(centers_df, annot=True, fmt=".2f", cmap="RdYlBu_r", center=0,
                linewidths=0.5, ax=ax, cbar_kws={"label": "Feature Value"},
                annot_kws={"size": 7})
    ax.set_title(f"K-Means Cluster Centers (k={best_k}) -- Feature Profiles\n(Phishing Website Detection)")
    ax.set_ylabel("Cluster")
    ax.set_xlabel("Feature")
    plt.tight_layout()
    path = os.path.join(FIGURES_DIR, "kmeans_cluster_centers.png")
    fig.savefig(path)
    plt.close(fig)
    print(f"\n[K-Means] Saved cluster centers heatmap -> {path}")

    # Characterize each cluster
    print(f"\n  Cluster Characterization (features with |value| > 0.3):")
    for c in range(best_k):
        row = centers_df.iloc[c]
        notable = row[row.abs() > 0.3].sort_values(key=abs, ascending=False)
        size = np.sum(labels_best == c)
        dominant_class_idx = pd.Series(y[labels_best == c]).mode().values
        dominant_class = [CLASS_NAMES[int(c_)] for c_ in dominant_class_idx]
        print(f"\n  Cluster {c} ({size} samples, dominant class={dominant_class}):")
        for feat, val in notable.head(10).items():
            direction = "LEGIT" if val > 0 else "SUSPICIOUS"
            print(f"    {direction:>10s} {feat:<35s} val={val:+.3f}")

    # -- 9. Save the best K-Means model ----------------------------------------
    model_path = os.path.join(DATA_DIR, "kmeans_model.pkl")
    joblib.dump(km_best, model_path)
    print(f"\n[K-Means] Saved best KMeans model (k={best_k}) -> {model_path}")

    # Also save the k=2 model for reference
    model_path_2 = os.path.join(DATA_DIR, f"kmeans_model_k{n_true_classes}.pkl")
    joblib.dump(km_2, model_path_2)
    print(f"[K-Means] Saved KMeans model (k={n_true_classes}) -> {model_path_2}")

    return km_best, km_2, labels_best, labels_2


# ==============================================================================
#  FINAL COMMENTARY
# ==============================================================================

def print_final_commentary(y, labels_best, labels_2, best_k, pca_full):
    """Print comprehensive commentary on the unsupervised analysis results."""

    evr = pca_full.explained_variance_ratio_
    cumulative = np.cumsum(evr)
    n_95 = int(np.argmax(cumulative >= 0.95)) + 1
    n_true_classes = len(np.unique(y))

    ari_best = adjusted_rand_score(y, labels_best)
    ari_2 = adjusted_rand_score(y, labels_2)
    nmi_best = normalized_mutual_info_score(y, labels_best)
    nmi_2 = normalized_mutual_info_score(y, labels_2)

    print("\n" + "=" * 70)
    print("  FINAL COMMENTARY -- UNSUPERVISED ANALYSIS OF PHISHING DATA")
    print("=" * 70)
    print(f"""
  +---------------------------------------------------------------+
  |  KEY FINDINGS                                                   |
  +---------------------------------------------------------------+

  1. PCA DIMENSIONALITY REDUCTION
     - The first 2 PCs capture {cumulative[1]*100:.1f}% of variance; {n_95} PCs capture 95%.
     - With 30 ternary features, this reveals moderate redundancy among
       the phishing indicators -- some features capture overlapping
       aspects of malicious website behavior.
     - The 2D projection shows how well phishing and legitimate sites
       separate in the principal component space.

  2. K-MEANS CLUSTERING
     - The optimal k={best_k} (by silhouette) {'matches' if best_k == n_true_classes else 'differs from'} the {n_true_classes} true
       classes, {'suggesting natural binary separation' if best_k == n_true_classes else 'suggesting sub-clusters within the phishing/legitimate categories'}.
     - Adjusted Rand Index: best_k -> {ari_best:.4f}, k={n_true_classes} -> {ari_2:.4f}
       (values near 1 indicate good alignment with true labels).
     - NMI: best_k -> {nmi_best:.4f}, k={n_true_classes} -> {nmi_2:.4f}
       (values closer to 1 indicate better cluster-to-class correspondence).

  3. DO PHISHING SITES NATURALLY CLUSTER?
     - {'High' if ari_2 > 0.3 else 'Moderate' if ari_2 > 0.1 else 'Low'} ARI ({ari_2:.4f}) for k={n_true_classes} indicates that phishing and
       legitimate sites {'do' if ari_2 > 0.3 else 'partially'} form distinct clusters based on their
       URL/website feature profiles.
     - This {'supports' if ari_2 > 0.3 else 'partially supports'} the hypothesis that phishing websites share
       common structural patterns that differentiate them from legitimate
       sites, even without using class labels during clustering.

  4. STRENGTHS OF THIS UNSUPERVISED APPROACH
     - Discovers natural groupings without label bias
     - Reveals the intrinsic structure of the URL feature space
     - PCA identifies which features drive the most variation
     - Shows how well phishing/legitimate sites separate
     - Cluster profiles reveal characteristic feature patterns

  5. WEAKNESSES AND LIMITATIONS
     - K-Means assumes spherical clusters -- phishing clusters may be
       irregularly shaped (DBSCAN or GMM might find different structure)
     - Must specify k in advance
     - Silhouette score can be misleading in high dimensions
     - PCA is linear -- t-SNE or UMAP might reveal additional non-linear
       structure
     - The ternary feature space creates a discrete lattice, which may
       not align well with K-Means' continuous centroid updates

  6. HOW THIS COMPLEMENTS SUPERVISED MODELS
     - The cluster-to-class alignment shows how much of the phishing
       distinction is captured by feature proximity alone
     - The PCA variance analysis guides feature engineering
     - Cluster membership could be used as an additional feature
     - The identified cluster structure helps interpret misclassifications

  +---------------------------------------------------------------+
  |  BOTTOM LINE: {'Phishing and legitimate sites DO form' if ari_2 > 0.3 else 'Phishing and legitimate sites PARTIALLY form'}     |
  |  {'distinct clusters, validating that the URL features capture' if ari_2 > 0.3 else 'distinct clusters. The URL features capture some but not all'}  |
  |  {'meaningful differences between malicious and legitimate sites.' if ari_2 > 0.3 else 'of the structural differences between the two categories.'}  |
  +---------------------------------------------------------------+
""")


# ==============================================================================
#  MAIN
# ==============================================================================

def main():
    """Run the complete PCA + K-Means unsupervised analysis pipeline."""

    print("=" * 70)
    print("  K-MEANS CLUSTERING & PCA ANALYSIS")
    print("  Phishing Website Dataset -- Unsupervised Analysis")
    print("=" * 70)

    # -- Load data (using get_train_test, combine train+test for unsupervised) -
    print("\n[Data] Loading phishing website dataset...")
    data = get_train_test()

    # Combine train and test for unsupervised analysis (no data leakage concern
    # since K-Means and PCA don't use labels during fitting)
    X = np.vstack([data["X_train"], data["X_test"]])
    y = np.concatenate([data["y_train"], data["y_test"]])
    feature_names = data["feature_names"]

    print(f"\n[Data] Dataset summary:")
    print(f"  Samples:  {X.shape[0]}")
    print(f"  Features: {X.shape[1]}")
    print(f"  Classes:  {sorted(np.unique(y))} ({len(np.unique(y))} classes: {CLASS_NAMES})")
    class_counts = pd.Series(y).value_counts().sort_index()
    print(f"  Class distribution:")
    for cls, count in class_counts.items():
        print(f"    {CLASS_NAMES[cls]} ({cls}): {count:>5d} ({count/len(y)*100:5.1f}%)")

    # -- Part 1: PCA ----------------------------------------------------------
    pca_full = run_pca_analysis(X, y, feature_names)

    # -- Part 2: K-Means ------------------------------------------------------
    km_best, km_2, labels_best, labels_2 = run_kmeans_analysis(X, y, feature_names, pca_full)

    best_k = km_best.n_clusters

    # -- Final Commentary ------------------------------------------------------
    print_final_commentary(y, labels_best, labels_2, best_k, pca_full)

    # -- Summary of saved artifacts --------------------------------------------
    n_true_classes = len(np.unique(y))
    print("\n" + "=" * 70)
    print("  SAVED ARTIFACTS")
    print("=" * 70)
    artifacts = [
        ("figures/pca_explained_variance.png", "PCA explained variance bar chart"),
        ("figures/pca_cumulative_variance.png", "PCA cumulative variance (95% threshold)"),
        ("figures/pca_2d_scatter.png", "PCA 2D scatter (Phishing vs Legitimate)"),
        ("figures/pca_3d_scatter.png", "PCA 3D scatter (Phishing vs Legitimate)"),
        ("figures/pca_loadings_heatmap.png", "PCA component loadings heatmap"),
        ("figures/pca_biplot.png", "PCA biplot (scores + feature loadings)"),
        ("figures/kmeans_elbow.png", "K-Means elbow method plot"),
        ("figures/kmeans_silhouette.png", "K-Means silhouette score vs k"),
        ("figures/kmeans_silhouette_diagrams.png", "Silhouette diagrams for selected k"),
        ("figures/kmeans_clusters_2d.png", "Cluster visualization in PCA 2D space"),
        ("figures/kmeans_cluster_centers.png", "Cluster center feature profiles"),
        ("data/kmeans_model.pkl", f"Best KMeans model (k={best_k})"),
        (f"data/kmeans_model_k{n_true_classes}.pkl", f"KMeans model (k={n_true_classes})"),
    ]
    for fpath, desc in artifacts:
        full = os.path.join(PROJECT_DIR, fpath)
        status = "OK" if os.path.exists(full) else "MISSING"
        print(f"  [{status}] {fpath:<50s} -- {desc}")

    print("\n" + "=" * 70)
    print("  UNSUPERVISED ANALYSIS COMPLETE")
    print("=" * 70)


if __name__ == "__main__":
    main()
