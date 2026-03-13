"""
Phishing Website Dataset Loader & Preprocessor
================================================
CSAI412 Machine Learning Group Project

Dataset: UCI Phishing Websites Dataset (Mohammad, Thabtah & McCluskey)
- Samples:  11,055
- Features: 30 (URL-based and website-based characteristics)
- Target:   Result (-1 = Phishing, 1 = Legitimate)
- Missing:  None (after dropping junk column "Unnamed: 31")

All features are integer-valued with values in {-1, 0, 1} representing:
  -1 = Suspicious / Phishing indicator
   0 = Neutral / Uncertain
   1 = Legitimate indicator

Reference: R.M. Mohammad, F. Thabtah, L. McCluskey. Phishing Websites Features.
           University of Huddersfield.
"""

import os
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import joblib

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(PROJECT_DIR, "data")
FIGURES_DIR = os.path.join(PROJECT_DIR, "figures")

# Ensure directories exist
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(FIGURES_DIR, exist_ok=True)

# Target column name and class names
# Original target: -1 = Phishing, 1 = Legitimate
# After remapping: 0 = Phishing, 1 = Legitimate
TARGET_COL = "Result"
CLASS_NAMES = ["Phishing", "Legitimate"]  # index 0 = Phishing, index 1 = Legitimate
NUM_CLASSES = 2

# Feature descriptions (all 30 features)
FEATURE_DESCRIPTIONS = {
    "having_IP_Address": "Whether the URL uses an IP address instead of a domain name (-1=yes, 1=no)",
    "URL_Length": "Length of the URL (-1=long/suspicious, 0=medium, 1=short/legitimate)",
    "Shortining_Service": "Whether a URL shortening service (e.g., TinyURL) is used (-1=yes, 1=no)",
    "having_At_Symbol": "Whether the URL contains an '@' symbol (-1=yes, 1=no)",
    "double_slash_redirecting": "Whether '//' appears in the URL path for redirection (-1=yes, 1=no)",
    "Prefix_Suffix": "Whether the domain name contains a '-' (dash) (-1=yes, 1=no)",
    "having_Sub_Domain": "Number of dots in the domain indicating subdomains (-1=many, 0=one, 1=none)",
    "SSLfinal_State": "SSL certificate trust level (-1=untrusted, 0=suspicious, 1=trusted)",
    "Domain_registeration_length": "Domain registration duration (-1=short, 1=long)",
    "Favicon": "Whether the favicon is loaded from an external domain (-1=yes, 1=no)",
    "port": "Whether a non-standard port is used (-1=yes, 1=no)",
    "HTTPS_token": "Whether 'HTTPS' token appears in the domain part of the URL (-1=yes, 1=no)",
    "Request_URL": "Percentage of external objects (images, videos) loaded from other domains (-1=high, 0=medium, 1=low)",
    "URL_of_Anchor": "Percentage of anchor tags with different domain or no link (-1=high, 0=medium, 1=low)",
    "Links_in_tags": "Percentage of links in <meta>, <script>, and <link> from different domains (-1=high, 0=medium, 1=low)",
    "SFH": "Server Form Handler — whether form action points to a different domain or is blank (-1=suspicious, 0=about:blank, 1=legitimate)",
    "Submitting_to_email": "Whether the form submits data to an email address (-1=yes, 1=no)",
    "Abnormal_URL": "Whether the URL hostname is not in the WHOIS record (-1=yes, 1=no)",
    "Redirect": "Number of redirects (0=few, 1=none; -1 not used here but kept for consistency)",
    "on_mouseover": "Whether onMouseOver changes the status bar (-1=yes, 1=no)",
    "RightClick": "Whether right-click is disabled (-1=yes, 1=no)",
    "popUpWidnow": "Whether pop-up windows contain text fields (-1=yes, 1=no)",
    "Iframe": "Whether the page uses invisible iframes (-1=yes, 1=no)",
    "age_of_domain": "Age of the domain (-1=young/suspicious, 1=old/legitimate)",
    "DNSRecord": "Whether the domain has a DNS record (-1=no record, 1=has record)",
    "web_traffic": "Website traffic rank from Alexa (-1=low/no rank, 0=medium, 1=high)",
    "Page_Rank": "Google PageRank score (-1=low, 1=high)",
    "Google_Index": "Whether the page is indexed by Google (-1=not indexed, 1=indexed)",
    "Links_pointing_to_page": "Number of external links pointing to the page (-1=none, 0=few, 1=many)",
    "Statistical_report": "Whether the host belongs to top phishing IPs/domains from statistical reports (-1=yes, 1=no)",
}


def load_data(force_reload=False):
    """
    Load the Phishing Websites dataset.

    Drops the junk column "Unnamed: 31" (caused by a trailing comma in the CSV).

    Returns:
        pd.DataFrame: Dataset with 11,055 rows, 30 feature columns + 'Result' target column
    """
    csv_path = os.path.join(DATA_DIR, "phishing.csv")

    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"Dataset not found at {csv_path}. "
            "Please place the phishing.csv file in the data/ directory."
        )

    df = pd.read_csv(csv_path)

    # Drop the junk column caused by the trailing comma in the CSV header
    if "Unnamed: 31" in df.columns:
        df = df.drop(columns=["Unnamed: 31"])
        print("[DataLoader] Dropped junk column 'Unnamed: 31'")

    print(f"[DataLoader] Loaded Phishing dataset: {df.shape[0]} samples, {df.shape[1]} columns")
    print(f"[DataLoader] Target values: {sorted(df[TARGET_COL].unique())} (-1=Phishing, 1=Legitimate)")

    return df


def preprocess_data(df, scale=True):
    """
    Preprocess the Phishing Websites dataset.

    Steps:
        1. Separate features (X) and target (y)
        2. Remap target from {-1, 1} to {0, 1} where 0=Phishing, 1=Legitimate
        3. Optionally apply StandardScaler to features

    Note: All features are already integer-valued in {-1, 0, 1}, so StandardScaler
    will center and scale them. This is still beneficial for distance-based models
    (KNN, SVM) even though the features are on a similar scale, as it ensures
    zero mean and unit variance across all features.

    Args:
        df (pd.DataFrame): Phishing dataset
        scale (bool): Whether to apply StandardScaler (default: True)

    Returns:
        tuple: (X, y, feature_names, scaler)
            - X: np.ndarray of shape (n_samples, 30) -- scaled if scale=True
            - y: np.ndarray of shape (n_samples,) -- class labels as integers (0=Phishing, 1=Legitimate)
            - feature_names: list of feature column names
            - scaler: fitted StandardScaler (or None if scale=False)
    """
    # Feature columns (all columns except target)
    feature_cols = [col for col in df.columns if col != TARGET_COL]

    X = df[feature_cols].values.astype(np.float64)

    # Remap target: -1 (Phishing) -> 0, 1 (Legitimate) -> 1
    y_raw = df[TARGET_COL].values
    y = np.where(y_raw == -1, 0, 1)

    scaler = None
    if scale:
        scaler = StandardScaler()
        X = scaler.fit_transform(X)
        print(f"[DataLoader] Applied StandardScaler to {len(feature_cols)} features")

    print(f"[DataLoader] Features shape: {X.shape}, Target shape: {y.shape}")
    print(f"[DataLoader] Target classes: {sorted(np.unique(y))} ({CLASS_NAMES})")
    print(f"[DataLoader] Class distribution: Phishing(0)={np.sum(y==0)}, Legitimate(1)={np.sum(y==1)}")
    print(f"[DataLoader] Feature names: {feature_cols}")

    return X, y, feature_cols, scaler


def get_train_test(df=None, test_size=0.2, random_state=42, scale=True, save=True):
    """
    Get train/test split with stratified sampling and optional scaling.

    Args:
        df (pd.DataFrame): Dataset (loads automatically if None)
        test_size (float): Test set proportion (default: 0.2)
        random_state (int): Random seed for reproducibility (default: 42)
        scale (bool): Whether to apply StandardScaler (default: True)
        save (bool): Whether to save processed data to CSVs (default: True)

    Returns:
        dict: {
            'X_train': np.ndarray,
            'X_test': np.ndarray,
            'y_train': np.ndarray,
            'y_test': np.ndarray,
            'feature_names': list,
            'scaler': StandardScaler or None
        }
    """
    if df is None:
        df = load_data()

    X, y, feature_names, scaler = preprocess_data(df, scale=scale)

    # Stratified train/test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=test_size,
        random_state=random_state,
        stratify=y
    )

    print(f"\n[DataLoader] Train/Test Split (stratified, random_state={random_state}):")
    print(f"  Training set:  {X_train.shape[0]} samples ({(1-test_size)*100:.0f}%)")
    print(f"  Test set:      {X_test.shape[0]} samples ({test_size*100:.0f}%)")

    # Verify stratification
    train_dist = pd.Series(y_train).value_counts(normalize=True).sort_index()
    test_dist = pd.Series(y_test).value_counts(normalize=True).sort_index()
    print(f"\n[DataLoader] Class distribution verification:")
    print(f"  {'Class':<12} {'Name':<12} {'Train %':<10} {'Test %':<10}")
    print(f"  {'-'*44}")
    for cls in sorted(np.unique(y)):
        t = train_dist.get(cls, 0) * 100
        te = test_dist.get(cls, 0) * 100
        name = CLASS_NAMES[cls] if cls < len(CLASS_NAMES) else "?"
        print(f"  {cls:<12} {name:<12} {t:<10.1f} {te:<10.1f}")

    if save:
        _save_processed_data(X_train, X_test, y_train, y_test, feature_names, scaler)

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


def _save_processed_data(X_train, X_test, y_train, y_test, feature_names, scaler):
    """Save processed train/test data to CSV and scaler to joblib."""
    # Save train set
    train_df = pd.DataFrame(X_train, columns=feature_names)
    train_df["class_label"] = y_train
    train_path = os.path.join(DATA_DIR, "train.csv")
    train_df.to_csv(train_path, index=False)

    # Save test set
    test_df = pd.DataFrame(X_test, columns=feature_names)
    test_df["class_label"] = y_test
    test_path = os.path.join(DATA_DIR, "test.csv")
    test_df.to_csv(test_path, index=False)

    # Save scaler
    if scaler is not None:
        scaler_path = os.path.join(DATA_DIR, "scaler.joblib")
        joblib.dump(scaler, scaler_path)
        print(f"\n[DataLoader] Saved scaler to: {scaler_path}")

    print(f"[DataLoader] Saved training data to: {train_path}")
    print(f"[DataLoader] Saved test data to: {test_path}")


def load_processed_data():
    """
    Load previously saved processed train/test data.

    Returns:
        dict: Same structure as get_train_test()
    """
    train_path = os.path.join(DATA_DIR, "train.csv")
    test_path = os.path.join(DATA_DIR, "test.csv")
    scaler_path = os.path.join(DATA_DIR, "scaler.joblib")

    if not os.path.exists(train_path) or not os.path.exists(test_path):
        print("[DataLoader] Processed data not found. Running get_train_test()...")
        return get_train_test()

    train_df = pd.read_csv(train_path)
    test_df = pd.read_csv(test_path)

    feature_names = [col for col in train_df.columns if col != "class_label"]

    X_train = train_df[feature_names].values
    y_train = train_df["class_label"].values
    X_test = test_df[feature_names].values
    y_test = test_df["class_label"].values

    scaler = None
    if os.path.exists(scaler_path):
        scaler = joblib.load(scaler_path)

    print(f"[DataLoader] Loaded processed data: {X_train.shape[0]} train, {X_test.shape[0]} test")

    return {
        "X_train": X_train,
        "X_test": X_test,
        "y_train": y_train,
        "y_test": y_test,
        "feature_names": feature_names,
        "scaler": scaler,
    }


# ---- Quick test when run directly ----
if __name__ == "__main__":
    print("=" * 60)
    print("Phishing Website Dataset Loader - Test Run")
    print("=" * 60)

    # Load raw data
    df = load_data(force_reload=True)
    print(f"\nDataset shape: {df.shape}")
    print(f"Columns: {list(df.columns)}")
    print(f"\nFirst 5 rows:")
    print(df.head())
    print(f"\nTarget distribution (original values):")
    print(df[TARGET_COL].value_counts().sort_index())

    # Get train/test split
    print("\n" + "=" * 60)
    data = get_train_test(df)

    print("\n" + "=" * 60)
    print("Data loader test complete!")
    print("=" * 60)
