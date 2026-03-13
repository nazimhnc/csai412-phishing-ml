# CSAI412 Machine Learning Group Project: Phishing Website Detection

## Dataset

**Phishing Websites Dataset** from the [UCI Machine Learning Repository](https://archive.ics.uci.edu/dataset/327/phishing+websites).

| Property | Value |
|----------|-------|
| **Samples** | 11,055 |
| **Features** | 30 (URL-based and website-based characteristics) |
| **Target** | `Result` (binary: -1 = Phishing, 1 = Legitimate) |
| **Missing Values** | None |
| **Class Balance** | Near-perfect — 4,898 Phishing (44.3%) vs 6,157 Legitimate (55.7%), ratio 1.26:1 |
| **Feature Type** | All integer-valued: {-1, 0, 1} representing suspicious/neutral/legitimate states |

### Features

All 30 features are categorical/ordinal with values in {-1, 0, 1}:
- **-1** = Suspicious / Phishing indicator
- **0** = Neutral / Uncertain
- **1** = Legitimate indicator

| Feature | Description |
|---------|-------------|
| having_IP_Address | Whether the URL uses an IP address instead of a domain name |
| URL_Length | Length of the URL (-1=long/suspicious, 0=medium, 1=short/legitimate) |
| Shortining_Service | Whether a URL shortening service (e.g., TinyURL) is used |
| having_At_Symbol | Whether the URL contains an '@' symbol |
| double_slash_redirecting | Whether '//' appears in the URL path for redirection |
| Prefix_Suffix | Whether the domain name contains a '-' (dash) |
| having_Sub_Domain | Number of dots in the domain indicating subdomains |
| SSLfinal_State | SSL certificate trust level (-1=untrusted, 0=suspicious, 1=trusted) |
| Domain_registeration_length | Domain registration duration (-1=short, 1=long) |
| Favicon | Whether the favicon is loaded from an external domain |
| port | Whether a non-standard port is used |
| HTTPS_token | Whether 'HTTPS' token appears in the domain part of the URL |
| Request_URL | Percentage of external objects loaded from other domains |
| URL_of_Anchor | Percentage of anchor tags with different domain or no link |
| Links_in_tags | Percentage of links in meta/script/link tags from different domains |
| SFH | Server Form Handler — whether form action points to a different domain |
| Submitting_to_email | Whether the form submits data to an email address |
| Abnormal_URL | Whether the URL hostname is not in the WHOIS record |
| Redirect | Number of redirects |
| on_mouseover | Whether onMouseOver changes the status bar |
| RightClick | Whether right-click is disabled |
| popUpWidnow | Whether pop-up windows contain text fields |
| Iframe | Whether the page uses invisible iframes |
| age_of_domain | Age of the domain (-1=young/suspicious, 1=old/legitimate) |
| DNSRecord | Whether the domain has a DNS record |
| web_traffic | Website traffic rank from Alexa (-1=low, 0=medium, 1=high) |
| Page_Rank | Google PageRank score |
| Google_Index | Whether the page is indexed by Google |
| Links_pointing_to_page | Number of external links pointing to the page |
| Statistical_report | Whether the host belongs to top phishing IPs/domains from reports |

### Reference

R.M. Mohammad, F. Thabtah, L. McCluskey. *Phishing Websites Features.* University of Huddersfield.

## Project Structure

```
csai412-phishing-project/
├── README.md                  # This file
├── requirements.txt           # Python dependencies
├── data/
│   ├── phishing.csv           # Raw Phishing Websites dataset (UCI)
│   ├── train.csv              # Preprocessed training set (80%)
│   ├── test.csv               # Preprocessed test set (20%)
│   └── scaler.joblib          # Fitted StandardScaler
├── src/
│   ├── __init__.py
│   ├── data_loader.py         # Data loading, preprocessing, train/test split
│   ├── eda.py                 # Comprehensive exploratory data analysis
│   ├── comparison.py          # Model comparison & evaluation
│   └── models/                # ML model implementations
├── notebooks/                 # Jupyter notebooks (experiments)
├── figures/                   # All generated EDA/model figures
├── gui/                       # Streamlit GUI application
├── deploy/                    # Deployment files
└── report/                    # Final project report
```

## Setup

### 1. Install Dependencies

```bash
pip install numpy pandas scikit-learn matplotlib seaborn scipy plotly streamlit joblib
```

Or use the requirements file:

```bash
pip install -r requirements.txt
```

### 2. Load Data & Run EDA

The data loader loads the Phishing Websites dataset from `data/phishing.csv`.

```bash
# Test the data loader (loads data, creates train/test split)
python3 src/data_loader.py

# Run comprehensive EDA (generates all figures)
python3 src/eda.py
```

### 3. Using the Data Loader in Your Code

```python
from src.data_loader import load_data, get_train_test, load_processed_data

# Load raw dataset
df = load_data()

# Get train/test split (stratified, scaled, 80/20)
data = get_train_test()
X_train, X_test = data['X_train'], data['X_test']
y_train, y_test = data['y_train'], data['y_test']
feature_names = data['feature_names']

# Load previously saved processed data (fast, no recomputation)
data = load_processed_data()
```

## Key EDA Findings

1. **No missing values** -- dataset is complete (11,055 samples)
2. **Near-perfect class balance** -- Phishing 44.3% vs Legitimate 55.7%, imbalance ratio 1.26:1
3. **All features are categorical** -- values in {-1, 0, 1} representing suspicious/neutral/legitimate states
4. **Top correlated features with target:**
   - SSLfinal_State, URL_of_Anchor, and web_traffic show strongest correlation with phishing result
5. **Low multicollinearity** -- unlike continuous datasets, features are largely independent
6. **All features statistically significant** -- Chi-squared test of independence, p < 0.001
7. **No traditional outliers** -- IQR method yields few outliers since features only take 3 values

## Preprocessing Pipeline

- **Scaling:** StandardScaler (zero mean, unit variance) on all 30 features
  - Note: Features are already on a similar scale {-1, 0, 1}, but StandardScaler ensures zero mean and unit variance which benefits distance-based models (KNN, SVM)
- **Target remapping:** -1 (Phishing) -> 0, 1 (Legitimate) -> 1
- **Split:** 80% train / 20% test, stratified by Result, random_state=42
- **No imputation needed** (no missing values)

## Generated Figures

All figures are saved to `figures/` after running `src/eda.py`:

- `class_distribution.png` -- Bar chart and pie chart of Phishing vs Legitimate distribution
- `correlation_heatmap.png` -- Full correlation matrix heatmap (30 features)
- `correlation_with_target.png` -- Feature correlation with phishing result
- `feature_distributions.png` -- Bar charts of feature value counts ({-1, 0, 1})
- `feature_distributions_by_class.png` -- Feature value proportions by class
- `feature_proportions_per_class.png` -- Stacked bar charts per class
- `box_plots_per_class.png` -- Box plots for each feature by class
- `feature_importance.png` -- Mutual information + random forest importance
- `summary_dashboard.png` -- Combined overview dashboard

## Team

CSAI412 Machine Learning Group Project

| Team Member | Contribution |
|-------------|-------------|
| **Nazim Ahmed** | Project setup, data preprocessing, EDA, Logistic Regression, SVM (Linear & RBF), GUI development, project coordination |
| **Danniyaal Ahmed** | KNN implementation, Decision Tree implementation, K-Means clustering & PCA analysis, comparative analysis |
| **Mohamed Talha** | MLP implementation, technical report writing, Jupyter notebook compilation, testing & validation |
