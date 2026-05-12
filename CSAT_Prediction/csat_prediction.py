# ================================================================
# CSAT Score Prediction - eCommerce Customer Support Data
# Project  : DeepCSAT DL Project
# Author   : Tanisha Goyal
# ================================================================

import os
import re
import string
import warnings
warnings.filterwarnings('ignore')

# ── STEP 1: IMPORT LIBRARIES ────────────────────────────────────
print("=" * 60)
print("STEP 1: Importing Libraries")
print("=" * 60)

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')          # non-interactive backend – no GUI popup, faster
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection  import train_test_split, RandomizedSearchCV
from sklearn.preprocessing    import LabelEncoder, StandardScaler, PowerTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics          import mean_squared_error, r2_score
from sklearn.ensemble         import ExtraTreesRegressor, RandomForestRegressor
from sklearn.decomposition    import PCA
from scipy.stats              import f_oneway, ttest_ind, chi2_contingency

print("Libraries imported successfully\n")


# ── STEP 2: LOAD DATASET ────────────────────────────────────────
print("=" * 60)
print("STEP 2: Loading Dataset")
print("=" * 60)

csv_path = os.path.join("data", "eCommerce_Customer_support_data.csv")

if os.path.exists(csv_path):
    df = pd.read_csv(csv_path)
    print("Dataset loaded successfully.")
else:
    raise FileNotFoundError(
        "Dataset not found. Place the CSV file inside the 'data/' folder.")

# Ensure outputs/ folder exists
os.makedirs("outputs", exist_ok=True)
print("outputs/ folder ready.\n")


# ── STEP 3: FIRST VIEW OF DATA ──────────────────────────────────
print("=" * 60)
print("STEP 3: First View of Dataset")
print("=" * 60)

print("\n--- First 5 Rows ---")
print(df.head().to_string())

rows, cols = df.shape
print(f"\nTotal Rows: {rows}, Total Columns: {cols}\n")


# ── STEP 4: DATASET INFO ────────────────────────────────────────
print("=" * 60)
print("STEP 4: Dataset Info")
print("=" * 60)

df.info()
print()


# ── STEP 5: DUPLICATE CHECK ─────────────────────────────────────
print("=" * 60)
print("STEP 5: Duplicate Check")
print("=" * 60)

duplicate_count = df.duplicated().sum()
print(f"Number of duplicate rows: {duplicate_count}\n")


# ── STEP 6: MISSING VALUES ──────────────────────────────────────
print("=" * 60)
print("STEP 6: Missing Values Per Column")
print("=" * 60)

missing_counts = df.isnull().sum()
print("Missing values per column:\n", missing_counts.to_string())
print()


# ── STEP 7: COLUMN NAMES ────────────────────────────────────────
print("=" * 60)
print("STEP 7: Column Names")
print("=" * 60)

print(df.columns.tolist())
print()


# ── STEP 8: STATISTICAL DESCRIPTION ────────────────────────────
print("=" * 60)
print("STEP 8: Statistical Description")
print("=" * 60)

print(df.describe().to_string())
print()


# ── STEP 9: UNIQUE VALUES PER COLUMN ───────────────────────────
print("=" * 60)
print("STEP 9: Unique Values Per Column")
print("=" * 60)

for col in df.columns:
    print(col, df[col].nunique())
print()


# ── STEP 10: BASIC CLEANING (lowercase + strip) ─────────────────
print("=" * 60)
print("STEP 10: Basic Text Cleaning (lowercase + strip)")
print("=" * 60)

object_cols = df.select_dtypes(include='object').columns
for col in object_cols:
    df[col] = df[col].astype(str).str.lower().str.strip()

print("Object columns after cleaning:")
print(df.select_dtypes(include='object').columns.tolist())
print()


# ── STEP 11: EDA VISUALISATIONS ─────────────────────────────────
print("=" * 60)
print("STEP 11: EDA Visualisations (saving to outputs/)")
print("=" * 60)

# Chart 1 - CSAT Score Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='CSAT Score', data=df, palette='Blues')
plt.title('Distribution of CSAT Scores')
plt.xlabel('CSAT Score')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/01_csat_distribution.png', dpi=80)
plt.close()
print("  Chart 1/14 saved: 01_csat_distribution.png")

# Chart 2 - Service Channel Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='channel_name', data=df, palette='pastel')
plt.title('Distribution of Service Channels')
plt.xlabel('Channel Name')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/02_channel_distribution.png', dpi=80)
plt.close()
print("  Chart 2/14 saved: 02_channel_distribution.png")

# Chart 3 - Product Category Distribution
plt.figure(figsize=(15, 6))
sns.countplot(x='Product_category', data=df,
              order=df['Product_category'].value_counts().index, palette='Set2')
plt.title('Distribution of Product Categories')
plt.xlabel('Product Category')
plt.ylabel('Count')
plt.xticks(rotation=30)
plt.tight_layout()
plt.savefig('outputs/03_product_category.png', dpi=80)
plt.close()
print("  Chart 3/14 saved: 03_product_category.png")

# Chart 4 - Agent Shift Distribution
plt.figure(figsize=(8, 5))
sns.countplot(x='Agent Shift', data=df, palette='cool')
plt.title('Agent Shift Distribution')
plt.xlabel('Agent Shift')
plt.ylabel('Count')
plt.tight_layout()
plt.savefig('outputs/04_agent_shift.png', dpi=80)
plt.close()
print("  Chart 4/14 saved: 04_agent_shift.png")

# Chart 5 - Item Price Distribution
plt.figure(figsize=(8, 5))
sns.histplot(df['Item_price'].dropna(), bins=50, color='springgreen')
plt.title('Distribution of Item Prices')
plt.xlabel('Item Price')
plt.ylabel('Frequency')
plt.tight_layout()
plt.savefig('outputs/05_item_price.png', dpi=80)
plt.close()
print("  Chart 5/14 saved: 05_item_price.png")

# Chart 6 - Item Price by CSAT Score
plt.figure(figsize=(8, 5))
sns.boxplot(x='CSAT Score', y='Item_price', data=df, color='lightslategray')
plt.title('Item Price by CSAT Score')
plt.xlabel('CSAT Score')
plt.ylabel('Item Price')
plt.tight_layout()
plt.savefig('outputs/06_price_vs_csat.png', dpi=80)
plt.close()
print("  Chart 6/14 saved: 06_price_vs_csat.png")

# Chart 7 - CSAT Score by Channel
plt.figure(figsize=(10, 5))
sns.countplot(x='channel_name', hue='CSAT Score', data=df, palette='Set1')
plt.title('CSAT Score by Channel')
plt.xlabel('Channel Name')
plt.ylabel('Count')
plt.legend(title='CSAT Score')
plt.tight_layout()
plt.savefig('outputs/07_csat_by_channel.png', dpi=80)
plt.close()
print("  Chart 7/14 saved: 07_csat_by_channel.png")

# Chart 8 - Average CSAT by Agent Shift
df_shift_csat = df.groupby('Agent Shift')['CSAT Score'].mean().reset_index()
plt.figure(figsize=(8, 5))
sns.barplot(x='Agent Shift', y='CSAT Score', data=df_shift_csat, palette='Set3')
plt.title('Average CSAT by Agent Shift')
plt.xlabel('Agent Shift')
plt.ylabel('Average CSAT Score')
plt.tight_layout()
plt.savefig('outputs/08_avg_csat_shift.png', dpi=80)
plt.close()
print("  Chart 8/14 saved: 08_avg_csat_shift.png")

# Chart 9 - CSAT by Category
plt.figure(figsize=(12, 6))
sns.countplot(x='category', hue='CSAT Score', data=df, palette='husl',
              order=df['category'].value_counts().index)
plt.title('CSAT Score Distribution by Category')
plt.xlabel('Category')
plt.ylabel('Count')
plt.legend(title='CSAT Score')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/09_csat_by_category.png', dpi=80)
plt.close()
print("  Chart 9/14 saved: 09_csat_by_category.png")

# Chart 10 - CSAT by Tenure Bucket
plt.figure(figsize=(8, 5))
sns.countplot(x='Tenure Bucket', hue='CSAT Score', data=df, palette='Spectral')
plt.title('CSAT Score by Agent Tenure')
plt.xlabel('Tenure Bucket')
plt.ylabel('Count')
plt.legend(title='CSAT Score')
plt.tight_layout()
plt.savefig('outputs/10_csat_by_tenure.png', dpi=80)
plt.close()
print("  Chart 10/14 saved: 10_csat_by_tenure.png")

# Chart 11 - Correlation Heatmap
plt.figure(figsize=(10, 8))
corr = df[['Item_price', 'connected_handling_time', 'CSAT Score']].corr()
sns.heatmap(corr, annot=True, cmap='coolwarm', fmt='.2f')
plt.title('Correlation Heatmap')
plt.tight_layout()
plt.savefig('outputs/11_correlation_heatmap.png', dpi=80)
plt.close()
print("  Chart 11/14 saved: 11_correlation_heatmap.png")

# Chart 12 - Pair Plot
sns.pairplot(df[['Item_price', 'connected_handling_time', 'CSAT Score']].dropna())
plt.suptitle('Pair Plot of Key Numeric Features', y=1.02)
plt.savefig('outputs/12_pairplot.png', dpi=80)
plt.close()
print("  Chart 12/14 saved: 12_pairplot.png")

# Chart 13 - Avg CSAT by Product & Channel
prod_ch = df.groupby(['Product_category', 'channel_name'])['CSAT Score'].mean().unstack()
prod_ch.plot(kind='bar', figsize=(12, 7))
plt.title('Avg CSAT by Product & Channel')
plt.ylabel('Avg CSAT Score')
plt.xlabel('Product Category')
plt.xticks(rotation=45)
plt.legend(title='Channel Name')
plt.tight_layout()
plt.savefig('outputs/13_csat_product_channel.png', dpi=80)
plt.close()
print("  Chart 13/14 saved: 13_csat_product_channel.png")

# Chart 14 - Avg CSAT by Shift & Category
shift_cat = df.groupby(['Agent Shift', 'category'])['CSAT Score'].mean().unstack()
shift_cat.plot(kind='bar', stacked=True, figsize=(14, 7), colormap='viridis')
plt.title('Avg CSAT by Shift & Issue Category')
plt.ylabel('Avg CSAT Score')
plt.xlabel('Agent Shift')
plt.xticks(rotation=45)
plt.tight_layout()
plt.savefig('outputs/14_csat_shift_category.png', dpi=80)
plt.close()
print("  Chart 14/14 saved: 14_csat_shift_category.png")

print("All EDA charts saved.\n")


# ── STEP 12: HYPOTHESIS TESTS ───────────────────────────────────
print("=" * 60)
print("STEP 12: Hypothesis Testing")
print("=" * 60)

# ANOVA: Does channel_name significantly affect CSAT?
channel_groups = [grp['CSAT Score'].dropna().values
                  for _, grp in df.groupby('channel_name')]
anova_result = f_oneway(*channel_groups)
print(f"ANOVA p-value: {anova_result.pvalue:.55f}")

# t-test: High-value vs low-value item price vs CSAT
high_value = df[df['Item_price'] >= 10000]['CSAT Score'].dropna()
low_value  = df[df['Item_price'] <  10000]['CSAT Score'].dropna()
t_stat, p_val = ttest_ind(high_value, low_value, equal_var=False)
print(f"t-test p-value: {p_val:.55f}")

# Chi-square: Agent Shift vs CSAT Score
shift_csat = pd.crosstab(df['Agent Shift'], df['CSAT Score'])
chi2, p, dof, expected = chi2_contingency(shift_csat)
print(f"Chi-square test p-value: {p:.55f}")
print()


# ── STEP 13: DATA PREPROCESSING ─────────────────────────────────
print("=" * 60)
print("STEP 13: Data Preprocessing & Feature Engineering")
print("=" * 60)

# Drop columns with extreme missingness (>60%)
drop_cols = ['connected_handling_time', 'Customer_City',
             'Product_category', 'Order_id', 'order_date_time', 'Item_price']
df_clean = df.drop(columns=drop_cols, errors='ignore').copy()

# Fill missing Customer Remarks
df_clean['Customer Remarks'] = df_clean['Customer Remarks'].fillna('no feedback')
print(f"Dropped high-missingness columns: {drop_cols}")

# Label encode categorical columns
le = LabelEncoder()
cat_cols = ['channel_name', 'category', 'Sub-category',
            'Agent_name', 'Supervisor', 'Manager', 'Tenure Bucket', 'Agent Shift']
for col in cat_cols:
    df_clean[col] = le.fit_transform(df_clean[col].astype(str))
print(f"Label encoded: {cat_cols}")

# Build feature matrix X (drop target + text + ID columns)
X = df_clean.drop(
    columns=['CSAT Score', 'Customer Remarks', 'Customer Remarks Tokens',
             'Customer Remarks POS', 'id', 'Unique id'],
    errors='ignore'
)

# Parse datetime columns
for col in ['Issue_reported at', 'issue_responded', 'Survey_response_Date']:
    X[col] = pd.to_datetime(X[col], errors='coerce', dayfirst=True)

# Extract time-based features
X['issue_reported_hour']       = X['Issue_reported at'].dt.hour
X['issue_reported_dayofweek']  = X['Issue_reported at'].dt.dayofweek
X['issue_responded_hour']      = X['issue_responded'].dt.hour
X['issue_responded_dayofweek'] = X['issue_responded'].dt.dayofweek

# Response time in hours (key engineered feature)
X['response_time_hrs'] = (
    X['issue_responded'] - X['Issue_reported at']
).dt.total_seconds() / 3600

# Drop raw datetime columns
X = X.drop(columns=['Issue_reported at', 'issue_responded', 'Survey_response_Date'],
           errors='ignore')

print(f"Feature matrix shape: {X.shape}")
print()


# ── STEP 14: NLP - CLEAN CUSTOMER REMARKS ───────────────────────
print("=" * 60)
print("STEP 14: NLP - Cleaning Customer Remarks")
print("=" * 60)

contractions = {
    "can't": "cannot", "won't": "will not",
    "don't": "do not", "i'm": "i am", "isn't": "is not"
}

def expand_contractions(text):
    for c, e in contractions.items():
        text = re.sub(r"\b" + c + r"\b", e, text)
    return text

df_clean['Customer Remarks'] = df_clean['Customer Remarks'].apply(expand_contractions)
df_clean['Customer Remarks'] = df_clean['Customer Remarks'].str.lower()
df_clean['Customer Remarks'] = df_clean['Customer Remarks'].str.replace(
    f"[{re.escape(string.punctuation)}]", "", regex=True)
df_clean['Customer Remarks'] = df_clean['Customer Remarks'].str.replace(
    r'http\S+|www\S+', '', regex=True)
df_clean['Customer Remarks'] = df_clean['Customer Remarks'].apply(
    lambda x: ' '.join([w for w in x.split() if not any(c.isdigit() for c in w)]))

# Remove stopwords
try:
    import nltk
    try:
        from nltk.corpus import stopwords
        stop = set(stopwords.words('english'))
    except LookupError:
        nltk.download('stopwords', quiet=True)
        from nltk.corpus import stopwords
        stop = set(stopwords.words('english'))
    df_clean['Customer Remarks'] = df_clean['Customer Remarks'].apply(
        lambda x: ' '.join([w for w in x.split() if w not in stop]))
    df_clean['Customer Remarks'] = df_clean['Customer Remarks'].str.strip()
    print("Stopwords removed successfully.")
except Exception as e:
    print(f"Stopword removal skipped: {e}")

print()


# ── STEP 15: TF-IDF VECTORISATION ───────────────────────────────
print("=" * 60)
print("STEP 15: TF-IDF Vectorisation of Customer Remarks")
print("=" * 60)

vectorizer    = TfidfVectorizer(max_features=100)
remarks_tfidf = vectorizer.fit_transform(df_clean['Customer Remarks'])
print(f"TF-IDF matrix shape: {remarks_tfidf.shape}")
print()


# ── STEP 16: SAVE CLEANED CSV ───────────────────────────────────
print("=" * 60)
print("STEP 16: Saving Cleaned Dataset to data/ folder")
print("=" * 60)

# Add response_time_hrs to df_clean before saving
df_clean['response_time_hrs'] = (
    pd.to_datetime(df_clean['issue_responded'], errors='coerce', dayfirst=True) -
    pd.to_datetime(df_clean['Issue_reported at'], errors='coerce', dayfirst=True)
).dt.total_seconds() / 3600

cleaned_path = os.path.join("data", "eCommerce_cleaned.csv")
df_clean.to_csv(cleaned_path, index=False)
print(f"Cleaned CSV saved -> {cleaned_path}")
print(f"Cleaned dataset shape: {df_clean.shape}")
print()


# ── STEP 17: FEATURE IMPORTANCE WITH EXTRATREES ─────────────────
print("=" * 60)
print("STEP 17: Feature Selection via ExtraTreesRegressor")
print("=" * 60)

y         = df_clean['CSAT Score']
X_numeric = X.select_dtypes(include=['number'])

# n_jobs=-1 uses all CPU cores - much faster
selector = ExtraTreesRegressor(n_estimators=50, random_state=42, n_jobs=-1)
selector.fit(X_numeric, y)

feat_importances = selector.feature_importances_
important_feats  = X_numeric.columns[np.argsort(feat_importances)[-8:]]
print(f"Important Features: {important_feats.tolist()}")
print()


# ── STEP 18: SCALING + PCA ──────────────────────────────────────
print("=" * 60)
print("STEP 18: PowerTransform -> StandardScaler -> PCA")
print("=" * 60)

# Power-transform skewed response_time_hrs
pt = PowerTransformer()
df_clean['response_time_hrs_trans'] = pt.fit_transform(
    df_clean[['response_time_hrs']].fillna(0))

# Scale all features to zero mean / unit variance
scaler   = StandardScaler()
X_scaled = scaler.fit_transform(X.fillna(0))

# PCA to 10 components
pca           = PCA(n_components=10)
X_pca         = pca.fit_transform(X_scaled)
explained_var = pca.explained_variance_ratio_.sum()
print(f"Explained Variance by PCA: {explained_var}")
print()


# ── STEP 19: TRAIN / TEST SPLIT ─────────────────────────────────
print("=" * 60)
print("STEP 19: Train / Test Split (80/20)")
print("=" * 60)

X_final = X_pca if X_pca.shape[1] > 0 else X_scaled
X_train, X_test, y_train, y_test = train_test_split(
    X_final, y, test_size=0.2, random_state=42)

print(f"X_train: {X_train.shape}  |  X_test: {X_test.shape}")
print()


# ── STEP 20: CATBOOST ───────────────────────────────────────────
print("=" * 60)
print("STEP 20: CatBoost - Baseline + Tuned")
print("=" * 60)

mse_cat = mse_cat_cv = r2_cat = r2_cat_cv = None
cat_cv = None

try:
    from catboost import CatBoostRegressor

    # Baseline - thread_count=-1 uses all CPU threads
    cat_reg = CatBoostRegressor(verbose=0, random_state=42, thread_count=-1)
    cat_reg.fit(X_train, y_train)
    y_pred_cat = cat_reg.predict(X_test)
    mse_cat = mean_squared_error(y_test, y_pred_cat)
    r2_cat  = r2_score(y_test, y_pred_cat)
    print(f"CatBoost Baseline  MSE: {mse_cat:.4f}, R2: {r2_cat:.4f}")

    # Baseline chart
    pd.DataFrame({'Metric': ['MSE', 'R2'], 'Baseline': [mse_cat, r2_cat]}) \
      .set_index('Metric').plot(kind='bar', legend=False,
                                title='CatBoost Baseline Metrics', color='limegreen')
    plt.tight_layout()
    plt.savefig('outputs/15_catboost_baseline.png', dpi=80)
    plt.close()

    # Tuned - reduced n_iter and cv=2 for speed
    param_dist = {
        'depth':         [3, 4, 5, 6],
        'iterations':    [50, 100],
        'learning_rate': [0.05, 0.1]
    }
    cat_cv = RandomizedSearchCV(
        CatBoostRegressor(verbose=0, random_state=42, thread_count=-1),
        param_dist,
        n_iter=6,
        scoring='neg_mean_squared_error',
        cv=2,              # cv=2 instead of 3 - biggest speed boost
        random_state=42,
        n_jobs=-1
    )
    cat_cv.fit(X_train, y_train)
    y_pred_cat_cv = cat_cv.predict(X_test)
    mse_cat_cv = mean_squared_error(y_test, y_pred_cat_cv)
    r2_cat_cv  = r2_score(y_test, y_pred_cat_cv)
    print(f"CatBoost Tuned     MSE: {mse_cat_cv:.4f}, R2: {r2_cat_cv:.4f}")

    # Comparison chart
    pd.DataFrame({'Metric': ['MSE', 'R2'],
                  'Baseline': [mse_cat, r2_cat],
                  'Tuned':    [mse_cat_cv, r2_cat_cv]}) \
      .set_index('Metric').plot(kind='bar', title='CatBoost: Baseline vs Tuned')
    plt.tight_layout()
    plt.savefig('outputs/16_catboost_tuned.png', dpi=80)
    plt.close()

except Exception as e:
    print(f"CatBoost failed: {e}")

print()


# ── STEP 21: RANDOM FOREST ──────────────────────────────────────
print("=" * 60)
print("STEP 21: Random Forest - Baseline + Tuned")
print("=" * 60)

mse_rf = mse_rf_cv = r2_rf = r2_rf_cv = None

try:
    # Baseline - n_jobs=-1 uses all CPU cores
    rf = RandomForestRegressor(n_estimators=100, random_state=42, n_jobs=-1)
    rf.fit(X_train, y_train)
    y_pred_rf = rf.predict(X_test)
    mse_rf = mean_squared_error(y_test, y_pred_rf)
    r2_rf  = r2_score(y_test, y_pred_rf)
    print(f"Random Forest Baseline  MSE: {mse_rf:.4f}, R2: {r2_rf:.4f}")

    # Baseline chart
    pd.DataFrame({'Metric': ['MSE', 'R2'], 'Baseline': [mse_rf, r2_rf]}) \
      .set_index('Metric').plot(kind='bar', legend=False,
                                title='Random Forest Baseline Metrics', color='limegreen')
    plt.tight_layout()
    plt.savefig('outputs/17_rf_baseline.png', dpi=80)
    plt.close()

    # Tuned - smaller search space, cv=2, n_jobs=-1
    param_dist_rf = {
        'n_estimators':      [50, 100],
        'max_depth':         [5, 10, 15],
        'min_samples_split': [5, 10]
    }
    rf_cv = RandomizedSearchCV(
        RandomForestRegressor(random_state=42, n_jobs=-1),
        param_dist_rf,
        n_iter=5,
        scoring='neg_mean_squared_error',
        cv=2,
        random_state=42,
        n_jobs=-1
    )
    rf_cv.fit(X_train, y_train)
    y_pred_rf_cv = rf_cv.predict(X_test)
    mse_rf_cv = mean_squared_error(y_test, y_pred_rf_cv)
    r2_rf_cv  = r2_score(y_test, y_pred_rf_cv)
    print(f"Random Forest Tuned     MSE: {mse_rf_cv:.4f}, R2: {r2_rf_cv:.4f}")

    # Comparison chart
    pd.DataFrame({'Metric': ['MSE', 'R2'],
                  'Baseline': [mse_rf, r2_rf],
                  'Tuned':    [mse_rf_cv, r2_rf_cv]}) \
      .set_index('Metric').plot(kind='bar', title='Random Forest: Baseline vs Tuned')
    plt.tight_layout()
    plt.savefig('outputs/18_rf_tuned.png', dpi=80)
    plt.close()

except Exception as e:
    print(f"Random Forest failed: {e}")

print()


# ── STEP 22: XGBOOST ────────────────────────────────────────────
print("=" * 60)
print("STEP 22: XGBoost - Baseline + Tuned")
print("=" * 60)

mse_xgb = mse_xgb_cv = r2_xgb = r2_xgb_cv = None

try:
    from xgboost import XGBRegressor

    # Baseline - nthread=-1 uses all CPU cores
    xgb = XGBRegressor(random_state=42, objective='reg:squarederror',
                       nthread=-1, verbosity=0)
    xgb.fit(X_train, y_train)
    y_pred_xgb = xgb.predict(X_test)
    mse_xgb = mean_squared_error(y_test, y_pred_xgb)
    r2_xgb  = r2_score(y_test, y_pred_xgb)
    print(f"XGBoost Baseline  MSE: {mse_xgb:.4f}, R2: {r2_xgb:.4f}")

    # Baseline chart
    pd.DataFrame({'Metric': ['MSE', 'R2'], 'Baseline': [mse_xgb, r2_xgb]}) \
      .set_index('Metric').plot(kind='bar', legend=False,
                                title='XGBoost Baseline Metrics', color='limegreen')
    plt.tight_layout()
    plt.savefig('outputs/19_xgb_baseline.png', dpi=80)
    plt.close()

    # Tuned - smaller search space, cv=2, n_jobs=-1
    param_dist_xgb = {
        'n_estimators':  [50, 100],
        'learning_rate': [0.05, 0.1],
        'max_depth':     [3, 5],
        'subsample':     [0.8, 1.0]
    }
    xgb_cv = RandomizedSearchCV(
        XGBRegressor(random_state=42, objective='reg:squarederror',
                     nthread=-1, verbosity=0),
        param_dist_xgb,
        n_iter=5,
        scoring='neg_mean_squared_error',
        cv=2,
        random_state=42,
        n_jobs=-1
    )
    xgb_cv.fit(X_train, y_train)
    y_pred_xgb_cv = xgb_cv.predict(X_test)
    mse_xgb_cv = mean_squared_error(y_test, y_pred_xgb_cv)
    r2_xgb_cv  = r2_score(y_test, y_pred_xgb_cv)
    print(f"XGBoost Tuned     MSE: {mse_xgb_cv:.4f}, R2: {r2_xgb_cv:.4f}")

    # Comparison chart
    pd.DataFrame({'Metric': ['MSE', 'R2'],
                  'Baseline': [mse_xgb, r2_xgb],
                  'Tuned':    [mse_xgb_cv, r2_xgb_cv]}) \
      .set_index('Metric').plot(kind='bar', title='XGBoost: Baseline vs Tuned')
    plt.tight_layout()
    plt.savefig('outputs/20_xgb_tuned.png', dpi=80)
    plt.close()

except Exception as e:
    print(f"XGBoost failed: {e}")

print()


# ── STEP 23: ANN (DEEP LEARNING) ────────────────────────────────
print("=" * 60)
print("STEP 23: ANN (Deep Learning) - Baseline + Tuned")
print("=" * 60)

mse_ann = mse_ann_tuned = r2_ann = r2_ann_tuned = None

try:
    import tensorflow as tf
    from tensorflow.keras.models import Sequential
    from tensorflow.keras.layers import Dense, Dropout, BatchNormalization
    from tensorflow.keras.callbacks import EarlyStopping
    from tensorflow.keras.optimizers import Adam
    os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # suppress TF info logs

    print(f"TensorFlow version: {tf.__version__}")

    # ── ANN Baseline ──────────────────────────────────────────────
    tf.random.set_seed(42)
    ann_baseline = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(64,  activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(32,  activation='relu'),
        Dense(1)
    ], name='ANN_Baseline')

    ann_baseline.compile(optimizer=Adam(learning_rate=0.001), loss='mse')
    es = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True, verbose=0)

    ann_baseline.fit(
        X_train, y_train,
        epochs=100, batch_size=64,
        validation_split=0.1,
        callbacks=[es],
        verbose=0
    )

    y_pred_ann = ann_baseline.predict(X_test, verbose=0).flatten()
    mse_ann    = mean_squared_error(y_test, y_pred_ann)
    r2_ann     = r2_score(y_test, y_pred_ann)
    print(f"ANN Baseline  MSE: {mse_ann:.4f}, R2: {r2_ann:.4f}")

    # Baseline chart
    pd.DataFrame({'Metric': ['MSE', 'R2'], 'Baseline': [mse_ann, r2_ann]}) \
      .set_index('Metric').plot(kind='bar', legend=False,
                                title='ANN Baseline Metrics', color='mediumpurple')
    plt.tight_layout()
    plt.savefig('outputs/23_ann_baseline.png', dpi=80)
    plt.close()
    print("  Chart saved: 23_ann_baseline.png")

    # ── ANN Tuned (deeper architecture + lower lr) ────────────────
    tf.random.set_seed(42)
    es_tuned = EarlyStopping(monitor='val_loss', patience=15, restore_best_weights=True, verbose=0)

    ann_tuned = Sequential([
        Dense(256, activation='relu', input_shape=(X_train.shape[1],)),
        BatchNormalization(),
        Dropout(0.3),
        Dense(128, activation='relu'),
        BatchNormalization(),
        Dropout(0.2),
        Dense(64,  activation='relu'),
        BatchNormalization(),
        Dropout(0.1),
        Dense(32,  activation='relu'),
        Dense(1)
    ], name='ANN_Tuned')

    ann_tuned.compile(optimizer=Adam(learning_rate=0.0005), loss='mse')
    history = ann_tuned.fit(
        X_train, y_train,
        epochs=150, batch_size=32,
        validation_split=0.1,
        callbacks=[es_tuned],
        verbose=0
    )

    y_pred_ann_tuned = ann_tuned.predict(X_test, verbose=0).flatten()
    mse_ann_tuned    = mean_squared_error(y_test, y_pred_ann_tuned)
    r2_ann_tuned     = r2_score(y_test, y_pred_ann_tuned)
    print(f"ANN Tuned     MSE: {mse_ann_tuned:.4f}, R2: {r2_ann_tuned:.4f}")

    # Comparison chart
    pd.DataFrame({'Metric': ['MSE', 'R2'],
                  'Baseline': [mse_ann, r2_ann],
                  'Tuned':    [mse_ann_tuned, r2_ann_tuned]}) \
      .set_index('Metric').plot(kind='bar', title='ANN: Baseline vs Tuned', color=['mediumpurple', 'darkorchid'])
    plt.tight_layout()
    plt.savefig('outputs/24_ann_tuned.png', dpi=80)
    plt.close()
    print("  Chart saved: 24_ann_tuned.png")

    # Training history (loss curve)
    plt.figure(figsize=(8, 5))
    plt.plot(history.history['loss'],     label='Train Loss', color='mediumpurple')
    plt.plot(history.history['val_loss'], label='Val Loss',   color='darkorchid', linestyle='--')
    plt.title('ANN Tuned — Training Loss Curve')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.legend()
    plt.tight_layout()
    plt.savefig('outputs/25_ann_loss_curve.png', dpi=80)
    plt.close()
    print("  Chart saved: 25_ann_loss_curve.png")

    # Save the tuned ANN model
    os.makedirs("models", exist_ok=True)
    ann_tuned.save("models/ann_tuned.keras")
    print("  ANN Tuned model saved -> models/ann_tuned.keras")

except ImportError:
    print("TensorFlow not installed. Falling back to MLPRegressor (sklearn ANN)...")
    try:
        from sklearn.neural_network import MLPRegressor

        # Baseline MLP
        mlp_base = MLPRegressor(
            hidden_layer_sizes=(128, 64, 32),
            activation='relu', solver='adam',
            max_iter=200, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=10
        )
        mlp_base.fit(X_train, y_train)
        y_pred_ann = mlp_base.predict(X_test)
        mse_ann    = mean_squared_error(y_test, y_pred_ann)
        r2_ann     = r2_score(y_test, y_pred_ann)
        print(f"ANN (MLP) Baseline  MSE: {mse_ann:.4f}, R2: {r2_ann:.4f}")

        # Tuned MLP
        mlp_tuned = MLPRegressor(
            hidden_layer_sizes=(256, 128, 64, 32),
            activation='relu', solver='adam',
            learning_rate_init=0.0005,
            max_iter=300, random_state=42,
            early_stopping=True, validation_fraction=0.1,
            n_iter_no_change=15
        )
        mlp_tuned.fit(X_train, y_train)
        y_pred_ann_tuned = mlp_tuned.predict(X_test)
        mse_ann_tuned    = mean_squared_error(y_test, y_pred_ann_tuned)
        r2_ann_tuned     = r2_score(y_test, y_pred_ann_tuned)
        print(f"ANN (MLP) Tuned     MSE: {mse_ann_tuned:.4f}, R2: {r2_ann_tuned:.4f}")

        pd.DataFrame({'Metric': ['MSE', 'R2'],
                      'Baseline': [mse_ann, r2_ann],
                      'Tuned':    [mse_ann_tuned, r2_ann_tuned]}) \
          .set_index('Metric').plot(kind='bar', title='ANN (MLP): Baseline vs Tuned', color=['mediumpurple', 'darkorchid'])
        plt.tight_layout()
        plt.savefig('outputs/24_ann_tuned.png', dpi=80)
        plt.close()

    except Exception as e2:
        print(f"ANN fallback also failed: {e2}")

except Exception as e:
    print(f"ANN failed: {e}")

print()


# ── STEP 24: CATBOOST FEATURE IMPORTANCE ────────────────────────
print("=" * 60)
print("STEP 24: CatBoost Feature Importance")
print("=" * 60)

try:
    importances           = cat_cv.best_estimator_.get_feature_importance()
    numeric_feature_names = cat_cv.best_estimator_.feature_names_

    original_feature_names = [
        'Unique id', 'channel_name', 'category', 'Sub-category', 'Customer Remarks',
        'Order_id', 'order_date_time', 'Issue_reported at', 'issue_responded',
        'Survey_response_Date', 'Customer_City', 'Product_category', 'Item_price',
        'connected_handling_time', 'Agent_name', 'Supervisor', 'Manager',
        'Tenure Bucket', 'Agent Shift', 'CSAT Score'
    ]
    feature_mapping       = {str(i): n for i, n in enumerate(original_feature_names)}
    mapped_feature_names  = [feature_mapping.get(f, f) for f in numeric_feature_names]

    feat_imp_df = (
        pd.DataFrame({'Feature': mapped_feature_names, 'Importance': importances})
        .sort_values('Importance', ascending=False)
        .head(10)
    )

    plt.figure(figsize=(10, 6))
    plt.barh(feat_imp_df['Feature'], feat_imp_df['Importance'], color='teal')
    plt.xlabel('Importance Score')
    plt.title('Top 10 Feature Importances (CatBoost)')
    plt.gca().invert_yaxis()
    plt.tight_layout()
    plt.savefig('outputs/26_catboost_feature_importance.png', dpi=80)
    plt.close()
    print("CatBoost feature importance chart saved.")

except Exception as e:
    print(f"Feature importance plot skipped: {e}")


# ── STEP 25: FINAL SUMMARY ──────────────────────────────────────
print()
print("=" * 60)
print("STEP 25: Final Model Comparison Summary")
print("=" * 60)

summary = pd.DataFrame({
    'Model': [
        'CatBoost Baseline', 'CatBoost Tuned',
        'Random Forest Baseline', 'Random Forest Tuned',
        'XGBoost Baseline', 'XGBoost Tuned',
        'ANN Baseline', 'ANN Tuned'
    ],
    'MSE': [mse_cat, mse_cat_cv, mse_rf, mse_rf_cv,
            mse_xgb, mse_xgb_cv, mse_ann, mse_ann_tuned],
    'R2':  [r2_cat,  r2_cat_cv,  r2_rf,  r2_rf_cv,
            r2_xgb,  r2_xgb_cv,  r2_ann, r2_ann_tuned]
}).dropna()

print(summary.to_string(index=False))

# Save final comparison chart
if not summary.empty:
    fig, axes = plt.subplots(1, 2, figsize=(16, 6))
    colors = ['steelblue' if 'ANN' not in m else 'mediumpurple' for m in summary['Model']]
    summary.plot(kind='bar', x='Model', y='MSE', ax=axes[0],
                 color=colors, legend=False,
                 title='MSE Comparison (lower is better)')
    summary.plot(kind='bar', x='Model', y='R2',  ax=axes[1],
                 color=colors, legend=False,
                 title='R2 Comparison (higher is better)')
    for ax in axes:
        ax.set_xticklabels(ax.get_xticklabels(), rotation=35, ha='right')
    plt.tight_layout()
    plt.savefig('outputs/27_model_comparison.png', dpi=80)
    plt.close()
    print("\nFinal model comparison chart saved -> outputs/27_model_comparison.png")

print()
print("=" * 60)
print("ALL STEPS COMPLETED SUCCESSFULLY.")
print("=" * 60)