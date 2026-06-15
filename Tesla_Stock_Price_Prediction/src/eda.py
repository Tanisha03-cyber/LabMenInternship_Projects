import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from utils import load_data
from feature_engineering import create_features

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "TSLA.csv"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

print("Loading Dataset...")

df = load_data(DATA_PATH)

df = create_features(df)

# --------------------------------------------------
# 1 Closing Price Trend
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Adj Close"]
)

plt.title(
    "Tesla Adjusted Closing Price"
)

plt.xlabel("Date")
plt.ylabel("Price")

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "closing_price_trend.png"
    )
)

plt.close()

# --------------------------------------------------
# 2 Volume Trend
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Volume"]
)

plt.title(
    "Trading Volume Trend"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "volume_trend.png"
    )
)

plt.close()

# --------------------------------------------------
# 3 Daily Returns
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Daily_Return"]
)

plt.title(
    "Daily Returns"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "daily_returns.png"
    )
)

plt.close()

# --------------------------------------------------
# 4 Distribution Plot
# --------------------------------------------------

plt.figure(figsize=(10, 6))

sns.histplot(
    df["Adj Close"],
    kde=True
)

plt.title(
    "Adjusted Close Distribution"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "distribution.png"
    )
)

plt.close()

# --------------------------------------------------
# 5 Box Plot
# --------------------------------------------------

plt.figure(figsize=(8, 6))

sns.boxplot(
    y=df["Adj Close"]
)

plt.title(
    "Boxplot of Adjusted Close"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "boxplot.png"
    )
)

plt.close()

# --------------------------------------------------
# 6 Correlation Heatmap
# --------------------------------------------------

plt.figure(figsize=(10, 8))

sns.heatmap(
    df.corr(numeric_only=True),
    annot=True,
    cmap="coolwarm"
)

plt.title(
    "Correlation Heatmap"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "correlation_heatmap.png"
    )
)

plt.close()

# --------------------------------------------------
# 7 Moving Averages
# --------------------------------------------------

plt.figure(figsize=(14, 6))

plt.plot(
    df.index,
    df["Adj Close"],
    label="Adj Close"
)

plt.plot(
    df.index,
    df["MA_10"],
    label="MA 10"
)

plt.plot(
    df.index,
    df["MA_20"],
    label="MA 20"
)

plt.plot(
    df.index,
    df["MA_50"],
    label="MA 50"
)

plt.legend()

plt.title(
    "Moving Average Analysis"
)

plt.savefig(
    os.path.join(
        OUTPUT_DIR,
        "moving_averages.png"
    )
)

plt.close()

print(
    "\nEDA Completed Successfully"
)