import pandas as pd
import numpy as np


def create_features(df):

    df = df.copy()

    # Moving Averages

    df["MA_10"] = (
        df["Adj Close"]
        .rolling(window=10)
        .mean()
    )

    df["MA_20"] = (
        df["Adj Close"]
        .rolling(window=20)
        .mean()
    )

    df["MA_50"] = (
        df["Adj Close"]
        .rolling(window=50)
        .mean()
    )

    # Exponential Moving Average

    df["EMA_10"] = (
        df["Adj Close"]
        .ewm(span=10)
        .mean()
    )

    # Daily Return

    df["Daily_Return"] = (
        df["Adj Close"]
        .pct_change()
    )

    # Volatility

    df["Volatility"] = (
        df["Daily_Return"]
        .rolling(window=10)
        .std()
    )

    df.dropna(inplace=True)

    return df