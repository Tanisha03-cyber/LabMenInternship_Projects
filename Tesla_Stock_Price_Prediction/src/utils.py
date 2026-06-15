import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler


def load_data(filepath):

    df = pd.read_csv(filepath)

    df["Date"] = pd.to_datetime(df["Date"])

    df.sort_values("Date", inplace=True)

    df.set_index("Date", inplace=True)

    return df


def check_missing_values(df):

    return df.isnull().sum()


def handle_missing_values(df):

    df = df.ffill().bfill()

    return df


def scale_data(df, target_column="Adj Close"):

    scaler = MinMaxScaler(feature_range=(0, 1))

    scaled_data = scaler.fit_transform(
        df[[target_column]]
    )

    return scaled_data, scaler


def create_sequences(
    data,
    lookback=60,
    horizon=1
):

    X = []
    y = []

    for i in range(
        lookback,
        len(data) - horizon
    ):

        X.append(
            data[i - lookback:i]
        )

        y.append(
            data[i + horizon]
        )

    X = np.array(X)
    y = np.array(y)

    return X, y


def train_test_split_time_series(
    X,
    y,
    train_ratio=0.8
):

    split_index = int(
        len(X) * train_ratio
    )

    X_train = X[:split_index]
    X_test = X[split_index:]

    y_train = y[:split_index]
    y_test = y[split_index:]

    return (
        X_train,
        X_test,
        y_train,
        y_test
    )