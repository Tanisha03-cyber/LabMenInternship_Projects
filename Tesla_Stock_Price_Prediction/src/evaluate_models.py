import os
import sys
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    r2_score
)

from tensorflow.keras.models import load_model

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

SRC_DIR = os.path.join(
    BASE_DIR,
    "src"
)

sys.path.append(SRC_DIR)

from utils import (
    load_data,
    scale_data,
    create_sequences,
    train_test_split_time_series
)

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "TSLA.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics"
)

FIGURE_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

os.makedirs(
    FIGURE_DIR,
    exist_ok=True
)

df = load_data(DATA_PATH)

scaled_data, scaler = scale_data(
    df,
    target_column="Adj Close"
)

results = []

for horizon in [1, 5, 10]:

    print(f"\nEvaluating Horizon {horizon}")

    X, y = create_sequences(
        scaled_data,
        lookback=60,
        horizon=horizon
    )

    (
        X_train,
        X_test,
        y_train,
        y_test
    ) = train_test_split_time_series(
        X,
        y
    )

    rnn_model = load_model(
        os.path.join(
            MODEL_DIR,
            f"best_rnn_h{horizon}.keras"
        )
    )

    lstm_model = load_model(
        os.path.join(
            MODEL_DIR,
            f"best_lstm_h{horizon}.keras"
        )
    )

    rnn_pred = rnn_model.predict(X_test)
    lstm_pred = lstm_model.predict(X_test)

    actual = scaler.inverse_transform(y_test)
    rnn_pred_inv = scaler.inverse_transform(rnn_pred)
    lstm_pred_inv = scaler.inverse_transform(lstm_pred)

    # ---------------------------
    # Metrics
    # ---------------------------

    rnn_mse = mean_squared_error(
        actual,
        rnn_pred_inv
    )

    lstm_mse = mean_squared_error(
        actual,
        lstm_pred_inv
    )

    rnn_rmse = np.sqrt(rnn_mse)
    lstm_rmse = np.sqrt(lstm_mse)

    rnn_mae = mean_absolute_error(
        actual,
        rnn_pred_inv
    )

    lstm_mae = mean_absolute_error(
        actual,
        lstm_pred_inv
    )

    rnn_r2 = r2_score(
        actual,
        rnn_pred_inv
    )

    lstm_r2 = r2_score(
        actual,
        lstm_pred_inv
    )

    results.append(
        [
            horizon,
            rnn_mse,
            rnn_rmse,
            rnn_mae,
            rnn_r2,
            lstm_mse,
            lstm_rmse,
            lstm_mae,
            lstm_r2
        ]
    )

    # ---------------------------
    # Plot
    # ---------------------------

    plt.figure(figsize=(12,6))

    plt.plot(
        actual,
        label="Actual"
    )

    plt.plot(
        lstm_pred_inv,
        label="LSTM Prediction"
    )

    plt.title(
        f"LSTM Actual vs Predicted ({horizon} Day)"
    )

    plt.legend()

    plt.savefig(
        os.path.join(
            FIGURE_DIR,
            f"actual_vs_predicted_h{horizon}.png"
        )
    )

    plt.close()

results_df = pd.DataFrame(
    results,
    columns=[
        "Horizon",
        "RNN_MSE",
        "RNN_RMSE",
        "RNN_MAE",
        "RNN_R2",
        "LSTM_MSE",
        "LSTM_RMSE",
        "LSTM_MAE",
        "LSTM_R2"
    ]
)

results_df.to_csv(
    os.path.join(
        OUTPUT_DIR,
        "evaluation_metrics.csv"
    ),
    index=False
)

print("\nEvaluation Completed")
print(results_df)