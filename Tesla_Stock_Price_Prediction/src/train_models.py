import os
import sys
import joblib
import pandas as pd

from tensorflow.keras.callbacks import (
    EarlyStopping,
    ModelCheckpoint
)

from sklearn.metrics import (
    mean_squared_error
)

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

from model_builder import (
    build_rnn,
    build_lstm
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

os.makedirs(
    MODEL_DIR,
    exist_ok=True
)

df = load_data(DATA_PATH)

scaled_data, scaler = scale_data(
    df,
    target_column="Adj Close"
)

joblib.dump(
    scaler,
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

results = []

for horizon in [1, 5, 10]:

    print(
        f"\nTraining Horizon = {horizon}"
    )

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

    # ====================
    # RNN
    # ====================

    rnn = build_rnn()

    rnn_checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            f"best_rnn_h{horizon}.keras"
        ),
        save_best_only=True,
        monitor="val_loss"
    )

    rnn.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        callbacks=[
            EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),
            rnn_checkpoint
        ],
        verbose=1
    )

    rnn_pred = rnn.predict(
        X_test
    )

    rnn_mse = mean_squared_error(
        y_test,
        rnn_pred
    )

    # ====================
    # LSTM
    # ====================

    lstm = build_lstm()

    lstm_checkpoint = ModelCheckpoint(
        filepath=os.path.join(
            MODEL_DIR,
            f"best_lstm_h{horizon}.keras"
        ),
        save_best_only=True,
        monitor="val_loss"
    )

    lstm.fit(
        X_train,
        y_train,
        validation_split=0.1,
        epochs=30,
        batch_size=32,
        callbacks=[
            EarlyStopping(
                patience=5,
                restore_best_weights=True
            ),
            lstm_checkpoint
        ],
        verbose=1
    )

    lstm_pred = lstm.predict(
        X_test
    )

    lstm_mse = mean_squared_error(
        y_test,
        lstm_pred
    )

    results.append(
        [
            horizon,
            rnn_mse,
            lstm_mse
        ]
    )

results_df = pd.DataFrame(
    results,
    columns=[
        "Prediction_Horizon",
        "RNN_MSE",
        "LSTM_MSE"
    ]
)

results_df.to_csv(
    os.path.join(
        MODEL_DIR,
        "model_comparison.csv"
    ),
    index=False
)

print(
    "\nTraining Completed"
)

print(results_df)