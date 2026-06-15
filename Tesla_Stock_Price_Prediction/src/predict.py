import os
import joblib
import pandas as pd

from tensorflow.keras.models import load_model

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

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

df = pd.read_csv(DATA_PATH)

scaler = joblib.load(
    os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )
)

close_prices = df["Adj Close"].values

last_60 = close_prices[-60:]

scaled = scaler.transform(
    last_60.reshape(-1,1)
)

X = scaled.reshape(
    1,
    60,
    1
)

for horizon in [1,5,10]:

    model = load_model(
        os.path.join(
            MODEL_DIR,
            f"best_lstm_h{horizon}.keras"
        )
    )

    prediction = model.predict(X)

    prediction = scaler.inverse_transform(
        prediction
    )

    print(
        f"\nPredicted Adj Close after {horizon} day(s): "
        f"${prediction[0][0]:.2f}"
    )