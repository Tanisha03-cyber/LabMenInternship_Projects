import os
import sys
import json

from scikeras.wrappers import KerasRegressor
from sklearn.model_selection import GridSearchCV

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
    create_sequences
)

from model_builder import build_lstm

# -------------------------
# Load Dataset
# -------------------------

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "TSLA.csv"
)

df = load_data(DATA_PATH)

scaled_data, scaler = scale_data(
    df,
    target_column="Adj Close"
)

X, y = create_sequences(
    scaled_data,
    lookback=60,
    horizon=1
)

split = int(
    len(X) * 0.8
)

X_train = X[:split]
y_train = y[:split]

# -------------------------
# Model Wrapper
# -------------------------

model = KerasRegressor(
    model=build_lstm,
    verbose=0
)

# -------------------------
# Hyperparameter Grid
# -------------------------

param_grid = {

    "model__units": [32, 50, 100],

    "model__dropout": [
        0.1,
        0.2,
        0.3
    ],

    "model__learning_rate": [
        0.001,
        0.0001
    ],

    "batch_size": [
        16,
        32
    ],

    "epochs": [
        20
    ]
}

grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=3,
    scoring="neg_mean_squared_error",
    n_jobs=1
)

grid.fit(
    X_train,
    y_train
)

print("\nBest Parameters")
print(grid.best_params_)

OUTPUT_DIR = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics"
)

os.makedirs(
    OUTPUT_DIR,
    exist_ok=True
)

with open(
    os.path.join(
        OUTPUT_DIR,
        "best_parameters.json"
    ),
    "w"
) as f:

    json.dump(
        grid.best_params_,
        f,
        indent=4
    )

print(
    "\nSaved best parameters."
)