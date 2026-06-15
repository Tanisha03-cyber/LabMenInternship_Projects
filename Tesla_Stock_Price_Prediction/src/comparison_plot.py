import os
import pandas as pd
import matplotlib.pyplot as plt

BASE_DIR = os.path.dirname(
    os.path.dirname(
        os.path.abspath(__file__)
    )
)

metrics_file = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics",
    "evaluation_metrics.csv"
)

fig_dir = os.path.join(
    BASE_DIR,
    "outputs",
    "figures"
)

df = pd.read_csv(metrics_file)

plt.figure(figsize=(10,6))

plt.plot(
    df["Horizon"],
    df["RNN_MSE"],
    marker="o",
    label="RNN"
)

plt.plot(
    df["Horizon"],
    df["LSTM_MSE"],
    marker="o",
    label="LSTM"
)

plt.xlabel(
    "Prediction Horizon"
)

plt.ylabel(
    "MSE"
)

plt.title(
    "RNN vs LSTM Comparison"
)

plt.legend()

plt.savefig(
    os.path.join(
        fig_dir,
        "rnn_vs_lstm_comparison.png"
    )
)

plt.close()

print("Comparison Plot Saved")