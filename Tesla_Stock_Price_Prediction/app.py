import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import plotly.express as px

from tensorflow.keras.models import load_model

# --------------------------------------------------
# Paths
# --------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_PATH = os.path.join(
    BASE_DIR,
    "data",
    "TSLA.csv"
)

MODEL_DIR = os.path.join(
    BASE_DIR,
    "models"
)

METRICS_PATH = os.path.join(
    BASE_DIR,
    "outputs",
    "metrics",
    "evaluation_metrics.csv"
)

# --------------------------------------------------
# Load Dataset
# --------------------------------------------------

df = pd.read_csv(DATA_PATH)

# --------------------------------------------------
# Sidebar
# --------------------------------------------------

st.sidebar.title("Tesla Stock Dashboard")

page = st.sidebar.radio(
    "Navigation",
    [
        "Dataset Overview",
        "EDA",
        "Model Performance",
        "Future Prediction",
        "Business Insights"
    ]
)

# ==================================================
# PAGE 1
# ==================================================

if page == "Dataset Overview":

    st.title(
        "Tesla Stock Price Prediction"
    )

    st.subheader(
        "Dataset Overview"
    )

    st.write(
        f"Rows: {df.shape[0]}"
    )

    st.write(
        f"Columns: {df.shape[1]}"
    )

    st.dataframe(
        df.head()
    )

    st.subheader(
        "Summary Statistics"
    )

    st.dataframe(
        df.describe()
    )

# ==================================================
# PAGE 2
# ==================================================

elif page == "EDA":

    st.title(
        "Exploratory Data Analysis"
    )

    fig = px.line(
        df,
        x="Date",
        y="Adj Close",
        title="Adjusted Close Price Trend"
    )

    st.plotly_chart(
        fig,
        use_container_width=True
    )

    fig2 = px.line(
        df,
        x="Date",
        y="Volume",
        title="Trading Volume Trend"
    )

    st.plotly_chart(
        fig2,
        use_container_width=True
    )

    df["Daily_Return"] = (
        df["Adj Close"]
        .pct_change()
    )

    fig3 = px.histogram(
        df,
        x="Daily_Return",
        title="Daily Returns Distribution"
    )

    st.plotly_chart(
        fig3,
        use_container_width=True
    )

# ==================================================
# PAGE 3
# ==================================================

elif page == "Model Performance":

    st.title(
        "Model Evaluation"
    )

    if os.path.exists(METRICS_PATH):

        metrics_df = pd.read_csv(
            METRICS_PATH
        )

        st.dataframe(
            metrics_df
        )

        fig = px.line(
            metrics_df,
            x="Horizon",
            y=[
                "RNN_MSE",
                "LSTM_MSE"
            ],
            markers=True,
            title="RNN vs LSTM MSE"
        )

        st.plotly_chart(
            fig,
            use_container_width=True
        )

    else:

        st.warning(
            "Run evaluate_models.py first."
        )

# ==================================================
# PAGE 4
# ==================================================

elif page == "Future Prediction":

    st.title(
        "Future Stock Prediction"
    )

    horizon = st.selectbox(
        "Select Horizon",
        [1,5,10]
    )

    model_path = os.path.join(
        MODEL_DIR,
        f"best_lstm_h{horizon}.keras"
    )

    scaler_path = os.path.join(
        MODEL_DIR,
        "scaler.pkl"
    )

    if os.path.exists(model_path):

        scaler = joblib.load(
            scaler_path
        )

        model = load_model(
            model_path
        )

        close_prices = (
            df["Adj Close"]
            .values
        )

        last_60 = close_prices[-60:]

        scaled = scaler.transform(
            last_60.reshape(-1,1)
        )

        X = scaled.reshape(
            1,
            60,
            1
        )

        prediction = model.predict(X)

        prediction = (
            scaler.inverse_transform(
                prediction
            )
        )

        st.metric(
            label=f"Predicted Price after {horizon} Day(s)",
            value=f"${prediction[0][0]:.2f}"
        )

    else:

        st.warning(
            "Train models first."
        )

# ==================================================
# PAGE 5
# ==================================================

elif page == "Business Insights":

    st.title(
        "Business Use Cases"
    )

    st.markdown(
        """
### Automated Trading

Use model predictions to create
algorithmic trading strategies.

### Portfolio Optimization

Predict future trends and adjust
portfolio allocations.

### Long-Term Investing

Analyze Tesla growth trends
for investment decisions.

### Competitor Benchmarking

Compare Tesla with:

- Rivian
- Lucid
- NIO

### Future Enhancements

- GRU Networks
- Transformer Models
- News Sentiment Analysis
- Social Media Analytics
- Macroeconomic Indicators
        """
    )