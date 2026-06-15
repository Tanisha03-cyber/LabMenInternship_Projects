from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Dense,
    Dropout,
    SimpleRNN,
    LSTM
)
from tensorflow.keras.optimizers import Adam


def build_rnn(
    units=50,
    dropout=0.2,
    learning_rate=0.001
):

    model = Sequential()

    model.add(
        SimpleRNN(
            units=units,
            input_shape=(60, 1)
        )
    )

    model.add(
        Dropout(dropout)
    )

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    return model


def build_lstm(
    units=50,
    dropout=0.2,
    learning_rate=0.001
):

    model = Sequential()

    model.add(
        LSTM(
            units=units,
            input_shape=(60, 1)
        )
    )

    model.add(
        Dropout(dropout)
    )

    model.add(Dense(1))

    model.compile(
        optimizer=Adam(
            learning_rate=learning_rate
        ),
        loss="mse"
    )

    return model