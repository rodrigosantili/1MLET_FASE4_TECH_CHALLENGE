import yfinance as yf
import numpy as np
import joblib
from sklearn.preprocessing import MinMaxScaler
from tensorflow.keras.layers import LSTM, Dense, Input
from tensorflow.keras.models import Sequential
from sklearn.metrics import mean_absolute_error, mean_squared_error, mean_absolute_percentage_error

# Coleta dos dados
SYMBOL = 'DIS'
START_DATE = '2018-01-01'
END_DATE = '2024-07-20'

# Hiperparâmetros
SEQ_LENGTH = 60
EPOCHS = 20
BATCH_SIZE = 32
LSTM_UNITS = 50
TRAIN_SPLIT = 0.8

# EXPORTAÇÃO DO MODELO
MODEL_PATH = "../api/src/ml/lstm_model.keras"
SCALER_PATH = "../api/src/ml/scaler.save"


def create_sequences(data, sequence_length):
    X = []
    y = []
    for i in range(sequence_length, len(data)):
        X.append(data[i - sequence_length:i, 0])
        y.append(data[i, 0])
    return np.array(X), np.array(y)


if __name__ == "__main__":
    df = yf.download(SYMBOL, start=START_DATE, end=END_DATE)
    df = df[['Close']]

    scaler = MinMaxScaler(feature_range=(0, 1))
    scaled_data = scaler.fit_transform(df)

    X, y = create_sequences(scaled_data, SEQ_LENGTH)

    X = np.reshape(X, (X.shape[0], X.shape[1], 1))

    train_size = int(len(X) * TRAIN_SPLIT)
    X_train, X_val = X[:train_size], X[train_size:]
    y_train, y_val = y[:train_size], y[train_size:]

    model = Sequential()
    model.add(Input(shape=(SEQ_LENGTH, 1)))
    model.add(LSTM(units=LSTM_UNITS, return_sequences=False))  # Correct import used
    model.add(Dense(1))
    model.compile(optimizer='adam', loss='mean_squared_error')
    model.summary()

    history = model.fit(X_train, y_train, epochs=EPOCHS, batch_size=BATCH_SIZE, validation_data=(X_val, y_val))

    y_pred = model.predict(X_val)
    y_val_rescaled = scaler.inverse_transform(y_val.reshape(-1, 1))
    y_pred_rescaled = scaler.inverse_transform(y_pred)

    mae = mean_absolute_error(y_val_rescaled, y_pred_rescaled)
    rmse = np.sqrt(mean_squared_error(y_val_rescaled, y_pred_rescaled))
    mape = mean_absolute_percentage_error(y_val_rescaled, y_pred_rescaled)

    print(f'MAE: {mae}')
    print(f'RMSE: {rmse}')
    print(f'MAPE: {mape}')

    model.save(MODEL_PATH)
    print(f"Modelo salvo em: {MODEL_PATH}")

    joblib.dump(scaler, SCALER_PATH)
    print(f"Scaler salvo em: {SCALER_PATH}")
