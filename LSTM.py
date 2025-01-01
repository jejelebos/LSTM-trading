import yfinance as yf
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
import tensorflow as tf
import matplotlib.pyplot as plt

# List of currencies to process
symbols = ['EURUSD=X']#, 'USDJPY=X', 'GBPUSD=X', 'AUDUSD=X', 'USDCAD=X']

for symbol in symbols:
    print(f"Processing {symbol}...")

    # Download the data from Yahoo Finance
    data = yf.download(symbol, start='2010-01-01', end='2023-01-01')

    # Use only the closing price column
    data_close = data['Close'].values.reshape(-1, 1)

    # Normalize the data between 0 and 1
    scaler = MinMaxScaler(feature_range=(0, 1))
    data_scaled = scaler.fit_transform(data_close)

    # Create time sequences
    def create_dataset(data, time_step=60):
        X, y = [], []
        for i in range(len(data) - time_step):
            X.append(data[i:(i + time_step), 0])
            y.append(data[i + time_step, 0])
        return np.array(X), np.array(y)

    time_step = 60  # Use the last 60 days to predict
    X, y = create_dataset(data_scaled, time_step)

    # Reshape the input to meet LSTM requirements
    X = X.reshape(X.shape[0], X.shape[1], 1)

    # Split the data into training and testing sets
    train_size = int(len(X) * 0.8)
    X_train, X_test = X[:train_size], X[train_size:]
    y_train, y_test = y[:train_size], y[train_size:]

    # Create the enhanced LSTM model
    model = tf.keras.Sequential([
        tf.keras.layers.LSTM(units=128, return_sequences=True, input_shape=(X_train.shape[1], 1)),
        tf.keras.layers.Dropout(0.2),  # Dropout to prevent overfitting
        tf.keras.layers.LSTM(units=128, return_sequences=True),
        tf.keras.layers.Dropout(0.2),  # Dropout
        tf.keras.layers.LSTM(units=128, return_sequences=False),
        tf.keras.layers.Dropout(0.2),  # Dropout
        tf.keras.layers.Dense(units=64, activation='relu'),
        tf.keras.layers.Dense(units=1)  # Price prediction
    ])

    # Compile the model with a lower learning rate
    model.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), loss='mean_squared_error')

    # Train the model with more epochs for better accuracy
    model.fit(X_train, y_train, epochs=50, batch_size=64)

    # Predictions
    predictions = model.predict(X_test)

    # Inverse the normalization to get the real prices
    predictions = scaler.inverse_transform(predictions)
    y_test = scaler.inverse_transform(y_test.reshape(-1, 1))

    # Visualize the results
    plt.figure(figsize=(12, 6))
    plt.plot(y_test, color='blue', label='Real price')
    plt.plot(predictions, color='red', label='Predicted price')
    plt.title(f'Price prediction with LSTM ({symbol})')
    plt.xlabel('Time')
    plt.ylabel('Price')
    plt.legend()
    plt.show()
