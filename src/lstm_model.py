
import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense
import warnings
warnings.filterwarnings("ignore")

class LSTMForecaster:
    def __init__(self, returns_train, returns_test, window_size=60, epochs=20, batch_size=32):
        self.returns_train = returns_train
        self.returns_test = returns_test
        self.window_size = window_size
        self.epochs = epochs
        self.batch_size = batch_size
        self.scaler = MinMaxScaler(feature_range=(-1, 1))
        self.model = None
        self.forecast_returns = None
        self.forecast_prices = None

    def _create_sequences(self, data):
        X, y = [], []
        for i in range(self.window_size, len(data)):
            X.append(data[i - self.window_size:i, 0])
            y.append(data[i, 0])
        return np.array(X), np.array(y)

    def fit(self):
        # Scale training returns
        train_scaled = self.scaler.fit_transform(self.returns_train.values.reshape(-1, 1))
        
        # Create sequences
        X_train, y_train = self._create_sequences(train_scaled)
        X_train = X_train.reshape((X_train.shape[0], X_train.shape[1], 1))
        
        # Build model
        self.model = Sequential([
            LSTM(50, activation='tanh', input_shape=(self.window_size, 1)),
            Dense(1)
        ])
        self.model.compile(optimizer='adam', loss='mse')
        
        # Train
        self.model.fit(
            X_train, y_train,
            epochs=self.epochs,
            batch_size=self.batch_size,
            validation_split=0.1,
            verbose=0
        )

    def forecast(self):
        # Prepare full scaled sequence (train + test)
        test_scaled = self.scaler.transform(self.returns_test.values.reshape(-1, 1))
        full_scaled = np.concatenate([self.scaler.transform(self.returns_train.values.reshape(-1, 1)), test_scaled], axis=0)
        
        # Create test sequences
        X_test, _ = self._create_sequences(full_scaled)
        X_test = X_test[-len(self.returns_test):]
        X_test = X_test.reshape((X_test.shape[0], X_test.shape[1], 1))
        
        # Predict
        y_pred_scaled = self.model.predict(X_test, verbose=0)
        self.forecast_returns = self.scaler.inverse_transform(y_pred_scaled).flatten()

    def convert_to_prices(self, last_price):
        prices = [last_price]
        for r in self.forecast_returns:
            prices.append(prices[-1] * (1 + r))
        self.forecast_prices = pd.Series(prices[1:], index=self.returns_test.index)

    def evaluate(self, actual_prices):
        mae = mean_absolute_error(actual_prices, self.forecast_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, self.forecast_prices))
        mape = np.mean(np.abs((actual_prices - self.forecast_prices) / actual_prices)) * 100
        print("\nLSTM Forecast Evaluation Metrics:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}