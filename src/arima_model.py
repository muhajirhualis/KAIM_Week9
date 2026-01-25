import pandas as pd
import numpy as np
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_absolute_error, mean_squared_error

class ARIMAForecaster:
    def __init__(self, returns_train, returns_test, order=(1, 0, 1)):
        self.returns_train = returns_train
        self.returns_test = returns_test
        self.order = order
        self.model = None
        self.forecast_returns = None
        self.forecast_prices = None

    def fit(self):
        self.model = ARIMA(self.returns_train, order=self.order)
        self.fitted_model = self.model.fit()
        print(self.fitted_model.summary())

    def forecast(self):
        steps = len(self.returns_test)
        self.forecast_returns = self.fitted_model.forecast(steps=steps)
        self.forecast_returns.index = self.returns_test.index

    def convert_to_prices(self, last_price):
        self.forecast_prices = last_price * (1 + self.forecast_returns).cumprod()

    def evaluate(self, actual_prices):
        mae = mean_absolute_error(actual_prices, self.forecast_prices)
        rmse = np.sqrt(mean_squared_error(actual_prices, self.forecast_prices))
        mape = np.mean(np.abs((actual_prices - self.forecast_prices) / actual_prices)) * 100
        print("\nForecast Evaluation Metrics:")
        print(f"  MAE: ${mae:.2f}")
        print(f"  RMSE: ${rmse:.2f}")
        print(f"  MAPE: {mape:.2f}%")
        return {"MAE": mae, "RMSE": rmse, "MAPE": mape}

