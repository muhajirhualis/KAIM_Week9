from statsmodels.tsa.stattools import adfuller
import numpy as np

class StationarityTester:
    @staticmethod
    def adf_test(series, name="Series"):
        result = adfuller(series.dropna())
        p_value = result[1]
        is_stationary = p_value < 0.05
        print(f"ADF Test for {name}:")
        print(f"  ADF Statistic: {result[0]:.6f}")
        print(f"  p-value: {p_value:.10f}")
        print(f"  Stationary: {'Yes' if is_stationary else 'No'}\n")
        return p_value, is_stationary
