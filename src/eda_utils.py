import matplotlib.pyplot as plt
import numpy as np

class EDA:
    def __init__(self, data_dict):
        self.data = data_dict

    def plot_closing_prices(self):
        plt.figure(figsize=(14, 6))
        for ticker, df in self.data.items():
            plt.plot(df.index, df['Close'], label=ticker)
        plt.title("Closing Prices (2015–2026)")
        plt.xlabel("Date")
        plt.ylabel("Price (USD)")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_daily_returns(self):
        plt.figure(figsize=(14, 5))
        for ticker, df in self.data.items():
            returns = df['Close'].pct_change().dropna()
            plt.plot(returns.index, returns, label=ticker, linewidth=0.7)
        plt.title("Daily Returns")
        plt.axhline(0, color='black', linestyle='-', linewidth=0.5)
        plt.xlabel("Date")
        plt.ylabel("Daily Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def plot_rolling_stats(self, window=30):
        plt.figure(figsize=(14, 6))
        for ticker, df in self.data.items():
            returns = df['Close'].pct_change().dropna()
            rolling_mean = returns.rolling(window).mean()
            rolling_std = returns.rolling(window).std()
            plt.plot(rolling_mean.index, rolling_mean, label=f"{ticker} Mean")
            plt.fill_between(
                rolling_std.index,
                rolling_mean - rolling_std,
                rolling_mean + rolling_std,
                alpha=0.2,
                label=f"{ticker} ±1σ"
            )
        plt.title(f"Rolling {window}-Day Mean and Std Dev of Returns")
        plt.xlabel("Date")
        plt.ylabel("Return")
        plt.legend()
        plt.grid(True, alpha=0.3)
        plt.tight_layout()
        plt.show()

    def detect_outliers(self, threshold=3):
        outliers = {}
        for ticker, df in self.data.items():
            returns = df['Close'].pct_change().dropna()
            rolling_std = returns.rolling(30).std()
            z_scores = returns / rolling_std
            extreme = returns[np.abs(z_scores) > threshold]
            outliers[ticker] = extreme.sort_values(key=abs, ascending=False).head(5)
            print(f"\nTop 5 Outliers for {ticker}:")
            print(extreme.head(5))
        return outliers