
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

class PortfolioBacktester:
    def __init__(self, prices_df, start_date, end_date):
        """
        Initialize backtester with price data and period.
        
        Parameters:
        - prices_df: DataFrame with columns ['TSLA', 'BND', 'SPY'] (daily closing prices)
        - start_date, end_date: str, e.g., "2025-01-01", "2026-01-15"
        """
        self.prices = prices_df[['TSLA', 'BND', 'SPY']].loc[start_date:end_date].copy()
        self.returns = self.prices.pct_change().dropna()
        self.dates = self.prices.index

    def simulate_portfolio(self, weights, name="Strategy"):
        """
        Simulate portfolio performance with fixed weights (no rebalancing).
        
        Parameters:
        - weights: dict like {'TSLA': 0.6, 'BND': 0.4, 'SPY': 0.0}
        - name: label for plotting
        
        Returns:
        - portfolio_value: Series of cumulative portfolio value (starting at 1.0)
        """
        # Align weights with asset order
        w = np.array([weights.get(asset, 0.0) for asset in ['TSLA', 'BND', 'SPY']])
        
        # Compute daily portfolio returns
        port_returns = (self.returns * w).sum(axis=1)
        
        # Cumulative value (start at $1)
        cum_value = (1 + port_returns).cumprod()
        cum_value.iloc[0] = 1.0  # Ensure starts at 1.0
        
        return cum_value

    def calculate_metrics(self, cum_value, risk_free_rate=0.02):
        """
        Calculate performance metrics from cumulative value series.
        """
        total_return = cum_value.iloc[-1] - 1.0
        n_days = len(cum_value)
        annualized_return = (1 + total_return) ** (252 / n_days) - 1
        
        daily_returns = cum_value.pct_change().dropna()
        sharpe = (daily_returns.mean() - risk_free_rate / 252) / daily_returns.std() * np.sqrt(252)
        
        # Max drawdown
        rolling_max = cum_value.cummax()
        drawdown = (cum_value - rolling_max) / rolling_max
        max_drawdown = drawdown.min()
        
        return {
            "Total Return": total_return,
            "Annualized Return": annualized_return,
            "Sharpe Ratio": sharpe,
            "Max Drawdown": max_drawdown
        }

    def plot_performance(self, strategy_cum, benchmark_cum, strategy_name="Strategy", benchmark_name="Benchmark"):
        plt.figure(figsize=(12, 6))
        plt.plot(strategy_cum.index, strategy_cum.values, label=strategy_name, linewidth=2)
        plt.plot(benchmark_cum.index, benchmark_cum.values, label=benchmark_name, linewidth=2, linestyle='--')
        plt.title("Portfolio Backtest: Strategy vs Benchmark (Jan 2025 – Jan 2026)")
        plt.xlabel("Date")
        plt.ylabel("Portfolio Value ($)")
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()