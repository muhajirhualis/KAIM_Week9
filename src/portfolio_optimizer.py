# src/portfolio_optimizer_scipy.py
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.optimize import minimize

class PortfolioOptimizer:
    def __init__(self, returns_df, forecasted_tsla_return, risk_free_rate=0.02):
        """
        Initialize with historical returns and forecasted TSLA return.
        
        Parameters:
        - returns_df: DataFrame of daily returns for ['TSLA', 'BND', 'SPY']
        - forecasted_tsla_return: Annualized expected return for TSLA (e.g., 1.413)
        - risk_free_rate: Annual risk-free rate (default = 2%)
        """
        self.returns = returns_df[['TSLA', 'BND', 'SPY']]
        self.tickers = ['TSLA', 'BND', 'SPY']
        self.risk_free_rate = risk_free_rate
        
        # Expected returns vector (annualized)
        hist_annual = self.returns.mean() * 252
        hist_annual['TSLA'] = forecasted_tsla_return
        self.expected_returns = hist_annual.values  # shape: (3,)
        
        # Covariance matrix (annualized)
        self.cov_matrix = self.returns.cov().values * 252  # shape: (3,3)

    def portfolio_return(self, weights):
        return np.dot(weights, self.expected_returns)

    def portfolio_volatility(self, weights):
        return np.sqrt(np.dot(weights.T, np.dot(self.cov_matrix, weights)))

    def portfolio_sharpe_ratio(self, weights):
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        return -(ret - self.risk_free_rate) / vol  # Negative for minimization

    def optimize_max_sharpe(self):
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})  # weights sum to 1
        bounds = (
            (0.0, 0.60),   # TSLA: max 60%
            (0.10, 1.0),   # BND: at least 10%
            (0.0, 1.0)     # SPY: 0–100%
        )
        init_guess = np.array([1/num_assets] * num_assets)

        result = minimize(
            self.portfolio_sharpe_ratio,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x, self._get_performance(result.x)

    def optimize_min_volatility(self):
        num_assets = len(self.tickers)
        constraints = ({'type': 'eq', 'fun': lambda w: np.sum(w) - 1})
        bounds = (
            (0.0, 0.60),   # TSLA: max 60%
            (0.10, 1.0),   # BND: at least 10%
            (0.0, 1.0)     # SPY
        )
        init_guess = np.array([1/num_assets] * num_assets)

        def volatility_objective(weights):
            return self.portfolio_volatility(weights)

        result = minimize(
            volatility_objective,
            init_guess,
            method='SLSQP',
            bounds=bounds,
            constraints=constraints
        )
        return result.x, self._get_performance(result.x)

    def _get_performance(self, weights):
        ret = self.portfolio_return(weights)
        vol = self.portfolio_volatility(weights)
        sharpe = (ret - self.risk_free_rate) / vol
        return ret, vol, sharpe

    def plot_efficient_frontier(self, n_portfolios=1000):
        num_assets = len(self.tickers)
        results = []
        weights_list = []

        for _ in range(n_portfolios):
            w = np.random.random(num_assets)
            w /= np.sum(w)
            ret, vol, _ = self._get_performance(w)
            results.append([ret, vol])
            weights_list.append(w)

        results = np.array(results)
        
        # Optimize key portfolios
        max_sharpe_w, max_sharpe_perf = self.optimize_max_sharpe()
        min_vol_w, min_vol_perf = self.optimize_min_volatility()

        # Plot
        plt.figure(figsize=(12, 8))
        plt.scatter(results[:, 1], results[:, 0], c=(results[:, 0] - self.risk_free_rate)/results[:, 1], 
                    cmap='viridis', alpha=0.6)
        plt.colorbar(label='Sharpe Ratio')
        
        # Mark key portfolios
        plt.scatter(min_vol_perf[1], min_vol_perf[0], color='red', marker='*', s=300, label='Min Volatility')
        plt.scatter(max_sharpe_perf[1], max_sharpe_perf[0], color='blue', marker='*', s=300, label='Max Sharpe')
        
        plt.title('Efficient Frontier (Scipy Implementation)')
        plt.xlabel('Annualized Volatility')
        plt.ylabel('Annualized Return')
        plt.legend()
        plt.grid(alpha=0.3)
        plt.tight_layout()
        plt.show()

        return {
            'max_sharpe': {'weights': max_sharpe_w, 'performance': max_sharpe_perf},
            'min_vol': {'weights': min_vol_w, 'performance': min_vol_perf}
        }