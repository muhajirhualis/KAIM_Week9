import numpy as np

class RiskMetrics:
    @staticmethod
    def value_at_risk(returns, confidence=0.95):
        """Compute historical VaR at given confidence level."""
        var = np.percentile(returns, (1 - confidence) * 100)
        return var

    @staticmethod
    def sharpe_ratio(returns, risk_free_rate=0.0):
        """Annualized Sharpe Ratio (252 trading days)."""
        excess_return = returns - risk_free_rate / 252
        sharpe = excess_return.mean() / returns.std()
        return sharpe * np.sqrt(252)