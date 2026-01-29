# **Portfolio Optimization with Time Series Forecasting**  
*Prepared By: Muhajer Hualis | 10 Academy AI Mastery Program*

---

##  Overview

This project develops a **data-driven portfolio strategy** for **GMF Investments**, a financial advisory firm specializing in personalized portfolio management. Using historical data for **TSLA (Tesla)**, **BND (Vanguard Bond ETF)**, and **SPY (S&P 500 ETF)** from **January 2015 to January 2026**, we:

1. **Forecast** TSLA’s future returns using ARIMA and LSTM models  
2. **Optimize** a 3-asset portfolio using Modern Portfolio Theory (MPT)  
3. **Backtest** the strategy against a passive benchmark  

Recognizing the **Efficient Market Hypothesis (EMH)**, forecasts are treated as **probabilistic inputs**—not deterministic predictions—to enhance risk-aware decision-making.

---

##  Methodology Summary

### **Task 1: Data Preprocessing & EDA**
- Fetched and cleaned daily price data via `yfinance`  
- Conducted EDA: closing prices, daily returns, rolling volatility  
- Performed outlier detection and ADF stationarity tests  
- Calculated risk metrics: **VaR (95%)** and **Sharpe Ratio**

### **Task 2: Time Series Forecasting**
- Implemented **ARIMA(1,0,1)** and **LSTM (60-day window)** on TSLA returns  
- Evaluated on **2025–2026 test period**:  
  - **LSTM MAPE = 17.69%** (best-performing model)  
  - **ARIMA MAPE = 45.82%**

### **Task 3: Future Trend Forecasting**
- Generated **12-month TSLA price forecast** using LSTM  
- Projected **+141.3% return** with rapidly widening confidence intervals  
- CI width expands **15.9×** over 12 months, highlighting long-term uncertainty

### **Task 4: Portfolio Optimization (MPT)**
- **Expected returns**:  
  - TSLA: **141.3%** (forecasted)  
  - BND/SPY: historical annualized mean  
- Applied **realistic constraints**:  
  - TSLA ≤ 60%, BND ≥ 10%  
- Recommended **60% TSLA / 40% BND** portfolio:  
  - **Expected Return**: 85.34%  
  - **Volatility**: 25.82%  
  - **Sharpe Ratio**: 3.23

### **Task 5: Strategy Backtesting**
- Backtested over **Jan 2025 – Jan 2026** (held-out data)  
- Compared vs **60% SPY / 40% BND benchmark**  
- **Results**:  
  | Metric | GMF Strategy | Benchmark |
  |--------|--------------|-----------|
  | Total Return | 14.72% | 14.94% |
  | Sharpe Ratio | 0.39 | 0.84 |
  | Max Drawdown | -31.4% | -11.3% |

> **Key Insight**: The strategy matched benchmark returns but with **significantly higher risk**, underscoring the challenge of translating forecasts into alpha.

---

##  Repository Structure

```
portfolio-optimization/
├── .gitignore
├── requirements.txt
├── README.md                 ← This file
├── data/
│   └── processed/            # Cleaned CSVs
├── notebooks/
│   ├── eda.ipynb       # EDA & risk metrics
│   ├── arima_lstm.ipynb # Model comparison
│   ├── forecast_trends.ipynb # 12-month forecast
│   ├── portfolio_optimization.ipynb # MPT with scipy
│   └── backtesting.ipynb # Strategy validation
└── src/
    ├── __init__.py
    ├── data_loader.py        # Fetch & clean data
    ├── eda.py                # Visualization methods
    ├── stationarity.py       # ADF testing
    ├── risk_metrics.py       # VaR, Sharpe
    ├── arima_model.py        # ARIMA forecaster
    ├── lstm_model.py         # LSTM forecaster
    ├── portfolio_optimizer.py # MPT with constraints
    └── backtester.py         # Strategy simulation
```

---

##  Setup Instructions

1. **Clone repository**
   ```bash
   git clone https://github.com/muhajirhualis/KAIM_Week9
   cd KAIM_Week9
   ```

2. **Create virtual environment**
   ```bash
   python -m venv venv
   source venv/bin/activate      # Linux/Mac
   # venv\Scripts\activate       # Windows
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Run notebooks**
   ```bash
   jupyter notebook
   ```
   Open notebooks in `notebooks/` to reproduce all results.

---

##  Dependencies (`requirements.txt`)

```txt
yfinance==0.2.37
pandas==2.1.4
numpy==1.26.4
matplotlib==3.8.2
scikit-learn==1.4.0
statsmodels==0.14.1
tensorflow==2.15.0
seaborn==0.13.2
scipy==1.17.0
```

> 💡 **Note**: Uses only `scipy.optimize` for portfolio optimization—no external solvers required.

---

##  Key Takeaways for GMF Investments

- **Forecasts are valuable but uncertain**: Use them for scenario planning, not precise targeting  
- **Diversification remains critical**: Even high-conviction views should be constrained  
- **Risk-adjusted returns > raw returns**: The benchmark’s superior Sharpe highlights this  
- **Model risk is real**: A strong in-sample forecast doesn’t guarantee out-of-sample alpha  

This project demonstrates a **rigorous, end-to-end workflow** from data to decision—aligning with GMF’s mission to deliver **personalized, evidence-based portfolio management**.

---
