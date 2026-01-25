

# **Portfolio Optimization with Time Series Forecasting**  
**Interim Submission – Week 9**  
**Prepared By:** Muhajer Hualis  
**Date:** 25 January 2026  

---

##  Overview

This project applies **time series forecasting** to historical financial data for three key assets—**TSLA (Tesla)**, **BND (Vanguard Bond ETF)**, and **SPY (S&P 500 ETF)**—to support **data-driven portfolio optimization** at GMF Investments.

Recognizing the **Efficient Market Hypothesis (EMH)**, this work treats forecasts not as deterministic price predictions, but as **probabilistic inputs** for:
- Volatility estimation  
- Risk-aware asset allocation  
- Enhanced decision-making within a **Modern Portfolio Theory (MPT)** framework  

This interim submission covers **Task 1 (Data Preprocessing & EDA)** and **initial progress on Task 2 (ARIMA modeling)**.

---

##  Data Source

- **Provider**: Yahoo Finance (`yfinance` Python library)  
- **Assets**: `TSLA`, `BND`, `SPY`  
- **Period**: January 1, 2015 – January 15, 2026  
- **Fields**: Open, High, Low, Close, Adj Close, Volume  

---

##  Completed Work (Interim)

### **Task 1: Data Preprocessing & Exploratory Data Analysis**
- Fetched and cleaned daily price data for all three assets  
- Handled missing values via **time-based interpolation**  
- Conducted comprehensive EDA:
  - Closing price trends (2015–2026)
  - Daily returns and volatility clustering
  - Rolling 30-day mean and standard deviation
- Performed **outlier detection** (e.g., TSLA +22.69% on 2025-04-09)
- Conducted **Augmented Dickey-Fuller (ADF) tests**:
  - Prices: **non-stationary** (p > 0.05)
  - Returns: **stationary** (p < 0.0001) → suitable for ARIMA
- Calculated risk metrics:
  | Asset | VaR (95%) | Sharpe Ratio |
  |-------|-----------|--------------|
  | TSLA  | -4.06%    | 0.71         |
  | BND   | -0.34%    | 0.34         |
  | SPY   | -1.22%    | 0.71         |

### **Task 2: Initial Forecasting Model (ARIMA)**
- Implemented **ARIMA(1,0,1)** on **TSLA daily returns**
- Chronological train/test split:
  - Train: 2015–2024  
  - Test: 2025–2026
- Converted return forecasts to price forecasts
- Evaluated performance:
  - **MAE**: \$153.28  
  - **RMSE**: \$163.61  
  - **MAPE**: 45.82%

> *Note: High MAPE reflects TSLA’s extreme volatility; model captures directional trend reasonably well.*

---

##  Repository Structure

```
portfolio-optimization/
├── .gitignore
├── requirements.txt
├── README.md                 ← This file
├── data/
│   └── processed/            # Cleaned CSVs: TSLA_clean.csv, etc.
├── notebooks/
│   ├── task1_eda.ipynb       # Full EDA & risk analysis
│   └── task2_arima_lstm.ipynb # ARIMA implementation (LSTM in progress)
└── src/
    ├── __init__.py
    ├── data_loader.py        # Fetch & clean data
    ├── eda.py                # Visualization methods
    ├── stationarity.py       # ADF testing
    ├── risk_metrics.py       # VaR, Sharpe Ratio
    └── arima_model.py        # ARIMA forecaster class
```

---

##  Setup Instructions

1. **Clone the repository**
   ```bash
   git clone https://github.com/muhajirhualis/kaim_week9.git
   cd kaim_week9
   ```

2. **Create and activate virtual environment**
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
   - Launch Jupyter:
     ```bash
     jupyter notebook
     ```
   - Open notebooks in `notebooks/` to reproduce results

---

##  Next Steps (Final Submission – Jan 27)

- [ ] Complete **LSTM model** for TSLA  
- [ ] Compare ARIMA vs LSTM (MAE, RMSE, MAPE)  
- [ ] Generate **6–12 month forecasts** with confidence intervals  
- [ ] Implement **portfolio optimization** (Efficient Frontier, Max Sharpe, Min Volatility)  
- [ ] **Backtest** strategy vs 60% SPY / 40% BND benchmark  

---

