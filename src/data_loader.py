import yfinance as yf
import pandas as pd
import os
from pathlib import Path

class DataLoader:
    def __init__(self, tickers, start_date, end_date, processed_dir="../data/processed"):
        self.tickers = tickers
        self.start_date = start_date
        self.end_date = end_date
        self.processed_dir = Path(processed_dir)
        self.data = {}
        self.processed_dir.mkdir(parents=True, exist_ok=True)

    def fetch_data(self):
        for ticker in self.tickers:
            print(f"Fetching {ticker}...")
            df = yf.download(ticker, start=self.start_date, end=self.end_date)
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.get_level_values(0)
            self.data[ticker] = df[["Open", "High", "Low", "Close","Volume"]]

    def clean_data(self):
        for ticker, df in self.data.items():
            # Ensure datetime index
            df.index = pd.to_datetime(df.index)
            # Handle missing values via time interpolation
            df = df.asfreq('D')
            df = df.interpolate(method='time')
            df = df.dropna()
            self.data[ticker] = df

    def save_processed(self):
        for ticker, df in self.data.items():
            df.to_csv(self.processed_dir / f"{ticker}_clean.csv")

    def load_processed(self):
        self.data = {}
        for ticker in self.tickers:
            path = self.processed_dir / f"{ticker}_clean.csv"
            if path.exists():
                self.data[ticker] = pd.read_csv(path, index_col=0, parse_dates=True)
            else:
                raise FileNotFoundError(f"Processed file not found: {path}")