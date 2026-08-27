import yfinance as yf
import numpy as np
import pandas as pd

US10Y_CSV = "data_us10y.csv"

TICKER = "SGOV"
PERIOD = "8y"
INTERVAL = "1mo"

data = yf.download(TICKER, period=PERIOD, interval=INTERVAL, auto_adjust=True)
prices = data["Close"].squeeze().dropna()

returns = prices.pct_change().dropna()
returns.index = returns.index.to_period("M")

us10y = pd.read_csv(US10Y_CSV)
us10y = us10y[us10y["indicator_ticker"] == "US_10Y"][["record_date", "value"]]

us10y["record_date"] = pd.to_datetime(us10y["record_date"])
us10y = us10y.sort_values("record_date").set_index("record_date")

us10y_monthly = us10y["value"].resample("ME").mean() / 100
us10y_monthly.index = us10y_monthly.index.to_period("M")

rf_monthly = (1 + us10y_monthly) ** (1 / 12) - 1

df = pd.concat([returns.rename("soxx_return"), rf_monthly.rename("rf_monthly")], axis=1).dropna()

df["excess_return"] = df["soxx_return"] - df["rf_monthly"]

mean_excess = df["excess_return"].mean()
std_excess = df["excess_return"].std()

periods_per_year = 12
annualized_mean_excess = mean_excess * periods_per_year
annualized_std_excess = std_excess * np.sqrt(periods_per_year)

sharpe_ratio = annualized_mean_excess / annualized_std_excess

print(f"Ticker: {TICKER}")
print(f"Periode data: {PERIOD}, interval: {INTERVAL}")
print(f"Jumlah bulan data (setelah join dengan US_10Y): {len(df)}")
print(f"Mean excess return (bulanan): {mean_excess:.4%}")
print(f"Std dev excess return (bulanan): {std_excess:.4%}")
print(f"Annualized mean excess return: {annualized_mean_excess:.4%}")
print(f"Annualized std dev: {annualized_std_excess:.4%}")
print(f"Sharpe Ratio (dynamic risk-free, US_10Y): {sharpe_ratio:.4f}")