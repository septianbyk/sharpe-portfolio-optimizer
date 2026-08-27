import yfinance as yf
import numpy as np
import pandas as pd
import time

TICKERS_FILE = "tickers.csv"
US10Y_CSV = "data_us10y.csv"
PERIOD = "2y"
INTERVAL = "1mo"
SLEEP_BETWEEN_REQUESTS = 0.5

tickers_df = pd.read_csv(TICKERS_FILE)

if "category" not in tickers_df.columns:
    if "sector" in tickers_df.columns:
        tickers_df["category"] = tickers_df["sector"]
    else:
        tickers_df["category"] = "Unknown"

if "label" not in tickers_df.columns:
    if "name" in tickers_df.columns:
        tickers_df["label"] = tickers_df["name"]
    else:
        tickers_df["label"] = tickers_df["ticker"]

us10y = pd.read_csv(US10Y_CSV)
us10y = us10y[us10y["indicator_ticker"] == "US_10Y"][["record_date", "value"]]

us10y["record_date"] = pd.to_datetime(us10y["record_date"])
us10y = us10y.sort_values("record_date").set_index("record_date")
us10y_monthly = us10y["value"].resample("ME").mean() / 100
us10y_monthly.index = us10y_monthly.index.to_period("M")
rf_monthly = (1 + us10y_monthly) ** (1 / 12) - 1

results = []

for _, row in tickers_df.iterrows():
    ticker = row["ticker"]
    category = row["category"]
    label = row["label"]

    try:
        data = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                            auto_adjust=True, progress=False)
        prices = data["Close"].squeeze().dropna()

        if len(prices) < 12:
            print(f"[SKIP]  {ticker}: data terlalu pendek ({len(prices)} bulan)")
            continue

        returns = prices.pct_change().dropna()
        returns.index = returns.index.to_period("M")

        df = pd.concat(
            [returns.rename("asset_return"), rf_monthly.rename("rf_monthly")],
            axis=1
        ).dropna()

        df["excess_return"] = df["asset_return"] - df["rf_monthly"]

        mean_excess = df["excess_return"].mean()
        std_excess = df["excess_return"].std()

        annualized_return = mean_excess * 12
        annualized_std = std_excess * np.sqrt(12)
        sharpe = annualized_return / annualized_std

        results.append({
            "ticker": ticker,
            "category": category,
            "label": label,
            "n_months": len(df),
            "annualized_return": annualized_return,
            "annualized_std": annualized_std,
            "sharpe_ratio": sharpe,
        })
        print(f"[OK]    {ticker:6s} ({label}): Sharpe = {sharpe:.4f}")

    except Exception as e:
        print(f"[ERROR] {ticker}: {e}")

    time.sleep(SLEEP_BETWEEN_REQUESTS)

results_df = pd.DataFrame(results).sort_values("sharpe_ratio", ascending=False)

display_df = results_df.copy()
display_df["annualized_return"] = display_df["annualized_return"].apply(lambda x: f"{x:.2%}")
display_df["annualized_std"] = display_df["annualized_std"].apply(lambda x: f"{x:.2%}")
display_df["sharpe_ratio"] = display_df["sharpe_ratio"].round(4)

print("\n" + "=" * 90)
print(f"HASIL SCAN SHARPE RATIO ({PERIOD}, diurutkan tertinggi ke terendah)")
print("=" * 90)
print(display_df.to_string(index=False))

results_df.to_csv("sharpe_ratio_scan_results.csv", index=False)
print("\nHasil disimpan ke sharpe_ratio_scan_results.csv")