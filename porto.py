import yfinance as yf
import numpy as np
import pandas as pd
from scipy.optimize import minimize

US10Y_CSV = "data_us10y.csv"
SCAN_RESULTS_CSV = "sharpe_ratio_scan_results.csv"
TOP_N = 4

TOTAL_CAPITAL = 6000
PERIOD = "8y"
INTERVAL = "1d"
MAX_WEIGHT_PER_ASSET = 0.50

scan_df = pd.read_csv(SCAN_RESULTS_CSV)
scan_df = scan_df.sort_values("sharpe_ratio", ascending=False)
CANDIDATE_TICKERS = scan_df["ticker"].head(TOP_N).tolist()

print(f"[INFO] Top {TOP_N} ticker dari {SCAN_RESULTS_CSV}: {CANDIDATE_TICKERS}\n")

us10y = pd.read_csv(US10Y_CSV)
us10y = us10y[us10y["indicator_ticker"] == "US_10Y"][["record_date", "value"]]
us10y["record_date"] = pd.to_datetime(us10y["record_date"])
us10y = us10y.sort_values("record_date").set_index("record_date")
rf_daily = (1 + us10y["value"] / 100) ** (1 / 252) - 1

price_data = {}
for ticker in CANDIDATE_TICKERS:
    data = yf.download(ticker, period=PERIOD, interval=INTERVAL,
                        auto_adjust=True, progress=False)
    prices = data["Close"].squeeze().dropna()
    price_data[ticker] = prices

prices_df = pd.DataFrame(price_data).dropna()
returns_df = prices_df.pct_change().dropna()

combined = returns_df.join(rf_daily.rename("rf_daily")).dropna()
excess_returns_df = combined[CANDIDATE_TICKERS].sub(combined["rf_daily"], axis=0)

n_assets = len(CANDIDATE_TICKERS)
n_days = len(excess_returns_df)

mean_excess_annual = excess_returns_df.mean() * 252
cov_matrix_annual = excess_returns_df.cov() * 252

def negative_sharpe(weights, mean_returns, cov_matrix):
    port_return = np.dot(weights, mean_returns)
    port_std = np.sqrt(weights.T @ cov_matrix @ weights)
    return -port_return / port_std

constraints = ({"type": "eq", "fun": lambda w: np.sum(w) - 1})
bounds = tuple((0, MAX_WEIGHT_PER_ASSET) for _ in range(n_assets))
initial_guess = np.array([1 / n_assets] * n_assets)

np.random.seed(42)
best_result = None
best_sharpe = -np.inf
N_RESTARTS = 20

for attempt in range(N_RESTARTS):
    if attempt == 0:
        start_point = initial_guess
    else:
        random_w = np.random.dirichlet(np.ones(n_assets))
        start_point = random_w

    res = minimize(
        negative_sharpe,
        start_point,
        args=(mean_excess_annual.values, cov_matrix_annual.values),
        method="SLSQP",
        bounds=bounds,
        constraints=constraints,
    )

    if res.success:
        candidate_sharpe = -res.fun
        if candidate_sharpe > best_sharpe:
            best_sharpe = candidate_sharpe
            best_result = res

if best_result is None:
    raise RuntimeError("Optimasi GAGAL konvergen di semua percobaan -- cek data/constraints.")

result = best_result
optimal_weights = result.x

print(f"[INFO] Optimasi konvergen: {result.success} | "
      f"Terbaik dari {N_RESTARTS} percobaan restart")
print(f"[INFO] Pesan solver: {result.message}\n")
optimal_port_return = np.dot(optimal_weights, mean_excess_annual.values)
optimal_port_std = np.sqrt(optimal_weights.T @ cov_matrix_annual.values @ optimal_weights)
optimal_sharpe = optimal_port_return / optimal_port_std

print("=" * 70)
print(f"MAX SHARPE PORTFOLIO -- {n_days} hari data, periode {PERIOD}")
print("=" * 70)
print(f"\n{'Ticker':<8}{'Bobot':>10}{'Alokasi ($)':>15}")
print("-" * 35)
for ticker, w in zip(CANDIDATE_TICKERS, optimal_weights):
    print(f"{ticker:<8}{w:>9.2%}{w * TOTAL_CAPITAL:>15,.2f}")

print("-" * 35)
print(f"{'TOTAL':<8}{sum(optimal_weights):>9.2%}{sum(optimal_weights) * TOTAL_CAPITAL:>15,.2f}")

print(f"\nExpected annualized excess return : {optimal_port_return:.2%}")
print(f"Expected annualized volatility     : {optimal_port_std:.2%}")
print(f"Portfolio Sharpe Ratio             : {optimal_sharpe:.4f}")

print("\n" + "=" * 70)
print("MATRIKS KORELASI ANTAR ASET (0 = tidak berkorelasi, 1 = bergerak identik)")
print("=" * 70)
corr_matrix = excess_returns_df.corr()
print(corr_matrix.round(2).to_string())

N_SIMULATIONS = 10000
np.random.seed(7)

sim_weights = np.random.dirichlet(np.ones(n_assets), size=N_SIMULATIONS)
sim_weights = np.clip(sim_weights, 0, MAX_WEIGHT_PER_ASSET)
sim_weights = sim_weights / sim_weights.sum(axis=1, keepdims=True)

sim_returns = sim_weights @ mean_excess_annual.values
sim_vols = np.sqrt(np.einsum("ij,jk,ik->i", sim_weights, cov_matrix_annual.values, sim_weights))
sim_sharpes = sim_returns / sim_vols

min_vol_idx = np.argmin(sim_vols)

import matplotlib.pyplot as plt

BG = "#0d1117"
FG = "#c9d1d9"
GRID = "#30363d"
ACCENT = "#39d353"
HIGHLIGHT = "#f0b90b"

fig, ax = plt.subplots(figsize=(11, 7), facecolor=BG)
ax.set_facecolor(BG)

scatter = ax.scatter(sim_vols, sim_returns, c=sim_sharpes, cmap="viridis",
                      s=8, alpha=0.6, edgecolors="none")

ax.scatter(optimal_port_std, optimal_port_return, c=HIGHLIGHT, marker="*",
           s=500, edgecolors="white", linewidths=1, label="Max Sharpe (SLSQP)", zorder=5)

ax.scatter(sim_vols[min_vol_idx], sim_returns[min_vol_idx], c=ACCENT, marker="D",
           s=120, edgecolors="white", linewidths=1, label="Min Volatility (simulasi)", zorder=5)

cbar = fig.colorbar(scatter, ax=ax)
cbar.set_label("Sharpe Ratio", color=FG)
cbar.ax.yaxis.set_tick_params(color=FG)
plt.setp(plt.getp(cbar.ax.axes, "yticklabels"), color=FG)

ax.set_xlabel("Annualized Volatility", color=FG, fontsize=11)
ax.set_ylabel("Annualized Excess Return", color=FG, fontsize=11)
ax.set_title(f"Efficient Frontier -- {N_SIMULATIONS:,} Simulasi Portofolio Random\n"
             f"Kandidat: {', '.join(CANDIDATE_TICKERS)}", color=FG, fontsize=13, pad=15)

ax.tick_params(colors=FG)
ax.grid(True, color=GRID, linewidth=0.5, alpha=0.7)
for spine in ax.spines.values():
    spine.set_color(GRID)

legend = ax.legend(facecolor=BG, edgecolor=GRID, labelcolor=FG, loc="best")

ax.xaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))
ax.yaxis.set_major_formatter(plt.FuncFormatter(lambda x, _: f"{x:.0%}"))

plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150, facecolor=BG)
print("\nGrafik efficient frontier disimpan ke efficient_frontier.png")

plt.tight_layout()
plt.savefig("efficient_frontier.png", dpi=150, facecolor=BG)
print("\nGrafik efficient frontier disimpan ke efficient_frontier.png")
print("Menampilkan pop-up window grafik...")
plt.show()