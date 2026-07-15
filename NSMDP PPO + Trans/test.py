import yfinance as yf
import pandas as pd
import numpy as np

SECTOR_TICKERS = {
    "XLB":  "Materials",
    "XLI":  "Industrials",
    "XLY":  "Consumer Discretionary",
    "XLP":  "Consumer Staples",
    "XLV":  "Health Care",
    "XLF":  "Financials",
    "XLK":  "Information Technology",
    "XLC":  "Communication Services",
    "XLU":  "Utilities",
    "XLRE": "Real Estate",
    "XLE":  "Energy",
    "^VIX": "Volatility Index",
}

START_CASH = 100_000
T = 60  # lookback period

tickers = list(SECTOR_TICKERS.keys())

raw = yf.download(tickers, start="2006-01-01", end="2021-12-31", auto_adjust=True)
prices = raw["Close"]
prices.columns = [SECTOR_TICKERS[t] for t in prices.columns]

complete = prices.columns[prices.notna().all()]
dropped = prices.columns.difference(complete).tolist()
prices = prices[complete]
if dropped:
    print(f"Dropped (incomplete history): {dropped}")

# Separate VIX from sector prices
vix = prices["Volatility Index"]
sector_prices = prices.drop(columns=["Volatility Index"])

log_returns = np.log(sector_prices / sector_prices.shift(1)).dropna()
returns = sector_prices.pct_change().dropna()

# vol20 and vol60: rolling std of S&P 500 log returns (paper Section 5.1)
sp500 = yf.download("^GSPC", start="2006-01-01", end="2021-12-31", auto_adjust=True)["Close"].squeeze()
sp500_log_returns = np.log(sp500 / sp500.shift(1)).dropna()
vol20 = sp500_log_returns.rolling(20).std().rename("vol20")
vol60 = sp500_log_returns.rolling(60).std().rename("vol60")

# ── State matrix S_t (paper Section 4.2) ──────────────────────────────────
# Shape: (n+1) x T,  n = sectors, T = 60
# Row i   : [w_i,  r_{i,t-1}, r_{i,t-2}, ..., r_{i,t-T+1}]
# Cash row: [w_c,  vol20_t,   vol20/vol60_t,  VIX_t, 0, ...]

n = len(sector_prices.columns)
sectors = sector_prices.columns.tolist()

# Initial portfolio: all cash
weights = np.zeros(n + 1)
weights[-1] = 1.0  # w_c = 1

t = log_returns.index[-1]
recent_log_returns = log_returns.loc[:t].iloc[-(T - 1):]  # (T-1) x n

state = np.zeros((n + 1, T))

for i, sector in enumerate(sectors):
    state[i, 0] = weights[i]
    state[i, 1:] = recent_log_returns[sector].values

state[n, 0] = weights[-1]
state[n, 1] = vol20.loc[t]
state[n, 2] = vol20.loc[t] / vol60.loc[t]
state[n, 3] = vix.loc[t]

col_labels = ["w"] + [f"t-{i}" for i in range(1, T)]
short_names = {
    "Materials": "Matl", "Industrials": "Indu",
    "Consumer Discretionary": "ConDisc", "Consumer Staples": "ConStp",
    "Health Care": "Hlth", "Financials": "Fin",
    "Information Technology": "Tech", "Utilities": "Util",
}
row_labels = [short_names.get(s, s) for s in sectors] + ["Cash"]
state_df = pd.DataFrame(state.round(5), index=row_labels, columns=col_labels)

print(f"Start cash: ${START_CASH:,}")
print(f"State matrix S_t at {t.date()}  shape: {state_df.shape}  (rows=assets+cash, cols=weight+59 log-returns)")
print(f"  Cash row cols:  w | vol20 | vol20/vol60 | VIX | 0 ...\n")
with pd.option_context("display.max_columns", 12, "display.width", 110,
                       "display.float_format", "{:>9.5f}".format):
    print(state_df)
