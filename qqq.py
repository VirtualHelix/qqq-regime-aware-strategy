import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Optional: silence chained assignment warnings globally (we also fix root cause with .copy()).
pd.options.mode.chained_assignment = None

# ==========================================
# 1. LOAD AND CLEAN DATA
# ==========================================
def load_data():
    # Load QQQ
    df_q1 = pd.read_csv("QQQ_Train_Validate.csv")
    df_q2 = pd.read_csv("QQQ_Blind_Test.csv")
    if "Downloaded" in str(df_q2.iloc[-1, 0]):
        df_q2 = df_q2[:-1]

    df_qqq = pd.concat([df_q1, df_q2], ignore_index=True)

    # Parse dates (explicit inference to avoid warnings / inconsistent parsing)
    df_qqq["Date"] = pd.to_datetime(df_qqq["Time"], infer_datetime_format=True, errors="coerce")
    df_qqq = df_qqq.dropna(subset=["Date"])
    df_qqq = df_qqq.set_index("Date").sort_index()

    # Handle duplicates by keeping the first occurrence
    df_qqq = df_qqq[~df_qqq.index.duplicated(keep="first")]

    # Rename for standard usage
    df_qqq = df_qqq.rename(
        columns={
            "Latest": "Close",
            "Volume": "Vol",
            "Open": "Open",
            "High": "High",
            "Low": "Low",
        }
    )
    df_qqq = df_qqq[["Open", "High", "Low", "Close", "Vol"]].copy()

    # Ensure numerics
    for c in ["Open", "High", "Low", "Close", "Vol"]:
        df_qqq[c] = pd.to_numeric(df_qqq[c], errors="coerce")

    # Load VIX
    raw = pd.read_csv("VIX.csv")
    raw_cols = raw.iloc[0].tolist()  # header row inside file
    df_vix = raw.iloc[1:].copy()
    df_vix.columns = raw_cols

    df_vix["Date"] = pd.to_datetime(df_vix["Time"], infer_datetime_format=True, errors="coerce")
    df_vix = df_vix.dropna(subset=["Date"])
    df_vix = df_vix.set_index("Date").sort_index()

    df_vix["VIX"] = pd.to_numeric(df_vix["Latest"], errors="coerce")
    df_vix = df_vix[["VIX"]].copy()

    # Load Rates (DGS20)
    df_rates = pd.read_csv("DGS20.csv")
    df_rates["Date"] = pd.to_datetime(df_rates["observation_date"], errors="coerce")
    df_rates = df_rates.dropna(subset=["Date"])
    df_rates = df_rates.set_index("Date").sort_index()

    df_rates["Yield"] = pd.to_numeric(df_rates["DGS20"], errors="coerce")
    df_rates = df_rates[["Yield"]].copy()

    # Merge Data (join everything to the QQQ calendar)
    df = df_qqq.join(df_vix, how="left")

    # Rates: forward fill aligned to QQQ calendar (signal freezes after last known value)
    df_rates_aligned = df_rates.reindex(df.index, method="ffill")
    df["Yield"] = df_rates_aligned["Yield"].ffill()

    # Forward fill VIX for any small gaps
    df["VIX"] = df["VIX"].ffill()

    # Final clean
    df = df.dropna().copy()
    return df


# ==========================================
# 2. SIGNAL GENERATION (At Close T)
# ==========================================
def calculate_signals(df):
    # --- Market State (Calculated at Close T) ---
    qqq_ret = df["Close"].pct_change()

    # 1. Rates Signal
    rate_trend = df["Yield"].diff(20)
    roll_corr = qqq_ret.rolling(60).corr(df["Yield"].diff()).fillna(0)
    # If rates rising fast AND correlation isn't negative (hedge), Risk Off.
    s_rates = np.where((rate_trend > 0.1) & (roll_corr > -0.2), -1, 1)

    # 2. Trend & Volatility Regime
    sma200 = df["Close"].rolling(200).mean()
    is_bull = df["Close"] > sma200

    s_trend = np.where(is_bull, 1, -1)
    s_grind = np.where((~is_bull) & (df["VIX"] < 25), -1, 0)  # bear + low vol = bleed
    s_vix = np.where(
        df["VIX"] < 20,
        1,
        np.where(df["VIX"] > 35, 1, -1),  # long calm or panic, short transition
    )

    # 3. Momentum & Serial
    s_mom = np.where(df["Close"] > df["Close"].shift(20), 1, -1)
    s_serial = np.where(
        qqq_ret.rolling(20).corr(qqq_ret.shift(1)).fillna(0) > 0, 1, 0
    )

    # --- Core Direction Score ---
    score = s_trend + s_rates + s_vix + s_mom + s_serial + (2 * s_grind)
    core_dir = np.select([score >= 2, score <= -2], [1, -1], default=0)

    # --- Position Sizing ---
    realized_vol = qqq_ret.rolling(20).std() * np.sqrt(252)
    target_vol = np.where(is_bull, 0.25, 0.15)  # target 25% bull, 15% bear
    max_lev = np.where(is_bull, 3.0, 1.0)  # max 3x bull, 1x bear

    vol_scalar = (target_vol / (realized_vol + 0.001)).clip(0.5, max_lev)

    # Iron Dome: If VIX > 40, cut size in half regardless of other signals
    risk_off = np.where(df["VIX"] > 40, 0.5, 1.0)

    # OVERNIGHT LEVERAGE (Determined at Close T)
    df["Lev_Overnight"] = core_dir * vol_scalar * risk_off

    # --- PREPARE FOR INTRADAY (T+1) ---
    # Gap from Close T to Open T+1 (used for intraday position held Open->Close T+1)
    gap = (df["Open"].shift(-1) / df["Close"]) - 1
    gap_mag = gap.abs()
    gap_sign = np.sign(gap)

    # Fade small gaps (<1%), follow large gaps (>1%)
    intra_sig = np.where(gap_mag < 0.01, -1 * gap_sign, 1 * gap_sign)

    # Intraday regime scaling
    k_intra = np.where(is_bull, 0.5, 0.3)

    # Blend Gap Signal (80%) with Core Trend (20%)
    raw_intra_lev = (intra_sig * k_intra * 0.8) + (core_dir * 0.2)
    df["Lev_Intraday"] = raw_intra_lev * risk_off

    return df


# ==========================================
# 3. BACKTEST ENGINE
# ==========================================
def run_backtest(df):
    # Returns for the NEXT period (T+1) aligned with signals at T

    # Return from Close T to Open T+1
    df["Ret_Overnight"] = df["Open"].shift(-1) / df["Close"] - 1

    # Return from Open T+1 to Close T+1
    df["Ret_Intraday"] = df["Close"].shift(-1) / df["Open"].shift(-1) - 1

    # Benchmark Return (Close T to Close T+1)
    df["Ret_Bench"] = df["Close"].shift(-1) / df["Close"] - 1

    # Drop last row (no T+1 data) and force a real copy to avoid chained assignment warnings
    df = df.dropna().copy()

    # Strategy daily return
    strat_overnight = df["Lev_Overnight"] * df["Ret_Overnight"]
    strat_intraday = df["Lev_Intraday"] * df["Ret_Intraday"]
    df["Strat_Daily_Ret"] = ((1 + strat_overnight) * (1 + strat_intraday)) - 1

    # Equity curves
    df["Equity_Curve"] = (1 + df["Strat_Daily_Ret"]).cumprod()
    df["Bench_Curve"] = (1 + df["Ret_Bench"]).cumprod()

    # Drawdowns (store for plotting/analysis)
    df["Strat_RollMax"] = df["Equity_Curve"].cummax()
    df["Bench_RollMax"] = df["Bench_Curve"].cummax()
    df["Strat_Drawdown"] = (df["Equity_Curve"] / df["Strat_RollMax"]) - 1
    df["Bench_Drawdown"] = (df["Bench_Curve"] / df["Bench_RollMax"]) - 1

    return df


# ==========================================
# 4. METRICS
# ==========================================
def calculate_metrics(equity_series):
    if equity_series.empty:
        return 0, 0, 0, 0

    total_ret = (equity_series.iloc[-1] / equity_series.iloc[0]) - 1
    days = (equity_series.index[-1] - equity_series.index[0]).days
    years = days / 365.25 if days > 0 else 0

    cagr = (1 + total_ret) ** (1 / years) - 1 if years > 0 else 0

    daily_rets = equity_series.pct_change().dropna()
    sharpe = (
        (daily_rets.mean() * 252) / (daily_rets.std() * np.sqrt(252))
        if daily_rets.std() != 0
        else 0
    )

    rolling_max = equity_series.cummax()
    drawdown = (equity_series - rolling_max) / rolling_max
    max_dd = drawdown.min()

    calmar = cagr / abs(max_dd) if max_dd != 0 else 0
    return cagr, sharpe, max_dd, calmar


# ==========================================
# 5. EXECUTION
# ==========================================
df = load_data()
df = calculate_signals(df)
df = run_backtest(df)

# Filter for Test Period
test_start_date = "2022-01-01"
df_test = df.loc[test_start_date:].copy()

# Metrics
s_cagr, s_sharpe, s_dd, s_calmar = calculate_metrics(df_test["Equity_Curve"])
b_cagr, b_sharpe, b_dd, b_calmar = calculate_metrics(df_test["Bench_Curve"])

# Output
print(f"\nBACKTEST RESULTS: {test_start_date} to {df_test.index[-1].date()}")
print("-" * 55)
print(f"{'Metric':<20} {'Strategy':<15} {'Benchmark (QQQ)':<15}")
print("-" * 55)
print(f"{'CAGR':<20} {s_cagr*100:.2f}%          {b_cagr*100:.2f}%")
print(f"{'Sharpe Ratio':<20} {s_sharpe:.2f}           {b_sharpe:.2f}")
print(f"{'Calmar Ratio':<20} {s_calmar:.2f}           {b_calmar:.2f}")
print(f"{'Max Drawdown':<20} {s_dd*100:.2f}%         {b_dd*100:.2f}%")
print("-" * 55)

# ==========================================
# 6. PLOTTING (SAVE + SHOW)
# ==========================================
os.makedirs("images", exist_ok=True)

# Normalize curves to 1.0 at start of test period
eq_s = df_test["Equity_Curve"] / df_test["Equity_Curve"].iloc[0]
eq_b = df_test["Bench_Curve"] / df_test["Bench_Curve"].iloc[0]

# 6.1 Equity curve (log scale)
plt.figure(figsize=(12, 6))
plt.plot(df_test.index, eq_s, label="Strategy", linewidth=1.5)
plt.plot(df_test.index, eq_b, label="QQQ", linewidth=1.5, alpha=0.8)
plt.title(f"QQQ Strategy Test Period (Sharpe: {s_sharpe:.2f} vs {b_sharpe:.2f})")
plt.ylabel("Normalized Equity (Log Scale)")
plt.yscale("log")
plt.legend()
plt.grid(True, which="both", alpha=0.3)
plt.tight_layout()
plt.savefig("images/equity_curve.png", dpi=200)
plt.show()

# 6.2 Drawdown curve
plt.figure(figsize=(12, 4))
plt.plot(df_test.index, df_test["Strat_Drawdown"], label="Strategy Drawdown", linewidth=1.5)
plt.plot(df_test.index, df_test["Bench_Drawdown"], label="QQQ Drawdown", linewidth=1.5, alpha=0.8)
plt.title("Drawdown Profile (Test)")
plt.ylabel("Drawdown")
plt.xlabel("Date")
plt.axhline(0, linewidth=1)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig("images/drawdown.png", dpi=200)
plt.show()
