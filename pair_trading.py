import numpy as np
import pandas as pd
import yfinance as yf
import statsmodels.api as sm
from statsmodels.tsa.stattools import adfuller
import matplotlib.pyplot as plt


def fetch_pair_data(ticker_a: str, ticker_b: str, start_date: str, end_date: str) -> pd.DataFrame:
    """Download historical adjusted closing prices for two tickers."""
    print(f"Fetching data for {ticker_a} and {ticker_b}...")
    df = yf.download([ticker_a, ticker_b], start=start_date, end=end_date)["Adj Close"]
    df = df.dropna()
    df.columns = ["Asset_A", "Asset_B"]
    return df


def calculate_hedge_ratio(series_a: pd.Series, series_b: pd.Series) -> float:
    """
    Calculate Hedge Ratio using OLS Regression: Asset_A = Hedge_Ratio * Asset_B + Intercept
    """
    X = sm.add_constant(series_b)
    model = sm.OLS(series_a, X).fit()
    hedge_ratio = model.params.iloc[1]
    return hedge_ratio


def test_stationarity(spread: pd.Series) -> dict:
    """Perform Augmented Dickey-Fuller (ADF) test on the spread series."""
    adf_result = adfuller(spread.dropna(), autolag="AIC")
    p_value = adf_result[1]
    is_stationary = p_value < 0.05
    
    return {
        "adf_statistic": adf_result[0],
        "p_value": p_value,
        "is_stationary": is_stationary,
        "critical_values": adf_result[4]
    }


def generate_trading_signals(
    df: pd.DataFrame, 
    hedge_ratio: float, 
    lookback: int = 30, 
    entry_z: float = 2.0, 
    exit_z: float = 0.5
) -> pd.DataFrame:
    """
    Compute spread, rolling Z-score, positions, and track mean reversion status.
    """
    data = df.copy()
    
    # 1. Compute Spread
    data["Spread"] = data["Asset_A"] - (hedge_ratio * data["Asset_B"])
    
    # 2. Rolling Spread Statistics & Z-Score
    data["Spread_Mean"] = data["Spread"].rolling(window=lookback).mean()
    data["Spread_Std"] = data["Spread"].rolling(window=lookback).std()
    data["Z_Score"] = (data["Spread"] - data["Spread_Mean"]) / data["Spread_Std"]
    
    # 3. Position Logic (-1: Short Spread, +1: Long Spread, 0: Neutral)
    data["Position"] = 0
    data["Mean_Reversion_Status"] = "Neutral"
    
    current_position = 0
    
    positions = []
    statuses = []
    
    for z in data["Z_Score"]:
        if np.isnan(z):
            positions.append(0)
            statuses.append("Insufficient Data")
            continue
            
        if current_position == 0:
            if z > entry_z:
                current_position = -1  # Short Spread: Sell A, Buy B
            elif z < -entry_z:
                current_position = 1   # Long Spread: Buy A, Sell B
        elif current_position != 0:
            # Check for Mean Reversion (Exit signal)
            if abs(z) < exit_z:
                current_position = 0
                
        positions.append(current_position)
        
        # Status Labeling
        if current_position == 1:
            statuses.append("Long Spread (Under-valued)")
        elif current_position == -1:
            statuses.append("Short Spread (Over-valued)")
        else:
            statuses.append("Mean Reverted / Neutral")
            
    data["Position"] = positions
    data["Mean_Reversion_Status"] = statuses
    
    return data


def plot_pair_trading(data: pd.DataFrame, ticker_a: str, ticker_b: str, entry_z: float = 2.0, exit_z: float = 0.5):
    """Plot Z-score and trading signals."""
    plt.figure(figsize=(14, 7))
    plt.plot(data.index, data["Z_Score"], label="Z-Score", color="blue", alpha=0.7)
    
    # Threshold lines
    plt.axhline(entry_z, color="red", linestyle="--", label=f"Short Threshold (+{entry_z})")
    plt.axhline(-entry_z, color="green", linestyle="--", label=f"Long Threshold (-{entry_z})")
    plt.axhline(exit_z, color="black", linestyle=":", label=f"Exit Bounds (+/-{exit_z})")
    plt.axhline(-exit_z, color="black", linestyle=":")
    plt.axhline(0, color="gray", alpha=0.5)
    
    plt.title(f"Pair Trading Strategy Z-Score & Signals ({ticker_a} / {ticker_b})")
    plt.xlabel("Date")
    plt.ylabel("Z-Score")
    plt.legend(loc="upper left")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.show()


# ==========================================
# Execution Pipeline
# ==========================================
if __name__ == "__main__":
    # Define Tickers and Date Range (e.g., Chevron & ExxonMobil)
    TICKER_A = "CVX"
    TICKER_B = "XOM"
    START_DATE = "2023-01-01"
    END_DATE = "2026-01-01"
    LOOKBACK_WINDOW = 30
    
    # 1. Fetch Data
    df_prices = fetch_pair_data(TICKER_A, TICKER_B, START_DATE, END_DATE)
    
    # 2. Calculate Hedge Ratio
    hedge_ratio = calculate_hedge_ratio(df_prices["Asset_A"], df_prices["Asset_B"])
    print(f"\nCalculated Hedge Ratio (Asset_A / Asset_B): {hedge_ratio:.4f}")
    
    # 3. Test Cointegration / Stationarity via ADF
    temp_spread = df_prices["Asset_A"] - (hedge_ratio * df_prices["Asset_B"])
    adf_results = test_stationarity(temp_spread)
    
    print("\n--- Augmented Dickey-Fuller (ADF) Test ---")
    print(f"ADF Statistic : {adf_results['adf_statistic']:.4f}")
    print(f"p-value       : {adf_results['p_value']:.4f}")
    print(f"Is Stationary : {adf_results['is_stationary']} (at 5% significance)")
    
    # 4. Generate Signals and Dataframe
    strategy_df = generate_trading_signals(
        df_prices, 
        hedge_ratio=hedge_ratio, 
        lookback=LOOKBACK_WINDOW, 
        entry_z=2.0, 
        exit_z=0.5
    )
    
    # 5. Display Structured Summary Table
    output_cols = ["Asset_A", "Asset_B", "Spread", "Z_Score", "Position", "Mean_Reversion_Status"]
    print("\n--- Recent Strategy Output (Last 10 Rows) ---")
    print(strategy_df[output_cols].tail(10).round(4).to_string())
    
    # 6. Plot Results
    plot_pair_trading(strategy_df, TICKER_A, TICKER_B)