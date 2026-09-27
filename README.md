Pair trading is a market-neutral statistical arbitrage strategy that profits from price divergence between two historically correlated assets, such as Chevron and ExxonMobil. Instead of betting on general market direction, the strategy tracks the relative pricing relationship (the spread) between the pair. By calculating a dynamic hedge ratio via Ordinary Least Squares (OLS) regression, you determine the precise balance required to construct a neutral spread where the assets move together over time.

To ensure the strategy remains mathematically viable, the system subjects the spread to an Augmented Dickey-Fuller (ADF) stationarity test. A stationary spread means the relationship exhibits mean-reverting behavior rather than trending indefinitely. Once stationarity is confirmed, a rolling Z-score normalizes current spread fluctuations against historical averages to trigger execution:

1.ADF Test Check:Validates that the spread is stationary (p value <=0.05) so mean reversion can be reliably exploited.

2.Long Spread Signal (Z<-2.0): Triggers when Asset A becomes significantly undervalued relative to Asset B, signaling a buy on A and a sell on B.

3.Short Spread Signal (Z> +2.0): Triggers when Asset A becomes overvalued relative to Asset B, signaling a sell on A and a buy on B.

4.Mean Reversion Exit (mod(Z)< 0.5): Closes open positions once the Z-score returns to near zero, locking in profits as prices re-align.