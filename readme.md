# Drift-Regimes on Boosting Cross-sectional Predictability

This framework transforms intraday dynamics into daily signals activated only during stock-specific drift regimes. It exploits microstructure shifts to enable robust, regime-conditional cross-sectional positioning at the market open.

**Target**

- Regime-Conditional Signal Extraction: Extract high-fidelity signals from intraday and daily data, activated exclusively during stock-specific drift regimes to unlock hidden predictability.
- Microstructure-Aware Modeling: Model time-varying dynamics and liquidity shifts within drift periods to filter noise and capture alpha driven by behavioral biases.
- Cross-Sectional Integration: Bridge microstructure dynamics with daily horizons by integrating high-frequency features into a regime-gated, market-neutral framework.
- Institutional Scalability Analysis: Evaluate strategy viability by modeling market impact and capacity constraints (targeting $100M–$500M AUM).

**Data**

High-frequency Trade and Quote (TAQ) and daily data for the S&P 500 constituents, ensuring results are based on the most liquid and economically significant segment of the US equity market.

**Strategy Pipeline**

## Replication Timeline
$$
\begin{array}{|c|c|c|c|c|c|c|}
| Weeks | Lit. Review | Feature Eng. | Vec Backtest | Metric Bldg. | Robustness | OOS Testing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Week 1 | ✓ | ✓ | | | | |
| Week 2 | | ✓ | ✓ | | | |
| Week 3 | | | ✓ | ✓ | | |
| Week 4 | | | | ✓ | ✓ | |
| Week 5 | | | | | ✓ | ✓ |