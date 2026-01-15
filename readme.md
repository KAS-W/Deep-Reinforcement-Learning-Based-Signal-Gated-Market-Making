# Signal-Gated Market Making: Exploiting Intraday Drift via Deep Neuroevolution

This framework transforms intraday volatility and trend dynamics into execution signals via a Deep Reinforcement Learning (DRL) agent. It exploits standalone predictive units to calibrate optimal bid/ask offsets, managing inventory risk while providing liquidity in the A-share market.

**Target**

- Regime-Conditional Signal Extraction: Extract high-fidelity signals using standalone units (XGBoost for price range and LSTM for trend) to gate execution decisions.
- Microstructure-Aware Modeling: Implement a continuous, tick-based action space to optimize bid/ask offsets relative to best-book prices, accounting for price-time priority.
- Inventory-Risk Management: Utilize an absolute value inventory penalty ($-\lambda|I_{t+1}|$) within the reward function to prioritize spread capturing over directional exposure.
- Robustness via Adversarial RL: Incorporate a "clairvoyant" adversary agent to strategically perturb quotes, enhancing policy generalization in non-stationary market regimes.

**Data**

A-share Limit Order Book (LOB) data for liquid constituents: 300ETF (Tushare code `510300.SH`) from `Tushare`. Leveraging multiple years of historical data to satisfy the sample density requirements for neuroevolutionary training.

Period: 20240401-20240630

**Strategy Pipeline**

1. **SGU Training**: Develop standalone XGBoost (Volatility) and LSTM (Trend) models to generate the 3D state space $[I, RR, TR]$.
2. **Environment Setup**: Construct an MDP environment utilizing a First-Passage Time (FPT) execution model for realistic limit order matching.
3. **Policy Evolution**: Train deep neural networks using population-based Genetic Algorithms (Neuroevolution) to map intraday states directly to bid/ask offsets.
4. **Adversarial Co-training**: Evolve an adversary agent to stress-test the market maker's robustness against strategic quote displacement and model uncertainty.

<!-- **SGU1 Features** -->

<!-- SGU1 (Signal Generation Unit 1) is designed to predict the future price realized range $RR_{t,k}$ using an XGBoost model. It incorporates 23 initial features derived from market microstructure and domain knowledge to capture various dimensions of price action and liquidity.

- **Log Returns**: Capture price momentum across multiple time horizons.
  $$r_{t,k} = \ln(P_t) - \ln(P_{t-k}), \quad k \in \{1, 5, 10, 30, 60\}$$

- **Historical Realized Volatility**: Measure the intensity of price fluctuations.
  $$\sigma_{t,k} = \sqrt{\frac{1}{k} \sum_{i=0}^{k-1} (r_{t-i, 1} - \bar{r})^2}, \quad k \in \{5, 10, 30, 60\}$$

- **Relative Price Position ($LP$)**: Determine the current price location relative to a historical window to identify overbought or oversold conditions.
  $$LP_{t,k} = \frac{P_t - \min(P_{t-k:t})}{\max(P_{t-k:t}) - \min(P_{t-k:t})}, \quad k \in \{5, 10, 30, 60\}$$

- **Number of Trades**: Represent market activity levels by counting total transactions over previous $p$ periods.
  $$N_{trades, p}, \quad p \in \{1, 2, 3, 5, 10\}$$

- **Bid-Ask Spread**: A proxy for market liquidity and friction costs at the start of the time-step.
  $$Spread_t = P_{ask,t} - P_{bid,t}$$

- **Traded Volume Imbalance**: Evaluate the relative strength of aggressive buying versus selling.
  $$V_{imb} = \frac{V_{buy} - V_{sell}}{V_{buy} + V_{sell}}$$

- **Volume-Weighted Average Price (VWAP)**: Indicate the average price weighted by volume over $r$ periods.
  $$VWAP_{t,r} = \frac{\sum_{i=0}^{r-1} P_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{r-1} V_{t-i}}, \quad r \in \{1, 3, 5\}$$

- **Price Slope**: The slope of a linear fit (OLS) of price versus time over $s$ periods to detect short-term trends.
  $$\text{Slope}_{t,s}, \quad s \in \{1, 3, 5\}$$

- **Time of Day**: Captured in hours to account for cyclical intraday patterns like opening and closing volatility.

- **Order Flow Statistics**: Includes total traded volume, percentage of upticks, and the count of large buys/sells. Note that experimental results suggested removing upticks and large order counts to improve gain.

- **Time-Lagged Labels**: The previous modified realized price ranges ($RR$) used to capture volatility clustering.
  $$RR_{t-L}, \quad L \in \{1, 2, 3, 4, 5\}$$ -->

**SGU1 Features**

SGU1 (Signal Generation Unit 1) is designed to predict the future modified realized price range $y_i$ using an XGBoost model. It incorporates 23 initial features derived from market microstructure and domain knowledge to capture various dimensions of price action and liquidity.

- **Number of Trades**: Represent market activity levels by counting total transactions over previous $p$ periods.
  $$N_{trades, p}, \quad p \in \{1, 2, 3, 5, 10\}$$

- **Bid-Ask Spread**: A proxy for market liquidity and friction costs at the start of the time-step.
  $$Spread_t = P_{ask,t} - P_{bid,t}$$

- **Traded Volume Imbalance**: Evaluate the relative strength of aggressive buying versus selling.
  $$V_{imb} = \frac{V_{buy} - V_{sell}}{V_{buy} + V_{sell}}$$

- **Volume-Weighted Average Price (VWAP)**: Indicate the average price weighted by volume over $r$ periods.
  $$VWAP_{t,r} = \frac{\sum_{i=0}^{r-1} P_{t-i} \cdot V_{t-i}}{\sum_{i=0}^{r-1} V_{t-i}}, \quad r \in \{1, 3, 5\}$$

- **Price Slope**: The slope of a linear fit (OLS) of price versus time over $s$ periods to detect short-term trends.
  $$\text{Slope}_{t,s}, \quad s \in \{1, 3, 5\}$$

- **Time of Day**: Captured in hours to account for cyclical intraday patterns like opening and closing volatility.

- **Total Traded Volume**: The cumulative amount of the security traded during the most recent period.

- **Order Flow Statistics**: Includes the percentage of upticks, the total number of large buys, and the total number of large sells. Note that experimental results suggested removing upticks and large order counts to improve gain.

- **Time-Lagged Labels**: The previous modified realized price ranges ($RR$) used to capture volatility clustering.
  $$RR_{t-L}, \quad L \in \{1, 2, 3, 4, 5\}$$

## Replication Timeline

| Weeks | Lit. Review | Feature Eng. | Vec Backtest | Metric Bldg. | Robustness | OOS Testing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Week 1 | ✓ | ✓ | | | | |
| Week 2 | | ✓ | ✓ | | | |
| Week 3 | | | ✓ | ✓ | | |
| Week 4 | | | | ✓ | ✓ | |
| Week 5 | | | | | ✓ | ✓ |