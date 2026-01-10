# Signal-Gated Market Making: Exploiting Intraday Drift via Deep Neuroevolution

This framework transforms intraday volatility and trend dynamics into execution signals via a Deep Reinforcement Learning (DRL) agent. It exploits standalone predictive units to calibrate optimal bid/ask offsets, managing inventory risk while providing liquidity in the A-share market.

**Target**

- Regime-Conditional Signal Extraction: Extract high-fidelity signals using standalone units (XGBoost for price range and LSTM for trend) to gate execution decisions.
- Microstructure-Aware Modeling: Implement a continuous, tick-based action space to optimize bid/ask offsets relative to best-book prices, accounting for price-time priority.
- Inventory-Risk Management: Utilize an absolute value inventory penalty ($-\lambda|I_{t+1}|$) within the reward function to prioritize spread capturing over directional exposure.
- Robustness via Adversarial RL: Incorporate a "clairvoyant" adversary agent to strategically perturb quotes, enhancing policy generalization in non-stationary market regimes.

**Data**

A-share 1-minute OHLCV data for liquid constituents (e.g., CSI 300), leveraging multiple years of historical data to satisfy the sample density requirements for neuroevolutionary training.

**Strategy Pipeline**

1. **SGU Training**: Develop standalone XGBoost (Volatility) and LSTM (Trend) models to generate the 3D state space $[I, RR, TR]$.
2. **Environment Setup**: Construct an MDP environment utilizing a First-Passage Time (FPT) execution model for realistic limit order matching.
3. **Policy Evolution**: Train deep neural networks using population-based Genetic Algorithms (Neuroevolution) to map intraday states directly to bid/ask offsets.
4. **Adversarial Co-training**: Evolve an adversary agent to stress-test the market maker's robustness against strategic quote displacement and model uncertainty.

## Replication Timeline

| Weeks | Lit. Review | Feature Eng. | Vec Backtest | Metric Bldg. | Robustness | OOS Testing |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| Week 1 | ✓ | ✓ | | | | |
| Week 2 | | ✓ | ✓ | | | |
| Week 3 | | | ✓ | ✓ | | |
| Week 4 | | | | ✓ | ✓ | |
| Week 5 | | | | | ✓ | ✓ |