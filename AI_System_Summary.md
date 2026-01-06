# Hedge Fund AI Agent - System Summary

## Can it really replace a human trader?
**After several days of training**, this AI is designed to reach a "Junior Trader" level. It will not have human intuition (news events, tweets), but it will have **superhuman discipline and mathematical precision**.

## What will it learn?
The **ResNet Architecture** (Deep Residual Network) allows the model to "see" charts like an image.

1.  **Regime Detection**: It will likely learn to distinguish between:
    -   **Trending**: Buying breakout pairs (e.g., EURUSD going up).
    -   **Mean Reversion**: Fading spikes in range-bound markets (e.g., AUDUSD at night).
    -   **Crash Avoidance**: Going "Neutral" (Weight 0.0) when volatility (ATR) spikes dangerously high.

2.  **Portfolio Hedging**:
    -   Since it trades 8 pairs simultaneously, it learns **Correlations**.
    -   *Example*: If it buys EURUSD, it might sell USDCHF (since they often move oppositely), effectively doubling its bet on a weak Dollar while hedging specific Euro risk.

3.  **Risk Management (The "Hedge Fund" Part)**:
    -   It optimizes the **Sortino Ratio** (Rewards vs Downside Risk).
    -   It will learn that taking huge losses destroys its reward, so it naturally learns to cut losers quickly.

## Technical Architecture

### 1. The Brain (Custom Policy)
-   **Type**: `ResNetFeatureExtractor`
-   **Depth**: 6 Residual Blocks (Deep Learning).
-   **Vision**: It looks at 30 candles of history across 15m, 1h, and 1d timeframes simultaneously.
-   **Input**: Price, Volume, RSI, MACD, Bollinger Bands, ATR.

### 2. The Curriculum (Training Plan)
-   **Phase 1 (University)**: Trained on **Hourly (1H)** data. Fast, noisy, general concepts.
-   **Phase 2 (Specialization)**: Fine-tuned on **15-Minute (15M)** data. Learning detailed entry/exit timing.

## How to measure "Success"?
Don't just look at Profit ($). Look at:
-   **Sharpe Ratio**: Is the return stable? (> 1.0 is good, > 2.0 is excellent).
-   **Max Drawdown**: How much did it lose at its worst point? (< 10% is the goal).

*Use the provided `evaluate_model.py` script to generate these metrics after training.*
