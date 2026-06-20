# Forex Regression Model — Automated EUR/USD Trading Bot

A quantitative trading bot that uses a **LightGBM regression model** to predict short-term pip movements on **EUR/USD**, then autonomously executes trades on **OANDA** with dynamic stop-loss and take-profit management.

---

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Configuration](#configuration)
- [Usage](#usage)
- [Model Details](#model-details)
- [Risk Management](#risk-management)
- [Logging](#logging)
- [Contributing](#contributing)
- [Disclaimer](#disclaimer)

---

## Overview

The bot operates on a **5-minute timeframe** and combines multi-timeframe analysis (5M, 1H, 4H) with a trained regression model to forecast expected pip movements. When the predicted move exceeds a configurable threshold, the bot places a **MARKET order** with ATR-based stop-loss and prediction-based take-profit via the OANDA v20 REST API.

### How It Works

```
┌─────────────────┐           ┌──────────────────┐     ┌────────────────────┐
│  Fetch Candles  │───────────▶  Engineer 25     ────▶│  LightGBM Model    │
│  (5M, 1H, 4H)   │           │  Features        │     │  Predicts Pips     │
└─────────────────┘           └──────────────────┘     └───────────┬────────┘
                                                                   │
                        ┌──────────────────┐                       │
                        │  Execute Trade   │◀──────────────────────┘
                        │  on OANDA        │   (if |pips| >= threshold)
                        └──────────────────┘
```

---

## Architecture

The bot follows a **single-class monolith** design centered around `OandaQuantBot`:

| Layer               | Responsibility                                                  |
|---------------------|-----------------------------------------------------------------|
| **News Calendar**   | Fetches high-impact economic events from Finnhub / Forex Factory |
| **Market Data**     | Pulls multi-timeframe OHLCV candles from OANDA                  |
| **Feature Engine**  | Computes 25 technical + contextual features                     |
| **Model Inference** | LightGBM regression predicts expected pip movement              |
| **Order Execution** | Places MARKET orders with SL/TP via OANDA v20 API              |
| **Exit Management** | Optional time-based exits after `k_candles` periods             |

---

## Features

- **Multi-Timeframe Analysis** — Merges 5-minute, 1-hour, and 4-hour data for richer context
- **25-Feature Engineering Pipeline** — Log returns, volatility ratios, RSI, MACD, Bollinger Bands, ATR, volume metrics, HTF alignment, trading session flags, and news proximity
- **News-Aware Trading** — Integrates Finnhub and Forex Factory economic calendars; computes minutes-until-next-event features for both US and EU releases
- **ATR-Based Risk Management** — Dynamic stop-loss calculated from current ATR × configurable multiplier
- **Configurable Model Parameters** — Pip threshold, holding period, exit strategy, and SL multiplier all externalized to `model_config.json`
- **Time-Based Exit Strategy** — Optional automatic position closure after a defined number of candles
- **Trade Limits** — Caps concurrent open positions (default: 2 trades)

---

## Project Structure

```
forex_model/
├── main.py                       # Bot entry point & OandaQuantBot class
├── requirements.txt              # Python dependencies
├── .env                          # API keys (not committed)
├── .gitignore                    # Ignores .env, .venv, model binaries
├── forex_bot.log                 # Runtime log output
│
├── model/
│   ├── forex_model_v6.pkl        # Trained LightGBM model (serialized)
│   ├── feature_names_v6.pkl      # Ordered feature name list
│   └── model_config.json         # Model & trading configuration
│
└── DataPrepper/
    └── dataset_preppr.ipynb      # Jupyter notebook for data prep & training
```

---

## Prerequisites

- **Python** 3.10+
- An **OANDA** practice or live account with API access
- A **Finnhub** API key (free tier works)

---

## Installation

1. **Clone the repository**

   ```bash
   git clone https://github.com/Bluel0la/Forex-Regression-Model.git
   cd Forex-Regression-Model
   ```

2. **Create and activate a virtual environment**

   ```bash
   python -m venv .venv

   # Windows
   .venv\Scripts\activate

   # macOS/Linux
   source .venv/bin/activate
   ```

3. **Install dependencies**

   ```bash
   pip install -r requirements.txt
   ```

4. **Set up environment variables**

   Create a `.env` file in the project root:

   ```env
   API_KEY=your_oanda_api_key
   ACCOUNT_ID=your_oanda_account_id
   FINNHUB_API_KEY=your_finnhub_api_key
   ```

---

## Configuration

All model and trading parameters are defined in [`model/model_config.json`](model/model_config.json):

```json
{
    "pip_threshold": 0.4,
    "k_candles": 12,
    "model_version": "v6",
    "use_time_exit": false,
    "atr_sl_multiplier": 1.5
}
```

| Parameter           | Type    | Default | Description                                                    |
|---------------------|---------|---------|----------------------------------------------------------------|
| `pip_threshold`     | float   | `0.4`   | Minimum predicted pip move to trigger a trade                  |
| `k_candles`         | int     | `12`    | Holding period in candles (for time-based exits)               |
| `model_version`     | string  | `"v6"`  | Model version identifier                                      |
| `use_time_exit`     | bool    | `false` | Enable automatic position closure after `k_candles` periods    |
| `atr_sl_multiplier` | float   | `1.5`   | Stop-loss distance = ATR × this multiplier                     |

---

## Usage

### Run the bot

```bash
python main.py
```

The bot will:
1. Load the trained model and config
2. Fetch the economic news calendar
3. Enter a continuous loop that:
   - Checks for time-based exits (if enabled)
   - On every 5-minute mark (+1 min offset), fetches fresh candles
   - Engineers features and runs the model
   - Executes BUY/SELL orders when conviction exceeds the threshold
   - Sleeps between cycles to respect API rate limits

### Switch to Live Trading

In `main.py`, change the environment parameter:

```python
bot = OandaQuantBot(
    api_key=API_KEY,
    account_id=ACCOUNT_ID,
    finnhub_api_key=FINNHUB_KEY,
    environment="live"  # Change from "practice" to "live"
)
```

> ⚠️ **Warning**: Live trading involves real money. Ensure thorough backtesting and paper trading before going live.

---

## Model Details

| Property             | Value                                          |
|----------------------|------------------------------------------------|
| **Algorithm**        | LightGBM (Gradient Boosted Decision Trees)     |
| **Task**             | Regression — predicts expected pip movement     |
| **Input Features**   | 25 engineered features                         |
| **Target**           | Forward pip return over the next `k` candles    |
| **Instrument**       | EUR/USD                                        |
| **Timeframe**        | 5-minute primary, with 1H and 4H overlays      |

### Feature Categories

| Category                | Features                                                            |
|-------------------------|---------------------------------------------------------------------|
| **Log Returns**         | `Return_1`, `Return_5`, `Return_12`, `Return_60`                   |
| **Volatility**          | `Vol_Short`, `Vol_Long`, `Vol_Ratio`                                |
| **Momentum**            | `RSI_14`, `RSI_Change_3`, `MACD_Hist_Norm`                         |
| **Bollinger Bands**     | `BB_Width_Norm`, `BB_Position`                                      |
| **Price Action**        | `ATR_5m_14`, `Range_Ratio`, `Close_Position`                       |
| **Volume**              | `Vol_MA_Ratio`, `Vol_Price_Corr`                                    |
| **Multi-Timeframe**     | `Norm_Dist_1H_EMA`, `Norm_Dist_4H_EMA`, `TF_Alignment`, `ATR_Ratio_1H_4H` |
| **Session**             | `Session_Asian`, `Session_London`, `Session_NY`                     |
| **News Proximity**      | `Min_Until_US_News`, `Min_Until_EU_News`                            |

### Retraining the Model

Use the Jupyter notebook at [`DataPrepper/dataset_preppr.ipynb`](DataPrepper/dataset_preppr.ipynb) to:

1. Fetch historical OANDA data
2. Engineer features with the same pipeline
3. Train/tune the LightGBM model
4. Export `forex_model_v6.pkl` and `feature_names_v6.pkl` to the `model/` directory

---

## Risk Management

| Control                | Setting                                                    |
|------------------------|------------------------------------------------------------|
| **Position Size**      | 10,000 units (1 mini lot ≈ $1/pip)                         |
| **Max Open Trades**    | 2 concurrent positions                                     |
| **Stop-Loss**          | ATR(14) × 1.5 (dynamic, adapts to volatility)              |
| **Take-Profit**        | Predicted pip distance (model output × 0.0001)             |
| **Time Exit**          | Optional — closes after `k_candles × 5` minutes            |
| **News Filter**        | Model uses minutes-to-event as a feature (not a hard block) |

---

## Logging

All activity is logged to both **stdout** and `forex_bot.log`:

```
2026-06-20 15:30:01 - INFO - Model Prediction: 0.62 pips expected move
2026-06-20 15:30:01 - WARNING - HIGH CONVICTION BUY SETUP: 0.62 pips expected
2026-06-20 15:30:02 - INFO - EXECUTED BUY at 1.08452. Trade ID: 1234. TP: 0.00062, SL: 0.00045
```

---

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/my-improvement`)
3. Commit your changes (`git commit -m "Add my improvement"`)
4. Push to your branch (`git push origin feature/my-improvement`)
5. Open a Pull Request

---

## Disclaimer

> **This software is provided for educational and research purposes only.** Trading foreign exchange (forex) carries a high level of risk and may not be suitable for all investors. Past performance of the model is not indicative of future results. You could sustain a loss of some or all of your investment. **Do not trade with money you cannot afford to lose.** The authors assume no responsibility for any financial losses incurred through the use of this software.

---

## License

This project is unlicensed. All rights reserved by the author.
