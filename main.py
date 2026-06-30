import json
import logging
import os
import time
from datetime import datetime, timedelta, timezone

import joblib
import pandas as pd
import requests
from dotenv import load_dotenv
from oandapyV20 import API
import oandapyV20.endpoints.accounts as accounts
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades

from core.features import EUR_NEWS_COUNTRIES, FEATURE_COLUMNS_V7, ForexFeatureEngineer, USD_NEWS_COUNTRIES


log_handlers = [logging.StreamHandler()]
if not os.getenv("VERCEL"):
    log_handlers.insert(0, logging.FileHandler("forex_bot.log"))

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=log_handlers,
)


def app(environ, start_response):
    """Small WSGI health endpoint for Vercel's Python runtime."""
    payload = {
        "status": "ok",
        "service": "forex-regression-model",
        "message": "Deploy is healthy. Run python main.py to start the trading bot loop.",
    }
    body = json.dumps(payload).encode("utf-8")
    headers = [
        ("Content-Type", "application/json"),
        ("Content-Length", str(len(body))),
    ]

    start_response("200 OK", headers)
    return [body]


class OandaQuantBot:
    def __init__(self, api_key, account_id, finnhub_api_key, environment="practice"):
        logging.info("Initializing OandaQuantBot (barrier v7)...")
        self.api = API(access_token=api_key, environment=environment)
        self.account_id = account_id
        self.finnhub_api_key = finnhub_api_key
        self.instrument = "EUR_USD"

        config = self.load_config()
        self.model_type = config.get("model_type", "barrier_v7")
        self.k_candles = config.get("k_candles", 12)
        self.use_time_exit = config.get("use_time_exit", False)
        self.atr_sl_multiplier = config.get("atr_sl_multiplier", 1.5)
        self.confidence_threshold = config.get("confidence_threshold", 0.55)
        self.risk_pct = min(config.get("risk_pct", 0.02), config.get("max_risk_pct", 0.10))
        self.max_risk_pct = config.get("max_risk_pct", 0.10)
        self.max_daily_drawdown = config.get("max_daily_drawdown", 0.20)
        self.max_open_trades = config.get("max_open_trades", 2)
        self.max_position_size = config.get("max_position_size", 100000)

        self.model = joblib.load("model/forex_model_v7.pkl")
        try:
            self.features = joblib.load("model/feature_names_v7.pkl")
        except FileNotFoundError:
            self.features = FEATURE_COLUMNS_V7
        self.feature_engineer = ForexFeatureEngineer(self.features)
        logging.info("Loaded %s model with %s features", self.model_type, len(self.features))

        self.daily_start_nav = None
        self.daily_start_date = None

        self.us_news_cache = pd.Series(dtype="datetime64[ns, UTC]")
        self.eu_news_cache = pd.Series(dtype="datetime64[ns, UTC]")
        self.last_news_fetch_time = None

    def load_config(self):
        defaults = {
            "model_type": "barrier_v7",
            "k_candles": 12,
            "atr_sl_multiplier": 1.5,
            "confidence_threshold": 0.55,
            "risk_pct": 0.02,
            "max_risk_pct": 0.10,
            "max_daily_drawdown": 0.20,
            "max_open_trades": 2,
            "max_position_size": 100000,
            "use_time_exit": False,
        }
        try:
            with open("model/model_config.json", "r", encoding="utf-8-sig") as f:
                config = json.load(f)
            logging.info("Loaded model config: %s", config)
            return config
        except FileNotFoundError:
            logging.warning("model_config.json not found. Using conservative v7 defaults.")
            return defaults
        except json.JSONDecodeError as exc:
            logging.warning("model_config.json is malformed (%s). Using defaults.", exc)
            return defaults

    # ------------------------------------------------------------------
    # NEWS CALENDAR
    # ------------------------------------------------------------------
    @staticmethod
    def normalize_news_country(country):
        mapping = {"USD": "US", "EUR": "EU"}
        return mapping.get(country, country)

    def fetch_news_calendar(self):
        loaded = False
        if self.finnhub_api_key:
            loaded = self._fetch_from_finnhub()
            if loaded:
                return
            logging.info("Falling back to Forex Factory calendar...")
        loaded = self._fetch_from_forex_factory()
        if not loaded:
            logging.warning(
                "No news calendar loaded. Using default 10000-minute news proximity until next refresh."
            )

    def _cache_news_events(self, df, time_column, country_column, high_impact_value):
        required_columns = {"impact", time_column, country_column}
        missing_columns = required_columns.difference(df.columns)
        if missing_columns:
            logging.warning("News calendar missing expected columns: %s", sorted(missing_columns))
            return False

        df = df[df["impact"] == high_impact_value].copy()
        if df.empty:
            return False

        df["event_time"] = pd.to_datetime(df[time_column], utc=True, errors="coerce")
        df = df.dropna(subset=["event_time"])
        df["country_norm"] = df[country_column].map(self.normalize_news_country)
        self.us_news_cache = df.loc[
            df["country_norm"].isin(USD_NEWS_COUNTRIES), "event_time"
        ].sort_values().reset_index(drop=True)
        self.eu_news_cache = df.loc[
            df["country_norm"].isin(EUR_NEWS_COUNTRIES), "event_time"
        ].sort_values().reset_index(drop=True)
        loaded = not (self.us_news_cache.empty and self.eu_news_cache.empty)
        if loaded:
            logging.info(
                "News calendar loaded: %s US events, %s EU events",
                len(self.us_news_cache),
                len(self.eu_news_cache),
            )
        return loaded

    def _fetch_from_finnhub(self):
        today = datetime.now(timezone.utc)
        start_str = today.strftime("%Y-%m-%d")
        end_str = (today + timedelta(days=3)).strftime("%Y-%m-%d")
        url = "https://finnhub.io/api/v1/calendar/economic"
        params = {"from": start_str, "to": end_str, "token": self.finnhub_api_key}
        try:
            response = requests.get(url, params=params, timeout=(3.05, 8))
            response.raise_for_status()
            data = response.json().get("economicCalendar", [])
            if not data:
                logging.info("Finnhub returned no economic calendar events for %s -> %s.", start_str, end_str)
                return False
            return self._cache_news_events(pd.DataFrame(data), "time", "country", "high")
        except requests.HTTPError as exc:
            status_code = exc.response.status_code if exc.response is not None else "unknown"
            if status_code in (401, 403):
                logging.warning("Finnhub rejected the API key or calendar access (HTTP %s).", status_code)
            else:
                logging.warning("Finnhub calendar request failed (HTTP %s).", status_code)
            return False
        except Exception as exc:
            logging.warning("Finnhub calendar request failed: %s", type(exc).__name__)
            return False

    def _fetch_from_forex_factory(self, max_retries=2):
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        headers = {"User-Agent": "Mozilla/5.0"}
        for attempt in range(max_retries):
            try:
                response = requests.get(url, headers=headers, timeout=(3.05, 8))
                if response.status_code == 429:
                    logging.warning("Forex Factory rate limited the calendar request (HTTP 429).")
                    if attempt < max_retries - 1:
                        time.sleep(2 ** attempt)
                    continue
                response.raise_for_status()
                if self._cache_news_events(pd.DataFrame(response.json()), "date", "country", "High"):
                    return True
            except Exception as exc:
                logging.warning("Forex Factory fetch attempt failed: %s", type(exc).__name__)
        return False

    # ------------------------------------------------------------------
    # OANDA ACCOUNT STATE
    # ------------------------------------------------------------------
    def get_open_trades(self):
        request = trades.OpenTrades(accountID=self.account_id)
        response = self.api.request(request)
        return [t for t in response.get("trades", []) if t.get("instrument") == self.instrument]

    def get_account_nav(self):
        request = accounts.AccountSummary(accountID=self.account_id)
        response = self.api.request(request)
        account = response.get("account", {})
        return float(account.get("NAV", account.get("balance")))

    def refresh_daily_start_nav(self):
        today = datetime.now(timezone.utc).date()
        if self.daily_start_date != today or self.daily_start_nav is None:
            self.daily_start_nav = self.get_account_nav()
            self.daily_start_date = today
            logging.info("Daily start NAV reset: %.2f", self.daily_start_nav)

    def daily_drawdown_allows_trade(self):
        self.refresh_daily_start_nav()
        current_nav = self.get_account_nav()
        drawdown_floor = self.daily_start_nav * (1 - self.max_daily_drawdown)
        if current_nav < drawdown_floor:
            logging.error(
                "Daily drawdown kill-switch active. Current NAV %.2f below floor %.2f.",
                current_nav,
                drawdown_floor,
            )
            return False
        return True

    def calculate_position_size(self, sl_distance_price):
        if sl_distance_price <= 0:
            raise ValueError("Stop-loss distance must be positive.")
        nav = self.get_account_nav()
        risk_amount = nav * min(self.risk_pct, self.max_risk_pct)
        units = int(risk_amount / sl_distance_price)
        return max(1, min(units, self.max_position_size))

    # ------------------------------------------------------------------
    # MARKET DATA
    # ------------------------------------------------------------------
    def _fetch_candles(self, granularity, count):
        params = {"granularity": granularity, "count": count}
        request = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
        self.api.request(request)
        rows = []
        for candle in request.response["candles"]:
            if candle["complete"]:
                rows.append(
                    {
                        "Datetime": pd.to_datetime(candle["time"], utc=True),
                        "Open": float(candle["mid"]["o"]),
                        "High": float(candle["mid"]["h"]),
                        "Low": float(candle["mid"]["l"]),
                        "Close": float(candle["mid"]["c"]),
                        "Volume": int(candle["volume"]),
                    }
                )
        return pd.DataFrame(rows).set_index("Datetime")

    def fetch_multi_tf_data(self):
        df_5m = self._fetch_candles("M5", 400)
        df_1h = self._fetch_candles("H1", 80)
        df_4h = self._fetch_candles("H4", 80)
        live_df = self.feature_engineer.build_live_frame(
            df_5m,
            df_1h,
            df_4h,
            us_news=self.us_news_cache,
            eu_news=self.eu_news_cache,
        )
        logging.info("Built live feature frame with %s rows", len(live_df))
        return live_df

    # ------------------------------------------------------------------
    # ORDER EXECUTION
    # ------------------------------------------------------------------
    def execute_trade(self, signal, current_price, current_atr):
        open_trades = self.get_open_trades()
        if len(open_trades) >= self.max_open_trades:
            logging.info("Max open trades (%s) reached. Skipping signal.", self.max_open_trades)
            return
        if not self.daily_drawdown_allows_trade():
            return

        sl_distance = current_atr * self.atr_sl_multiplier
        units_abs = self.calculate_position_size(sl_distance)
        units = units_abs if signal == 1 else -units_abs
        direction = "BUY" if signal == 1 else "SELL"

        data = {
            "order": {
                "units": str(units),
                "instrument": self.instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "takeProfitOnFill": {"distance": f"{sl_distance:.5f}"},
                "stopLossOnFill": {"distance": f"{sl_distance:.5f}"},
            }
        }

        request = orders.OrderCreate(self.account_id, data=data)
        try:
            response = self.api.request(request)
            if "orderCancelTransaction" in response:
                reason = response["orderCancelTransaction"].get("reason", "Unknown reason")
                logging.error("Order cancelled by OANDA. Reason: %s", reason)
                return

            fill = response.get("orderFillTransaction", {})
            trade_id = fill.get("tradeOpened", {}).get("tradeID")
            price = fill.get("price", current_price)
            logging.info(
                "EXECUTED %s %s units at %s. Trade ID: %s. TP/SL distance: %.5f",
                direction,
                units_abs,
                price,
                trade_id,
                sl_distance,
            )
        except Exception as exc:
            logging.error("Order execution failed: %s", exc)

    # ------------------------------------------------------------------
    # EXIT MANAGEMENT
    # ------------------------------------------------------------------
    def manage_time_based_exits(self):
        current_time = datetime.now(timezone.utc)
        max_age = timedelta(minutes=self.k_candles * 5)
        for trade in self.get_open_trades():
            trade_id = trade.get("id")
            open_time = pd.to_datetime(trade.get("openTime"), utc=True).to_pydatetime()
            if current_time - open_time < max_age:
                continue

            logging.info("Time exit reached for Trade %s. Closing position...", trade_id)
            request = trades.TradeClose(accountID=self.account_id, tradeID=trade_id, data={"units": "ALL"})
            try:
                self.api.request(request)
                logging.info("Successfully closed Trade %s.", trade_id)
            except Exception as exc:
                logging.error("Failed to close Trade %s: %s", trade_id, exc)

    # ------------------------------------------------------------------
    # MAIN LOOP
    # ------------------------------------------------------------------
    def run(self):
        logging.info("Bot started. Scanning for barrier-classification setups...")
        while True:
            try:
                current_time = datetime.now(timezone.utc)

                if self.last_news_fetch_time is None or (current_time - self.last_news_fetch_time).days >= 1:
                    self.fetch_news_calendar()
                    self.last_news_fetch_time = current_time

                if self.use_time_exit:
                    self.manage_time_based_exits()

                if current_time.minute % 5 == 1:
                    df = self.fetch_multi_tf_data()
                    X, atr_series = self.feature_engineer.feature_matrix(df)
                    current_features = X.tail(1)
                    current_price = df["Close"].iloc[-1]
                    current_atr = atr_series.iloc[-1]

                    probs = self.model.predict(current_features.values)[0]
                    p_no_trade, p_long, p_short = probs[0], probs[1], probs[2]
                    logging.info(
                        "Model probabilities: NO_TRADE=%.3f LONG=%.3f SHORT=%.3f",
                        p_no_trade,
                        p_long,
                        p_short,
                    )

                    if p_long > self.confidence_threshold and p_long >= p_short:
                        logging.warning("HIGH CONFIDENCE BUY SETUP: %.2f%%", p_long * 100)
                        self.execute_trade(1, current_price, current_atr)
                    elif p_short > self.confidence_threshold:
                        logging.warning("HIGH CONFIDENCE SELL SETUP: %.2f%%", p_short * 100)
                        self.execute_trade(-1, current_price, current_atr)

                    time.sleep(240)
                else:
                    time.sleep(10)
            except Exception as exc:
                logging.error("Error in main loop: %s", exc)
                time.sleep(60)


if __name__ == "__main__":
    load_dotenv()

    API_KEY = os.getenv("API_KEY")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")

    if not all([API_KEY, ACCOUNT_ID, FINNHUB_KEY]):
        logging.error("Missing environment variables. Ensure API_KEY, ACCOUNT_ID, and FINNHUB_API_KEY are set.")
        raise SystemExit(1)

    bot = OandaQuantBot(
        api_key=API_KEY,
        account_id=ACCOUNT_ID,
        finnhub_api_key=FINNHUB_KEY,
        environment="practice",
    )
    bot.run()
