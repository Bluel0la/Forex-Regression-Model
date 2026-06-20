import joblib
import pandas as pd
import numpy as np
import time
import logging
import requests
import json
from datetime import datetime, timedelta, timezone
from oandapyV20 import API
import oandapyV20.endpoints.instruments as instruments
import oandapyV20.endpoints.orders as orders
import oandapyV20.endpoints.trades as trades
import os
from dotenv import load_dotenv

# Setup logging
logging.basicConfig(
    level=logging.INFO, 
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[logging.FileHandler('forex_bot.log'), logging.StreamHandler()]
)

class OandaQuantBot:
    # Finnhub country codes that map to EUR-area events
    EUR_COUNTRIES = ['EU', 'ERL', 'DE', 'FR', 'IT', 'ES']

    def __init__(self, api_key, account_id, finnhub_api_key, environment="practice"):
        logging.info("Initializing OandaQuantBot (v6)...")
        self.api = API(access_token=api_key, environment=environment)
        self.account_id = account_id
        self.finnhub_api_key = finnhub_api_key
        
        # Load the Model Configuration
        try:
            with open('model/model_config.json', 'r') as f:
                config = json.load(f)
            self.pip_threshold = config.get("pip_threshold", 0.4)
            self.k_candles = config.get("k_candles", 12)
            self.use_time_exit = config.get("use_time_exit", False)
            self.atr_sl_multiplier = config.get("atr_sl_multiplier", 1.5)
            logging.info(f"Loaded config: Threshold={self.pip_threshold} pips, Holding={self.k_candles} candles")
        except FileNotFoundError:
            logging.warning("model_config.json not found! Using default safe values.")
            self.pip_threshold = 0.4
            self.k_candles = 12
            self.use_time_exit = False
            self.atr_sl_multiplier = 1.5

        # Load the Brain
        self.model = joblib.load('model/forex_model_v6.pkl')
        self.features = joblib.load('model/feature_names_v6.pkl')
        logging.info(f"Model loaded. Expected features: {len(self.features)}")
        
        # Risk & Trade Management
        self.instrument = "EUR_USD"
        self.trade_size = 10000   # 10,000 units = 1 Mini Lot ($1 per pip)
        self.max_open_trades = 2  # Strictly 2 trades at a time (matches realistic backtest)
        
        # Memory state for Time-Based Exits
        self.active_trades = {}   # Format: {trade_id: expiry_datetime}
        
        # News Cache State — separate series for US and EU events
        self.us_news_cache = pd.Series(dtype='datetime64[ns, UTC]')
        self.eu_news_cache = pd.Series(dtype='datetime64[ns, UTC]')
        self.last_news_fetch_time = None

    # -------------------------------------------------------------------------
    # NEWS CALENDAR (Finnhub)
    # -------------------------------------------------------------------------
    def fetch_news_calendar(self):
        if self.finnhub_api_key:
            if self._fetch_from_finnhub(): return
            logging.warning("Finnhub failed. Falling back to Forex Factory...")
        self._fetch_from_forex_factory()

    def _fetch_from_finnhub(self):
        today = datetime.now(timezone.utc)
        start_str = today.strftime('%Y-%m-%d')
        end_str = (today + timedelta(days=3)).strftime('%Y-%m-%d')
        url = f"https://finnhub.io/api/v1/calendar/economic?from={start_str}&to={end_str}&token={self.finnhub_api_key}"
        try:
            r = requests.get(url, timeout=15)
            r.raise_for_status()
            data = r.json().get('economicCalendar', [])
            if not data: return False
            df = pd.DataFrame(data)
            df = df[df['impact'] == 'high'].copy()
            if df.empty: return False
            df['event_time'] = pd.to_datetime(df['time'], utc=True)
            self.us_news_cache = df.loc[df['country'] == 'US', 'event_time'].sort_values().reset_index(drop=True)
            self.eu_news_cache = df.loc[df['country'].isin(self.EUR_COUNTRIES), 'event_time'].sort_values().reset_index(drop=True)
            return True
        except Exception as e:
            logging.error(f"Finnhub fetch failed: {type(e).__name__}")
            return False

    def _fetch_from_forex_factory(self, max_retries=3):
        url = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
        headers = {'User-Agent': 'Mozilla/5.0'}
        for attempt in range(max_retries):
            try:
                r = requests.get(url, headers=headers, timeout=15)
                if r.status_code == 429:
                    time.sleep(2 ** (attempt + 1))
                    continue
                r.raise_for_status()
                df = pd.DataFrame(r.json())
                df = df[df['impact'] == 'High'].copy()
                if df.empty: return
                df['event_time'] = pd.to_datetime(df['date'], utc=True)
                self.us_news_cache = df.loc[df['country'] == 'USD', 'event_time'].sort_values().reset_index(drop=True)
                self.eu_news_cache = df.loc[df['country'] == 'EUR', 'event_time'].sort_values().reset_index(drop=True)
                return
            except Exception as e:
                pass

    # -------------------------------------------------------------------------
    # MARKET DATA
    # -------------------------------------------------------------------------
    def _fetch_candles(self, granularity, count):
        params = {"granularity": granularity, "count": count}
        r = instruments.InstrumentsCandles(instrument=self.instrument, params=params)
        self.api.request(r)
        data = []
        for c in r.response['candles']:
            if c['complete']:
                data.append({
                    'Datetime': pd.to_datetime(c['time'], utc=True),
                    'Open': float(c['mid']['o']),
                    'High': float(c['mid']['h']),
                    'Low': float(c['mid']['l']),
                    'Close': float(c['mid']['c']),
                    'Volume': int(c['volume'])
                })
        return pd.DataFrame(data).set_index('Datetime')

    def fetch_multi_tf_data(self):
        """Fetches 5M, 1H, and 4H candles and computes higher-TF indicators."""
        df_5m = self._fetch_candles("M5", 400)   # Need enough history for 288-period rolling features
        df_1h = self._fetch_candles("H1", 50)
        df_4h = self._fetch_candles("H4", 50)

        df_1h['1H_EMA20'] = df_1h['Close'].ewm(span=20, adjust=False).mean()
        tr_1h = pd.concat([
            df_1h['High'] - df_1h['Low'],
            (df_1h['High'] - df_1h['Close'].shift(1)).abs(),
            (df_1h['Low']  - df_1h['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df_1h['1H_ATR14'] = tr_1h.rolling(14).mean()

        df_4h['4H_EMA20'] = df_4h['Close'].ewm(span=20, adjust=False).mean()
        tr_4h = pd.concat([
            df_4h['High'] - df_4h['Low'],
            (df_4h['High'] - df_4h['Close'].shift(1)).abs(),
            (df_4h['Low']  - df_4h['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df_4h['4H_ATR14'] = tr_4h.rolling(14).mean()

        df_1h_join = df_1h[['1H_EMA20', '1H_ATR14']]
        df_4h_join = df_4h[['4H_EMA20', '4H_ATR14']]

        # Join and forward fill with strict limits (matching dataset_preppr.ipynb)
        df = df_5m.join(df_1h_join, how='left')
        df[['1H_EMA20', '1H_ATR14']] = df[['1H_EMA20', '1H_ATR14']].ffill(limit=12)
        
        df = df.join(df_4h_join, how='left')
        df[['4H_EMA20', '4H_ATR14']] = df[['4H_EMA20', '4H_ATR14']].ffill(limit=48)

        # Drop rows with NaN HTF indicators
        return df.dropna(subset=['1H_ATR14', '4H_ATR14'])

    # -------------------------------------------------------------------------
    # FEATURE ENGINEERING (v6 - 25 Features)
    # -------------------------------------------------------------------------
    @staticmethod
    def _minutes_until_next(reference_time, event_series, default=10000):
        future = event_series[event_series > reference_time]
        if not future.empty: return (future.iloc[0] - reference_time).total_seconds() / 60.0
        return default

    def engineer_features(self, df):
        df_p = df.copy()

        df_p['Return_1'] = np.log(df_p['Close'] / df_p['Close'].shift(1))
        df_p['Return_5'] = np.log(df_p['Close'] / df_p['Close'].shift(5))
        df_p['Return_12'] = np.log(df_p['Close'] / df_p['Close'].shift(12))
        df_p['Return_60'] = np.log(df_p['Close'] / df_p['Close'].shift(60))

        df_p['Vol_Short'] = df_p['Return_1'].rolling(window=12).std()
        df_p['Vol_Long']  = df_p['Return_1'].rolling(window=288).std()
        df_p['Vol_Ratio'] = df_p['Vol_Short'] / (df_p['Vol_Long'] + 1e-8)

        delta = df_p['Close'].diff()
        gain = delta.clip(lower=0)
        loss = -delta.clip(upper=0)
        avg_gain = gain.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        avg_loss = loss.ewm(alpha=1/14, min_periods=14, adjust=False).mean()
        rs = avg_gain / (avg_loss + 1e-8)
        df_p['RSI_14'] = 100 - (100 / (1 + rs))
        df_p['RSI_Change_3'] = df_p['RSI_14'] - df_p['RSI_14'].shift(3)

        ema12 = df_p['Close'].ewm(span=12, adjust=False).mean()
        ema26 = df_p['Close'].ewm(span=26, adjust=False).mean()
        macd_line = ema12 - ema26
        signal_line = macd_line.ewm(span=9, adjust=False).mean()
        df_p['MACD_Hist_Norm'] = (macd_line - signal_line) / (df_p['1H_ATR14'] + 1e-8)

        bb_mid = df_p['Close'].rolling(20).mean()
        bb_std = df_p['Close'].rolling(20).std()
        bb_upper = bb_mid + 2 * bb_std
        bb_lower = bb_mid - 2 * bb_std
        df_p['BB_Width_Norm'] = (bb_upper - bb_lower) / (df_p['Close'] + 1e-8)
        df_p['BB_Position'] = (df_p['Close'] - bb_lower) / (bb_upper - bb_lower + 1e-8)

        tr_5m = pd.concat([
            df_p['High'] - df_p['Low'],
            (df_p['High'] - df_p['Close'].shift(1)).abs(),
            (df_p['Low'] - df_p['Close'].shift(1)).abs()
        ], axis=1).max(axis=1)
        df_p['ATR_5m_14'] = tr_5m.rolling(14).mean()
        df_p['Range_Ratio'] = (df_p['High'] - df_p['Low']) / (df_p['ATR_5m_14'] + 1e-8)
        df_p['Close_Position'] = (df_p['Close'] - df_p['Low']) / (df_p['High'] - df_p['Low'] + 1e-8)

        df_p['Vol_MA_Ratio'] = df_p['Volume'] / (df_p['Volume'].rolling(20).mean() + 1e-8)
        df_p['Vol_Price_Corr'] = df_p['Volume'].rolling(20).corr(df_p['Return_1'].abs())

        df_p['Norm_Dist_1H_EMA'] = (df_p['Close'] - df_p['1H_EMA20']) / (df_p['1H_ATR14'] + 1e-8)
        df_p['Norm_Dist_4H_EMA'] = (df_p['Close'] - df_p['4H_EMA20']) / (df_p['4H_ATR14'] + 1e-8)
        df_p['TF_Alignment'] = np.sign(df_p['Norm_Dist_1H_EMA']) * np.sign(df_p['Norm_Dist_4H_EMA'])
        df_p['ATR_Ratio_1H_4H'] = df_p['1H_ATR14'] / (df_p['4H_ATR14'] + 1e-8)

        hour = df_p.index.hour
        df_p['Session_Asian']  = ((hour >= 0)  & (hour < 8)).astype(int)
        df_p['Session_London'] = ((hour >= 8)  & (hour < 16)).astype(int)
        df_p['Session_NY']     = ((hour >= 12) & (hour < 20)).astype(int)

        latest_time = df_p.index[-1]
        df_p['Min_Until_US_News'] = self._minutes_until_next(latest_time, self.us_news_cache)
        df_p['Min_Until_EU_News'] = self._minutes_until_next(latest_time, self.eu_news_cache)

        try:
            return df_p[self.features], df_p['ATR_5m_14']
        except KeyError as e:
            logging.error(f"Feature mismatch! Missing columns: {e}")
            raise

    # -------------------------------------------------------------------------
    # ORDER EXECUTION
    # -------------------------------------------------------------------------
    def execute_trade(self, signal, predicted_pips, current_price, current_atr):
        if len(self.active_trades) >= self.max_open_trades:
            logging.info(f"Max open trades ({self.max_open_trades}) reached. Skipping signal.")
            return

        units = self.trade_size if signal == 1 else -self.trade_size
        direction = "BUY" if signal == 1 else "SELL"
        
        # 1 pip = 0.0001 for EUR/USD
        tp_distance = abs(predicted_pips) * 0.0001
        sl_distance = (current_atr * self.atr_sl_multiplier)

        data = {
            "order": {
                "units": str(units),
                "instrument": self.instrument,
                "timeInForce": "FOK",
                "type": "MARKET",
                "positionFill": "DEFAULT",
                "takeProfitOnFill": {
                    "distance": f"{tp_distance:.5f}"
                },
                "stopLossOnFill": {
                    "distance": f"{sl_distance:.5f}"
                }
            }
        }
        
        r = orders.OrderCreate(self.account_id, data=data)
        try:
            response = self.api.request(r)
            if 'orderCancelTransaction' in response:
                reason = response['orderCancelTransaction'].get('reason', 'Unknown reason')
                logging.error(f"Order Cancelled by Oanda. Reason: {reason}")
                return

            trade_id = response['orderFillTransaction']['id']
            price = response['orderFillTransaction']['price']
            
            exit_time = datetime.now(timezone.utc) + timedelta(minutes=self.k_candles * 5)
            self.active_trades[trade_id] = exit_time
            
            logging.info(f"EXECUTED {direction} at {price}. Trade ID: {trade_id}. TP Distance: {tp_distance:.5f}, SL Distance: {sl_distance:.5f}")
        except Exception as e:
            logging.error(f"Order Execution Failed: {e}. Response: {response if 'response' in locals() else 'No response'}")

    # -------------------------------------------------------------------------
    # EXIT MANAGEMENT
    # -------------------------------------------------------------------------
    def manage_time_based_exits(self):
        current_time = datetime.now(timezone.utc)
        trades_to_close = [tid for tid, exit_t in self.active_trades.items() if current_time >= exit_t]
                
        for trade_id in trades_to_close:
            logging.info(f"Time exit reached for Trade {trade_id}. Closing position...")
            data = {"units": "ALL"}
            r = trades.TradeClose(accountID=self.account_id, tradeID=trade_id, data=data)
            try:
                self.api.request(r)
                logging.info(f"Successfully closed Trade {trade_id}.")
            except Exception as e:
                logging.error(f"Failed to close Trade {trade_id}: {e}")
            finally:
                del self.active_trades[trade_id]

    # -------------------------------------------------------------------------
    # MAIN LOOP
    # -------------------------------------------------------------------------
    def run(self):
        logging.info("Bot started. Scanning for high-conviction setups...")
        while True:
            try:
                current_time = datetime.now(timezone.utc)
                
                if (self.last_news_fetch_time is None or (current_time - self.last_news_fetch_time).days >= 1):
                    self.fetch_news_calendar()
                    self.last_news_fetch_time = current_time
                
                if self.use_time_exit:
                    self.manage_time_based_exits()
                
                if current_time.minute % 5 == 1:
                    df = self.fetch_multi_tf_data()
                    X, atr_series = self.engineer_features(df)
                    
                    current_features = X.tail(1)
                    current_price = df['Close'].iloc[-1]
                    current_atr = atr_series.iloc[-1]
                    
                    # Regression model returns predicted pips
                    predicted_pips = self.model.predict(current_features.values)[0]
                    
                    logging.info(f"Model Prediction: {predicted_pips:.2f} pips expected move")
                    
                    if predicted_pips >= self.pip_threshold:
                        logging.warning(f"HIGH CONVICTION BUY SETUP: {predicted_pips:.2f} pips expected")
                        self.execute_trade(1, predicted_pips, current_price, current_atr)
                    elif predicted_pips <= -self.pip_threshold:
                        logging.warning(f"HIGH CONVICTION SELL SETUP: {predicted_pips:.2f} pips expected")
                        self.execute_trade(-1, predicted_pips, current_price, current_atr)
                    
                    time.sleep(240)
                else:
                    time.sleep(10)
                
            except Exception as e:
                logging.error(f"Error in main loop: {e}")
                time.sleep(60)

# --- EXECUTION ---
if __name__ == "__main__":
    load_dotenv()
    
    API_KEY = os.getenv("API_KEY")
    ACCOUNT_ID = os.getenv("ACCOUNT_ID")
    FINNHUB_KEY = os.getenv("FINNHUB_API_KEY")
    
    if not all([API_KEY, ACCOUNT_ID, FINNHUB_KEY]):
        logging.error("Missing environment variables. Ensure API_KEY, ACCOUNT_ID, and FINNHUB_API_KEY are set.")
        exit(1)
    
    bot = OandaQuantBot(
        api_key=API_KEY,
        account_id=ACCOUNT_ID,
        finnhub_api_key=FINNHUB_KEY,
        environment="practice"
    )
    bot.run()