from __future__ import annotations

import numpy as np
import pandas as pd


FEATURE_COLUMNS_V7 = [
    "Return_1",
    "Return_5",
    "Return_12",
    "Return_60",
    "Vol_Short",
    "Vol_Long",
    "Vol_Ratio",
    "RSI_14",
    "RSI_Change_3",
    "MACD_Hist_Norm",
    "BB_Width_Norm",
    "BB_Position",
    "Range_Ratio",
    "Close_Position",
    "Vol_MA_Ratio",
    "Vol_Price_Corr",
    "Norm_Dist_1H_EMA",
    "Norm_Dist_4H_EMA",
    "TF_Alignment",
    "ATR_Ratio_1H_4H",
    "Session_Asian",
    "Session_London",
    "Session_NY",
    "Min_Until_US_News",
    "Min_Until_EU_News",
]

USD_NEWS_COUNTRIES = ["US"]
EUR_NEWS_COUNTRIES = ["EU", "DE", "FR", "IT", "ES"]


def shift_htf_index(df: pd.DataFrame, hours: int) -> pd.DataFrame:
    shifted = df.copy()
    shifted.index = shifted.index + pd.Timedelta(hours=hours)
    return shifted


def _true_range(df: pd.DataFrame) -> pd.Series:
    return pd.concat(
        [
            df["High"] - df["Low"],
            (df["High"] - df["Close"].shift(1)).abs(),
            (df["Low"] - df["Close"].shift(1)).abs(),
        ],
        axis=1,
    ).max(axis=1)


def compute_htf_indicators(
    df_1h: pd.DataFrame, df_4h: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    df_1h_ind = df_1h.copy()
    df_4h_ind = df_4h.copy()

    df_1h_ind["1H_EMA20"] = df_1h_ind["Close"].ewm(span=20, adjust=False).mean()
    df_1h_ind["1H_ATR14"] = _true_range(df_1h_ind).rolling(14).mean()

    df_4h_ind["4H_EMA20"] = df_4h_ind["Close"].ewm(span=20, adjust=False).mean()
    df_4h_ind["4H_ATR14"] = _true_range(df_4h_ind).rolling(14).mean()

    return (
        df_1h_ind[["1H_EMA20", "1H_ATR14"]],
        df_4h_ind[["4H_EMA20", "4H_ATR14"]],
    )


def join_timeframes(
    df_5m: pd.DataFrame, df_1h: pd.DataFrame, df_4h: pd.DataFrame
) -> pd.DataFrame:
    df_1h_ind, df_4h_ind = compute_htf_indicators(df_1h, df_4h)
    df_1h_ind = shift_htf_index(df_1h_ind, hours=1)
    df_4h_ind = shift_htf_index(df_4h_ind, hours=4)

    df = df_5m.join(df_1h_ind, how="left")
    df[["1H_EMA20", "1H_ATR14"]] = df[["1H_EMA20", "1H_ATR14"]].ffill(limit=12)

    df = df.join(df_4h_ind, how="left")
    df[["4H_EMA20", "4H_ATR14"]] = df[["4H_EMA20", "4H_ATR14"]].ffill(limit=48)

    return df.dropna(subset=["1H_ATR14", "4H_ATR14"])


def _align_event_times(event_times, target_index: pd.DatetimeIndex) -> pd.DatetimeIndex:
    events = pd.DatetimeIndex(pd.to_datetime(pd.Series(event_times), errors="coerce").dropna())
    if events.empty:
        return events

    if target_index.tz is not None:
        if events.tz is None:
            events = events.tz_localize("UTC")
        else:
            events = events.tz_convert("UTC")
    elif events.tz is not None:
        events = events.tz_convert("UTC").tz_localize(None)

    return events.sort_values()


def _minutes_until_next(
    reference_times: pd.DatetimeIndex, event_times, default: float = 10000.0
) -> np.ndarray:
    events = _align_event_times(event_times, reference_times)
    if events.empty:
        return np.full(len(reference_times), default, dtype=float)

    ref_ns = reference_times.view("int64")
    event_ns = events.view("int64")
    positions = np.searchsorted(event_ns, ref_ns, side="left")

    minutes = np.full(len(reference_times), default, dtype=float)
    valid = positions < len(event_ns)
    minutes[valid] = (event_ns[positions[valid]] - ref_ns[valid]) / 1_000_000_000 / 60.0
    return minutes


def compute_news_features(
    df: pd.DataFrame, us_news, eu_news, default: float = 10000.0
) -> pd.DataFrame:
    enriched = df.copy()
    enriched["Min_Until_US_News"] = _minutes_until_next(enriched.index, us_news, default)
    enriched["Min_Until_EU_News"] = _minutes_until_next(enriched.index, eu_news, default)
    return enriched


def engineer_25_features(
    df: pd.DataFrame, us_news=None, eu_news=None, dropna: bool = True
) -> pd.DataFrame:
    df_p = df.copy()

    df_p["Return_1"] = np.log(df_p["Close"] / df_p["Close"].shift(1))
    df_p["Return_5"] = np.log(df_p["Close"] / df_p["Close"].shift(5))
    df_p["Return_12"] = np.log(df_p["Close"] / df_p["Close"].shift(12))
    df_p["Return_60"] = np.log(df_p["Close"] / df_p["Close"].shift(60))

    df_p["Vol_Short"] = df_p["Return_1"].rolling(window=12).std()
    df_p["Vol_Long"] = df_p["Return_1"].rolling(window=288).std()
    df_p["Vol_Ratio"] = df_p["Vol_Short"] / (df_p["Vol_Long"] + 1e-8)

    delta = df_p["Close"].diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / 14, min_periods=14, adjust=False).mean()
    rs = avg_gain / (avg_loss + 1e-8)
    df_p["RSI_14"] = 100 - (100 / (1 + rs))
    df_p["RSI_Change_3"] = df_p["RSI_14"] - df_p["RSI_14"].shift(3)

    ema12 = df_p["Close"].ewm(span=12, adjust=False).mean()
    ema26 = df_p["Close"].ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    df_p["MACD_Hist_Norm"] = (macd_line - signal_line) / (df_p["1H_ATR14"] + 1e-8)

    bb_mid = df_p["Close"].rolling(20).mean()
    bb_std = df_p["Close"].rolling(20).std()
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    df_p["BB_Width_Norm"] = (bb_upper - bb_lower) / (df_p["Close"] + 1e-8)
    df_p["BB_Position"] = (df_p["Close"] - bb_lower) / (bb_upper - bb_lower + 1e-8)

    df_p["ATR_5m_14"] = _true_range(df_p).rolling(14).mean()
    df_p["Range_Ratio"] = (df_p["High"] - df_p["Low"]) / (df_p["ATR_5m_14"] + 1e-8)
    df_p["Close_Position"] = (df_p["Close"] - df_p["Low"]) / (
        df_p["High"] - df_p["Low"] + 1e-8
    )

    df_p["Vol_MA_Ratio"] = df_p["Volume"] / (df_p["Volume"].rolling(20).mean() + 1e-8)
    df_p["Vol_Price_Corr"] = df_p["Volume"].rolling(20).corr(df_p["Return_1"].abs())

    df_p["Norm_Dist_1H_EMA"] = (df_p["Close"] - df_p["1H_EMA20"]) / (
        df_p["1H_ATR14"] + 1e-8
    )
    df_p["Norm_Dist_4H_EMA"] = (df_p["Close"] - df_p["4H_EMA20"]) / (
        df_p["4H_ATR14"] + 1e-8
    )
    df_p["TF_Alignment"] = np.sign(df_p["Norm_Dist_1H_EMA"]) * np.sign(
        df_p["Norm_Dist_4H_EMA"]
    )
    df_p["ATR_Ratio_1H_4H"] = df_p["1H_ATR14"] / (df_p["4H_ATR14"] + 1e-8)

    hour = df_p.index.hour
    df_p["Session_Asian"] = ((hour >= 0) & (hour < 8)).astype(int)
    df_p["Session_London"] = ((hour >= 8) & (hour < 16)).astype(int)
    df_p["Session_NY"] = ((hour >= 12) & (hour < 20)).astype(int)

    df_p = compute_news_features(df_p, [] if us_news is None else us_news, [] if eu_news is None else eu_news)

    if dropna:
        df_p = df_p.dropna(subset=FEATURE_COLUMNS_V7 + ["ATR_5m_14"])

    return df_p


def generate_barrier_labels(
    df: pd.DataFrame, k_candles: int, atr_multiplier: float
) -> pd.Series:
    close = df["Close"].to_numpy()
    high = df["High"].to_numpy()
    low = df["Low"].to_numpy()
    atr = df["ATR_5m_14"].to_numpy()

    labels = np.full(len(df), np.nan)
    last_start = len(df) - k_candles - 1

    for i in range(max(last_start + 1, 0)):
        if np.isnan(atr[i]):
            continue

        upper_barrier = close[i] + atr[i] * atr_multiplier
        lower_barrier = close[i] - atr[i] * atr_multiplier
        label = 0

        for j in range(i + 1, i + k_candles + 1):
            long_hit = high[j] >= upper_barrier
            short_hit = low[j] <= lower_barrier
            if long_hit and short_hit:
                label = 0
                break
            if long_hit:
                label = 1
                break
            if short_hit:
                label = -1
                break

        labels[i] = label

    return pd.Series(labels, index=df.index, name="Barrier_Label")


def labels_to_lgb_classes(labels: pd.Series) -> pd.Series:
    return labels.map({0: 0, 1: 1, -1: 2}).astype("int64")


class ForexFeatureEngineer:
    def __init__(self, feature_columns=None):
        self.feature_columns = feature_columns or FEATURE_COLUMNS_V7

    def build_live_frame(self, df_5m, df_1h, df_4h, us_news=None, eu_news=None):
        joined = join_timeframes(df_5m, df_1h, df_4h)
        return engineer_25_features(joined, us_news=us_news, eu_news=eu_news)

    def feature_matrix(self, df: pd.DataFrame) -> tuple[pd.DataFrame, pd.Series]:
        missing = [col for col in self.feature_columns if col not in df.columns]
        if missing:
            raise KeyError(f"Missing feature columns: {missing}")
        return df[self.feature_columns], df["ATR_5m_14"]
