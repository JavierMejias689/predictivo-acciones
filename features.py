"""Feature engineering avanzado para clasificación de tendencia diaria."""

from __future__ import annotations

import numpy as np
import pandas as pd

import config


# ---------------------------------------------------------------------------
# Indicadores técnicos
# ---------------------------------------------------------------------------

def compute_rsi(series: pd.Series, window: int = 14) -> pd.Series:
    """RSI clásico de Wilder."""
    delta = series.diff()
    gains = delta.clip(lower=0)
    losses = -delta.clip(upper=0)
    avg_gain = gains.rolling(window=window, min_periods=window).mean()
    avg_loss = losses.rolling(window=window, min_periods=window).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def compute_macd(series: pd.Series) -> tuple[pd.Series, pd.Series, pd.Series]:
    """MACD (12, 26, 9)."""
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    macd_signal = macd.ewm(span=9, adjust=False).mean()
    macd_hist = macd - macd_signal
    return macd, macd_signal, macd_hist


def compute_bollinger_bands(series: pd.Series, window: int = 20, num_std: float = 2.0) -> tuple[pd.Series, pd.Series, pd.Series, pd.Series]:
    """Bollinger Bands: upper, lower, %B, bandwidth."""
    sma = series.rolling(window).mean()
    std = series.rolling(window).std()
    upper = sma + num_std * std
    lower = sma - num_std * std
    pct_b = (series - lower) / (upper - lower).replace(0, np.nan)
    bandwidth = (upper - lower) / sma.replace(0, np.nan)
    return upper, lower, pct_b, bandwidth


def compute_atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average True Range."""
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


def compute_stochastic(high: pd.Series, low: pd.Series, close: pd.Series,
                       k_window: int = 14, d_window: int = 3) -> tuple[pd.Series, pd.Series]:
    """Stochastic %K y %D."""
    lowest_low = low.rolling(k_window).min()
    highest_high = high.rolling(k_window).max()
    denom = (highest_high - lowest_low).replace(0, np.nan)
    k = 100 * (close - lowest_low) / denom
    d = k.rolling(d_window).mean()
    return k, d


def compute_adx(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """Average Directional Index (simplificado)."""
    prev_high = high.shift(1)
    prev_low = low.shift(1)
    prev_close = close.shift(1)

    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)

    up_move = high - prev_high
    down_move = prev_low - low
    plus_dm = np.where((up_move > down_move) & (up_move > 0), up_move, 0.0)
    minus_dm = np.where((down_move > up_move) & (down_move > 0), down_move, 0.0)

    plus_dm_s = pd.Series(plus_dm, index=high.index).rolling(window).mean()
    minus_dm_s = pd.Series(minus_dm, index=high.index).rolling(window).mean()
    atr = tr.rolling(window).mean()

    plus_di = 100 * plus_dm_s / atr.replace(0, np.nan)
    minus_di = 100 * minus_dm_s / atr.replace(0, np.nan)
    dx = 100 * (plus_di - minus_di).abs() / (plus_di + minus_di).replace(0, np.nan)
    adx = dx.rolling(window).mean()
    return adx


def compute_obv(close: pd.Series, volume: pd.Series) -> pd.Series:
    """On Balance Volume."""
    direction = np.sign(close.diff()).fillna(0)
    return (direction * volume).cumsum()


# ---------------------------------------------------------------------------
# Candlestick patterns
# ---------------------------------------------------------------------------

def add_candlestick_features(data: pd.DataFrame) -> tuple[pd.DataFrame, list[str]]:
    """Métricas y patrones de velas."""
    out = data.copy()

    out["Candle_Body"] = (out["Close"] - out["Open"]).abs()
    out["Candle_Range"] = (out["High"] - out["Low"]).replace(0, np.nan)
    out["Upper_Wick"] = out["High"] - out[["Open", "Close"]].max(axis=1)
    out["Lower_Wick"] = out[["Open", "Close"]].min(axis=1) - out["Low"]
    out["Body_Range_Ratio"] = out["Candle_Body"] / out["Candle_Range"]
    out["Wick_Imbalance"] = (out["Lower_Wick"] - out["Upper_Wick"]) / out["Candle_Range"]

    bullish = out["Close"] > out["Open"]
    bearish = out["Close"] < out["Open"]
    prev_open = out["Open"].shift(1)
    prev_close = out["Close"].shift(1)
    prev_bullish = prev_close > prev_open
    prev_bearish = prev_close < prev_open

    out["Doji"] = (out["Body_Range_Ratio"] <= 0.10).astype(int)
    out["Hammer"] = (
        (out["Lower_Wick"] >= 2.0 * out["Candle_Body"])
        & (out["Upper_Wick"] <= out["Candle_Body"])
        & (out["Body_Range_Ratio"] <= 0.45)
    ).astype(int)
    out["Shooting_Star"] = (
        (out["Upper_Wick"] >= 2.0 * out["Candle_Body"])
        & (out["Lower_Wick"] <= out["Candle_Body"])
        & (out["Body_Range_Ratio"] <= 0.45)
    ).astype(int)
    out["Bullish_Engulfing"] = (
        prev_bearish & bullish
        & (out["Open"] <= prev_close)
        & (out["Close"] >= prev_open)
    ).astype(int)
    out["Bearish_Engulfing"] = (
        prev_bullish & bearish
        & (out["Open"] >= prev_close)
        & (out["Close"] <= prev_open)
    ).astype(int)

    candle_feature_cols = [
        "Candle_Body", "Upper_Wick", "Lower_Wick",
        "Body_Range_Ratio", "Wick_Imbalance",
        "Doji", "Hammer", "Shooting_Star",
        "Bullish_Engulfing", "Bearish_Engulfing",
    ]
    return out, candle_feature_cols


# ---------------------------------------------------------------------------
# Builder principal
# ---------------------------------------------------------------------------

def build_features(df: pd.DataFrame, horizon_days: int | None = None) -> tuple[pd.DataFrame, list[str], str]:
    """Genera features avanzados y target binario."""
    horizon_days = horizon_days or config.PREDICTION_HORIZON_DAYS
    data = df.copy()

    # --- Retornos y volatilidad base ---
    data["Return"] = data["Close"].pct_change()
    data["Volatility"] = data["Return"].rolling(config.VOLATILITY_WINDOW).std()

    # --- SMAs ---
    data["SMA_10"] = data["Close"].rolling(10).mean()
    data["SMA_20"] = data["Close"].rolling(20).mean()
    data["SMA_50"] = data["Close"].rolling(50).mean()

    # --- RSI ---
    data["RSI"] = compute_rsi(data["Close"], window=config.RSI_WINDOW)

    # --- MACD ---
    data["MACD"], data["MACD_Signal"], data["MACD_Hist"] = compute_macd(data["Close"])

    # --- Bollinger Bands ---
    data["BB_Upper"], data["BB_Lower"], data["BB_PctB"], data["BB_Bandwidth"] = compute_bollinger_bands(data["Close"])

    # --- ATR ---
    data["ATR"] = compute_atr(data["High"], data["Low"], data["Close"])

    # --- Stochastic ---
    data["Stoch_K"], data["Stoch_D"] = compute_stochastic(data["High"], data["Low"], data["Close"])

    # --- ADX ---
    data["ADX"] = compute_adx(data["High"], data["Low"], data["Close"])

    # --- OBV ---
    data["OBV"] = compute_obv(data["Close"], data["Volume"])
    data["OBV_SMA20"] = data["OBV"].rolling(20).mean()
    data["OBV_Momentum"] = data["OBV"] - data["OBV_SMA20"]

    # --- Rate of Change ---
    for period in [5, 10, 20]:
        data[f"ROC_{period}"] = data["Close"].pct_change(period)

    # --- Lag features (retornos pasados) ---
    for lag in [1, 2, 3, 5]:
        data[f"Return_Lag{lag}"] = data["Return"].shift(lag)

    # --- Rolling stats de retornos ---
    for win in [5, 10, 20]:
        data[f"Return_Mean_{win}"] = data["Return"].rolling(win).mean()
        data[f"Return_Std_{win}"] = data["Return"].rolling(win).std()

    # --- Volume features ---
    vol_sma20 = data["Volume"].rolling(20).mean()
    data["Volume_Ratio"] = data["Volume"] / vol_sma20.replace(0, np.nan)
    data["Volume_Change"] = data["Volume"].pct_change()

    # --- Price position ---
    data["Dist_SMA50"] = (data["Close"] - data["SMA_50"]) / data["SMA_50"].replace(0, np.nan)
    data["Dist_High20"] = (data["Close"] - data["High"].rolling(20).max()) / data["Close"].replace(0, np.nan)
    data["Dist_Low20"] = (data["Close"] - data["Low"].rolling(20).min()) / data["Close"].replace(0, np.nan)

    # --- Efecto calendario (día de la semana) ---
    data["DayOfWeek"] = data.index.dayofweek

    # --- Retornos de contexto macro/global ---
    context_close_cols = [
        col for col in data.columns
        if col.endswith("_Close") and col not in {"Close"}
    ]
    context_return_cols: list[str] = []
    for col in context_close_cols:
        ret_col = f"{col}_Ret1"
        data[ret_col] = data[col].pct_change()
        context_return_cols.append(ret_col)

    # --- Candlestick patterns ---
    data, candle_feature_cols = add_candlestick_features(data)

    # --- Target ---
    data["Forward_Return"] = data["Close"].shift(-horizon_days) / data["Close"] - 1
    data["Target"] = (data["Forward_Return"] > 0).astype(int)

    # Agregar retorno relativo vs IPSA como feature (no como target)
    extra_features: list[str] = []
    if "IPSA_Close" in data.columns:
        data["IPSA_Forward_Return"] = data["IPSA_Close"].shift(-horizon_days) / data["IPSA_Close"] - 1
        data["Relative_Return_vs_IPSA"] = data["Return"] - data["IPSA_Close"].pct_change()
        extra_features.append("Relative_Return_vs_IPSA")

    # --- Columnas de features ---
    feature_cols = [
        # Precio base
        "Open", "High", "Low", "Close", "Volume",
        # Macro
        "IPSA_Close", "USDCLP_Close", "COPPER_Close",
        # SMAs
        "SMA_10", "SMA_20", "SMA_50",
        # Momentum
        "RSI", "MACD", "MACD_Signal", "MACD_Hist",
        "Stoch_K", "Stoch_D", "ADX",
        # Volatilidad
        "Volatility", "ATR",
        "BB_Upper", "BB_Lower", "BB_PctB", "BB_Bandwidth",
        # Volumen avanzado
        "OBV", "OBV_SMA20", "OBV_Momentum",
        "Volume_Ratio", "Volume_Change",
        # Retornos y momentum
        "Return",
        "ROC_5", "ROC_10", "ROC_20",
        "Return_Lag1", "Return_Lag2", "Return_Lag3", "Return_Lag5",
        # Rolling stats
        "Return_Mean_5", "Return_Mean_10", "Return_Mean_20",
        "Return_Std_5", "Return_Std_10", "Return_Std_20",
        # Price position
        "Dist_SMA50", "Dist_High20", "Dist_Low20",
        # Calendario
        "DayOfWeek",
    ] + context_close_cols + context_return_cols + candle_feature_cols + extra_features

    target_col = "Target"
    # Filtrar solo features que realmente existen en el dataframe
    feature_cols = [c for c in feature_cols if c in data.columns]
    # Limpiar infinitos y NaN
    for c in feature_cols:
        data[c] = data[c].replace([np.inf, -np.inf], np.nan)
    data = data.dropna(subset=feature_cols + [target_col, "Forward_Return"]).copy()
    return data, feature_cols, target_col
