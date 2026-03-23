"""Carga de datos de mercado y macro desde Yahoo Finance."""

from __future__ import annotations

from typing import Dict

import pandas as pd
import yfinance as yf

import config


def _normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Normaliza columnas OHLCV tras descarga."""
    if isinstance(df.columns, pd.MultiIndex):
        df.columns = df.columns.get_level_values(0)
    keep_cols = ["Open", "High", "Low", "Close", "Volume"]
    df = df[[c for c in keep_cols if c in df.columns]].copy()
    return df


def download_ohlcv(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Descarga serie OHLCV para un ticker."""
    start = start or config.START_DATE
    end = end or config.END_DATE
    interval = interval or config.INTERVAL

    data = yf.download(
        ticker,
        start=start,
        end=end,
        interval=interval,
        auto_adjust=False,
        progress=False,
        threads=False,
    )
    if data.empty:
        raise ValueError(f"No se pudo descargar información para {ticker}")

    data = _normalize_columns(data)
    data.index = pd.to_datetime(data.index).tz_localize(None)
    data = data.sort_index()
    return data


def download_close_data(
    tickers: Dict[str, str],
    start: str | None = None,
    end: str | None = None,
    interval: str | None = None,
) -> pd.DataFrame:
    """Descarga un diccionario de tickers y retorna sus cierres diarios."""
    frames = []
    failed = []
    for name, yahoo_ticker in tickers.items():
        try:
            data_df = download_ohlcv(
                ticker=yahoo_ticker,
                start=start,
                end=end,
                interval=interval,
            )
            close = data_df[["Close"]].rename(columns={"Close": f"{name}_Close"})
            frames.append(close)
        except Exception:
            failed.append(yahoo_ticker)
            continue

    if not frames:
        raise ValueError(f"No se pudieron descargar tickers de contexto: {failed}")

    merged = pd.concat(frames, axis=1).sort_index()
    merged = merged.ffill()
    return merged


def load_market_data(
    ticker: str,
    start: str | None = None,
    end: str | None = None,
) -> pd.DataFrame:
    """Une datos accionarios + contexto macro/global por fecha, sin backfill."""
    stock_df = download_ohlcv(ticker=ticker, start=start, end=end)
    macro_df = download_close_data(config.MACRO_TICKERS, start=start, end=end)
    global_df = download_close_data(config.CONTEXT_TICKERS, start=start, end=end)

    df = stock_df.join(macro_df, how="left")
    df = df.join(global_df, how="left")
    # Solo forward-fill para no usar datos del futuro (sin backfill).
    df = df.ffill()
    return df
