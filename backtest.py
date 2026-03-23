"""Backtesting simple de señales de clasificación."""

from __future__ import annotations

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score


def run_backtest(
    df_test: pd.DataFrame,
    y_true: pd.Series,
    y_pred: pd.Series,
    horizon_days: int = 1,
) -> tuple[pd.DataFrame, dict[str, float]]:
    """Estrategia:
    - Si pred=1 => comprar para el horizonte objetivo.
    - Si pred=0 => fuera del mercado.
    """
    bt = df_test.copy()
    bt["y_true"] = y_true
    bt["y_pred"] = y_pred

    bt["forward_return"] = bt["Close"].shift(-horizon_days) / bt["Close"] - 1
    bt = bt.dropna(subset=["forward_return"]).copy()

    bt["strategy_return"] = bt["y_pred"] * bt["forward_return"]
    bt["buy_hold_return"] = bt["forward_return"]

    bt["equity_curve"] = (1 + bt["strategy_return"]).cumprod()
    bt["buy_hold_curve"] = (1 + bt["buy_hold_return"]).cumprod()

    metrics = {
        "accuracy": float(accuracy_score(bt["y_true"], bt["y_pred"])),
        "precision": float(precision_score(bt["y_true"], bt["y_pred"], zero_division=0)),
        "cumulative_return": float(bt["equity_curve"].iloc[-1] - 1),
        "buy_hold_return": float(bt["buy_hold_curve"].iloc[-1] - 1),
    }
    return bt, metrics


def plot_equity_curve(bt: pd.DataFrame, output_path: str, title: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(bt.index, bt["equity_curve"], label="Estrategia (IA)", linewidth=2)
    plt.plot(bt.index, bt["buy_hold_curve"], label="Buy & Hold", linestyle="--")
    plt.title(title)
    plt.xlabel("Fecha")
    plt.ylabel("Equity")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()
