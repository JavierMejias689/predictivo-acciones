"""Walk-forward backtest para evaluar confiabilidad fuera de muestra.

Uso:
  python walk_forward_backtest.py
  python walk_forward_backtest.py --ticker SQM-B.SN --threshold 0.60 --cost-bps 15
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score

import config
from data_loader import load_market_data
from feature_selector import select_features
from features import build_features
from model import TrendClassifier


@dataclass
class WFConfig:
    train_window: int
    test_window: int
    threshold: float
    cost_bps: float
    non_overlapping: bool


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


def max_drawdown(equity_curve: pd.Series) -> float:
    running_max = equity_curve.cummax()
    drawdown = equity_curve / running_max - 1.0
    return float(drawdown.min())


def sharpe_ratio(returns: pd.Series, annual_factor: float = 252.0) -> float:
    if returns.std(ddof=0) == 0:
        return 0.0
    return float((returns.mean() / returns.std(ddof=0)) * np.sqrt(annual_factor))


def walk_forward_predict(
    data: pd.DataFrame,
    feature_cols: list[str],
    target_col: str,
    wf_cfg: WFConfig,
) -> pd.DataFrame:
    """Genera predicciones walk-forward:
    - Entrena en ventana histórica fija (rolling).
    - Predice en bloque siguiente.
    """
    rows = []
    n = len(data)

    start_test = wf_cfg.train_window
    while start_test < n:
        end_test = min(start_test + wf_cfg.test_window, n)

        train_slice = data.iloc[start_test - wf_cfg.train_window : start_test]
        test_slice = data.iloc[start_test:end_test]
        if train_slice.empty or test_slice.empty:
            break

        X_train_raw = train_slice[feature_cols]
        y_train = train_slice[target_col]

        # Feature selection por ventana
        selected = select_features(X_train_raw, y_train, n_top=config.N_TOP_FEATURES)
        X_train = X_train_raw[selected]
        X_test = test_slice[selected]
        y_test = test_slice[target_col]

        clf = TrendClassifier(config.MODEL_TYPE)
        clf.fit(X_train, y_train)

        probs = clf.predict_proba(X_test)
        prob_up = probs[:, 1]
        y_pred = (prob_up >= wf_cfg.threshold).astype(int)

        chunk = pd.DataFrame(
            {
                "Date": test_slice.index,
                "Close": test_slice["Close"].values,
                "Forward_Return": test_slice["Forward_Return"].values,
                "y_true": y_test.values,
                "prob_up": prob_up,
                "y_pred": y_pred,
            }
        ).set_index("Date")
        rows.append(chunk)
        start_test = end_test

    if not rows:
        return pd.DataFrame()
    out = pd.concat(rows).sort_index()
    out = out[~out.index.duplicated(keep="first")]
    return out


def apply_strategy_returns(pred_df: pd.DataFrame, wf_cfg: WFConfig) -> pd.DataFrame:
    """Aplica estrategia long/cash con costo por cambio de posición."""
    bt = pred_df.copy()

    if wf_cfg.non_overlapping:
        step = max(config.PREDICTION_HORIZON_DAYS, 1)
        bt = bt.iloc[::step].copy()

    bt["position"] = bt["y_pred"].astype(float)
    one_way_cost = wf_cfg.cost_bps / 10000.0
    bt["turnover"] = bt["position"].diff().abs().fillna(bt["position"])

    bt["strategy_return_gross"] = bt["position"] * bt["Forward_Return"]
    bt["strategy_return_net"] = bt["strategy_return_gross"] - (bt["turnover"] * one_way_cost)
    bt["buy_hold_return"] = bt["Forward_Return"]

    bt["equity_curve"] = (1 + bt["strategy_return_net"]).cumprod()
    bt["buy_hold_curve"] = (1 + bt["buy_hold_return"]).cumprod()
    return bt


def evaluate_backtest(bt: pd.DataFrame) -> dict[str, float]:
    if bt.empty:
        return {
            "accuracy": 0.0,
            "precision": 0.0,
            "recall": 0.0,
            "cumulative_return_net": 0.0,
            "buy_hold_return": 0.0,
            "max_drawdown": 0.0,
            "sharpe": 0.0,
            "n_trades": 0.0,
        }

    return {
        "accuracy": float(accuracy_score(bt["y_true"], bt["y_pred"])),
        "precision": float(precision_score(bt["y_true"], bt["y_pred"], zero_division=0)),
        "recall": float(recall_score(bt["y_true"], bt["y_pred"], zero_division=0)),
        "cumulative_return_net": float(bt["equity_curve"].iloc[-1] - 1),
        "buy_hold_return": float(bt["buy_hold_curve"].iloc[-1] - 1),
        "max_drawdown": max_drawdown(bt["equity_curve"]),
        "sharpe": sharpe_ratio(bt["strategy_return_net"]),
        "n_trades": float(bt["turnover"].sum()),
    }


def plot_walk_forward_equity(bt: pd.DataFrame, ticker: str, output_path: str) -> None:
    if bt.empty:
        return
    plt.figure(figsize=(12, 6))
    plt.plot(bt.index, bt["equity_curve"], label="Estrategia neta (walk-forward)", linewidth=2)
    plt.plot(bt.index, bt["buy_hold_curve"], label="Buy & Hold", linestyle="--", linewidth=1.4)
    plt.title(f"{ticker} - Walk-Forward Equity Curve")
    plt.xlabel("Fecha")
    plt.ylabel("Equity")
    plt.grid(alpha=0.2)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


def run_one_ticker(ticker: str, wf_cfg: WFConfig) -> dict[str, float | str]:
    raw_df = load_market_data(ticker)
    feat_df, feature_cols, target_col = build_features(
        raw_df, horizon_days=config.PREDICTION_HORIZON_DAYS
    )

    pred_df = walk_forward_predict(
        data=feat_df,
        feature_cols=feature_cols,
        target_col=target_col,
        wf_cfg=wf_cfg,
    )
    bt = apply_strategy_returns(pred_df, wf_cfg)
    metrics = evaluate_backtest(bt)

    safe_ticker = sanitize_ticker(ticker)
    bt_path = config.REPORTS_DIR / f"{safe_ticker}_walk_forward_trades.csv"
    bt.to_csv(bt_path, encoding="utf-8")
    plot_walk_forward_equity(
        bt,
        ticker=ticker,
        output_path=str(config.PLOTS_DIR / f"{safe_ticker}_walk_forward_equity.png"),
    )

    report = {
        "ticker": ticker,
        "horizon_days": config.PREDICTION_HORIZON_DAYS,
        "train_window": wf_cfg.train_window,
        "test_window": wf_cfg.test_window,
        "threshold": wf_cfg.threshold,
        "cost_bps": wf_cfg.cost_bps,
        "non_overlapping": wf_cfg.non_overlapping,
        "n_samples_eval": int(len(bt)),
        **metrics,
        "trades_path": str(bt_path),
    }
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Walk-forward backtest de señales alcista/bajista.")
    parser.add_argument(
        "--ticker",
        type=str,
        default="ALL",
        help="Ticker específico (ej: SQM-B.SN) o ALL para todos los de config.",
    )
    parser.add_argument("--train-window", type=int, default=756, help="Ventana de entrenamiento (días).")
    parser.add_argument("--test-window", type=int, default=63, help="Bloque de prueba por iteración (días).")
    parser.add_argument(
        "--threshold",
        type=float,
        default=config.PREDICTION_THRESHOLD,
        help="Umbral de probabilidad alcista para tomar posición.",
    )
    parser.add_argument(
        "--cost-bps",
        type=float,
        default=10.0,
        help="Costo one-way por cambio de posición en basis points (bps).",
    )
    parser.add_argument(
        "--allow-overlap",
        action="store_true",
        help="Permite operaciones superpuestas (menos realista para horizonte >1).",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    wf_cfg = WFConfig(
        train_window=args.train_window,
        test_window=args.test_window,
        threshold=args.threshold,
        cost_bps=args.cost_bps,
        non_overlapping=not args.allow_overlap,
    )

    tickers = config.TICKERS if args.ticker.upper() == "ALL" else [args.ticker.upper()]
    reports: list[dict[str, float | str]] = []

    print("=== Walk-Forward Backtest ===")
    print(
        f"Horizonte={config.PREDICTION_HORIZON_DAYS}d | "
        f"train_window={wf_cfg.train_window} | test_window={wf_cfg.test_window} | "
        f"threshold={wf_cfg.threshold:.2f} | cost={wf_cfg.cost_bps:.1f} bps | "
        f"non_overlapping={wf_cfg.non_overlapping}"
    )

    for ticker in tickers:
        try:
            rep = run_one_ticker(ticker, wf_cfg)
            reports.append(rep)
            print(
                f"[OK] {ticker}: acc={rep['accuracy']:.4f} prec={rep['precision']:.4f} "
                f"rec={rep['recall']:.4f} ret_net={rep['cumulative_return_net']:.2%} "
                f"mdd={rep['max_drawdown']:.2%} sharpe={rep['sharpe']:.2f}"
            )
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    if not reports:
        raise RuntimeError("No se pudo evaluar ningún ticker en walk-forward.")

    summary = pd.DataFrame(reports).sort_values("cumulative_return_net", ascending=False)
    summary_path = config.OUTPUT_DIR / "walk_forward_summary.csv"
    summary.to_csv(summary_path, index=False, encoding="utf-8")
    print(f"\nResumen walk-forward guardado en: {summary_path}")


if __name__ == "__main__":
    main()
