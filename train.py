"""Entrena modelos por ticker con ensemble y feature selection."""

from __future__ import annotations

import json
from datetime import datetime

import matplotlib
import numpy as np
import pandas as pd
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score

import config
from backtest import plot_equity_curve, run_backtest
from data_loader import load_market_data
from feature_selector import select_features
from features import build_features
from model import TrendClassifier


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


def find_best_threshold(y_true: pd.Series, prob_up: np.ndarray) -> tuple[float, dict[str, float]]:
    """Selecciona umbral con foco en precision, penalizando cobertura muy baja."""
    best_threshold = config.PREDICTION_THRESHOLD
    best_score = -1.0
    best_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "signal_rate": 0.0}

    for th in config.THRESHOLD_GRID:
        y_pred = (prob_up >= th).astype(int)
        signal_rate = float(np.mean(y_pred))
        if signal_rate < config.MIN_SIGNAL_RATE:
            continue

        prec = float(precision_score(y_true, y_pred, zero_division=0))
        rec = float(recall_score(y_true, y_pred, zero_division=0))
        acc = float(accuracy_score(y_true, y_pred))
        score = (0.40 * acc) + (0.40 * prec) + (0.20 * rec)
        if score > best_score:
            best_score = score
            best_threshold = float(th)
            best_metrics = {
                "accuracy": acc, "precision": prec,
                "recall": rec, "signal_rate": signal_rate,
            }

    if best_score < 0:
        y_pred = (prob_up >= config.PREDICTION_THRESHOLD).astype(int)
        best_metrics = {
            "accuracy": float(accuracy_score(y_true, y_pred)),
            "precision": float(precision_score(y_true, y_pred, zero_division=0)),
            "recall": float(recall_score(y_true, y_pred, zero_division=0)),
            "signal_rate": float(np.mean(y_pred)),
        }
        best_threshold = config.PREDICTION_THRESHOLD

    return best_threshold, best_metrics


# ---------------------------------------------------------------------------
# Plots
# ---------------------------------------------------------------------------

def plot_historical_price(df: pd.DataFrame, output_path: str, ticker: str) -> None:
    plt.figure(figsize=(12, 6))
    plt.plot(df.index, df["Close"], linewidth=1.6)
    plt.title(f"{ticker} - Precio Histórico (Close)")
    plt.xlabel("Fecha")
    plt.ylabel("Precio")
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


def plot_pred_vs_real(y_true: pd.Series, y_pred: pd.Series, output_path: str, ticker: str) -> None:
    plt.figure(figsize=(12, 5))
    plt.plot(y_true.index, y_true.values, label="Real", linewidth=1.4)
    plt.plot(y_pred.index, y_pred.values, label="Predicción", linewidth=1.2, alpha=0.9)
    plt.title(f"{ticker} - Predicción vs Realidad (Test)")
    plt.xlabel("Fecha")
    plt.ylabel("Clase (0/1)")
    plt.legend()
    plt.grid(alpha=0.2)
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


def plot_feature_importance(clf: TrendClassifier, feature_cols: list[str], output_path: str, ticker: str) -> None:
    fi = clf.get_feature_importances(feature_cols)
    if fi.empty or fi["importance"].sum() == 0:
        return

    fi = fi.head(25)
    plt.figure(figsize=(10, 7))
    plt.barh(fi["feature"], fi["importance"])
    plt.gca().invert_yaxis()
    plt.title(f"{ticker} - Importancia de Variables (Top 25)")
    plt.xlabel("Importancia")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


def plot_candlestick_with_signals(
    df: pd.DataFrame, y_pred: pd.Series, output_path: str,
    ticker: str, horizon_days: int, max_bars: int = 180,
) -> None:
    plot_df = df.loc[y_pred.index].copy()
    if plot_df.empty:
        return
    plot_df["Prediction"] = y_pred
    plot_df = plot_df.tail(max_bars).copy()
    if plot_df.empty:
        return

    x = np.arange(len(plot_df))
    bull = plot_df["Close"] >= plot_df["Open"]
    body_bottom = np.where(bull, plot_df["Open"], plot_df["Close"])
    body_height = (plot_df["Close"] - plot_df["Open"]).abs()

    plt.figure(figsize=(14, 7))
    ax = plt.gca()
    for i, (_, row) in enumerate(plot_df.iterrows()):
        ax.plot([x[i], x[i]], [row["Low"], row["High"]], color="#4c566a", linewidth=1.0, zorder=1)

    colors = np.where(bull, "#2ca02c", "#d62728")
    ax.bar(x, body_height, bottom=body_bottom, color=colors, width=0.7, linewidth=0.0, align="center", zorder=2)

    long_idx = plot_df["Prediction"] == 1
    flat_idx = plot_df["Prediction"] == 0
    ax.scatter(x[long_idx], (plot_df.loc[long_idx, "Low"] * 0.995), marker="^", s=35, color="#1f77b4", label="Señal alcista IA", zorder=3)
    ax.scatter(x[flat_idx], (plot_df.loc[flat_idx, "High"] * 1.005), marker="v", s=30, color="#7f7f7f", label="Señal bajista IA", zorder=3)

    bull_pat = (plot_df.get("Hammer", 0) == 1) | (plot_df.get("Bullish_Engulfing", 0) == 1)
    bear_pat = (plot_df.get("Shooting_Star", 0) == 1) | (plot_df.get("Bearish_Engulfing", 0) == 1)
    ax.scatter(x[bull_pat], (plot_df.loc[bull_pat, "Low"] * 0.99), marker="o", s=18, color="#17becf", label="Patrón vela alcista", zorder=4)
    ax.scatter(x[bear_pat], (plot_df.loc[bear_pat, "High"] * 1.01), marker="o", s=18, color="#9467bd", label="Patrón vela bajista", zorder=4)

    tick_step = max(1, len(plot_df) // 12)
    tick_pos = x[::tick_step]
    tick_labels = [d.strftime("%Y-%m-%d") for d in plot_df.index[::tick_step]]
    ax.set_xticks(tick_pos)
    ax.set_xticklabels(tick_labels, rotation=40, ha="right")
    ax.set_title(f"{ticker} - Velas + Señales IA ({horizon_days} días)")
    ax.set_xlabel("Fecha")
    ax.set_ylabel("Precio")
    ax.grid(alpha=0.2)
    ax.legend(loc="best")
    plt.tight_layout()
    plt.savefig(output_path, dpi=130)
    plt.close()


# ---------------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------------

def train_one_ticker(
    ticker: str,
    start_date: str | None = None,
    end_date: str | None = None,
) -> dict:
    raw_df = load_market_data(ticker, start=start_date, end=end_date)
    feat_df, feature_cols, target_col = build_features(raw_df, horizon_days=config.PREDICTION_HORIZON_DAYS)

    # Split temporal
    base_clf = TrendClassifier(config.MODEL_TYPE)
    X_train, X_test, y_train, y_test = base_clf.temporal_train_test_split(
        feat_df, feature_cols, target_col, config.TEST_SIZE
    )

    # Feature selection sobre train
    selected_features = select_features(X_train, y_train, n_top=config.N_TOP_FEATURES)
    X_train = X_train[selected_features]
    X_test = X_test[selected_features]

    # Threshold tuning en validación temporal
    split_val = int(len(X_train) * 0.8)
    X_subtrain, y_subtrain = X_train.iloc[:split_val], y_train.iloc[:split_val]
    X_val, y_val = X_train.iloc[split_val:], y_train.iloc[split_val:]

    threshold = config.PREDICTION_THRESHOLD
    threshold_metrics = {"accuracy": 0.0, "precision": 0.0, "recall": 0.0, "signal_rate": 0.0}

    # Entrenar todos los modelos y seleccionar el mejor por validación
    model_types_to_try = ["random_forest", "xgboost", "lightgbm"]
    best_val_acc = -1.0
    best_model_type = config.MODEL_TYPE
    model_val_accs = {}

    for mt in model_types_to_try:
        try:
            trial_clf = TrendClassifier(mt)
            trial_clf.fit(X_subtrain, y_subtrain)
            val_result = trial_clf.evaluate(X_val, y_val)
            model_val_accs[mt] = val_result.accuracy
            if val_result.accuracy > best_val_acc:
                best_val_acc = val_result.accuracy
                best_model_type = mt
        except Exception:
            model_val_accs[mt] = 0.0

    # Threshold tuning con el mejor modelo
    if len(X_subtrain) > 50 and len(X_val) > 20:
        tune_clf = TrendClassifier(best_model_type)
        tune_clf.fit(X_subtrain, y_subtrain)
        val_prob_up = tune_clf.predict_proba(X_val)[:, 1]
        threshold, threshold_metrics = find_best_threshold(y_val, val_prob_up)

    # Entrenamiento final con el mejor modelo seleccionado
    # Usar ensemble con pesos basados en accuracy de validación
    clf = TrendClassifier("ensemble")
    clf.fit(X_train, y_train)

    # Ajustar pesos: dar mucho más peso al mejor modelo
    total_acc = sum(model_val_accs.values()) or 1.0
    new_weights = {}
    for name, acc in model_val_accs.items():
        # Elevar al cubo para acentuar diferencias
        new_weights[name] = (acc / total_acc) ** 3
    total_w = sum(new_weights.values()) or 1.0
    new_weights = {name: w / total_w for name, w in new_weights.items()}
    clf.set_ensemble_weights(new_weights)

    cv_metrics = clf.time_series_cross_validation(X_train, y_train)

    # Evaluación en test
    test_prob_up = clf.predict_proba(X_test)[:, 1]
    y_pred_arr = (test_prob_up >= threshold).astype(int)
    y_pred = pd.Series(y_pred_arr, index=y_test.index, name="Prediction")

    accuracy_test = float(accuracy_score(y_test, y_pred_arr))
    precision_test = float(precision_score(y_test, y_pred_arr, zero_division=0))
    recall_test = float(recall_score(y_test, y_pred_arr, zero_division=0))
    cm_test = confusion_matrix(y_test, y_pred_arr)

    # Sub-model metrics
    sub_metrics = {}
    sub_results = clf.evaluate_sub_models(X_test, y_test)
    for name, res in sub_results.items():
        sub_metrics[name] = {
            "accuracy": res.accuracy, "precision": res.precision,
            "recall": res.recall, "weight": new_weights.get(name, 0),
            "val_accuracy": model_val_accs.get(name, 0),
        }

    # Backtest
    backtest_df, bt_metrics = run_backtest(
        df_test=feat_df.loc[X_test.index],
        y_true=y_test, y_pred=y_pred,
        horizon_days=config.PREDICTION_HORIZON_DAYS,
    )

    # Save model
    safe_ticker = sanitize_ticker(ticker)
    model_path = config.MODELS_DIR / f"{safe_ticker}.joblib"
    clf.save(
        str(model_path),
        metadata={
            "ticker": ticker,
            "feature_cols": selected_features,
            "target_col": target_col,
            "trained_at": datetime.utcnow().isoformat(),
            "threshold": threshold,
            "horizon_days": config.PREDICTION_HORIZON_DAYS,
            "model_type": config.MODEL_TYPE,
            "n_features_selected": len(selected_features),
        },
    )

    # Plots
    plot_historical_price(raw_df, str(config.PLOTS_DIR / f"{safe_ticker}_price.png"), ticker)
    plot_pred_vs_real(y_test, y_pred, str(config.PLOTS_DIR / f"{safe_ticker}_pred_vs_real.png"), ticker)
    plot_feature_importance(clf, selected_features, str(config.PLOTS_DIR / f"{safe_ticker}_feature_importance.png"), ticker)
    plot_candlestick_with_signals(
        feat_df, y_pred, str(config.PLOTS_DIR / f"{safe_ticker}_candles_signals.png"),
        ticker=ticker, horizon_days=config.PREDICTION_HORIZON_DAYS,
    )
    plot_equity_curve(
        backtest_df, str(config.PLOTS_DIR / f"{safe_ticker}_equity_curve.png"),
        title=f"{ticker} - Equity Curve (Test, horizonte {config.PREDICTION_HORIZON_DAYS}d)",
    )

    # Report
    report = {
        "ticker": ticker,
        "train_start_date": start_date or config.START_DATE,
        "train_end_date": end_date,
        "model_type": config.MODEL_TYPE,
        "horizon_days": config.PREDICTION_HORIZON_DAYS,
        "n_samples": int(len(feat_df)),
        "train_samples": int(len(X_train)),
        "test_samples": int(len(X_test)),
        "n_features": len(selected_features),
        "selected_features": selected_features,
        "selected_threshold": threshold,
        "threshold_tuning_metrics": threshold_metrics,
        "accuracy_test": accuracy_test,
        "precision_test": precision_test,
        "recall_test": recall_test,
        "confusion_matrix": cm_test.tolist(),
        "sub_model_metrics": sub_metrics,
        "backtest_accuracy": bt_metrics["accuracy"],
        "backtest_precision": bt_metrics["precision"],
        "backtest_cumulative_return": bt_metrics["cumulative_return"],
        "backtest_buy_hold_return": bt_metrics["buy_hold_return"],
        **cv_metrics,
        "model_path": str(model_path),
    }
    report_path = config.REPORTS_DIR / f"{safe_ticker}_metrics.json"
    with open(report_path, "w", encoding="utf-8") as f:
        json.dump(report, f, ensure_ascii=False, indent=2)
    return report


def main() -> None:
    reports = []
    print("=== Entrenamiento de modelos (Bolsa de Santiago) ===")
    print(f"    Modelo: {config.MODEL_TYPE}")
    print(f"    Features máximo: {config.N_TOP_FEATURES}")
    print(f"    Horizonte: {config.PREDICTION_HORIZON_DAYS} días")
    print()

    for ticker in config.TICKERS:
        try:
            report = train_one_ticker(ticker)
            reports.append(report)
            line = (
                f"[OK] {ticker}: acc={report['accuracy_test']:.4f} "
                f"prec={report['precision_test']:.4f} "
                f"rec={report['recall_test']:.4f} "
                f"ret={report['backtest_cumulative_return']:.2%} "
                f"features={report['n_features']}"
            )
            if report.get("sub_model_metrics"):
                for name, m in report["sub_model_metrics"].items():
                    line += f"\n      {name}: acc={m['accuracy']:.4f}"
            print(line)
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    if not reports:
        raise RuntimeError("No se pudo entrenar ningún ticker.")

    summary_df = pd.DataFrame([
        {k: v for k, v in r.items() if k not in ("selected_features", "confusion_matrix", "sub_model_metrics", "threshold_tuning_metrics")}
        for r in reports
    ]).sort_values("accuracy_test", ascending=False)
    summary_path = config.OUTPUT_DIR / "training_summary.csv"
    summary_df.to_csv(summary_path, index=False, encoding="utf-8")

    avg_acc = summary_df["accuracy_test"].mean()
    print(f"\n{'='*50}")
    print(f"Accuracy promedio: {avg_acc:.4f} ({avg_acc:.1%})")
    print(f"Resumen: {summary_path}")
    print(f"Modelos: {config.MODELS_DIR}")
    print(f"Gráficos: {config.PLOTS_DIR}")
    print(f"Reportes: {config.REPORTS_DIR}")


if __name__ == "__main__":
    main()
