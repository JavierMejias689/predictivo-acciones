"""Motor de aprendizaje continuo: registra predicciones, evalúa resultados,
ajusta thresholds y pesos del ensemble, detecta drift y genera log de
pensamiento auditables.

Uso:
    from learning_engine import LearningEngine
    engine = LearningEngine()
    engine.record_prediction(...)
    engine.evaluate_past_predictions()
    adjustments = engine.compute_adjustments("SQM-B.SN")
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

import config


HISTORY_CSV = config.HISTORY_DIR / "prediction_history.csv"
LEARNING_LOG = config.HISTORY_DIR / "learning_log.json"
WEIGHTS_FILE = config.HISTORY_DIR / "ensemble_weights.json"

HISTORY_COLUMNS = [
    "timestamp", "ticker", "signal_date", "horizon_days",
    "prob_up", "prob_down", "threshold", "recommendation",
    "actual_return", "actual_direction", "was_correct",
    "model_type",
]


class LearningEngine:
    """Motor de aprendizaje continuo que registra, evalúa y ajusta."""

    def __init__(self) -> None:
        self._history = self._load_history()
        self._log: list[dict] = self._load_log()

    # ------------------------------------------------------------------
    # Persistence helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _load_history() -> pd.DataFrame:
        if HISTORY_CSV.exists():
            df = pd.read_csv(HISTORY_CSV, parse_dates=["timestamp", "signal_date"])
            return df
        return pd.DataFrame(columns=HISTORY_COLUMNS)

    @staticmethod
    def _load_log() -> list[dict]:
        if LEARNING_LOG.exists():
            with open(LEARNING_LOG, "r", encoding="utf-8") as f:
                return json.load(f)
        return []

    def _save_history(self) -> None:
        self._history.to_csv(HISTORY_CSV, index=False, encoding="utf-8")

    def _save_log(self) -> None:
        with open(LEARNING_LOG, "w", encoding="utf-8") as f:
            json.dump(self._log, f, ensure_ascii=False, indent=2, default=str)

    def _add_log_entry(self, action: str, ticker: str, detail: str, data: dict | None = None) -> None:
        entry = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "action": action,
            "ticker": ticker,
            "detail": detail,
            "data": data or {},
        }
        self._log.append(entry)
        self._save_log()

    # ------------------------------------------------------------------
    # Recording predictions
    # ------------------------------------------------------------------

    def record_prediction(
        self,
        ticker: str,
        signal_date: str,
        prob_up: float,
        prob_down: float,
        threshold: float,
        recommendation: str,
        model_type: str = "ensemble",
    ) -> None:
        """Registra una nueva predicción en el histórico."""
        row = {
            "timestamp": datetime.now().isoformat(timespec="seconds"),
            "ticker": ticker,
            "signal_date": pd.Timestamp(signal_date),
            "horizon_days": config.PREDICTION_HORIZON_DAYS,
            "prob_up": prob_up,
            "prob_down": prob_down,
            "threshold": threshold,
            "recommendation": recommendation,
            "actual_return": np.nan,
            "actual_direction": np.nan,
            "was_correct": np.nan,
            "model_type": model_type,
        }
        self._history = pd.concat([self._history, pd.DataFrame([row])], ignore_index=True)
        self._save_history()

    # ------------------------------------------------------------------
    # Evaluating past predictions
    # ------------------------------------------------------------------

    def evaluate_past_predictions(self, price_data: dict[str, pd.DataFrame] | None = None) -> int:
        """Evalúa predicciones pasadas cuyo horizonte ya venció.

        Args:
            price_data: dict ticker -> DataFrame con columna Close indexada por fecha.
                        Si None, intenta descargar automáticamente.

        Returns:
            Número de predicciones evaluadas en esta pasada.
        """
        if price_data is None:
            price_data = self._download_prices()

        evaluated = 0
        mask = self._history["was_correct"].isna()
        for idx in self._history[mask].index:
            row = self._history.loc[idx]
            ticker = row["ticker"]
            signal_date = pd.Timestamp(row["signal_date"])
            horizon = int(row["horizon_days"])

            if ticker not in price_data:
                continue

            prices = price_data[ticker]
            target_date = signal_date + pd.Timedelta(days=horizon)

            # Buscar precio más cercano a signal_date y target_date
            if signal_date not in prices.index:
                candidates = prices.index[prices.index <= signal_date]
                if candidates.empty:
                    continue
                signal_date = candidates[-1]

            future_candidates = prices.index[prices.index >= target_date]
            if future_candidates.empty:
                continue  # aún no hay datos futuros
            actual_date = future_candidates[0]

            price_signal = float(prices.loc[signal_date, "Close"])
            price_actual = float(prices.loc[actual_date, "Close"])
            actual_return = (price_actual / price_signal) - 1
            actual_direction = 1 if actual_return > 0 else 0
            predicted_direction = 1 if row["recommendation"] == "ALCISTA" else 0
            was_correct = int(predicted_direction == actual_direction)

            self._history.at[idx, "actual_return"] = actual_return
            self._history.at[idx, "actual_direction"] = actual_direction
            self._history.at[idx, "was_correct"] = was_correct
            evaluated += 1

        if evaluated > 0:
            self._save_history()
            self._add_log_entry(
                "evaluate", "ALL",
                f"Evaluadas {evaluated} predicciones pasadas.",
                {"evaluated_count": evaluated},
            )
        return evaluated

    def _download_prices(self) -> dict[str, pd.DataFrame]:
        """Descarga precios recientes para evaluar predicciones."""
        from data_loader import download_ohlcv
        result = {}
        for ticker in config.TICKERS:
            try:
                df = download_ohlcv(ticker)
                result[ticker] = df
            except Exception:
                pass
        return result

    # ------------------------------------------------------------------
    # Analytics
    # ------------------------------------------------------------------

    def get_rolling_accuracy(self, ticker: str, window: int = 30) -> float | None:
        """Accuracy rolling de las últimas N predicciones evaluadas para un ticker."""
        evaluated = self._history[
            (self._history["ticker"] == ticker) & self._history["was_correct"].notna()
        ].tail(window)
        if len(evaluated) < config.MIN_HISTORY_FOR_ADJUSTMENT:
            return None
        return float(evaluated["was_correct"].mean())

    def get_calibration_error(self, ticker: str, window: int = 60) -> float | None:
        """Error de calibración: diferencia entre prob_up promedio y tasa real de aciertos alcistas."""
        evaluated = self._history[
            (self._history["ticker"] == ticker) & self._history["was_correct"].notna()
        ].tail(window)
        if len(evaluated) < config.MIN_HISTORY_FOR_ADJUSTMENT:
            return None

        avg_prob_up = float(evaluated["prob_up"].mean())
        actual_up_rate = float(evaluated["actual_direction"].mean())
        return avg_prob_up - actual_up_rate

    def get_ticker_stats(self, ticker: str) -> dict:
        """Estadísticas completas de un ticker."""
        evaluated = self._history[
            (self._history["ticker"] == ticker) & self._history["was_correct"].notna()
        ]
        total = len(evaluated)
        if total == 0:
            return {"total": 0, "accuracy": None, "windows": {}}

        stats: dict = {
            "total": total,
            "accuracy": float(evaluated["was_correct"].mean()),
            "avg_prob_up": float(evaluated["prob_up"].mean()),
            "actual_up_rate": float(evaluated["actual_direction"].mean()),
            "windows": {},
        }
        for w in config.LEARNING_ROLLING_WINDOWS:
            acc = self.get_rolling_accuracy(ticker, w)
            if acc is not None:
                stats["windows"][f"acc_{w}d"] = acc
        return stats

    # ------------------------------------------------------------------
    # Adjustments
    # ------------------------------------------------------------------

    def compute_adjustments(self, ticker: str) -> dict:
        """Calcula ajustes de threshold y pesos basados en historial.

        Returns:
            dict con keys: threshold_adjustment, weight_adjustments, should_retrain, reasoning
        """
        result = {
            "threshold_adjustment": 0.0,
            "weight_adjustments": {},
            "should_retrain": False,
            "reasoning": [],
        }

        acc = self.get_rolling_accuracy(ticker, window=30)
        if acc is None:
            result["reasoning"].append(
                f"Insuficiente historial para {ticker} (min {config.MIN_HISTORY_FOR_ADJUSTMENT} evaluaciones)."
            )
            return result

        # Threshold adjustment basado en accuracy reciente
        if acc < 0.45:
            result["threshold_adjustment"] = 0.05
            result["reasoning"].append(
                f"Accuracy 30d={acc:.1%} muy baja — subir threshold +0.05 para filtrar señales débiles."
            )
        elif acc < 0.50:
            result["threshold_adjustment"] = 0.02
            result["reasoning"].append(
                f"Accuracy 30d={acc:.1%} bajo azar — subir threshold +0.02."
            )
        elif acc > 0.65:
            result["threshold_adjustment"] = -0.02
            result["reasoning"].append(
                f"Accuracy 30d={acc:.1%} excelente — bajar threshold -0.02 para capturar más señales."
            )

        # Calibration check
        cal_err = self.get_calibration_error(ticker, window=60)
        if cal_err is not None and abs(cal_err) > 0.10:
            if cal_err > 0:
                result["reasoning"].append(
                    f"Modelo sobreestima prob_up en {cal_err:.1%} — sesgo optimista detectado."
                )
                result["threshold_adjustment"] += 0.02
            else:
                result["reasoning"].append(
                    f"Modelo subestima prob_up en {abs(cal_err):.1%} — sesgo pesimista detectado."
                )
                result["threshold_adjustment"] -= 0.01

        # Drift detection
        acc_60 = self.get_rolling_accuracy(ticker, window=60)
        acc_90 = self.get_rolling_accuracy(ticker, window=90)
        if acc_60 is not None and acc_90 is not None:
            drift = acc_90 - acc
            if drift > config.DRIFT_THRESHOLD:
                result["should_retrain"] = True
                result["reasoning"].append(
                    f"Drift detectado: accuracy cayó de {acc_90:.1%} (90d) a {acc:.1%} (30d) — recomendar retrain."
                )

        self._add_log_entry(
            "adjustment", ticker,
            "; ".join(result["reasoning"]) if result["reasoning"] else "Sin ajustes necesarios.",
            {
                "accuracy_30d": acc,
                "threshold_adj": result["threshold_adjustment"],
                "should_retrain": result["should_retrain"],
            },
        )

        return result

    def compute_ensemble_weight_adjustments(self) -> dict[str, float]:
        """Ajusta pesos del ensemble basado en accuracy global por sub-modelo.

        Lee los reportes de entrenamiento para obtener accuracy por sub-modelo.
        """
        weights = dict(config.ENSEMBLE_WEIGHTS)
        report_files = list(config.REPORTS_DIR.glob("*_metrics.json"))
        if not report_files:
            return weights

        sub_accs: dict[str, list[float]] = {}
        for f in report_files:
            with open(f, "r", encoding="utf-8") as fh:
                report = json.load(fh)
            for name, metrics in report.get("sub_model_metrics", {}).items():
                sub_accs.setdefault(name, []).append(metrics["accuracy"])

        if not sub_accs:
            return weights

        avg_accs = {name: np.mean(accs) for name, accs in sub_accs.items()}
        total = sum(avg_accs.values())
        if total > 0:
            weights = {name: acc / total for name, acc in avg_accs.items()}

        # Guardar pesos
        with open(WEIGHTS_FILE, "w", encoding="utf-8") as f:
            json.dump(weights, f, indent=2)

        self._add_log_entry(
            "weight_update", "ALL",
            f"Pesos ensemble actualizados: {weights}",
            {"weights": weights, "avg_accuracies": avg_accs},
        )
        return weights

    def load_saved_weights(self) -> dict[str, float] | None:
        """Carga pesos guardados del ensemble."""
        if WEIGHTS_FILE.exists():
            with open(WEIGHTS_FILE, "r", encoding="utf-8") as f:
                return json.load(f)
        return None

    def get_adaptive_threshold(self, ticker: str, base_threshold: float) -> float:
        """Retorna threshold ajustado para un ticker."""
        adj = self.compute_adjustments(ticker)
        new_threshold = base_threshold + adj["threshold_adjustment"]
        return max(0.40, min(0.75, new_threshold))

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Resumen legible del estado del learning engine."""
        lines = ["=== Learning Engine Summary ==="]
        total = len(self._history)
        evaluated = self._history["was_correct"].notna().sum()
        pending = total - evaluated
        lines.append(f"Total predicciones: {total} (evaluadas: {evaluated}, pendientes: {pending})")

        if evaluated > 0:
            overall_acc = float(self._history.loc[self._history["was_correct"].notna(), "was_correct"].mean())
            lines.append(f"Accuracy global: {overall_acc:.1%}")

        for ticker in config.TICKERS:
            stats = self.get_ticker_stats(ticker)
            if stats["total"] > 0:
                acc_str = f"{stats['accuracy']:.1%}" if stats["accuracy"] is not None else "N/A"
                lines.append(f"  {ticker}: {stats['total']} eval, acc={acc_str}")
                for wname, wacc in stats.get("windows", {}).items():
                    lines.append(f"    {wname}: {wacc:.1%}")

        return "\n".join(lines)
