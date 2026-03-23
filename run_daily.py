"""Orquestador diario: reentrena, genera señales con aprendizaje continuo.

Ejemplos:
  python run_daily.py
  python run_daily.py --skip-train
  python run_daily.py --threshold 0.60
  python run_daily.py --no-learning
"""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import config
from data_loader import load_market_data
from features import build_features
from learning_engine import LearningEngine
from model import TrendClassifier
from train import main as train_main


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


def recommendation(prob_up: float, threshold: float) -> str:
    return "ALCISTA" if prob_up >= threshold else "BAJISTA"


def predict_one_ticker(
    ticker: str,
    threshold_override: float | None = None,
    engine: LearningEngine | None = None,
) -> dict[str, str | float]:
    model_path = config.MODELS_DIR / f"{sanitize_ticker(ticker)}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(f"Modelo no encontrado para {ticker}: {model_path}")

    payload = TrendClassifier.load(str(model_path))
    metadata = payload["metadata"]
    feature_cols = metadata["feature_cols"]
    horizon_days = int(metadata.get("horizon_days", config.PREDICTION_HORIZON_DAYS))
    base_threshold = float(metadata.get("threshold", config.PREDICTION_THRESHOLD))

    # Reconstruir clasificador
    clf = TrendClassifier.from_payload(payload)

    # Aplicar pesos adaptativos del ensemble
    if engine and clf.model_type == "ensemble":
        saved_weights = engine.load_saved_weights()
        if saved_weights:
            clf.set_ensemble_weights(saved_weights)

    # Threshold: override > adaptive > base
    if threshold_override is not None:
        threshold = float(threshold_override)
    elif engine:
        threshold = engine.get_adaptive_threshold(ticker, base_threshold)
    else:
        threshold = base_threshold

    df = load_market_data(ticker)
    feat_df, _, _ = build_features(df, horizon_days=horizon_days)
    latest = feat_df.iloc[[-1]]
    probs = clf.predict_proba(latest[feature_cols])[0]

    prob_down = float(probs[0])
    prob_up = float(probs[1])
    reco = recommendation(prob_up, threshold)

    signal_date = pd.Timestamp(latest.index[-1]).date().isoformat()
    now_ts = datetime.now().isoformat(timespec="seconds")
    return {
        "generated_at": now_ts,
        "ticker": ticker,
        "signal_date": signal_date,
        "horizon_days": horizon_days,
        "threshold": threshold,
        "prob_up": prob_up,
        "prob_down": prob_down,
        "recommendation": reco,
        "model_path": str(model_path),
        "model_type": metadata.get("model_type", "unknown"),
    }


def append_signals(signals: list[dict[str, str | float]]) -> str:
    out_path = config.SIGNALS_DIR / "signals_daily.csv"
    df = pd.DataFrame(signals)
    if out_path.exists():
        prev = pd.read_csv(out_path)
        df = pd.concat([prev, df], ignore_index=True)
    df.to_csv(out_path, index=False, encoding="utf-8")
    return str(out_path)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pipeline diario de entrenamiento y señales.")
    parser.add_argument("--skip-train", action="store_true", help="No reentrena modelos.")
    parser.add_argument("--threshold", type=float, default=None, help="Umbral global opcional.")
    parser.add_argument("--no-learning", action="store_true", help="Desactiva learning engine.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    engine = None if args.no_learning else LearningEngine()

    print("=== Pipeline Diario (Santiago) ===")
    print(f"    Modelo: {config.MODEL_TYPE}")
    print(f"    Learning engine: {'OFF' if args.no_learning else 'ON'}")
    print()

    # 0. Evaluar predicciones pasadas
    if engine:
        print("0) Evaluando predicciones pasadas...")
        try:
            n_eval = engine.evaluate_past_predictions()
            print(f"   {n_eval} predicciones evaluadas.")
        except Exception as exc:
            print(f"   [WARN] {exc}")

    # 1. Retrain
    if not args.skip_train:
        print("\n1) Reentrenando modelos...")
        train_main()
    else:
        print("\n1) Reentrenamiento omitido (--skip-train).")

    # 2. Ajustar pesos ensemble
    if engine:
        print("\n   Ajustando pesos del ensemble...")
        try:
            weights = engine.compute_ensemble_weight_adjustments()
            for name, w in weights.items():
                print(f"     {name}: {w:.3f}")
        except Exception:
            pass

    # 3. Generar señales
    print("\n2) Generando señales...")
    signals: list[dict[str, str | float]] = []
    for ticker in config.TICKERS:
        try:
            sig = predict_one_ticker(ticker, threshold_override=args.threshold, engine=engine)
            signals.append(sig)
            print(
                f"[OK] {ticker}: prob_up={sig['prob_up']:.2%} "
                f"th={sig['threshold']:.2f} "
                f"reco={sig['recommendation']}"
            )
            # Registrar en learning engine
            if engine:
                engine.record_prediction(
                    ticker=ticker,
                    signal_date=str(sig["signal_date"]),
                    prob_up=float(sig["prob_up"]),
                    prob_down=float(sig["prob_down"]),
                    threshold=float(sig["threshold"]),
                    recommendation=str(sig["recommendation"]),
                    model_type=str(sig.get("model_type", config.MODEL_TYPE)),
                )
        except Exception as exc:
            print(f"[ERROR] {ticker}: {exc}")

    if not signals:
        raise RuntimeError("No se generaron señales.")

    signals_path = append_signals(signals)
    print(f"\nSeñales guardadas en: {signals_path}")

    # 4. Resumen del learning engine
    if engine:
        print(f"\n{engine.summary()}")


if __name__ == "__main__":
    main()
