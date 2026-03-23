"""Predicción diaria de tendencia para un ticker específico."""

from __future__ import annotations

import argparse
from datetime import datetime

import pandas as pd

import config
from data_loader import load_market_data
from features import build_features
from learning_engine import LearningEngine
from model import TrendClassifier


def sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


def build_recommendation(prob_up: float, threshold: float) -> str:
    if prob_up >= threshold:
        return "ALCISTA (comprar / mantener exposición)"
    return "BAJISTA (no comprar / reducir exposición)"


def build_telegram_signal(ticker: str, signal_date: pd.Timestamp, prob_up: float, prob_down: float, reco: str) -> str:
    return (
        f"Señal IA {ticker}\n"
        f"Fecha señal: {signal_date.date()}\n"
        f"Prob. alcista: {prob_up:.2%}\n"
        f"Prob. bajista: {prob_down:.2%}\n"
        f"Recomendación: {reco}\n"
        f"Generado: {datetime.now().isoformat(timespec='seconds')}"
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Predicción de tendencia para acciones chilenas.")
    parser.add_argument("ticker", type=str, help="Ticker Yahoo Finance, ejemplo: SQM-B.SN")
    parser.add_argument("--no-adaptive", action="store_true", help="No usar threshold adaptativo del learning engine.")
    args = parser.parse_args()
    ticker = args.ticker.strip().upper()

    model_path = config.MODELS_DIR / f"{sanitize_ticker(ticker)}.joblib"
    if not model_path.exists():
        raise FileNotFoundError(
            f"No existe modelo entrenado para {ticker} en {model_path}. "
            "Ejecuta primero: python train.py"
        )

    payload = TrendClassifier.load(str(model_path))
    metadata = payload["metadata"]
    feature_cols = metadata["feature_cols"]
    base_threshold = float(metadata.get("threshold", config.PREDICTION_THRESHOLD))
    horizon_days = int(metadata.get("horizon_days", config.PREDICTION_HORIZON_DAYS))

    # Reconstruir clasificador
    clf = TrendClassifier.from_payload(payload)

    # Cargar pesos adaptativos del ensemble si existen
    engine = LearningEngine()
    saved_weights = engine.load_saved_weights()
    if saved_weights and clf.model_type == "ensemble":
        clf.set_ensemble_weights(saved_weights)

    # Threshold adaptativo
    if not args.no_adaptive:
        threshold = engine.get_adaptive_threshold(ticker, base_threshold)
        if threshold != base_threshold:
            print(f"(Threshold adaptado: {base_threshold:.2f} -> {threshold:.2f})")
    else:
        threshold = base_threshold

    df = load_market_data(ticker)
    feat_df, _, _ = build_features(df, horizon_days=horizon_days)
    latest_row = feat_df.iloc[[-1]]
    X_latest = latest_row[feature_cols]

    probs = clf.predict_proba(X_latest)[0]
    prob_down = float(probs[0])
    prob_up = float(probs[1])
    reco = build_recommendation(prob_up, threshold=threshold)
    signal_date = latest_row.index[-1]

    print(f"=== Predicción de tendencia (próximos {horizon_days} días) ===")
    print(f"Ticker: {ticker}")
    print(f"Fecha de señal: {signal_date.date()}")
    print(f"Horizonte objetivo: {horizon_days} días")
    print(f"Modelo: {metadata.get('model_type', 'unknown')}")
    print(f"Probabilidad alcista (1): {prob_up:.4f} ({prob_up:.2%})")
    print(f"Probabilidad bajista (0): {prob_down:.4f} ({prob_down:.2%})")
    print(f"Threshold: {threshold:.2f}")
    print(f"Recomendación final: {reco}")

    if config.ENABLE_TELEGRAM_INTEGRATION:
        msg = build_telegram_signal(ticker, signal_date, prob_up, prob_down, reco)
        print("\nMensaje listo para Telegram:")
        print(msg)


if __name__ == "__main__":
    main()
