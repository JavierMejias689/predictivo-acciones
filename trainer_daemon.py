"""Daemon de entrenamiento continuo.

Ejecuta ciclos de: entrenar → predecir → registrar → evaluar → ajustar → dormir.
Diseñado para correr dentro del contenedor Docker en background.

Variables de entorno:
    TRAIN_INTERVAL_HOURS: horas entre ciclos (default: 4)
"""

from __future__ import annotations

import os
import signal
import sys
import time
import traceback
from datetime import datetime

import config
from learning_engine import LearningEngine
from run_daily import predict_one_ticker
from train import main as train_main


INTERVAL_HOURS = int(os.environ.get("TRAIN_INTERVAL_HOURS", "4"))
INTERVAL_SECONDS = INTERVAL_HOURS * 3600
RUNNING = True


def handle_signal(signum: int, frame) -> None:
    global RUNNING
    print(f"\n[DAEMON] Señal {signum} recibida — deteniendo tras ciclo actual...")
    RUNNING = False


signal.signal(signal.SIGTERM, handle_signal)
signal.signal(signal.SIGINT, handle_signal)


def run_cycle(cycle_num: int) -> None:
    """Ejecuta un ciclo completo de entrenamiento + aprendizaje."""
    engine = LearningEngine()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*70}")
    print(f"[DAEMON] Ciclo #{cycle_num} — {now}")
    print(f"{'='*70}")

    # 1. Evaluar predicciones pasadas
    print("\n[1/5] Evaluando predicciones pasadas...")
    try:
        n_eval = engine.evaluate_past_predictions()
        print(f"       {n_eval} predicciones evaluadas.")
    except Exception as exc:
        print(f"       [WARN] {exc}")

    # 2. Entrenar modelos
    print("\n[2/5] Entrenando modelos (todos los tickers)...")
    try:
        train_main()
    except Exception as exc:
        print(f"       [ERROR] Entrenamiento fallido: {exc}")
        traceback.print_exc()

    # 3. Ajustar pesos del ensemble
    print("\n[3/5] Ajustando pesos del ensemble...")
    try:
        weights = engine.compute_ensemble_weight_adjustments()
        for name, w in weights.items():
            print(f"       {name}: {w:.3f}")
    except Exception as exc:
        print(f"       [WARN] {exc}")

    # 4. Generar señales y registrar en learning engine
    print("\n[4/5] Generando señales...")
    for ticker in config.TICKERS:
        try:
            sig = predict_one_ticker(ticker, engine=engine)
            engine.record_prediction(
                ticker=ticker,
                signal_date=str(sig["signal_date"]),
                prob_up=float(sig["prob_up"]),
                prob_down=float(sig["prob_down"]),
                threshold=float(sig["threshold"]),
                recommendation=str(sig["recommendation"]),
                model_type=str(sig.get("model_type", config.MODEL_TYPE)),
            )
            print(
                f"       [OK] {ticker}: prob_up={sig['prob_up']:.2%} "
                f"reco={sig['recommendation']}"
            )
        except Exception as exc:
            print(f"       [ERROR] {ticker}: {exc}")

    # 5. Calcular ajustes adaptativos
    print("\n[5/5] Calculando ajustes adaptativos...")
    retrain_needed = []
    for ticker in config.TICKERS:
        adj = engine.compute_adjustments(ticker)
        if adj["reasoning"]:
            for r in adj["reasoning"]:
                print(f"       {ticker}: {r}")
        if adj["should_retrain"]:
            retrain_needed.append(ticker)

    if retrain_needed:
        print(f"\n       Drift detectado — reentrenando: {retrain_needed}")
        from train import train_one_ticker
        for ticker in retrain_needed:
            try:
                report = train_one_ticker(ticker)
                print(f"       [OK] {ticker}: acc={report['accuracy_test']:.4f}")
            except Exception as exc:
                print(f"       [ERROR] {ticker}: {exc}")

    # Resumen
    print(f"\n{engine.summary()}")
    print(f"\n[DAEMON] Ciclo #{cycle_num} completado. Próximo en {INTERVAL_HOURS}h.")


def main() -> None:
    print(f"[DAEMON] Iniciando trainer daemon (intervalo: {INTERVAL_HOURS}h)")
    print(f"[DAEMON] Tickers: {config.TICKERS}")
    print(f"[DAEMON] Modelo: {config.MODEL_TYPE}")

    cycle = 0
    while RUNNING:
        cycle += 1
        try:
            run_cycle(cycle)
        except Exception as exc:
            print(f"[DAEMON] Error no manejado en ciclo #{cycle}: {exc}")
            traceback.print_exc()

        if not RUNNING:
            break

        # Dormir en intervalos cortos para responder a señales rápido
        remaining = INTERVAL_SECONDS
        while remaining > 0 and RUNNING:
            sleep_chunk = min(remaining, 30)
            time.sleep(sleep_chunk)
            remaining -= sleep_chunk

    print("[DAEMON] Daemon detenido limpiamente.")


if __name__ == "__main__":
    main()
