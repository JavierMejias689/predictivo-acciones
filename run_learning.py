"""Script de aprendizaje continuo: evalúa predicciones pasadas, ajusta modelos,
detecta drift y reentrena cuando es necesario.

Puede ejecutarse manualmente, por cron, o en un loop continuo:
    python run_learning.py                    # una pasada
    python run_learning.py --loop --interval 3600   # cada hora
    python run_learning.py --retrain          # forzar retrain si hay drift
"""

from __future__ import annotations

import argparse
import time
from datetime import datetime

import config
from learning_engine import LearningEngine


def run_once(force_retrain: bool = False) -> None:
    """Ejecuta una pasada completa del ciclo de aprendizaje."""
    engine = LearningEngine()
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    print(f"\n{'='*60}")
    print(f"[{now}] === Ciclo de Aprendizaje Continuo ===")
    print(f"{'='*60}")

    # 1. Evaluar predicciones pasadas
    print("\n1) Evaluando predicciones pasadas vs resultados reales...")
    try:
        n_eval = engine.evaluate_past_predictions()
        print(f"   {n_eval} predicciones evaluadas en esta pasada.")
    except Exception as exc:
        print(f"   [ERROR] No se pudieron evaluar predicciones: {exc}")

    # 2. Calcular ajustes por ticker
    print("\n2) Calculando ajustes adaptativos por ticker...")
    retrain_tickers: list[str] = []
    for ticker in config.TICKERS:
        adj = engine.compute_adjustments(ticker)
        reasons = adj["reasoning"]
        if reasons:
            print(f"   {ticker}:")
            for r in reasons:
                print(f"     - {r}")
        else:
            print(f"   {ticker}: sin ajustes necesarios.")
        if adj["should_retrain"]:
            retrain_tickers.append(ticker)

    # 3. Ajustar pesos del ensemble
    print("\n3) Ajustando pesos del ensemble...")
    try:
        new_weights = engine.compute_ensemble_weight_adjustments()
        for name, w in new_weights.items():
            print(f"   {name}: {w:.3f}")
    except Exception as exc:
        print(f"   [ERROR] {exc}")

    # 4. Retrain si hay drift
    if retrain_tickers or force_retrain:
        tickers_to_retrain = retrain_tickers if retrain_tickers else config.TICKERS
        print(f"\n4) {'Forzando retrain' if force_retrain else 'Drift detectado'} — reentrenando: {tickers_to_retrain}")
        try:
            from train import train_one_ticker
            for ticker in tickers_to_retrain:
                try:
                    report = train_one_ticker(ticker)
                    print(
                        f"   [OK] {ticker}: acc={report['accuracy_test']:.4f} "
                        f"prec={report['precision_test']:.4f}"
                    )
                except Exception as exc:
                    print(f"   [ERROR] {ticker}: {exc}")
        except Exception as exc:
            print(f"   [ERROR] Retrain fallido: {exc}")
    else:
        print("\n4) No se requiere retrain (sin drift detectado).")

    # 5. Resumen
    print(f"\n{engine.summary()}")
    print(f"\nHistorial: {config.HISTORY_DIR / 'prediction_history.csv'}")
    print(f"Log de pensamiento: {config.HISTORY_DIR / 'learning_log.json'}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Aprendizaje continuo del sistema predictivo.")
    parser.add_argument("--loop", action="store_true", help="Ejecutar en loop continuo.")
    parser.add_argument("--interval", type=int, default=3600, help="Segundos entre ciclos en modo loop (default: 3600).")
    parser.add_argument("--retrain", action="store_true", help="Forzar reentrenamiento de todos los modelos.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    if args.loop:
        print(f"Modo loop activado — ciclo cada {args.interval}s. Ctrl+C para detener.")
        while True:
            try:
                run_once(force_retrain=args.retrain)
                print(f"\nPróximo ciclo en {args.interval}s...")
                time.sleep(args.interval)
            except KeyboardInterrupt:
                print("\nLoop detenido por el usuario.")
                break
    else:
        run_once(force_retrain=args.retrain)


if __name__ == "__main__":
    main()
