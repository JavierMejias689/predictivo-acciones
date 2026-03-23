#!/bin/bash
set -e

echo "=================================================="
echo " Predictivo Acciones - Bolsa de Santiago"
echo " Modelo: ensemble (RF + XGBoost + LightGBM)"
echo " Intervalo entrenamiento: ${TRAIN_INTERVAL_HOURS:-4}h"
echo "=================================================="

# Crear directorios necesarios
mkdir -p models outputs/plots outputs/reports outputs/signals outputs/history

# Arrancar trainer daemon en background
echo "[ENTRYPOINT] Iniciando trainer daemon..."
python trainer_daemon.py &
TRAINER_PID=$!
echo "[ENTRYPOINT] Trainer daemon PID: $TRAINER_PID"

# Trap para shutdown limpio
cleanup() {
    echo "[ENTRYPOINT] Deteniendo procesos..."
    kill -TERM "$TRAINER_PID" 2>/dev/null || true
    wait "$TRAINER_PID" 2>/dev/null || true
    echo "[ENTRYPOINT] Shutdown completo."
    exit 0
}
trap cleanup SIGTERM SIGINT

# Arrancar API en foreground
echo "[ENTRYPOINT] Iniciando API en puerto ${API_PORT:-8000}..."
exec uvicorn api:app \
    --host 0.0.0.0 \
    --port "${API_PORT:-8000}" \
    --workers 1 \
    --log-level info
