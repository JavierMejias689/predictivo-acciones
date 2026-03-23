"""Configuración central del proyecto."""

from __future__ import annotations

from pathlib import Path

# Tickers accionarios de la Bolsa de Santiago (Yahoo Finance usa sufijo .SN)
TICKERS = [
    "SQM-B.SN",
    "COPEC.SN",
    "FALABELLA.SN",
    "CMPC.SN",
    "BSANTANDER.SN",
]

# Variables macro relevantes para Chile
MACRO_TICKERS = {
    "IPSA": "^IPSA",
    "USDCLP": "CLP=X",
    "COPPER": "HG=F",
}

# Contexto global (risk-on/risk-off) para mejorar señales en acciones chilenas
CONTEXT_TICKERS = {
    "SP500": "^GSPC",
    "VIX": "^VIX",
    "US10Y": "^TNX",
    "DXY": "DX-Y.NYB",
    "WTI": "CL=F",
    "GOLD": "GC=F",
    "ECH": "ECH",       # iShares MSCI Chile ETF — proxy directo del mercado chileno
    "EEM": "EEM",       # iShares Emerging Markets — sentimiento emergentes
    "XLB": "XLB",       # Materials Select Sector — correlación con minería chilena
    "XLF": "XLF",       # Financial Select Sector — correlación con banca chilena
    "HYG": "HYG",       # High Yield Corporate Bond — proxy spread de crédito/riesgo
}

# Horizonte histórico
START_DATE = "2015-01-01"
END_DATE = None  # None => hasta la fecha actual disponible
INTERVAL = "1d"

# ---------------------------------------------------------------------------
# Modelado
# ---------------------------------------------------------------------------
MODEL_TYPE = "ensemble"  # opciones: random_forest, xgboost, lightgbm, ensemble
PREDICTION_HORIZON_DAYS = 5  # sweet spot señal/ruido
TEST_SIZE = 0.2
RANDOM_STATE = 42
CV_SPLITS = 5
PREDICTION_THRESHOLD = 0.5
THRESHOLD_GRID = [0.45, 0.48, 0.50, 0.52, 0.55, 0.58, 0.60, 0.63, 0.65, 0.70]
MIN_SIGNAL_RATE = 0.08

# Random Forest — balance entre bias y varianza
RF_N_ESTIMATORS = 600
RF_MAX_DEPTH = 8
RF_MIN_SAMPLES_LEAF = 15
RF_MAX_FEATURES = "sqrt"
RF_CLASS_WEIGHT = "balanced"

# XGBoost — regularización moderada
XGB_N_ESTIMATORS = 500
XGB_MAX_DEPTH = 5
XGB_LEARNING_RATE = 0.03
XGB_SUBSAMPLE = 0.7
XGB_COLSAMPLE_BYTREE = 0.6
XGB_REG_ALPHA = 0.5
XGB_REG_LAMBDA = 3.0
XGB_MIN_CHILD_WEIGHT = 10
XGB_GAMMA = 0.3

# LightGBM — regularización moderada
LGBM_N_ESTIMATORS = 500
LGBM_MAX_DEPTH = 6
LGBM_LEARNING_RATE = 0.03
LGBM_NUM_LEAVES = 31
LGBM_SUBSAMPLE = 0.7
LGBM_COLSAMPLE_BYTREE = 0.6
LGBM_REG_ALPHA = 0.5
LGBM_REG_LAMBDA = 3.0
LGBM_MIN_CHILD_SAMPLES = 15

# Ensemble — pesos iniciales (se ajustan por learning engine)
ENSEMBLE_WEIGHTS = {"random_forest": 0.33, "xgboost": 0.34, "lightgbm": 0.33}

# ---------------------------------------------------------------------------
# Feature engineering
# ---------------------------------------------------------------------------
RSI_WINDOW = 14
VOLATILITY_WINDOW = 20
N_TOP_FEATURES = 15  # sweet spot entre información y ruido
TARGET_RETURN_THRESHOLD = 0.0  # sin filtro de ruido (evita desbalance)
PURGE_DAYS = 5  # gap entre train/test para evitar leakage del horizonte
EARLY_STOPPING_ROUNDS = 0  # desactivado — consume datos de training

# ---------------------------------------------------------------------------
# Learning engine — aprendizaje continuo
# ---------------------------------------------------------------------------
LEARNING_ROLLING_WINDOWS = [30, 60, 90]  # días para calcular accuracy rolling
DRIFT_THRESHOLD = 0.10       # caída de accuracy que dispara retrain
CALIBRATION_BINS = 5         # bins para calibración de probabilidades
MIN_HISTORY_FOR_ADJUSTMENT = 20  # predicciones mínimas antes de ajustar

# ---------------------------------------------------------------------------
# Rutas
# ---------------------------------------------------------------------------
BASE_DIR = Path(__file__).resolve().parent
MODELS_DIR = BASE_DIR / "models"
OUTPUT_DIR = BASE_DIR / "outputs"
PLOTS_DIR = OUTPUT_DIR / "plots"
REPORTS_DIR = OUTPUT_DIR / "reports"
SIGNALS_DIR = OUTPUT_DIR / "signals"
HISTORY_DIR = OUTPUT_DIR / "history"

# Preparación para automatización diaria y futuras señales por Telegram
ENABLE_TELEGRAM_INTEGRATION = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""

for path in (MODELS_DIR, OUTPUT_DIR, PLOTS_DIR, REPORTS_DIR, SIGNALS_DIR, HISTORY_DIR):
    path.mkdir(parents=True, exist_ok=True)
