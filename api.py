"""API REST para consultar predicciones, accuracy y estado del modelo.

Ejecutar:
    uvicorn api:app --host 0.0.0.0 --port 8000
"""

from __future__ import annotations

import json
import os
import unicodedata
from datetime import datetime, timedelta
from typing import Any, Literal

import numpy as np
import pandas as pd
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

import config
from data_loader import load_market_data
from features import build_features
from learning_engine import LearningEngine
from model import TrendClassifier

app = FastAPI(
    title="Predictivo Acciones - Bolsa de Santiago",
    description="API de predicciÃ³n de tendencia para acciones chilenas con aprendizaje continuo",
    version="1.0.0",
)

# Estado compartido con el trainer daemon
LAST_TRAINING_TS: str = ""
TRAINING_INTERVAL_HOURS = int(os.environ.get("TRAIN_INTERVAL_HOURS", "4"))


def _sanitize_ticker(ticker: str) -> str:
    return ticker.replace(".", "_").replace("-", "_")


def _load_report(ticker: str) -> dict | None:
    """Carga el Ãºltimo reporte de mÃ©tricas de un ticker."""
    safe = _sanitize_ticker(ticker)
    path = config.REPORTS_DIR / f"{safe}_metrics.json"
    if path.exists():
        with open(path, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


def _get_last_training_time() -> str:
    """Busca el timestamp del Ãºltimo modelo entrenado."""
    latest = ""
    for ticker in config.TICKERS:
        report = _load_report(ticker)
        if report:
            trained_at = report.get("model_path", "")
            path = config.MODELS_DIR / f"{_sanitize_ticker(ticker)}.joblib"
            if path.exists():
                mtime = datetime.fromtimestamp(path.stat().st_mtime)
                ts = mtime.isoformat(timespec="seconds")
                if ts > latest:
                    latest = ts
    return latest or "never"


def _ascii_text(value: str) -> str:
    """Convierte texto a ASCII para evitar problemas de encoding en clientes."""
    replacements = {
        "Ã¡": "a",
        "Ã©": "e",
        "Ã­": "i",
        "Ã³": "o",
        "Ãº": "u",
        "Ã±": "n",
        "Ã¼": "u",
        "Â": "",
        "â€”": "-",
        "â†’": "->",
    }
    for old, new in replacements.items():
        value = value.replace(old, new)
    value = unicodedata.normalize("NFKD", value)
    return value.encode("ascii", "ignore").decode("ascii")


def _asciiize_payload(value: Any) -> Any:
    """Normaliza strings dentro de dict/list para respuestas API sin acentos."""
    if isinstance(value, str):
        return _ascii_text(value)
    if isinstance(value, list):
        return [_asciiize_payload(v) for v in value]
    if isinstance(value, dict):
        return {k: _asciiize_payload(v) for k, v in value.items()}
    return value


def _parse_input_date(value: str) -> str:
    """Acepta dd-mm-yyyy o yyyy-mm-dd y retorna yyyy-mm-dd."""
    raw = value.strip()
    for fmt in ("%d-%m-%Y", "%d/%m/%Y", "%Y-%m-%d", "%Y/%m/%d"):
        try:
            return datetime.strptime(raw, fmt).date().isoformat()
        except ValueError:
            continue
    raise HTTPException(
        status_code=400,
        detail="start_date invalida. Usa formato dd-mm-yyyy o yyyy-mm-dd.",
    )


def _tickers_for_market(market: str) -> list[str]:
    mk = market.upper()
    tickers = list(config.TICKERS)
    if mk in {"CL", "CHILE", "SN"}:
        return [t for t in tickers if t.endswith(".SN")]
    if mk in {"US", "USA"}:
        return [t for t in tickers if not t.endswith(".SN")]
    if mk in {"ALL", "*"}:
        return tickers
    raise HTTPException(status_code=400, detail=f"Mercado no soportado: {market}")


class LearnFromDateRequest(BaseModel):
    start_date: str = Field(..., description="Fecha inicial: dd-mm-yyyy o yyyy-mm-dd")
    market: Literal["CL", "CHILE", "SN", "US", "USA", "ALL"] = "CL"
    tickers: list[str] | None = Field(default=None, description="Lista opcional de tickers a procesar")
    max_tickers: int = Field(default=5, ge=1, le=50, description="Limite de tickers para controlar volumen")


# -----------------------------------------------------------------------
# EstimaciÃ³n de fecha de venta
# -----------------------------------------------------------------------


def _estimate_sell(feat_df: pd.DataFrame, signal_date: str, recommendation: str) -> dict:
    """Estima cuÃ¡ndo vender basado en indicadores tÃ©cnicos del Ãºltimo dato.

    Analiza RSI, MACD, Bollinger %B, Stochastic, ADX y momentum
    para sugerir cuÃ¡ntos dÃ­as mantener la posiciÃ³n.

    Returns:
        dict con: sell_in_days, sell_date, confidence, reasoning
    """
    if recommendation == "BAJISTA":
        return {
            "sell_in_days": 0,
            "sell_date": None,
            "action": "NO COMPRAR â€” esperar seÃ±al alcista",
            "confidence": "N/A",
            "reasoning": ["SeÃ±al bajista: no se recomienda entrar en posiciÃ³n."],
        }

    row = feat_df.iloc[-1]
    base_date = pd.Timestamp(signal_date)
    reasons: list[str] = []

    # PuntuaciÃ³n: valores altos = vender pronto, valores bajos = mantener mÃ¡s
    urgency = 0.0  # rango aprox -3 a +5

    # --- RSI (sobrecompra/sobreventa) ---
    rsi = row.get("RSI", np.nan)
    if not np.isnan(rsi):
        if rsi >= 75:
            urgency += 2.0
            reasons.append(f"RSI={rsi:.0f} â€” zona de sobrecompra fuerte, vender pronto")
        elif rsi >= 65:
            urgency += 1.0
            reasons.append(f"RSI={rsi:.0f} â€” acercÃ¡ndose a sobrecompra")
        elif rsi >= 50:
            urgency -= 0.5
            reasons.append(f"RSI={rsi:.0f} â€” momentum saludable, mantener")
        else:
            urgency += 0.5
            reasons.append(f"RSI={rsi:.0f} â€” momentum dÃ©bil, posiciÃ³n cautelosa")

    # --- MACD Histogram (direcciÃ³n del momentum) ---
    macd_hist = row.get("MACD_Hist", np.nan)
    if not np.isnan(macd_hist):
        # Revisar si el histograma estÃ¡ declinando (Ãºltimas 3 barras)
        if len(feat_df) >= 3:
            hist_recent = feat_df["MACD_Hist"].iloc[-3:].values
            if len(hist_recent) == 3 and all(np.isfinite(hist_recent)):
                if hist_recent[2] < hist_recent[1] < hist_recent[0]:
                    urgency += 1.5
                    reasons.append("MACD histograma declinando 3 barras â€” momentum agotÃ¡ndose")
                elif hist_recent[2] > hist_recent[1] > hist_recent[0]:
                    urgency -= 1.0
                    reasons.append("MACD histograma creciendo â€” momentum acelerando")

        if macd_hist < 0:
            urgency += 1.0
            reasons.append(f"MACD histograma negativo ({macd_hist:.4f}) â€” presiÃ³n bajista")

    # --- Bollinger %B (posiciÃ³n dentro de las bandas) ---
    bb_pctb = row.get("BB_PctB", np.nan)
    if not np.isnan(bb_pctb):
        if bb_pctb >= 0.95:
            urgency += 1.5
            reasons.append(f"Bollinger %B={bb_pctb:.2f} â€” tocando banda superior, reversiÃ³n probable")
        elif bb_pctb >= 0.80:
            urgency += 0.5
            reasons.append(f"Bollinger %B={bb_pctb:.2f} â€” cerca de banda superior")
        elif bb_pctb <= 0.20:
            urgency -= 0.5
            reasons.append(f"Bollinger %B={bb_pctb:.2f} â€” cerca de banda inferior, potencial rebote")

    # --- Stochastic (sobrecompra) ---
    stoch_k = row.get("Stoch_K", np.nan)
    if not np.isnan(stoch_k):
        if stoch_k >= 80:
            urgency += 1.0
            reasons.append(f"Stochastic %K={stoch_k:.0f} â€” zona de sobrecompra")
        elif stoch_k <= 20:
            urgency -= 0.5
            reasons.append(f"Stochastic %K={stoch_k:.0f} â€” zona de sobreventa, potencial alcista")

    # --- ADX (fuerza de tendencia) ---
    adx = row.get("ADX", np.nan)
    if not np.isnan(adx):
        if adx >= 40:
            urgency -= 0.5
            reasons.append(f"ADX={adx:.0f} â€” tendencia fuerte, puede extenderse")
        elif adx < 20:
            urgency += 0.5
            reasons.append(f"ADX={adx:.0f} â€” tendencia dÃ©bil, movimiento limitado")

    # --- Convertir urgency a dÃ­as de hold ---
    # urgency: -3 (mantener mucho) a +5 (vender ya)
    # Mapeo: urgency <= -2 â†’ 10 dÃ­as, urgency >= 4 â†’ 1 dÃ­a
    hold_days = max(1, min(10, round(5 - urgency)))

    # Calcular fecha de venta (solo dÃ­as hÃ¡biles L-V)
    sell_date = base_date
    biz_days_added = 0
    while biz_days_added < hold_days:
        sell_date += timedelta(days=1)
        if sell_date.weekday() < 5:  # Lunes=0 a Viernes=4
            biz_days_added += 1

    # Confianza en la estimaciÃ³n
    n_indicators = len(reasons)
    if n_indicators >= 4:
        confidence = "alta" if abs(urgency) >= 2.5 else "media"
    elif n_indicators >= 2:
        confidence = "media"
    else:
        confidence = "baja"

    # Días hábiles restantes desde hoy (evita mensajes desfasados si la señal es antigua)
    today = pd.Timestamp.today().normalize()
    sell_day = pd.Timestamp(sell_date).normalize()
    remaining_biz_days = int(np.busday_count(today.date(), sell_day.date()))

    if remaining_biz_days <= 0:
        action_text = f"VENDER hoy ({sell_date.strftime('%d/%m/%Y')})"
    else:
        action_text = f"VENDER en ~{remaining_biz_days} días hábiles ({sell_date.strftime('%d/%m/%Y')})"

    return {
        "sell_in_days": hold_days,  # días estimados desde signal_date
        "sell_in_days_remaining": max(0, remaining_biz_days),  # días restantes desde hoy
        "sell_date": sell_date.date().isoformat(),
        "action": action_text,
        "confidence": confidence,
        "reasoning": reasons,
    }


# -----------------------------------------------------------------------
# Endpoints
# -----------------------------------------------------------------------


@app.get("/health")
def health():
    """Healthcheck."""
    return _asciiize_payload({"status": "ok", "timestamp": datetime.now().isoformat(timespec="seconds")})


@app.get("/predict/{ticker}")
def predict(ticker: str):
    """PredicciÃ³n de tendencia para un ticker.

    Ejemplo: GET /predict/COPEC.SN
    """
    ticker = ticker.strip().upper()
    if ticker not in config.TICKERS:
        raise HTTPException(
            status_code=404,
            detail=f"Ticker '{ticker}' no soportado. Disponibles: {config.TICKERS}",
        )

    model_path = config.MODELS_DIR / f"{_sanitize_ticker(ticker)}.joblib"
    if not model_path.exists():
        raise HTTPException(
            status_code=503,
            detail=f"Modelo no entrenado aun para {ticker}. Esperando primer ciclo de entrenamiento.",
        )

    try:
        # Cargar modelo
        payload = TrendClassifier.load(str(model_path))
        metadata = payload["metadata"]
        feature_cols = metadata["feature_cols"]
        horizon_days = int(metadata.get("horizon_days", config.PREDICTION_HORIZON_DAYS))
        base_threshold = float(metadata.get("threshold", config.PREDICTION_THRESHOLD))

        clf = TrendClassifier.from_payload(payload)

        # Pesos adaptativos
        engine = LearningEngine()
        saved_weights = engine.load_saved_weights()
        if saved_weights and clf.model_type == "ensemble":
            clf.set_ensemble_weights(saved_weights)

        # Threshold adaptativo
        threshold = engine.get_adaptive_threshold(ticker, base_threshold)

        # Predecir
        df = load_market_data(ticker)
        feat_df, _, _ = build_features(df, horizon_days=horizon_days)
        latest = feat_df.iloc[[-1]]
        probs = clf.predict_proba(latest[feature_cols])[0]
        prob_up = float(probs[1])
        prob_down = float(probs[0])
        recommendation = "ALCISTA" if prob_up >= threshold else "BAJISTA"

        # MÃ©tricas del reporte
        report = _load_report(ticker)
        accuracy = report.get("accuracy_test", None) if report else None
        precision = report.get("precision_test", None) if report else None
        recall = report.get("recall_test", None) if report else None

        # MÃ©tricas del learning engine
        stats = engine.get_ticker_stats(ticker)
        live_accuracy = stats.get("accuracy")

        signal_date = pd.Timestamp(latest.index[-1]).date().isoformat()

        # Accuracy y precision como porcentaje legible
        accuracy_pct = f"{accuracy * 100:.1f}%" if accuracy else "entrenando..."
        precision_pct = f"{precision * 100:.1f}%" if precision else "entrenando..."
        live_accuracy_pct = f"{live_accuracy * 100:.1f}%" if live_accuracy else "acumulando datos..."

        # EstimaciÃ³n de fecha de venta
        sell_info = _estimate_sell(feat_df, signal_date, recommendation)

        buy_action = "COMPRAR / MANTENER" if recommendation == "ALCISTA" else "NO COMPRAR"
        response = {
            "ticker": ticker,
            "tendencia": recommendation,
            "compra": {
                "action": buy_action,
                "reason": f"prob_up={prob_up:.2%} vs threshold={threshold:.2%}",
            },
            "signal_date": signal_date,
            "horizon_days": horizon_days,
            "prob_up": round(prob_up, 4),
            "prob_down": round(prob_down, 4),
            "threshold": round(threshold, 4),
            "accuracy": accuracy_pct,
            "precision": precision_pct,
            "live_accuracy": live_accuracy_pct,
            "venta": sell_info,
            "model_type": metadata.get("model_type", "unknown"),
            "n_features": len(feature_cols),
            "metrics_raw": {
                "accuracy_test": round(accuracy, 4) if accuracy else None,
                "precision_test": round(precision, 4) if precision else None,
                "recall_test": round(recall, 4) if recall else None,
                "live_accuracy": round(live_accuracy, 4) if live_accuracy else None,
                "live_predictions_evaluated": stats.get("total", 0),
            },
            "last_trained": metadata.get("trained_at", "unknown"),
        }
        return _asciiize_payload(response)

    except HTTPException:
        raise
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc))


@app.get("/status")
def status():
    """Estado general del sistema: accuracy promedio, predicciones, Ãºltimo entrenamiento."""
    engine = LearningEngine()
    history = engine._history

    total_predictions = len(history)
    evaluated = int(history["was_correct"].notna().sum())
    pending = total_predictions - evaluated

    overall_accuracy = None
    if evaluated > 0:
        overall_accuracy = round(float(history.loc[history["was_correct"].notna(), "was_correct"].mean()), 4)

    # Accuracy por ticker del Ãºltimo entrenamiento
    ticker_metrics = {}
    for ticker in config.TICKERS:
        report = _load_report(ticker)
        stats = engine.get_ticker_stats(ticker)
        ticker_metrics[ticker] = {
            "accuracy_test": round(report["accuracy_test"], 4) if report else None,
            "precision_test": round(report["precision_test"], 4) if report else None,
            "live_accuracy": round(stats["accuracy"], 4) if stats.get("accuracy") is not None else None,
            "live_predictions": stats.get("total", 0),
        }

    avg_accuracy_test = None
    accs = [m["accuracy_test"] for m in ticker_metrics.values() if m["accuracy_test"] is not None]
    if accs:
        avg_accuracy_test = round(sum(accs) / len(accs), 4)

    response = {
        "model_type": config.MODEL_TYPE,
        "horizon_days": config.PREDICTION_HORIZON_DAYS,
        "tickers": config.TICKERS,
        "avg_accuracy_test": avg_accuracy_test,
        "live_accuracy": overall_accuracy,
        "total_predictions": total_predictions,
        "evaluated_predictions": evaluated,
        "pending_predictions": pending,
        "training_interval_hours": TRAINING_INTERVAL_HOURS,
        "last_training": _get_last_training_time(),
        "ticker_details": ticker_metrics,
    }
    return _asciiize_payload(response)


@app.get("/status/{ticker}")
def status_ticker(ticker: str):
    """Stats detallados de un ticker especÃ­fico."""
    ticker = ticker.strip().upper()
    if ticker not in config.TICKERS:
        raise HTTPException(status_code=404, detail=f"Ticker '{ticker}' no soportado.")

    engine = LearningEngine()
    stats = engine.get_ticker_stats(ticker)
    report = _load_report(ticker)

    response = {
        "ticker": ticker,
        "training_metrics": {
            "accuracy_test": report.get("accuracy_test") if report else None,
            "precision_test": report.get("precision_test") if report else None,
            "recall_test": report.get("recall_test") if report else None,
            "selected_threshold": report.get("selected_threshold") if report else None,
            "n_features": report.get("n_features") if report else None,
            "sub_model_metrics": report.get("sub_model_metrics") if report else None,
        },
        "learning_engine": {
            "total_evaluated": stats.get("total", 0),
            "live_accuracy": stats.get("accuracy"),
            "avg_prob_up": stats.get("avg_prob_up"),
            "actual_up_rate": stats.get("actual_up_rate"),
            "rolling_windows": stats.get("windows", {}),
        },
    }
    return _asciiize_payload(response)


@app.get("/signals")
def signals():
    """Ãšltimas seÃ±ales generadas."""
    path = config.SIGNALS_DIR / "signals_daily.csv"
    if not path.exists():
        return _asciiize_payload({"signals": [], "count": 0})

    df = pd.read_csv(path)
    # Ãšltimas seÃ±ales por ticker
    latest = df.sort_values("generated_at", ascending=False).groupby("ticker").first().reset_index()
    records = latest.to_dict(orient="records")
    return _asciiize_payload({"signals": records, "count": len(records)})


@app.post("/learn/from-date")
def learn_from_date(req: LearnFromDateRequest):
    """Reentrena modelos desde una fecha dada hasta hoy para aprender patrones recientes."""
    start_date_iso = _parse_input_date(req.start_date)

    today_iso = datetime.now().date().isoformat()
    if start_date_iso > today_iso:
        raise HTTPException(status_code=400, detail="start_date no puede ser futura.")

    if req.tickers:
        requested = [t.strip().upper() for t in req.tickers if t.strip()]
        unsupported = [t for t in requested if t not in config.TICKERS]
        if unsupported:
            raise HTTPException(
                status_code=400,
                detail=f"Tickers no soportados: {unsupported}. Disponibles: {config.TICKERS}",
            )
        selected = requested
    else:
        selected = _tickers_for_market(req.market)

    if not selected:
        raise HTTPException(status_code=400, detail="No hay tickers para el mercado seleccionado.")

    tickers_to_train = selected[: req.max_tickers]
    train_results: list[dict] = []
    errors: list[dict] = []

    from train import train_one_ticker

    for ticker in tickers_to_train:
        try:
            fallback_used = False
            try:
                report = train_one_ticker(ticker, start_date=start_date_iso, end_date=None)
            except Exception as first_exc:
                # Fallback robusto si la ventana pedida deja pocas muestras tras feature engineering
                if "0 sample" not in str(first_exc).lower():
                    raise
                report = train_one_ticker(ticker, start_date=config.START_DATE, end_date=None)
                fallback_used = True

            train_results.append(
                {
                    "ticker": ticker,
                    "accuracy_test": round(float(report.get("accuracy_test", 0.0)), 4),
                    "precision_test": round(float(report.get("precision_test", 0.0)), 4),
                    "recall_test": round(float(report.get("recall_test", 0.0)), 4),
                    "train_start_date": report.get("train_start_date", start_date_iso),
                    "fallback_to_full_history": fallback_used,
                }
            )
        except Exception as exc:
            errors.append({"ticker": ticker, "error": str(exc)})

    # Actualiza learning engine tras reentrenar
    engine = LearningEngine()
    try:
        n_eval = engine.evaluate_past_predictions()
    except Exception:
        n_eval = 0

    try:
        weights = engine.compute_ensemble_weight_adjustments()
    except Exception:
        weights = {}

    response = {
        "ok": len(errors) == 0,
        "start_date": start_date_iso,
        "end_date": today_iso,
        "market": req.market,
        "requested_count": len(selected),
        "processed_count": len(tickers_to_train),
        "trained_count": len(train_results),
        "errors_count": len(errors),
        "train_results": train_results,
        "errors": errors,
        "evaluated_predictions_after_train": n_eval,
        "ensemble_weights": weights,
    }
    return _asciiize_payload(response)

