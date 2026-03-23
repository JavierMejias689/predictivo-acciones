"""Entrenamiento y evaluación de modelos de clasificación con ensemble."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, confusion_matrix, precision_score, recall_score
from sklearn.model_selection import TimeSeriesSplit, cross_validate
from sklearn.preprocessing import StandardScaler

import config


@dataclass
class ModelResult:
    accuracy: float
    precision: float
    recall: float
    confusion_matrix: np.ndarray


class TrendClassifier:
    """Wrapper que soporta RF, XGBoost, LightGBM y Ensemble."""

    def __init__(self, model_type: str | None = None):
        self.model_type = (model_type or config.MODEL_TYPE).lower()
        self.scaler: StandardScaler | None = None
        if self.model_type == "ensemble":
            self._sub_models: dict[str, Any] = {}
            self._weights: dict[str, float] = dict(config.ENSEMBLE_WEIGHTS)
            for name in self._weights:
                self._sub_models[name] = self._build_single(name)
            self.model = None
        else:
            self.model = self._build_single(self.model_type)
            self._sub_models = {}
            self._weights = {}

    @staticmethod
    def _build_single(model_type: str) -> Any:
        if model_type == "random_forest":
            return RandomForestClassifier(
                n_estimators=config.RF_N_ESTIMATORS,
                max_depth=config.RF_MAX_DEPTH,
                min_samples_leaf=config.RF_MIN_SAMPLES_LEAF,
                max_features=config.RF_MAX_FEATURES,
                class_weight=config.RF_CLASS_WEIGHT,
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
            )
        if model_type == "xgboost":
            from xgboost import XGBClassifier
            return XGBClassifier(
                n_estimators=config.XGB_N_ESTIMATORS,
                max_depth=config.XGB_MAX_DEPTH,
                learning_rate=config.XGB_LEARNING_RATE,
                subsample=config.XGB_SUBSAMPLE,
                colsample_bytree=config.XGB_COLSAMPLE_BYTREE,
                reg_alpha=config.XGB_REG_ALPHA,
                reg_lambda=config.XGB_REG_LAMBDA,
                min_child_weight=config.XGB_MIN_CHILD_WEIGHT,
                gamma=config.XGB_GAMMA,
                use_label_encoder=False,
                eval_metric="logloss",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbosity=0,
            )
        if model_type == "lightgbm":
            from lightgbm import LGBMClassifier
            return LGBMClassifier(
                n_estimators=config.LGBM_N_ESTIMATORS,
                max_depth=config.LGBM_MAX_DEPTH,
                learning_rate=config.LGBM_LEARNING_RATE,
                num_leaves=config.LGBM_NUM_LEAVES,
                subsample=config.LGBM_SUBSAMPLE,
                colsample_bytree=config.LGBM_COLSAMPLE_BYTREE,
                reg_alpha=config.LGBM_REG_ALPHA,
                reg_lambda=config.LGBM_REG_LAMBDA,
                min_child_samples=config.LGBM_MIN_CHILD_SAMPLES,
                class_weight="balanced",
                random_state=config.RANDOM_STATE,
                n_jobs=-1,
                verbose=-1,
            )
        raise ValueError(f"Tipo de modelo no soportado: {model_type}")

    @staticmethod
    def temporal_train_test_split(
        df: pd.DataFrame,
        feature_cols: list[str],
        target_col: str,
        test_size: float | None = None,
    ) -> tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        """Divide en train/test respetando orden temporal con purge gap."""
        test_size = test_size or config.TEST_SIZE
        purge = getattr(config, "PURGE_DAYS", 0)
        n = len(df)
        split_idx = max(int(n * (1 - test_size)), 1)
        X = df[feature_cols].copy()
        y = df[target_col].copy()
        # Purge: eliminar últimas `purge` filas del train para evitar leakage
        train_end = max(split_idx - purge, 1)
        return X.iloc[:train_end], X.iloc[split_idx:], y.iloc[:train_end], y.iloc[split_idx:]

    def _scale(self, X: pd.DataFrame, fit: bool = False) -> pd.DataFrame:
        """Escala features con StandardScaler, preservando nombres de columnas."""
        if fit:
            self.scaler = StandardScaler()
            scaled = self.scaler.fit_transform(X)
        elif self.scaler is not None:
            scaled = self.scaler.transform(X)
        else:
            return X.copy()
        return pd.DataFrame(scaled, index=X.index, columns=X.columns)

    def fit(self, X_train: pd.DataFrame, y_train: pd.Series) -> None:
        X_scaled = self._scale(X_train, fit=True)
        # Reservar 15% final para early stopping de GBM
        es_rounds = getattr(config, "EARLY_STOPPING_ROUNDS", 0)
        es_split = int(len(X_scaled) * 0.85)
        X_es_train, X_es_val = X_scaled.iloc[:es_split], X_scaled.iloc[es_split:]
        y_es_train, y_es_val = y_train.iloc[:es_split], y_train.iloc[es_split:]

        if self.model_type == "ensemble":
            for name, mdl in self._sub_models.items():
                if es_rounds and name in ("xgboost", "lightgbm") and len(X_es_val) > 10:
                    self._fit_with_early_stopping(mdl, name, X_es_train, y_es_train, X_es_val, y_es_val, es_rounds)
                else:
                    mdl.fit(X_scaled, y_train)
        else:
            if es_rounds and self.model_type in ("xgboost", "lightgbm") and len(X_es_val) > 10:
                self._fit_with_early_stopping(self.model, self.model_type, X_es_train, y_es_train, X_es_val, y_es_val, es_rounds)
            else:
                self.model.fit(X_scaled, y_train)

    @staticmethod
    def _fit_with_early_stopping(mdl: Any, model_type: str,
                                  X_train: pd.DataFrame, y_train: pd.Series,
                                  X_val: pd.DataFrame, y_val: pd.Series,
                                  n_rounds: int) -> None:
        """Entrena XGBoost/LightGBM con early stopping."""
        if model_type == "xgboost":
            mdl.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                verbose=False,
            )
        elif model_type == "lightgbm":
            mdl.fit(
                X_train, y_train,
                eval_set=[(X_val, y_val)],
                callbacks=[
                    __import__("lightgbm").early_stopping(n_rounds, verbose=False),
                    __import__("lightgbm").log_evaluation(0),
                ],
            )

    def predict(self, X: pd.DataFrame) -> np.ndarray:
        probs = self.predict_proba(X)
        return (probs[:, 1] >= 0.5).astype(int)

    def predict_proba(self, X: pd.DataFrame) -> np.ndarray:
        X_scaled = self._scale(X)
        if self.model_type == "ensemble":
            # Usar el sub-modelo con mayor peso (best model selection)
            # y complementar con voting ponderado de los demás
            weighted_probs = np.zeros((len(X), 2))
            total_weight = sum(self._weights.values())
            for name, mdl in self._sub_models.items():
                w = self._weights[name] / total_weight
                weighted_probs += w * mdl.predict_proba(X_scaled)
            return weighted_probs
        return self.model.predict_proba(X_scaled)

    def evaluate(self, X_test: pd.DataFrame, y_test: pd.Series) -> ModelResult:
        y_pred = self.predict(X_test)
        return ModelResult(
            accuracy=float(accuracy_score(y_test, y_pred)),
            precision=float(precision_score(y_test, y_pred, zero_division=0)),
            recall=float(recall_score(y_test, y_pred, zero_division=0)),
            confusion_matrix=confusion_matrix(y_test, y_pred),
        )

    def evaluate_sub_models(self, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, ModelResult]:
        """Evalúa cada sub-modelo del ensemble por separado."""
        results = {}
        if self.model_type != "ensemble":
            return results
        X_scaled = self._scale(X_test)
        for name, mdl in self._sub_models.items():
            y_pred = mdl.predict(X_scaled)
            results[name] = ModelResult(
                accuracy=float(accuracy_score(y_test, y_pred)),
                precision=float(precision_score(y_test, y_pred, zero_division=0)),
                recall=float(recall_score(y_test, y_pred, zero_division=0)),
                confusion_matrix=confusion_matrix(y_test, y_pred),
            )
        return results

    def time_series_cross_validation(self, X_train: pd.DataFrame, y_train: pd.Series) -> dict[str, float]:
        """CV temporal. Para ensemble usa el primer sub-modelo RF como proxy rápido."""
        if self.model_type == "ensemble":
            proxy = self._sub_models.get("random_forest", list(self._sub_models.values())[0])
        else:
            proxy = self.model
        X_scaled = self._scale(X_train)
        tscv = TimeSeriesSplit(n_splits=config.CV_SPLITS)
        scoring = {"acc": "accuracy", "prec": "precision", "rec": "recall"}
        cv = cross_validate(proxy, X_scaled, y_train, cv=tscv, scoring=scoring, n_jobs=-1)
        return {
            "cv_accuracy_mean": float(cv["test_acc"].mean()),
            "cv_precision_mean": float(cv["test_prec"].mean()),
            "cv_recall_mean": float(cv["test_rec"].mean()),
        }

    def get_feature_importances(self, feature_cols: list[str]) -> pd.DataFrame:
        """Importancia combinada de features (promedio ponderado para ensemble)."""
        if self.model_type == "ensemble":
            importances = np.zeros(len(feature_cols))
            total_w = sum(self._weights.values())
            for name, mdl in self._sub_models.items():
                if hasattr(mdl, "feature_importances_"):
                    w = self._weights[name] / total_w
                    importances += w * mdl.feature_importances_
            return pd.DataFrame({
                "feature": feature_cols,
                "importance": importances,
            }).sort_values("importance", ascending=False)
        if hasattr(self.model, "feature_importances_"):
            return pd.DataFrame({
                "feature": feature_cols,
                "importance": self.model.feature_importances_,
            }).sort_values("importance", ascending=False)
        return pd.DataFrame({"feature": feature_cols, "importance": 0.0})

    def set_ensemble_weights(self, weights: dict[str, float]) -> None:
        """Actualiza pesos del ensemble (usado por learning engine)."""
        if self.model_type == "ensemble":
            self._weights.update(weights)

    def save(self, path: str, metadata: dict[str, Any]) -> None:
        if self.model_type == "ensemble":
            payload = {
                "model_type": self.model_type,
                "sub_models": self._sub_models,
                "weights": self._weights,
                "scaler": self.scaler,
                "metadata": metadata,
            }
        else:
            payload = {"model_type": self.model_type, "model": self.model, "scaler": self.scaler, "metadata": metadata}
        joblib.dump(payload, path)

    @staticmethod
    def load(path: str) -> dict[str, Any]:
        return joblib.load(path)

    @classmethod
    def from_payload(cls, payload: dict[str, Any]) -> "TrendClassifier":
        """Reconstruye un TrendClassifier desde un payload cargado."""
        obj = cls.__new__(cls)
        obj.model_type = payload["model_type"]
        obj.scaler = payload.get("scaler")
        if obj.model_type == "ensemble":
            obj._sub_models = payload["sub_models"]
            obj._weights = payload["weights"]
            obj.model = None
        else:
            obj.model = payload["model"]
            obj._sub_models = {}
            obj._weights = {}
        return obj
