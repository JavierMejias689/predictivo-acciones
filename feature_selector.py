"""Selección automática de features para reducir ruido y mejorar generalización."""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.feature_selection import mutual_info_classif

import config


def _remove_highly_correlated(X: pd.DataFrame, threshold: float = 0.95) -> list[str]:
    """Identifica columnas con correlación > threshold para eliminar."""
    corr = X.corr().abs()
    upper = corr.where(np.triu(np.ones(corr.shape), k=1).astype(bool))
    to_drop = [col for col in upper.columns if any(upper[col] > threshold)]
    return to_drop


def _mutual_info_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Calcula mutual information entre cada feature y el target."""
    mi = mutual_info_classif(X, y, random_state=config.RANDOM_STATE, n_neighbors=5)
    return pd.Series(mi, index=X.columns)


def _rf_importance_scores(X: pd.DataFrame, y: pd.Series) -> pd.Series:
    """Importancia de features usando un RandomForest rápido."""
    rf = RandomForestClassifier(
        n_estimators=200,
        max_depth=8,
        random_state=config.RANDOM_STATE,
        n_jobs=-1,
    )
    rf.fit(X, y)
    return pd.Series(rf.feature_importances_, index=X.columns)


def select_features(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_top: int | None = None,
) -> list[str]:
    """Selecciona los mejores features combinando mutual info + RF importance.

    1. Elimina features con correlación > 0.95
    2. Rankea por mutual information + RF importance (promedio de rankings)
    3. Retorna los top-N features
    """
    n_top = n_top or config.N_TOP_FEATURES

    # Paso 1: eliminar altamente correlacionados
    corr_drop = _remove_highly_correlated(X_train)
    X_filtered = X_train.drop(columns=corr_drop, errors="ignore")

    if X_filtered.shape[1] <= n_top:
        return list(X_filtered.columns)

    # Paso 2: scores
    mi_scores = _mutual_info_scores(X_filtered, y_train)
    rf_scores = _rf_importance_scores(X_filtered, y_train)

    # Paso 3: ranking combinado (promedio de posiciones normalizadas)
    mi_rank = mi_scores.rank(ascending=False)
    rf_rank = rf_scores.rank(ascending=False)
    combined_rank = (mi_rank + rf_rank) / 2.0
    combined_rank = combined_rank.sort_values()

    selected = list(combined_rank.head(n_top).index)
    return selected
