
"""Model creation, cross-validation, and fitting utilities."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xgboost as xgb
from sklearn.base import clone
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold, cross_val_score
from sklearn.neural_network import MLPRegressor
from sklearn.pipeline import Pipeline
from sklearn.svm import SVR
from sklearn.preprocessing import StandardScaler

from config import MODEL_ORDER, MODEL_PARAMS, SEED


def make_pipeline(model_name: str) -> Pipeline:
    """Build a StandardScaler + estimator pipeline for the given model."""
    params = MODEL_PARAMS[model_name]

    if model_name == "LR":
        estimator = LinearRegression()
    elif model_name == "SVR":
        estimator = SVR(**params)
    elif model_name == "MLP":
        estimator = MLPRegressor(random_state=SEED, **params)
    elif model_name == "RF":
        estimator = RandomForestRegressor(random_state=SEED, **params)
    elif model_name == "GBRT":
        estimator = GradientBoostingRegressor(random_state=SEED, **params)
    elif model_name == "XGBoost":
        estimator = xgb.XGBRegressor(random_state=SEED, objective="reg:squarederror", **params)
    else:
        raise KeyError(f"Unknown model: {model_name}")

    return Pipeline([
        ("scaler", StandardScaler()),
        ("model", estimator),
    ])


def build_and_fit_models(X_raw: np.ndarray, y: np.ndarray) -> tuple[dict[str, Pipeline], dict[str, dict]]:
    """
    Fit all models with the same preprocessing pipeline.
    Returns fitted pipelines and performance metrics.
    """
    fitted = {}
    perf: dict[str, dict] = {}

    for name in MODEL_ORDER:
        pipe = make_pipeline(name)
        cv = cross_val_score(pipe, X_raw, y, cv=10, scoring="r2")
        pipe.fit(X_raw, y)
        fitted[name] = pipe

        train_r2 = pipe.score(X_raw, y)
        perf[name] = {
            "train": float(train_r2),
            "cv": float(cv.mean()),
            "cv_std": float(cv.std()),
        }

    return fitted, perf


def get_transformed_X(fitted_pipeline: Pipeline, X_raw: np.ndarray) -> np.ndarray:
    """Get the standardized feature matrix used by a fitted pipeline."""
    return fitted_pipeline.named_steps["scaler"].transform(X_raw)


def get_estimator(fitted_pipeline: Pipeline):
    """Extract the fitted estimator from the pipeline."""
    return fitted_pipeline.named_steps["model"]


def predict_all(fitted_models: dict[str, Pipeline], X_raw: np.ndarray) -> dict[str, np.ndarray]:
    return {name: pipe.predict(X_raw) for name, pipe in fitted_models.items()}


def prediction_r2_matrix(preds: dict[str, np.ndarray], model_order: list[str]) -> np.ndarray:
    n = len(model_order)
    mat = np.zeros((n, n), dtype=float)
    for i, m1 in enumerate(model_order):
        for j, m2 in enumerate(model_order):
            a = preds[m1].reshape(-1)
            b = preds[m2].reshape(-1)
            ss_res = np.sum((a - b) ** 2)
            ss_tot = np.sum((a - a.mean()) ** 2)
            mat[i, j] = max(0.0, 1.0 - ss_res / (ss_tot + 1e-12))
    return mat
