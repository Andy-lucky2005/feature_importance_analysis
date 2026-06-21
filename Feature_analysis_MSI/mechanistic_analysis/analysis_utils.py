"""SHAP, PDP, nonlinearity, and attribution utilities."""

from __future__ import annotations

import numpy as np
import pandas as pd
import shap
from scipy.stats import pearsonr
from sklearn.metrics import r2_score
from statsmodels.stats.outliers_influence import variance_inflation_factor

from config import BG_SIZE, FCOLS, KE_NSAMP, MODEL_ORDER
def compute_pdp_store(fitted_models: dict, X_std: np.ndarray, n_grid: int = 100) -> dict:
    """Compute PDP in standardized space for all models and features."""
    pdp_store = {}
    for name in MODEL_ORDER:
        est = fitted_models[name].named_steps["model"]
        for j, feat in enumerate(FCOLS):
            g = np.linspace(X_std[:, j].min(), X_std[:, j].max(), n_grid)
            pdp = np.zeros_like(g)
            for k, gv in enumerate(g):
                Xt = X_std.copy()
                Xt[:, j] = gv
                pdp[k] = est.predict(Xt).mean()
            pdp_store[(name, feat)] = (g, pdp)
    return pdp_store

def compute_ice_store(fitted_models: dict, X_std: np.ndarray, n_grid: int = 100) -> dict:
    """
    Compute ICE (Individual Conditional Expectation) for all models and features.
    Returns a dictionary: key=(model,feature), value=(grid, ice_array)
    ice_array.shape = (n_samples, n_grid)
    """
    ice_store = {}
    for name in MODEL_ORDER:
        est = fitted_models[name].named_steps["model"]
        for j, feat in enumerate(FCOLS):
            # 构建网格
            g = np.linspace(X_std[:, j].min(), X_std[:, j].max(), n_grid)
            ice = np.zeros((X_std.shape[0], n_grid))
            for k, gv in enumerate(g):
                Xt = X_std.copy()
                Xt[:, j] = gv
                ice[:, k] = est.predict(Xt)
            ice_store[(name, feat)] = (g, ice)
    return ice_store

def rank_features_from_attribution(signed_contribs: dict[str, np.ndarray]) -> np.ndarray:
    mat_all = np.array([signed_contribs[m] for m in MODEL_ORDER])
    return np.argsort(np.abs(mat_all).mean(axis=0))[::-1]
