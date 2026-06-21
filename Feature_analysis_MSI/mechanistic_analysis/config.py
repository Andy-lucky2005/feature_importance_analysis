
"""Central configuration for model-mechanism analysis."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

DATA_PATH = Path("Science_feature_data.xlsx")
OUT_DIR = Path("mech_analysis")
OUT_DIR.mkdir(parents=True, exist_ok=True)

SEED = 1412
N_GRID = 100
BG_SIZE = 50
KE_NSAMP = 120

MODEL_ORDER = ["LR", "SVR", "MLP", "RF", "GBRT", "XGBoost"]
MODEL_COLORS = {
    "LR": "#3498db",
    "SVR": "#9b59b6",
    "MLP": "#e74c3c",
    "RF": "#27ae60",
    "GBRT": "#f39c12",
    "XGBoost": "#2c3e50",
}

FCOLS = [
    "Xp_M", "Xp_Mp", "IE_M", "IE_Mp",
    "r_M", "r_Mp", "Hf_MO", "Hf_MpO", "Hf_MpM",
    "Hsub_M", "Hsub_Mp", "gamma_M", "Nws_M", "Eg_MpO",
]

FLABELS = {
    "Xp_M":    r"$\chi_p^M$",
    "Xp_Mp":   r"$\chi_p^{M'}$",
    "IE_M":    r"$IE^M$",
    "IE_Mp":  r"$IE^{M'}$",
    "r_M":     r"$r^M$",
    "r_Mp":    r"$r^{M'}$",
    "Hf_MO":   r"$\Delta H_f^{MO}$",
    "Hf_MpO":  r"$\Delta H_f^{M'O}$",
    "Hf_MpM":  r"$\Delta H_f^{M'M}$",
    "Hsub_M":  r"$\Delta H_{sub}^M$",
    "Hsub_Mp": r"$\Delta H_{sub}^{M'}$",
    "gamma_M": r"$\gamma^M$",
    "Nws_M":   r"$n_{ws}^M$",
    "Eg_MpO":  r"$E_g^{M'O}$",
}

# Hyperparameters are centralized here so reviewer-facing changes only need to be made once.
MODEL_PARAMS = {
    "LR": {},
    "SVR": {
        "kernel": "rbf",
        "C": 28.92583740213864,
        "epsilon": 0.09661424638635481,
        "gamma": 0.01042110123339548,
    },
    "MLP": {
        "hidden_layer_sizes": (40, 8),
        "activation": "relu",
        "solver": "lbfgs",
        "max_iter": 5000,
        "alpha": 0.009885444029062893,
    },
    "RF": {
        "n_estimators": 66,
        "max_depth": 9,
        "max_features": 7,
        "min_samples_split": 2,
        "min_samples_leaf": 1,
    },
    "GBRT": {
        "n_estimators": 266,
        "learning_rate": 0.07400928184149287,
        "max_depth": 3,
        "subsample": 0.8314787464970493,
        "min_samples_split": 9,
        "min_samples_leaf": 3,
    },
    "XGBoost": {
        "max_depth": 3,
        "learning_rate": 0.09222929445288928,
        "n_estimators": 199,
        "subsample": 0.5003718198735058,
        "colsample_bytree": 0.8310917506470121,
        "gamma": 0.13566644039562242,
        "reg_alpha": 0.029867618709022974,
        "reg_lambda": 0.0005818694744638248,
    },
}
