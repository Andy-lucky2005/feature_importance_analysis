
"""Data loading and standardization utilities."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from config import FCOLS, FLABELS

EXPECTED_COLUMNS = [
    "Sample", "Eadh",
    "Xp_M", "Xp_Mp", "IE_M", "IE_Mp",
    "r_M", "r_Mp", "Hf_MO", "Hf_MpO", "Hf_MpM",
    "Hsub_M", "Hsub_Mp", "gamma_M", "Nws_M", "Eg_MpO",
]


def load_dataset(data_path: Path) -> tuple[pd.DataFrame, np.ndarray, np.ndarray]:
    """Load the Excel file and return raw X and y."""
    df = pd.read_excel(data_path)

    # Keep compatibility with the original file format.
    if df.shape[1] == len(EXPECTED_COLUMNS):
        df.columns = EXPECTED_COLUMNS
    else:
        # Do not fail silently: the user should know if the file layout changed.
        missing = set(EXPECTED_COLUMNS) - set(df.columns)
        if missing:
            raise ValueError(
                f"Input file structure does not match the expected columns. Missing: {sorted(missing)}"
            )

    X_raw = df[FCOLS].to_numpy(dtype=float)
    y = df["Eadh"].to_numpy(dtype=float)
    return df, X_raw, y


def standardize_features(X_raw: np.ndarray) -> tuple[StandardScaler, np.ndarray]:
    """Fit a StandardScaler and transform the raw features."""
    scaler = StandardScaler()
    X_std = scaler.fit_transform(X_raw)
    return scaler, X_std


def feature_names() -> list[str]:
    return [FLABELS[c] for c in FCOLS]


def inverse_standardize_feature(values_std: np.ndarray, scaler: StandardScaler, feature_idx: int) -> np.ndarray:
    """Map a single feature from standardized space back to raw units."""
    return values_std * scaler.scale_[feature_idx] + scaler.mean_[feature_idx]
