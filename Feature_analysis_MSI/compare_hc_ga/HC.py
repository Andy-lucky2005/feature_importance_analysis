from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.cluster.hierarchy import linkage, optimal_leaf_ordering, leaves_list
from scipy.spatial.distance import squareform

# =========================
# Config
# =========================
INPUT_XLSX = "all_sort.xlsx"
USE_ABS_CORR = True
RANDOM_SEED = 42

LINKAGE_METHODS = [
    "single",
    "complete",
    "average",
    "weighted",
    "centroid",
    "median",
    "ward",
]

# =========================
# Data
# =========================
def load_data(path: str) -> pd.DataFrame:
    df = pd.read_excel(path, header=0)
    df = df.iloc[:, 1:]
    df = df.apply(pd.to_numeric, errors="coerce")
    df = df.dropna(axis=1, how="all").dropna(axis=0)
    return df


def compute_corr(df: pd.DataFrame) -> pd.DataFrame:
    corr = df.corr(method="spearman")
    return corr.abs() if USE_ABS_CORR else corr


def corr_to_distance(corr: np.ndarray) -> np.ndarray:
    dist = 1.0 - corr
    np.fill_diagonal(dist, 0.0)
    return dist


# =========================
# HC ordering
# =========================
def hierarchical_order(corr_df: pd.DataFrame, method: str):
    dist = corr_to_distance(corr_df.values)
    condensed = squareform(dist, checks=False)

    Z = linkage(condensed, method=method)
    Z = optimal_leaf_ordering(Z, condensed)

    order = leaves_list(Z)
    return order


# =========================
# Loss (same as paper)
# =========================
def generate_target_matrix(n, decay_rate=0.04):
    target = np.zeros((n, n))
    for i in range(n):
        for j in range(n):
            d = abs((n - 1) - (i + j))
            target[i, j] = max(0.0, 1.0 - decay_rate * d)
    return target


def compute_loss(matrix, target):
    loss = 0.0
    n = matrix.shape[0]

    for i in range(n):
        for j in range(n):
            if target[i, j] > 0.5:
                diff = target[i, j] - matrix[i, j]
                if diff > 0:
                    loss += diff ** 2
            else:
                diff = matrix[i, j] - target[i, j]
                if diff > 0:
                    loss += diff ** 2
    return loss


# =========================
# Main
# =========================
def main():
    np.random.seed(RANDOM_SEED)

    df = load_data(INPUT_XLSX)
    method_names = df.columns.tolist()

    corr = compute_corr(df)

    target = generate_target_matrix(len(method_names))

    results = []

    print("\n=== HC Comparison Across Linkage Methods ===")

    for method in LINKAGE_METHODS:

        order = hierarchical_order(corr, method)

        ordered_corr = corr.values[np.ix_(order, order)]
        ordered_corr = np.fliplr(ordered_corr)

        loss = compute_loss(ordered_corr, target)

        results.append({
            "linkage": method,
            "loss": loss
        })

        print(f"{method:10s} | Loss = {loss:.6f}")

    # summary table
    df_res = pd.DataFrame(results).sort_values("loss")
    print("\n=== Summary (sorted) ===")
    print(df_res.to_string(index=False))


if __name__ == "__main__":
    main()