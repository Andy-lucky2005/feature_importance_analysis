"""Main entry point for the modular model-mechanism analysis."""

from __future__ import annotations
from analysis_utils import (
    compute_pdp_store,
    compute_ice_store,
)
from config import DATA_PATH, FCOLS, FLABELS, MODEL_ORDER, OUT_DIR, SEED
from data_utils import feature_names, load_dataset, standardize_features
from model_utils import (
    build_and_fit_models,
)
from plot_utils_plus import (
    plot_f2_pdp_grid,
    plot_f3_overlay,
    export_pdp_json,          # ← NEW: exports JSON for Plotly web page
)
from plot_utils_plotly import plot_f2_pdp_grid_plotly


def main():
    print("=" * 72)
    print("Step 1: Loading data")
    print("=" * 72)
    df, X_raw, y = load_dataset(DATA_PATH)
    scaler, X_std = standardize_features(X_raw)

    fnames = feature_names()
    print(f"Samples : {len(df)}")
    print(f"Features: {len(FCOLS)}")
    print(f"Target  : [{y.min():.3f}, {y.max():.3f}] J/m²")

    print("\n" + "=" * 72)
    print("Step 2: Fitting models")
    print("=" * 72)
    fitted_models, perf = build_and_fit_models(X_raw, y)

    print("\n" + "=" * 72)
    print("Step 3: PDP / ICE computation")
    print("=" * 72)
    # Pre-computed feature importance order (skip SHAP for speed)
    feat_rank_idx = [5, 3, 9, 7, 0, 1, 4, 12, 13,  10,11, 6,8, 2 ]
    # feat_rank_idx = feat_rank_idx[-2:]
    feat_rank_idx_plot3 = [10, 6, 9, 7, 3, 2, 4, 12, 13, 8, 1, 11, 5, 0]
    pdp_store = compute_pdp_store(fitted_models, X_std, n_grid=100)
    ice_store = compute_ice_store(fitted_models, X_std, n_grid=100)

    print("\n" + "=" * 72)
    print("Step 4: Plotting figures")
    print("=" * 72)

    # F2: pass X_raw (raw units) so SISSO PDP is evaluated correctly,
    #     and y for normalisation + per-model Pearson/Spearman computation.
    plot_f2_pdp_grid(pdp_store, feat_rank_idx, scaler, X_raw, y, OUT_DIR)
    print("  F2 saved.")

    # F3 overlay: now accepts X_raw and y for normalised scatter
    plot_f3_overlay(pdp_store, feat_rank_idx_plot3, scaler, X_raw, y, OUT_DIR)
    print("  F3 overlay saved.")

    # ── Export JSON for the interactive Plotly web page ───────────────────
    json_path = OUT_DIR / "pdp_data.json"
    export_pdp_json(pdp_store, feat_rank_idx, scaler, X_raw, y, json_path)
    print(f"  JSON exported → {json_path}")

    print("Step 5: Plotly interactive F2 export (HTML for GitHub Pages)")
    print("=" * 72)
    plot_f2_pdp_grid_plotly(pdp_store, feat_rank_idx, scaler, X_raw, y, OUT_DIR)

if __name__ == "__main__":
    main()