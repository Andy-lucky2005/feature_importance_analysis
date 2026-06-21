"""Figure generation utilities."""

from __future__ import annotations

from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import matplotlib.ticker as ticker
from matplotlib.colors import LinearSegmentedColormap
from scipy.stats import pearsonr, spearmanr

from config import FCOLS, FLABELS, MODEL_COLORS, MODEL_ORDER


# ─────────────────────────────────────────────────────────────────────────────
# Internal helpers
# ─────────────────────────────────────────────────────────────────────────────

def _cmap_div():
    return LinearSegmentedColormap.from_list("div", ["#1a6faf", "white", "#c0392b"])

def _norm_curve(arr: np.ndarray) -> np.ndarray:
    """Normalise a 1-D array to [0, 1] using its own min/max."""
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)

def _norm_scatter(y: np.ndarray) -> np.ndarray:
    """Normalise the experimental y vector to [0, 1] using global min/max."""
    lo, hi = float(y.min()), float(y.max())
    return (y - lo) / (hi - lo + 1e-12)

def _norm_feature(x: np.ndarray) -> np.ndarray:
    lo = np.min(x)
    hi = np.max(x)
    return (x - lo) / (hi - lo + 1e-12)


# ─────────────────────────────────────────────────────────────────────────────
# SISSO formula   (operates in raw, un-standardised feature space)
# ─────────────────────────────────────────────────────────────────────────────

_SISSO_IDX: dict[str, int] = {
    "Xp_M": 0, "Xp_Mp": 1, "IE_M": 2,
    "Hf_MO": 6, "Hf_MpO": 7, "Hf_MpM": 8,
    "Hsub_M": 9, "Hsub_Mp": 10, "gamma_M": 11,
}
SISSO_FEAT_IDX_SET: set[int] = set(_SISSO_IDX.values())


def _sisso_predict(X_raw: np.ndarray) -> np.ndarray:
    """Evaluate the SISSO formula on every row of X_raw."""
    a1       = 9.85 * 10**-2
    a2       = -3.06 * 10**-3
    constant = 16 * 0.0160217

    gamma_M = X_raw[:, 11];  Xp_Mp  = X_raw[:, 1];  Hf_MO   = X_raw[:, 6]
    Hsub_M  = X_raw[:, 9];   Xp_M   = X_raw[:, 0];  IE_M    = X_raw[:, 2]
    Hf_MpM  = X_raw[:, 8];   Hsub_Mp= X_raw[:, 10]; Hf_MpO  = X_raw[:, 7]

    term1 = (a1 * gamma_M * Xp_Mp * (Hf_MO - Hsub_M)) / Xp_M
    term2 = (a2 * IE_M * (Hf_MpM - Hsub_M - Hsub_Mp)) / Hf_MpO * 16.0217
    return abs(term1 + term2 + constant)


def _sisso_pdp(X_raw: np.ndarray, feat_idx: int,
               n_grid: int = 100) -> tuple[np.ndarray, np.ndarray]:
    """PDP of the SISSO formula for feature `feat_idx` (raw units)."""
    g   = np.linspace(X_raw[:, feat_idx].min(),
                      X_raw[:, feat_idx].max(), n_grid)
    pdp = np.zeros(n_grid)
    for k, gv in enumerate(g):
        Xt = X_raw.copy()
        Xt[:, feat_idx] = gv
        pdp[k] = _sisso_predict(Xt).mean()
    return g, pdp

# ─────────────────────────────────────────────────────────────────────────────
# Figure 2   PDP grid
# ─────────────────────────────────────────────────────────────────────────────
def _apply_panel_style(ax):
    ax.set_xlim(-0.07, 1.07)
    ax.set_ylim(-0.07, 1.07)
    ax.set_xticks([0,  0.5,  1])
    ax.set_yticks([0, 0.5,  1])
    ax.grid(alpha=0.2, lw=0.5)


def plot_f2_pdp_grid(
    pdp_store: dict,
    feat_rank_idx: np.ndarray,
    scaler,
    X_raw: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
):
    """
    14 features × (6 ML models + 1 SISSO) PDP grid.

    Normalisation:
      • Each PDP curve normalised independently to [0, 1]  →  _norm_curve
      • Experimental scatter y normalised globally         →  _norm_scatter
      • Feature x normalised per-feature to [0, 1]

    Layout:
      • 14 rows × 7 cols (6 ML + SISSO)
      • Y-axis tick labels: leftmost column only
      • X-axis tick labels: bottom row only
      • SISSO column styled identically to ML columns
      • Features absent from SISSO shown as grey "N/A" panels
    """
    all_cols  = [FCOLS[i] for i in feat_rank_idx]
    n_feats   = len(all_cols)      # 14
    n_ml      = len(MODEL_ORDER)   # 6
    n_cols    = n_ml + 1           # 7  (6 ML + SISSO)
    sisso_color = "#16a085"

    y_norm_scatter = _norm_scatter(y)

    fig, axes = plt.subplots(n_feats, n_cols, figsize=(28, n_feats * 3.4))
    for row, feat in enumerate(all_cols):
        fidx  = FCOLS.index(feat)
        x_raw = X_raw[:, fidx]

        # ── Per-feature x normalisation (shared by all columns in this row) ──
        x_min  = float(x_raw.min())
        x_max  = float(x_raw.max())
        x_rng  = x_max - x_min + 1e-12
        x_norm = (x_raw - x_min) / x_rng

        # ── ML model columns 0–5 ──────────────────────────────────────────
        for col, name in enumerate(MODEL_ORDER):
            ax    = axes[row, col]
            color = MODEL_COLORS[name]

            g_std, pv_raw = pdp_store[(name, feat)]
            g_raw  = g_std * scaler.scale_[fidx] + scaler.mean_[fidx]
            pv_norm = _norm_curve(pv_raw)
            g_norm  = (g_raw - x_min) / x_rng

            ax.scatter(x_norm, y_norm_scatter,
                       s=14, alpha=0.40, color="black", zorder=2, linewidths=0,
                       label="Exp" if (row == 0 and col == 0) else None)
            ax.plot(g_norm, pv_norm, color=color, lw=2.2, zorder=3)

            _apply_panel_style(ax)

            if row == 0:
                ax.set_title(name, fontsize=25, fontweight="bold", color=color, pad=6)
            if col == 0:
                ax.set_ylabel(FLABELS[feat], fontsize=27)

            # ── Tick-label visibility ─────────────────────────────────────
            if col != 0:
                ax.tick_params(axis='y', left=False, labelleft=False)
            if row != n_feats - 1:
                ax.tick_params(axis='x', bottom=False, labelbottom=False)
            ax.tick_params(axis='both', labelsize=23.5)

        # ── SISSO column (col = 6) ────────────────────────────────────────
        ax_s = axes[row, n_ml]

        if fidx in SISSO_FEAT_IDX_SET:
            # Feature is in the SISSO formula → plot PDP
            g_s, pv_s_raw = _sisso_pdp(X_raw, fidx, n_grid=100)
            pv_s_norm = _norm_curve(pv_s_raw)
            g_s_norm  = (g_s - x_min) / x_rng

            ax_s.scatter(x_norm, y_norm_scatter,
                         s=14, alpha=0.40, color="black", zorder=2, linewidths=0)
            ax_s.plot(g_s_norm, pv_s_norm, color=sisso_color, lw=2.2, zorder=3)
        else:
            # Feature absent from SISSO formula → grey placeholder
            # ax_s.set_facecolor("#f0f0f0")
            ax_s.text(0.5, 0.5, "Feature\n(not in\nSISSO)",
                      ha="center", va="center", transform=ax_s.transAxes,
                      fontsize=13, color="#c0c0c0", style="italic")

        # ── Unified SISSO styling  (rightmost column → no y-axis labels) ──
        _apply_panel_style(ax_s)
        ax_s.tick_params(axis='y', left=False, labelleft=False)
        if row != n_feats - 1:
            ax_s.tick_params(axis='x', bottom=False, labelbottom=False)
        ax_s.tick_params(axis='both', labelsize=23.5)

        if row == 0:
            ax_s.set_title("SISSO", fontsize=23, fontweight="bold",
                           color=sisso_color, pad=6)

    plt.subplots_adjust(
        hspace = 0.05,
        wspace = 0.05,
        left   = 0.14,
        right  = 0.99,
        top    = 0.97,
        bottom=max(0.04, 0.05 * (n_feats / 14))
    )
    # 全局 y 轴含义标签：放在最左侧中间
    fig.text(
        0.05, 0.5,
        "Normalized $E_{adh}$",
        rotation=90,
        va="center",
        ha="center",
        fontsize=26
    )

    # One global x-axis label at the very bottom of the figure
    fig.text(0.55, 0.03, "Normalized feature value", ha="center", va="bottom", fontsize=25)
    fig.savefig(out_dir / "F2_pdp_6models_14features.pdf",
                dpi=300, bbox_inches="tight")
    plt.close(fig)


# ─────────────────────────────────────────────────────────────────────────────
# Figure 3   6-model PDP overlay
# ─────────────────────────────────────────────────────────────────────────────
def plot_f3_overlay(
    pdp_store: dict,
    feat_rank_idx: np.ndarray,
    scaler,
    X_raw: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
):
    """
    Top-6 feature PDP overlay, all 6 models in one panel per feature.

    Normalisation:
      • Each PDP curve normalised independently to [0, 1]  (_norm_curve)
      • Experimental scatter y normalised globally         (_norm_scatter)

    Labels:
      • Feature name shown as bold text annotation inside each panel
      • Y-axis label shown on leftmost column only  (ii % 3 == 0)
      • One global x-axis label at the figure bottom via fig.text
    """
    top6_cols = [FCOLS[i] for i in feat_rank_idx[:6]]
    y_norm_scatter = _norm_scatter(y)

    fig, axes = plt.subplots(2, 3, figsize=(14, 9.2), sharex=True, sharey=True)

    for ii, feat in enumerate(top6_cols):
        fidx  = FCOLS.index(feat)
        ax    = axes[ii // 3, ii % 3]
        x_raw = X_raw[:, fidx]
        x_norm = _norm_feature(x_raw)

        ax.scatter(x_norm, y_norm_scatter,
                   s=20, alpha=0.45, color="#555555",
                   zorder=1, linewidths=0, label="Experiment")

        for name in MODEL_ORDER:
            g_std, pv_raw = pdp_store[(name, feat)]
            pv_norm = _norm_curve(pv_raw)
            g_raw   = g_std * scaler.scale_[fidx] + scaler.mean_[fidx]
            x_min   = x_raw.min()
            x_max   = x_raw.max()
            g_norm  = (g_raw - x_min) / (x_max - x_min + 1e-12)
            ax.plot(g_norm, pv_norm,
                    color=MODEL_COLORS[name], lw=2.2,
                    label=name, alpha=0.90, zorder=2)

        ax.text(0.03, 0.97, FLABELS[feat],
                transform=ax.transAxes, fontsize=22, fontweight="bold",
                va="top", ha="left",
                bbox=dict(boxstyle="round,pad=0.2", fc="white", alpha=0.75, ec="none"))

        _apply_panel_style(ax)
        ax.tick_params(axis='both', labelsize=14)

        # 只保留最左侧一列的 y 刻度/刻度值
        if ii % 3 != 0:
            ax.tick_params(axis='y', which='both', left=False, labelleft=False)

        # 只保留最底下一行的 x 刻度/刻度值
        if ii // 3 != 1:
            ax.tick_params(axis='x', which='both', bottom=False, labelbottom=False)

    # 左侧全局 y 轴标签
    fig.text(
        0.04, 0.5,
        "Normalized $E_{adh}$",
        rotation=90,
        va="center",
        ha="center",
        fontsize=15
    )

    # 底部全局 x 轴标签
    fig.text(
        0.5, 0.02,
        "Normalized feature value",
        ha="center",
        va="bottom",
        fontsize=15
    )

    # 更紧凑
    plt.subplots_adjust(
        left=0.08,
        right=0.995,
        bottom=0.15,
        top=0.97,
        wspace=0.03,
        hspace=0.03
    )

    fig.savefig(out_dir / "F3_pdp_overlay.pdf", dpi=300, bbox_inches="tight")
    plt.close(fig)

# ─────────────────────────────────────────────────────────────────────────────
# JSON export   (for Plotly web page)  — unchanged
# ─────────────────────────────────────────────────────────────────────────────

def export_pdp_json(
    pdp_store: dict,
    feat_rank_idx: np.ndarray,
    scaler,
    X_raw: np.ndarray,
    y: np.ndarray,
    out_path: Path,
):
    """
    Serialise all PDP data (ML models + SISSO) to JSON for the Plotly page.

    Normalisation matches the Python plots:
      • PDP curves  → _norm_curve  (each curve independently [0,1])
      • Scatter y   → _norm_scatter (global y min/max)
    """
    import json

    y_norm_scatter = _norm_scatter(y)
    feat_order     = [FCOLS[i] for i in feat_rank_idx]
    sisso_feats    = [FCOLS[i] for i in sorted(SISSO_FEAT_IDX_SET)]
    model_colors_ext = {**MODEL_COLORS, "SISSO": "#16a085"}
    models_ext       = list(MODEL_ORDER) + ["SISSO"]

    out: dict = {
        "feat_order":   feat_order,
        "feat_labels":  {k: v for k, v in FLABELS.items()},
        "sisso_feats":  sisso_feats,
        "models":       models_ext,
        "model_colors": model_colors_ext,
        "features":     {},
    }

    for feat in feat_order:
        fidx  = FCOLS.index(feat)
        x_raw = X_raw[:, fidx]

        try:
            pr_exp, _ = pearsonr(x_raw, y)
        except Exception:
            pr_exp = float("nan")
        try:
            sr_exp, _ = spearmanr(x_raw, y)
        except Exception:
            sr_exp = float("nan")

        feat_entry: dict = {
            "x_raw":        x_raw.tolist(),
            "y_norm":       y_norm_scatter.tolist(),
            "pearson_exp":  round(float(pr_exp), 4),
            "spearman_exp": round(float(sr_exp), 4),
            "models":       {},
        }

        # ML models
        for name in MODEL_ORDER:
            g_std, pv_raw = pdp_store[(name, feat)]
            g_raw   = g_std * scaler.scale_[fidx] + scaler.mean_[fidx]
            pv_norm = _norm_curve(pv_raw)
            x_min_f = float(X_raw[:, fidx].min())
            x_max_f = float(X_raw[:, fidx].max())
            g_norm  = (g_raw - x_min_f) / (x_max_f - x_min_f + 1e-12)
            feat_entry["models"][name] = {
                "g_norm":  g_norm.tolist(),
                "pv_norm": pv_norm.tolist(),
            }

        # SISSO
        if fidx in SISSO_FEAT_IDX_SET:
            g_s, pv_s_raw = _sisso_pdp(X_raw, fidx, n_grid=100)
            pv_s_norm = _norm_curve(pv_s_raw)
            x_min_f = float(X_raw[:, fidx].min())
            x_max_f = float(X_raw[:, fidx].max())
            g_s_norm = (g_s - x_min_f) / (x_max_f - x_min_f + 1e-12)
            feat_entry["models"]["SISSO"] = {
                "g_norm":  g_s_norm.tolist(),
                "pv_norm": pv_s_norm.tolist(),
            }
        else:
            feat_entry["models"]["SISSO"] = None

        out["features"][feat] = feat_entry

    with open(out_path, "w", encoding="utf-8") as fh:
        json.dump(out, fh, ensure_ascii=False, indent=2)

    print(f"[export_pdp_json] Written → {out_path}  "
          f"({out_path.stat().st_size // 1024} KB)")