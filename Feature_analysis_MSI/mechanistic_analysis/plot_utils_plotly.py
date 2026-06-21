from __future__ import annotations

from pathlib import Path

import numpy as np
import plotly.graph_objects as go
from plotly.subplots import make_subplots

from config import FCOLS, FLABELS, MODEL_COLORS, MODEL_ORDER
from plot_utils_plus import SISSO_FEAT_IDX_SET, _sisso_pdp   # reuse from main module


# ─── Inline normalisers — identical to plot_utils.py ───────────────────────

def _norm_curve(arr: np.ndarray) -> np.ndarray:
    lo, hi = arr.min(), arr.max()
    return (arr - lo) / (hi - lo + 1e-12)


def _norm_scatter(y: np.ndarray) -> np.ndarray:
    lo, hi = float(y.min()), float(y.max())
    return (y - lo) / (hi - lo + 1e-12)


# ─── Main function ──────────────────────────────────────────────────────────

def plot_f2_pdp_grid_plotly(
    pdp_store: dict,
    feat_rank_idx,             # list[int] | np.ndarray
    scaler,                    # fitted StandardScaler
    X_raw: np.ndarray,
    y: np.ndarray,
    out_dir: Path,
) -> None:
    """
    Build and save a standalone HTML interactive figure that exactly
    reproduces the visual layout of plot_f2_pdp_grid().

    Parameters are identical to the matplotlib version.
    Output:  <out_dir>/F2_pdp_interactive.html
    """
    SISSO_COLOR = "#16a085"

    all_cols   = [FCOLS[i] for i in feat_rank_idx]
    n_rows     = len(all_cols)       # 14
    n_ml       = len(MODEL_ORDER)    # 6
    n_cols     = n_ml + 1            # 7  (6 ML + SISSO)

    all_models = list(MODEL_ORDER) + ["SISSO"]
    all_colors = {**MODEL_COLORS, "SISSO": SISSO_COLOR}

    y_norm = _norm_scatter(y)

    # ── 1. Build 14 × 7 subplot grid ──────────────────────────────────────
    # horizontal_spacing / vertical_spacing: fractions of total plot area.
    # 0.008 / 0.005 produces the same tight packing as
    # subplots_adjust(hspace=0.04, wspace=0.04) in matplotlib.
    fig = make_subplots(
        rows               = n_rows,
        cols               = n_cols,
        horizontal_spacing = 0.008,
        vertical_spacing   = 0.005,
    )

    # ── 2. Add traces ──────────────────────────────────────────────────────
    for ri, feat in enumerate(all_cols):
        fidx  = FCOLS.index(feat)
        x_raw = X_raw[:, fidx]
        r     = ri + 1

        # Per-feature x normalisation — same for all 7 columns in this row
        x_min  = float(x_raw.min())
        x_max  = float(x_raw.max())
        x_rng  = x_max - x_min + 1e-12
        x_norm = (x_raw - x_min) / x_rng

        # ── ML columns 1–6 ────────────────────────────────────────────────
        for ci, name in enumerate(MODEL_ORDER):
            c     = ci + 1
            g_std, pv_raw = pdp_store[(name, feat)]
            color = MODEL_COLORS[name]
            g_raw = g_std * scaler.scale_[fidx] + scaler.mean_[fidx]
            pv_n  = _norm_curve(pv_raw)
            g_n   = (g_raw - x_min) / x_rng

            # Experimental scatter (black, semi-transparent)
            fig.add_trace(go.Scatter(
                x             = x_norm.tolist(),
                y             = y_norm.tolist(),
                mode          = "markers",
                marker        = dict(size=3, color="rgba(0,0,0,0.40)",
                                     line=dict(width=0)),
                showlegend    = False,
                hovertemplate = "x̃: %{x:.2f}<br>Ẽadh: %{y:.2f}<extra>Exp</extra>",
            ), row=r, col=c)

            # PDP curve (model colour, lw=2.2 matches matplotlib)
            fig.add_trace(go.Scatter(
                x             = g_n.tolist(),
                y             = pv_n.tolist(),
                mode          = "lines",
                line          = dict(color=color, width=2.2),
                showlegend    = False,
                hovertemplate = (
                    f"x̃: %{{x:.2f}}<br>PDP: %{{y:.2f}}<extra>{name}</extra>"
                ),
            ), row=r, col=c)

        # ── SISSO column (col 7) ──────────────────────────────────────────
        c_s = n_cols

        if fidx in SISSO_FEAT_IDX_SET:
            g_s, pv_s_raw = _sisso_pdp(X_raw, fidx, n_grid=100)
            pv_s_n = _norm_curve(pv_s_raw)
            g_s_n  = (g_s - x_min) / x_rng

            fig.add_trace(go.Scatter(
                x             = x_norm.tolist(),
                y             = y_norm.tolist(),
                mode          = "markers",
                marker        = dict(size=4, color="rgba(0,0,0,0.40)",
                                     line=dict(width=0)),
                showlegend    = False,
                hovertemplate = "x̃: %{x:.2f}<br>Ẽadh: %{y:.2f}<extra>Exp</extra>",
            ), row=r, col=c_s)

            fig.add_trace(go.Scatter(
                x             = g_s_n.tolist(),
                y             = pv_s_n.tolist(),
                mode          = "lines",
                line          = dict(color=SISSO_COLOR, width=2.2),
                showlegend    = False,
                hovertemplate = "x̃: %{x:.2f}<br>PDP: %{y:.2f}<extra>SISSO</extra>",
            ), row=r, col=c_s)

        else:
            # Feature absent from SISSO formula:
            # white background + italic centred placeholder (mirrors matplotlib)
            fig.add_trace(go.Scatter(
                x             = [0.5],
                y             = [0.5],
                mode          = "text",
                textposition  = "middle center",
                text          = ["<i>Feature<br>(not in<br>SISSO)</i>"],
                textfont      = dict(size=11, color="rgba(192,192,192,1)"),
                showlegend    = False,
                hoverinfo     = "skip",
            ), row=r, col=c_s)

    # ── 3. Axis styling — mirrors _apply_panel_style ───────────────────────
    # range=[−0.05, 1.05], ticks=[0, 0.5, 1], light grid, axis box visible
    _common = dict(
        range     = [-0.02, 1.02],
        tickvals  = [0, 0.5, 1],
        ticktext  = ["0", "0.5", "1"],
        showgrid  = True,
        gridcolor = "rgba(0,0,0,0.10)",
        gridwidth = 0.5,
        zeroline  = False,
        showline  = True,
        linecolor = "rgba(0,0,0,0.30)",
        linewidth = 0.8,
        mirror    = True,               # draws box around each cell
    )

    for r in range(1, n_rows + 1):
        for c in range(1, n_cols + 1):
            is_bottom = (r == n_rows)
            is_left   = (c == 1)

            fig.update_xaxes(
                **_common,
                showticklabels = is_bottom,   # only bottom row
                tickfont       = dict(size=11),
                row=r, col=c,
            )
            fig.update_yaxes(
                **_common,
                showticklabels = is_left,     # only leftmost column
                tickfont       = dict(size=11),
                row=r, col=c,
            )

    # ── 4. Y-axis feature labels (leftmost column) ─────────────────────────
    # Equivalent to: if col == 0: ax.set_ylabel(FLABELS[feat], fontsize=23)
    for ri, feat in enumerate(all_cols):
        fig.update_yaxes(
            title_text     = FLABELS[feat],
            title_font     = dict(size=13),
            title_standoff = 5,
            row = ri + 1,
            col = 1,
        )

    # ── 5. Column header annotations (bold, model colour) ─────────────────
    # Equivalent to: if row == 0: ax.set_title(name, fontsize=23, color=color)
    for ci, name in enumerate(all_models):
        ax_key = "xaxis" if ci == 0 else f"xaxis{ci + 1}"
        try:
            dom   = fig.layout[ax_key].domain    # [x0, x1] in paper coords
            x_ctr = (dom[0] + dom[1]) / 2
        except (AttributeError, TypeError):
            x_ctr = (ci + 0.5) / n_cols          # fallback: uniform spacing

        fig.add_annotation(
            x         = x_ctr,
            y         = 1.012,
            xref      = "paper",
            yref      = "paper",
            text      = f"<b>{name}</b>",
            showarrow = False,
            font      = dict(size=15, color=all_colors[name]),
            xanchor   = "center",
            yanchor   = "bottom",
        )

    # ── 6. Global axis labels ──────────────────────────────────────────────
    # Mirrors:
    #   fig.text(0.05, 0.5, "Normalized $E_{adh}$", rotation=90, ...)
    #   fig.text(0.55, 0.03, "Normalized feature value", ...)
    #
    # Paper-coordinate mapping (width=1500, margin l=170, r=15, t=50, b=85):
    #   plot-area x: [0.113, 0.990]  centre ≈ 0.552
    #   plot-area y: [0.034, 0.980]  centre ≈ 0.507
    #   left-margin centre x ≈ 0.057  →  use −0.05 relative to plot for clarity

    # Rotated y-axis global label
    fig.add_annotation(
        x         = -0.055,
        y         = 0.5,
        xref      = "paper",
        yref      = "paper",
        text      = "Normalized E<sub>adh</sub>",
        showarrow = False,
        font      = dict(size=18),
        textangle = -90,
        xanchor   = "center",
        yanchor   = "middle",
    )

    # Bottom x-axis global label
    fig.add_annotation(
        x         = 0.525,
        y         = -0.032,
        xref      = "paper",
        yref      = "paper",
        text      = "Normalized feature value",
        showarrow = False,
        font      = dict(size=18),
        xanchor   = "center",
        yanchor   = "middle",
    )

    # ── 7. Overall layout ──────────────────────────────────────────────────
    # Aspect ratio targets matplotlib's 26 in × (14 × 3.2) in = 26 × 44.8 in
    # → width/height ≈ 0.58  →  1500 / 2580 ≈ 0.58 ✓
    fig.update_layout(
        height        = n_rows * 175,     # 14 × 175 = 2450 px
        width         = 1500,
        plot_bgcolor  = "white",
        paper_bgcolor = "white",
        margin        = dict(l=170, r=15, t=50, b=85),
        showlegend    = False,
        font          = dict(family="Arial, sans-serif", size=11),
    )

    # ── 8. Write self-contained HTML ───────────────────────────────────────
    out_path = out_dir / "F2_pdp_interactive.html"
    fig.write_html(
        str(out_path),
        include_plotlyjs = "cdn",       # loads Plotly.js from CDN
        include_mathjax  = "cdn",       # loads MathJax 2 for $...$ in labels
        full_html        = True,
        config           = {"scrollZoom": True, "responsive": True},
    )
    print(f"✓  Interactive HTML saved → {out_path}")