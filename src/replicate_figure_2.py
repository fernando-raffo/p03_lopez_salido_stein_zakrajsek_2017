"""
Replicate Figure II of Lopez-Salido, Stein & Zakrajsek (2017): credit-market
sentiment at t-2 vs. real GDP-per-capita growth at t, together with the
fitted line implied by column (1) of Table II.

Both plotted variables come straight from `replicate_table_2.run_table_2`'s
column-(1) results.

Flexible like `replicate_table_2.run_table_2`: pass any `start`/`end`
window to `plot_figure_2`. `main()` saves the 1929-2015 replication figure
and a 1929-present extension, as PDFs, to `_output/`, once for the
Baa-Treasury spread and once for the Aaa-Treasury spread (both via
`replicate_table_2.build_panel`'s `spread_col` argument).
"""

import numpy as np
import pandas as pd
import statsmodels.api as sm
from matplotlib import pyplot as plt

from plot_style import (
    HIGHLIGHT_COLOR,
    MARKER_COLOR,
    set_paper_style,
    style_axes,
)
from replicate_table_2 import (
    EXT_END,
    OUTPUT_DIR,
    REP_END,
    REP_START,
    build_panel,
    run_table_2,
)

set_paper_style()

# Axis range/ticks of the published Figure II (p. 1392), reproduced here so
# our version sits on the same scale as the paper's rather than matplotlib's
# data-dependent autoscale.
XLIM = (-1.2, 0.6)
YLIM = (-15, 12)
XTICK_STEP = 0.3
YTICK_STEP = 3


def find_influential(res, var):
    """Return the index labels of observations whose |DFBETAS| on `var`
    exceeds the size-adjusted Belsley, Kuh, and Welsch (1980) cutoff of
    2/sqrt(T) (T = number of observations in `res`), i.e. the same
    thresholding rule used for Figure III of the paper."""
    dfbetas = res.get_influence().dfbetas
    var_idx = res.model.exog_names.index(var)
    s = pd.Series(dfbetas[:, var_idx], index=res.model.data.row_labels)
    cutoff = 2 / np.sqrt(res.nobs)
    return s[s.abs() > cutoff].index


def orthogonalize(res, var):
    """Residualize `var` and the model's dependent variable against every
    *other* regressor in the fitted `statsmodels` OLS result `res` (an
    added-variable / partial-regression transform).

    By the Frisch-Waugh-Lovell theorem, regressing the resulting
    endog-residual on the `var`-residual recovers `res`'s own coefficient on
    `var` exactly, with a zero intercept, since both residual series are
    mean zero by construction.

    Returns
    -------
    (x_resid, y_resid) : pandas.Series
        Residualized `var` and residualized endog, indexed like `res`.
    """
    data = res.model.data.orig_exog.join(res.model.data.orig_endog)
    other_vars = [v for v in res.model.exog_names if v not in ("const", var)]
    controls = (
        sm.add_constant(data[other_vars]) if other_vars else data[[]].assign(const=1.0)
    )
    endog_name = res.model.endog_names
    x_resid = sm.OLS(data[var], controls).fit().resid
    y_resid = sm.OLS(data[endog_name], controls).fit().resid
    return x_resid, y_resid


def plot_figure_2(df, start=REP_START, end=REP_END):
    """
    Build the Figure II scatter plot (credit-market sentiment at t-2 vs.
    real GDP-per-capita growth at t, both orthogonalized against column
    (1)'s other regressor) over `df.loc[start:end]`, using column (1) of
    Table II (`replicate_table_2.run_table_2`) for both the plotted points
    and the fitted line.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `replicate_table_2.build_panel`.
    start, end : int
        First and last calendar year (inclusive) of the sample plotted.

    Returns
    -------
    matplotlib.figure.Figure
    """
    res1 = run_table_2(df, start, end)["col1"]

    x, y = orthogonalize(res1, "d_s_hat")
    influential_years = find_influential(res1, "d_s_hat")
    beta_s = res1.params["d_s_hat"]

    fig, ax = plt.subplots(figsize=(7, 7))
    ax.axhline(0, color="grey", lw=0.8, zorder=1)
    ax.axvline(0, color="grey", lw=0.8, zorder=1)

    is_influential = x.index.isin(influential_years)
    ax.scatter(
        x[~is_influential],
        y[~is_influential],
        color=MARKER_COLOR,
        s=20,
        zorder=2,
    )
    ax.scatter(
        x[influential_years],
        y[influential_years],
        color=HIGHLIGHT_COLOR,
        marker="*",
        s=170,
        label="Influential observations",
        zorder=3,
    )
    for yr in influential_years:
        ax.annotate(
            str(yr),
            (x.loc[yr], y.loc[yr]),
            textcoords="offset points",
            xytext=(6, 6),
            fontsize=9,
            color=HIGHLIGHT_COLOR,
        )

    y_line = [beta_s * xv for xv in XLIM]
    ax.plot(XLIM, y_line, color=HIGHLIGHT_COLOR, lw=1.5, zorder=2)

    ax.set_xlim(XLIM)
    ax.set_ylim(YLIM)
    ax.set_xticks(np.arange(XLIM[0], XLIM[1] + 1e-9, XTICK_STEP))
    ax.set_yticks(np.arange(YLIM[0], YLIM[1] + 1e-9, YTICK_STEP))
    ax.set_xlabel("Credit-market sentiment at $t-2$ (pps.)")
    ax.set_ylabel("Growth in real GDP per capita at $t$ (pct.)")
    style_axes(ax)
    ax.legend(loc="lower left", frameon=False, fontsize=9.5)
    fig.tight_layout()
    return fig


# (label tag, spread column) pairs. The Baa tag is empty so its filenames
# match the original, unsuffixed `figure_2_*.pdf` names.
SPREAD_VARIANTS = [
    ("", "BAA_Treasury_spread"),
    ("aaa", "AAA_Treasury_spread"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    windows = [
        (REP_START, REP_END, "replication"),
        (REP_START, EXT_END, "extended"),
    ]
    for spread_tag, spread_col in SPREAD_VARIANTS:
        df = build_panel(spread_col=spread_col)
        for start, end, window_label in windows:
            fig = plot_figure_2(df, start, end)
            label = "_".join(p for p in (spread_tag, window_label) if p)
            out = OUTPUT_DIR / f"figure_2_{label}.pdf"
            fig.savefig(out)
            plt.close(fig)
            print(f"{label} ({start}-{end}): -> {out.name}")


if __name__ == "__main__":
    main()
