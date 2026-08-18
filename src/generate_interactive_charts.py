"""Generate interactive (Plotly/HTML) versions of Figures I and II for the
ChartBook site, to sit alongside the static PDF versions produced by
`replicate_figure_1.py` and `replicate_figure_2.py`.

Reuses the same data-selection and estimation logic as the PDF figures
(`plot_style.recession_spans`, `replicate_table_2.build_panel`/`run_table_2`,
`replicate_figure_2.orthogonalize`/`find_influential`) so both versions of
each figure show identical data; only the rendering differs.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go

from plot_style import (
    HIGHLIGHT_COLOR,
    LINE_COLOR,
    MARKER_COLOR,
    RECESSION_COLOR,
    recession_spans,
)

# XLIM/YLIM are Figure II's published axis range (p. 1392), reused here so
# both renderings share the same fixed scale.
from replicate_figure_2 import XLIM, YLIM, find_influential, orthogonalize
from replicate_table_2 import EXT_END, REP_END, REP_START, build_panel, run_table_2
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
BUFFER_START = config("BUFFER_START_DATE")
FIG1_REP_END = config("REPLICATION_END_DATE")
FIG1_EXT_END = config("EXTENSION_END_DATE")

# (label tag, spread column, display label) triples, shared by both figures.
SPREAD_VARIANTS = [
    ("", "BAA_Treasury_spread", "Baa"),
    ("aaa", "AAA_Treasury_spread", "Aaa"),
]


def make_figure_1_chart(df, start, end, spread_col, spread_label):
    """Build an interactive line chart of the credit spread over
    `df.loc[start:end]`, with NBER recessions shaded, mirroring
    `replicate_figure_1.plot_figure_1`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    window = df.loc[(df.index >= start) & (df.index <= end)]
    spread = window[spread_col].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=spread.index,
            y=spread.values,
            mode="lines",
            line=dict(color=LINE_COLOR, width=1.3),
            name=spread_label,
            hovertemplate="%{x|%b %Y}: %{y:.2f} pp<extra></extra>",
        )
    )
    for s, e in recession_spans(window):
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=RECESSION_COLOR,
            opacity=1,
            layer="below",
            line_width=0,
        )

    y_max = int(np.ceil(spread.max()))
    fig.update_layout(
        template="simple_white",
        showlegend=False,
        hovermode="x unified",
        margin=dict(t=60, r=30, b=40, l=60),
    )
    fig.update_yaxes(title_text="Percentage points", range=[0, y_max])
    fig.update_xaxes(title_text="")
    return fig


def make_figure_2_chart(df, start, end):
    """Build an interactive scatter chart of credit-market sentiment at
    t-2 vs. GDP-per-capita growth at t, with the fitted line and
    influential observations highlighted, mirroring
    `replicate_figure_2.plot_figure_2`.

    Returns
    -------
    plotly.graph_objects.Figure
    """
    res1 = run_table_2(df, start, end)["col1"]
    x, y = orthogonalize(res1, "d_s_hat")
    influential_years = find_influential(res1, "d_s_hat")
    beta_s = res1.params["d_s_hat"]

    is_influential = x.index.isin(influential_years)

    fig = go.Figure()
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.add_vline(x=0, line_color="grey", line_width=0.8)
    fig.add_trace(
        go.Scatter(
            x=x[~is_influential],
            y=y[~is_influential],
            mode="markers",
            marker=dict(color=MARKER_COLOR, size=7),
            name="Observations",
            text=[str(yr) for yr in x[~is_influential].index],
            hovertemplate="%{text}: (%{x:.2f}, %{y:.2f})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x[influential_years],
            y=y[influential_years],
            mode="markers+text",
            marker=dict(color=HIGHLIGHT_COLOR, size=14, symbol="star"),
            text=[str(yr) for yr in influential_years],
            textposition="top center",
            textfont=dict(color=HIGHLIGHT_COLOR),
            name="Influential observations",
            hovertemplate="%{text}: (%{x:.2f}, %{y:.2f})<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=list(XLIM),
            y=[beta_s * xv for xv in XLIM],
            mode="lines",
            line=dict(color=HIGHLIGHT_COLOR, width=2),
            name="Fitted line",
            hoverinfo="skip",
        )
    )

    fig.update_layout(
        template="simple_white",
        legend=dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
        margin=dict(t=60, r=30, b=60, l=60),
    )
    fig.update_xaxes(
        title_text="Credit-market sentiment at t-2 (pps.)", range=list(XLIM)
    )
    fig.update_yaxes(
        title_text="Growth in real GDP per capita at t (pct.)", range=list(YLIM)
    )
    return fig


def _write_html(fig, chart_id):
    out = OUTPUT_DIR / f"{chart_id}.html"
    fig.write_html(str(out), include_plotlyjs="cdn")
    print(f"-> {out.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    monthly = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet")
    fig1_windows = [
        (BUFFER_START, FIG1_REP_END, "replication"),
        (BUFFER_START, FIG1_EXT_END, "extended"),
    ]
    for spread_tag, spread_col, spread_label in SPREAD_VARIANTS:
        for start, end, window_tag in fig1_windows:
            label = "_".join(p for p in (spread_tag, window_tag) if p)
            fig = make_figure_1_chart(monthly, start, end, spread_col, spread_label)
            _write_html(fig, f"figure_1_{label}")

    fig2_windows = [
        (REP_START, REP_END, "replication"),
        (REP_START, EXT_END, "extended"),
    ]
    for spread_tag, spread_col, _spread_label in SPREAD_VARIANTS:
        panel = build_panel(spread_col=spread_col)
        for start, end, window_tag in fig2_windows:
            label = "_".join(p for p in (spread_tag, window_tag) if p)
            fig = make_figure_2_chart(panel, start, end)
            _write_html(fig, f"figure_2_{label}")


if __name__ == "__main__":
    main()
