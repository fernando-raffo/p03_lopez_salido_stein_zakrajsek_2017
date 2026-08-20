"""
Data walkthrough: summary statistics and one chart per processed data
file that feeds the whole LSZ (2017) replication pipeline:

    - fred_final_series_annual.parquet   (GDP, prices, yields; 1929-present)
    - fred_final_series_monthly.parquet  (Baa/Aaa-Treasury spreads; 1925-present)
    - greenwood_hanson_hys.parquet       (high-yield issuance share; 1926-present)
    - shiller_data_annual.parquet        (S&P price/dividend, CAPE; 1929-present)

Four standalone tables and four standalone charts are produced -- one of
each per file (`summary_statistics_{slug}.tex`/`.pdf`/`.html`), rather than
one combined table or one combined figure, so each pair can carry its own
caption/description later on when this is dropped into the LaTeX report and
the ChartBook site -- neither is added here.

Charts intentionally carry no in-figure titles, matching the house style of
`replicate_figure_1.py`/`replicate_figure_2.py`, whose captions live outside
the image (report caption, ChartBook chart metadata).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from matplotlib import pyplot as plt

from helper_functions import log_total_return, year_over_year_growth
from latex_format import latex_escape
from plot_style import (
    HIGHLIGHT_COLOR,
    LINE_COLOR,
    MARKER_COLOR,
    RECESSION_COLOR,
    recession_spans,
    set_paper_style,
    style_axes,
)
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))

set_paper_style()

# ---------------------------------------------------------------------------
# Summary-statistics table
# ---------------------------------------------------------------------------
# Each spec is (column, label, units, scale, decimals). `scale` converts the
# raw column into the display units (e.g. dollars -> $tn, share -> pct.)
# before the mean/std/min/median/max are computed and formatted to
# `decimals` places.

FRED_ANNUAL_SPECS = [
    ("GDP", "Real GDP", "$tn", 1e-12, 2),
    ("Population", "Population", "millions", 1e-6, 1),
    ("GDP_per_capita", "Real GDP per capita", "$000s", 1e-3, 1),
    ("CPI_inflation", "CPI inflation (Dec/Dec)", "pct.", 100.0, 2),
    ("BAA", "Baa corporate bond yield", "pct.", 1.0, 2),
    ("AAA", "Aaa corporate bond yield", "pct.", 1.0, 2),
    ("Treasury_10yr", "10-year Treasury yield", "pct.", 1.0, 2),
    ("Treasury_3mo", "3-month Treasury yield", "pct.", 1.0, 2),
    ("BAA_Treasury_spread", "Baa-Treasury spread", "pp.", 1.0, 2),
    ("AAA_Treasury_spread", "Aaa-Treasury spread", "pp.", 1.0, 2),
]

FRED_MONTHLY_SPECS = [
    ("BAA_Treasury_spread", "Baa-Treasury spread", "pp.", 1.0, 2),
    ("AAA_Treasury_spread", "Aaa-Treasury spread", "pp.", 1.0, 2),
    ("BAA", "Baa corporate bond yield", "pct.", 1.0, 2),
    ("AAA", "Aaa corporate bond yield", "pct.", 1.0, 2),
    ("Treasury_10yr", "10-year Treasury yield", "pct.", 1.0, 2),
    (
        "hist_recession_indicator",
        "Months in NBER recession",
        "pct. of months",
        100.0,
        1,
    ),
]

GHY_SPECS = [
    ("hy_share", "High-yield issuance share", "pct.", 100.0, 1),
    ("ln_hy_share", "Log high-yield share", "log", 1.0, 2),
]

SHILLER_SPECS = [
    ("sp500_price", "S&P Composite price index", "index level", 1.0, 2),
    ("dividend", "S&P Composite dividend", "$ per share", 1.0, 2),
    ("pe10", "Cyclically adj. P/E (CAPE)", "ratio", 1.0, 2),
    ("sp_return", "S&P total return (log)", "pct.", 1.0, 2),
]


def load_data():
    """Load the four processed data files keyed by a short mnemonic."""
    shiller = pd.read_parquet(PROCESSED_DATA_DIR / "shiller_data_annual.parquet")
    shiller["sp_return"] = log_total_return(shiller["sp500_price"], shiller["dividend"])
    return {
        "fred_annual": pd.read_parquet(
            PROCESSED_DATA_DIR / "fred_final_series_annual.parquet"
        ),
        "fred_monthly": pd.read_parquet(
            PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet"
        ),
        "ghy": pd.read_parquet(PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet"),
        "shiller": shiller,
    }


def summary_dataframe(df, specs):
    """Mean/Std/Min/Q1/Median/Q3/Max for each spec, scaled to display units,
    as an (unrounded) DataFrame indexed by variable label. Shared by the
    LaTeX table builder below and by the notebook walkthrough, so both show
    numbers computed the same way."""
    records = []
    for col, label, units, scale, _dec in specs:
        s = df[col].astype(float) * scale
        records.append(
            {
                "Variable": label,
                "Units": units,
                "Mean": s.mean(),
                "Std. Dev.": s.std(),
                "Min": s.min(),
                "Q1": s.quantile(0.25),
                "Median": s.median(),
                "Q3": s.quantile(0.75),
                "Max": s.max(),
            }
        )
    return pd.DataFrame.from_records(records).set_index("Variable")


def table_rows(df, specs):
    """One LaTeX row per spec: label & units & mean & std & min & Q1 & median & Q3 & max."""
    stats = summary_dataframe(df, specs)
    lines = []
    for (_col, label, units, _scale, dec), (_, row) in zip(specs, stats.iterrows()):
        lines.append(
            f"{latex_escape(label)} & {latex_escape(units)} & {row['Mean']:,.{dec}f} & "
            f"{row['Std. Dev.']:,.{dec}f} & {row['Min']:,.{dec}f} & {row['Q1']:,.{dec}f} & "
            f"{row['Median']:,.{dec}f} & {row['Q3']:,.{dec}f} & {row['Max']:,.{dec}f} \\\\"
        )
    return lines


def build_table(df, specs, title):
    lines = [
        "\\begin{tabular}{llrrrrrrr}",
        "\\toprule",
        f"\\multicolumn{{9}}{{l}}{{\\textit{{{title}}}}} \\\\",
        "\\midrule",
        "Variable & Units & Mean & Std.\\ Dev. & Min & Q1 & Median & Q3 & Max \\\\",
        "\\midrule",
        *table_rows(df, specs),
        "\\bottomrule",
        "\\end{tabular}",
    ]
    return "\n".join(lines)


# (slug, data key, specs, title fn) -- one entry per processed file, each
# producing its own standalone table (matching the chart slugs below).
TABLE_SPECS = [
    (
        "credit_spreads",
        "fred_monthly",
        FRED_MONTHLY_SPECS,
        lambda fm: (
            f"FRED Monthly Series, {fm.index.min():%Y:%m}--{fm.index.max():%Y:%m} ($N={len(fm):,}$)"
        ),
    ),
    (
        "gdp_growth",
        "fred_annual",
        FRED_ANNUAL_SPECS,
        lambda fa: (
            f"FRED Annual Series, {int(fa.index.min())}--{int(fa.index.max())} ($N={len(fa)}$)"
        ),
    ),
    (
        "hy_share",
        "ghy",
        GHY_SPECS,
        lambda ghy: (
            f"Greenwood-Hanson High-Yield Share, {int(ghy.index.min())}--{int(ghy.index.max())} "
            f"($N={len(ghy)}$)"
        ),
    ),
    (
        "cape",
        "shiller",
        SHILLER_SPECS,
        lambda sh: (
            f"Shiller Annual Stock-Market Series, {sh.index.min().year}--{sh.index.max().year} ($N={len(sh)}$)"
        ),
    ),
]


def emit_tables(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, key, specs, title_fn in TABLE_SPECS:
        df = data[key]
        tex = build_table(df, specs, title_fn(df))
        path = OUTPUT_DIR / f"summary_statistics_{slug}.tex"
        comment = f"% Summary statistics: {slug}\n"
        path.write_text(comment + tex + "\n")
        print(f"-> {path.name}")


# ---------------------------------------------------------------------------
# Charts -- one standalone figure per data file (static PDF + interactive
# HTML), rather than one combined multi-panel figure.
# ---------------------------------------------------------------------------


def build_credit_spreads_figure(fm):
    """FRED monthly: Baa- and Aaa-Treasury credit spreads, with NBER
    recessions shaded. Mirrors `replicate_figure_1.plot_figure_1`'s styling."""
    baa = fm["BAA_Treasury_spread"].dropna()
    aaa = fm["AAA_Treasury_spread"].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(baa.index, baa.values, color=LINE_COLOR, lw=1.1, label="Baa-Treasury")
    ax.plot(aaa.index, aaa.values, color=HIGHLIGHT_COLOR, lw=1.1, label="Aaa-Treasury")
    for s, e in recession_spans(fm):
        ax.axvspan(s, e, color=RECESSION_COLOR, lw=0)

    first_year, last_year = fm.index.year.min(), fm.index.year.max()
    xticks = pd.to_datetime([f"{y}-01-01" for y in range(first_year, last_year + 1, 6)])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(y) for y in xticks.year])

    y_max = int(np.ceil(max(baa.max(), aaa.max())))
    ax.set_ylim(0, y_max)
    ax.set_yticks(range(y_max + 1))
    ax.margins(x=0.01)

    ax.legend(fontsize=9, loc="upper right")
    ax.text(
        1.0,
        1.02,
        "Percentage points",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
    )
    style_axes(ax)
    fig.tight_layout()
    return fig


def build_credit_spreads_interactive(fm):
    baa = fm["BAA_Treasury_spread"].dropna()
    aaa = fm["AAA_Treasury_spread"].dropna()

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=baa.index,
            y=baa.values,
            mode="lines",
            line=dict(color=LINE_COLOR, width=1.3),
            name="Baa-Treasury",
            hovertemplate="%{x|%b %Y}: %{y:.2f} pp<extra>Baa</extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=aaa.index,
            y=aaa.values,
            mode="lines",
            line=dict(color=HIGHLIGHT_COLOR, width=1.3),
            name="Aaa-Treasury",
            hovertemplate="%{x|%b %Y}: %{y:.2f} pp<extra>Aaa</extra>",
        )
    )
    for s, e in recession_spans(fm):
        fig.add_vrect(
            x0=s,
            x1=e,
            fillcolor=RECESSION_COLOR,
            opacity=1,
            layer="below",
            line_width=0,
        )

    y_max = int(np.ceil(max(baa.max(), aaa.max())))
    fig.update_layout(
        template="simple_white",
        hovermode="x unified",
        margin=dict(t=40, r=30, b=40, l=60),
    )
    fig.update_yaxes(title_text="Percentage points", range=[0, y_max])
    return fig


def build_gdp_growth_figure(fa):
    """FRED annual: year-over-year growth in real GDP per capita -- the
    dependent variable throughout Tables I and II."""
    g = year_over_year_growth(fa["GDP_per_capita"]).dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.axhline(0, color="grey", lw=0.7)
    ax.plot(g.index, g.values, color=LINE_COLOR, lw=1.1)
    ax.margins(x=0.01)
    ax.text(
        1.0,
        1.02,
        "Log growth, y/y (pct.)",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
    )
    style_axes(ax)
    fig.tight_layout()
    return fig


def build_gdp_growth_interactive(fa):
    g = year_over_year_growth(fa["GDP_per_capita"]).dropna()

    fig = go.Figure()
    fig.add_hline(y=0, line_color="grey", line_width=0.8)
    fig.add_trace(
        go.Scatter(
            x=g.index,
            y=g.values,
            mode="lines",
            line=dict(color=LINE_COLOR, width=1.3),
            showlegend=False,
            hovertemplate="%{x}: %{y:.2f} pct.<extra></extra>",
        )
    )
    fig.update_layout(template="simple_white", margin=dict(t=40, r=30, b=40, l=60))
    fig.update_yaxes(title_text="Log growth, y/y (pct.)")
    return fig


def build_hy_share_figure(ghy):
    """Greenwood-Hanson high-yield issuance share, colored by source series
    to make the 2009 Greenwood-Hanson (2013) -> Mergent FISD splice visible."""
    gh2013 = ghy.loc[ghy["source"] == "gh2013", "hy_share"] * 100
    fisd = ghy.loc[ghy["source"] == "fisd", "hy_share"] * 100

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(
        gh2013.index,
        gh2013.values,
        color=LINE_COLOR,
        lw=1.1,
        label="Greenwood-Hanson (2013)",
    )
    if not fisd.empty:
        ax.plot(
            fisd.index,
            fisd.values,
            color=HIGHLIGHT_COLOR,
            lw=1.1,
            label="Mergent FISD splice",
        )
        ax.axvline(fisd.index.min(), color="grey", lw=0.8, ls="--")
    ax.margins(x=0.01)
    ax.legend(fontsize=9, loc="upper right")
    ax.text(
        1.0,
        1.02,
        "Pct. of issuance",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
    )
    style_axes(ax)
    fig.tight_layout()
    return fig


def build_hy_share_interactive(ghy):
    gh2013 = ghy.loc[ghy["source"] == "gh2013", "hy_share"] * 100
    fisd = ghy.loc[ghy["source"] == "fisd", "hy_share"] * 100

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=gh2013.index,
            y=gh2013.values,
            mode="lines",
            line=dict(color=LINE_COLOR, width=1.3),
            name="Greenwood-Hanson (2013)",
            hovertemplate="%{x}: %{y:.1f}%<extra>Greenwood-Hanson</extra>",
        )
    )
    if not fisd.empty:
        fig.add_trace(
            go.Scatter(
                x=fisd.index,
                y=fisd.values,
                mode="lines",
                line=dict(color=HIGHLIGHT_COLOR, width=1.3),
                name="Mergent FISD splice",
                hovertemplate="%{x}: %{y:.1f}%<extra>FISD splice</extra>",
            )
        )
        fig.add_vline(
            x=fisd.index.min(), line_color="grey", line_width=0.9, line_dash="dash"
        )
    fig.update_layout(
        template="simple_white",
        legend=dict(x=0.02, y=0.02, xanchor="left", yanchor="bottom"),
        margin=dict(t=40, r=30, b=40, l=60),
    )
    fig.update_yaxes(title_text="Pct. of issuance")
    return fig


CAPE_FORWARD_HORIZON = 10


def cape_forward_return(sh, horizon=CAPE_FORWARD_HORIZON):
    """CAPE at year t paired with the cumulative S&P total log return
    realized from t to t+`horizon` (sum of `sp_return` over the following
    `horizon` years, since log returns are additive). Years within
    `horizon` of the end of the sample have no future window and are
    dropped."""
    fwd = sh["sp_return"].rolling(horizon).sum().shift(-horizon)
    pts = pd.concat([sh["pe10"], fwd], axis=1, keys=["pe10", "fwd_return"])
    return pts.dropna()


def build_cape_figure(sh):
    """CAPE at year t vs. the cumulative S&P total return realized over the
    following `CAPE_FORWARD_HORIZON` years -- the textbook case for CAPE as
    a valuation signal: rich markets (high CAPE) have historically been
    followed by weaker subsequent returns, and cheap markets by stronger
    ones, with a fitted line showing the (negative) average relationship."""
    pts = cape_forward_return(sh)
    slope, intercept = np.polyfit(pts["pe10"], pts["fwd_return"], 1)
    x_fit = np.array([pts["pe10"].min(), pts["pe10"].max()])

    fig, ax = plt.subplots(figsize=(7, 6))
    ax.scatter(pts["pe10"], pts["fwd_return"], color=MARKER_COLOR, s=22, zorder=2)
    ax.plot(x_fit, slope * x_fit + intercept, color=HIGHLIGHT_COLOR, lw=1.6, zorder=3)
    ax.set_xlabel("CAPE at year $t$ (ratio)")
    ax.set_ylabel(
        f"Cumulative S&P total return, $t$ to $t+{CAPE_FORWARD_HORIZON}$ (pct.)"
    )
    corr = pts["pe10"].corr(pts["fwd_return"])
    ax.text(
        0.98,
        0.96,
        f"corr = {corr:.2f}",
        transform=ax.transAxes,
        ha="right",
        va="top",
        fontsize=9.5,
        color=HIGHLIGHT_COLOR,
    )
    style_axes(ax)
    fig.tight_layout()
    return fig


def build_cape_interactive(sh):
    pts = cape_forward_return(sh)
    slope, intercept = np.polyfit(pts["pe10"], pts["fwd_return"], 1)
    x_fit = np.array([pts["pe10"].min(), pts["pe10"].max()])

    fig = go.Figure()
    fig.add_trace(
        go.Scatter(
            x=pts["pe10"],
            y=pts["fwd_return"],
            mode="markers",
            marker=dict(color=MARKER_COLOR, size=6),
            text=[str(y) for y in pts.index.year],
            showlegend=False,
            hovertemplate="CAPE in %{text}: %{x:.2f}<br>fwd. return: %{y:.1f} pct.<extra></extra>",
        )
    )
    fig.add_trace(
        go.Scatter(
            x=x_fit,
            y=slope * x_fit + intercept,
            mode="lines",
            line=dict(color=HIGHLIGHT_COLOR, width=2),
            showlegend=False,
            hoverinfo="skip",
        )
    )
    fig.update_layout(template="simple_white", margin=dict(t=40, r=30, b=40, l=60))
    fig.update_xaxes(title_text="CAPE at year t (ratio)")
    fig.update_yaxes(
        title_text=f"Cumulative S&P total return, t to t+{CAPE_FORWARD_HORIZON} (pct.)"
    )
    return fig


# (slug, static builder, interactive builder, data key) -- one entry per
# processed file, each producing its own PDF and HTML chart.
CHART_SPECS = [
    (
        "credit_spreads",
        build_credit_spreads_figure,
        build_credit_spreads_interactive,
        "fred_monthly",
    ),
    (
        "gdp_growth",
        build_gdp_growth_figure,
        build_gdp_growth_interactive,
        "fred_annual",
    ),
    ("hy_share", build_hy_share_figure, build_hy_share_interactive, "ghy"),
    ("cape", build_cape_figure, build_cape_interactive, "shiller"),
]


def emit_charts(data):
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for slug, static_builder, interactive_builder, key in CHART_SPECS:
        df = data[key]

        fig = static_builder(df)
        pdf_path = OUTPUT_DIR / f"summary_statistics_{slug}.pdf"
        fig.savefig(pdf_path)
        plt.close(fig)
        print(f"-> {pdf_path.name}")

        ifig = interactive_builder(df)
        html_path = OUTPUT_DIR / f"summary_statistics_{slug}.html"
        ifig.write_html(str(html_path), include_plotlyjs="cdn")
        print(f"-> {html_path.name}")


def main():
    data = load_data()
    for name, df in data.items():
        print(f"{name}: n={len(df)}, columns={list(df.columns)}")
    emit_tables(data)
    emit_charts(data)


if __name__ == "__main__":
    main()
