"""Replicate Figure I of Lopez-Salido, Stein & Zakrajsek (2017):
the Baa-Treasury credit spread over the 1925-2015 replication sample.

Reads the cleaned monthly FRED series produced by process_fred_data_monthly.py
and shades NBER recessions using the hist_recession_indicator column.

Styling is shared with Figure II via `plot_style.py` so both figures match
the look of the printed article.

Flexible like `replicate_figure_2.plot_figure_2`: pass any `start`/`end`
window and `spread_col` to `plot_figure_1`. `main()` saves both the
1925-2015 replication figure and a 1925-present extension, as PDFs, to
`_output/`, once for the Baa-Treasury spread and once for the Aaa-Treasury
spread.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

from plot_style import (
    LINE_COLOR,
    RECESSION_COLOR,
    recession_spans,
    set_paper_style,
    style_axes,
)
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
BUFFER_START = config("BUFFER_START_DATE")
REP_END = config("REPLICATION_END_DATE")
EXT_END = config("EXTENSION_END_DATE")

set_paper_style()


def plot_figure_1(df, start=BUFFER_START, end=REP_END, spread_col="BAA_Treasury_spread"):
    """
    Build the Figure I line plot (credit spread, with NBER recessions
    shaded) over `df.loc[start:end]`.

    Parameters
    ----------
    df : pandas.DataFrame
        `fred_final_series_monthly.parquet`, indexed by date, with
        `spread_col` and `hist_recession_indicator` columns.
    start, end : datetime-like
        First and last date (inclusive) of the sample plotted.
    spread_col : str, default "BAA_Treasury_spread"
        Column of `df` to plot as the credit spread, e.g.
        "BAA_Treasury_spread" or "AAA_Treasury_spread".

    Returns
    -------
    (fig, spread) : (matplotlib.figure.Figure, pandas.Series)
    """
    window = df.loc[(df.index >= start) & (df.index <= end)]
    spread = window[spread_col].dropna()

    fig, ax = plt.subplots(figsize=(10, 5))
    ax.plot(spread.index, spread.values, color=LINE_COLOR, lw=1.1)

    # Shade NBER recessions directly from the data
    for s, e in recession_spans(window):
        ax.axvspan(s, e, color=RECESSION_COLOR, lw=0)

    # X-axis: a tick every 6 years, exactly as in the printed Figure I.
    first_year = spread.index.year.min()
    last_year = spread.index.year.max()
    xticks = pd.to_datetime([f"{y}-01-01" for y in range(first_year, last_year + 1, 6)])
    ax.set_xticks(xticks)
    ax.set_xticklabels([str(y) for y in xticks.year])

    # Y-axis: integer ticks from 0 to just above the sample maximum.
    y_max = int(np.ceil(spread.max()))
    ax.set_ylim(0, y_max)
    ax.set_yticks(range(y_max + 1))

    ax.set_xlabel("")
    ax.set_ylabel("")
    ax.margins(x=0.01)
    style_axes(ax)

    # The paper labels the y-axis unit above the axis rather than with a
    # rotated ylabel.
    ax.text(
        1.0,
        1.02,
        "Percentage points",
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=9.5,
    )

    fig.tight_layout()
    return fig, spread


# (label tag, spread column) pairs. The Baa tag is empty so its filenames
# match the original, unsuffixed `figure_1_*.pdf` names.
SPREAD_VARIANTS = [
    ("", "BAA_Treasury_spread"),
    ("aaa", "AAA_Treasury_spread"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet")

    windows = [
        (BUFFER_START, REP_END, "replication"),
        (BUFFER_START, EXT_END, "extended"),
    ]
    for spread_tag, spread_col in SPREAD_VARIANTS:
        for start, end, window_label in windows:
            fig, spread = plot_figure_1(df, start, end, spread_col=spread_col)
            label = "_".join(p for p in (spread_tag, window_label) if p)
            out = OUTPUT_DIR / f"figure_1_{label}.pdf"
            fig.savefig(out)
            plt.close(fig)
            print(
                f"{label} ({start.date()}..{end.date()}): saved {out.name}  "
                f"(n={len(spread)}, {spread.index.min().date()}..{spread.index.max().date()})"
            )


if __name__ == "__main__":
    main()
