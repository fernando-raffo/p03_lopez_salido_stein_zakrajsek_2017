"""
Shared matplotlib styling for the replicated figures.

Both `replicate_figure_1.py` and `replicate_figure_2.py` render their plots
through this module so that our version of each figure matches the plain,
serif, box-axis look shared by the printed QJE figures in Lopez-Salido,
Stein, and Zakrajsek (2017).
"""

import textwrap

import matplotlib.pyplot as plt

# Colors used across both figures, chosen to match the printed article:
# a black data line/markers, pale blue-grey NBER recession shading, and a
# red highlight for the influential-observation markers/fitted line in
# Figure II.
LINE_COLOR = "black"
MARKER_COLOR = "black"
RECESSION_COLOR = "#c6d4e1"
HIGHLIGHT_COLOR = "#c0392b"


def set_paper_style():
    """Reset matplotlib rcParams to the plain serif/box-axis look shared by
    the paper's figures. Call once per script, before creating any axes."""
    plt.rcdefaults()
    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.size": 11,
            "axes.facecolor": "white",
            "figure.facecolor": "white",
            "savefig.facecolor": "white",
            "axes.edgecolor": "black",
            "axes.linewidth": 0.9,
            "axes.grid": False,
            "xtick.direction": "in",
            "ytick.direction": "in",
            "xtick.top": True,
            "ytick.right": True,
            "legend.frameon": False,
        }
    )


def recession_spans(window, indicator_col="hist_recession_indicator"):
    """Return `(start, end)` pairs for each contiguous recession spell in
    `window[indicator_col]`, for shading recession periods on a plot.

    Parameters
    ----------
    window : pandas.DataFrame
        Indexed by date, containing at least `indicator_col`.
    indicator_col : str, default "hist_recession_indicator"
        Column of `window` equal to 1 during NBER recessions, 0 otherwise.

    Returns
    -------
    list of (pandas.Timestamp, pandas.Timestamp)
        Start and end date of each recession spell found in `window`.
    """
    rec = window[indicator_col].fillna(0).astype(int)
    in_rec = rec.eq(1)
    if not in_rec.any():
        return []
    starts = window.index[in_rec & ~in_rec.shift(1, fill_value=False)]
    ends = window.index[in_rec & ~in_rec.shift(-1, fill_value=False)]
    return list(zip(starts, ends))


def style_axes(ax):
    """Apply the box-style spines / inward ticks used across the paper's
    figures to a single Axes."""
    for spine in ax.spines.values():
        spine.set_visible(True)
        spine.set_color("black")
        spine.set_linewidth(0.9)
    ax.tick_params(direction="in", which="both", top=True, right=True, length=4)
    ax.grid(False)


def add_caption(
    fig,
    figure_number,
    title,
    note,
    note_width=95,
    bottom=0.24,
    gap=0.05,
    line_step=0.035,
):
    """Typeset a QJE-style caption ("FIGURE <n>" + title + a small italic
    note) centered below the axes, and reserve room for it at the bottom of
    `fig`.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
        Figure to caption; its axes should already be laid out (e.g. via
        `fig.tight_layout()`) before calling this.
    figure_number : int or str
        Roman-style figure label, e.g. `1` -> "FIGURE 1".
    title : str
        One-line descriptive title, printed under the bold "FIGURE n" line.
    note : str
        Explanatory note, wrapped and printed in a smaller italic font.
    note_width : int
        Character width used to wrap `note` into multiple lines.
    bottom : float
        Fraction of the figure height reserved below the axes for the
        caption (passed to `fig.subplots_adjust(bottom=...)`).
    gap : float
        Vertical fraction of figure height left blank between the bottom of
        the axes (and its x-axis label, if any) and the "FIGURE n" line.
    line_step : float
        Vertical fraction of figure height between the "FIGURE n" line and
        the title line below it.
    """
    fig.subplots_adjust(bottom=bottom)
    wrapped_note = "\n".join(textwrap.wrap(note, width=note_width))
    y0 = bottom - gap

    fig.text(
        0.5,
        y0,
        f"FIGURE {figure_number}",
        ha="center",
        va="top",
        fontsize=11,
        fontweight="bold",
    )
    fig.text(
        0.5,
        y0 - line_step,
        title,
        ha="center",
        va="top",
        fontsize=10.5,
    )
    fig.text(
        0.5,
        y0 - 2 * line_step,
        wrapped_note,
        ha="center",
        va="top",
        fontsize=8.5,
        style="italic",
        linespacing=1.5,
    )
