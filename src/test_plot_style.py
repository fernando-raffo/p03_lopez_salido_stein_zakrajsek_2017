"""
Unit tests for `plot_style`, the shared styling/recession-shading helper
used by both `replicate_figure_1.py` and `replicate_figure_2.py`.

`recession_spans` is a pure function (no plotting, no data files) and feeds
directly into Figure I's NBER-recession shading, so it gets a real test
rather than being exercised only incidentally through a figure-generation
smoke test.
"""

import pandas as pd

import plot_style as ps


def test_recession_spans_finds_contiguous_runs():
    dates = pd.date_range("2000-01-01", periods=6, freq="MS")
    df = pd.DataFrame({"hist_recession_indicator": [0, 1, 1, 0, 1, 0]}, index=dates)

    spans = ps.recession_spans(df)

    assert spans == [(dates[1], dates[2]), (dates[4], dates[4])]


def test_recession_spans_handles_no_recessions():
    dates = pd.date_range("2000-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"hist_recession_indicator": [0, 0, 0]}, index=dates)

    assert ps.recession_spans(df) == []


def test_recession_spans_handles_recession_at_series_boundary():
    """A recession spell already underway at the start (or still ongoing at
    the end) of the window must still be captured as a single span."""
    dates = pd.date_range("2000-01-01", periods=4, freq="MS")
    df = pd.DataFrame({"hist_recession_indicator": [1, 1, 0, 1]}, index=dates)

    spans = ps.recession_spans(df)

    assert spans == [(dates[0], dates[1]), (dates[3], dates[3])]


def test_recession_spans_treats_missing_values_as_no_recession():
    dates = pd.date_range("2000-01-01", periods=3, freq="MS")
    df = pd.DataFrame({"hist_recession_indicator": [1.0, None, 1.0]}, index=dates)

    spans = ps.recession_spans(df)

    assert spans == [(dates[0], dates[0]), (dates[2], dates[2])]


def test_set_paper_style_sets_serif_font_and_white_background():
    ps.set_paper_style()
    import matplotlib.pyplot as plt

    assert plt.rcParams["font.family"] == ["serif"]
    assert plt.rcParams["axes.facecolor"] == "white"
    assert plt.rcParams["axes.grid"] is False
