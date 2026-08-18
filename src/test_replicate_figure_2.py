"""Tests for `replicate_figure_2`.

Figure II plots credit-market sentiment at t-2 against real GDP-per-capita
growth at t (both orthogonalized against Table II column (1)'s other
regressor), with the fitted line from that same column, and highlights the
"influential observations" the paper calls out by name. Three things about
it are directly checkable against the paper:

1. The axis window is hardcoded (p. 1392) rather than autoscaled -- a
   regression test that nobody has quietly nudged it off the paper's scale.
2. `orthogonalize` must satisfy the Frisch-Waugh-Lovell identity it's named
   for: regressing the endogenous residual on the d_s_hat residual recovers
   Table II column (1)'s own coefficient on d_s_hat exactly.
3. `find_influential`, using the Belsley-Kuh-Welsch cutoff, should recover
   the paper's own named influential years: "two of these overly influential
   observations occur ... in 1932 and 1934; the remaining one is in 1977"
   (p. 1393). This repo's replication flags one additional borderline year
   (1947) under today's (revised) data, so the test checks the paper's three
   years are a subset of ours rather than requiring an exact match.
"""

from pathlib import Path

import numpy as np
import pytest
from matplotlib import pyplot as plt

import replicate_figure_2 as f2
from replicate_table_2 import REP_END, REP_START, build_panel, run_table_2
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))

_NEEDED = [
    "fred_final_series_annual.parquet",
    "shiller_data_annual.parquet",
    "greenwood_hanson_hys.parquet",
]


def _data_ready():
    return all((PROCESSED_DATA_DIR / f).exists() for f in _NEEDED)


requires_data = pytest.mark.skipif(
    not _data_ready(),
    reason="processed parquet data not built; run `doit` first",
)


# --------------------------------------------------------------------------- #
# Pure unit test (no data needed)
# --------------------------------------------------------------------------- #
def test_axis_ranges_match_published_figure():
    assert f2.XLIM == (-1.2, 0.6)
    assert f2.YLIM == (-15, 12)


# --------------------------------------------------------------------------- #
# Integration: does the replication match the paper?
# --------------------------------------------------------------------------- #
@pytest.fixture(scope="module")
def col1_results():
    df = build_panel("BAA_Treasury_spread")
    return run_table_2(df, REP_START, REP_END)["col1"]


@requires_data
def test_orthogonalize_recovers_original_slope(col1_results):
    """By the Frisch-Waugh-Lovell theorem, regressing the endog residual on
    the d_s_hat residual (both mean zero by construction, so no intercept
    is needed) must recover column (1)'s own d_s_hat coefficient exactly --
    this is precisely the slope Figure II draws as its fitted line."""
    x_resid, y_resid = f2.orthogonalize(col1_results, "d_s_hat")

    assert x_resid.mean() == pytest.approx(0.0, abs=1e-8)
    assert y_resid.mean() == pytest.approx(0.0, abs=1e-8)

    slope = np.polyfit(x_resid, y_resid, 1)[0]
    assert slope == pytest.approx(col1_results.params["d_s_hat"], rel=1e-6)


@requires_data
def test_find_influential_recovers_published_outlier_years(col1_results):
    influential = set(f2.find_influential(col1_results, "d_s_hat"))
    assert {1932, 1934, 1977}.issubset(influential)
    # Shouldn't be flagging a large fraction of the ~85-year sample.
    assert len(influential) <= 6


@requires_data
def test_plot_figure_2_runs_and_returns_a_figure():
    df = build_panel("BAA_Treasury_spread")
    fig = f2.plot_figure_2(df, REP_START, REP_END)
    try:
        assert fig is not None
        assert len(fig.axes) == 1
    finally:
        plt.close(fig)
