"""
Tests for `replicate_figure_1`.

Figure I itself is a single time series (the Baa-Treasury credit spread)
with NBER recessions shaded, so there is no regression coefficient to check
against the paper the way Tables I/II have. Instead these tests check the
figure's *content* against what the printed Figure I (p. 1384) actually
shows: a spread that never goes negative, peaks around 7-8 percentage
points during the Great Depression -- well above its next-highest peak,
during the 2008-09 financial crisis -- and stays below the Aaa-Treasury
spread (the extension's variant) at every point, since Aaa bonds carry less
default risk than Baa bonds.

Like the Table I/II integration tests, these read the processed monthly
FRED parquet file and so are skipped until the pipeline has been run.
"""

from pathlib import Path

import pandas as pd
import pytest
from matplotlib import pyplot as plt

import replicate_figure_1 as f1
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))

_NEEDED = ["fred_final_series_monthly.parquet"]


def _data_ready():
    return all((PROCESSED_DATA_DIR / f).exists() for f in _NEEDED)


requires_data = pytest.mark.skipif(
    not _data_ready(),
    reason="processed parquet data not built; run `doit` first",
)


@pytest.fixture(scope="module")
def monthly_df():
    return pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_monthly.parquet")


@requires_data
def test_baa_spread_peaks_during_great_depression_above_gfc_peak(monthly_df):
    """The paper's Figure I shows the Depression-era peak (~1932) as the
    series' historical high, distinctly above the next-highest peak during
    the 2008-09 financial crisis."""
    spread = monthly_df["BAA_Treasury_spread"].dropna()
    depression_peak = spread.loc["1929":"1937"].max()
    gfc_peak = spread.loc["2008":"2010"].max()

    assert 6.5 < depression_peak < 8.5
    assert 5.0 < gfc_peak < 7.0
    assert depression_peak > gfc_peak


@requires_data
def test_baa_spread_is_nonnegative_over_full_sample(monthly_df):
    spread = monthly_df["BAA_Treasury_spread"].dropna()
    assert (spread >= 0).all()


@requires_data
def test_aaa_spread_never_exceeds_baa_spread(monthly_df):
    """Aaa-rated bonds carry less default risk than Baa-rated bonds, so the
    Aaa-Treasury spread (Figure I's extension variant) should sit at or
    below the Baa-Treasury spread in (essentially) every month."""
    both = monthly_df[["BAA_Treasury_spread", "AAA_Treasury_spread"]].dropna()
    assert (both["AAA_Treasury_spread"] <= both["BAA_Treasury_spread"]).mean() > 0.98


@requires_data
def test_plot_figure_1_spans_full_replication_window(monthly_df):
    fig, spread = f1.plot_figure_1(
        monthly_df, f1.BUFFER_START, f1.REP_END, spread_col="BAA_Treasury_spread"
    )
    try:
        assert spread.index.min().year <= 1926
        assert spread.index.max().year == 2015
        # Roughly monthly cadence over ~90 years.
        assert len(spread) > 1000
    finally:
        plt.close(fig)


@requires_data
def test_plot_figure_1_aaa_variant_runs(monthly_df):
    fig, spread = f1.plot_figure_1(
        monthly_df, f1.BUFFER_START, f1.REP_END, spread_col="AAA_Treasury_spread"
    )
    try:
        assert len(spread) > 1000
    finally:
        plt.close(fig)
