"""Tests for `replicate_table_1`.

Two kinds of test live here:

* Pure unit tests for the small helpers (`newey_west_lags`), which need no
  data and always run.
* Integration tests that the replicated Table I *matches the published QJE
  numbers*. These read the processed parquet files, so they are skipped unless
  the pipeline has been run (``doit``). The wired-in ``run_tests`` doit task
  depends on the data + table tasks, so these execute there.

Published QJE Table I, 1929-2015 (from the paper and the module docstring):
    col (1)  Delta s_{t-1}  = -1.997
    col (2)  r_t^SP         =  0.081
    col (3)  Delta s_{t-1}  = -2.061
    col (3)  r_t^SP         =  0.029
Tolerances below are set wide enough to admit this repo's own (documented)
replication values, while still failing on a sign flip or a gross regression.
"""

from pathlib import Path

import numpy as np
import pytest

import replicate_table_1 as t1
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
REP_START = t1.REP_START
REP_END = t1.REP_END

_NEEDED = ["fred_final_series_annual.parquet", "shiller_data_annual.parquet"]


def _data_ready():
    return all((PROCESSED_DATA_DIR / f).exists() for f in _NEEDED)


requires_data = pytest.mark.skipif(
    not _data_ready(),
    reason="processed parquet data not built; run `doit` first",
)

BASE = ["gdp_pc_growth"]
COL3_REGRESSORS = [
    "d_credit_spread",
    "sp_return",
    "d_treasury_3mo",
    "d_treasury_10yr",
    "CPI_inflation",
] + BASE


# --------------------------------------------------------------------------- #
# Pure unit tests (no data needed)
# --------------------------------------------------------------------------- #
def test_newey_west_lags_matches_rule():
    for n in (30, 87, 100, 250, 500):
        expected = max(int(np.floor(4 * (n / 100.0) ** (2 / 9))), 1)
        assert t1.newey_west_lags(n) == expected


def test_newey_west_lags_floor_is_one():
    assert t1.newey_west_lags(1) >= 1
    assert t1.newey_west_lags(5) >= 1


# --------------------------------------------------------------------------- #
# Integration: does the replication match the published paper?
# --------------------------------------------------------------------------- #
@requires_data
def test_table_1_matches_published_qje():
    df = t1.build_panel("BAA_Treasury_spread")
    col1 = t1.run_regression(df, ["d_credit_spread"] + BASE, REP_START, REP_END)
    col2 = t1.run_regression(df, ["sp_return"] + BASE, REP_START, REP_END)
    col3 = t1.run_regression(df, COL3_REGRESSORS, REP_START, REP_END)

    # Point estimates close to the published QJE values.
    assert col1.params["d_credit_spread"] == pytest.approx(-1.997, abs=0.25)
    assert col2.params["sp_return"] == pytest.approx(0.081, abs=0.03)
    assert col3.params["d_credit_spread"] == pytest.approx(-2.061, abs=0.30)
    assert col3.params["sp_return"] == pytest.approx(0.029, abs=0.03)


@requires_data
def test_table_1_credit_beats_equity():
    """The paper's headline (col 3): with both predictors in, the credit
    coefficient stays large and significant while the equity coefficient
    collapses toward zero."""
    df = t1.build_panel("BAA_Treasury_spread")
    col2 = t1.run_regression(df, ["sp_return"] + BASE, REP_START, REP_END)
    col3 = t1.run_regression(df, COL3_REGRESSORS, REP_START, REP_END)

    # Credit: right sign and significant.
    assert col3.params["d_credit_spread"] < 0
    assert col3.pvalues["d_credit_spread"] < 0.05
    # Equity coefficient shrinks once credit is included...
    assert abs(col3.params["sp_return"]) < abs(col2.params["sp_return"])
    # ...and is dwarfed by the credit effect.
    assert abs(col3.params["d_credit_spread"]) > 10 * abs(col3.params["sp_return"])


@requires_data
def test_build_panel_has_expected_columns():
    df = t1.build_panel("BAA_Treasury_spread")
    for col in ("gdp_pc_growth", "d_credit_spread", "sp_return", "dy_next"):
        assert col in df.columns
    # dy_next is next-year growth, i.e. gdp_pc_growth shifted back one year.
    aligned = df["dy_next"].dropna()
    assert len(aligned) > 50
