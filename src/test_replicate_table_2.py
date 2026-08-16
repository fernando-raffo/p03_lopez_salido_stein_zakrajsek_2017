"""Tests for `replicate_table_2`.

The Table II first stage is the "sentiment engine": two auxiliary regressions
whose fitted values drive the second-step growth regressions. The
paper-matching tests assert the LSZ signs and the headline second-step result;
they read processed parquet files and so are skipped until the pipeline has
been run (``doit``).

LSZ (2017) Table II, first stage, 1929-2015:
    Delta s_t  on  ln(HYS)_{t-2}  -> positive  (froth predicts widening)
    Delta s_t  on  s_{t-2}        -> negative  (wide spreads mean-revert)
    r_t^SP     on  ln[P/E10]_{t-2}-> negative  (published display value -0.134)
Second step, col (3): the fitted spread change forecasts *lower* growth
(negative, significant).
"""

from pathlib import Path

import numpy as np
import pytest

import replicate_table_2 as t2
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
# Pure unit tests (no data needed)
# --------------------------------------------------------------------------- #
def test_newey_west_lags_matches_rule():
    for n in (40, 85, 120, 300):
        expected = max(int(np.floor(4 * (n / 100.0) ** (2 / 9))), 1)
        assert t2.newey_west_lags(n) == expected


def test_column_regressors_are_wired_to_fitted_values():
    # The second-step columns must use the *fitted* first-stage regressors.
    assert "d_s_hat" in t2.COLUMN_REGRESSORS["col1"]
    assert "r_sp_hat" in t2.COLUMN_REGRESSORS["col2"]
    assert set(t2.COLUMN_REGRESSORS["col3"]) >= {"d_s_hat", "r_sp_hat"}


# --------------------------------------------------------------------------- #
# Integration: does the sentiment engine match the paper?
# --------------------------------------------------------------------------- #
@requires_data
def test_table_2_first_stage_signs():
    df = t2.build_panel("BAA_Treasury_spread")
    res = t2.run_table_2(df)

    aux_spread = res["aux_spread"]
    # Froth predicts widening; wide spreads mean-revert.
    assert aux_spread.params["ln_hys_lag2"] > 0
    assert aux_spread.params["spread_lag2"] < 0

    aux_return = res["aux_return"]
    # A high CAPE forecasts lower subsequent equity returns.
    assert aux_return.params["ln_pe10_lag2"] < 0
    # Display-scaled (x0.01) coefficient sits near the paper's -0.134.
    assert -0.30 < aux_return.params["ln_pe10_lag2"] * 0.01 < -0.02


@requires_data
def test_table_2_second_stage_credit_forecasts_growth():
    df = t2.build_panel("BAA_Treasury_spread")
    res = t2.run_table_2(df)
    col3 = res["col3"]
    # Fitted spread widening predicts lower growth: negative and significant.
    assert col3.params["d_s_hat"] < 0
    assert col3.pvalues["d_s_hat"] < 0.10


@requires_data
def test_table_2_aaa_variant_runs():
    """The Aaa-spread variant should build and estimate without error and keep
    the mean-reversion sign."""
    df = t2.build_panel("AAA_Treasury_spread")
    res = t2.run_table_2(df)
    assert res["aux_spread"].params["spread_lag2"] < 0
    for col in ("col1", "col2", "col3", "col4"):
        assert res[col].nobs > 30
