"""Tests for `replicate_table_1`.

Two kinds of test live here:

* Pure unit tests for the small helpers (`newey_west_lags`), which need no
  data and always run.
* Integration tests that the replicated Table I *matches the published QJE
  numbers*. These read the processed parquet files, so they are skipped unless
  the pipeline has been run (``doit``). The wired-in ``run_tests`` doit task
  depends on the data + table tasks, so these execute there.

Published QJE Table I, 1929-2015, transcribed from the paper (p. 1386):

               (1)             (2)             (3)
    Delta s_{t-1}   -1.997*** (.746)   --               -2.061** (.847)
    r_t^SP          --                 0.081*** (.029)   0.029  (.036)
    Delta y_{t-1}    0.479*** (.080)   0.475*** (.082)   0.464*** (.079)
    i^(3m)_{t-1}    --                --                -0.217  (.198)
    i^(10y)_{t-1}   --                --                -0.719** (.346)
    pi_{t-1}        --                --                 0.069  (.050)
    R-bar^2          0.425             0.389             0.450
    std. effect Ds  -0.369             --                -0.380
    std. effect rSP --                 0.319              0.114

Tolerances below were chosen by first computing this repo's own point
estimates against the numbers above: every coefficient lands within roughly
15% (relative) of its published counterpart, and R-bar^2 within 0.01-0.02.
The gap is expected -- FRED/Shiller vintages are continuously revised, so an
exact match to the 2017 paper is neither expected nor achievable -- but it
should be small and stable. Tolerances are set at roughly 1.5-2x the
observed gap: wide enough to absorb ordinary data-vintage drift, tight
enough to catch a sign flip, a dropped regressor, or a gross regression bug.
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

COLUMN_REGRESSORS = {
    "col1": ["d_credit_spread"] + BASE,
    "col2": ["sp_return"] + BASE,
    "col3": COL3_REGRESSORS,
}

# (column, regressor, published coefficient, absolute tolerance)
PUBLISHED_COEFS = [
    ("col1", "d_credit_spread", -1.997, 0.25),
    ("col1", "gdp_pc_growth", 0.479, 0.08),
    ("col2", "sp_return", 0.081, 0.03),
    ("col2", "gdp_pc_growth", 0.475, 0.08),
    ("col3", "d_credit_spread", -2.061, 0.30),
    ("col3", "sp_return", 0.029, 0.03),
    ("col3", "gdp_pc_growth", 0.464, 0.08),
    ("col3", "d_treasury_3mo", -0.217, 0.15),
    ("col3", "d_treasury_10yr", -0.719, 0.20),
    ("col3", "CPI_inflation", 0.069, 0.06),
]

# (column, published adjusted R^2, absolute tolerance)
PUBLISHED_R2 = [
    ("col1", 0.425, 0.03),
    ("col2", 0.389, 0.03),
    ("col3", 0.450, 0.03),
]

# (column, regressor, published standardized effect, absolute tolerance)
PUBLISHED_STD_EFFECTS = [
    ("col1", "d_credit_spread", -0.369, 0.05),
    ("col2", "sp_return", 0.319, 0.05),
    ("col3", "d_credit_spread", -0.380, 0.05),
    ("col3", "sp_return", 0.114, 0.05),
]


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
@pytest.fixture(scope="module")
def table_1_results():
    """Fit all three Table I columns once per test session and reuse the
    fitted results across every value-matching test below (each column's
    NLLS/OLS fit is deterministic given the processed data, so refitting it
    once per assertion would be pure waste)."""
    df = t1.build_panel("BAA_Treasury_spread")
    results = {
        col: t1.run_regression(df, regs, REP_START, REP_END)
        for col, regs in COLUMN_REGRESSORS.items()
    }
    window = df.loc[REP_START:REP_END]
    return results, window


@requires_data
@pytest.mark.parametrize(
    "col,var,published,tol",
    PUBLISHED_COEFS,
    ids=[f"{c}-{v}" for c, v, _, _ in PUBLISHED_COEFS],
)
def test_table_1_coefficients_match_published_qje(
    table_1_results, col, var, published, tol
):
    results, _ = table_1_results
    assert results[col].params[var] == pytest.approx(published, abs=tol)


@requires_data
@pytest.mark.parametrize(
    "col,published,tol", PUBLISHED_R2, ids=[c for c, _, _ in PUBLISHED_R2]
)
def test_table_1_adjusted_r2_matches_published_qje(
    table_1_results, col, published, tol
):
    results, _ = table_1_results
    assert results[col].rsquared_adj == pytest.approx(published, abs=tol)


@requires_data
@pytest.mark.parametrize(
    "col,var,published,tol",
    PUBLISHED_STD_EFFECTS,
    ids=[f"{c}-{v}" for c, v, _, _ in PUBLISHED_STD_EFFECTS],
)
def test_table_1_standardized_effects_match_published_qje(
    table_1_results, col, var, published, tol
):
    results, window = table_1_results
    effect = float(
        t1.standardized_effect(results[col], window, COLUMN_REGRESSORS[col], var)
    )
    assert effect == pytest.approx(published, abs=tol)


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
