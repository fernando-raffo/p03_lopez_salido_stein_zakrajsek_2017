# Methodology

## Approach

The paper's thesis is that "frothy" credit markets - narrow credit spreads
and a high share of below-investment-grade issuance - mean-revert, and that
this reversion forecasts a slowdown in real activity roughly two years later.
The replication reconstructs this in two stages, following the paper's
Tables I and II:

- **Table I** regresses one-year-ahead real GDP-per-capita growth on the
  lagged change in the Baa-Treasury credit spread and the lagged S&P total
  return, separately and together with treasury rates/inflation as control variables, using
  Newey-West HAC standard errors (`replicate_table_1.py`).
- **Table II** is the paper's two-step "credit-market sentiment" design
  (`replicate_table_2.py`):
  - *Auxiliary (first-step) regressions*, fit by OLS: the change in the
    credit spread on `ln(HYS)_{t-2}` and the spread level `s_{t-2}`; the S&P
    return on `ln[P/E10]_{t-2}`.
  - *Second-step regression*: GDP-per-capita growth on the fitted values
    from the two auxiliary regressions, plus lagged growth, short- and
    long-rate changes, and inflation.
  - The paper estimates all of this jointly by nonlinear least squares "to
    take into account the generated-regressor nature of the expected
    returns" (p. 1388, footnote 12). Because the system is block-recursive,
    the NLLS point estimates coincide with simple plug-in two-step OLS - but
    the plug-in approach understates the second-step standard errors, since
    it treats the first-step fitted values as data rather than estimates
    with their own sampling variance. `replicate_table_2.py` corrects for
    this with a stacked M-estimator (the classic Murphy and Topel (1985)
    generated-regressors correction), so the point estimates match plug-in
    OLS but the reported standard errors match the paper's joint-NLLS
    methodology.
- **Figures I and II** are the visual counterparts: the credit spread over
  time with NBER recessions shaded, and lagged sentiment plotted against
  subsequent GDP growth, with the fitted line and influential observations
  highlighted.

Every regression and chart is produced twice: once with the paper's original
Baa-Treasury spread, and once with an analogous measure built from the
Aaa-Treasury spread. Each of these is additionally produced over two sample windows - the paper's 1929-2015
replication window and an extended window through the most recent available
data. A further case study (`04_case_study.ipynb`) applies the Table II
first-step regression, estimated and held fixed on 1929-2015 data, out of
sample to 2020-2022, comparing its predicted change in the credit spread
against the realized change through the COVID-19 shock and the subsequent
period of market froth.

## Implementation Notes

- **Orchestration.** The full pipeline is driven by `doit` (`dodo.py`): each
  task declares its file dependencies and targets, so `doit` only reruns
  steps whose inputs changed. The stage order is pull → process → summary
  statistics / Table I / Table II / Figure I / Figure II → interactive charts
  → notebooks → LaTeX report → ChartBook site → tests.
- **Two spread variants throughout.** Rather than branching the analysis, the
  Baa and Aaa variants are computed side by side at every stage - each
  replication script emits both by default, and every chart/table in
  `chartbook.toml` has a Baa version and an `_aaa_` counterpart.
- **Vintage and series-stitching judgment calls.** Several FRED series (long-
  and short-term interest rates, GDP, population) only cover part of the
  1929-2025 window on their own and are stitched together from multiple FRED
  series to reach further back or forward than a single series allows. 
- **Splicing the high-yield share.** The Mergent FISD reconstruction cannot
  reach back to 1929 on its own, so the final `ln(HYS)` series splices the
  published Greenwood-Hanson (2013) values (1926-2008) with the FISD
  reconstruction (2009-present) - see [Data Sources](data_sources.md).
- **Caveats and limitations.**
  - Replicated coefficients will not match the published QJE numbers
    exactly: several of the underlying FRED, Shiller, and Greenwood-Hanson
    series are a newer vintage than what the original authors used, and the
    FISD reconstruction's rating/denominator conventions differ subtly from
    the original Greenwood-Hanson methodology.
  - A WRDS subscription is required to reconstruct the post-2008 high-yield
    share from primary data; without one, `source="historical"` falls back to
    the published series through 2008 (see [Data Sources](data_sources.md)).
  - The integration tests (`test_replicate_table_1.py`,
    `test_replicate_table_2.py`) check the replicated coefficients against
    the published values with tolerances wide enough to accept this repo's
    documented replication values while still catching a sign flip or a
    gross regression error. They read the processed parquet files, so they
    skip automatically if the pipeline has not been run yet, and otherwise
    run as the final step of `doit`.
