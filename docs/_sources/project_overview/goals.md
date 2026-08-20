# Goals

## Objectives

- Replicate Figures I-II and Tables I-II of López-Salido, Stein, and Zakrajšek
  (2017, *QJE*) from primary sources. Pull the underlying data from FRED,
  Robert Shiller's data website, and the Greenwood-Hanson high-yield share
  (published historical series plus a Mergent FISD/WRDS reconstruction).
- Extend the sample period beyond the original paper's 2015 cutoff through
  the most recent data available from each source, so the replicated and
  extended series reflect current market conditions rather than stopping at
  the original publication's end date.
- Extend the replication along two axes: (1) an analogous credit-market
  sentiment measure built from the Aaa-Treasury spread instead of the
  Baa-Treasury spread, produced in parallel through every stage of the
  pipeline, and (2) a case study applying the 1929-2015-fit sentiment signal
  out of sample to the 2020-2022 COVID shock and its aftermath.
- Package the whole project as a reproducible, one-command ChartBook
  pipeline (`doit`) that pulls data, runs the replication, renders guided
  notebooks, compiles a LaTeX report, and publishes a documented site with
  interactive charts and dataframe provenance.

## Success Criteria

- A clean clone with valid WRDS credentials can run `doit` end to end and
  produce every table, figure, notebook, and the compiled `reports/report.pdf`
  without manual intervention.
- The replicated Table I and Table II coefficients fall within the
  tolerances checked by the automated integration tests
  (`test_replicate_table_1.py`, `test_replicate_table_2.py`), which compare
  them against the published QJE values as part of `doit`'s `run_pytest` step.
- Every replication figure and table has a corresponding Aaa-spread variant
  and an extended-sample (through the most recent available data) variant.
- The COVID case study (`04_case_study.ipynb`) shows whether the sentiment
  signal's predicted change in the credit spread tracks the realized change
  over 2020-2022, with its caveats documented alongside the result.
- `chartbook build` publishes a site where every dataframe, chart, and
  notebook is documented.
