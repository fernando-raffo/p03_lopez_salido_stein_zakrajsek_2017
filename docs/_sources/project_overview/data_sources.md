# Data Sources

## Overview

All data is pulled by code rather than committed to the repo: raw and
processed files live under `_data/`, which is git-ignored and fully
regenerable by rerunning the pipeline. The only data tracked in Git is
`data_manual/`, reserved for manually-collected inputs that cannot be
re-pulled automatically.

## Datasets

| Dataset | Source | Frequency | Description |
|---------|--------|-----------|-------------|
| FRED macro/financial series | [FRED](https://fred.stlouisfed.org/), via `pandas_datareader` (`pull_fred.py`) | Monthly, Quarterly | Moody's Aaa/Baa seasoned corporate bond yields, the 10-year Treasury yield, the 3-month T-bill rate, CPI, population, real GDP, and the NBER recession indicator. |
| Robert Shiller stock-market data | `ie_data.xls` from [Robert Shiller's data website](https://shillerdata.com/) (`pull_shiller.py`) | Monthly | S&P Composite price, dividends, earnings, CPI, GS10, and the cyclically adjusted price-earnings ratio (CAPE / P/E10). |
| Greenwood-Hanson high-yield share (HYS) | Published [Greenwood & Hanson (2013) series (1926-2008)](https://www.hbs.edu/faculty/Pages/item.aspx?num=44245) spliced with a Mergent FISD reconstruction via [WRDS](https://wrds-www.wharton.upenn.edu/) (2009-present) (`pull_greenwood_hanson.py`) | Annual | Fraction of gross nonfinancial corporate bond issuance rated below investment grade. |

## Data Pipeline

### FRED data (`pull_fred.py`)

- **Pull.** Downloads each series in `series_to_pull` (Aaa/Baa yields, GS10,
  historical long-term/short-term bond-yield series used to extend coverage
  before the modern series begin, TB3MS, CPI, population, GDP, USREC) via
  `pandas_datareader.data.DataReader`.
- **Store.** Cached to `_data/raw_data/fred.parquet` / `.csv`, with a data
  dictionary written to `_data/data_dictionaries/fred_data_dictionary.md`.
- **Process.** `process_fred_data_monthly.py` and `process_fred_data_annual.py`
  build the cleaned Baa/Aaa-Treasury credit spreads, GDP-per-capita growth,
  inflation, and the NBER recession indicator used throughout the
  replication, cached to `_data/processed_data/fred_final_series_monthly.parquet`
  and `fred_final_series_annual.parquet`.

### Robert Shiller's data (`pull_shiller.py`)

- **Pull.** `pull_shiller` downloads `ie_data.xls` from
  <https://shillerdata.com/> (default, overridable via the `SHILLER_URL`
  setting). Since shillerdata.com serves the workbook from a versioned CDN
  link that changes whenever Shiller updates the file, the puller scrapes the
  current link from the landing page rather than hard-coding it, then parses
  the `Data` sheet.
- **Structure.** The `Data` sheet is one row per month from 1871-01 onward. The
  raw `Date` column encodes October as `YYYY.1`, so the monthly index is rebuilt
  from row order rather than parsed from that column.
- **Process.** `process_shiller_annual` collapses to annual frequency (year-end
  by default) and adds `ln_pe10 = ln(P/E10)`.
- **Store.** Cached to `_data/raw_data/shiller_data.parquet` (monthly) and
  `_data/processed_data/shiller_data_annual.parquet`.

### Greenwood-Hanson high-yield share (`pull_greenwood_hanson.py`)

- **Pull.** There is no free public CSV/API for the high-yield share; it is
  *constructed* from bond-level gross issuance. The authoritative source for construction components is
  **Mergent FISD via WRDS** (`source="fisd"`, needs `WRDS_USERNAME`). Because FISD's usable coverage only reaches back to the
  early 1980s, the module also ships the published Greenwood & Hanson (2013)
  historical series (1926-2008), transcribed from printed sources for years
  before FISD coverage begins.
- **Process.** Restricts to U.S. nonfinancial corporate issues, flags each as
  high yield if rated below investment grade at issuance (using each issue's
  first Moody's rating), then computes `HYS = high-yield issuance / total
  issuance` per year. The default `source="spliced"` returns the
  published series through 2008 and appends the FISD reconstruction from 2009
  onward, giving a continuous series covering the full 1929-2015 replication
  sample.
- **Store.** The historical and FISD components are cached separately to
  `_data/raw_data/`, and the final spliced series used downstream is cached to
  `_data/processed_data/greenwood_hanson_hys.parquet`.
- **Known limitations.** The FISD reconstruction only accepts issues with a
  Moody's rating (issues rated only by S&P or Fitch are dropped from the
  denominator), and its earliest years contain too few issues to be reliable
  (the `n_issues` column is provided so thin years can be screened out).

### Notes

- All pulled and derived data live in `_data/`, which is git-ignored, so **no
  data is committed to the repo**.
- Run all three pulls together with `doit pull_data`, or run the full
  pipeline (pulls, processing, replication, notebooks, report, and site) with
  a single `doit`.
