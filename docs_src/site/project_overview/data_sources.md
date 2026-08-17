# Data Sources

## Overview

Describe the datasets used in this project, how they are obtained, and any
relevant licensing or access considerations.

## Datasets

| Dataset | Source | Frequency | Description |
|---------|--------|-----------|-------------|
| FRED macro/finance series | FRED (`pull_fred.py`) | Daily/Monthly | GDP, CPI, rates, Fed balance sheet, etc. |
| Shiller stock-market data (CAPE / P/E10) | Robert Shiller's website `ie_data.xls` (`pull_shiller.py`) | Monthly (also annual) | S&P price, dividends, earnings, CPI, GS10, and the cyclically adjusted P/E (`P/E10`). Provides `ln[P/E10]`, the first-step stock-return predictor in Lopez-Salido, Stein, and Zakrajsek (2017). |
| Greenwood-Hanson high-yield share (HYS) | Mergent FISD via WRDS, or a raw issuance file (`pull_greenwood_hanson.py`) | Annual | Fraction of gross nonfinancial corporate bond issuance rated below investment grade. Provides `ln(HYS)`, a first-step predictor of changes in the Baa-Treasury spread. |

## Data Pipeline

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
- **Store.** Cached to `_data/shiller_pe.parquet` / `.csv` (monthly) and
  `_data/shiller_pe_annual.parquet` / `.csv` (annual). Git-ignored.

### Greenwood-Hanson high-yield share (`pull_greenwood_hanson.py`)

- **Pull.** There is no free public CSV/API. The series is *constructed* from
  bond-level gross issuance. The authoritative source is **Mergent FISD via
  WRDS** (`source="fisd"`, needs `WRDS_USERNAME`). Collaborators without WRDS can
  set `source="raw"` and drop an issuance file into `data_manual/`.
- **Process.** Restrict to U.S. nonfinancial corporate issues, flag each as high
  yield if rated below investment grade, then for each year compute
  `HYS = high-yield issuance / total issuance`. `compute_hy_share` is a pure,
  unit-tested function; `ln_hy_share = ln(HYS)`.
- **Store.** Cached to `_data/greenwood_hanson_hys.parquet` / `.csv`. Git-ignored.

### Notes

- All pulled and derived data live in `_data/`, which is git-ignored, so **no
  data is committed to the repo** (see issue #3). The raw high-yield-share file
  in `data_manual/` is also explicitly git-ignored.
- Run the pulls with `doit pull` (or individually,
  `doit pull:shiller`, `doit pull:greenwood_hanson`).
