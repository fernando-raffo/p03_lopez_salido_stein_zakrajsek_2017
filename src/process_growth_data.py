"""
Growth-rate processing for the LSZ (2017) replication (issue #5).

Builds the YOY / QOQ / first-difference transforms that were missing from the
cleaned FRED data and that Table 1 needs: next-year GDP growth, the change in
the Baa-Treasury spread, and the change in the 3-month Treasury yield.
"""
from pathlib import Path

import numpy as np
import pandas as pd

import pull_fred
from settings import config

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))


def year_over_year_growth(series):
    """Annual YOY log growth (percent) of a level series indexed by year."""
    s = series.astype(float).sort_index()
    return 100.0 * np.log(s).diff()


def quarter_over_quarter_growth(series):
    """QOQ log growth (percent) of a level series indexed by quarterly dates."""
    s = series.astype(float).sort_index()
    return 100.0 * np.log(s).diff()


def forward_cumulative_growth(level_series, horizon):
    """Cumulative log growth (percent) from t to t+horizon."""
    s = np.log(level_series.astype(float).sort_index())
    return 100.0 * (s.shift(-horizon) - s)


def build_annual_growth_panel(annual):
    df = annual.copy()
    # delta-y_t : YOY log growth of real GDP per capita (percent)
    df["gdp_pc_growth"] = year_over_year_growth(df["GDP_per_capita"])
    # s_t and its change; the Baa-Treasury credit spread (pps)
    df["credit_spread"] = df["BAA"] - df["Treasury_10yr"]
    df["d_credit_spread"] = df["credit_spread"].diff()
    # term spread and short-rate change (Table 1 cols 3-4 controls)
    df["term_spread"] = df["Treasury_10yr"] - df["Treasury_3mo"]
    df["d_treasury_3mo"] = df["Treasury_3mo"].diff()
    # forward cumulative growth (kept for later horizon analysis)
    for h in (1, 2, 3):
        df[f"gdp_pc_fwd_{h}yr"] = forward_cumulative_growth(df["GDP_per_capita"], h)
    return df


def build_quarterly_growth_panel(data_dir=RAW_DATA_DIR):
    raw = pull_fred.load_fred(data_dir=data_dir)
    gdp_col = "GDPC1" if "GDPC1" in raw.columns else next(
        (c for c in raw.columns if "GDP" in c.upper() and "CAP" not in c.upper()),
        None,
    )
    if gdp_col is None:
        raise SystemExit(
            f"No quarterly real GDP column in raw FRED data. "
            f"Available: {list(raw.columns)}"
        )
    gdp = raw[gdp_col].dropna()
    out = pd.DataFrame({gdp_col: gdp})
    out["gdp_qoq_growth"] = quarter_over_quarter_growth(gdp)
    return out


ANNUAL_README = """# FRED Growth Series (Annual) README

Documents `fred_growth_series_annual.parquet`, produced by `process_growth_data.py`
from `fred_final_series_annual.parquet` (issue #5).

| Column | Description |
| --- | --- |
| gdp_pc_growth | YOY log growth (percent) of real GDP per capita = delta-y_t. |
| credit_spread | Baa minus 10-year Treasury yield (pps) = s_t. |
| d_credit_spread | Annual change in the credit spread = delta-s_t. |
| term_spread | 10-year minus 3-month Treasury yield. |
| d_treasury_3mo | Annual change in the 3-month Treasury yield. |
| gdp_pc_fwd_{1,2,3}yr | Cumulative log growth (percent) of GDP/capita, t to t+h. |

All original columns of the annual cleaned file are retained.
"""

QUARTERLY_README = """# FRED Growth Series (Quarterly) README

Documents `fred_growth_series_quarterly.parquet`, produced by `process_growth_data.py`.

| Column | Description |
| --- | --- |
| GDPC1 | Real Gross Domestic Product (quarterly), from the raw FRED pull. |
| gdp_qoq_growth | QOQ log growth (percent) of real GDP. |
"""


def main():
    PROCESSED_DATA_DIR.mkdir(parents=True, exist_ok=True)

    annual = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
    annual_growth = build_annual_growth_panel(annual)
    annual_growth.to_parquet(PROCESSED_DATA_DIR / "fred_growth_series_annual.parquet")
    (PROCESSED_DATA_DIR / "fred_growth_series_annual_readme.md").write_text(ANNUAL_README)

    quarterly_growth = build_quarterly_growth_panel()
    quarterly_growth.to_parquet(PROCESSED_DATA_DIR / "fred_growth_series_quarterly.parquet")
    (PROCESSED_DATA_DIR / "fred_growth_series_quarterly_readme.md").write_text(QUARTERLY_README)

    print("wrote annual growth panel:", annual_growth.shape,
          "| quarterly growth panel:", quarterly_growth.shape)


if __name__ == "__main__":
    main()
