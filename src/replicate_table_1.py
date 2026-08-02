"""
Replicate Table I of Lopez-Salido, Stein & Zakrajsek (2017).

Three columns matching the published QJE table, Newey-West (HAC) errors:
    (1) credit only, (2) equity only, (3) both + rate/inflation controls.
Published QJE (1929-2015): col1 d_s=-1.997, col2 r^SP=0.081, col3 d_s=-2.061.

Two war-dummy variants x two windows (1929-2015, 1929-2025) are emitted.
Shiller inputs: sp500_price + dividend, to build the annual S&P 500 total log
return r_t = 100*log((P_t + D_t)/P_{t-1}). All other controls come from FRED.
Note: Shiller data ends 2023, so the equity columns of the extended window
effectively stop at 2023 (post-2023 rows drop out on the join).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from helper_functions import year_over_year_growth
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
REP_START = config("REPLICATION_START_DATE").year
REP_END = config("REPLICATION_END_DATE").year
EXT_END = config("EXTENSION_END_DATE").year

# Shiller annual file (from pull_shiller.py) and the columns we use.
SHILLER_FILE = PROCESSED_DATA_DIR / "shiller_data_annual.parquet"
PRICE_COL = "sp500_price"
DIV_COL = "dividend"


def newey_west_lags(n_obs):
    return max(int(np.floor(4 * (n_obs / 100.0) ** (2 / 9))), 1)


def load_sp_return():
    """Annual S&P 500 total log return (percent) from Shiller price + dividend."""
    sh = pd.read_parquet(SHILLER_FILE)
    if not np.issubdtype(sh.index.dtype, np.integer):
        sh.index = pd.to_datetime(sh.index).year
    if PRICE_COL not in sh.columns or DIV_COL not in sh.columns:
        raise SystemExit(
            f"Expected '{PRICE_COL}' and '{DIV_COL}' in {SHILLER_FILE.name}. "
            f"Available: {sh.columns.tolist()}"
        )
    P, D = sh[PRICE_COL].astype(float), sh[DIV_COL].astype(float)
    return (100.0 * np.log((P + D) / P.shift(1))).rename("sp_return")


def build_panel():
    df = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
    df["gdp_pc_growth"] = year_over_year_growth(df["GDP_per_capita"])
    df["d_credit_spread"] = df["BAA_Treasury_spread"].diff()
    df["d_treasury_3mo"] = df["Treasury_3mo"].diff()
    df["d_treasury_10yr"] = df["Treasury_10yr"].diff()
    df = df.join(load_sp_return(), how="left")
    df["dy_next"] = df["gdp_pc_growth"].shift(-1)
    df["wwii"] = df.index.isin(range(1941, 1946)).astype(int)
    df["korea"] = df.index.isin(range(1950, 1954)).astype(int)
    return df


def run_regression(df, regressors, start, end):
    d = df.loc[start:end, ["dy_next"] + regressors].dropna()
    return sm.OLS(d["dy_next"], sm.add_constant(d[regressors])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))}
    )


def cell(res, name):
    if name not in res.params:
        return "", ""
    return f"{res.params[name]:.3f}", f"({res.bse[name]:.3f})"


def emit(df, start, end, use_dummies, label):
    base = ["gdp_pc_growth"] + (["wwii", "korea"] if use_dummies else [])
    specs = {
        "(1)": ["d_credit_spread"] + base,
        "(2)": ["sp_return"] + base,
        "(3)": [
            "d_credit_spread",
            "sp_return",
            "d_treasury_3mo",
            "d_treasury_10yr",
            "CPI_inflation",
        ]
        + base,
    }
    rows = [
        "$\\Delta s_t$",
        "\\quad (s.e.)",
        "$r^{SP}_t$",
        "\\quad (s.e.) ",
        "$\\Delta y_t$",
        "$\\bar R^2$",
        "N",
    ]
    table = {}
    for col, regs in specs.items():
        res = run_regression(df, regs, start, end)
        ds, ds_se = cell(res, "d_credit_spread")
        rs, rs_se = cell(res, "sp_return")
        table[col] = {
            "$\\Delta s_t$": ds,
            "\\quad (s.e.)": ds_se,
            "$r^{SP}_t$": rs,
            "\\quad (s.e.) ": rs_se,
            "$\\Delta y_t$": f"{res.params['gdp_pc_growth']:.3f}",
            "$\\bar R^2$": f"{res.rsquared_adj:.3f}",
            "N": str(int(res.nobs)),
        }
    out = pd.DataFrame(table).reindex(rows)
    out.columns.name = f"Dep. var: $\\Delta y_{{t+1}}$, {start}-{end}"
    path = OUTPUT_DIR / f"table_1_{label}.tex"
    path.write_text(out.to_latex(escape=False))
    print(
        f"{label}: col1 d_s={table['(1)']['$\\Delta s_t$']}, "
        f"col2 r_sp={table['(2)']['$r^{SP}_t$']}, "
        f"col3 d_s={table['(3)']['$\\Delta s_t$']}/r_sp={table['(3)']['$r^{SP}_t$']} "
        f"-> {path.name}"
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_panel()
    for use_dummies in (True, False):
        suffix = "_dummies" if use_dummies else ""
        emit(df, REP_START, REP_END, use_dummies, f"replication{suffix}")
        emit(df, REP_START, EXT_END, use_dummies, f"extended{suffix}")


if __name__ == "__main__":
    main()
