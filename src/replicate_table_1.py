"""
Replicate Table I of Lopez-Salido, Stein & Zakrajsek (2017).

Table I forecasts next-year real GDP-per-capita growth on the current-year
change in the Baa-Treasury spread with lagged growth, Newey-West (HAC) errors:

    dy_{t+1} = b1 * d_credit_spread_t + b2 * dy_t (+ war dummies) + e

Published QJE (1929-2015, col 1): d_s = -1.997 (0.746), R2bar = 0.425.

Two spec choices are emitted side by side because the published Table I note is
ambiguous about controls (it lists only "a constant"):
  - "dummies": includes WWII (1941-45) and Korea (1950-53), as in the FEDS
    working paper. Reproduces our earlier -1.984 with dy_t ~ 0.549.
  - "nodummies": constant only, the literal reading of the QJE note. Expected
    to move dy_t toward the published 0.479.

Each spec is run over two windows: the 1929-2015 replication sample and a
1929-2025 extension. Equity columns (2)-(3), which need the Shiller S&P 500
return, are not built here (that data lands via #3).
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


def newey_west_lags(n_obs):
    """Newey-West (1994) automatic bandwidth rule of thumb."""
    return max(int(np.floor(4 * (n_obs / 100.0) ** (2 / 9))), 1)


def build_panel():
    df = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
    df["gdp_pc_growth"] = year_over_year_growth(df["GDP_per_capita"])
    df["d_credit_spread"] = df["BAA_Treasury_spread"].diff()
    df["dy_next"] = df["gdp_pc_growth"].shift(-1)
    df["wwii"] = df.index.isin(range(1941, 1946)).astype(int)
    df["korea"] = df.index.isin(range(1950, 1954)).astype(int)
    return df


def run_column_1(df, start, end, use_dummies):
    regressors = ["d_credit_spread", "gdp_pc_growth"]
    if use_dummies:
        regressors += ["wwii", "korea"]
    d = df.loc[start:end, ["dy_next"] + regressors].dropna()
    X = sm.add_constant(d[regressors])
    return sm.OLS(d["dy_next"], X).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))}
    )


def emit(res, start, end, label):
    table = pd.DataFrame(
        {
            "(1) Credit only": {
                "$\\Delta s_t$": f"{res.params['d_credit_spread']:.3f}",
                "\\quad (s.e.)": f"({res.bse['d_credit_spread']:.3f})",
                "$\\Delta y_t$": f"{res.params['gdp_pc_growth']:.3f}",
                "\\quad (s.e.) ": f"({res.bse['gdp_pc_growth']:.3f})",
                "$\\bar R^2$": f"{res.rsquared_adj:.3f}",
                "N": str(int(res.nobs)),
            }
        }
    )
    table.columns.name = (
        f"Dep. var: $\\Delta y_{{t+1}}$, real GDP p.c. (pct.), {start}-{end}"
    )
    out = OUTPUT_DIR / f"table_1_{label}.tex"
    out.write_text(table.to_latex(escape=False))
    print(f"{label}: d_s={res.params['d_credit_spread']:.3f}, "
          f"dy_t={res.params['gdp_pc_growth']:.3f}, N={int(res.nobs)} -> {out.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_panel()
    for use_dummies, tag in [(True, "dummies"), (False, "nodummies")]:
        emit(run_column_1(df, REP_START, REP_END, use_dummies),
             REP_START, REP_END, f"replication_{tag}")
        emit(run_column_1(df, REP_START, EXT_END, use_dummies),
             REP_START, EXT_END, f"extended_{tag}")


if __name__ == "__main__":
    main()
