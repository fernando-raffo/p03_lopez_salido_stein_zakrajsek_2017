"""
Replicate Table 1 of Lopez-Salido, Stein & Zakrajsek (2017) (issue #5).

Table 1 forecasts next-year real GDP-per-capita growth (delta-y_{t+1}) with the
current-year change in the Baa-Treasury spread (delta-s_t), lagged growth, and
WWII / Korean-War dummies, using Newey-West (HAC) standard errors.

    delta-y_{t+1} = b1*delta-s_t + b2*delta-y_t + controls + e

Paper target (FEDS 1929-2013, col 1): delta-s_t = -2.007 (0.744), R2bar = 0.501.
This repo runs 1929-2015, so expect a value close to but not equal to -2.0.

SCOPE: Table 1 cols (2)-(4) pit the credit spread against the value-weighted
stock-market return r^M_t. Per the paper's data appendix that series is from
CRSP; it is NOT in the FRED pipeline yet (see issue #3 / Shiller & equity data).
Only the FRED-buildable credit-only column (1) is produced here; the equity
columns are left for #3.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
REP_START = config("REPLICATION_START_DATE").year
REP_END = config("REPLICATION_END_DATE").year


def newey_west_lags(n_obs):
    """Newey-West (1994) automatic bandwidth rule of thumb."""
    return max(int(np.floor(4 * (n_obs / 100.0) ** (2 / 9))), 1)


def run_column_1(df):
    regressors = ["d_credit_spread", "gdp_pc_growth", "wwii", "korea"]
    d = df.loc[REP_START:REP_END, ["dy_next"] + regressors].dropna()
    X = sm.add_constant(d[regressors])
    y = d["dy_next"]
    return sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))}
    )


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = pd.read_parquet(PROCESSED_DATA_DIR / "fred_growth_series_annual.parquet")

    df["wwii"] = df.index.isin(range(1941, 1946)).astype(int)
    df["korea"] = df.index.isin(range(1950, 1954)).astype(int)
    df["dy_next"] = df["gdp_pc_growth"].shift(-1)  # delta-y_{t+1}

    res = run_column_1(df)

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
        f"Dep. var: $\\Delta y_{{t+1}}$, real GDP p.c. (pct.), {REP_START}-{REP_END}"
    )
    (OUTPUT_DIR / "table_1.tex").write_text(table.to_latex(escape=False))

    print(table)
    print(
        "\ndelta-s_t =",
        round(res.params["d_credit_spread"], 3),
        "(paper col 1 target ~ -2.0)",
    )
    print("wrote", OUTPUT_DIR / "table_1.tex")
    print("NOTE: cols 2-4 need the CRSP/Shiller equity return (issue #3).")


if __name__ == "__main__":
    main()
