"""
Replicate Table II of Lopez-Salido, Stein & Zakrajsek (2017).

Table II is a "two-step" regression of contemporaneous real GDP-per-capita
growth on financial-market sentiment measures that are themselves
predicted from twice-lagged information:

    Auxiliary (first-step) regressions, fit by OLS:
        Delta s_t = a0 + a1 * ln(HYS)_{t-2}   + a2 * s_{t-2}  + u_t
        r_t^SP    = b0 + b1 * ln[P/E10]_{t-2}                 + v_t

    Second-step regression (Newey-West HAC standard errors):
        Delta y_t = c0 + c1 * Delta-hat s_t + c2 * r-hat_t^SP
                        + c3 * Delta y_{t-1}
                        + c4 * Delta i_{t-1}^(3m) + c5 * Delta i_{t-1}^(10y)
                        + c6 * pi_{t-1} + e_t

where Delta-hat s_t and r-hat_t^SP are the *fitted* values from the two
auxiliary regressions above; columns (1)-(4) use different subsets of these
regressors and controls, see `COLUMN_REGRESSORS`.

Estimation-method caveat
-------------------------
The paper notes both steps "are estimated jointly ... by NLLS," which lets
the second-step standard errors account for sampling uncertainty in the
first-step (auxiliary) coefficients. This script instead runs the simpler,
standard two-step "plug-in" procedure: fit each auxiliary regression by OLS,
substitute its fitted values into the growth regression, then fit the growth
regression by OLS. Because the system is block-recursive (the auxiliary
coefficients solve their own normal equations regardless of the second
step), the point estimates should match the joint-NLLS estimates; only the
standard errors differ, since ours do not correct for the extra "generated
regressor" uncertainty from the first step.

Because `fred_final_series_annual.parquet` is trimmed to start in 1929 (the
replication start year), `s_{t-2}` is unavailable for 1929-1930, so the
effective estimation sample used here begins in 1931 rather than 1929.

Data: all series come from the annual parquet files in
`_data/processed_data`:
    - fred_final_series_annual.parquet: GDP_per_capita, CPI_inflation,
      BAA_Treasury_spread, Treasury_10yr, Treasury_3mo
    - greenwood_hanson_hys.parquet: ln_hy_share (ln HYS)
    - shiller_data_annual.parquet: sp500_price, dividend, ln_pe10
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from helper_functions import log_total_return, to_percent, year_over_year_growth
from settings import config

PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
OUTPUT_DIR = Path(config("OUTPUT_DIR"))
REP_START = config("REPLICATION_START_DATE").year
REP_END = config("REPLICATION_END_DATE").year
EXT_END = config("EXTENSION_END_DATE").year

# Regressors used in each second-step (growth) column, keyed to the columns
# of the panel built by `build_panel`.
COLUMN_REGRESSORS = {
    "col1": ["d_s_hat", "dy_lag1"],
    "col2": ["r_sp_hat", "dy_lag1"],
    "col3": ["d_s_hat", "r_sp_hat", "dy_lag1"],
    "col4": ["d_s_hat", "dy_lag1", "d_3mo_lag1", "d_10yr_lag1", "inflation_pct_lag1"],
}

# Row order and LaTeX labels for the main (second-step) table.
_MAIN_ROWS = [
    ("d_s_hat", r"$\Delta \hat s_t$"),
    ("r_sp_hat", r"$\hat r_t^{SP}$"),
    ("dy_lag1", r"$\Delta y_{t-1}$"),
    ("d_3mo_lag1", r"$\Delta i_{t-1}^{(3m)}$"),
    ("d_10yr_lag1", r"$\Delta i_{t-1}^{(10y)}$"),
    ("inflation_pct_lag1", r"$\pi_{t-1}$"),
]

# Row order and LaTeX labels for the auxiliary (first-step) table.
_AUX_ROWS = [
    ("ln_hys_lag2", r"$\ln \mathrm{HYS}_{t-2}$"),
    ("spread_lag2", r"$s_{t-2}$"),
    ("ln_pe10_lag2", r"$\ln[P/E10]_{t-2}$"),
]


def newey_west_lags(n_obs):
    """Newey-West (1994) automatic bandwidth rule of thumb."""
    return max(int(np.floor(4 * (n_obs / 100.0) ** (2 / 9))), 1)


def build_panel():
    """Assemble the annual panel of series needed for Table II."""
    fred = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
    hys = pd.read_parquet(PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet")
    shiller = pd.read_parquet(PROCESSED_DATA_DIR / "shiller_data_annual.parquet")

    shiller = shiller.copy()
    shiller.index = shiller.index.year
    shiller.index.name = "year"

    df = fred.copy()
    df["dy"] = year_over_year_growth(df["GDP_per_capita"])
    df["d_spread"] = df["BAA_Treasury_spread"].diff()
    df["d_3mo"] = df["Treasury_3mo"].diff()
    df["d_10yr"] = df["Treasury_10yr"].diff()
    df["inflation_pct"] = to_percent(df["CPI_inflation"])

    df["ln_hys"] = hys["ln_hy_share"]
    df["sp_return"] = log_total_return(shiller["sp500_price"], shiller["dividend"])
    df["ln_pe10"] = shiller["ln_pe10"]

    # Auxiliary-regression predictors, lagged two years per the paper.
    df["ln_hys_lag2"] = df["ln_hys"].shift(2)
    df["spread_lag2"] = df["BAA_Treasury_spread"].shift(2)
    df["ln_pe10_lag2"] = df["ln_pe10"].shift(2)

    # Predetermined (t-1) controls for the second-step regression.
    df["dy_lag1"] = df["dy"].shift(1)
    df["d_3mo_lag1"] = df["d_3mo"].shift(1)
    df["d_10yr_lag1"] = df["d_10yr"].shift(1)
    df["inflation_pct_lag1"] = df["inflation_pct"].shift(1)
    return df


def _fit_ols_hac(y, X):
    """OLS with a constant and Newey-West (HAC) standard errors."""
    X = sm.add_constant(X)
    return sm.OLS(y, X).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(y))}
    )


def run_table_2(df, start=REP_START, end=REP_END):
    """
    Estimate Table II over `df.loc[start:end]`: the two auxiliary
    (first-step) forecasting regressions, then the four second-step growth
    regressions that use their fitted values as regressors.

    Parameters
    ----------
    df : pandas.DataFrame
        Output of `build_panel`.
    start, end : int
        First and last calendar year (inclusive) of the estimation window.

    Returns
    -------
    dict
        Keys "aux_spread", "aux_return", "col1".."col4", each a fitted
        `statsmodels` OLS results object.
    """
    window = df.loc[start:end]

    aux_spread_vars = ["ln_hys_lag2", "spread_lag2"]
    d = window[["d_spread"] + aux_spread_vars].dropna()
    res_aux_spread = _fit_ols_hac(d["d_spread"], d[aux_spread_vars])

    aux_return_vars = ["ln_pe10_lag2"]
    d = window[["sp_return"] + aux_return_vars].dropna()
    res_aux_return = _fit_ols_hac(d["sp_return"], d[aux_return_vars])

    reg = window.assign(
        d_s_hat=res_aux_spread.fittedvalues.reindex(window.index),
        r_sp_hat=res_aux_return.fittedvalues.reindex(window.index),
    )

    results = {"aux_spread": res_aux_spread, "aux_return": res_aux_return}
    for col, regressors in COLUMN_REGRESSORS.items():
        d = reg[["dy"] + regressors].dropna()
        results[col] = _fit_ols_hac(d["dy"], d[regressors])
    return results


def _coef_and_se(res, var):
    if var not in res.params.index:
        return "--", ""
    return f"{res.params[var]:.3f}", f"({res.bse[var]:.3f})"


def _main_table(results):
    columns = {}
    for i, col in enumerate(["col1", "col2", "col3", "col4"], start=1):
        res = results[col]
        rows = {}
        for var, label in _MAIN_ROWS:
            coef, se = _coef_and_se(res, var)
            rows[label] = coef
            rows[f"{label} (se)"] = se
        rows["$R^2$"] = f"{res.rsquared:.3f}"
        rows["$N$"] = str(int(res.nobs))
        columns[f"({i})"] = rows
    table = pd.DataFrame(columns)
    table.columns.name = r"Dependent variable: $\Delta y_t$"
    return table


def _aux_table(results):
    res_s = results["aux_spread"]
    res_r = results["aux_return"]

    def col(res):
        rows = {}
        for var, label in _AUX_ROWS:
            coef, se = _coef_and_se(res, var)
            rows[label] = coef
            rows[f"{label} (se)"] = se
        rows["$R^2$"] = f"{res.rsquared:.3f}"
        return rows

    table = pd.DataFrame({r"$\Delta s_t$": col(res_s), r"$r_t^{SP}$": col(res_r)})
    table.columns.name = "Auxiliary regressions"
    return table


def emit_table_2(results, start, end, label):
    main_table = _main_table(results)
    aux_table = _aux_table(results)

    out = OUTPUT_DIR / f"table_2_{label}.tex"
    text = (
        f"% Table II replication ({label}): {start}-{end}\n"
        + main_table.to_latex(escape=False)
        + "\n"
        + aux_table.to_latex(escape=False)
    )
    out.write_text(text)

    print(f"{label} ({start}-{end}):")
    for i, col in enumerate(["col1", "col2", "col3", "col4"], start=1):
        res = results[col]
        coefs = ", ".join(f"{v}={res.params[v]:.3f}" for v in COLUMN_REGRESSORS[col])
        print(f"  ({i}) N={int(res.nobs)} R2={res.rsquared:.3f} {coefs}")
    aux_s, aux_r = results["aux_spread"], results["aux_return"]
    print(
        f"  aux Delta s_t: N={int(aux_s.nobs)} R2={aux_s.rsquared:.3f} "
        f"ln_hys_lag2={aux_s.params['ln_hys_lag2']:.3f} "
        f"spread_lag2={aux_s.params['spread_lag2']:.3f}"
    )
    print(
        f"  aux r_t^SP:    N={int(aux_r.nobs)} R2={aux_r.rsquared:.3f} "
        f"ln_pe10_lag2={aux_r.params['ln_pe10_lag2']:.3f}"
    )
    print(f"  -> {out.name}")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    df = build_panel()
    emit_table_2(run_table_2(df, REP_START, REP_END), REP_START, REP_END, "replication")
    emit_table_2(run_table_2(df, REP_START, EXT_END), REP_START, EXT_END, "extended")


if __name__ == "__main__":
    main()
