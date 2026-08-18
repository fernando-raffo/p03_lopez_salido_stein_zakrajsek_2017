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

Estimation method: joint NLLS, footnote 12
-------------------------------------------
The paper estimates equations (2)-(4) *jointly* by nonlinear least squares
(NLLS) "to take into account the generated-regressor nature of the expected
returns" (p. 1388), with inference based on "a heteroskedasticity- and
autocorrelation-consistent asymptotic covariance matrix computed according
to Newey and West (1987), using the automatic lag selection method of Newey
and West (1994)" (footnote 12).

Because the system is block-recursive -- the auxiliary (first-step)
equations don't involve the second-step coefficients, so their normal
equations pin down theta1 (spread) and theta2 (return) on their own -- the
NLLS *point* estimates coincide with the simple two-step "plug-in" OLS
procedure: fit each auxiliary regression by OLS, substitute its fitted
values into the growth regression, and fit that by OLS too. This script
still does exactly that for the point estimates (`_fit_ols_hac` below).

What plug-in OLS gets wrong is the *second-step standard errors*: treating
`d_s_hat`/`r_sp_hat` as if they were data, rather than estimates with their
own sampling variance, understates the second step's coefficient
uncertainty. `_joint_step2_inference` corrects this by treating the whole
system as one exactly identified GMM/M-estimation problem, i.e. the
first-order conditions of joint NLLS:

    g1_t(theta1)              = z1,t-2 * (Delta s_t - theta1'z1,t-2)
    g2_t(theta2)              = z2,t-2 * (r_t^SP - theta2'z2,t-2)
    g3_t(theta1, theta2, psi) = w_t     * (Delta y_t - w_t(theta1,theta2)'psi)

stacked into m_t(theta1, theta2, psi) = [g1_t; g2_t; g3_t] (only the blocks
a given column actually uses), where w_t is that column's second-step
regressor vector with `d_s_hat`/`r_sp_hat` written as theta1'z1,t-2 /
theta2'z2,t-2 rather than plugged-in numbers. Solving Sum_t m_t = 0
reproduces the plug-in OLS estimates exactly (each theta solves its own
normal equations regardless of psi; psi then solves its own normal
equations given theta) -- confirming NLLS and plug-in OLS point estimates
agree -- but the asymptotic covariance of the stacked M-estimator,

    Avar(theta1, theta2, psi) = G^-1 S (G^-1)',

with G = d(Sum_t m_t)/d(params) (a numerically differentiated Jacobian,
block lower-triangular since g3 depends on theta1/theta2 but not vice versa)
and S the Newey-West HAC "meat" matrix of m_t (same automatic-lag rule as
every other regression in this repo), correctly propagates the first-step
sampling uncertainty into the psi (= second-step coefficient) block. This is
exactly the classic Murphy and Topel (1985)/generated-regressors correction,
and it is what "estimated jointly by NLLS" buys over plug-in OLS: the point
estimates are unchanged, but the reported standard errors are larger and
match the paper's methodology rather than a two-step approximation to it.

The auxiliary-regression coefficients' own standard errors are *unaffected*
by this correction (their block of `Avar` collapses back to the plain
single-equation HAC covariance, since G is block lower-triangular), so
`_fit_ols_hac`'s HAC standard errors for `aux_spread`/`aux_return` already
match the joint-NLLS ones and are left as is.

Because `fred_final_series_annual.parquet` is trimmed to start in 1929 (the
replication start year), `s_{t-2}` is unavailable for 1929-1930, so the
effective estimation sample used here begins in 1931 rather than 1929.

Data: all series come from the annual parquet files in
`_data/processed_data`:
    - fred_final_series_annual.parquet: GDP_per_capita, CPI_inflation,
      BAA_Treasury_spread / AAA_Treasury_spread, Treasury_10yr, Treasury_3mo
    - greenwood_hanson_hys.parquet: ln_hy_share (ln HYS)
    - shiller_data_annual.parquet: sp500_price, dividend, ln_pe10

`build_panel`'s `spread_col` argument picks which of the two credit spreads
feeds `s_t`; `main()` emits one full set of tables for each, the Aaa set
suffixed with "aaa" (e.g. table_2_aaa_replication.tex).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm
from scipy import stats

from helper_functions import log_total_return, to_percent, year_over_year_growth
from latex_format import coef_se_rows, regression_table_df, style_table, two_row_header
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

# Twice-lagged predictors of the two auxiliary (first-step) regressions,
# equations (2)-(3) of the paper.
AUX_SPREAD_VARS = ["ln_hys_lag2", "spread_lag2"]
AUX_RETURN_VARS = ["ln_pe10_lag2"]

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
#
# `ln_pe10_lag2` carries a 0.01 display scale to match the paper's own
# convention. LSZ fit Delta y_t, Delta s_t and r_t^SP as decimal (fractional)
# growth/return rates and only describe their magnitudes "as a percentage"
# in prose; this repo instead computes those series in percentage-point
# units throughout (see `helper_functions.year_over_year_growth` and
# `log_total_return`). That 100x rescaling is invisible in every other cell
# of Table II because it cancels whenever it appears on both sides of a
# regression (e.g. Delta y_t on Delta y_{t-1}, or the second-step Delta y_t
# on r_sp_hat, itself a fitted value of the rescaled r_t^SP). It does *not*
# cancel here, because ln[P/E10] is a dimensionless log-ratio that is never
# itself rescaled: the fitted coefficient on ln_pe10_lag2 comes out ~100x
# the paper's published -0.134 (confirmed against the authors' 2015 FEDS
# working-paper draft, which reports an analogous -0.136 for log[P/E]).
# Scaling only this row's *display* by 0.01 reproduces the paper's own
# convention of showing a rescaled coefficient for one specific auxiliary
# regressor (they multiply their ln HYS row by 100; we divide this one by
# 100) without touching the fitted values used anywhere downstream.
_AUX_ROWS = [
    ("ln_hys_lag2", r"$\ln \mathrm{HYS}_{t-2}$"),
    ("spread_lag2", r"$s_{t-2}$"),
    ("ln_pe10_lag2", r"$\ln[P/E10]_{t-2}$", 0.01),
]


def newey_west_lags(n_obs):
    """Newey-West (1994) automatic bandwidth rule of thumb."""
    return max(int(np.floor(4 * (n_obs / 100.0) ** (2 / 9))), 1)


def build_panel(spread_col="BAA_Treasury_spread"):
    """Assemble the annual panel of series needed for Table II.

    Parameters
    ----------
    spread_col : str, default "BAA_Treasury_spread"
        Column of `fred_final_series_annual.parquet` to use as the credit
        spread `s_t`, e.g. "BAA_Treasury_spread" or "AAA_Treasury_spread".
    """
    fred = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
    hys = pd.read_parquet(PROCESSED_DATA_DIR / "greenwood_hanson_hys.parquet")
    shiller = pd.read_parquet(PROCESSED_DATA_DIR / "shiller_data_annual.parquet")

    shiller = shiller.copy()
    shiller.index = shiller.index.year
    shiller.index.name = "year"

    df = fred.copy()
    df["dy"] = year_over_year_growth(df["GDP_per_capita"])
    df["d_spread"] = df[spread_col].diff()
    df["d_3mo"] = df["Treasury_3mo"].diff()
    df["d_10yr"] = df["Treasury_10yr"].diff()
    df["inflation_pct"] = to_percent(df["CPI_inflation"])

    df["ln_hys"] = hys["ln_hy_share"]
    df["sp_return"] = log_total_return(shiller["sp500_price"], shiller["dividend"])
    df["ln_pe10"] = shiller["ln_pe10"]

    # Auxiliary-regression predictors, lagged two years per the paper.
    df["ln_hys_lag2"] = df["ln_hys"].shift(2)
    df["spread_lag2"] = df[spread_col].shift(2)
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


def _hac_meat(scores, nlags):
    """Newey-West (1987) HAC "meat" matrix (Bartlett kernel, sum
    convention) of a T x q score/moment array, i.e. Gamma_0 + sum_{l=1}^L
    w_l (Gamma_l + Gamma_l'), the same object statsmodels forms internally
    for `cov_type="HAC"` -- just generalized here to an arbitrary stacked
    moment vector instead of a single equation's X*resid."""
    S = scores.T @ scores
    for lag in range(1, nlags + 1):
        w = 1.0 - lag / (nlags + 1)
        gamma = scores[lag:].T @ scores[:-lag]
        S += w * (gamma + gamma.T)
    return S


def _numeric_jacobian(func, x0, rel_step=1e-6):
    """Central-difference Jacobian of `func: R^q -> R^q` at `x0`."""
    q = x0.shape[0]
    jac = np.empty((q, q))
    for j in range(q):
        step = rel_step * max(abs(x0[j]), 1.0)
        dx = np.zeros(q)
        dx[j] = step
        jac[:, j] = (func(x0 + dx) - func(x0 - dx)) / (2 * step)
    return jac


def _joint_step2_inference(window, res_aux_spread, res_aux_return, res_col, regressors):
    """Generated-regressor-corrected (co)variance of a second-step column's
    coefficients, per the paper's joint-NLLS system (footnote 12; see the
    module docstring for the derivation). Point estimates are unaffected --
    only `res_col`'s standard errors are being redone here -- so this takes
    the already-fit auxiliary and second-step OLS results and returns
    `(bse, pvalues)` `pandas.Series` indexed like `res_col.params`.
    """
    spread_used = "d_s_hat" in regressors
    return_used = "r_sp_hat" in regressors
    other = [r for r in regressors if r not in ("d_s_hat", "r_sp_hat")]

    raw_cols = list(other) + ["dy"]
    if spread_used:
        raw_cols += ["d_spread"] + AUX_SPREAD_VARS
    if return_used:
        raw_cols += ["sp_return"] + AUX_RETURN_VARS
    joint = window[raw_cols].dropna()
    n_obs = len(joint)

    theta1 = res_aux_spread.params.to_numpy() if spread_used else None
    theta2 = res_aux_return.params.to_numpy() if return_used else None
    if spread_used:
        Z1 = sm.add_constant(joint[AUX_SPREAD_VARS], has_constant="add").to_numpy()
        d_spread = joint["d_spread"].to_numpy()
    if return_used:
        Z2 = sm.add_constant(joint[AUX_RETURN_VARS], has_constant="add").to_numpy()
        sp_return = joint["sp_return"].to_numpy()

    b_names = res_col.params.index.tolist()
    b_hat = res_col.params.to_numpy()
    dy = joint["dy"].to_numpy()
    k1 = theta1.shape[0] if spread_used else 0
    k2 = theta2.shape[0] if return_used else 0
    kb = len(b_names)

    def build_Xc(theta1, theta2):
        cols = []
        for name in b_names:
            if name == "d_s_hat":
                cols.append(Z1 @ theta1)
            elif name == "r_sp_hat":
                cols.append(Z2 @ theta2)
            elif name == "const":
                cols.append(np.ones(n_obs))
            else:
                cols.append(joint[name].to_numpy())
        return np.column_stack(cols)

    def moments(psi):
        i = 0
        parts = []
        theta1 = psi[i : i + k1] if spread_used else None
        i += k1
        theta2 = psi[i : i + k2] if return_used else None
        i += k2
        b = psi[i : i + kb]
        if spread_used:
            u1 = d_spread - Z1 @ theta1
            parts.append(Z1 * u1[:, None])
        if return_used:
            u2 = sp_return - Z2 @ theta2
            parts.append(Z2 * u2[:, None])
        Xc = build_Xc(theta1, theta2)
        u3 = dy - Xc @ b
        parts.append(Xc * u3[:, None])
        return np.hstack(parts)

    psi0 = np.concatenate([a for a in (theta1, theta2, b_hat) if a is not None])
    m0 = moments(psi0)
    nlags = newey_west_lags(n_obs)
    S = _hac_meat(m0, nlags)
    G = _numeric_jacobian(lambda p: moments(p).sum(axis=0), psi0)
    G_inv = np.linalg.inv(G)
    avar = G_inv @ S @ G_inv.T
    avar_b = avar[-kb:, -kb:]

    bse = pd.Series(np.sqrt(np.diag(avar_b)), index=b_names)
    df_resid = n_obs - kb
    tvalues = pd.Series(b_hat, index=b_names) / bse
    pvalues = pd.Series(
        2 * stats.t.sf(np.abs(tvalues.to_numpy()), df_resid), index=b_names
    )
    return bse, pvalues


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
        `statsmodels` OLS results object. Each "col*" result additionally
        carries `.bse_joint`/`.pvalues_joint` attributes -- the second-step
        standard errors/p-values corrected for generated-regressor sampling
        uncertainty per the paper's joint-NLLS system (see
        `_joint_step2_inference`); `latex_format.coef_cell` prefers these
        over the plug-in `.bse`/`.pvalues` when present. `.params`, `.bse`,
        `.pvalues`, `.rsquared`, `.nobs`, and all other statsmodels
        internals (e.g. `.get_influence()`, used by `replicate_figure_2.py`)
        are untouched plug-in-OLS values.
    """
    window = df.loc[start:end]

    d = window[["d_spread"] + AUX_SPREAD_VARS].dropna()
    res_aux_spread = _fit_ols_hac(d["d_spread"], d[AUX_SPREAD_VARS])

    d = window[["sp_return"] + AUX_RETURN_VARS].dropna()
    res_aux_return = _fit_ols_hac(d["sp_return"], d[AUX_RETURN_VARS])

    reg = window.assign(
        d_s_hat=res_aux_spread.fittedvalues.reindex(window.index),
        r_sp_hat=res_aux_return.fittedvalues.reindex(window.index),
    )

    results = {"aux_spread": res_aux_spread, "aux_return": res_aux_return}
    for col, regressors in COLUMN_REGRESSORS.items():
        d = reg[["dy"] + regressors].dropna()
        res_col = _fit_ols_hac(d["dy"], d[regressors])
        res_col.bse_joint, res_col.pvalues_joint = _joint_step2_inference(
            window, res_aux_spread, res_aux_return, res_col, regressors
        )
        results[col] = res_col
    return results


def _main_table_lines(results):
    cols = ["col1", "col2", "col3", "col4"]
    res_list = [results[c] for c in cols]
    ncols = len(cols)

    lines = [f"\\begin{{tabular}}{{l{'c' * ncols}}}", "\\toprule"]
    lines.append(two_row_header(ncols, r"Dependent variable: $\Delta y_t$"))
    lines.append("\\midrule")
    lines.extend(coef_se_rows(res_list, _MAIN_ROWS))
    lines.append("\\midrule")
    r2_vals = " & ".join(f"{res.rsquared:.3f}" for res in res_list)
    lines.append(f"$R^2$ & {r2_vals} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return lines


def _aux_table_lines(results):
    res_s = results["aux_spread"]
    res_r = results["aux_return"]

    lines = [
        "\\begin{tabular}{lcc}",
        "\\toprule",
        r" & $\Delta s_t$ & $r_t^{SP}$ \\",
        "\\midrule",
    ]
    lines.extend(coef_se_rows([res_s, res_r], _AUX_ROWS))
    lines.append("\\midrule")
    lines.append(f"$R^2$ & {res_s.rsquared:.3f} & {res_r.rsquared:.3f} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")
    return lines


def _panel_header(title):
    """A bold, centered panel title, on its own line above a `tabular` --
    used to separate Table II's two sub-tables (they have different column
    counts, so they can't share one `tabular`) so they read as two clearly
    labeled panels of one table rather than an unlabeled, oddly-indented
    second block stacked under the first."""
    return [r"\begin{center}\textbf{%s}\end{center}" % title, r"\vspace{2pt}"]


def emit_table_2(results, start, end, label, spread_col="BAA_Treasury_spread"):
    lines = (
        _panel_header("Panel A: Second-step (growth) regressions")
        + _main_table_lines(results)
        + ["", r"\vspace{12pt}", ""]
        + _panel_header("Panel B: Auxiliary (first-step) regressions")
        + _aux_table_lines(results)
    )

    out = OUTPUT_DIR / f"table_2_{label}.tex"
    text = (
        f"% Table II replication ({label}): {start}-{end}, "
        f"credit spread = {spread_col}\n" + "\n".join(lines) + "\n"
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


def pretty_table_2(results, start, end, spread_col="BAA_Treasury_spread"):
    """The same two tables `emit_table_2` writes to `_output/table_2_*.tex`
    (the second-step growth regressions, then the auxiliary regressions),
    as styled `DataFrame`s for direct notebook display via `display(...)`
    -- every row and column, not a hand-picked subset, built with the same
    `coef_cell` formatting the `.tex` output uses.

    Returns `(main_styler, aux_styler)`.
    """
    main_res = [results[c] for c in ("col1", "col2", "col3", "col4")]
    main_df = regression_table_df(
        main_res, _MAIN_ROWS, "Dependent variable: Δy<sub>t</sub>"
    )
    main_footer = [("R²", [f"{res.rsquared:.3f}" for res in main_res])]
    main_caption = (
        f"Table II -- second-step (growth) regressions: "
        f"{start}-{end}, credit spread = {spread_col}"
    )
    main_styler = style_table(main_df, footer=main_footer, caption=main_caption)

    aux_res = [results["aux_spread"], results["aux_return"]]
    aux_df = regression_table_df(
        aux_res, _AUX_ROWS, ["Δs<sub>t</sub>", "r<sup>SP</sup><sub>t</sub>"]
    )
    aux_footer = [("R²", [f"{res.rsquared:.3f}" for res in aux_res])]
    aux_caption = (
        f"Table II -- auxiliary (first-step) regressions: "
        f"{start}-{end}, credit spread = {spread_col}"
    )
    aux_styler = style_table(aux_df, footer=aux_footer, caption=aux_caption)

    return main_styler, aux_styler


# (label tag, spread column) pairs. The Baa tag is empty so its filenames
# match the original, unsuffixed `table_2_*.tex` names.
SPREAD_VARIANTS = [
    ("", "BAA_Treasury_spread"),
    ("aaa", "AAA_Treasury_spread"),
]


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    for spread_tag, spread_col in SPREAD_VARIANTS:
        df = build_panel(spread_col=spread_col)
        for window_tag, start, end in (
            ("replication", REP_START, REP_END),
            ("extended", REP_START, EXT_END),
        ):
            label = "_".join(p for p in (spread_tag, window_tag) if p)
            emit_table_2(
                run_table_2(df, start, end), start, end, label, spread_col=spread_col
            )


if __name__ == "__main__":
    main()
