"""
Replicate Table I of Lopez-Salido, Stein & Zakrajsek (2017).

Three columns matching the published QJE table, Newey-West (HAC) errors:
    (1) credit only, (2) equity only, (3) both + rate/inflation controls.
Published QJE (1929-2015): col1 d_s=-1.997, col2 r^SP=0.081, col3 d_s=-2.061.

Two windows (1929-2015, 1929-2025) are emitted for both the Baa- and
Aaa-Treasury spread variants.
Shiller inputs: sp500_price + dividend, to build the annual S&P 500 total log
return r_t = 100*log((P_t + D_t)/P_{t-1}). All other controls come from FRED.
Note: Shiller data ends 2023, so the equity columns of the extended window
effectively stop at 2023 (post-2023 rows drop out on the join).
"""

from pathlib import Path

import numpy as np
import pandas as pd
import statsmodels.api as sm

from helper_functions import to_percent, year_over_year_growth
from latex_format import (
    coef_se_rows,
    pretty_label,
    regression_table_df,
    style_table,
    two_row_header,
)
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


def build_panel(spread_col="BAA_Treasury_spread"):
    df = pd.read_parquet(PROCESSED_DATA_DIR / "fred_final_series_annual.parquet")
    df["gdp_pc_growth"] = year_over_year_growth(df["GDP_per_capita"])
    df["d_credit_spread"] = df[spread_col].diff()
    df["d_treasury_3mo"] = df["Treasury_3mo"].diff()
    df["d_treasury_10yr"] = df["Treasury_10yr"].diff()
    df["CPI_inflation"] = to_percent(df["CPI_inflation"])
    df = df.join(load_sp_return(), how="left")
    df["dy_next"] = df["gdp_pc_growth"].shift(-1)
    return df


def run_regression(df, regressors, start, end):
    d = df.loc[start:end, ["dy_next"] + regressors].dropna()
    return sm.OLS(d["dy_next"], sm.add_constant(d[regressors])).fit(
        cov_type="HAC", cov_kwds={"maxlags": newey_west_lags(len(d))}
    )


# Row order and LaTeX labels for the main table. r^{SP}_t keeps the paper's
# own (contemporaneous-looking) subscript rather than t-1, matching Table I.
MAIN_ROWS = [
    ("d_credit_spread", r"$\Delta s_{t-1}$"),
    ("sp_return", r"$r_t^{SP}$"),
    ("gdp_pc_growth", r"$\Delta y_{t-1}$"),
    ("d_treasury_3mo", r"$\Delta i_{t-1}^{(3m)}$"),
    ("d_treasury_10yr", r"$\Delta i_{t-1}^{(10y)}$"),
    ("CPI_inflation", r"$\pi_{t-1}$"),
]

# Rows shown in the "Standardized effect" panel.
STD_ROWS = [
    ("d_credit_spread", r"$\Delta s_{t-1}$"),
    ("sp_return", r"$r_t^{SP}$"),
]


def standardized_effect(res, window, regressors, var):
    """Coefficient rescaled by StdDev(regressor)/StdDev(dep. var) over the
    regression's own estimation sample, or "---" if var isn't in the spec."""
    if var not in res.params.index:
        return "---"
    d = window[["dy_next"] + regressors].dropna()
    effect = res.params[var] * d[var].std() / d["dy_next"].std()
    return f"{effect:.3f}"


def _column_specs():
    """Regressors for each of Table I's three columns, keyed by its column
    label -- shared by `emit` (LaTeX) and `pretty_table_1` (notebook
    display) so the two can't drift apart."""
    base = ["gdp_pc_growth"]
    return {
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


def emit(df, start, end, label, spread_col="BAA_Treasury_spread"):
    specs = _column_specs()
    window = df.loc[start:end]
    results = {col: run_regression(df, regs, start, end) for col, regs in specs.items()}
    res_list = list(results.values())
    ncols = len(specs)

    lines = [f"\\begin{{tabular}}{{l{'c' * ncols}}}", "\\toprule"]
    lines.append(
        two_row_header(ncols, r"Dependent variable: $\Delta y_t$", "Regressors")
    )
    lines.append("\\midrule")
    lines.extend(coef_se_rows(res_list, MAIN_ROWS))
    lines.append("\\midrule")
    r2_vals = " & ".join(f"{res.rsquared_adj:.3f}" for res in res_list)
    lines.append(f"$\\bar R^2$ & {r2_vals} \\\\")
    lines.append("\\addlinespace[4pt]")
    lines.append(
        f"\\multicolumn{{{ncols + 1}}}{{l}}{{\\textit{{Standardized effect on $\\Delta y_t$}}}} \\\\"
    )
    for var, row_label in STD_ROWS:
        vals = " & ".join(
            standardized_effect(results[col], window, specs[col], var) for col in specs
        )
        lines.append(f"{row_label} & {vals} \\\\")
    lines.append("\\bottomrule")
    lines.append("\\end{tabular}")

    path = OUTPUT_DIR / f"table_1_{label}.tex"
    comment = f"% Table I replication ({label}): {start}-{end}, credit spread = {spread_col}\n"
    path.write_text(comment + "\n".join(lines) + "\n")

    r1, r3 = results["(1)"], results["(3)"]
    print(
        f"{label}: col1 d_s={r1.params['d_credit_spread']:.3f}, "
        f"col2 r_sp={results['(2)'].params['sp_return']:.3f}, "
        f"col3 d_s={r3.params['d_credit_spread']:.3f}/r_sp={r3.params['sp_return']:.3f} "
        f"-> {path.name}"
    )


def pretty_table_1(df, start, end, spread_col="BAA_Treasury_spread"):
    """The same table `emit` writes to `_output/table_1_*.tex`, as a styled
    `DataFrame` for direct notebook display via `display(...)` -- every row
    and column, not a hand-picked subset, built with the same `coef_cell`
    formatting the `.tex` output uses.
    """
    specs = _column_specs()
    window = df.loc[start:end]
    results = [run_regression(df, regs, start, end) for regs in specs.values()]

    main_df = regression_table_df(
        results, MAIN_ROWS, "Dependent variable: Δy<sub>t</sub>"
    )
    footer = [("Adj. R²", [f"{res.rsquared_adj:.3f}" for res in results])]
    footer.append(("Standardized effect on Δy<sub>t</sub>", [""] * len(results)))
    for var, label in STD_ROWS:
        footer.append(
            (
                pretty_label(label),
                [
                    standardized_effect(res, window, regs, var)
                    for res, regs in zip(results, specs.values())
                ],
            )
        )

    caption = f"Table I -- {start}-{end}, credit spread = {spread_col}"
    return style_table(main_df, footer=footer, caption=caption)


# (label tag, spread column) pairs. The Baa tag is empty so its filenames
# match the original, unsuffixed `table_1_*.tex` names.
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
            emit(df, start, end, label, spread_col=spread_col)


if __name__ == "__main__":
    main()
