"""
Shared LaTeX formatting helpers for the Table I / Table II regression tables,
built to reproduce the published QJE layout: a two-row header ("Dependent
variable: ..." spanning the numeric columns, then column numbers), a
coefficient row followed by a parenthesized-s.e. row for each regressor,
significance stars on coefficients, and an em-dash for regressors omitted
from a given column.
"""

import numpy as np

STAR_THRESHOLDS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))


def stars(pvalue):
    """Significance stars for a p-value: *p<.10, **p<.05, ***p<.01."""
    if pvalue is None or np.isnan(pvalue):
        return ""
    for threshold, mark in STAR_THRESHOLDS:
        if pvalue < threshold:
            return mark
    return ""


def coef_cell(res, name, scale=1.0):
    """(coefficient string with stars, s.e. string) for a regressor in a
    fitted statsmodels result, or ("---", "") if the regressor isn't in the
    specification.

    `scale` rescales the displayed coefficient and s.e. only (stars are
    computed from the unscaled p-value); use it to match a paper's own
    display convention for a regressor, e.g. LSZ report their HYS coefficient
    "multiplied by 100" relative to the raw fitted value.
    """
    if name not in res.params.index:
        return "---", ""
    coef = f"{res.params[name] * scale:.3f}{stars(res.pvalues[name])}"
    se = f"({res.bse[name] * scale:.3f})"
    return coef, se


def two_row_header(ncols, dep_var_label, second_row_corner=""):
    """Two-line booktabs header: a multicolumn 'Dependent variable: ...'
    row spanning the numeric columns (with a cmidrule under it), then a row
    of column numbers '(1)', '(2)', ...."""
    row1 = f" & \\multicolumn{{{ncols}}}{{c}}{{{dep_var_label}}} \\\\"
    rule = f"\\cmidrule(lr){{2-{ncols + 1}}}"
    numbers = " & ".join(f"({i})" for i in range(1, ncols + 1))
    row2 = f"{second_row_corner} & {numbers} \\\\"
    return "\n".join([row1, rule, row2])


def coef_se_rows(results, row_specs):
    """A coefficient line and a (blank-label) s.e. line for each row spec in
    row_specs, across a list of fitted results (one per column). Rows where
    every column omits the regressor are skipped.

    Each row spec is (varname, label) or (varname, label, scale); see
    `coef_cell` for what `scale` does. `scale` defaults to 1.0.
    """
    lines = []
    for spec in row_specs:
        var, label, *rest = spec
        scale = rest[0] if rest else 1.0
        coefs, ses = zip(*(coef_cell(res, var, scale) for res in results))
        if all(c == "---" for c in coefs):
            continue
        lines.append(label + " & " + " & ".join(coefs) + " \\\\")
        lines.append(" & " + " & ".join(ses) + " \\\\")
    return lines
