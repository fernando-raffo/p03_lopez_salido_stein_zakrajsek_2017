"""
Shared LaTeX formatting helpers.

Covers the Table I / Table II regression tables' layout (a two-row header
"Dependent variable: ..." spanning the numeric columns, then column numbers;
a coefficient row followed by a parenthesized-s.e. row for each regressor;
significance stars; an em-dash for regressors omitted from a given column),
plus `latex_escape` for any plain-text label/unit that ends up as a table
cell in a `.tex` file emitted by this repo -- so callers can write specs as
plain text (e.g. "S&P", "$tn") instead of pre-escaped LaTeX source. Hand-
authored LaTeX source strings (e.g. row labels like r"$\\Delta s_{t-1}$")
should NOT be run through `latex_escape` -- it is only for data/labels that
are plain text.

Also covers `regression_table_df`/`style_table`, a notebook-facing display
path for the same tables: a rendered (not raw-LaTeX-source) `pandas` table
built from the same `coef_cell` formatting the `.tex` output uses, so the
two can't drift apart -- only the presentation differs.
"""

import numpy as np
import pandas as pd

STAR_THRESHOLDS = ((0.01, "***"), (0.05, "**"), (0.10, "*"))

_LATEX_SPECIAL_CHARS = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
    "\\": r"\textbackslash{}",
}


def latex_escape(text):
    """Escape LaTeX special characters in plain text bound for a table cell."""
    return "".join(_LATEX_SPECIAL_CHARS.get(ch, ch) for ch in str(text))


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

    If `res` carries `.bse_joint`/`.pvalues_joint` attributes (as
    `replicate_table_2.run_table_2`'s second-step results do, to correct for
    generated-regressor sampling uncertainty), those are used in place of
    the plain `.bse`/`.pvalues`.
    """
    if name not in res.params.index:
        return "---", ""
    bse = getattr(res, "bse_joint", res.bse)
    pvalues = getattr(res, "pvalues_joint", res.pvalues)
    coef = f"{res.params[name] * scale:.3f}{stars(pvalues[name])}"
    se = f"({bse[name] * scale:.3f})"
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


# ---------------------------------------------------------------------------
# Notebook display: a rendered `pandas` table instead of raw LaTeX source.
# ---------------------------------------------------------------------------

# Plain-HTML rendering of every hand-authored LaTeX row label used by
# `replicate_table_1.MAIN_ROWS`/`STD_ROWS` and `replicate_table_2._MAIN_ROWS`/
# `_AUX_ROWS` -- the closed set of regressor labels these tables ever show.
_PRETTY_LABELS = {
    r"$\Delta s_{t-1}$": "Δs<sub>t−1</sub>",
    r"$r_t^{SP}$": "r<sup>SP</sup><sub>t</sub>",
    r"$\Delta y_{t-1}$": "Δy<sub>t−1</sub>",
    r"$\Delta i_{t-1}^{(3m)}$": "Δi<sub>t−1</sub><sup>(3m)</sup>",
    r"$\Delta i_{t-1}^{(10y)}$": "Δi<sub>t−1</sub><sup>(10y)</sup>",
    r"$\pi_{t-1}$": "π<sub>t−1</sub>",
    r"$\Delta \hat s_t$": "Δŝ<sub>t</sub>",
    r"$\hat r_t^{SP}$": "r̂<sup>SP</sup><sub>t</sub>",
    r"$\ln \mathrm{HYS}_{t-2}$": "ln HYS<sub>t−2</sub>",
    r"$s_{t-2}$": "s<sub>t−2</sub>",
    r"$\ln[P/E10]_{t-2}$": "ln[P/E10]<sub>t−2</sub>",
}


def pretty_label(label):
    """Plain-HTML rendering of a hand-authored LaTeX row label (e.g.
    r"$\\Delta s_{t-1}$" -> "Δs<sub>t−1</sub>"), for notebook display.
    Falls back to the raw label, unchanged, for anything not in the lookup.
    """
    return _PRETTY_LABELS.get(label, label)


def regression_table_df(results, row_specs, dep_var_label):
    """A coefficient/s.e. table matching `coef_se_rows`'s LaTeX rows, as a
    plain `DataFrame` for notebook display -- every `row_specs` entry
    present in any column (not a hand-picked subset), via the same
    `coef_cell` formatting the `.tex` output uses.

    `dep_var_label`: a single string for one "Dependent variable: ..."
    header spanning every column (mirrors `two_row_header`), or a sequence
    of one label per result when each column has its own dependent
    variable (Table II's auxiliary regressions).
    """
    if isinstance(dep_var_label, str):
        columns = pd.MultiIndex.from_arrays(
            [
                [dep_var_label] * len(results),
                [f"({i})" for i in range(1, len(results) + 1)],
            ]
        )
    else:
        columns = pd.Index(list(dep_var_label))

    rows, index = [], []
    for spec in row_specs:
        var, label, *rest = spec
        scale = rest[0] if rest else 1.0
        coefs, ses = zip(*(coef_cell(res, var, scale) for res in results))
        if all(c == "---" for c in coefs):
            continue
        rows.append(list(coefs))
        index.append(pretty_label(label))
        rows.append(list(ses))
        index.append("")
    return pd.DataFrame(rows, index=index, columns=columns)


def style_table(df, footer=None, caption=None):
    """A booktabs-ish styled `Styler` for `regression_table_df`'s output,
    ready for `display()` in a notebook.

    `footer`: an ordered list of `(label, values)` rows appended below a
    rule (e.g. an R^2 row, or a standardized-effect block).
    """
    if footer:
        foot_df = pd.DataFrame(
            [values for _, values in footer],
            index=[label for label, _ in footer],
            columns=df.columns,
        )
        full = pd.concat([df, foot_df])
        rule_row = len(df)
    else:
        full = df
        rule_row = None

    table_styles = [
        {
            "selector": "caption",
            "props": "caption-side: top; text-align: left; "
            "font-weight: 600; padding-bottom: 6px;",
        },
        {"selector": "th, td", "props": "padding: 3px 14px; text-align: center;"},
        {
            "selector": "th.row_heading",
            "props": "text-align: left; font-weight: normal;",
        },
        {
            "selector": "thead tr:first-child th",
            "props": "border-top: 1.5px solid #333;",
        },
        {
            "selector": "thead tr:last-child th",
            "props": "border-bottom: 1px solid #333;",
        },
        {
            "selector": "tbody tr:last-child td, tbody tr:last-child th",
            "props": "border-bottom: 1.5px solid #333;",
        },
    ]
    if rule_row is not None:
        table_styles.append(
            {
                "selector": (
                    f"tbody tr:nth-child({rule_row + 1}) td, "
                    f"tbody tr:nth-child({rule_row + 1}) th"
                ),
                "props": "border-top: 1px solid #999;",
            }
        )

    styler = full.style
    if caption:
        styler = styler.set_caption(caption)
    return styler.set_table_styles(table_styles, overwrite=False)
