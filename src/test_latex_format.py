"""Unit tests for `latex_format`, the shared LaTeX table helpers.

These use lightweight fake results (a namespace with `params`, `bse`,
`pvalues` Series) so they need no statsmodels fit and no data.
"""

from types import SimpleNamespace

import pandas as pd

import latex_format as lf


def _fake_res(params, bse, pvalues):
    return SimpleNamespace(
        params=pd.Series(params),
        bse=pd.Series(bse),
        pvalues=pd.Series(pvalues),
    )


def test_stars_thresholds():
    assert lf.stars(0.001) == "***"
    assert lf.stars(0.03) == "**"
    assert lf.stars(0.08) == "*"
    assert lf.stars(0.50) == ""


def test_stars_handles_missing_pvalue():
    assert lf.stars(None) == ""
    assert lf.stars(float("nan")) == ""


def test_latex_escape_special_characters():
    assert lf.latex_escape("S&P") == r"S\&P"
    assert lf.latex_escape("100%") == r"100\%"
    assert lf.latex_escape("a_b") == r"a\_b"
    assert lf.latex_escape("$tn") == r"\$tn"


def test_coef_cell_present_regressor():
    res = _fake_res({"x": 2.0}, {"x": 0.5}, {"x": 0.001})
    coef, se = lf.coef_cell(res, "x")
    assert coef == "2.000***"
    assert se == "(0.500)"


def test_coef_cell_absent_regressor():
    res = _fake_res({"x": 2.0}, {"x": 0.5}, {"x": 0.001})
    assert lf.coef_cell(res, "not_there") == ("---", "")


def test_coef_cell_scale_affects_display_not_stars():
    # p = 0.5 -> no stars even though the scaled coefficient is small.
    res = _fake_res({"x": 100.0}, {"x": 10.0}, {"x": 0.5})
    coef, se = lf.coef_cell(res, "x", scale=0.01)
    assert coef == "1.000"
    assert se == "(0.100)"


def test_two_row_header_structure():
    header = lf.two_row_header(3, "Dependent variable: $y$")
    assert r"\multicolumn{3}{c}{Dependent variable: $y$}" in header
    assert r"\cmidrule(lr){2-4}" in header
    assert "(1) & (2) & (3)" in header


def test_two_row_header_corner_label():
    header = lf.two_row_header(2, "Dep", second_row_corner="Regressors")
    assert header.strip().splitlines()[-1].startswith("Regressors &")


def test_coef_se_rows_emits_coef_then_se_per_present_row():
    r1 = _fake_res({"a": 1.0}, {"a": 0.1}, {"a": 0.2})
    r2 = _fake_res({"a": 2.0}, {"a": 0.2}, {"a": 0.2})
    rows = lf.coef_se_rows([r1, r2], [("a", "A")])
    assert len(rows) == 2  # one coefficient line, one s.e. line
    assert rows[0].startswith("A & ")
    assert rows[1].strip().startswith("&")  # s.e. line has a blank label


def test_coef_se_rows_skips_rows_absent_from_every_column():
    r1 = _fake_res({"a": 1.0}, {"a": 0.1}, {"a": 0.2})
    r2 = _fake_res({"a": 2.0}, {"a": 0.2}, {"a": 0.2})
    rows = lf.coef_se_rows([r1, r2], [("a", "A"), ("missing", "M")])
    text = "\n".join(rows)
    assert "A &" in text
    assert "M" not in text  # entirely-absent regressor row is dropped


def test_coef_se_rows_applies_scale():
    r = _fake_res({"a": 100.0}, {"a": 10.0}, {"a": 0.5})
    rows = lf.coef_se_rows([r], [("a", "A", 0.01)])
    assert rows[0] == "A & 1.000 \\\\"
    assert rows[1] == " & (0.100) \\\\"


# --------------------------------------------------------------------------- #
# Notebook display helpers
# --------------------------------------------------------------------------- #
def test_pretty_label_known_and_unknown():
    assert lf.pretty_label(r"$\Delta s_{t-1}$") == "Δs<sub>t−1</sub>"
    assert lf.pretty_label("not a known label") == "not a known label"


def test_regression_table_df_shared_dependent_variable_header():
    r1 = _fake_res({"a": 1.0}, {"a": 0.1}, {"a": 0.2})
    r2 = _fake_res({"a": 2.0}, {"a": 0.2}, {"a": 0.2})
    df = lf.regression_table_df([r1, r2], [("a", "A")], "Dep. var: y")
    assert list(df.columns) == [("Dep. var: y", "(1)"), ("Dep. var: y", "(2)")]
    assert list(df.index) == ["A", ""]
    assert df.iloc[0].tolist() == ["1.000", "2.000"]
    assert df.iloc[1].tolist() == ["(0.100)", "(0.200)"]


def test_regression_table_df_per_column_dependent_variables():
    r1 = _fake_res({"a": 1.0}, {"a": 0.1}, {"a": 0.2})
    r2 = _fake_res({"b": 2.0}, {"b": 0.2}, {"b": 0.2})
    df = lf.regression_table_df([r1, r2], [("a", "A"), ("b", "B")], ["y1", "y2"])
    assert list(df.columns) == ["y1", "y2"]
    # Both rows are kept: each is present in at least one column.
    assert list(df.index) == ["A", "", "B", ""]


def test_regression_table_df_drops_rows_absent_everywhere():
    r1 = _fake_res({"a": 1.0}, {"a": 0.1}, {"a": 0.2})
    df = lf.regression_table_df([r1], [("a", "A"), ("missing", "M")], "y")
    assert "M" not in df.index


def test_style_table_appends_footer_rows():
    r1 = _fake_res({"a": 1.0}, {"a": 0.1}, {"a": 0.2})
    df = lf.regression_table_df([r1], [("a", "A")], "y")
    styled = lf.style_table(df, footer=[("R2", ["0.500"])], caption="A table")
    full = styled.data
    assert list(full.index) == ["A", "", "R2"]
    assert full.iloc[2].tolist() == ["0.500"]
