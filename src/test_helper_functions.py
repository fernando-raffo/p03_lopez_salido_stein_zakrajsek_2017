"""
Unit tests for `helper_functions`, the growth-rate and return
transformation helpers shared by the Table I/II regression scripts.
"""

import numpy as np
import pandas as pd
import pytest

import helper_functions as hf


def test_year_over_year_growth_matches_log_diff():
    s = pd.Series([100.0, 110.0, 121.0], index=[2010, 2011, 2012])

    result = hf.year_over_year_growth(s)

    assert np.isnan(result.loc[2010])
    assert result.loc[2011] == pytest.approx(100.0 * (np.log(110.0) - np.log(100.0)))
    assert result.loc[2012] == pytest.approx(100.0 * (np.log(121.0) - np.log(110.0)))


def test_year_over_year_growth_sorts_by_index_first():
    s = pd.Series([121.0, 100.0, 110.0], index=[2012, 2010, 2011])

    result = hf.year_over_year_growth(s)

    assert result.loc[2011] == pytest.approx(100.0 * (np.log(110.0) - np.log(100.0)))
    assert result.loc[2012] == pytest.approx(100.0 * (np.log(121.0) - np.log(110.0)))


def test_log_total_return_matches_formula():
    price = pd.Series([100.0, 110.0, 105.0], index=[2010, 2011, 2012])
    income = pd.Series([2.0, 2.5, 3.0], index=[2010, 2011, 2012])

    result = hf.log_total_return(price, income)

    assert np.isnan(result.loc[2010])
    assert result.loc[2011] == pytest.approx(100.0 * np.log(112.5 / 100.0))
    assert result.loc[2012] == pytest.approx(100.0 * np.log(108.0 / 110.0))


def test_log_total_return_sorts_by_index_first():
    price = pd.Series([105.0, 100.0, 110.0], index=[2012, 2010, 2011])
    income = pd.Series([3.0, 2.0, 2.5], index=[2012, 2010, 2011])

    result = hf.log_total_return(price, income)

    assert result.loc[2011] == pytest.approx(100.0 * np.log(112.5 / 100.0))
    assert result.loc[2012] == pytest.approx(100.0 * np.log(108.0 / 110.0))


def test_to_percent_scales_by_100():
    s = pd.Series([0.05, -0.02, 0.0], index=[2010, 2011, 2012])

    result = hf.to_percent(s)

    assert result.tolist() == pytest.approx([5.0, -2.0, 0.0])
