import numpy as np
import pandas as pd
import pytest

import process_fred_data_annual as annual
import process_fred_data_monthly as monthly


# ---------------------------------------------------------------------------
# process_fred_data_annual
# ---------------------------------------------------------------------------


def test_final_gdp_series_combines_annual_and_quarterly():
    df = pd.DataFrame(
        {
            "GDPCA": pd.Series([100.0], index=pd.to_datetime(["1940-01-01"])),
            "GDPC1": pd.Series(
                [90.0, 100.0, 110.0, 120.0],
                index=pd.to_datetime(
                    ["2010-01-01", "2010-04-01", "2010-07-01", "2010-10-01"]
                ),
            ),
        }
    )

    gdp = annual.final_gdp_series(df)

    assert gdp.loc[1940] == pytest.approx(100.0 * 1e9)
    assert gdp.loc[2010] == pytest.approx(105.0 * 1e9)
    assert gdp.index.name == "year"
    assert gdp.name == "GDP"


def test_final_population_series_annual_and_quarterly_branches():
    df = pd.DataFrame(
        {
            "POPH": pd.Series(
                [140.0, 141.0, 142.0, 143.0],
                index=pd.to_datetime(
                    ["1945-07-01", "1946-07-01", "1947-07-01", "1948-07-01"]
                ),
            ),
            "B230RC0Q173SBEA": pd.Series(
                [200.0, 201.0, 202.0, 203.0],
                index=pd.to_datetime(
                    ["2010-01-01", "2010-04-01", "2010-07-01", "2010-10-01"]
                ),
            ),
        }
    )

    pop = annual.final_population_series(df)

    # Post-threshold (quarterly) years are an exact average, rescaled to persons.
    assert pop.loc[2010] == pytest.approx((200.0 + 201.0 + 202.0 + 203.0) / 4 * 1000)

    # Pre-threshold (spline-interpolated) years should be present and roughly
    # track the nearly-linear input trend, without NaNs.
    assert {1945, 1946, 1947, 1948}.issubset(set(pop.index))
    assert pop.notna().all()
    assert pop.loc[1946] == pytest.approx(141.0, abs=1.0)


def test_calculate_gdp_per_capita():
    gdp = pd.Series({2010: 200.0, 2011: 220.0}, name="GDP")
    population = pd.Series({2010: 100.0, 2011: 110.0}, name="Population")

    result = annual.calculate_gdp_per_capita(gdp, population)

    assert result.loc[2010] == pytest.approx(2.0)
    assert result.loc[2011] == pytest.approx(2.0)
    assert result.name == "GDP_per_capita"


def test_final_cpi_series_is_december_log_change():
    df = pd.DataFrame(
        {"CPIAUCNS": [200.0, 204.0, 210.0]},
        index=pd.to_datetime(["2010-12-01", "2011-12-01", "2012-12-01"]),
    )

    cpi = annual.final_cpi_series(df)

    assert 2010 not in cpi.index
    assert cpi.loc[2011] == pytest.approx(np.log(204.0) - np.log(200.0))
    assert cpi.loc[2012] == pytest.approx(np.log(210.0) - np.log(204.0))


def test_final_baa_and_aaa_series_annual_take_december_value():
    df = pd.DataFrame(
        {
            "BAA": [5.0, 5.5, 5.7],
            "AAA": [4.0, 4.5, 4.7],
        },
        index=pd.to_datetime(["2010-06-01", "2010-12-01", "2011-12-01"]),
    )

    baa = annual.final_baa_series_annual(df)
    aaa = annual.final_aaa_series(df)

    assert list(baa.index) == [2010, 2011]
    assert baa.loc[2010] == pytest.approx(5.5)
    assert baa.loc[2011] == pytest.approx(5.7)
    assert aaa.loc[2010] == pytest.approx(4.5)
    assert aaa.loc[2011] == pytest.approx(4.7)


def test_final_10yr_treasury_series_annual_prefers_latest_column():
    df = pd.DataFrame(
        {
            "GS10": [3.5, np.nan, np.nan],
            "M1333BUSM156NNBR": [9.9, 4.0, np.nan],
            "M1333AUSM156NNBR": [9.9, 9.9, 2.0],
        },
        index=pd.to_datetime(["2010-12-01", "2011-12-01", "2012-12-01"]),
    )

    treasury = annual.final_10yr_treasury_series_annual(df)

    assert treasury.loc[2010] == pytest.approx(3.5)
    assert treasury.loc[2011] == pytest.approx(4.0)
    assert treasury.loc[2012] == pytest.approx(2.0)


def test_final_3mo_treasury_series_prefers_latest_column():
    df = pd.DataFrame(
        {
            "TB3MS": [np.nan, 2.0],
            "M1329AUSM193NNBR": [1.5, 9.9],
        },
        index=pd.to_datetime(["2010-12-01", "2011-12-01"]),
    )

    def bond_equivalent(discount_pct):
        d = discount_pct / 100
        return 100 * (365 * d) / (360 - 91 * d)

    result = annual.final_3mo_treasury_series(df)

    assert result.loc[2010] == pytest.approx(bond_equivalent(1.5))
    assert result.loc[2011] == pytest.approx(bond_equivalent(2.0))


def test_clean_fred_data_annual_produces_expected_columns():
    df = pd.DataFrame(
        {
            "GDPCA": pd.Series([100.0], index=pd.to_datetime(["1940-01-01"])),
            "GDPC1": pd.Series(
                [90.0, 100.0, 110.0, 120.0],
                index=pd.to_datetime(
                    ["2010-01-01", "2010-04-01", "2010-07-01", "2010-10-01"]
                ),
            ),
            "POPH": pd.Series(
                [140.0, 141.0, 142.0, 143.0],
                index=pd.to_datetime(
                    ["1945-07-01", "1946-07-01", "1947-07-01", "1948-07-01"]
                ),
            ),
            "B230RC0Q173SBEA": pd.Series(
                [200.0, 201.0, 202.0, 203.0],
                index=pd.to_datetime(
                    ["2010-01-01", "2010-04-01", "2010-07-01", "2010-10-01"]
                ),
            ),
            "CPIAUCNS": pd.Series(
                [200.0, 204.0], index=pd.to_datetime(["2009-12-01", "2010-12-01"])
            ),
            "BAA": pd.Series(
                [5.0, 5.5], index=pd.to_datetime(["2009-12-01", "2010-12-01"])
            ),
            "AAA": pd.Series(
                [4.0, 4.5], index=pd.to_datetime(["2009-12-01", "2010-12-01"])
            ),
            "GS10": pd.Series(
                [3.0, 3.5], index=pd.to_datetime(["2009-12-01", "2010-12-01"])
            ),
            "TB3MS": pd.Series(
                [1.0, 1.5], index=pd.to_datetime(["2009-12-01", "2010-12-01"])
            ),
            "M1333BUSM156NNBR": pd.Series(dtype=float, index=pd.DatetimeIndex([])),
            "M1333AUSM156NNBR": pd.Series(dtype=float, index=pd.DatetimeIndex([])),
            "M1329AUSM193NNBR": pd.Series(dtype=float, index=pd.DatetimeIndex([])),
        }
    )

    cleaned = annual.clean_fred_data_annual(df)

    expected_columns = {
        "GDP",
        "Population",
        "GDP_per_capita",
        "CPI_inflation",
        "BAA",
        "AAA",
        "Treasury_10yr",
        "Treasury_3mo",
        "BAA_Treasury_spread",
        "AAA_Treasury_spread",
    }
    assert expected_columns == set(cleaned.columns)
    assert cleaned.loc[2010, "GDP_per_capita"] == pytest.approx(
        cleaned.loc[2010, "GDP"] / cleaned.loc[2010, "Population"]
    )
    assert cleaned.loc[2010, "BAA_Treasury_spread"] == pytest.approx(
        cleaned.loc[2010, "BAA"] - cleaned.loc[2010, "Treasury_10yr"]
    )
    assert cleaned.loc[2010, "AAA_Treasury_spread"] == pytest.approx(
        cleaned.loc[2010, "AAA"] - cleaned.loc[2010, "Treasury_10yr"]
    )


def test_calculate_baa_treasury_spread_annual():
    years = [2009, 2010]
    baa = pd.Series([5.0, 5.5], index=years)
    treasury = pd.Series([3.0, 3.2], index=years)

    spread = annual.calculate_baa_treasury_spread(baa, treasury)

    assert spread.loc[2009] == pytest.approx(2.0)
    assert spread.loc[2010] == pytest.approx(2.3)
    assert spread.name == "BAA_Treasury_spread"


def test_calculate_aaa_treasury_spread_annual():
    years = [2009, 2010]
    aaa = pd.Series([4.0, 4.5], index=years)
    treasury = pd.Series([3.0, 3.2], index=years)

    spread = annual.calculate_aaa_treasury_spread(aaa, treasury)

    assert spread.loc[2009] == pytest.approx(1.0)
    assert spread.loc[2010] == pytest.approx(1.3)
    assert spread.name == "AAA_Treasury_spread"


def test_save_data_readme_annual(tmp_path):
    df = pd.DataFrame({"GDP": [1.0], "MADE_UP_COLUMN": [2.0]}, index=[2010])

    file_path = annual.save_data_readme(df, data_dir=tmp_path)

    assert file_path == tmp_path / "fred_final_series_annual_readme.md"
    text = file_path.read_text()
    assert "GDP" in text
    assert "MADE_UP_COLUMN" in text
    assert "Unknown series" in text


# ---------------------------------------------------------------------------
# process_fred_data_monthly
# ---------------------------------------------------------------------------


def test_final_recession_indicator_series():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01", "2010-03-01"])
    df = pd.DataFrame({"USREC": [0, 1, 0]}, index=dates)

    result = monthly.final_recession_indicator_series(df)

    assert list(result.values) == [0, 1, 0]
    assert result.name == "hist_recession_indicator"


def test_final_10yr_treasury_series_monthly_prefers_latest_column():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01"])
    df = pd.DataFrame(
        {
            "GS10": [3.0, np.nan],
            "M1333BUSM156NNBR": [9.9, 2.5],
            "M1333AUSM156NNBR": [9.9, 9.9],
        },
        index=dates,
    )

    result = monthly.final_10yr_treasury_series(df)

    assert result.loc[dates[0]] == pytest.approx(3.0)
    assert result.loc[dates[1]] == pytest.approx(2.5)


def test_final_baa_series_monthly():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01"])
    df = pd.DataFrame({"BAA": [5.0, 5.2]}, index=dates)

    result = monthly.final_baa_series(df)

    assert list(result.values) == [5.0, 5.2]
    assert result.name == "BAA"


def test_calculate_baa_treasury_spread():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01"])
    baa = pd.Series([5.0, 5.5], index=dates)
    treasury = pd.Series([3.0, 3.2], index=dates)

    spread = monthly.calculate_baa_treasury_spread(baa, treasury)

    assert spread.loc[dates[0]] == pytest.approx(2.0)
    assert spread.loc[dates[1]] == pytest.approx(2.3)
    assert spread.name == "BAA_Treasury_spread"


def test_final_aaa_series_monthly():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01"])
    df = pd.DataFrame({"AAA": [4.0, 4.2]}, index=dates)

    result = monthly.final_aaa_series(df)

    assert list(result.values) == [4.0, 4.2]
    assert result.name == "AAA"


def test_calculate_aaa_treasury_spread():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01"])
    aaa = pd.Series([4.0, 4.5], index=dates)
    treasury = pd.Series([3.0, 3.2], index=dates)

    spread = monthly.calculate_aaa_treasury_spread(aaa, treasury)

    assert spread.loc[dates[0]] == pytest.approx(1.0)
    assert spread.loc[dates[1]] == pytest.approx(1.3)
    assert spread.name == "AAA_Treasury_spread"


def test_clean_fred_data_monthly_produces_expected_columns():
    dates = pd.to_datetime(["2010-01-01", "2010-02-01"])
    df = pd.DataFrame(
        {
            "USREC": [0, 1],
            "GS10": [3.0, 3.1],
            "M1333BUSM156NNBR": [np.nan, np.nan],
            "M1333AUSM156NNBR": [np.nan, np.nan],
            "BAA": [5.0, 5.2],
            "AAA": [4.0, 4.1],
        },
        index=dates,
    )

    cleaned = monthly.clean_fred_data_monthly(df)

    expected_columns = {
        "hist_recession_indicator",
        "Treasury_10yr",
        "BAA",
        "BAA_Treasury_spread",
        "AAA",
        "AAA_Treasury_spread",
    }
    assert expected_columns == set(cleaned.columns)
    assert cleaned.loc[dates[0], "BAA_Treasury_spread"] == pytest.approx(2.0)
    assert cleaned.loc[dates[0], "AAA_Treasury_spread"] == pytest.approx(1.0)


def test_save_data_readme_monthly(tmp_path):
    dates = pd.to_datetime(["2010-01-01"])
    df = pd.DataFrame(
        {"BAA": [5.0], "MADE_UP_COLUMN": [2.0]},
        index=dates,
    )

    file_path = monthly.save_data_readme(df, data_dir=tmp_path)

    assert file_path == tmp_path / "fred_final_series_monthly_readme.md"
    text = file_path.read_text()
    assert "BAA" in text
    assert "MADE_UP_COLUMN" in text
    assert "Unknown series" in text
