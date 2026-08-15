"""
Process raw FRED data pulled by `pull_fred.py` into the annual series used in
the replication.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import CubicSpline

from pull_fred import load_fred
from settings import config

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
DATA_DICTIONARY_DIR = Path(config("DATA_DICTIONARY_DIR"))
DATA_START_DATE = config("BUFFER_START_DATE")
REPLICATION_START_DATE = config("REPLICATION_START_DATE")
END_DATE = config("EXTENSION_END_DATE")

# `cleaned_df` should span full calendar years only: from
# REPLICATION_START_DATE through the last full year before END_DATE (e.g.
# if END_DATE falls mid-year, that year is still incomplete and is
# excluded).
REPLICATION_START_YEAR = REPLICATION_START_DATE.year
TARGET_END_YEAR = END_DATE.year


# Maps each column of the cleaned DataFrame returned by
# `clean_fred_data_annual` to a human-readable description of how it is
# constructed. Used by `save_data_readme` to document
# `fred_final_series_annual.parquet`.
column_descriptions = {
    "GDP": (
        "Annual real GDP, in complete 2009 dollars. Annual GDPCA is used as-is for 1929-1947; "
        "from 1948 onward, quarterly GDPC1 (SAAR) is averaged over the "
        "four quarters of each calendar year."
    ),
    "Population": (
        "Annual population. For pre-1952, the"
        "annual (July 1) POPH observations are cubic-spline-interpolated "
        "to monthly frequency and averaged over each calendar year; from "
        "1952 onward, quarterly B230RC0Q173SBEA is averaged over the four "
        "quarters of each calendar year."
    ),
    "GDP_per_capita": ("GDP divided by Population for each year."),
    "CPI_inflation": (
        "Annual CPI inflation, computed as the December-to-December "
        "log-change of the not-seasonally-adjusted "
        "CPIAUCNS."
    ),
    "BAA": (
        "Annual Moody's Seasoned Baa Corporate Bond Yield, computed "
        "as the December value of the monthly "
        "BAA series."
    ),
    "AAA": (
        "Annual Moody's Seasoned Aaa Corporate Bond Yield, computed as "
        "the December value of the monthly AAA "
        "series."
    ),
    "Treasury_10yr": (
        "Annual 10-year Treasury (long-term government bond) yield, "
        "computed as the December value of a monthly series combining GS10, M1333BUSM156NNBR, and "
        "M1333AUSM156NNBR (in order of preference, most recent first)."
    ),
    "Treasury_3mo": (
        "Annual 3-month Treasury bill bond-equivalent yield, computed "
        "from the monthly discount rates "
        "TB3MS and M1329AUSM193NNBR (TB3MS preferred where both are "
        "available), converted to a bond-equivalent yield and then to "
        "the December value of each calendar year."
    ),
    "BAA_Treasury_spread": (
        "Annual spread between the Baa corporate bond yield and the "
        "10-year Treasury yield, computed "
        "as BAA minus Treasury_10yr."
    ),
    "AAA_Treasury_spread": (
        "Annual spread between the Aaa corporate bond yield and the "
        "10-year Treasury yield, computed "
        "as AAA minus Treasury_10yr."
    ),
}


def _trim_to_replication_range(series, end_year_cap=TARGET_END_YEAR):
    """
    Restrict a series indexed by year to the replication window.

    The lower bound is `REPLICATION_START_YEAR`. The upper bound is the
    latest year at or before `end_year_cap` that is actually present in
    `series` -- so if the raw data does not yet have complete data for
    `end_year_cap` (and it is therefore absent from `series`), the series
    is effectively truncated to the most recent complete year before it.

    Parameters
    ----------
    series : pandas.Series
        Series indexed by year.
    end_year_cap : int, default TARGET_END_YEAR
        Latest year that may appear in the returned series.

    Returns
    -------
    pandas.Series
        The series restricted to `[REPLICATION_START_YEAR, end_year_cap]`.
    """
    series = series[series.index >= REPLICATION_START_YEAR]
    series = series[series.index <= end_year_cap]
    return series


def final_gdp_series(df, annual_col="GDPCA", quarterly_col="GDPC1"):
    """
    Build a single annual real GDP series, in complete 2009 dollars (not
    billions), from the annual (1929 onward) and quarterly (1947 onward)
    FRED series.

    For 1929-1947, annual data is used as-is, since quarterly NIPA data is
    not available before 1947. From 1948 onward, the quarterly,
    seasonally-adjusted-annual-rate series is averaged over the four
    quarters of each calendar year (using whatever quarters are available
    for the most recent, potentially incomplete, year). The series is
    further restricted to `REPLICATION_START_YEAR` through the latest year
    at or before `TARGET_END_YEAR` that is present in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `annual_col` and
        `quarterly_col`.
    annual_col : str, default "GDPCA"
        Column name of the annual real GDP series.
    quarterly_col : str, default "GDPC1"
        Column name of the quarterly real GDP series.

    Returns
    -------
    pandas.Series
        Real GDP in complete dollars, indexed by year, spanning 1929
        through the last year of available quarterly data.
    """
    # GDPCA and GDPC1 are both reported in billions of dollars.
    annual = df[annual_col].dropna().copy() * 1e9
    annual.index = annual.index.year
    annual = annual[annual.index < 1948]

    quarterly = df[quarterly_col].dropna() * 1e9
    quarterly_annual = quarterly.resample("YE").mean()
    quarterly_annual.index = quarterly_annual.index.year
    quarterly_annual = quarterly_annual[quarterly_annual.index >= 1948]

    gdp = pd.concat([annual, quarterly_annual]).sort_index()
    gdp.index.name = "year"
    gdp.name = "GDP"
    gdp = _trim_to_replication_range(gdp)
    return gdp


def final_population_series(
    df, annual_col="POPH", quarterly_col="B230RC0Q173SBEA", threshold_year=1952
):
    """
    Build a single annual population series from Census Bureau annual data
    (recorded as of July 1 each year, through 1951) and the FRED quarterly
    population series (1952 onward).

    The pre-1952 annual (July 1) observations are interpolated to monthly
    frequency using a cubic spline, and the resulting monthly series is
    averaged over the 12 months of each calendar year. From 1952 onward,
    the quarterly series is averaged over the four quarters of each
    calendar year (using whatever quarters are available for the most
    recent, potentially incomplete, year). The series is further
    restricted to `REPLICATION_START_YEAR` through the latest year at or
    before `TARGET_END_YEAR` that is present in the data.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `annual_col` and
        `quarterly_col`.
    annual_col : str, default "POPH"
        Column name of the annual population series (recorded as of July 1).
    quarterly_col : str, default "B230RC0Q173SBEA"
        Column name of the quarterly population series.
    threshold_year : int, default 1952
        First year for which the quarterly series is used instead of the
        spline-interpolated annual series.

    Returns
    -------
    pandas.Series
        Population indexed by year.
    """
    annual = df[annual_col].dropna()
    annual = annual[annual.index.year < threshold_year]

    # Each annual observation pertains to July 1 of its year, per the
    # Census Bureau's mid-year population estimate convention.
    july_dates = pd.to_datetime([f"{y}-07-01" for y in annual.index.year])
    x = july_dates.map(pd.Timestamp.toordinal).to_numpy()
    spline = CubicSpline(x, annual.to_numpy())

    monthly_dates = pd.date_range(
        f"{annual.index.year.min()}-01-01",
        f"{annual.index.year.max()}-12-01",
        freq="MS",
    )
    monthly_x = monthly_dates.map(pd.Timestamp.toordinal).to_numpy()
    monthly = pd.Series(spline(monthly_x), index=monthly_dates)
    annual_from_monthly = monthly.resample("YE").mean()
    annual_from_monthly.index = annual_from_monthly.index.year

    # B230RC0Q173SBEA is reported in thousands of persons, while POPH is
    # reported in persons; rescale so the combined series has one unit.
    quarterly = df[quarterly_col].dropna() * 1000
    quarterly_annual = quarterly.resample("YE").mean()
    quarterly_annual.index = quarterly_annual.index.year
    quarterly_annual = quarterly_annual[quarterly_annual.index >= threshold_year]

    pop = pd.concat([annual_from_monthly, quarterly_annual]).sort_index()
    pop.index.name = "year"
    pop.name = "Population"
    pop = _trim_to_replication_range(pop)
    return pop


def calculate_gdp_per_capita(gdp_series, population_series):
    """
    Calculate GDP per capita by dividing the GDP series by the population series.

    Parameters
    ----------
    gdp_series : pandas.Series
        Series containing GDP values indexed by year.
    population_series : pandas.Series
        Series containing population values indexed by year.

    Returns
    -------
    pandas.Series
        GDP per capita indexed by year.
    """
    gdp_per_capita = gdp_series / population_series
    gdp_per_capita.name = "GDP_per_capita"
    return gdp_per_capita


def final_cpi_series(df, cpi_col="CPIAUCNS"):
    """
    Build an annual CPI inflation series as the December-to-December
    log-change of the not-seasonally-adjusted CPI-U (1982-84=100), which
    is available monthly from ALFRED/FRED since 1913.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `cpi_col`.
    cpi_col : str, default "CPIAUCNS"
        Column name of the not-seasonally-adjusted monthly CPI index.

    Returns
    -------
    pandas.Series
        Annual CPI inflation (log-change), indexed by year.
    """
    cpi = df[cpi_col].dropna()
    december = cpi[cpi.index.month == 12].copy()
    december.index = december.index.year

    inflation = np.log(december).diff().dropna()
    inflation.index.name = "year"
    inflation.name = "CPI_inflation"
    inflation = _trim_to_replication_range(inflation)
    return inflation


def _annual_december_value(series):
    """
    Convert a monthly series to annual frequency by taking the December
    value of each calendar year.
    """
    december = series.dropna()
    december = december[december.index.month == 12].copy()
    december.index = december.index.year
    december.index.name = "year"
    return december


def final_baa_series_annual(df, baa_col="BAA"):
    """
    Build an annual series of Moody's Seasoned Baa Corporate Bond Yield by
    taking the December value of the monthly (average) FRED series, which
    is available since 1919. Annual changes are thus December-to-December
    changes of the monthly series.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `baa_col`.
    baa_col : str, default "BAA"
        Column name of the monthly Baa corporate bond yield series.

    Returns
    -------
    pandas.Series
        Annual Baa corporate bond yield, indexed by year.
    """
    baa = _annual_december_value(df[baa_col])
    baa.name = "BAA"
    baa = _trim_to_replication_range(baa)
    return baa


def final_aaa_series(df, aaa_col="AAA"):
    """
    Build an annual series of Moody's Seasoned Aaa Corporate Bond Yield by
    taking the December value of the monthly (average) FRED series, which
    is available since 1919. Annual changes are thus December-to-December
    changes of the monthly series.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `aaa_col`.
    aaa_col : str, default "AAA"
        Column name of the monthly Aaa corporate bond yield series.

    Returns
    -------
    pandas.Series
        Annual Aaa corporate bond yield, indexed by year.
    """
    aaa = _annual_december_value(df[aaa_col])
    aaa.name = "AAA"
    aaa = _trim_to_replication_range(aaa)
    return aaa


def final_10yr_treasury_series_annual(
    df,
    latest_col="GS10",
    middle_col="M1333BUSM156NNBR",
    earliest_col="M1333AUSM156NNBR",
):
    """
    Build a single annual 10-year Treasury (long-term government bond)
    yield series from three overlapping monthly FRED series, taking the
    December value of each calendar year (so annual changes are
    December-to-December changes of the monthly series).

    Where more than one series has a value for a given month, the more
    recent series is preferred: `latest_col` takes priority over
    `middle_col`, which in turn takes priority over `earliest_col`.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `latest_col`,
        `middle_col`, and `earliest_col`.
    latest_col : str, default "GS10"
        Column name of the most recent monthly 10-year Treasury yield
        series.
    middle_col : str, default "M1333BUSM156NNBR"
        Column name of the middle monthly long-term government bond
        yield series.
    earliest_col : str, default "M1333AUSM156NNBR"
        Column name of the earliest monthly long-term government bond
        yield series.

    Returns
    -------
    pandas.Series
        Annual 10-year Treasury yield, indexed by year.
    """
    monthly = (
        df[latest_col].combine_first(df[middle_col]).combine_first(df[earliest_col])
    )

    treasury_10yr = _annual_december_value(monthly)
    treasury_10yr.name = "Treasury_10yr"
    treasury_10yr = _trim_to_replication_range(treasury_10yr)
    return treasury_10yr


def final_3mo_treasury_series(df, latest_col="TB3MS", older_col="M1329AUSM193NNBR"):
    """
    Build a single annual 3-month Treasury bill yield series from two
    overlapping monthly FRED series of the bank-discount rate, converted
    to a bond-equivalent yield.

    Where both series have a value for a given month, the more recent
    series is preferred: `latest_col` takes priority over `older_col`.
    Both FRED series are already annualized discount rates, in percent, on
    a bank-discount basis (i.e. computed off the face value using a
    360-day year), so the combined series can be built directly without
    any rescaling.

    Following the standard market-convention formula for converting a
    T-bill discount rate to a bond-equivalent (coupon-equivalent) yield
    for bills of 182 days or less, and using a fixed 91-day convention for
    the "3-month" tenor:

        BEY = 365 * d / (360 - 91 * d)

    where `d` is the discount rate expressed as a decimal. The resulting
    bond-equivalent series is then converted to monthly frequency by
    averaging the available values within each month, and to annual
    frequency by taking the December value of each calendar year (so
    annual changes are December-to-December changes of the monthly
    series).

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `latest_col` and
        `older_col`.
    latest_col : str, default "TB3MS"
        Column name of the most recent monthly 3-month Treasury bill
        discount rate series.
    older_col : str, default "M1329AUSM193NNBR"
        Column name of the older monthly 3-month Treasury bill (and, for
        its earliest years, Treasury note/certificate) discount rate
        series.

    Returns
    -------
    pandas.Series
        Annual 3-month Treasury bill bond-equivalent yield, indexed by
        year.
    """
    discount = df[latest_col].combine_first(df[older_col]) / 100

    n_days = 91
    bond_equivalent = 100 * (365 * discount) / (360 - n_days * discount)

    monthly = bond_equivalent.resample("MS").mean()

    treasury_3mo = _annual_december_value(monthly)
    treasury_3mo.name = "Treasury_3mo"
    treasury_3mo = _trim_to_replication_range(treasury_3mo)
    return treasury_3mo


def calculate_baa_treasury_spread(baa_series, treasury_series):
    """
    Calculate the annual Baa-Treasury spread by subtracting the 10-year
    Treasury yield from the Baa corporate bond yield.

    Parameters
    ----------
    baa_series : pandas.Series
        Series containing annual Baa corporate bond yield values, indexed
        by year.
    treasury_series : pandas.Series
        Series containing annual 10-year Treasury yield values, indexed
        by year.

    Returns
    -------
    pandas.Series
        Annual Baa-Treasury spread, indexed by year.
    """
    spread = baa_series - treasury_series
    spread.name = "BAA_Treasury_spread"
    return spread


def calculate_aaa_treasury_spread(aaa_series, treasury_series):
    """
    Calculate the annual Aaa-Treasury spread by subtracting the 10-year
    Treasury yield from the Aaa corporate bond yield.

    Parameters
    ----------
    aaa_series : pandas.Series
        Series containing annual Aaa corporate bond yield values, indexed
        by year.
    treasury_series : pandas.Series
        Series containing annual 10-year Treasury yield values, indexed
        by year.

    Returns
    -------
    pandas.Series
        Annual Aaa-Treasury spread, indexed by year.
    """
    spread = aaa_series - treasury_series
    spread.name = "AAA_Treasury_spread"
    return spread


def save_data_readme(df, data_dir=DATA_DICTIONARY_DIR):
    """
    Write a Markdown README describing how each column of `df` is
    constructed.

    For every column in `df` that has a known description (i.e. is a key
    in `column_descriptions`), record how it is derived from the raw FRED
    series. This makes it easy to understand what each column of
    `fred_final_series.parquet` represents without reading this module's
    source code.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame returned by `clean_fred_data_annual`.
    data_dir : str or Path, default DATA_DICTIONARY_DIR
        Directory to write `fred_final_series_annual_readme.md` into.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    file_path = filedir / "fred_final_series_annual_readme.md"

    lines = [
        "## Overview",
        "",
        "- **File:** `_data/processed_data/fred_final_series_annual.parquet`",
        "- **Source:** Derived from `fred_macroeconomic_variables`, "
        "itself pulled from [FRED](https://fred.stlouisfed.org/)",
        "- **Generated by:** `process_fred_data_annual.py`",
        "- **Frequency:** Annual",
        "- **Index:** `year`",
        "## Column Dictionary",
        "",
        "| Column | Description |",
        "| --- | --- |",
    ]
    for column in df.columns:
        description = column_descriptions.get(column, "Unknown series")
        lines.append(f"| {column} | {description} |")

    file_path.write_text("\n".join(lines) + "\n")
    return file_path


def clean_fred_data_annual(df):
    """
    Build the full cleaned annual series DataFrame used in the replication.

    Combines `final_gdp_series`, `final_population_series`,
    `calculate_gdp_per_capita`, `final_cpi_series`, `final_baa_series_annual`,
    `final_aaa_series`, `final_10yr_treasury_series_annual`,
    `final_3mo_treasury_series`, `calculate_baa_treasury_spread`, and
    `calculate_aaa_treasury_spread` into a single DataFrame indexed by year.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing the raw FRED series required
        by each of the individual `final_*_series` functions.

    Returns
    -------
    pandas.DataFrame
        Annual DataFrame with columns GDP, Population, GDP_per_capita,
        CPI_inflation, BAA, AAA, Treasury_10yr, Treasury_3mo,
        BAA_Treasury_spread, and AAA_Treasury_spread.
    """
    gdp = final_gdp_series(df)
    pop = final_population_series(df)
    gdp_per_capita = calculate_gdp_per_capita(gdp, pop)
    cpi_inflation = final_cpi_series(df)
    baa = final_baa_series_annual(df)
    aaa = final_aaa_series(df)
    treasury_10yr = final_10yr_treasury_series_annual(df)
    treasury_3mo = final_3mo_treasury_series(df)
    baa_treasury_spread = calculate_baa_treasury_spread(baa, treasury_10yr)
    aaa_treasury_spread = calculate_aaa_treasury_spread(aaa, treasury_10yr)

    return pd.DataFrame(
        {
            "GDP": gdp,
            "Population": pop,
            "GDP_per_capita": gdp_per_capita,
            "CPI_inflation": cpi_inflation,
            "BAA": baa,
            "AAA": aaa,
            "Treasury_10yr": treasury_10yr,
            "Treasury_3mo": treasury_3mo,
            "BAA_Treasury_spread": baa_treasury_spread,
            "AAA_Treasury_spread": aaa_treasury_spread,
        }
    )


if __name__ == "__main__":
    fred_df = load_fred(RAW_DATA_DIR)
    cleaned_df = clean_fred_data_annual(fred_df)
    filedir = Path(PROCESSED_DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(filedir / "fred_final_series_annual.parquet")
    cleaned_df.to_csv(filedir / "fred_final_series_annual.csv")
    save_data_readme(cleaned_df, DATA_DICTIONARY_DIR)
