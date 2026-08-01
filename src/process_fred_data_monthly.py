"""
Process raw FRED data pulled by `pull_fred.py` into the monthly series
used in the replication.
"""

from pathlib import Path

import pandas as pd

from pull_fred import load_fred
from settings import config

RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
GRAPH_START_DATE = config("BUFFER_START_DATE")
END_DATE = config("EXTENSION_END_DATE")


# Maps each column of the cleaned DataFrame returned by
# `clean_fred_data_monthly` to a human-readable description of how it is
# constructed. Used by `save_data_readme` to document
# `fred_final_series_monthly.parquet`.
column_descriptions = {
    "hist_recession_indicator": (
        "Monthly NBER-based recession indicator, taken as-is from the "
        "monthly USREC series by `final_recession_indicator_series`."
    ),
    "Treasury_10yr": (
        "Monthly 10-year Treasury (long-term government bond) yield, "
        "computed by `final_10yr_treasury_series` by combining GS10, "
        "M1333BUSM156NNBR, and M1333AUSM156NNBR (in order of preference, "
        "most recent first) at monthly frequency."
    ),
    "BAA": (
        "Monthly Moody's Seasoned Baa Corporate Bond Yield, taken as-is "
        "from the monthly BAA series by `final_baa_series`."
    ),
    "BAA_Treasury_spread": (
        "Monthly spread between the Baa corporate bond yield and the "
        "10-year Treasury yield, computed by "
        "`calculate_baa_treasury_spread` as BAA minus Treasury_10yr."
    ),
    "AAA": (
        "Monthly Moody's Seasoned Aaa Corporate Bond Yield, taken as-is "
        "from the monthly AAA series by `final_aaa_series`."
    ),
    "AAA_Treasury_spread": (
        "Monthly spread between the Aaa corporate bond yield and the "
        "10-year Treasury yield, computed by "
        "`calculate_aaa_treasury_spread` as AAA minus Treasury_10yr."
    ),
}


def _trim_to_replication_range(series):
    """
    Restrict a series indexed by date to the replication window.

    The lower bound is `GRAPH_START_DATE`. The upper bound is
    `END_DATE`; any months beyond the latest data actually pulled from
    FRED are already absent from `series` (since raw values not yet
    released are `NaN` and are dropped before this function is called).

    Parameters
    ----------
    series : pandas.Series
        Series indexed by month.

    Returns
    -------
    pandas.Series
        The series restricted to `[GRAPH_START_DATE, END_DATE]`.
    """
    series = series[series.index >= GRAPH_START_DATE]
    series = series[series.index <= END_DATE]
    return series


def final_recession_indicator_series(df, recession_col="USREC"):
    """
    Build a monthly historical recession indicator series from the NBER
    based recession indicator, which is available monthly since 1854.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `recession_col`.
    recession_col : str, default "USREC"
        Column name of the monthly recession indicator series.

    Returns
    -------
    pandas.Series
        Monthly recession indicator, indexed by month.
    """
    recession = df[recession_col].dropna().copy()
    recession.index.name = "date"
    recession.name = "hist_recession_indicator"
    recession = _trim_to_replication_range(recession)
    return recession


def final_10yr_treasury_series(
    df,
    latest_col="GS10",
    middle_col="M1333BUSM156NNBR",
    earliest_col="M1333AUSM156NNBR",
):
    """
    Build a single monthly 10-year Treasury (long-term government bond)
    yield series from three overlapping monthly FRED series.

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
        Monthly 10-year Treasury yield, indexed by month.
    """
    treasury_10yr = (
        df[latest_col].combine_first(df[middle_col]).combine_first(df[earliest_col])
    )
    treasury_10yr = treasury_10yr.dropna().copy()
    treasury_10yr.index.name = "date"
    treasury_10yr.name = "Treasury_10yr"
    treasury_10yr = _trim_to_replication_range(treasury_10yr)
    return treasury_10yr


def final_baa_series(df, baa_col="BAA"):
    """
    Build a monthly series of Moody's Seasoned Baa Corporate Bond Yield,
    which is available monthly since 1919.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `baa_col`.
    baa_col : str, default "BAA"
        Column name of the monthly Baa corporate bond yield series.

    Returns
    -------
    pandas.Series
        Monthly Baa corporate bond yield, indexed by month.
    """
    baa = df[baa_col].dropna().copy()
    baa.index.name = "date"
    baa.name = "BAA"
    baa = _trim_to_replication_range(baa)
    return baa


def final_aaa_series(df, aaa_col="AAA"):
    """
    Build a monthly series of Moody's Seasoned Aaa Corporate Bond Yield,
    which is available monthly since 1919.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing at least `aaa_col`.
    aaa_col : str, default "AAA"
        Column name of the monthly Aaa corporate bond yield series.

    Returns
    -------
    pandas.Series
        Monthly Aaa corporate bond yield, indexed by month.
    """
    aaa = df[aaa_col].dropna().copy()
    aaa.index.name = "date"
    aaa.name = "AAA"
    aaa = _trim_to_replication_range(aaa)
    return aaa


def calculate_baa_treasury_spread(baa_series, treasury_series):
    """
    Calculate the monthly Baa-Treasury spread by subtracting the 10-year
    Treasury yield from the Baa corporate bond yield.

    Parameters
    ----------
    baa_series : pandas.Series
        Series containing monthly Baa corporate bond yield values,
        indexed by month.
    treasury_series : pandas.Series
        Series containing monthly 10-year Treasury yield values, indexed
        by month.

    Returns
    -------
    pandas.Series
        Monthly Baa-Treasury spread, indexed by month.
    """
    spread = baa_series - treasury_series
    spread.name = "BAA_Treasury_spread"
    return spread


def calculate_aaa_treasury_spread(aaa_series, treasury_series):
    """
    Calculate the monthly Aaa-Treasury spread by subtracting the 10-year
    Treasury yield from the Aaa corporate bond yield.

    Parameters
    ----------
    aaa_series : pandas.Series
        Series containing monthly Aaa corporate bond yield values,
        indexed by month.
    treasury_series : pandas.Series
        Series containing monthly 10-year Treasury yield values, indexed
        by month.

    Returns
    -------
    pandas.Series
        Monthly Aaa-Treasury spread, indexed by month.
    """
    spread = aaa_series - treasury_series
    spread.name = "AAA_Treasury_spread"
    return spread


def save_data_readme(df, data_dir=PROCESSED_DATA_DIR):
    """
    Write a Markdown README describing how each column of `df` is
    constructed.

    For every column in `df` that has a known description (i.e. is a key
    in `column_descriptions`), record how it is derived from the raw FRED
    series. This makes it easy to understand what each column of
    `fred_final_series_monthly.parquet` represents without reading this
    module's source code.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame returned by `clean_fred_data_monthly`.
    data_dir : str or Path, default PROCESSED_DATA_DIR
        Directory to write `fred_final_series_monthly_readme.md` into.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    file_path = filedir / "fred_final_series_monthly_readme.md"

    lines = [
        "# FRED Cleaned Monthly Series README",
        "",
        "This file documents the columns found in "
        "`fred_final_series_monthly.parquet`, generated by "
        "`process_fred_data_monthly.py` from the raw series pulled by "
        "`pull_fred.py`.",
        "",
        "| Column | Description |",
        "| --- | --- |",
    ]
    for column in df.columns:
        description = column_descriptions.get(column, "Unknown series")
        lines.append(f"| {column} | {description} |")

    file_path.write_text("\n".join(lines) + "\n")
    return file_path


def clean_fred_data_monthly(df):
    """
    Build the full cleaned monthly series DataFrame used in the
    replication.

    Combines `final_recession_indicator_series`, `final_10yr_treasury_series`,
    `final_baa_series`, `calculate_baa_treasury_spread`, `final_aaa_series`,
    and `calculate_aaa_treasury_spread` into a single DataFrame indexed by
    month.

    Parameters
    ----------
    df : pandas.DataFrame
        DataFrame indexed by date, containing the raw FRED series required
        by each of the individual series-building functions.

    Returns
    -------
    pandas.DataFrame
        Monthly DataFrame with columns hist_recession_indicator,
        Treasury_10yr, BAA, BAA_Treasury_spread, AAA, and
        AAA_Treasury_spread.
    """
    recession_indicator = final_recession_indicator_series(df)
    treasury_10yr = final_10yr_treasury_series(df)
    baa = final_baa_series(df)
    baa_treasury_spread = calculate_baa_treasury_spread(baa, treasury_10yr)
    aaa = final_aaa_series(df)
    aaa_treasury_spread = calculate_aaa_treasury_spread(aaa, treasury_10yr)

    return pd.DataFrame(
        {
            "hist_recession_indicator": recession_indicator,
            "Treasury_10yr": treasury_10yr,
            "BAA": baa,
            "BAA_Treasury_spread": baa_treasury_spread,
            "AAA": aaa,
            "AAA_Treasury_spread": aaa_treasury_spread,
        }
    )


if __name__ == "__main__":
    fred_df = load_fred(RAW_DATA_DIR)
    cleaned_df = clean_fred_data_monthly(fred_df)
    filedir = Path(PROCESSED_DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    cleaned_df.to_parquet(filedir / "fred_final_series_monthly.parquet")
    save_data_readme(cleaned_df, filedir)
