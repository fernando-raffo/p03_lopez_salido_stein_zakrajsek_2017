"""
Pull and cache macro/financial time series from FRED (Federal Reserve
Economic Data) via `pandas_datareader`.

The set of series to download is defined in `series_to_pull`, which maps
each FRED series ID to a human-readable description. Running this module
as a script will:

1. Download all series in `series_to_pull` between START_DATE and today.
2. Save the resulting DataFrame to `fred.parquet` in RAW_DATA_DIR.
3. Write a `fred_data_dictionary.md` file to RAW_DATA_DIR that documents
   which column in the saved DataFrame corresponds to which FRED series
   and description.

Other modules should use `load_fred` to read the cached parquet file
rather than re-pulling from FRED.
"""

from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web

from settings import config

DATA_DIR = Path(config("RAW_DATA_DIR"))
START_DATE = config("BUFFER_START_DATE")
END_DATE = config("EXTENSION_END_DATE")


# Maps each FRED series ID to a human-readable description. Used both to
# request the series from FRED and to label the corresponding column when
# generating the data dictionary in `save_data_dictionary`.
series_to_pull = {
    "AAA": "Moody's Seasoned Aaa Corporate Bond Yield (monthly)",
    "BAA": "Moody's Seasoned Baa Corporate Bond Yield (monthly)",
    "GS10": "Market Yield on U.S. Treasury Securities at 10-Year Constant Maturity (monthly)",
    "M1333BUSM156NNBR": "Yield on Long-Term United States Bonds for United States (monthly, 1942-1967)",
    "M1333AUSM156NNBR": "Yield on Long-Term United States Bonds for United States (monthly, 1919-1944)",
    "M1329AUSM193NNBR": "Yields on Short-Term United States Securities, Three-Six Month Treasury Notes and Certificates, Three Month Treasury Bills (monthly, 1920-1934)",
    "TB3MS": "3-Month Treasury Bill Secondary Market Rate, Discount Basis (monthly)",
    "CPIAUCNS": "Consumer Price Index for All Urban Consumers: All Items in U.S. City Average (monthly)",
    "B230RC0Q173SBEA": "Population (quarterly)",
    "POPH": "National Population (annual)",
    "GDPC1": "Real Gross Domestic Product (quarterly)",
    "GDPCA": "Real Gross Domestic Product (annual)",
}


def pull_fred(start_date=START_DATE, end_date=END_DATE, ffill=False):
    """
    Download all series listed in `series_to_pull` from FRED.

    Parameters
    ----------
    start_date : str or datetime, default START_DATE
        First date of the requested date range.
    end_date : str or datetime, default END_DATE
        Last date of the requested date range.
    ffill : bool, default False
        Unused placeholder for forward-filling lower-frequency series
        (e.g. quarterly/annual) up to the daily/monthly index. Kept as a
        parameter for callers that may want to opt into this behavior.

    Returns
    -------
    pandas.DataFrame
        DataFrame indexed by date, with one column per FRED series ID
        in `series_to_pull`.
    """
    df = web.DataReader(list(series_to_pull.keys()), "fred", start_date, end_date)
    return df


def load_fred(data_dir=DATA_DIR):
    """
    Load the previously pulled FRED data from disk.

    Must first run this module as main (or call `pull_fred` and save its
    output) to create `fred.parquet` in `data_dir`.

    Parameters
    ----------
    data_dir : str or Path, default DATA_DIR
        Directory containing `fred.parquet`.

    Returns
    -------
    pandas.DataFrame
        The cached FRED DataFrame, indexed by date.
    """
    file_path = Path(data_dir) / "fred.parquet"
    df = pd.read_parquet(file_path)
    return df


def save_data_dictionary(df, data_dir=DATA_DIR):
    """
    Write a Markdown data dictionary describing each column of `df`.

    For every column in `df` that corresponds to a known FRED series
    (i.e. is a key in `series_to_pull`), record its FRED series ID and
    human-readable description in a Markdown table. This makes it easy
    to look up what each column of `fred.parquet` represents without
    cross-referencing this module's source code.

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame returned by `pull_fred`, whose columns are FRED
        series IDs.
    data_dir : str or Path, default DATA_DIR
        Directory to write `fred_data_dictionary.md` into.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    file_path = filedir / "fred_data_dictionary.md"

    lines = [
        "# FRED Data Dictionary",
        "",
        "This file documents the columns found in `fred.parquet`, generated "
        "by `pull_fred.py`.",
        "",
        "| Column (FRED Series ID) | Description |",
        "| --- | --- |",
    ]
    for column in df.columns:
        description = series_to_pull.get(column, "Unknown series")
        lines.append(f"| {column} | {description} |")

    file_path.write_text("\n".join(lines) + "\n")
    return file_path


if __name__ == "__main__":
    df = pull_fred(START_DATE, END_DATE)
    filedir = Path(DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filedir / "fred.parquet")
    save_data_dictionary(df, filedir)
