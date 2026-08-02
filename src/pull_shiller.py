"""Pull, read, load, and process Robert Shiller's stock-market data set.

This module downloads the monthly U.S. stock-market data set that Robert
Shiller distributes alongside *Irrational Exuberance* (the file historically
named ``ie_data.xls``). The data set runs from January 1871 to the present and
contains the S&P Composite price, dividends, earnings, the CPI, the 10-year
Treasury (long) interest rate, and Shiller's cyclically adjusted
price-earnings ratio, the CAPE (a.k.a. ``P/E10`` or ``PE10``).

Why we need it
--------------
Lopez-Salido, Stein, and Zakrajsek (2017) use the log of the cyclically
adjusted price-earnings ratio, ``ln[P/E10]_{t-2}``, as the first-step predictor
of stock-market returns (Shiller 2000). See Table I and Table II of the paper.

Data structure
--------------
The workbook has a sheet named ``"Data"``. After a few header rows, each row is
one calendar month, in consecutive order starting 1871-01. The columns of
interest (by position, left to right) are:

    0. Date            -- encoded as ``YYYY.MM`` (note: October shows as
                          ``YYYY.1`` in the raw file, which is why we rebuild
                          the monthly index from row order instead of parsing
                          this column).
    1. P               -- S&P Composite price
    2. D               -- Dividend
    3. E               -- Earnings
    4. CPI             -- Consumer Price Index
    5. Date Fraction
    6. Long Rate (GS10)-- 10-year Treasury yield
    7. Real Price
    8. Real Dividend
    9. Real Total Return Price
    10. Real Earnings
    11. Real TR Scaled Earnings
    12. CAPE           -- cyclically adjusted P/E (P/E10)

Newer vintages append additional columns (TR CAPE, Excess CAPE Yield, etc.),
so we select the columns we need by position rather than assuming a fixed
width.

Naming conventions
------------------
- ``pull_shiller`` downloads from the web and returns a monthly DataFrame.
- ``load_shiller`` reads the cached copy from the ``_data`` directory.
- ``process_shiller_annual`` collapses the monthly data to annual frequency and
  adds ``ln_pe10``.

Running this file as a script pulls the data and caches it to ``DATA_DIR``
(the ``_data`` folder, which is git-ignored, so the data is never committed).
"""

from io import BytesIO
from pathlib import Path

import numpy as np
import pandas as pd
import requests

from settings import config


def _config_or(var_name, default):
    """Return ``config(var_name)`` if it is defined anywhere (CLI, env, or
    ``.env``), otherwise fall back to ``default``. Lets optional settings keep
    the project's precedence rules without erroring when they are simply unset.
    """
    try:
        return config(var_name)
    except ValueError:
        return default


RAW_DATA_DIR = Path(config("RAW_DATA_DIR"))
PROCESSED_DATA_DIR = Path(config("PROCESSED_DATA_DIR"))
START_DATE = config("REPLICATION_START_DATE")
END_DATE = config("REPLICATION_END_DATE")

# The canonical location of Shiller's spreadsheet. Kept configurable because
# Shiller has moved the file in the past (it is now also mirrored on
# https://shillerdata.com/). Override with the SHILLER_URL environment
# variable or a --SHILLER_URL command-line argument if the link changes.
SHILLER_URL = _config_or(
    "SHILLER_URL",
    "http://www.econ.yale.edu/~shiller/data/ie_data.xls",
)

# A browser-like header avoids the occasional 403 from the host.
_REQUEST_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (compatible; research-replication/1.0; +https://github.com/)"
    )
}

# Position -> clean column name for the columns we keep from the "Data" sheet.
_COLUMN_MAP = {
    1: "sp500_price",
    2: "dividend",
    3: "earnings",
    4: "cpi",
    6: "gs10",
    7: "real_price",
    8: "real_dividend",
    10: "real_earnings",
    12: "pe10",
}


def _download_workbook(url=SHILLER_URL):
    """Download the raw Excel workbook and return it as an in-memory buffer."""
    response = requests.get(url, headers=_REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    return BytesIO(response.content)


def parse_shiller_data_sheet(workbook):
    """Parse the ``Data`` sheet of Shiller's workbook into a monthly DataFrame.

    Parameters
    ----------
    workbook : path, buffer, or ``pandas.ExcelFile``
        Anything ``pandas.read_excel`` accepts that points at Shiller's file.

    Returns
    -------
    pandas.DataFrame
        Monthly data indexed by a ``Datetime(month-start)`` ``date`` index, with
        the columns listed in ``_COLUMN_MAP``.

    Notes
    -----
    The raw ``Date`` column encodes October as ``YYYY.1`` (indistinguishable
    from January's ``YYYY.10`` after Excel drops the trailing zero), so we do
    *not* trust it. Instead we keep only the rows whose first column is a
    plausible ``YYYY.MM`` date fraction, verify the series starts in 1871, and
    rebuild a clean consecutive monthly index from row order.
    """
    raw = pd.read_excel(workbook, sheet_name="Data", header=None)

    # Keep only genuine data rows: the first column is a date fraction such as
    # 1871.01 .. 2025.xx. Header rows and the trailing notes are dropped.
    first_col = pd.to_numeric(raw.iloc[:, 0], errors="coerce")
    is_data_row = first_col.between(1871, 2100)
    data = raw.loc[is_data_row].reset_index(drop=True)

    start_year = int(np.floor(first_col.loc[is_data_row].iloc[0]))
    if start_year != 1871:
        raise ValueError(
            f"Expected Shiller data to start in 1871, found {start_year}. "
            "The workbook layout may have changed; inspect the 'Data' sheet."
        )

    df = pd.DataFrame(index=data.index)
    for position, name in _COLUMN_MAP.items():
        df[name] = pd.to_numeric(data.iloc[:, position], errors="coerce")

    # Rebuild a clean monthly index from 1871-01 forward (row order is monthly).
    df["date"] = pd.date_range("1871-01-01", periods=len(df), freq="MS")
    df = df.set_index("date")
    return df


def pull_shiller(url=SHILLER_URL, start_date=START_DATE, end_date=END_DATE):
    """Download and parse Shiller's monthly stock-market data set.

    Parameters
    ----------
    url : str
        Location of the ``ie_data.xls`` workbook.
    start_date, end_date : datetime-like
        Inclusive date range used to trim the (very long) monthly series.

    Returns
    -------
    pandas.DataFrame
        Monthly data indexed by ``date`` (see :func:`parse_shiller_data_sheet`).
    """
    workbook = _download_workbook(url)
    df = parse_shiller_data_sheet(workbook)
    df = df.loc[
        str(pd.Timestamp(start_date).date()) : str(pd.Timestamp(end_date).date())
    ]
    return df


def process_shiller_annual(df_monthly, how="last"):
    """Collapse monthly Shiller data to annual frequency and add ``ln_pe10``.

    Parameters
    ----------
    df_monthly : pandas.DataFrame
        Output of :func:`pull_shiller` / :func:`load_shiller`.
    how : {"last", "mean"}
        ``"last"`` takes the December (year-end) observation of each year;
        ``"mean"`` takes the calendar-year average. Greenwood-Hanson-style
        specifications typically use a year-end level, which is the default.

    Returns
    -------
    pandas.DataFrame
        Annual data indexed by year-end date, with an added ``ln_pe10`` column
        (the natural log of ``P/E10``) used as the first-step stock-return
        predictor in Lopez-Salido, Stein, and Zakrajsek (2017).

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    >>> m = pd.DataFrame({"pe10": np.arange(1.0, 25.0)}, index=idx)
    >>> m.index.name = "date"
    >>> a = process_shiller_annual(m, how="last")
    >>> a["pe10"].tolist()
    [12.0, 24.0]
    >>> bool(np.isclose(a["ln_pe10"].iloc[-1], np.log(24.0)))
    True
    """
    if how == "last":
        annual = df_monthly.resample("YE").last()
    elif how == "mean":
        annual = df_monthly.resample("YE").mean()
    else:
        raise ValueError("`how` must be 'last' or 'mean'.")

    annual["ln_pe10"] = np.log(annual["pe10"])
    annual.index.name = "date"
    return annual


def load_shiller(data_dir=RAW_DATA_DIR):
    """Load the cached monthly Shiller data from the ``_data`` directory.

    Must first run this module as ``__main__`` to pull and save the data.
    """
    file_path = Path(data_dir) / "shiller_data.parquet"
    return pd.read_parquet(file_path)


def load_shiller_annual(data_dir=PROCESSED_DATA_DIR):
    """Load the cached annual Shiller data from the ``_data`` directory."""
    file_path = Path(data_dir) / "shiller_data_annual.parquet"
    return pd.read_parquet(file_path)


def _demo():
    df = load_shiller()
    print(df.tail())


if __name__ == "__main__":
    today = pd.Timestamp.today().strftime("%Y-%m-%d")
    df_monthly = pull_shiller(SHILLER_URL, START_DATE, today)
    df_annual = process_shiller_annual(df_monthly, how="last")
    df_annual = df_annual[["sp500_price", "dividend", "pe10", "ln_pe10"]]

    filedir = Path(RAW_DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df_monthly.to_parquet(filedir / "shiller_data.parquet")
    df_monthly.to_csv(filedir / "shiller_data.csv")

    filedir = Path(PROCESSED_DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df_annual.to_parquet(filedir / "shiller_data_annual.parquet")
    df_annual.to_csv(filedir / "shiller_data_annual.csv")
