"""
Pull, read, load, and process Robert Shiller's stock-market data set.

Downloads the monthly U.S. stock-market data set Robert Shiller distributes
alongside *Irrational Exuberance* (the file historically named
``ie_data.xls``), covering January 1871 to the present: S&P Composite price,
dividends, earnings, the CPI, the 10-year Treasury (long) rate, and Shiller's
cyclically adjusted price-earnings ratio (CAPE, a.k.a. ``P/E10``). Lopez-
Salido, Stein, and Zakrajsek (2017) use ``ln[P/E10]_{t-2}`` as the first-step
predictor of stock-market returns in Tables I and II.

The workbook's ``"Data"`` sheet has one row per calendar month starting
1871-01; columns are selected by position (see ``_COLUMN_MAP``) rather than
name, since newer vintages append extra columns (TR CAPE, Excess CAPE Yield,
etc.). The raw ``Date`` column encodes October as ``YYYY.1`` (ambiguous with
January's truncated ``YYYY.10``), so the monthly index is rebuilt from row
order instead of parsed from that column.

Naming conventions
------------------
- ``pull_shiller`` downloads from the web and returns a monthly DataFrame.
- ``load_shiller`` reads the cached copy from the ``_data`` directory.
- ``process_shiller_annual`` collapses to annual frequency and adds ``ln_pe10``.
- ``save_data_dictionary`` / ``save_data_dictionary_annual`` write Markdown
  data dictionaries for the raw monthly and processed annual parquet files.

Running this file as a script pulls the data and caches it to ``DATA_DIR``
(the ``_data`` folder, which is git-ignored, so the data is never committed).
"""

import re
from io import BytesIO
from pathlib import Path
from urllib.parse import urljoin

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
DATA_DICTIONARY_DIR = Path(config("DATA_DICTIONARY_DIR"))
START_DATE = config("BUFFER_START_DATE")
END_DATE = config("EXTENSION_END_DATE")
PROCESSED_START_DATE = config("REPLICATION_START_DATE")

# The canonical location of Shiller's data. Points at his data website's
# landing page rather than a direct link to the workbook: shillerdata.com
# serves ie_data.xls from a versioned CDN URL (a `?ver=...` query string) that
# changes every time Shiller updates the file, so `_download_workbook` scrapes
# the current link from this page instead of hard-coding it. Override with the
# SHILLER_URL environment variable or a --SHILLER_URL command-line argument
# (accepts either this landing page or a direct .xls link) if it changes.
SHILLER_URL = _config_or(
    "SHILLER_URL",
    "https://shillerdata.com/",
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

# Human-readable description of each column of `shiller_data.parquet` (the
# raw monthly pull). Used by `save_data_dictionary`.
_RAW_COLUMN_DESCRIPTIONS = {
    "sp500_price": "S&P Composite (S&P 500 predecessor) nominal price index, monthly.",
    "dividend": "S&P Composite nominal dividend, monthly, as reported by Shiller.",
    "earnings": "S&P Composite nominal earnings, monthly, as reported by Shiller.",
    "cpi": (
        "Consumer Price Index (CPI-U), monthly, as reported in Shiller's "
        "data set; used to construct the real (inflation-adjusted) series."
    ),
    "gs10": "10-year U.S. Treasury (long-term government bond) yield, monthly.",
    "real_price": "S&P Composite price, deflated to real (CPI-adjusted) terms by Shiller.",
    "real_dividend": "S&P Composite dividend, deflated to real (CPI-adjusted) terms by Shiller.",
    "real_earnings": "S&P Composite earnings, deflated to real (CPI-adjusted) terms by Shiller.",
    "pe10": (
        "Shiller's cyclically adjusted price-earnings ratio (CAPE / P/E10): "
        "real price divided by the 10-year moving average of real earnings."
    ),
}

# Human-readable description of each column of `shiller_data_annual.parquet`
# (the processed annual series). Used by `save_data_dictionary_annual`.
_ANNUAL_COLUMN_DESCRIPTIONS = {
    "sp500_price": ("S&P Composite nominal price index, year-end (December) value."),
    "dividend": ("S&P Composite nominal dividend, year-end (December) value."),
    "pe10": (
        "Shiller's cyclically adjusted price-earnings ratio (CAPE / P/E10), "
        "year-end (December) value."
    ),
    "ln_pe10": (
        "Natural log of `pe10`. Used "
        "as `ln[P/E10]_{t-2}`, the first-step stock-return predictor in "
        "Lopez-Salido, Stein, and Zakrajsek (2017)."
    ),
}


# Matches the ie_data.xls link embedded in shillerdata.com's landing page,
# e.g. `//img1.wsimg.com/.../ie_data.xls?ver=1785857394436`. The page also
# mentions "ie_data.xls" in plain prose ("File is ie_data.xls below:"), so the
# pattern requires a preceding "/" (present in any real URL path, absent in
# prose) and excludes whitespace/quotes/angle brackets to avoid overrunning
# into surrounding markup.
_IE_DATA_LINK_RE = re.compile(r'[^\s"\'<>]*/ie_data\.xls[^\s"\'<>]*')


def _resolve_shiller_xls_url(page_url):
    """Scrape shillerdata.com's landing page for the current ie_data.xls link.

    The page serves the workbook from a versioned CDN URL that changes every
    time Shiller updates the file, so the direct link can't be hard-coded;
    this re-derives it from the page on every pull instead.
    """
    response = requests.get(page_url, headers=_REQUEST_HEADERS, timeout=60)
    response.raise_for_status()
    match = _IE_DATA_LINK_RE.search(response.text)
    if not match:
        raise ValueError(
            f"Could not find a link to 'ie_data.xls' on {page_url}. "
            "shillerdata.com's page layout may have changed; inspect it "
            "directly, or set SHILLER_URL to a direct .xls link."
        )
    return urljoin(page_url, match.group(0))


def _download_workbook(url=SHILLER_URL):
    """Download the raw Excel workbook and return it as an in-memory buffer.

    ``url`` may point either directly at an ``.xls`` file or at Shiller's data
    landing page (e.g. the default ``https://shillerdata.com/``), in which
    case the actual workbook link is scraped from the page first.
    """
    if not url.lower().split("?")[0].endswith(".xls"):
        url = _resolve_shiller_xls_url(url)
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

    Only calendar years with the monthly data needed to actually compute the
    requested annual value are kept. A year missing its December observation
    (e.g. the most recent year, if the pull lags behind or ``END_DATE`` cuts
    it off mid-year) is dropped rather than silently substituted with an
    earlier month's value for ``how="last"``, or averaged over a partial year
    for ``how="mean"``.

    Parameters
    ----------
    df_monthly : pandas.DataFrame
        Output of :func:`pull_shiller` / :func:`load_shiller`.
    how : {"last", "mean"}
        ``"last"`` takes the December (year-end) observation of each
        complete year; ``"mean"`` takes the calendar-year average of each
        year that has all 12 months present. Greenwood-Hanson-style
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

    A trailing partial year (here, only through June 2001) is dropped rather
    than treated as if June were the year-end value:

    >>> m2 = m.iloc[:18]  # 2000 complete, 2001 only through June
    >>> a2 = process_shiller_annual(m2, how="last")
    >>> a2.index.year.tolist()
    [2000]
    """
    months_by_year = pd.Series(df_monthly.index.month, index=df_monthly.index.year)

    if how == "last":
        complete_years = months_by_year.eq(12).groupby(level=0).any()
        annual = df_monthly.resample("YE").last()
    elif how == "mean":
        complete_years = months_by_year.groupby(level=0).size().eq(12)
        annual = df_monthly.resample("YE").mean()
    else:
        raise ValueError("`how` must be 'last' or 'mean'.")

    annual = annual.loc[annual.index.year.map(complete_years).fillna(False)]

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


def save_data_dictionary(df, data_dir=DATA_DICTIONARY_DIR):
    """Write a Markdown data dictionary describing each column of the raw
    monthly Shiller pull (``shiller_data.parquet``).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame returned by :func:`pull_shiller`.
    data_dir : str or Path, default DATA_DICTIONARY_DIR
        Directory to write ``shiller_data_dictionary.md`` into.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    file_path = filedir / "shiller_data_dictionary.md"

    lines = [
        "## Overview",
        "",
        "- **File:** `_data/raw_data/shiller_data.parquet`",
        "- **Source:** [Robert Shiller's data website](https://shillerdata.com/) "
        "(the `ie_data.xls` workbook, `Data` sheet)",
        "- **Pulled by:** `pull_shiller.py`",
        "- **Frequency:** Monthly, from 1871-01 onward",
        "- **Index:** `date`",
        "## Column Dictionary",
        "",
        "| Column | Description |",
        "| --- | --- |",
    ]
    for column in df.columns:
        description = _RAW_COLUMN_DESCRIPTIONS.get(column, "Unknown series")
        lines.append(f"| {column} | {description} |")

    file_path.write_text("\n".join(lines) + "\n")
    return file_path


def save_data_dictionary_annual(df, data_dir=DATA_DICTIONARY_DIR):
    """Write a Markdown data dictionary describing each column of the
    processed annual Shiller series (``shiller_data_annual.parquet``).

    Parameters
    ----------
    df : pandas.DataFrame
        The DataFrame returned by :func:`process_shiller_annual`.
    data_dir : str or Path, default DATA_DICTIONARY_DIR
        Directory to write ``shiller_data_annual_dictionary.md`` into.

    Returns
    -------
    Path
        Path to the written Markdown file.
    """
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    file_path = filedir / "shiller_data_annual_dictionary.md"

    lines = [
        "## Overview",
        "",
        "- **File:** `_data/processed_data/shiller_data_annual.parquet`",
        "- **Source:** Derived from `shiller_market_variables`, "
        "itself pulled from "
        "[Robert Shiller's data website](https://shillerdata.com/)",
        "- **Generated by:** `pull_shiller.py`",
        "- **Frequency:** Annual (year-end / December value of each year)",
        "- **Index:** `date`",
        "",
        "This file documents the columns found in `shiller_data_annual.parquet`, "
        "generated by `process_shiller_annual` from the raw monthly series "
        "pulled by `pull_shiller.py`.",
        "",
        "## Column Dictionary",
        "",
        "| Column | Description |",
        "| --- | --- |",
    ]
    for column in df.columns:
        description = _ANNUAL_COLUMN_DESCRIPTIONS.get(column, "Unknown series")
        lines.append(f"| {column} | {description} |")

    file_path.write_text("\n".join(lines) + "\n")
    return file_path


if __name__ == "__main__":
    df_monthly = pull_shiller(SHILLER_URL, START_DATE, END_DATE)
    df_annual = process_shiller_annual(df_monthly, how="last")
    df_annual = df_annual.loc[
        str(pd.Timestamp(PROCESSED_START_DATE).date()) : str(
            pd.Timestamp(END_DATE).date()
        )
    ]
    df_annual = df_annual[["sp500_price", "dividend", "pe10", "ln_pe10"]]

    filedir = Path(RAW_DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df_monthly.to_parquet(filedir / "shiller_data.parquet")
    df_monthly.to_csv(filedir / "shiller_data.csv")
    save_data_dictionary(df_monthly, DATA_DICTIONARY_DIR)

    filedir = Path(PROCESSED_DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df_annual.to_parquet(filedir / "shiller_data_annual.parquet")
    df_annual.to_csv(filedir / "shiller_data_annual.csv")
    save_data_dictionary_annual(df_annual, DATA_DICTIONARY_DIR)
