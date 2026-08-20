"""
Pull, read, load, and process the Greenwood-Hanson high-yield share (HYS).

The *high-yield share* (``HYS``) is the fraction of gross nonfinancial
corporate bond issuance in a given year that is rated below investment grade
("high yield" / "junk"). Greenwood and Hanson (2013, *Review of Financial
Studies*) show it forecasts corporate-bond returns; Lopez-Salido, Stein, and
Zakrajsek (2017) use ``ln(HYS)_{t-2}`` as a first-step predictor of changes
in the Baa-Treasury credit spread (the auxiliary regressions in Tables II
and V).

There is no free public CSV/API for the HYS -- it is *constructed* from
bond-level issuance data. This module reconstructs it from **Mergent FISD**
(Fixed Income Securities Database), accessed through **WRDS** (the project's
``.env.example`` anticipates a ``WRDS_USERNAME``): restrict to U.S.
nonfinancial corporate issues (dropping financials, convertibles, non-USD,
asset-backed, and government/agency debt), assign each issue a rating at
issuance from its first Moody's rating, and compute
``HYS = (high-yield issuance) / (total issuance)`` per year. Issues with no
Moody's rating are excluded from the denominator entirely, following
Greenwood and Hanson (2013); the ``n_issues`` column flags thin early years
for screening (e.g. ``df.loc[df.n_issues >= 25]``).

FISD's usable coverage effectively begins in the early 1980s, so it cannot
reach the 1929 start of the LSZ sample on its own. To cover the full sample,
this module also ships the published Greenwood-Hanson (2013) high-yield
share for 1926-2008 (their Table 2, sourced from Hickman 1960, Atkinson
1967, and hand-collected Moody's Bond Surveys pre-1983). ``source="spliced"``
(the default) returns those published values through 2008 and appends the
local FISD reconstruction for later years, giving a continuous series
covering the full 1929-2015 sample; ``source="historical"`` returns just the
published 1926-2008 series and needs no WRDS access.

Naming conventions
------------------
- ``pull_greenwood_hanson`` obtains the data (FISD via WRDS, or the
  published-historical series) and returns an annual DataFrame.
- ``load_greenwood_hanson`` reads the cached, combined copy from the
  ``_data/processed_data`` directory.
- ``compute_hy_share`` is the pure aggregation step and is unit-tested with
  synthetic data (it needs no network or credentials).
- ``save_data_dictionary_historical`` / ``save_data_dictionary_fisd`` /
  ``save_data_dictionary_combined`` write Markdown data dictionaries
  documenting the columns of each of the three parquet files.

Running this file as a script pulls the data and caches it under ``_data``
(git-ignored, so nothing is ever committed): the historical and FISD series
go to ``RAW_DATA_DIR``, and the final spliced annual series to
``PROCESSED_DATA_DIR``.
"""

import warnings
from pathlib import Path

import numpy as np
import pandas as pd

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
END_DATE = config("EXTENSION_END_DATE")
PROCESSED_START_DATE = config("REPLICATION_START_DATE")

# Which source to use by default: "spliced", "fisd" (WRDS Mergent FISD), or
# "historical" (no WRDS needed). Configurable via env var GH_HYS_SOURCE or
# --GH_HYS_SOURCE.
GH_HYS_SOURCE = _config_or("GH_HYS_SOURCE", "spliced")
WRDS_USERNAME = _config_or("WRDS_USERNAME", "")

# Moody's / S&P letter grades at or above these are investment grade; anything
# strictly below is high yield. FISD stores agency ratings as text codes.
_INVESTMENT_GRADE_MOODYS = {
    "Aaa",
    "Aa1",
    "Aa2",
    "Aa3",
    "A1",
    "A2",
    "A3",
    "Baa1",
    "Baa2",
    "Baa3",
}
_INVESTMENT_GRADE_SP = {
    "AAA",
    "AA+",
    "AA",
    "AA-",
    "A+",
    "A",
    "A-",
    "BBB+",
    "BBB",
    "BBB-",
}
_INVESTMENT_GRADE = _INVESTMENT_GRADE_MOODYS | _INVESTMENT_GRADE_SP

# Human-readable description of each column, keyed by which parquet file it
# appears in. Used by the `save_data_dictionary_*` functions below.
_HISTORICAL_COLUMN_DESCRIPTIONS = {
    "hy_share": (
        "Published Greenwood-Hanson (2013) annual high-yield share (fraction "
        "of nonfinancial corporate bond issuance rated below investment "
        "grade), from Table 2 of Greenwood and Hanson (2013)."
    ),
    "ln_hy_share": "Natural log of `hy_share`.",
    "source": (
        "Label identifying the data's provenance; always 'gh2013' for this "
        "published historical series."
    ),
}

_FISD_COLUMN_DESCRIPTIONS = {
    "hy_issuance": (
        "Total face amount of nonfinancial U.S. corporate bonds issued in "
        "the year that are rated below investment grade (high yield), "
        "reconstructed from Mergent FISD via WRDS."
    ),
    "total_issuance": (
        "Total face amount of all rated nonfinancial U.S. corporate bonds "
        "issued in the year, reconstructed from Mergent FISD via WRDS."
    ),
    "n_issues": (
        "Number of bond issues underlying the year's `total_issuance`; "
        "useful for screening out thin early years, e.g. "
        "`df.loc[df.n_issues >= 25]`."
    ),
    "hy_share": "High-yield share for the year, computed as `hy_issuance / total_issuance`.",
    "ln_hy_share": "Natural log of `hy_share` (NaN when `hy_share` is 0).",
}

_COMBINED_COLUMN_DESCRIPTIONS = {
    "hy_share": (
        "Annual high-yield share, spliced from the "
        "published Greenwood-Hanson (2013) series through 2008, followed by "
        "the Mergent FISD reconstruction from 2009 onward."
    ),
    "ln_hy_share": (
        "Natural log of `hy_share`. Used as `ln(HYS)_{t-2}`, a first-step "
        "predictor of changes in the Baa-Treasury credit spread in "
        "Lopez-Salido, Stein, and Zakrajsek (2017)."
    ),
    "source": (
        "Label identifying which underlying series each year's value comes "
        "from: 'gh2013' (published Greenwood-Hanson historical series) or "
        "'fisd' (Mergent FISD reconstruction)."
    ),
}


def is_high_yield(rating):
    """Return ``True`` if a Moody's/S&P letter rating is below investment grade.

    Unrated issues (``None``/``NaN``/empty) return ``False`` (treated as not
    high yield), matching the convention of counting only *rated* speculative
    issuance in the numerator.

    Examples
    --------
    >>> is_high_yield("Ba1"), is_high_yield("BB+")
    (True, True)
    >>> is_high_yield("Baa3"), is_high_yield("AAA")
    (False, False)
    >>> is_high_yield(None), is_high_yield(float("nan"))
    (False, False)
    """
    if rating is None:
        return False
    if isinstance(rating, float) and np.isnan(rating):
        return False
    rating = str(rating).strip()
    if rating == "" or rating.upper() in {"NR", "NA", "NAN"}:
        return False
    return rating not in _INVESTMENT_GRADE


def compute_hy_share(issues):
    """Aggregate issue-level data into the annual high-yield share.

    Parameters
    ----------
    issues : pandas.DataFrame
        One row per bond issue with at least:

        - ``year`` : int, the calendar year of the offering
        - ``offering_amt`` : float, face amount issued (any consistent unit)
        - ``high_yield`` : bool, whether the issue is below investment grade

        An optional ``nonfinancial`` boolean column, if present, restricts the
        universe to nonfinancial issuers (rows where it is ``False`` are
        dropped) as in Greenwood and Hanson (2013).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``year`` with columns ``hy_issuance``, ``total_issuance``,
        ``hy_share`` (in [0, 1]) and ``ln_hy_share``.

    Examples
    --------
    >>> import pandas as pd
    >>> issues = pd.DataFrame({
    ...     "year": [1990, 1990, 1990, 1991],
    ...     "offering_amt": [100.0, 300.0, 100.0, 50.0],
    ...     "high_yield": [True, False, True, False],
    ... })
    >>> out = compute_hy_share(issues)
    >>> float(out.loc[1990, "hy_share"])
    0.4
    >>> float(out.loc[1991, "hy_share"])
    0.0
    """
    issues = issues.copy()
    if "nonfinancial" in issues.columns:
        # Coerce via pandas' nullable "boolean" dtype so this works whether the
        # caller passes plain numpy bools or the nullable dtypes that database
        # drivers (e.g. WRDS/psycopg2) return. Unknown (NA) is treated as
        # nonfinancial so that issues with a missing SIC code are kept rather
        # than silently dropped; see `_clean_fisd_issues`.
        keep = issues["nonfinancial"].astype("boolean").fillna(True).astype(bool)
        issues = issues.loc[keep]

    issues["offering_amt"] = pd.to_numeric(issues["offering_amt"], errors="coerce")
    issues["high_yield"] = (
        issues["high_yield"].astype("boolean").fillna(False).astype(bool)
    )
    issues = issues.dropna(subset=["year", "offering_amt"])
    issues["year"] = issues["year"].astype(int)

    grouped = issues.groupby("year")
    total = grouped["offering_amt"].sum().rename("total_issuance")
    n_issues = grouped.size().rename("n_issues")
    hy = (
        issues.loc[issues["high_yield"]]
        .groupby("year")["offering_amt"]
        .sum()
        .reindex(total.index, fill_value=0.0)
        .rename("hy_issuance")
    )

    out = pd.concat([hy, total, n_issues], axis=1)
    # Cast to plain float64 first: nullable dtypes make numpy emit a spurious
    # "divide by zero encountered in log" warning for the zero-share years.
    out["hy_issuance"] = out["hy_issuance"].astype("float64")
    out["total_issuance"] = out["total_issuance"].astype("float64")
    out["n_issues"] = out["n_issues"].astype("int64")
    out["hy_share"] = out["hy_issuance"] / out["total_issuance"]
    out["ln_hy_share"] = np.log(out["hy_share"].where(out["hy_share"] > 0))
    out.index.name = "year"
    return out


# ---------------------------------------------------------------------------
# Source 1: Mergent FISD via WRDS (authoritative)
# ---------------------------------------------------------------------------

# SQL selecting gross corporate bond issuance with the first rating assigned at
# (or just after) issuance. Table / column names follow the standard WRDS FISD
# schema; adjust here if your WRDS FISD vintage differs.
# U.S. *corporate* bond_type codes in FISD. This is an allowlist rather than a
# blocklist: FISD also carries agency debentures (ADEB/AMTN/ASPZ), Treasury
# issues (USBN/USBL/USNT/USSP/USSI), foreign government debt (FGOV), preferred
# stock (PS/PSTK) and trust-preferred securities (TPCS), none of which belong in
# corporate bond issuance. Verified against the bond_type frequency counts in
# the WRDS FISD vintage used here.
_CORPORATE_BOND_TYPES = (
    "CDEB",  # U.S. corporate debenture (the bulk of straight corporate debt)
    "CMTN",  # U.S. corporate medium-term note
    "CMTZ",  # U.S. corporate MTN, zero coupon
    "CZ",  # U.S. corporate zero coupon
    "CS",  # U.S. corporate structured
    "CPAS",  # U.S. corporate pass-through
    "CPIK",  # U.S. corporate payment-in-kind
    "CCOV",  # U.S. corporate covered
    "CCUR",  # U.S. corporate currency-linked
)

#   - convertible = 'N'                      -> convertibles are only folded in
#     for 1966-1982; in the FISD period they are not part of the base
#   - foreign_currency = 'N'                 -> USD issuance
#   - exchange offers are excluded too, but the column that flags them varies
#     across FISD vintages, so that predicate is added at runtime by
#     `_exchange_offer_clause` (see below) rather than hard-coded here.
# Financials (SIC 6000-6999) are dropped later, in `_clean_fisd_issues`, once
# the issuer SIC code is joined on. rating_type = 'MR' selects Moody's, and the
# denominator is restricted to *rated* issuance (HY + IG) by dropping issues
# with no Moody's rating, exactly as in Equation 9.
_FISD_SQL = """
    WITH corp AS (
        SELECT i.issue_id,
               i.issuer_id,
               i.offering_amt,
               i.offering_date
        FROM fisd.fisd_mergedissue AS i
        WHERE i.offering_date IS NOT NULL
          AND i.offering_amt IS NOT NULL
          AND i.bond_type IN %(bond_types)s
          AND i.convertible = 'N'
          AND i.asset_backed = 'N'
          AND i.foreign_currency = 'N'{exchange_offer_clause}
    ),
    first_moody AS (
        SELECT DISTINCT ON (r.issue_id)
               r.issue_id,
               r.rating,
               r.rating_date
        FROM fisd.fisd_ratings AS r
        WHERE r.rating_type = 'MR'
          AND r.rating IS NOT NULL
        ORDER BY r.issue_id, r.rating_date
    )
    SELECT c.issue_id,
           c.offering_amt,
           c.offering_date,
           fm.rating,
           iss.sic_code,
           iss.country_domicile
    FROM corp AS c
    LEFT JOIN first_moody AS fm
           ON c.issue_id = fm.issue_id
    LEFT JOIN fisd.fisd_mergedissuer AS iss
           ON c.issuer_id = iss.issuer_id
"""

# Candidate column names FISD has used to flag exchange offers, in preference
# order. Greenwood and Hanson (2013) exclude exchange offers; whichever of these
# the local FISD vintage exposes is used, and if none is present the exclusion
# is skipped (with a warning) rather than erroring.
_EXCHANGE_OFFER_COLUMNS = ("exchange_offer", "exchangeable", "exchange")


def _exchange_offer_clause(db):
    """Return a SQL predicate excluding exchange offers, adapted to the FISD
    vintage. Returns an empty string (and warns) if no known column exists."""
    try:
        cols = set(db.describe_table("fisd", "fisd_mergedissue")["name"])
    except Exception:  # noqa: BLE001 - fall back to no predicate on any error
        cols = set()
    for col in _EXCHANGE_OFFER_COLUMNS:
        if col in cols:
            return f"\n          AND i.{col} = 'N'"
    warnings.warn(
        "No exchange-offer column found in fisd.fisd_mergedissue; exchange "
        "offers will NOT be excluded. Greenwood and Hanson (2013) exclude them, "
        "so the high-yield share may be slightly biased.",
        stacklevel=2,
    )
    return ""


def _clean_fisd_issues(raw, max_year=None):
    """Turn the raw FISD query result into the tidy frame ``compute_hy_share``
    expects (one row per issue with ``year``, ``offering_amt``, ``high_yield``,
    ``nonfinancial``).

    The SQL already returns one row per issue with its first Moody's rating, so
    this step only applies the sample filters:

    - **Rated issues only.** Issues with no Moody's rating are dropped, so the
      high-yield share is the fraction of *rated* nonfinancial issuance that is
      speculative grade. Keeping unrated issues in the denominator would bias
      the share downward.
    - **U.S. issuers only**, via ``country_domicile``.
    - **Plausible offering years only.** FISD contains a small number of
      malformed or forward-dated offering dates (the raw column spans 1894 to
      2030), which would otherwise create spurious years.
    """
    if max_year is None:
        max_year = END_DATE.year

    issues = raw.copy()
    issues["offering_date"] = pd.to_datetime(issues["offering_date"], errors="coerce")
    issues["year"] = issues["offering_date"].dt.year

    # Rated issues only (see docstring).
    issues = issues.dropna(subset=["rating"])
    issues["high_yield"] = issues["rating"].map(is_high_yield)

    sic = pd.to_numeric(issues["sic_code"], errors="coerce").astype("float64")
    # `between` on a nullable dtype propagates NA, so resolve it explicitly:
    # an issue whose issuer has no SIC code is treated as nonfinancial (kept),
    # rather than being dropped from the sample.
    is_financial = sic.between(6000, 6999).fillna(False).astype(bool)
    issues["nonfinancial"] = ~is_financial

    domicile = issues["country_domicile"].fillna("USA").str.upper()
    issues = issues.loc[domicile.isin({"USA", "US", "UNITED STATES"})]

    issues = issues.loc[issues["year"].between(1900, max_year)]

    return issues[["year", "offering_amt", "high_yield", "nonfinancial"]]


def pull_hy_share_from_fisd(wrds_username=WRDS_USERNAME):
    """Reconstruct the annual high-yield share from Mergent FISD via WRDS.

    Requires the ``wrds`` package and valid WRDS credentials (set
    ``WRDS_USERNAME`` in your ``.env`` file). Returns the annual HYS frame from
    :func:`compute_hy_share`.
    """
    try:
        import wrds
    except ImportError as exc:  # pragma: no cover - depends on optional dep
        raise ImportError(
            "The 'wrds' package is required to pull the high-yield share from "
            "Mergent FISD. Install it (`pip install wrds`) and set WRDS_USERNAME "
            "in your .env file, or use source='historical' to fall back to the "
            "published Greenwood-Hanson series (1926-2008, no WRDS needed)."
        ) from exc

    db = wrds.Connection(wrds_username=wrds_username or None)
    try:
        sql = _FISD_SQL.format(exchange_offer_clause=_exchange_offer_clause(db))
        raw = db.raw_sql(sql, params={"bond_types": tuple(_CORPORATE_BOND_TYPES)})
    finally:
        db.close()

    issues = _clean_fisd_issues(raw)
    return compute_hy_share(issues)


# ---------------------------------------------------------------------------
# Source 3: the published Greenwood-Hanson historical series (1926-2008)
# ---------------------------------------------------------------------------

# The pre-1983 high-yield share cannot be pulled from any database: it comes
# from printed NBER studies. Greenwood and Hanson (2013) publish the full
# spliced annual series in Table 2 of "Issuer Quality and Corporate Bond
# Returns" (Review of Financial Studies 26(6), 1483-1525). Their splice is:
#   1926-1943  Hickman (1960), Table V2
#   1944-1965  Atkinson (1967), Table B-1
#   1966-1982  hand-collected from Moody's Bond Surveys
#   1983-2008  Mergent FISD (the same source `pull_hy_share_from_fisd` uses)
#
# These are historical facts, not a copyrightable work, and are reproduced here
# because they are the canonical series the target paper (Lopez-Salido, Stein,
# and Zakrajsek 2017) relies on for years before FISD coverage begins, and they
# exist only in print. Values are the dollar fraction of nonfinancial corporate
# bond issuance rated high yield by Moody's (Ba1/BB+ or lower). They are pulled
# from their published location (see `pull_greenwood_hanson_historical` below)
# rather than re-derived here.


# Historical Greenwood-Hanson high-yield share, pulled from its published
# location. The URL is configurable via GH_HISTORICAL_URL and points at
# InvestorCreditSentiment.xlsx on the HBS Behavioral Finance & Financial
# Stability site.
GH_HISTORICAL_URL_DEFAULT = (
    "https://www.hbs.edu/behavioral-finance-and-financial-stability/"
    "Documents/ChartData/LineCharts/InvestorCreditSentiment.xlsx"
)


def pull_greenwood_hanson_historical(url=None, sheet="Annual Data"):
    """Download the published Greenwood-Hanson high-yield share (1926-2008).

    Pulled from its published location (the HBS ``InvestorCreditSentiment``
    workbook), mirroring ``pull_shiller``. Reads the ``Annual Data`` sheet's
    ``HYS`` column and returns a frame indexed by ``year`` with ``hy_share``,
    ``ln_hy_share`` and ``source`` ('gh2013').

    Note: this is a newer vintage than the paper's Table II, so some values
    differ slightly from the originally published series.
    """
    from io import BytesIO

    import requests

    if url is None:
        try:
            url = config("GH_HISTORICAL_URL")
        except Exception:
            url = GH_HISTORICAL_URL_DEFAULT

    headers = {"User-Agent": "Mozilla/5.0 (compatible; research script)"}
    resp = requests.get(url, headers=headers, timeout=60)
    resp.raise_for_status()

    raw = pd.read_excel(BytesIO(resp.content), sheet_name=sheet)
    raw.columns = [str(c).strip().lower() for c in raw.columns]
    val = next(
        (c for c in ("hy_share", "hys", "high yield share") if c in raw.columns), None
    )
    if "year" not in raw.columns or val is None:
        raise ValueError(
            f"Expected a 'year' and HYS column from {url}; got {list(raw.columns)}"
        )
    out = raw[["year", val]].rename(columns={val: "hy_share"}).dropna()
    out["year"] = out["year"].astype(int)
    out = out.set_index("year").sort_index()
    out.index.name = "year"
    out["ln_hy_share"] = np.log(out["hy_share"].where(out["hy_share"] > 0))
    out["source"] = "gh2013"
    return out


def splice_hy_share(fisd=None, first_fisd_year=2009, historical=None):
    """Splice the published historical series with the FISD reconstruction.

    Produces a single continuous high-yield share running from 1926 to whatever
    the FISD reconstruction covers, so the series spans the full sample of
    Lopez-Salido, Stein, and Zakrajsek (2017) rather than starting in the 1980s.

    The published Greenwood-Hanson values are used through ``first_fisd_year - 1``
    (they already incorporate FISD for 1983-2008 using the authors' exact
    filters), and the locally reconstructed FISD series is appended from
    ``first_fisd_year`` onward. Defaulting the handoff to 2009 means every year
    the original paper covers comes straight from the published series, and only
    the post-2008 extension relies on the local reconstruction.

    Parameters
    ----------
    fisd : pandas.DataFrame, optional
        Output of :func:`pull_hy_share_from_fisd` / :func:`compute_hy_share`.
        If ``None``, only the historical series is returned.
    first_fisd_year : int
        First year to take from ``fisd`` rather than the published series.
    historical : pandas.DataFrame, optional
        Override the historical series (defaults to the published GH table).

    Returns
    -------
    pandas.DataFrame
        Indexed by ``year`` with ``hy_share``, ``ln_hy_share`` and ``source``
        ('gh2013' or 'fisd') marking where each year came from.

    Examples
    --------
    >>> import pandas as pd, numpy as np
    >>> historical = pd.DataFrame(
    ...     {"hy_share": [0.20, 0.25]},
    ...     index=pd.Index([2007, 2008], name="year"),
    ... )
    >>> historical["ln_hy_share"] = np.log(historical["hy_share"])
    >>> historical["source"] = "gh2013"
    >>> fisd = pd.DataFrame(
    ...     {"hy_share": [0.30, 0.35]},
    ...     index=pd.Index([2009, 2010], name="year"),
    ... )
    >>> fisd["ln_hy_share"] = np.log(fisd["hy_share"])
    >>> full = splice_hy_share(fisd=fisd, historical=historical)
    >>> int(full.index.min()), int(full.index.max())
    (2007, 2010)
    >>> full.loc[2008, "source"], full.loc[2009, "source"]
    ('gh2013', 'fisd')
    """
    if historical is None:
        historical = pull_greenwood_hanson_historical()
    hist = historical.loc[historical.index < first_fisd_year].copy()

    if fisd is None:
        return hist

    tail = fisd.loc[fisd.index >= first_fisd_year, ["hy_share", "ln_hy_share"]].copy()
    tail["source"] = "fisd"

    full = pd.concat([hist[["hy_share", "ln_hy_share", "source"]], tail])
    full = full.sort_index()
    full.index.name = "year"
    return full


def pull_greenwood_hanson(source=GH_HYS_SOURCE, first_fisd_year=2009, **kwargs):
    """Obtain the annual Greenwood-Hanson high-yield share.

    Parameters
    ----------
    source : {"spliced", "fisd", "historical"}
        ``"spliced"`` (recommended for the full sample) returns the published
        1926-2008 series spliced with the local FISD reconstruction for later
        years, covering 1929 onward as the paper requires;
        ``"fisd"`` reconstructs the series from Mergent FISD via WRDS (1983+);
        ``"historical"`` returns only the published 1926-2008 series (no WRDS
        needed).
    first_fisd_year : int
        For ``source="spliced"``, the first year taken from the FISD
        reconstruction rather than the published series.
    **kwargs
        Forwarded to the FISD puller.
    """
    if source == "historical":
        return pull_greenwood_hanson_historical()
    if source == "spliced":
        fisd = pull_hy_share_from_fisd(**kwargs)
        return splice_hy_share(fisd=fisd, first_fisd_year=first_fisd_year)
    if source == "fisd":
        return pull_hy_share_from_fisd(**kwargs)
    raise ValueError("`source` must be 'spliced', 'fisd', or 'historical'.")


def load_greenwood_hanson(data_dir=PROCESSED_DATA_DIR):
    """Load the cached, final combined high-yield-share data from the
    ``_data/processed_data`` directory.

    Must first run this module as ``__main__`` to pull and save the data.
    """
    file_path = Path(data_dir) / "greenwood_hanson_hys.parquet"
    return pd.read_parquet(file_path)


def _write_markdown_dictionary(file_path, overview_lines, column_descriptions, df):
    """Shared helper that writes a "## Overview" + "## Column Dictionary"
    Markdown file, used by the `save_data_dictionary_*` functions below."""
    lines = [
        "## Overview",
        "",
        *overview_lines,
        "",
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


def save_data_dictionary_historical(df, data_dir=DATA_DICTIONARY_DIR):
    """Write a Markdown data dictionary for the published Greenwood-Hanson
    historical series (``greenwood_hanson_hys_historical.parquet``)."""
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    overview = [
        "- **File:** `_data/raw_data/greenwood_hanson_hys_historical.parquet`",
        "- **Source:** Greenwood, Robin, and Samuel G. Hanson (2013), "
        '"Issuer Quality and Corporate Bond Returns," *Review of Financial '
        "Studies* 26(6), 1483-1525, Table 2. A new vintage of the published series is pulled from the [HBS](https://www.hbs.edu/behavioral-finance-and-financial-stability/"
        "Documents/ChartData/LineCharts/InvestorCreditSentiment.xlsx).",
        "- **Pulled by:** `pull_greenwood_hanson.py`",
        "- **Frequency:** Annual, 1926-2008",
        "- **Index:** `year`",
    ]
    return _write_markdown_dictionary(
        filedir / "greenwood_hanson_hys_historical_dictionary.md",
        overview,
        _HISTORICAL_COLUMN_DESCRIPTIONS,
        df,
    )


def save_data_dictionary_fisd(df, data_dir=DATA_DICTIONARY_DIR):
    """Write a Markdown data dictionary for the Mergent FISD reconstruction
    of the high-yield share (``greenwood_hanson_hys_fisd.parquet``)."""
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    overview = [
        "- **File:** `_data/raw_data/greenwood_hanson_hys_fisd.parquet`",
        "- **Source:** Mergent FISD (Fixed Income Securities Database), via WRDS",
        "- **Pulled by:** `pull_greenwood_hanson.py`",
        "- **Frequency:** Annual, effectively from the early 1980s onward",
        "- **Index:** `year`",
    ]
    return _write_markdown_dictionary(
        filedir / "greenwood_hanson_hys_fisd_dictionary.md",
        overview,
        _FISD_COLUMN_DESCRIPTIONS,
        df,
    )


def save_data_dictionary_combined(df, data_dir=DATA_DICTIONARY_DIR):
    """Write a Markdown data dictionary for the final spliced high-yield
    share used in the replication (``greenwood_hanson_hys.parquet``)."""
    filedir = Path(data_dir)
    filedir.mkdir(parents=True, exist_ok=True)
    overview = [
        "- **File:** `_data/processed_data/greenwood_hanson_hys.parquet`",
        "- **Source:** Spliced from `greenwood_hanson_hys` "
        "(1926-2008) and `mergent_fisd_data` (2009 onward).",
        "- **Generated by:** `pull_greenwood_hanson.py`",
        "- **Frequency:** Annual, continuous 1926-present",
        "- **Index:** `year`",
    ]
    return _write_markdown_dictionary(
        filedir / "greenwood_hanson_hys_dictionary.md",
        overview,
        _COMBINED_COLUMN_DESCRIPTIONS,
        df,
    )


if __name__ == "__main__":
    raw_dir = Path(RAW_DATA_DIR)
    raw_dir.mkdir(parents=True, exist_ok=True)
    processed_dir = Path(PROCESSED_DATA_DIR)
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Source 3: always cache the published historical series (needs no WRDS;
    # pulled straight from the manual transcription in MANUAL_DATA_DIR).
    historical = pull_greenwood_hanson_historical()
    historical.to_parquet(raw_dir / "greenwood_hanson_hys_historical.parquet")
    historical.to_csv(raw_dir / "greenwood_hanson_hys_historical.csv")
    save_data_dictionary_historical(historical, DATA_DICTIONARY_DIR)

    # Source 1: the FISD reconstruction and the full spliced series require
    # WRDS. If that is unavailable, still leave the historical (1926-2008)
    # series in place as the combined output.
    try:
        fisd = pull_hy_share_from_fisd()
        fisd.to_parquet(raw_dir / "greenwood_hanson_hys_fisd.parquet")
        fisd.to_csv(raw_dir / "greenwood_hanson_hys_fisd.csv")
        save_data_dictionary_fisd(fisd, DATA_DICTIONARY_DIR)

        full = splice_hy_share(fisd=fisd, historical=historical)
    except Exception as exc:  # noqa: BLE001 - want a clear message, keep going
        print(f"FISD pull unavailable ({exc}); writing historical series only.")
        full = historical

    # Trim the processed series to the replication sample's start year (1929);
    # the raw historical/FISD parquet files above are left uncut.
    full = full.loc[full.index >= PROCESSED_START_DATE.year]

    full.to_parquet(processed_dir / "greenwood_hanson_hys.parquet")
    full.to_csv(processed_dir / "greenwood_hanson_hys.csv")
    save_data_dictionary_combined(full, DATA_DICTIONARY_DIR)
