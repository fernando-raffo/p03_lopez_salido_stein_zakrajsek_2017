"""Pull, read, load, and process the Greenwood-Hanson high-yield share (HYS).

The *high-yield share* (``HYS``) is the fraction of gross nonfinancial
corporate bond issuance in a given year that is rated below investment grade
("high yield" / "junk"). Greenwood and Hanson (2013, *Review of Financial
Studies*) show it forecasts corporate-bond returns; Lopez-Salido, Stein, and
Zakrajsek (2017) use ``ln(HYS)_{t-2}`` as a first-step predictor of future
changes in the Baa-Treasury credit spread (see the auxiliary regressions in
Tables II and V, where "HYS_t = fraction of debt that is rated as high yield
(Greenwood and Hanson 2013)").

Where the data comes from
-------------------------
There is no free public CSV/API for the HYS. It is *constructed* from
bond-level issuance data. The authoritative source is **Mergent FISD** (Fixed
Income Securities Database), accessed through **WRDS** (the project's
``.env.example`` already anticipates a ``WRDS_USERNAME``). This module builds
the series from FISD; it also supports processing a raw issuance file (e.g. an
export of the Greenwood-Hanson replication data, or SIFMA investment-grade /
high-yield issuance totals) placed in ``MANUAL_DATA_DIR`` for anyone without
WRDS access.

Construction logic (following Greenwood and Hanson 2013)
--------------------------------------------------------
1. Start from gross corporate bond issues (offering amount and offering date).
2. Restrict to U.S. nonfinancial corporate issues (drop financials, SIC
   6000-6999), excluding convertibles, non-USD, asset-backed, and government /
   agency debt.
3. Assign each issue a rating at issuance and flag it high yield if it is below
   investment grade (below Baa3 / BBB-).
4. For each year, ``HYS = (high-yield issuance) / (total issuance)``.

Known limitations of the FISD reconstruction
--------------------------------------------
1. **Coverage.** FISD's usable issuance history effectively begins in the early
   1980s. Stray offering dates reach back to 1902, but those years contain a
   handful of issues and produce degenerate shares of exactly 0 or 1. The
   ``n_issues`` column is provided so these thin years can be screened out
   (e.g. ``df.loc[df.n_issues >= 25]``). This series therefore cannot cover the
   1929 start of the sample in Lopez-Salido, Stein, and Zakrajsek (2017); their
   pre-1980s high-yield share comes from other historical sources.
2. **Moody's-rated denominator.** Following Greenwood and Hanson (2013), an
   issue's grade is taken from its first Moody's rating (``rating_type = 'MR'``)
   and issues with no Moody's rating are excluded entirely. Issues rated only by
   S&P or Fitch therefore drop out of the denominator, which can bias the
   computed share upward relative to a definition that accepts any agency
   rating. Worth checking before the series is used in the first-step
   regression.

Reaching 1929: the published historical series
----------------------------------------------
Because FISD cannot reach the 1929 start of the sample, this module also ships
the published Greenwood-Hanson (2013) high-yield share for 1926-2008 (their
Table 2), whose pre-1983 values come from printed NBER studies (Hickman 1960;
Atkinson 1967) and hand-collected Moody's Bond Surveys. ``source="spliced"``
(the default) returns those published values through 2008 and appends the local
FISD reconstruction for later years, giving a continuous series that covers the
full 1929-2015 sample. ``source="historical"`` returns just the published
1926-2008 series and needs no WRDS access.

Naming conventions
------------------
- ``pull_greenwood_hanson`` obtains the data from an external source (FISD via
  WRDS, or a raw manual file) and returns an annual DataFrame.
- ``load_greenwood_hanson`` reads the cached copy from the ``_data`` directory.
- ``compute_hy_share`` is the pure aggregation step and is unit-tested with
  synthetic data (it needs no network or credentials).

Running this file as a script pulls the data and caches it to ``DATA_DIR``
(the git-ignored ``_data`` folder), so no data is ever committed to the repo.
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


DATA_DIR = Path(config("DATA_DIR"))
MANUAL_DATA_DIR = Path(config("MANUAL_DATA_DIR"))
START_DATE = config("START_DATE")
END_DATE = config("END_DATE")

# Which source to use by default: "fisd" (WRDS Mergent FISD) or "raw"
# (a manually supplied issuance file in MANUAL_DATA_DIR). Configurable via env
# var GH_HYS_SOURCE or --GH_HYS_SOURCE.
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
        max_year = pd.Timestamp.today().year

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
            "in your .env file, or use source='raw' with a manual issuance file."
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
# Source 2: a manually supplied raw issuance file (fallback, no WRDS needed)
# ---------------------------------------------------------------------------


def pull_hy_share_from_raw(raw_path=None, manual_data_dir=MANUAL_DATA_DIR):
    """Build the high-yield share from a manually supplied issuance file.

    The file may be either:

    - **issue-level** (one row per bond) with columns ``year``,
      ``offering_amt`` and either ``high_yield`` (bool) or ``rating`` (letter
      grade, converted via :func:`is_high_yield`); or
    - **pre-aggregated annual** totals with columns ``year``,
      ``hy_issuance`` and ``total_issuance`` (or directly ``hy_share``).

    Accepts ``.csv``, ``.parquet``, ``.xls`` and ``.xlsx``. This path lets
    collaborators without WRDS reproduce the series (e.g. from SIFMA
    investment-grade vs. high-yield issuance totals, or an export of the
    Greenwood-Hanson replication data).
    """
    if raw_path is None:
        raw_path = Path(manual_data_dir) / "gh_high_yield_share_raw.csv"
    raw_path = Path(raw_path)
    if not raw_path.exists():
        raise FileNotFoundError(
            f"No raw high-yield-share file found at {raw_path}. Provide one, or "
            "use source='fisd' to reconstruct it from Mergent FISD via WRDS."
        )

    suffix = raw_path.suffix.lower()
    if suffix == ".csv":
        raw = pd.read_csv(raw_path)
    elif suffix == ".parquet":
        raw = pd.read_parquet(raw_path)
    elif suffix in {".xls", ".xlsx"}:
        raw = pd.read_excel(raw_path)
    else:
        raise ValueError(f"Unsupported raw file type: {suffix}")

    cols = {c.lower(): c for c in raw.columns}
    raw = raw.rename(columns={v: k for k, v in cols.items()})

    # Case 1: already contains an annual hy_share.
    if "hy_share" in raw.columns and "year" in raw.columns:
        out = raw.set_index("year").sort_index()
        out["ln_hy_share"] = np.log(out["hy_share"].where(out["hy_share"] > 0))
        return out

    # Case 2: pre-aggregated annual issuance totals.
    if {"year", "hy_issuance", "total_issuance"}.issubset(raw.columns):
        out = raw.set_index("year").sort_index()
        out["hy_share"] = out["hy_issuance"] / out["total_issuance"]
        out["ln_hy_share"] = np.log(out["hy_share"].where(out["hy_share"] > 0))
        return out

    # Case 3: issue-level data -> aggregate.
    if "high_yield" not in raw.columns and "rating" in raw.columns:
        raw["high_yield"] = raw["rating"].map(is_high_yield)
    return compute_hy_share(raw)


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
# bond issuance rated high yield by Moody's (Ba1/BB+ or lower).
_GH2013_TABLE2_HYS = {
    1926: 0.182,
    1927: 0.177,
    1928: 0.270,
    1929: 0.262,
    1930: 0.135,
    1931: 0.108,
    1932: 0.229,
    1933: 0.639,
    1934: 0.212,
    1935: 0.150,
    1936: 0.062,
    1937: 0.129,
    1938: 0.053,
    1939: 0.261,
    1940: 0.151,
    1941: 0.045,
    1942: 0.137,
    1943: 0.104,
    1944: 0.026,
    1945: 0.044,
    1946: 0.037,
    1947: 0.007,
    1948: 0.010,
    1949: 0.023,
    1950: 0.031,
    1951: 0.023,
    1952: 0.013,
    1953: 0.011,
    1954: 0.044,
    1955: 0.076,
    1956: 0.107,
    1957: 0.077,
    1958: 0.041,
    1959: 0.146,
    1960: 0.079,
    1961: 0.056,
    1962: 0.030,
    1963: 0.082,
    1964: 0.166,
    1965: 0.210,
    1966: 0.193,
    1967: 0.214,
    1968: 0.137,
    1969: 0.141,
    1970: 0.033,
    1971: 0.081,
    1972: 0.056,
    1973: 0.038,
    1974: 0.002,
    1975: 0.002,
    1976: 0.006,
    1977: 0.062,
    1978: 0.128,
    1979: 0.099,
    1980: 0.139,
    1981: 0.132,
    1982: 0.146,
    1983: 0.217,
    1984: 0.294,
    1985: 0.353,
    1986: 0.264,
    1987: 0.424,
    1988: 0.561,
    1989: 0.394,
    1990: 0.049,
    1991: 0.074,
    1992: 0.285,
    1993: 0.336,
    1994: 0.389,
    1995: 0.258,
    1996: 0.420,
    1997: 0.496,
    1998: 0.409,
    1999: 0.306,
    2000: 0.180,
    2001: 0.200,
    2002: 0.250,
    2003: 0.395,
    2004: 0.493,
    2005: 0.391,
    2006: 0.375,
    2007: 0.337,
    2008: 0.177,
}


def load_greenwood_hanson_historical():
    """Return the published Greenwood-Hanson (2013) high-yield share, 1926-2008.

    This is the authoritative spliced series from Table 2 of Greenwood and
    Hanson (2013). Use it for the pre-FISD period (before 1983), which cannot be
    reconstructed from any database, or as a ready-made 1926-2008 series.

    Returns
    -------
    pandas.DataFrame
        Indexed by ``year`` with ``hy_share``, ``ln_hy_share`` and a ``source``
        label ('gh2013').

    Examples
    --------
    >>> h = load_greenwood_hanson_historical()
    >>> int(h.index.min()), int(h.index.max())
    (1926, 2008)
    >>> float(h.loc[1929, "hy_share"])
    0.262
    """
    out = pd.DataFrame({"hy_share": pd.Series(_GH2013_TABLE2_HYS)}).sort_index()
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
    >>> fisd = pd.DataFrame(
    ...     {"hy_share": [0.30, 0.35]},
    ...     index=pd.Index([2009, 2010], name="year"),
    ... )
    >>> fisd["ln_hy_share"] = np.log(fisd["hy_share"])
    >>> full = splice_hy_share(fisd=fisd)
    >>> int(full.index.min()), int(full.index.max())
    (1926, 2010)
    >>> full.loc[2008, "source"], full.loc[2009, "source"]
    ('gh2013', 'fisd')
    """
    if historical is None:
        historical = load_greenwood_hanson_historical()
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
    source : {"spliced", "fisd", "raw", "historical"}
        ``"spliced"`` (recommended for the full sample) returns the published
        1926-2008 series spliced with the local FISD reconstruction for later
        years, covering 1929 onward as the paper requires;
        ``"fisd"`` reconstructs the series from Mergent FISD via WRDS (1983+);
        ``"raw"`` builds it from a manual issuance file in ``MANUAL_DATA_DIR``;
        ``"historical"`` returns only the published 1926-2008 series (no WRDS
        needed).
    first_fisd_year : int
        For ``source="spliced"``, the first year taken from the FISD
        reconstruction rather than the published series.
    **kwargs
        Forwarded to the FISD puller.
    """
    if source == "historical":
        return load_greenwood_hanson_historical()
    if source == "spliced":
        fisd = pull_hy_share_from_fisd(**kwargs)
        return splice_hy_share(fisd=fisd, first_fisd_year=first_fisd_year)
    if source == "fisd":
        return pull_hy_share_from_fisd(**kwargs)
    if source == "raw":
        return pull_hy_share_from_raw(**kwargs)
    raise ValueError("`source` must be 'spliced', 'fisd', 'raw', or 'historical'.")


def load_greenwood_hanson(data_dir=DATA_DIR):
    """Load the cached high-yield-share data from the ``_data`` directory.

    Must first run this module as ``__main__`` to pull and save the data.
    """
    file_path = Path(data_dir) / "greenwood_hanson_hys.parquet"
    return pd.read_parquet(file_path)


def _demo():
    df = load_greenwood_hanson()
    print(df.tail())


if __name__ == "__main__":
    filedir = Path(DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)

    # Always cache the published historical series (needs no WRDS).
    historical = load_greenwood_hanson_historical()
    historical.to_parquet(filedir / "greenwood_hanson_hys_historical.parquet")
    historical.to_csv(filedir / "greenwood_hanson_hys_historical.csv")

    # The FISD reconstruction and the full spliced series require WRDS. If that
    # is unavailable, still leave the historical (1926-2008) series in place.
    try:
        fisd = pull_hy_share_from_fisd()
        fisd.to_parquet(filedir / "greenwood_hanson_hys_fisd.parquet")
        fisd.to_csv(filedir / "greenwood_hanson_hys_fisd.csv")

        full = splice_hy_share(fisd=fisd)
    except Exception as exc:  # noqa: BLE001 - want a clear message, keep going
        print(f"FISD pull unavailable ({exc}); writing historical series only.")
        full = historical

    full.to_parquet(filedir / "greenwood_hanson_hys.parquet")
    full.to_csv(filedir / "greenwood_hanson_hys.csv")
