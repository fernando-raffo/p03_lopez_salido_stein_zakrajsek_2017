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
GH_HYS_SOURCE = _config_or("GH_HYS_SOURCE", "fisd")
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

# One row per corporate issue, carrying the *first Moody's* rating assigned to
# that issue (the rating at issuance). Notes on the joins:
#   - fisd_ratings holds ~4.4m rows because every rating action is a row, so we
#     collapse to one rating per issue with DISTINCT ON (PostgreSQL).
#   - rating_type = 'MR' selects Moody's, matching Greenwood and Hanson (2013).
#   - the `investment_grade` column is left unused: it is NULL for the large
#     majority of rows in this vintage, so grade is inferred from the letter
#     rating instead (see :func:`is_high_yield`).
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
          AND i.foreign_currency = 'N'
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
        raw = db.raw_sql(_FISD_SQL, params={"bond_types": tuple(_CORPORATE_BOND_TYPES)})
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


def pull_greenwood_hanson(source=GH_HYS_SOURCE, **kwargs):
    """Obtain the annual Greenwood-Hanson high-yield share.

    Parameters
    ----------
    source : {"fisd", "raw"}
        ``"fisd"`` reconstructs the series from Mergent FISD via WRDS;
        ``"raw"`` builds it from a manual issuance file in ``MANUAL_DATA_DIR``.
    **kwargs
        Forwarded to the underlying puller.
    """
    if source == "fisd":
        return pull_hy_share_from_fisd(**kwargs)
    if source == "raw":
        return pull_hy_share_from_raw(**kwargs)
    raise ValueError("`source` must be 'fisd' or 'raw'.")


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
    df = pull_greenwood_hanson(GH_HYS_SOURCE)

    filedir = Path(DATA_DIR)
    filedir.mkdir(parents=True, exist_ok=True)
    df.to_parquet(filedir / "greenwood_hanson_hys.parquet")
    df.to_csv(filedir / "greenwood_hanson_hys.csv")
