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
        issues = issues.loc[issues["nonfinancial"].astype(bool)]

    issues["offering_amt"] = pd.to_numeric(issues["offering_amt"], errors="coerce")
    issues["high_yield"] = issues["high_yield"].astype(bool)
    issues = issues.dropna(subset=["year", "offering_amt"])
    issues["year"] = issues["year"].astype(int)

    grouped = issues.groupby("year")
    total = grouped["offering_amt"].sum().rename("total_issuance")
    hy = (
        issues.loc[issues["high_yield"]]
        .groupby("year")["offering_amt"]
        .sum()
        .reindex(total.index, fill_value=0.0)
        .rename("hy_issuance")
    )

    out = pd.concat([hy, total], axis=1)
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
_FISD_SQL = """
    SELECT i.issue_id,
           i.offering_amt,
           i.offering_date,
           r.rating,
           r.rating_type,
           iss.sic_code,
           iss.country_domicile
    FROM fisd.fisd_mergedissue AS i
    LEFT JOIN fisd.fisd_mergedissuer AS iss
           ON i.issuer_id = iss.issuer_id
    LEFT JOIN fisd.fisd_ratings AS r
           ON i.issue_id = r.issue_id
    WHERE i.offering_date IS NOT NULL
      AND i.offering_amt IS NOT NULL
      AND i.bond_type NOT IN ('TXMU', 'USBN', 'USBL', 'ABS', 'CMO')
      AND i.convertible = 'N'
      AND i.foreign_currency = 'N'
"""


def _clean_fisd_issues(raw):
    """Turn the raw FISD query result into the tidy frame ``compute_hy_share``
    expects (one row per issue with ``year``, ``offering_amt``, ``high_yield``,
    ``nonfinancial``)."""
    raw = raw.copy()
    raw["offering_date"] = pd.to_datetime(raw["offering_date"], errors="coerce")

    # Keep the first (earliest) rating per issue as the "rating at issuance".
    raw = raw.sort_values(["issue_id", "offering_date"])
    first_rating = raw.dropna(subset=["rating"]).groupby("issue_id")["rating"].first()

    issues = (
        raw.drop(columns=["rating", "rating_type"])
        .drop_duplicates(subset="issue_id")
        .set_index("issue_id")
    )
    issues["rating"] = first_rating
    issues["year"] = issues["offering_date"].dt.year
    issues["high_yield"] = issues["rating"].map(is_high_yield)

    sic = pd.to_numeric(issues["sic_code"], errors="coerce")
    issues["nonfinancial"] = ~sic.between(6000, 6999)

    domicile = issues["country_domicile"].fillna("USA").str.upper()
    issues = issues.loc[domicile.isin({"USA", "US", "UNITED STATES"})]

    return issues.reset_index()[["year", "offering_amt", "high_yield", "nonfinancial"]]


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
        raw = db.raw_sql(_FISD_SQL)
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
