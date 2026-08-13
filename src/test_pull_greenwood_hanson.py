import numpy as np
import pandas as pd
import pytest

import pull_greenwood_hanson as gh
from settings import config

DATA_DIR = config("DATA_DIR")


def test_is_high_yield():
    # Below investment grade -> True
    assert gh.is_high_yield("Ba1")
    assert gh.is_high_yield("BB+")
    assert gh.is_high_yield("Caa2")
    # Investment grade -> False
    assert not gh.is_high_yield("Baa3")
    assert not gh.is_high_yield("AAA")
    assert not gh.is_high_yield("A2")
    # Unrated / missing -> False
    assert not gh.is_high_yield(None)
    assert not gh.is_high_yield(float("nan"))
    assert not gh.is_high_yield("NR")
    assert not gh.is_high_yield("")


def test_compute_hy_share_basic():
    issues = pd.DataFrame(
        {
            "year": [1990, 1990, 1990, 1991, 1991],
            "offering_amt": [100.0, 300.0, 100.0, 50.0, 50.0],
            "high_yield": [True, False, True, False, True],
        }
    )
    out = gh.compute_hy_share(issues)
    assert out.loc[1990, "hy_share"] == pytest.approx(0.4)  # 200 / 500
    assert out.loc[1991, "hy_share"] == pytest.approx(0.5)  # 50 / 100
    assert np.isclose(out.loc[1990, "ln_hy_share"], np.log(0.4))
    assert list(out.columns) == [
        "hy_issuance",
        "total_issuance",
        "n_issues",
        "hy_share",
        "ln_hy_share",
    ]
    assert out.loc[1990, "n_issues"] == 3


def test_compute_hy_share_nonfinancial_filter():
    issues = pd.DataFrame(
        {
            "year": [2000, 2000, 2000],
            "offering_amt": [100.0, 100.0, 100.0],
            "high_yield": [True, False, True],
            "nonfinancial": [True, True, False],  # last row is a financial issuer
        }
    )
    out = gh.compute_hy_share(issues)
    # Only the two nonfinancial rows count: 100 HY / 200 total = 0.5
    assert out.loc[2000, "total_issuance"] == pytest.approx(200.0)
    assert out.loc[2000, "hy_share"] == pytest.approx(0.5)


def test_compute_hy_share_zero_hy():
    issues = pd.DataFrame(
        {
            "year": [2010, 2010],
            "offering_amt": [10.0, 20.0],
            "high_yield": [False, False],
        }
    )
    out = gh.compute_hy_share(issues)
    assert out.loc[2010, "hy_share"] == pytest.approx(0.0)
    # ln(0) is undefined and stored as NaN rather than -inf.
    assert np.isnan(out.loc[2010, "ln_hy_share"])


def test_compute_hy_share_nullable_dtypes():
    """WRDS/psycopg2 return pandas nullable dtypes, where a missing SIC code
    makes `between` evaluate to NA rather than False. Guards against the
    'cannot convert float NaN to bool' failure that caused."""
    issues = pd.DataFrame(
        {
            "year": pd.array([2000, 2000, 2000], dtype="Int64"),
            "offering_amt": pd.array([100.0, 100.0, 100.0], dtype="Float64"),
            "high_yield": pd.array([True, False, True], dtype="boolean"),
            "nonfinancial": pd.array([True, True, None], dtype="boolean"),
        }
    )
    out = gh.compute_hy_share(issues)
    # The NA row is treated as nonfinancial and kept: 200 HY / 300 total.
    assert out.loc[2000, "total_issuance"] == pytest.approx(300.0)
    assert out.loc[2000, "hy_share"] == pytest.approx(2 / 3)


def test_clean_fisd_issues_missing_sic():
    """A missing SIC code should not drop the issue or raise."""
    raw = pd.DataFrame(
        {
            "issue_id": [1, 2],
            "offering_amt": pd.array([100.0, 100.0], dtype="Float64"),
            "offering_date": ["1995-03-01", "1995-06-01"],
            "rating": ["Ba1", "Aaa"],
            "sic_code": pd.array([None, "6021"], dtype="string"),
            "country_domicile": ["USA", "USA"],
        }
    )
    out = gh._clean_fisd_issues(raw, max_year=2026)
    assert len(out) == 2
    assert bool(out["nonfinancial"].iloc[0]) is True  # missing SIC -> kept
    assert bool(out["nonfinancial"].iloc[1]) is False  # 6021 -> financial


def test_pull_greenwood_hanson_bad_source():
    with pytest.raises(ValueError):
        gh.pull_greenwood_hanson(source="nonsense")


def test_load_greenwood_hanson_invalid_dir():
    with pytest.raises(FileNotFoundError):
        gh.load_greenwood_hanson(data_dir="invalid_directory")


def test_save_data_dictionary_historical(tmp_path):
    df = pd.DataFrame(
        {"hy_share": [0.2], "SOME_UNKNOWN_COLUMN": [1.0]}, index=[1990]
    )

    file_path = gh.save_data_dictionary_historical(df, data_dir=tmp_path)

    assert file_path == tmp_path / "greenwood_hanson_hys_historical_dictionary.md"
    text = file_path.read_text()
    assert "hy_share" in text
    assert gh._HISTORICAL_COLUMN_DESCRIPTIONS["hy_share"] in text
    assert "SOME_UNKNOWN_COLUMN" in text
    assert "Unknown series" in text


def test_save_data_dictionary_fisd(tmp_path):
    df = pd.DataFrame(
        {"n_issues": [10], "SOME_UNKNOWN_COLUMN": [1.0]}, index=[2010]
    )

    file_path = gh.save_data_dictionary_fisd(df, data_dir=tmp_path)

    assert file_path == tmp_path / "greenwood_hanson_hys_fisd_dictionary.md"
    text = file_path.read_text()
    assert "n_issues" in text
    assert gh._FISD_COLUMN_DESCRIPTIONS["n_issues"] in text
    assert "SOME_UNKNOWN_COLUMN" in text
    assert "Unknown series" in text


def test_save_data_dictionary_combined(tmp_path):
    df = pd.DataFrame(
        {"source": ["fisd"], "SOME_UNKNOWN_COLUMN": [1.0]}, index=[2010]
    )

    file_path = gh.save_data_dictionary_combined(df, data_dir=tmp_path)

    assert file_path == tmp_path / "greenwood_hanson_hys_dictionary.md"
    text = file_path.read_text()
    assert "source" in text
    assert gh._COMBINED_COLUMN_DESCRIPTIONS["source"] in text
    assert "SOME_UNKNOWN_COLUMN" in text
    assert "Unknown series" in text
