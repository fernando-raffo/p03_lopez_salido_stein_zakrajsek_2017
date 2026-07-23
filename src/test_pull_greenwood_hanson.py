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
        "hy_share",
        "ln_hy_share",
    ]


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


def test_pull_hy_share_from_raw_issue_level(tmp_path):
    raw = pd.DataFrame(
        {
            "year": [1995, 1995, 1996],
            "offering_amt": [100.0, 100.0, 100.0],
            "rating": ["Ba1", "AAA", "BBB-"],  # HY, IG, IG
        }
    )
    path = tmp_path / "gh_high_yield_share_raw.csv"
    raw.to_csv(path, index=False)

    out = gh.pull_hy_share_from_raw(raw_path=path)
    assert out.loc[1995, "hy_share"] == pytest.approx(0.5)
    assert out.loc[1996, "hy_share"] == pytest.approx(0.0)


def test_pull_hy_share_from_raw_preaggregated(tmp_path):
    raw = pd.DataFrame(
        {
            "year": [2001, 2002],
            "hy_issuance": [30.0, 10.0],
            "total_issuance": [100.0, 100.0],
        }
    )
    path = tmp_path / "agg.csv"
    raw.to_csv(path, index=False)

    out = gh.pull_hy_share_from_raw(raw_path=path)
    assert out.loc[2001, "hy_share"] == pytest.approx(0.3)
    assert out.loc[2002, "hy_share"] == pytest.approx(0.1)


def test_pull_hy_share_from_raw_missing_file(tmp_path):
    with pytest.raises(FileNotFoundError):
        gh.pull_hy_share_from_raw(raw_path=tmp_path / "does_not_exist.csv")


def test_pull_greenwood_hanson_bad_source():
    with pytest.raises(ValueError):
        gh.pull_greenwood_hanson(source="nonsense")


def test_load_greenwood_hanson_invalid_dir():
    with pytest.raises(FileNotFoundError):
        gh.load_greenwood_hanson(data_dir="invalid_directory")
