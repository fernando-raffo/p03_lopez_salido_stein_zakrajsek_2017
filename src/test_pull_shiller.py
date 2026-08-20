"""
Unit tests for `pull_shiller`, the Shiller stock-market data module.
Builds a small in-memory workbook mimicking `ie_data.xls`'s layout
so the parser can be tested without hitting the network.
"""

from io import BytesIO

import numpy as np
import pandas as pd
import pytest

import pull_shiller


def _make_fake_shiller_workbook(n_months=27):
    """Build an in-memory workbook that mimics the layout of ie_data.xls.

    Includes a few header rows and reproduces the quirk that October is encoded
    as ``YYYY.1`` in the raw ``Date`` column, so we can verify the parser
    rebuilds the monthly index from row order rather than trusting that column.
    """
    header = [
        [
            "Robert Shiller data",
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
            None,
        ],
        [None] * 13,
        [
            "Date",
            "P",
            "D",
            "E",
            "CPI",
            "Fraction",
            "Rate GS10",
            "Real Price",
            "Real Div",
            "Real TR",
            "Real Earnings",
            "Real TR Scaled",
            "CAPE",
        ],
    ]
    rows = []
    for k in range(n_months):
        year = 1871 + k // 12
        month = k % 12 + 1
        date_fraction = year + month / 100.0  # October (10) -> .1, a real quirk
        pe10 = float(k + 1)  # deterministic, easy to check
        rows.append(
            [
                date_fraction,
                4.0,
                0.2,
                0.4,
                12.0,
                year + 0.04,
                5.3,
                80.0,
                4.0,
                90.0,
                8.0,
                8.5,
                pe10,
            ]
        )
    # A trailing notes row that must be ignored by the parser.
    notes = ["Note: data compiled by R. Shiller"] + [None] * 12

    frame = pd.DataFrame(header + rows + [notes])
    buffer = BytesIO()
    with pd.ExcelWriter(buffer, engine="openpyxl") as writer:
        frame.to_excel(writer, sheet_name="Data", header=False, index=False)
    buffer.seek(0)
    return buffer


def test_parse_shiller_data_sheet_structure():
    df = pull_shiller.parse_shiller_data_sheet(_make_fake_shiller_workbook(27))

    assert isinstance(df, pd.DataFrame)
    # Header and trailing notes rows are dropped; 27 monthly rows remain.
    assert len(df) == 27
    # Monthly index rebuilt from row order, starting Jan 1871.
    assert df.index.min() == pd.Timestamp("1871-01-01")
    assert df.index[12] == pd.Timestamp("1872-01-01")
    # Expected clean columns are present.
    for col in ["sp500_price", "dividend", "earnings", "cpi", "gs10", "pe10"]:
        assert col in df.columns
    # pe10 was set to row-number + 1, so the 13th month should be 13.
    assert df["pe10"].iloc[12] == pytest.approx(13.0)


def test_process_shiller_annual():
    idx = pd.date_range("2000-01-01", periods=24, freq="MS")
    monthly = pd.DataFrame({"pe10": np.arange(1.0, 25.0)}, index=idx)
    monthly.index.name = "date"

    annual = pull_shiller.process_shiller_annual(monthly, how="last")
    # Year-end (December) values: month 12 -> 12.0, month 24 -> 24.0.
    assert annual["pe10"].tolist() == [12.0, 24.0]
    assert np.isclose(annual["ln_pe10"].iloc[-1], np.log(24.0))

    annual_mean = pull_shiller.process_shiller_annual(monthly, how="mean")
    assert np.isclose(annual_mean["pe10"].iloc[0], np.arange(1.0, 13.0).mean())


def test_process_shiller_annual_drops_incomplete_trailing_year_last():
    # 2000 has all 12 months; 2001 only has data through June.
    idx = pd.date_range("2000-01-01", periods=18, freq="MS")
    monthly = pd.DataFrame({"pe10": np.arange(1.0, 19.0)}, index=idx)
    monthly.index.name = "date"

    annual = pull_shiller.process_shiller_annual(monthly, how="last")

    assert annual.index.year.tolist() == [2000]
    assert annual["pe10"].tolist() == [12.0]


def test_process_shiller_annual_drops_incomplete_trailing_year_mean():
    # 2000 has all 12 months; 2001 only has data through June.
    idx = pd.date_range("2000-01-01", periods=18, freq="MS")
    monthly = pd.DataFrame({"pe10": np.arange(1.0, 19.0)}, index=idx)
    monthly.index.name = "date"

    annual = pull_shiller.process_shiller_annual(monthly, how="mean")

    assert annual.index.year.tolist() == [2000]
    assert np.isclose(annual["pe10"].iloc[0], np.arange(1.0, 13.0).mean())


def test_process_shiller_annual_drops_year_missing_december_mid_gap():
    # A year that has 11 months but is missing December specifically (not
    # just a trailing partial year) must still be dropped for how="last".
    idx = pd.date_range("2000-01-01", periods=11, freq="MS")  # Jan-Nov 2000
    monthly = pd.DataFrame({"pe10": np.arange(1.0, 12.0)}, index=idx)
    monthly.index.name = "date"

    annual = pull_shiller.process_shiller_annual(monthly, how="last")

    assert annual.empty


def test_process_shiller_annual_bad_how():
    monthly = pd.DataFrame(
        {"pe10": [1.0, 2.0]},
        index=pd.date_range("2000-01-01", periods=2, freq="MS"),
    )
    with pytest.raises(ValueError):
        pull_shiller.process_shiller_annual(monthly, how="median")


def test_load_shiller_invalid_dir():
    with pytest.raises(FileNotFoundError):
        pull_shiller.load_shiller(data_dir="invalid_directory")


def test_save_data_dictionary_documents_known_and_unknown_columns(tmp_path):
    df = pd.DataFrame({"pe10": [1.0], "SOME_UNKNOWN_COLUMN": [2.0]})

    file_path = pull_shiller.save_data_dictionary(df, data_dir=tmp_path)

    assert file_path == tmp_path / "shiller_data_dictionary.md"
    text = file_path.read_text()
    assert "pe10" in text
    assert pull_shiller._RAW_COLUMN_DESCRIPTIONS["pe10"] in text
    assert "SOME_UNKNOWN_COLUMN" in text
    assert "Unknown series" in text


def test_save_data_dictionary_annual_documents_known_and_unknown_columns(tmp_path):
    df = pd.DataFrame({"ln_pe10": [1.0], "SOME_UNKNOWN_COLUMN": [2.0]})

    file_path = pull_shiller.save_data_dictionary_annual(df, data_dir=tmp_path)

    assert file_path == tmp_path / "shiller_data_annual_dictionary.md"
    text = file_path.read_text()
    assert "ln_pe10" in text
    assert pull_shiller._ANNUAL_COLUMN_DESCRIPTIONS["ln_pe10"] in text
    assert "SOME_UNKNOWN_COLUMN" in text
    assert "Unknown series" in text
