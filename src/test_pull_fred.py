"""
Unit tests for `pull_fred`, the FRED data-pulling module. `DataReader`
itself is monkeypatched, so these need no network access.
"""

import pandas as pd
import pytest

import pull_fred


def test_pull_fred_calls_datareader_with_all_series(monkeypatch):
    captured = {}

    def fake_datareader(series_ids, source, start, end):
        captured["series_ids"] = series_ids
        captured["source"] = source
        captured["start"] = start
        captured["end"] = end
        return pd.DataFrame({sid: [1.0, 2.0] for sid in series_ids})

    monkeypatch.setattr(pull_fred.web, "DataReader", fake_datareader)

    df = pull_fred.pull_fred(start_date="2000-01-01", end_date="2000-12-31")

    assert isinstance(df, pd.DataFrame)
    assert captured["source"] == "fred"
    assert captured["start"] == "2000-01-01"
    assert captured["end"] == "2000-12-31"
    assert captured["series_ids"] == list(pull_fred.series_to_pull.keys())
    assert list(df.columns) == list(pull_fred.series_to_pull.keys())


def test_load_fred_reads_cached_parquet(tmp_path):
    fake_df = pd.DataFrame(
        {"AAA": [1.0, 2.0], "BAA": [3.0, 4.0]},
        index=pd.to_datetime(["2000-01-01", "2000-02-01"]),
    )
    fake_df.to_parquet(tmp_path / "fred.parquet")

    result = pull_fred.load_fred(data_dir=tmp_path)

    pd.testing.assert_frame_equal(result, fake_df)


def test_load_fred_missing_file_raises(tmp_path):
    with pytest.raises(FileNotFoundError):
        pull_fred.load_fred(data_dir=tmp_path / "does_not_exist")


def test_save_data_dictionary_documents_known_and_unknown_columns(tmp_path):
    df = pd.DataFrame({"AAA": [1.0], "SOME_UNKNOWN_SERIES": [2.0]})

    file_path = pull_fred.save_data_dictionary(df, data_dir=tmp_path)

    assert file_path == tmp_path / "fred_data_dictionary.md"
    text = file_path.read_text()
    assert "AAA" in text
    assert pull_fred.series_to_pull["AAA"] in text
    assert "SOME_UNKNOWN_SERIES" in text
    assert "Unknown series" in text
