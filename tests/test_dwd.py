"""Unit tests for ``skyflux.dwd`` on synthetic inputs.

Covers unit conversion, archive discovery, station-list parsing, ZIP parsing
and the midpoint/missing/quality handling of ``normalize_observations``.
Network access and real DWD archives are deliberately not exercised.
"""
from io import BytesIO
from zipfile import ZipFile

import pandas as pd
import pytest

from skyflux.dwd import (
    FACTOR_JCM2_TO_WM2,
    discover_archive,
    extract_station_metadata,
    normalize_observations,
    read_solar_zip,
)


def test_conversion():
    assert FACTOR_JCM2_TO_WM2 == pytest.approx(16.666666666666668)
    assert 60 * FACTOR_JCM2_TO_WM2 == pytest.approx(1000)


def test_archive_discovery():
    html = '<a href="10minutenwerte_SOLAR_00342_20200101_20251231_hist.zip">archive</a>'
    assert discover_archive("00342", html) == "10minutenwerte_SOLAR_00342_20200101_20251231_hist.zip"
    with pytest.raises(ValueError):
        discover_archive("00183", html)


def test_station_metadata_latin1():
    data = "00342 20101202 20251231            103     52.3170    8.1694 Belm                                     Niedersachsen\n".encode("latin-1")
    result = extract_station_metadata(data, "00342")
    assert result["station_name"] == "Belm"
    assert result["longitude"] == pytest.approx(8.1694)
    assert result["elevation_m"] == 103


def test_midpoint_utc_and_missing_quality():
    rows = pd.DataFrame({
        "MESS_DATUM": ["202001010010", "202001010020", "202001010030", "202001010040"],
        "QN": [3, 3, 1, 3],
        "GS_10": ["60", "-999", "20", "-1"],
    })
    grid, daily, stats = normalize_observations(rows, 2020, 2020)
    first = pd.Timestamp("2020-01-01T00:05:00Z")
    assert grid.loc[first, "ghi_wm2"] == pytest.approx(1000)
    assert grid.loc[first, "timestamp_end_utc"] == pd.Timestamp("2020-01-01T00:10:00Z")
    assert pd.isna(grid.loc[first + pd.Timedelta(minutes=10), "ghi_wm2"])
    assert stats["missing_value_sentinel"] == 1
    assert stats["negative_non_sentinel"] == 1
    assert stats["accepted_qn2_or_qn3_measurements"] == 1
    assert not daily.iloc[0]["candidate_day"]


def test_zip_parser(tmp_path):
    archive = tmp_path / "data.zip"
    with ZipFile(archive, "w") as zf:
        zf.writestr("produkt_zehn_min_sd_20200101_20251231_00342.txt",
                    "STATIONS_ID;MESS_DATUM;QN;GS_10;DS_10\n00342;202001010010;3;60;20\n")
    frame = read_solar_zip(archive)
    assert frame.iloc[0]["GS_10"] == 60
