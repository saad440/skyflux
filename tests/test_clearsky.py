"""Unit tests for ``skyflux.clearsky`` on synthetic inputs.

Exercises the cloud-factor masking, the daily 144-slot status encoding, the
daylight-based inclusion screen and the chronological splits without network
access. One test evaluates the real pvlib Ineichen model at the Belm site,
which works offline because the Linke turbidity climatology ships with pvlib.
"""
import numpy as np
import pandas as pd
import pytest

from skyflux.clearsky import (
    DAYLIGHT_G_CLEAR_THRESHOLD_WM2,
    STATUS_DAYLIGHT_MISSING,
    STATUS_DAYLIGHT_VALID,
    STATUS_NIGHT,
    build_daily_arrays,
    chronological_splits,
    clear_day_ratio_diagnostic,
    clear_sky_ghi,
    cloud_factor,
    daily_inclusion,
    daylight_mask,
    k_statistics,
    monthly_k_table,
)


def synthetic_two_days():
    """Two UTC days of clear-sky-like, daylight, GHI and k series."""
    index = pd.date_range("2020-06-01 00:05", "2020-06-02 23:55", freq="10min", tz="UTC")
    minute_of_day = index.hour * 60 + index.minute
    # Cosine bump peaking at 12:00 UTC; zero outside a ~13 h window.
    phase = (minute_of_day - 720) / 720 * np.pi * 0.9
    g_clear = pd.Series(np.maximum(0.0, 700.0 * np.cos(phase)), index=index)
    daylight = daylight_mask(g_clear)
    ghi = g_clear * 0.8
    ghi[ghi < DAYLIGHT_G_CLEAR_THRESHOLD_WM2] = 0.0  # measured night zeros
    # One daylight slot missing, one daylight slot with a measured zero.
    missing_slot = pd.Timestamp("2020-06-01 09:35Z")
    zero_slot = pd.Timestamp("2020-06-02 10:05Z")
    ghi[missing_slot] = np.nan
    ghi[zero_slot] = 0.0
    return index, ghi, g_clear, daylight, missing_slot, zero_slot


def test_cloud_factor_only_on_valid_daylight():
    index = pd.date_range("2020-06-01 12:05", periods=4, freq="10min", tz="UTC")
    g_clear = pd.Series([800.0, 600.0, 10.0, 500.0], index=index)
    ghi = pd.Series([1000.0, np.nan, 400.0, 0.0], index=index)
    daylight = pd.Series([True, True, False, True], index=index)
    k = cloud_factor(ghi, g_clear, daylight)
    assert k.iloc[0] == pytest.approx(1.25)  # cloud enhancement preserved
    assert pd.isna(k.iloc[1])  # missing measurement
    assert pd.isna(k.iloc[2])  # below daylight threshold: ratio never formed
    assert k.iloc[3] == 0.0  # measured zero is a valid k of zero


def test_daily_array_encoding_distinguishes_slot_kinds():
    index, ghi, g_clear, daylight, missing_slot, zero_slot = synthetic_two_days()
    arrays = build_daily_arrays(index, ghi, g_clear, daylight)

    assert arrays["k"].shape == arrays["ghi"].shape == arrays["g_clear"].shape
    assert arrays["status"].shape == (2, 144)
    assert arrays["dates"][0] == np.datetime64("2020-06-01", "D")
    assert arrays["k"].dtype == np.float32
    assert arrays["status"].dtype == np.int8

    status = arrays["status"]
    k = arrays["k"]
    # k is finite exactly on valid daylight slots (measured zeros included).
    assert np.array_equal(np.isfinite(k), status == STATUS_DAYLIGHT_VALID)
    assert np.all(np.isnan(k[status == STATUS_DAYLIGHT_MISSING]))
    assert np.all(np.isnan(k[status == STATUS_NIGHT]))

    slot = lambda stamp: stamp.hour * 6 + stamp.minute // 10
    assert status[0, slot(missing_slot)] == STATUS_DAYLIGHT_MISSING
    assert np.isnan(k[0, slot(missing_slot)])
    assert status[1, slot(zero_slot)] == STATUS_DAYLIGHT_VALID
    assert k[1, slot(zero_slot)] == pytest.approx(0.0)

    # Reference retained everywhere; measured night zeros kept; night is night.
    assert np.isfinite(arrays["g_clear"]).all()
    night_columns = status == STATUS_NIGHT
    assert np.all(arrays["ghi"][night_columns] == 0.0)
    assert np.all(arrays["g_clear"][night_columns] < DAYLIGHT_G_CLEAR_THRESHOLD_WM2)


def test_build_daily_arrays_rejects_duplicates():
    index, ghi, g_clear, daylight, *_ = synthetic_two_days()
    duplicated = pd.DatetimeIndex(index.append(index[:1]))  # repeat one midpoint
    with pytest.raises(ValueError):
        build_daily_arrays(
            duplicated,
            pd.concat([ghi, ghi.iloc[:1]]),
            pd.concat([g_clear, g_clear.iloc[:1]]),
            pd.concat([daylight, daylight.iloc[:1]]),
        )


def test_build_daily_arrays_rejects_missing_slot_and_skipped_day():
    index, ghi, g_clear, daylight, *_ = synthetic_two_days()
    # One dropped daylight timestamp must not be silently encoded as night.
    keep = index != pd.Timestamp("2020-06-01 09:35Z")
    with pytest.raises(ValueError, match="complete unique"):
        build_daily_arrays(index[keep], ghi[keep], g_clear[keep], daylight[keep])

    # A skipped whole day violates the contiguous-day grid contract.
    third_day = pd.date_range("2020-06-03 00:05", "2020-06-03 23:55", freq="10min", tz="UTC")
    index3 = index.append(third_day)

    def extend(series):
        return pd.concat([series, pd.Series(series.iloc[:144].to_numpy(), index=third_day)])

    keep3 = np.asarray(index3.day != 2)
    with pytest.raises(ValueError, match="contiguous"):
        build_daily_arrays(
            index3[keep3], extend(ghi)[keep3], extend(g_clear)[keep3], extend(daylight)[keep3]
        )


def test_daylight_inclusion_screen():
    status = np.full((3, 144), STATUS_NIGHT, dtype=np.int8)
    status[0, :100] = STATUS_DAYLIGHT_VALID  # 100% valid daylight -> included
    status[1, :100] = STATUS_DAYLIGHT_VALID
    status[1, 97:100] = STATUS_DAYLIGHT_MISSING  # 97/100 -> below 98%
    status[2, :] = STATUS_NIGHT  # degenerate all-night day -> excluded
    result = daily_inclusion(status)
    assert result.loc[0, "include_day"]
    assert not result.loc[1, "include_day"]
    assert result.loc[1, "valid_daylight_fraction"] == pytest.approx(0.97)
    assert not result.loc[2, "include_day"]


def test_chronological_splits_cover_window_exactly():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D").to_numpy(dtype="datetime64[D]")
    masks = chronological_splits(dates)
    sizes = {name: int(mask.sum()) for name, mask in masks.items()}
    assert sizes == {"train": 1461, "validation": 366, "test": 365}  # leap years 2020, 2024
    stacked = np.column_stack(list(masks.values()))
    assert stacked.sum(axis=1).min() == 1  # every day in exactly one split
    with pytest.raises(ValueError):
        chronological_splits(np.append(dates, np.datetime64("2026-01-01", "D")))


def test_chronological_splits_reject_overlapping_ranges():
    dates = pd.date_range("2020-01-01", "2025-12-31", freq="D").to_numpy(dtype="datetime64[D]")
    overlapping = {
        "train": (2020, 2023),
        "validation": (2022, 2024),  # 2022-2023 belong to two splits
        "test": (2025, 2025),
    }
    with pytest.raises(ValueError, match="overlapping"):
        chronological_splits(dates, splits=overlapping)


def test_clear_day_ratio_diagnostic_flags_low_reference():
    index, ghi, g_clear, daylight, *_ = synthetic_two_days()
    # Bias day 1 upwards (clear, reference too low) and day 2 downwards.
    ghi[index.day == 1] = (g_clear * 1.2).where(daylight, 0.0)[index.day == 1]
    result = clear_day_ratio_diagnostic(ghi, g_clear, daylight, days_per_year=1)
    assert result["n_days"] == 1  # selects only the clearest day
    assert result["median_ratio"] == pytest.approx(1.2, abs=1e-6)


def test_clear_day_ratio_diagnostic_ignores_night_observations():
    index, ghi, g_clear, daylight, *_ = synthetic_two_days()
    # Day 1: clear (ratio 1.2) with ordinary night zeros. Day 2: lower
    # daylight peaks but absurd night values; night observations must not be
    # able to win the clearest-day selection.
    ghi[index.day == 1] = (g_clear * 1.2).where(daylight, 0.0)[index.day == 1]
    ghi[index.day == 2] = (g_clear * 0.6).where(daylight, 3000.0)[index.day == 2]
    result = clear_day_ratio_diagnostic(ghi, g_clear, daylight, days_per_year=1)
    assert result["n_days"] == 1
    assert result["median_ratio"] == pytest.approx(1.2, abs=1e-6)


def test_clear_sky_ghi_zeroes_night_and_rejects_daytime_nan(monkeypatch):
    import pvlib

    times = pd.date_range("2020-06-01 00:05", periods=144, freq="10min", tz="UTC")
    real = pvlib.location.Location.get_clearsky

    def with_daytime_nan(self, midpoints, model=None):
        out = real(self, midpoints, model=model).copy()
        out.loc[times[72], "ghi"] = np.nan  # 12:05 UTC, sun well up in June
        return out

    monkeypatch.setattr(pvlib.location.Location, "get_clearsky", with_daytime_nan)
    with pytest.raises(ValueError, match="NaN"):
        clear_sky_ghi(times, latitude=52.317, longitude=8.1694, elevation_m=103)

    def with_night_nan(self, midpoints, model=None):
        out = real(self, midpoints, model=model).copy()
        out.loc[times[0], "ghi"] = np.nan  # 00:05 UTC, sun below the horizon
        return out

    monkeypatch.setattr(pvlib.location.Location, "get_clearsky", with_night_nan)
    series = clear_sky_ghi(times, latitude=52.317, longitude=8.1694, elevation_m=103)
    assert series.loc[times[0]] == 0.0
    assert series.notna().all()


def test_ineichen_clear_sky_belml():
    july = pd.date_range("2020-07-01 00:05", periods=144, freq="10min", tz="UTC")
    december = pd.date_range("2020-12-21 00:05", periods=144, freq="10min", tz="UTC")
    g_clear = clear_sky_ghi(
        july.append(december), latitude=52.317, longitude=8.1694, elevation_m=103
    )
    assert g_clear.notna().all()
    assert (g_clear >= 0).all()
    july = g_clear.loc["2020-07-01"]
    december = g_clear.loc["2020-12-21"]
    # Physically required magnitudes for a 52 N site: summer noon well above
    # winter noon, both in a plausible clear-sky range, night exactly zero.
    assert july.max() > 700
    assert 100 < december.max() < 350
    assert july.max() > december.max()
    assert g_clear[pd.Timestamp("2020-12-21 02:05Z")] == 0.0
    assert (july >= DAYLIGHT_G_CLEAR_THRESHOLD_WM2).sum() > (
        december >= DAYLIGHT_G_CLEAR_THRESHOLD_WM2
    ).sum()


def test_k_statistics_and_monthly_table():
    values = np.array([0.5, 1.0, 1.2, np.nan, 3.0])
    stats = k_statistics(values)
    assert stats["count"] == 4
    assert stats["fraction_k_gt_1"] == pytest.approx(0.5)
    assert stats["fraction_k_gt_1_05"] == pytest.approx(0.5)
    assert stats["median"] == pytest.approx(1.1)
    assert stats["max"] == pytest.approx(3.0)

    index = pd.DatetimeIndex(
        pd.date_range("2020-06-01 12:05", periods=2, freq="10min", tz="UTC").append(
            pd.date_range("2020-07-01 12:05", periods=1, freq="10min", tz="UTC")
        )
    )
    k = pd.Series([1.2, np.nan, 0.9], index=index)
    table = monthly_k_table(index, k)
    assert table.loc[6, "n"] == 1
    assert table.loc[6, "frac_k_gt_1"] == pytest.approx(1.0)
    assert table.loc[7, "median_k"] == pytest.approx(0.9)
