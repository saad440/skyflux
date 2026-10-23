"""Unit tests for the daily-representation layer and the bootstrap baseline."""
import numpy as np
import pytest

from skyflux.baselines import (
    BootstrapSampler,
    DailyDataset,
    GeneratedDay,
    load_daily_dataset,
)


def make_dataset(n_days: int = 6) -> DailyDataset:
    """Small synthetic dataset: fixed daylight window, one distinct k per day."""
    dates = np.arange(
        np.datetime64("2020-06-01"), np.datetime64("2020-06-01") + np.timedelta64(n_days, "D")
    ).astype("datetime64[D]")
    valid = np.zeros((n_days, 144), dtype=bool)
    valid[:, 30:110] = True
    # Distinct per-day offset so sampled rows are attributable to their day.
    offsets = np.broadcast_to(
        np.linspace(0.0, 0.2, n_days, dtype=np.float32)[:, None], (n_days, 144)
    )
    k = np.where(valid, 0.7 + offsets, np.nan).astype(np.float32)
    g_clear = np.where(valid, 600.0, 0.0).astype(np.float32)
    include = np.ones(n_days, dtype=bool)
    split_id = np.zeros(n_days, dtype=np.int8)
    return DailyDataset(dates, k, valid, g_clear, include, split_id)


def write_npz(path, k_matrix, valid, dates, split_id, include=None):
    status = np.where(valid, 1, 0).astype(np.int8)
    np.savez(
        path,
        dates=dates,
        k_calibrated=k_matrix,
        status=status,
        g_clear_calibrated=np.where(valid, 500.0, 0.0).astype(np.float32),
        include_day=np.ones(len(dates), dtype=bool) if include is None else include,
        split_id=split_id,
    )


def test_load_daily_dataset_enforces_representation_contract(tmp_path):
    n_days = 4
    dates = np.arange(
        np.datetime64("2020-06-01"), np.datetime64("2020-06-01") + np.timedelta64(n_days, "D")
    ).astype("datetime64[D]")
    valid = np.zeros((n_days, 144), dtype=bool)
    valid[:, 40:100] = True
    k = np.where(valid, 0.8, np.nan).astype(np.float32)

    good = tmp_path / "good.npz"
    write_npz(good, k, valid, dates, np.zeros(n_days, dtype=np.int8))
    dataset = load_daily_dataset(good)
    assert np.array_equal(np.isfinite(dataset.k), dataset.valid)
    assert np.array_equal(dataset.valid, valid)
    assert np.isnan(dataset.k[~dataset.valid]).all()

    # A finite value on a non-valid slot must be rejected, not coerced.
    broken = k.copy()
    broken[0, 0] = 1.0  # night slot carrying a value
    bad = tmp_path / "bad.npz"
    write_npz(bad, broken, valid, dates, np.zeros(n_days, dtype=np.int8))
    with pytest.raises(ValueError, match="finite exactly"):
        load_daily_dataset(bad)

    # Split ids must agree with the year ranges of clearsky.SPLIT_YEARS.
    wrong = tmp_path / "wrong.npz"
    write_npz(wrong, k, valid, dates, np.full(n_days, 2, dtype=np.int8))  # 2020 != test 2025
    with pytest.raises(ValueError, match="test"):
        load_daily_dataset(wrong)


def test_subset_selects_included_days_of_one_split():
    dataset = make_dataset(n_days=6)
    dataset = DailyDataset(
        dates=dataset.dates,
        k=dataset.k,
        valid=dataset.valid,
        g_clear_cal=dataset.g_clear_cal,
        include_day=np.array([True, True, True, False, False, False]),
        split_id=np.array([0, 0, 1, 0, 1, 1], dtype=np.int8),
    )
    train = dataset.subset("train")
    validation = dataset.subset("validation")
    assert len(train.dates) == 2  # day 3 is train but excluded
    assert len(validation.dates) == 1  # day 6 is validation but excluded
    assert (train.split_id == 0).all()
    with pytest.raises(ValueError, match="unknown split"):
        dataset.subset("holdout")


def test_bootstrap_is_reproducible_and_stays_on_training_days():
    dataset = make_dataset(n_days=6)
    first = BootstrapSampler(dataset, seed=7).sample(50)
    second = BootstrapSampler(dataset, seed=7).sample(50)
    assert [d.date for d in first] == [d.date for d in second]
    for a, b in zip(first, second):
        np.testing.assert_array_equal(a.k, b.k)

    other = BootstrapSampler(dataset, seed=8).sample(50)
    assert [d.date for d in other] != [d.date for d in first]

    # Memorization property: every drawn profile equals some training row.
    pool = {day.tobytes() for day in dataset.k}
    assert all(day.k.tobytes() in pool for day in first)
    sampled_dates = np.array([d.date for d in first])
    assert np.isin(sampled_dates, dataset.dates).all()


def test_bootstrap_rejects_empty_dataset():
    dataset = make_dataset(n_days=2)
    empty = DailyDataset(
        dates=dataset.dates[:0],
        k=dataset.k[:0],
        valid=dataset.valid[:0],
        g_clear_cal=dataset.g_clear_cal[:0],
        include_day=dataset.include_day[:0],
        split_id=dataset.split_id[:0],
    )
    with pytest.raises(ValueError, match="empty"):
        BootstrapSampler(empty).sample(3)


def test_generated_day_irradiance_uses_only_daylight_slots():
    dataset = make_dataset(n_days=1)
    day = GeneratedDay(
        date=dataset.dates[0],
        k=np.full(144, 2.0, dtype=np.float32),  # 2.0 everywhere, night included
        daylight=dataset.valid[0],
        g_clear_cal=dataset.g_clear_cal[0],
    )
    ghi = day.irradiance()
    assert ghi.dtype == np.float32
    day_slots = dataset.valid[0]
    np.testing.assert_allclose(ghi[day_slots], 2.0 * 600.0)
    assert (ghi[~day_slots] == 0.0).all()  # night-slot output never reaches physics
    assert np.isfinite(ghi).all()
