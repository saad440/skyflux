"""Unit tests for the daily-representation layer and the bootstrap baseline."""
import numpy as np
import pytest

from skyflux.baselines import (
    BootstrapSampler,
    DailyDataset,
    GeneratedDay,
    PPCA,
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


def bump_shape(lo: int = 30, hi: int = 110) -> np.ndarray:
    """Zero-mean profile bump supported on slots lo..hi (for rank-1 tests)."""
    shape = np.zeros(144)
    shape[lo:hi] = np.sin(np.linspace(0.0, np.pi, hi - lo))
    return shape


def make_rank_one_dataset(n_days: int, seed: int, valid: np.ndarray) -> DailyDataset:
    """k = 0.6 + a_i * shape + tiny noise on the given validity support."""
    rng = np.random.default_rng(seed)
    shape = bump_shape()
    amplitude = rng.uniform(-0.3, 0.3, n_days)
    k = np.where(valid, 0.6 + amplitude[:, None] * shape[None, :], np.nan)
    noise = rng.normal(0.0, 0.01, (n_days, 144))
    k = np.where(valid, k + noise, np.nan).astype(np.float32)
    dates = np.arange(
        np.datetime64("2021-06-01"), np.datetime64("2021-06-01") + np.timedelta64(n_days, "D")
    ).astype("datetime64[D]")
    g_clear = np.where(valid, 600.0, 0.0).astype(np.float32)
    return DailyDataset(
        dates=dates,
        k=k,
        valid=valid,
        g_clear_cal=g_clear,
        include_day=np.ones(n_days, dtype=bool),
        split_id=np.zeros(n_days, dtype=np.int8),
    )


def test_ppca_recovers_rank_one_direction():
    base = make_dataset(n_days=80)
    dataset = make_rank_one_dataset(80, seed=3, valid=base.valid)
    model = PPCA(n_components=1, seed=0, max_iter=400).fit(dataset)
    window = np.arange(30, 110)
    direction = model.W[window, 0] / np.linalg.norm(model.W[window, 0])
    target = bump_shape()[window]
    target = target / np.linalg.norm(target)
    assert abs(float(direction @ target)) > 0.99  # direction recovered up to sign
    assert model.sigma2 < 1e-3  # residual noise scale, not structure
    assert model.n_iterations <= 400


def test_ppca_ties_days_with_varying_daylight_masks():
    # Widths 40-89 centered near noon: the always-observed core is 52..92.
    n_days = 60
    valid = np.zeros((n_days, 144), dtype=bool)
    for i in range(n_days):
        width = 40 + (i * 53) % 50
        start = 72 - width // 2
        valid[i, start : start + width] = True
    dataset = make_rank_one_dataset(n_days, seed=4, valid=valid)
    model = PPCA(n_components=2, seed=0, max_iter=400).fit(dataset)
    core = np.arange(52, 93)
    target = bump_shape(52, 93)[core]
    target /= np.linalg.norm(target)
    # The signal direction must appear in one of the two components.
    alignments = []
    for column in range(model.n_components):
        direction = model.W[core, column] / np.linalg.norm(model.W[core, column])
        alignments.append(abs(float(direction @ target)))
    assert max(alignments) > 0.9


def test_ppca_fit_and_sampling_are_reproducible():
    base = make_dataset(n_days=50)
    dataset = make_rank_one_dataset(50, seed=5, valid=base.valid)
    first = PPCA(n_components=2, seed=11).fit(dataset)
    second = PPCA(n_components=2, seed=11).fit(dataset)
    np.testing.assert_array_equal(first.W, second.W)
    assert first.sigma2 == second.sigma2
    assert first.log_likelihood == second.log_likelihood

    reference = make_dataset(n_days=9)
    days_a = first.sample(15, reference)
    days_b = second.sample(15, reference)
    assert [d.date for d in days_a] == [d.date for d in days_b]
    for a, b in zip(days_a, days_b):
        np.testing.assert_array_equal(a.k, b.k)

    other_seed = PPCA(n_components=2, seed=12).fit(dataset)
    assert not np.array_equal(first.W, other_seed.W)


def test_ppca_sample_contract():
    base = make_dataset(n_days=40)
    dataset = make_rank_one_dataset(40, seed=6, valid=base.valid)
    model = PPCA(n_components=2, seed=0, max_iter=200).fit(dataset)
    reference = make_dataset(n_days=9)
    days = model.sample(25, reference)
    assert len(days) == 25
    reference_dates = set(reference.dates.tolist())
    for day in days:
        assert day.k.shape == (144,)
        assert day.k.dtype == np.float32
        assert np.isfinite(day.k).all()  # raw output finite on every slot
        assert day.date.astype("datetime64[D]").tolist() in reference_dates
    noisy = model.sample(25, reference, include_noise=True)
    assert all(np.isfinite(day.k).all() for day in noisy)

    with pytest.raises(ValueError, match="fitted"):
        PPCA(n_components=2).sample(3, reference)
    with pytest.raises(ValueError, match="fitted"):
        PPCA(n_components=2).latent_variances()
    empty = DailyDataset(
        dates=reference.dates[:0],
        k=reference.k[:0],
        valid=reference.valid[:0],
        g_clear_cal=reference.g_clear_cal[:0],
        include_day=reference.include_day[:0],
        split_id=reference.split_id[:0],
    )
    with pytest.raises(ValueError, match="empty"):
        model.sample(3, empty)
    with pytest.raises(ValueError, match="empty"):
        PPCA(n_components=2).fit(empty)

    variances = model.latent_variances()
    assert variances.shape == (2,)
    assert (variances[:-1] >= variances[1:]).all()  # descending
