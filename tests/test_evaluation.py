"""Unit tests for the masked evaluation metrics on synthetic collections."""
import numpy as np
import pytest

from skyflux.baselines import DailyDataset, GeneratedDay
from skyflux.evaluation import (
    DayCollection,
    autocorrelation_k,
    daily_energy,
    distribution_summary,
    nearest_training_distances,
    physical_checks,
    ramp_magnitudes,
)


def make_collection(k_rows: list, valid_width: int = 4, start: int = 2, g: float = 500.0):
    """Days sharing one mask; k_rows supplies values on the valid slots."""
    n = len(k_rows)
    valid = np.zeros((n, 144), dtype=bool)
    valid[:, start : start + valid_width] = True
    k = np.full((n, 144), np.nan)
    for i, row in enumerate(k_rows):
        k[i, start : start + valid_width] = row
    g_clear = np.where(valid, g, 0.0)
    dates = np.arange(
        np.datetime64("2024-06-01"), np.datetime64("2024-06-01") + np.timedelta64(n, "D")
    ).astype("datetime64[D]")
    return DayCollection(dates=dates, k=k, valid=valid, g_clear_cal=g_clear)


def test_distribution_summary_quantiles():
    summary = distribution_summary(np.array([1.0, 2.0, 3.0, 4.0]))
    assert summary["count"] == 4
    assert summary["median"] == 2.5
    assert summary["min"] == 1.0
    assert summary["max"] == 4.0
    assert distribution_summary(np.array([np.nan]))["count"] == 0


def test_daily_energy_sums_valid_slots_only():
    collection = make_collection([[1.0, 1.0, 1.0, 1.0], [2.0, 2.0, 2.0, 2.0]])
    energy = daily_energy(collection)
    # 4 slots x 500 W/m^2 x 600 s = 1.2e6 J/m^2 per unit of k.
    np.testing.assert_allclose(energy, [1.2e6, 2.4e6])


def test_ramp_magnitudes_skip_invalid_pairs():
    # k pattern translated into G via g=500: ramps are |dk| * 500.
    collection = make_collection([[0.2, 0.6, 0.2, 0.6]])
    np.testing.assert_allclose(ramp_magnitudes(collection), [200.0, 200.0, 200.0])

    # A middle invalid slot must not create a two-slot ramp.
    collection.k[0, 3] = np.nan
    collection.valid[0, 3] = False
    np.testing.assert_allclose(ramp_magnitudes(collection), [200.0])


def test_autocorrelation_alternating_series():
    collection = make_collection([[0.2, 0.8] * 2])
    lags = autocorrelation_k(collection, max_lag=2)
    assert lags[1] == pytest.approx(-1.0)
    assert lags[2] == pytest.approx(1.0)


def test_physical_checks_count_artifacts():
    collection = make_collection([[0.5, -0.1, 1.2, 1.06]])
    checks = physical_checks(collection)
    assert checks["fraction_k_negative"] == 0.25
    assert checks["fraction_k_gt_1"] == 0.5
    assert checks["fraction_k_gt_1_05"] == 0.5
    assert checks["max_k"] == 1.2
    # g = 500 -> max GHI 600 W/m^2, below the audit marker.
    assert checks["count_ghi_above_audit_marker"] == 0
    assert checks["max_ghi_wm2"] == 600.0


def test_nearest_training_distances_detect_duplicates_and_offsets():
    training = make_collection(
        [[0.4, 0.4, 0.4, 0.4], [0.8, 0.8, 0.8, 0.8], [0.6, 0.2, 0.6, 0.2]]
    )
    duplicate = make_collection([[0.8, 0.8, 0.8, 0.8]])
    offset = make_collection([[0.9, 0.9, 0.9, 0.9]])  # 0.1 from the 0.8 day
    dup_distance = nearest_training_distances(duplicate, training)
    off_distance = nearest_training_distances(offset, training)
    assert dup_distance[0] == 0.0  # memorized training day
    assert abs(off_distance[0] - 0.1) < 1e-12

    # Mask intersection: a narrower generated day still matches on overlap.
    narrow = make_collection([[0.8, 0.8]], valid_width=2)
    assert nearest_training_distances(narrow, training)[0] == 0.0


def test_generated_collection_from_days():
    valid = np.zeros(144, dtype=bool)
    valid[10:20] = True
    g_clear = np.where(valid, 700.0, 0.0).astype(np.float32)
    day = GeneratedDay(
        date=np.datetime64("2024-07-01"),
        k=np.full(144, 1.0, dtype=np.float32),
        daylight=valid,
        g_clear_cal=g_clear,
    )
    collection = DayCollection.from_generated([day])
    assert collection.k.shape == (1, 144)
    assert np.isnan(collection.k[0, :10]).all()  # outside the mask: NaN
    assert np.allclose(collection.k[0][valid], 1.0)
    ghi = collection.ghi()[0]
    np.testing.assert_allclose(ghi[valid], 700.0)
    assert np.isnan(ghi[~valid]).all()


def test_from_dataset_preserves_mask_semantics():
    from tests.test_baselines import make_dataset

    dataset = make_dataset(n_days=3)
    collection = DayCollection.from_dataset(dataset)
    assert np.array_equal(np.isfinite(collection.k), dataset.valid)
    assert collection.k.dtype == np.float64
