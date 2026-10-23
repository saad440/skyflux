"""Masked evaluation metrics for daily irradiance profiles.

All metrics operate on a ``DayCollection`` — days of 144 slots with a
validity mask, observed or generated — and are computed only where the
mask is true, so night geometry and missing data never contaminate
statistics. Generated days are evaluated on their reference date's mask,
which keeps real and generated comparisons on the same support
conventions. The layer is model-agnostic: the VAE can be evaluated with
the same functions.
"""
from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from .baselines import DailyDataset, GeneratedDay

# DWD audit marker for implausibly high irradiance; not a hard limit.
GHI_AUDIT_MARKER_WM2 = 1500.0
SLOT_SECONDS = 600.0


@dataclass(frozen=True)
class DayCollection:
    """Days of 144 slots; ``k`` is NaN outside ``valid`` for both sources."""

    dates: np.ndarray  # (n,) datetime64[D]
    k: np.ndarray  # (n, 144) float64, NaN outside valid
    valid: np.ndarray  # (n, 144) bool
    g_clear_cal: np.ndarray  # (n, 144) float64

    @classmethod
    def from_dataset(cls, dataset: DailyDataset) -> "DayCollection":
        """Observed days from a dataset subset (e.g. the validation year)."""
        return cls(
            dates=dataset.dates,
            k=dataset.k.astype(np.float64),
            valid=dataset.valid,
            g_clear_cal=dataset.g_clear_cal.astype(np.float64),
        )

    @classmethod
    def from_generated(cls, days: Sequence[GeneratedDay]) -> "DayCollection":
        """Generated days, evaluated on their reference dates' masks."""
        n = len(days)
        k = np.full((n, 144), np.nan)
        valid = np.zeros((n, 144), dtype=bool)
        g_clear = np.zeros((n, 144))
        dates = np.empty(n, dtype="datetime64[D]")
        for i, day in enumerate(days):
            dates[i] = day.date
            valid[i] = day.daylight
            k[i] = np.where(day.daylight, day.k, np.nan)
            g_clear[i] = day.g_clear_cal
        return cls(dates=dates, k=k, valid=valid, g_clear_cal=g_clear)

    def ghi(self) -> np.ndarray:
        """W/m^2: k * G_clear_cal on valid slots, NaN elsewhere."""
        return np.where(self.valid, self.k * self.g_clear_cal, np.nan)


def distribution_summary(values: np.ndarray) -> dict:
    """Quantile/mean/std summary of finite values."""
    values = np.asarray(values, dtype=np.float64)
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    quantiles = np.percentile(values, [1, 5, 25, 50, 75, 95, 99])
    return {
        "count": int(values.size),
        "mean": float(values.mean()),
        "std": float(values.std()),
        "p01": float(quantiles[0]),
        "p05": float(quantiles[1]),
        "p25": float(quantiles[2]),
        "median": float(quantiles[3]),
        "p75": float(quantiles[4]),
        "p95": float(quantiles[5]),
        "p99": float(quantiles[6]),
        "min": float(values.min()),
        "max": float(values.max()),
    }


def daily_energy(collection: DayCollection) -> np.ndarray:
    """Daily irradiation in J/m^2: sum of G * 600 s over valid slots.

    Slots outside the validity mask contribute nothing; days with missing
    daylight slots would be underestimated (included days in this project
    are complete within daylight, so this does not arise in practice).
    """
    ghi = np.where(collection.valid, collection.k * collection.g_clear_cal, 0.0)
    return ghi.sum(axis=1) * SLOT_SECONDS


def ramp_magnitudes(collection: DayCollection) -> np.ndarray:
    """|dG| in W/m^2 between consecutive valid slots within each day.

    Pairs spanning an invalid slot or a day boundary are excluded, so data
    gaps can never appear as physical ramps.
    """
    ghi = collection.ghi()
    both = collection.valid[:, :-1] & collection.valid[:, 1:]
    diffs = np.abs(np.diff(ghi, axis=1))
    return diffs[both]


def autocorrelation_k(collection: DayCollection, max_lag: int = 6) -> dict:
    """Pooled Pearson autocorrelation of k at 1..max_lag slots (10-60 min).

    Correlated over all within-day valid slot pairs separated by the lag;
    pooling across days and dates means mild diurnal and seasonal structure
    enters both real and generated collections the same way. Lags are in
    slots of 10 minutes.
    """
    result = {}
    for lag in range(1, max_lag + 1):
        left = collection.k[:, :-lag]
        right = collection.k[:, lag:]
        both = collection.valid[:, :-lag] & collection.valid[:, lag:]
        x, y = left[both], right[both]
        if x.size < 2:
            result[lag] = None
            continue
        x_centered = x - x.mean()
        y_centered = y - y.mean()
        denominator = np.sqrt(float(np.sum(x_centered**2) * np.sum(y_centered**2)))
        result[lag] = (
            float(np.sum(x_centered * y_centered) / denominator) if denominator > 0 else None
        )
    return result


def physical_checks(collection: DayCollection) -> dict:
    """Fractions of physically implausible model output (never clipped)."""
    k = collection.k[collection.valid]
    ghi = collection.ghi()[collection.valid]
    return {
        "fraction_k_negative": float(np.mean(k < 0.0)) if k.size else None,
        "fraction_k_gt_1": float(np.mean(k > 1.0)) if k.size else None,
        "fraction_k_gt_1_05": float(np.mean(k > 1.05)) if k.size else None,
        "max_k": float(np.max(k)) if k.size else None,
        "count_ghi_above_audit_marker": int(np.sum(ghi > GHI_AUDIT_MARKER_WM2)) if ghi.size else 0,
        "max_ghi_wm2": float(np.max(ghi)) if ghi.size else None,
    }


def nearest_training_distances(
    collection: DayCollection, training: DayCollection
) -> np.ndarray:
    """Per-day nearest-neighbour RMS k distance to the training days.

    Distance between a generated day and a training day is the RMS
    difference of k over the intersection of their validity masks, so
    seasonal mask differences cannot inflate the distance. Zero-distance
    days are memorized training days (expected for the bootstrap);
    the distance from real held-out days to training provides the scale of
    natural day-to-day novelty for comparison.
    """
    distances = np.full(len(collection.dates), np.nan)
    for i in range(len(collection.dates)):
        intersection = collection.valid[i][None, :] & training.valid
        diffs = np.where(intersection, collection.k[i][None, :] - training.k, 0.0)
        counts = intersection.sum(axis=1)
        with np.errstate(invalid="ignore", divide="ignore"):
            rms = np.sqrt((diffs**2).sum(axis=1) / np.maximum(counts, 1))
        rms[counts == 0] = np.inf
        distances[i] = float(np.min(rms))
    return distances


def summarize_collection(
    collection: DayCollection, training: DayCollection | None = None
) -> dict:
    """Full metric block for one collection, optionally with diversity."""
    energy = daily_energy(collection)
    summary = {
        "n_days": int(len(collection.dates)),
        "n_valid_slots": int(collection.valid.sum()),
        "k": distribution_summary(collection.k[collection.valid]),
        "ghi_wm2": distribution_summary(collection.ghi()[collection.valid]),
        "daily_energy_j_m2": distribution_summary(energy),
        "daily_energy_kwh_m2": distribution_summary(energy / 3.6e6),
        "abs_ramp_wm2_per_slot": distribution_summary(ramp_magnitudes(collection)),
        "autocorrelation_k": autocorrelation_k(collection),
        "physical": physical_checks(collection),
    }
    if training is not None:
        nn = nearest_training_distances(collection, training)
        summary["nearest_training_rms_k"] = distribution_summary(nn)
        summary["fraction_duplicate_training_days"] = float(np.mean(nn < 1e-3))
    return summary
