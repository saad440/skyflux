"""Statistical baselines on the daily cloud-factor representation.

Model input contract: one sample is one UTC day, a 144-slot vector of the
calibrated cloud factor ``k_calibrated``. ``k`` is finite exactly where the
slot carries a valid daylight measurement (status 1) and NaN elsewhere
(night and missing); a parallel boolean mask records that support. Nothing
is imputed — night is structurally absent from the model input and must
never appear as ``k = 0``, which would conflate geometry with overcast
physics.

Generators model cloudiness only. A generated 144-vector becomes physical
exclusively through a reference date from the comparison period: the
vector is read on that date's valid daylight slots and irradiance is
``G = k * G_clear_calibrated`` (zero elsewhere, per the physical
reference). Geometry is therefore assigned, never learned.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .clearsky import SPLIT_YEARS, STATUS_DAYLIGHT_VALID

# Day-to-split ids used in the daily npz; matches the enumeration order of
# clearsky.SPLIT_YEARS (train, validation, test) used by prepare_daily.py.
SPLIT_IDS = {"train": 0, "validation": 1, "test": 2}


@dataclass(frozen=True)
class DailyDataset:
    """Daily k profiles with validity masks, geometry and split assignment.

    ``k`` is (n_days, 144) float32, finite exactly where ``valid`` is true.
    ``g_clear_cal`` carries the calibrated clear-sky reference for every
    slot (zero at night) so irradiance can be reconstructed from any k
    profile combined with any reference date.
    """

    dates: np.ndarray  # (n_days,) datetime64[D]
    k: np.ndarray  # (n_days, 144) float32, NaN where not valid
    valid: np.ndarray  # (n_days, 144) bool, true on valid daylight slots
    g_clear_cal: np.ndarray  # (n_days, 144) float32
    include_day: np.ndarray  # (n_days,) bool
    split_id: np.ndarray  # (n_days,) int8

    def subset(self, split: str) -> "DailyDataset":
        """Included days of one chronological split (train/validation/test)."""
        try:
            wanted = SPLIT_IDS[split]
        except KeyError:
            raise ValueError(f"unknown split {split!r}; expected one of {sorted(SPLIT_IDS)}") from None
        selected = (self.split_id == wanted) & self.include_day
        return DailyDataset(
            dates=self.dates[selected],
            k=self.k[selected],
            valid=self.valid[selected],
            g_clear_cal=self.g_clear_cal[selected],
            include_day=self.include_day[selected],
            split_id=self.split_id[selected],
        )


@dataclass(frozen=True)
class GeneratedDay:
    """A generated cloud-factor profile bound to a reference date.

    ``k`` is the raw (144,) model output; ``daylight`` and ``g_clear_cal``
    come from the reference date. Physical irradiance is available only
    through ``irradiance()``, which reads ``k`` on the reference date's
    valid daylight slots — generated values on other slots are never used,
    so a model's night-slot output cannot leak into physics.
    """

    date: np.datetime64
    k: np.ndarray  # (144,) float32
    daylight: np.ndarray  # (144,) bool
    g_clear_cal: np.ndarray  # (144,) float32

    def irradiance(self) -> np.ndarray:
        """W/m^2 per slot: k on valid daylight slots, zero elsewhere."""
        return np.where(self.daylight, self.k * self.g_clear_cal, 0.0).astype(np.float32)


def load_daily_dataset(npz_path) -> DailyDataset:
    """Load the daily npz written by ``prepare_daily.py``.

    Validates the representation contract (k finite exactly on valid
    daylight slots) and the split-id/year correspondence, so an encoding
    change upstream cannot silently propagate into the generators.
    """
    npz = np.load(npz_path)
    k = np.asarray(npz["k_calibrated"], dtype=np.float32)
    valid = np.asarray(npz["status"]) == STATUS_DAYLIGHT_VALID
    if not np.array_equal(np.isfinite(k), valid):
        raise ValueError("k_calibrated must be finite exactly on valid daylight slots")
    dates = np.asarray(npz["dates"])
    split_id = np.asarray(npz["split_id"])
    years = dates.astype("datetime64[Y]").astype(int) + 1970
    for name, identifier in SPLIT_IDS.items():
        first, last = SPLIT_YEARS[name]
        in_split = years[split_id == identifier]
        if in_split.size and (in_split.min() < first or in_split.max() > last):
            raise ValueError(f"split_id {identifier} does not match years {first}-{last} for {name!r}")
    return DailyDataset(
        dates=dates,
        k=np.where(valid, k, np.nan).astype(np.float32),
        valid=valid,
        g_clear_cal=np.asarray(npz["g_clear_calibrated"], dtype=np.float32),
        include_day=np.asarray(npz["include_day"], dtype=bool),
        split_id=split_id,
    )


class BootstrapSampler:
    """Non-novel resampling baseline over whole training days.

    Draws days with replacement; each draw carries the day's own k profile,
    date and clear-sky reference, so decoded irradiance is exact by
    construction. This is the memorization floor for distributional
    metrics and the zero-novelty reference for diversity metrics: it
    reproduces training-day distributions, ramps and autocorrelation under
    resampling variability, but cannot generate new fluctuation patterns.
    Pass the training subset (``dataset.subset("train")``); nothing else
    should ever be exposed to a generator.
    """

    def __init__(self, dataset: DailyDataset, seed: int = 0):
        self._dataset = dataset
        self._rng = np.random.default_rng(seed)

    def sample(self, n_days: int) -> list[GeneratedDay]:
        """Draw ``n_days`` training days with replacement."""
        if len(self._dataset.dates) == 0:
            raise ValueError("cannot bootstrap-sample an empty dataset")
        rows = self._rng.integers(0, len(self._dataset.dates), size=n_days)
        return [
            GeneratedDay(
                date=self._dataset.dates[row],
                k=self._dataset.k[row].copy(),
                daylight=self._dataset.valid[row],
                g_clear_cal=self._dataset.g_clear_cal[row],
            )
            for row in rows
        ]
