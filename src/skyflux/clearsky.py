"""Clear-sky reference, cloud factor and daily slot representation.

This stage turns the QC'd 10-minute observations into the SkyFlux working
representation ``G(t) = G_clear(t) * k(t)``. The clear-sky reference
``G_clear`` is the pvlib Ineichen model driven by station metadata; the
cloud-modification factor ``k = G / G_clear`` is only formed where the
reference is strong enough to give a stable ratio. Observations are then
packed into per-UTC-day 144-slot arrays with an explicit status encoding so
that night, valid, missing and measured-zero slots remain distinguishable.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .dwd import FIRST_YEAR, LAST_YEAR

# Provisional daylight criterion: slots with a clear-sky reference below this
# value are excluded from k so that near-sunrise/sunset ratios (unstable
# denominators) never enter the representation. The threshold acts on modelled
# G_clear, not on solar elevation; no fixed solar-elevation equivalent is
# implied. Its practical effect (daylight slots per day) is measured by the
# preparation script and reported alongside the arrays.
DAYLIGHT_G_CLEAR_THRESHOLD_WM2 = 25.0

# Slot status codes for the daily arrays. Measured zero in daylight is a valid
# measurement and is encoded as DAYLIGHT_VALID with k = 0, which stays
# distinguishable from NaN (missing/invalid) and from night slots.
STATUS_NIGHT = 0
STATUS_DAYLIGHT_VALID = 1
STATUS_DAYLIGHT_MISSING = 2

# Chronological splits by whole calendar years over the observation window:
# train 2020-2023, validation 2024, test 2025. Statistics or scalings fitted
# downstream may use training days only.
SPLIT_YEARS = {
    "train": (FIRST_YEAR, LAST_YEAR - 2),
    "validation": (LAST_YEAR - 1, LAST_YEAR - 1),
    "test": (LAST_YEAR, LAST_YEAR),
}


def clear_sky_ghi(
    midpoints_utc: pd.DatetimeIndex,
    latitude: float,
    longitude: float,
    elevation_m: float,
) -> pd.Series:
    """Return Ineichen clear-sky GHI in W/m^2 at the interval midpoints.

    ``midpoints_utc`` must be timezone-aware UTC timestamps (the 10-minute
    interval midpoints ``MESS_DATUM - 5 min``). The model is evaluated with
    pvlib using the station coordinates and elevation. Linke turbidity comes
    from the monthly climatology bundled with pvlib (Remund et al., SoDA
    world map, interpolated between monthly values by pvlib's default); it is
    an approximate climatology, not a site measurement, so the reference GHI
    inherits that uncertainty. Values below the horizon are exactly zero.
    """
    import pvlib

    location = pvlib.location.Location(latitude, longitude, tz="UTC", altitude=elevation_m)
    clear = location.get_clearsky(midpoints_utc, model="ineichen")
    ghi = clear["ghi"]
    # Below-horizon geometry can in principle produce NaN or tiny negatives
    # depending on pvlib version; physically the reference is zero.
    ghi = ghi.fillna(0.0).clip(lower=0.0)
    ghi.name = "g_clear_wm2"
    return ghi


def clear_day_ratio_diagnostic(
    ghi_wm2: pd.Series,
    g_clear: pd.Series,
    daylight: pd.Series,
    days_per_year: int = 15,
    quantile: float = 0.9,
    min_valid_slots: int = 50,
) -> dict:
    """Median observed/clear-sky ratio on the clearest days of each year.

    Clearest days are selected per calendar year by the ``quantile`` quantile
    of valid observed GHI per day (clear days have the highest peaks), then
    their valid daylight slots are pooled. A median ratio well above 1 means
    the Ineichen reference with the bundled turbidity climatology is biased
    low at the site, which inflates k and any k > 1 statistics. This is a
    diagnostic only; no scaling is fitted or applied here.
    """
    observed = ghi_wm2.dropna()
    per_day = observed.groupby(observed.index.date)
    scores = per_day.quantile(quantile)
    counts = per_day.size()
    years = pd.DatetimeIndex(scores.index).year
    daily_ratio = (ghi_wm2 / g_clear.where(daylight)).where(daylight & ghi_wm2.notna())
    selected, ratios = [], []
    for year in sorted(set(years)):
        eligible = scores[(years == year) & (counts >= min_valid_slots)]
        for date in eligible.nlargest(days_per_year).index:
            day_slice = daily_ratio.loc[str(date)]
            ratios.append(day_slice.dropna())
            if len(day_slice.dropna()):
                selected.append(date)
    pooled = pd.concat(ratios) if ratios else pd.Series(dtype=float)
    if pooled.empty:
        return {"n_days": 0, "n_slots": 0}
    return {
        "n_days": len(selected),
        "n_slots": int(pooled.size),
        "median_ratio": float(pooled.median()),
        "p90_ratio": float(pooled.quantile(0.9)),
    }


def daylight_mask(g_clear: pd.Series, threshold: float = DAYLIGHT_G_CLEAR_THRESHOLD_WM2) -> pd.Series:
    """Boolean mask of slots whose clear-sky reference reaches ``threshold``."""
    return g_clear >= threshold


def cloud_factor(ghi_wm2: pd.Series, g_clear: pd.Series, daylight: pd.Series) -> pd.Series:
    """Return ``k = G / G_clear`` on valid daylight slots, NaN elsewhere.

    ``ghi_wm2`` must already carry NaN in invalid slots (as produced by the
    DWD QC stage). The ratio is only computed where ``daylight`` is true, so
    the denominator is at least the daylight threshold. A measured zero in
    daylight yields k = 0, which is a valid value, not a mask. k may exceed 1
    (cloud enhancement); values are neither clipped nor filtered here.
    """
    k = ghi_wm2 / g_clear.where(daylight)
    return k.where(daylight & ghi_wm2.notna()).rename("k")


def build_daily_arrays(
    midpoints_utc: pd.DatetimeIndex,
    ghi_wm2: pd.Series,
    g_clear: pd.Series,
    daylight: pd.Series,
) -> dict:
    """Pack 10-minute series into per-UTC-day 144-slot matrices.

    ``midpoints_utc`` must be a complete, duplicate-free 10-minute grid
    (``00:05 ... 23:55`` UTC, contiguous days), as produced by the DWD QC
    stage; slots missing from the input would remain NaN with night status.

    Returns a dict with:

    - ``dates``: (n_days,) datetime64[D] UTC calendar days
    - ``k``: (n_days, 144) float32, finite only where status is
      DAYLIGHT_VALID (includes measured zeros, k = 0)
    - ``ghi``: (n_days, 144) float32, measured GHI where the measurement is
      valid (whole day, night zeros included), NaN otherwise
    - ``g_clear``: (n_days, 144) float32, clear-sky reference everywhere
    - ``status``: (n_days, 144) int8 slot codes NIGHT / DAYLIGHT_VALID /
      DAYLIGHT_MISSING

    The physical reference is retained for every slot so that actual
    irradiance can later be reconstructed from k, including night-time values
    which are never modelled statistically.
    """
    frame = pd.DataFrame(
        {"ghi": ghi_wm2, "g_clear": g_clear, "daylight": daylight},
        index=pd.DatetimeIndex(midpoints_utc, name="midpoint_utc"),
    ).sort_index()
    index = pd.DatetimeIndex(frame.index)
    if index.has_duplicates:
        raise ValueError("midpoint grid contains duplicate timestamps")
    day_stamps = index.floor("D")
    days = pd.DatetimeIndex(sorted(day_stamps.unique()))
    # Compare as timezone-free datetime64 arrays (pandas deprecates .values
    # for tz-aware indexes); both sides are UTC wall time.
    day_position = np.searchsorted(
        days.tz_convert(None).to_numpy(), day_stamps.tz_convert(None).to_numpy()
    )
    slot = index.hour * 6 + index.minute // 10
    n_days, n_slots = len(days), 144
    if len(np.unique(day_position * n_slots + slot)) != len(frame):
        raise ValueError("midpoint grid is not a complete unique 10-minute day grid")

    k = cloud_factor(frame["ghi"], frame["g_clear"], frame["daylight"])
    valid = frame["daylight"] & frame["ghi"].notna()
    status = np.where(
        valid, STATUS_DAYLIGHT_VALID,
        np.where(frame["daylight"], STATUS_DAYLIGHT_MISSING, STATUS_NIGHT),
    ).astype(np.int8)

    def scatter(values, fill: float, dtype) -> np.ndarray:
        matrix = np.full((n_days, n_slots), fill, dtype=dtype)
        matrix[day_position, slot] = values
        return matrix

    return {
        "dates": days.tz_localize(None).to_numpy(dtype="datetime64[D]"),
        "k": scatter(k.to_numpy(dtype=np.float64), np.nan, np.float32),
        "ghi": scatter(frame["ghi"].to_numpy(dtype=np.float64), np.nan, np.float32),
        "g_clear": scatter(frame["g_clear"].to_numpy(dtype=np.float64), np.nan, np.float32),
        "status": scatter(status, STATUS_NIGHT, np.int8),
    }


def daily_inclusion(
    status: np.ndarray, min_valid_daylight_fraction: float = 0.98
) -> pd.DataFrame:
    """Per-day daylight-based inclusion screen from slot status codes.

    A day is included when at least ``min_valid_daylight_fraction`` of its
    daylight slots carry a valid measurement. Unlike the earlier full-day
    98% completeness screen this ignores night-time slots, so short winter
    days are not diluted by the long nights. Duplicate-timestamp flags from
    the QC stage must be merged by the caller.
    """
    daylight_slots = status != STATUS_NIGHT
    valid_slots = status == STATUS_DAYLIGHT_VALID
    n_daylight = daylight_slots.sum(axis=1).astype(np.float64)
    n_valid = valid_slots.sum(axis=1).astype(np.float64)
    fraction = np.divide(
        n_valid, n_daylight, out=np.zeros_like(n_valid), where=n_daylight > 0
    )
    return pd.DataFrame(
        {
            "n_daylight_slots": n_daylight.astype(np.int32),
            "n_valid_daylight_slots": n_valid.astype(np.int32),
            "valid_daylight_fraction": fraction,
            "include_day": fraction >= min_valid_daylight_fraction,
        }
    )


def chronological_splits(dates: np.ndarray, splits: dict = SPLIT_YEARS) -> dict:
    """Boolean day masks for the chronological train/validation/test splits.

    ``dates`` is an array of datetime64[D] values. Every date must fall in
    exactly one split; dates outside the observation window raise
    ``ValueError`` so silent split leakage cannot go unnoticed.
    """
    years = pd.DatetimeIndex(dates).year.to_numpy()
    masks = {
        name: (years >= first) & (years <= last)
        for name, (first, last) in splits.items()
    }
    if not np.logical_or.reduce(list(masks.values())).all():
        unassigned = pd.DatetimeIndex(dates)[
            ~np.logical_or.reduce(list(masks.values()))
        ]
        raise ValueError(f"dates outside all splits, e.g. {unassigned[0].date()}")
    return masks


def k_statistics(k: np.ndarray) -> dict:
    """Distribution summary of finite k values (NaNs ignored).

    Reports cloud-enhancement fractions (k > 1 and k > 1.05) alongside
    quantiles; nothing is clipped, so the extremes document the raw ratio
    behaviour near the daylight threshold.
    """
    values = np.asarray(k, dtype=np.float64).ravel()
    values = values[np.isfinite(values)]
    if values.size == 0:
        return {"count": 0}
    quantiles = np.percentile(values, [1, 25, 50, 75, 99])
    return {
        "count": int(values.size),
        "fraction_k_gt_1": float(np.mean(values > 1.0)),
        "fraction_k_gt_1_05": float(np.mean(values > 1.05)),
        "min": float(values.min()),
        "p01": float(quantiles[0]),
        "p25": float(quantiles[1]),
        "median": float(quantiles[2]),
        "p75": float(quantiles[3]),
        "p99": float(quantiles[4]),
        "max": float(values.max()),
        "mean": float(values.mean()),
    }


def monthly_k_table(midpoints_utc: pd.DatetimeIndex, k: pd.Series) -> pd.DataFrame:
    """Monthly (UTC) k diagnostics: sample count, median and enhancement fractions."""
    frame = pd.DataFrame({"k": k}, index=pd.DatetimeIndex(midpoints_utc))
    months = pd.DatetimeIndex(frame.index).month
    monthly = frame.groupby(months)["k"].agg(
        n="count",
        median_k="median",
        # Fractions over finite values only; missing slots must not dilute them.
        frac_k_gt_1=lambda v: float((v.dropna() > 1.0).mean()),
        frac_k_gt_1_05=lambda v: float((v.dropna() > 1.05).mean()),
    )
    monthly.index.name = "month_utc"
    return monthly
