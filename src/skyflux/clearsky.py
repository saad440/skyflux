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

# The reference calibration is fitted only on clear slots at or above this
# G_clear level. Train-year detector bins show the observed/modelled ratio
# flat at ~1.08 from ~145 to ~840 W/m^2 and rising below that (dawn/dusk edge
# regime mixed with winter midday); 150 W/m^2 sits at the knee. Below it the
# measured ratios are a documented residual rather than something a
# multiplicative correction should absorb.
CALIBRATION_G_CLEAR_MIN_WM2 = 150.0

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
    inherits that uncertainty. Values below the horizon are exactly zero; a
    NaN with the sun up (model failure) raises ``ValueError`` instead of
    being read as night.
    """
    import pvlib

    location = pvlib.location.Location(latitude, longitude, tz="UTC", altitude=elevation_m)
    clear = location.get_clearsky(midpoints_utc, model="ineichen")
    ghi = clear["ghi"]
    # Below the horizon the physical reference is exactly zero; substitute it
    # there explicitly (some pvlib versions emit NaN for these stamps). A NaN
    # with the sun up would be a model failure, not night: zeroing it would
    # silently shrink the daylight mask, so it is rejected instead.
    below_horizon = location.get_solarposition(midpoints_utc)["apparent_zenith"] >= 90.0
    ghi = ghi.mask(below_horizon, 0.0)
    if ghi.isna().any():
        first_bad = ghi.index[ghi.isna()][0]
        raise ValueError(f"clear-sky model returned NaN with the sun up at {first_bad}")
    ghi = ghi.clip(lower=0.0)
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
    of valid observed GHI over each day's daylight slots only (night zeros
    carry no clearness information and would dilute the score by an amount
    that depends on day length, biasing selection towards long days), then
    their valid daylight slots are pooled. Days with fewer than
    ``min_valid_slots`` valid daylight observations are ineligible. Peak
    irradiance is only a proxy for cloud-free conditions — a brief
    cloud-enhancement event can also raise a day's peak — so the pooled
    median over many slots, not any single selected day, is the meaningful
    quantity. A median ratio well above 1 means the Ineichen reference with
    the bundled turbidity climatology is biased low at the site, which
    inflates k and any k > 1 statistics. This is a diagnostic only; no
    scaling is fitted or applied here.
    """
    observed = ghi_wm2.where(daylight).dropna()
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


def clearest_days_mask(
    ghi_wm2: pd.Series,
    g_clear: pd.Series,
    daylight: pd.Series,
    days_per_month: int = 10,
    quantile: float = 0.9,
    min_valid_slots: int = 50,
) -> pd.Series:
    """Mark valid daylight slots on the clearest days of each calendar month.

    Days are scored by the ``quantile`` quantile of valid observed GHI over
    their daylight slots — an observed-peak criterion, because selecting on
    the observed/modelled ratio itself would bias any ratio measured on the
    selected days. The ``days_per_month`` highest-scoring eligible days per
    calendar month, pooled over all years present, are kept. Month-stratified
    selection guarantees winter coverage, unlike the per-year selection of
    ``clear_day_ratio_diagnostic``, which long summer days dominate.

    Returns a boolean Series over ``ghi_wm2.index``, true exactly on valid
    daylight slots of the selected days.
    """
    observed = ghi_wm2.where(daylight).dropna()
    per_day = observed.groupby(observed.index.date)
    scores = per_day.quantile(quantile)
    counts = per_day.size()
    months = pd.DatetimeIndex(scores.index).month
    selected_dates = []
    for month in range(1, 13):
        eligible = scores[(months == month) & (counts >= min_valid_slots)]
        selected_dates.extend(eligible.nlargest(days_per_month).index)
    selected = pd.DatetimeIndex(sorted(selected_dates))
    day_key = ghi_wm2.index.tz_convert(None).normalize()
    is_selected_day = pd.Series(day_key.isin(selected), index=ghi_wm2.index)
    return (daylight & ghi_wm2.notna() & is_selected_day).rename("clear_day")


def clear_ratio_by_month(
    clear_slots: pd.Series, ghi_wm2: pd.Series, g_clear: pd.Series
) -> pd.DataFrame:
    """Monthly observed/clear-sky ratio statistics on marked slots.

    ``clear_slots`` is a boolean Series whose index is a subset of
    ``ghi_wm2`` (from ``clearest_days_mask`` or ``detect_clear_slots``).
    Unlike the all-sky monthly k table these ratios hold under clear
    conditions, so seasonal structure here reflects reference bias rather
    than cloud climatology. ``p90_ratio`` doubles as an envelope-offset
    sensitivity.
    """
    slots = clear_slots.fillna(False).astype(bool)
    ratio = (ghi_wm2 / g_clear).loc[slots.index].where(slots).dropna()
    if ratio.empty:
        return pd.DataFrame(columns=["n_days", "n_slots", "median_ratio", "p25_ratio", "p90_ratio"])
    per_slot = pd.DataFrame({"ratio": ratio, "month": ratio.index.month, "day": ratio.index.date})
    table = per_slot.groupby("month").agg(
        n_days=("day", "nunique"),
        n_slots=("ratio", "size"),
        median_ratio=("ratio", "median"),
        p25_ratio=("ratio", lambda v: float(v.quantile(0.25))),
        p90_ratio=("ratio", lambda v: float(v.quantile(0.90))),
    )
    table.index.name = "month_utc"
    return table


def clear_ratio_by_reference_bin(
    clear_slots: pd.Series,
    ghi_wm2: pd.Series,
    g_clear: pd.Series,
    n_bins: int = 8,
) -> pd.DataFrame:
    """Median observed/clear-sky ratio by clear-sky-irradiance bin on marked slots.

    Equal-count bins over the ``g_clear`` values of the marked slots. A
    median ratio roughly flat across bins supports a purely multiplicative
    reference correction; a systematic trend with ``g_clear`` would indicate
    a zenith-dependent model error that no single factor can absorb.
    """
    empty = pd.DataFrame(columns=["g_clear_low", "g_clear_high", "n_slots", "median_ratio", "p90_ratio"])
    slots = clear_slots.fillna(False).astype(bool)
    marked = pd.DataFrame({"g_clear": g_clear, "ratio": ghi_wm2 / g_clear}).loc[slots.index][slots]
    marked = marked.dropna()
    marked = marked[marked["g_clear"] > 0]
    if marked.empty:
        return empty
    quantiles = np.unique(np.quantile(marked["g_clear"], np.linspace(0, 1, n_bins + 1)))
    if len(quantiles) < 2:
        return empty
    bins = pd.cut(marked["g_clear"], quantiles, include_lowest=True)
    table = marked.groupby(bins, observed=False).agg(
        n_slots=("ratio", "size"),
        median_ratio=("ratio", "median"),
        p90_ratio=("ratio", lambda v: float(v.quantile(0.90))),
    )
    table = table[table["n_slots"] > 0]
    intervals = table.index
    table = table.reset_index(drop=True)
    table.insert(0, "g_clear_high", [float(interval.right) for interval in intervals])
    table.insert(0, "g_clear_low", [float(interval.left) for interval in intervals])
    return table


def detect_clear_slots(
    ghi_wm2: pd.Series,
    g_clear: pd.Series,
    daylight: pd.Series,
    min_daylight_slots: int = 12,
) -> tuple[pd.Series, dict]:
    """Detect clear-sky slots with the Reno & Hansen algorithm, per day.

    pvlib's ``detect_clearsky`` compares windowed statistics of measured and
    modelled GHI; with ``infer_limits=True`` its thresholds are interpolated
    for the sampling interval from Jordan & Hansen (2023), which covers our
    10-minute cadence. The detector needs equally spaced timestamps, so it is
    applied to the contiguous daylight subsequence of each day whose daylight
    block is complete and fully valid (measured zeros count as valid); nights
    are excluded because measured and modelled zeros agree trivially. Days
    with an incomplete daylight block cannot be fed to the detector and are
    skipped.

    Returns the boolean clear-slot mask over ``ghi_wm2.index`` and an audit
    dict: days evaluated, days the detector failed on, days with detections,
    clear-slot counts per calendar month, and the median per-day reference
    scaling fitted internally by the detector (its alpha).
    """
    import pvlib

    mask = pd.Series(False, index=ghi_wm2.index, name="detector_clear")
    by_month = {month: 0 for month in range(1, 13)}
    alphas: list[float] = []
    evaluated = failed = 0
    day_keys = ghi_wm2.index.date
    for _day, ghi_day in ghi_wm2.groupby(day_keys):
        day_index = ghi_day.index
        dl = daylight.loc[day_index].to_numpy()
        positions = np.flatnonzero(dl)
        if len(positions) < min_daylight_slots:
            continue
        valid_daylight = (ghi_day.notna().to_numpy())[positions]
        # One contiguous run: a gap would break the detector's equal spacing.
        if not np.all(np.diff(positions) == 1) or not valid_daylight.all():
            continue
        block = day_index[positions[0] : positions[-1] + 1]
        evaluated += 1
        try:
            is_clear, _components, alpha = pvlib.clearsky.detect_clearsky(
                ghi_day.loc[block], g_clear.loc[block], infer_limits=True, return_components=True
            )
        except Exception:
            failed += 1
            continue
        clear_block = pd.Series(np.asarray(is_clear), index=block).fillna(False).astype(bool)
        clear_stamps = clear_block.index[clear_block.to_numpy()]
        if len(clear_stamps) == 0:
            continue
        mask.loc[clear_stamps] = True
        alphas.append(float(alpha))
        month_values, month_counts = np.unique(clear_stamps.month, return_counts=True)
        for month, count in zip(month_values, month_counts):
            by_month[int(month)] += int(count)
    audit = {
        "n_days_evaluated": evaluated,
        "n_days_failed": failed,
        "n_days_with_clear_slots": len(alphas),
        "n_clear_slots": int(mask.sum()),
        "clear_slots_by_month": by_month,
        "median_alpha": float(np.median(alphas)) if alphas else None,
    }
    return mask, audit


def fit_reference_scale(
    clear_slots: pd.Series,
    ghi_wm2: pd.Series,
    g_clear: pd.Series,
    g_clear_min: float = CALIBRATION_G_CLEAR_MIN_WM2,
) -> dict:
    """Fit the multiplicative clear-sky reference correction.

    The scale is the median observed/modelled ratio over the marked clear
    slots at or above ``g_clear_min``, so applying ``scale * G_clear`` makes
    the median ratio on exactly those slots 1. A median (not a mean) keeps
    residual misclassified cloudy slots from dragging the fit, and the
    ``g_clear_min`` floor keeps the low-sun regime — where ratios rise from
    within-day geometry rather than reference bias — out of the fit.

    Returns the scale with the number of fitted slots and their G_clear
    range as provenance. Raises ``ValueError`` when fewer than 100 slots
    remain: too little evidence to calibrate on.
    """
    slots = clear_slots.fillna(False).astype(bool)
    usable = slots & (g_clear >= g_clear_min) & ghi_wm2.notna() & (g_clear > 0)
    ratio = (ghi_wm2 / g_clear).where(usable).dropna()
    if ratio.size < 100:
        raise ValueError(
            f"only {ratio.size} clear slots at or above {g_clear_min} W/m^2; cannot calibrate"
        )
    return {
        "scale": float(ratio.median()),
        "n_slots": int(ratio.size),
        "g_clear_min_wm2": float(g_clear_min),
        "g_clear_max_wm2": float(g_clear[usable].max()),
    }


def apply_reference_scale(g_clear: pd.Series, scale: float) -> pd.Series:
    """Return the calibrated clear-sky reference ``scale * G_clear``."""
    return (scale * g_clear).rename("g_clear_calibrated_wm2")


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
    g_clear_calibrated: pd.Series | None = None,
) -> dict:
    """Pack 10-minute series into per-UTC-day 144-slot matrices.

    ``midpoints_utc`` must be a complete, duplicate-free 10-minute grid
    (``00:05 ... 23:55`` UTC, contiguous days), as produced by the DWD QC
    stage. Incomplete grids or skipped whole days raise ``ValueError``: a
    missing daylight timestamp would otherwise keep the night fill value and
    escape the daylight inclusion screen as a fake night slot.

    Returns a dict with:

    - ``dates``: (n_days,) datetime64[D] UTC calendar days
    - ``k``: (n_days, 144) float32, finite only where status is
      DAYLIGHT_VALID (includes measured zeros, k = 0)
    - ``ghi``: (n_days, 144) float32, measured GHI where the measurement is
      valid (whole day, night zeros included), NaN otherwise
    - ``g_clear``: (n_days, 144) float32, clear-sky reference everywhere
    - ``status``: (n_days, 144) int8 slot codes NIGHT / DAYLIGHT_VALID /
      DAYLIGHT_MISSING
    - ``g_clear_calibrated``/``k_calibrated``: (n_days, 144) float32, packed
      only when a calibrated reference is passed. The daylight mask stays
      computed from the raw reference, so ``k_calibrated`` is finite on
      exactly the slots ``k`` is.

    The physical reference is retained for every slot so that actual
    irradiance can later be reconstructed from k, including night-time values
    which are never modelled statistically.
    """
    columns = {"ghi": ghi_wm2, "g_clear": g_clear, "daylight": daylight}
    if g_clear_calibrated is not None:
        columns["g_clear_cal"] = g_clear_calibrated
    frame = pd.DataFrame(
        columns, index=pd.DatetimeIndex(midpoints_utc, name="midpoint_utc")
    ).sort_index()
    index = pd.DatetimeIndex(frame.index)
    if index.has_duplicates:
        raise ValueError("midpoint grid contains duplicate timestamps")
    day_stamps = index.floor("D")
    days = pd.DatetimeIndex(sorted(day_stamps.unique()))
    # Compare as timezone-free datetime64 arrays (pandas deprecates .values
    # for tz-aware indexes); both sides are UTC wall time.
    day_values = days.tz_convert(None).to_numpy()
    day_position = np.searchsorted(day_values, day_stamps.tz_convert(None).to_numpy())
    slot = index.hour * 6 + index.minute // 10
    n_days, n_slots = len(days), 144
    # Unit-independent one-day gap check (index resolution may be us or ns).
    if n_days > 1 and (np.diff(day_values) != np.timedelta64(1, "D")).any():
        raise ValueError("midpoint grid skips whole days; days must be contiguous")
    # Uniqueness of the (day, slot) positions rules out duplicate or off-grid
    # stamps; combined with the exact count n_days * 144 it proves that every
    # slot of every day is present exactly once, so no timestamp can be
    # dropped and read back as a night slot.
    positions = day_position * n_slots + slot
    if len(np.unique(positions)) != len(frame) or len(frame) != n_days * n_slots:
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

    arrays = {
        "dates": days.tz_localize(None).to_numpy(dtype="datetime64[D]"),
        "k": scatter(k.to_numpy(dtype=np.float64), np.nan, np.float32),
        "ghi": scatter(frame["ghi"].to_numpy(dtype=np.float64), np.nan, np.float32),
        "g_clear": scatter(frame["g_clear"].to_numpy(dtype=np.float64), np.nan, np.float32),
        "status": scatter(status, STATUS_NIGHT, np.int8),
    }
    if "g_clear_cal" in frame.columns:
        k_cal = cloud_factor(frame["ghi"], frame["g_clear_cal"], frame["daylight"])
        arrays["g_clear_calibrated"] = scatter(
            frame["g_clear_cal"].to_numpy(dtype=np.float64), np.nan, np.float32
        )
        arrays["k_calibrated"] = scatter(k_cal.to_numpy(dtype=np.float64), np.nan, np.float32)
    return arrays


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
    exactly one split; unassigned dates and overlapping split ranges raise
    ``ValueError`` so silent split leakage cannot go unnoticed.
    """
    years = pd.DatetimeIndex(dates).year.to_numpy()
    masks = {
        name: (years >= first) & (years <= last)
        for name, (first, last) in splits.items()
    }
    # Exactly one split per date: an unassigned date would silently shrink a
    # split, an overlapping assignment would leak days between splits.
    n_splits = np.column_stack(list(masks.values())).sum(axis=1)
    if not (n_splits == 1).all():
        dates_index = pd.DatetimeIndex(dates)
        unassigned = dates_index[n_splits == 0]
        if len(unassigned):
            raise ValueError(f"dates outside all splits, e.g. {unassigned[0].date()}")
        overlapping = dates_index[n_splits > 1]
        raise ValueError(f"overlapping splits, e.g. {overlapping[0].date()}")
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
