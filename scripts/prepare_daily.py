"""Compute the clear-sky reference, cloud factor k and daily slot arrays.

Reads the QC'd 10-minute table produced by ``validate_dwd.py``, evaluates the
pvlib Ineichen clear-sky reference at the UTC interval midpoints with station
metadata from the validation report, and writes:

- ``data/processed/clearsky_<id>_<first>_<last>.csv.gz``: midpoint-level
  measured GHI, clear-sky GHI, daylight mask and k
- ``data/processed/skyflux_daily_<id>_<first>_<last>.npz``: per-day 144-slot
  matrices (k, GHI, G_clear, status codes, the calibrated reference and its
  k), inclusion flags and split ids
- ``reports/clearsky_<id>_report.json``: provenance, daylight-threshold
  effect, screen comparison, k diagnostics, and the clear-sky reference
  calibration with per-split transfer ratios
- ``reports/clearsky_<id>_monthly.csv``: monthly k and daylight statistics
- ``reports/clearsky_<id>_yearly_inclusion.csv``: retained days per year
  under both completeness screens
- ``figures/clearsky_<id>_diagnostics.png``: four-panel diagnostic figure
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from skyflux.clearsky import (
    CALIBRATION_G_CLEAR_MIN_WM2,
    DAYLIGHT_G_CLEAR_THRESHOLD_WM2,
    SPLIT_YEARS,
    STATUS_DAYLIGHT_MISSING,
    STATUS_DAYLIGHT_VALID,
    STATUS_NIGHT,
    apply_reference_scale,
    build_daily_arrays,
    chronological_splits,
    clear_day_ratio_diagnostic,
    clear_ratio_by_month,
    clear_ratio_by_reference_bin,
    clear_sky_ghi,
    clearest_days_mask,
    cloud_factor,
    daily_inclusion,
    daylight_mask,
    detect_clear_slots,
    fit_reference_scale,
    k_statistics,
    monthly_k_table,
)
from skyflux.dwd import FIRST_YEAR, LAST_YEAR


def load_qc_grid(root: Path, station_id: str) -> pd.DataFrame:
    path = root / "data/processed" / f"dwd_{station_id}_qc_{FIRST_YEAR}_{LAST_YEAR}.csv.gz"
    if not path.is_file():
        raise SystemExit(f"Missing {path}; run scripts/validate_dwd.py first")
    grid = pd.read_csv(path, index_col=0)
    grid.index = pd.to_datetime(grid.index, utc=True)
    return grid.sort_index()


def save_figure(
    dates, inclusion, monthly, arrays, example_days, threshold, target: Path
) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.dates as mdates
    import matplotlib.pyplot as plt

    day_stamps = pd.DatetimeIndex(dates)
    fig, axes = plt.subplots(2, 2, figsize=(12.5, 7.5))

    ax = axes[0, 0]
    ax.scatter(day_stamps, inclusion["n_daylight_slots"], s=2, color="tab:blue")
    ax.set_ylim(0, 144)
    ax.set_ylabel("daylight slots per UTC day")
    ax.set_title(f"Effect of G_clear >= {threshold:.0f} W/m$^2$ daylight mask")
    ax.grid(alpha=0.3)
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    ax = axes[0, 1]
    k_values = arrays["k"].ravel()
    k_values = k_values[np.isfinite(k_values)]
    ax.hist(k_values, bins=np.linspace(0, 2.5, 101), color="tab:gray")
    ax.set_yscale("log")
    for line in (1.0, 1.05):
        ax.axvline(line, color="tab:red", linestyle="--", linewidth=1)
    ax.set_xlabel("k = G / G_clear (daylight slots)")
    ax.set_ylabel("10-minute slots (log)")
    ax.set_title("Cloud-factor distribution (dashed: 1.0 and 1.05)")
    ax.grid(alpha=0.3)

    ax = axes[1, 0]
    ax.plot(monthly.index, monthly["frac_k_gt_1"], marker="o", label="k > 1")
    ax.plot(monthly.index, monthly["frac_k_gt_1_05"], marker="s", label="k > 1.05")
    ax.set_xticks(range(1, 13), labels="JFMAMJJASOND")
    ax.set_ylabel("fraction of daylight slots")
    ax.set_title("Cloud enhancement by calendar month (all years)")
    ax.legend()
    ax.grid(alpha=0.3)

    ax = axes[1, 1]
    for row, label in zip(example_days["rows"], example_days["labels"]):
        hours = (np.arange(144) * 10 + 5) / 60
        ax.plot(hours, arrays["ghi"][row], ".", markersize=3, label=f"G measured ({label})")
        ax.plot(hours, arrays["g_clear"][row], linewidth=1.2, label=f"G_clear ({label})")
    ax.set_xlabel("UTC hour of interval midpoint")
    ax.set_ylabel("W/m$^2$")
    ax.set_title("Example days: clearest vs most overcast (by daily median k)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)

    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="00342", help="Five-digit DWD station ID")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    args = parser.parse_args()
    station_id = args.station
    root: Path = args.root

    report_path = root / "reports" / f"dwd_{station_id}_validation.json"
    if not report_path.is_file():
        raise SystemExit(f"Missing {report_path}; run scripts/validate_dwd.py first")
    station = json.loads(report_path.read_text())["station"]

    grid = load_qc_grid(root, station_id)
    g_clear = clear_sky_ghi(
        grid.index,
        latitude=station["latitude"],
        longitude=station["longitude"],
        elevation_m=station["elevation_m"],
    )
    daylight = daylight_mask(g_clear)
    k = cloud_factor(grid["ghi_wm2"], g_clear, daylight)

    # Clear-condition diagnostics and the reference calibration. Everything
    # that feeds the calibration (detector clear slots) is restricted to the
    # training years; later years enter only as transfer diagnostics.
    train_range = SPLIT_YEARS["train"]
    in_train = (grid.index.year >= train_range[0]) & (grid.index.year <= train_range[1])
    clear_day_mask_all = clearest_days_mask(grid["ghi_wm2"], g_clear, daylight)
    clear_day_mask_train = clearest_days_mask(
        grid["ghi_wm2"][in_train], g_clear[in_train], daylight[in_train]
    )
    detector_mask, detector_audit = detect_clear_slots(
        grid["ghi_wm2"][in_train], g_clear[in_train], daylight[in_train]
    )
    calibration = fit_reference_scale(detector_mask, grid["ghi_wm2"], g_clear)
    g_clear_cal = apply_reference_scale(g_clear, calibration["scale"])
    k_cal = cloud_factor(grid["ghi_wm2"], g_clear_cal, daylight)

    arrays = build_daily_arrays(
        grid.index, grid["ghi_wm2"], g_clear, daylight, g_clear_calibrated=g_clear_cal
    )
    inclusion = daily_inclusion(arrays["status"])

    # Merge the QC duplicate-day flag and compare with the earlier full-day
    # 98% completeness screen (night included), which stays a heuristic.
    daily_qc_path = root / "reports" / f"dwd_{station_id}_daily.csv"
    if not daily_qc_path.is_file():
        raise SystemExit(f"Missing {daily_qc_path}; run scripts/validate_dwd.py first")
    daily_qc = pd.read_csv(daily_qc_path, index_col=0)
    daily_qc.index = pd.to_datetime(daily_qc.index, utc=True).tz_localize(None)
    if not (daily_qc.index.to_numpy(dtype="datetime64[D]") == arrays["dates"]).all():
        raise SystemExit("Daily QC table does not match the QC grid days")
    inclusion["has_duplicate"] = daily_qc["has_duplicate"].to_numpy()
    inclusion["candidate_day_full_screen"] = daily_qc["candidate_day"].to_numpy()
    inclusion["include_day"] = inclusion["include_day"] & ~inclusion["has_duplicate"]

    masks = chronological_splits(arrays["dates"])
    split_id = np.full(len(arrays["dates"]), -1, dtype=np.int8)
    for i, name in enumerate(SPLIT_YEARS):
        split_id[masks[name]] = i

    processed = root / "data/processed"
    processed.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        processed / f"skyflux_daily_{station_id}_{FIRST_YEAR}_{LAST_YEAR}.npz",
        dates=arrays["dates"],
        k=arrays["k"],
        ghi=arrays["ghi"],
        g_clear=arrays["g_clear"],
        status=arrays["status"],
        g_clear_calibrated=arrays["g_clear_calibrated"],
        k_calibrated=arrays["k_calibrated"],
        include_day=inclusion["include_day"].to_numpy(),
        split_id=split_id,
    )
    # Midpoint-level table: measured GHI stays NaN where invalid (no imputation).
    pd.DataFrame(
        {
            "ghi_wm2": grid["ghi_wm2"],
            "g_clear_wm2": g_clear,
            "g_clear_calibrated_wm2": g_clear_cal,
            "daylight": daylight,
            "k": k,
            "k_calibrated": k_cal,
        }
    ).to_csv(
        processed / f"clearsky_{station_id}_{FIRST_YEAR}_{LAST_YEAR}.csv.gz",
        compression="gzip", index_label="midpoint_utc",
    )

    monthly = monthly_k_table(grid.index, k)
    daylight_per_day = inclusion["n_daylight_slots"].groupby(
        pd.DatetimeIndex(arrays["dates"]).month
    ).mean()
    monthly["mean_daylight_slots_per_day"] = daylight_per_day

    inclusion["year"] = pd.DatetimeIndex(arrays["dates"]).year
    yearly = inclusion.groupby("year").agg(
        days=("include_day", "size"),
        included_daylight_screen=("include_day", "sum"),
        candidates_full_day_screen=("candidate_day_full_screen", "sum"),
        mean_daylight_slots=("n_daylight_slots", "mean"),
    )

    report_dir = root / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    monthly.to_csv(report_dir / f"clearsky_{station_id}_monthly.csv", index_label="month_utc")
    yearly.to_csv(report_dir / f"clearsky_{station_id}_yearly_inclusion.csv", index_label="year")

    full = inclusion["candidate_day_full_screen"].to_numpy()
    dayl = inclusion["include_day"].to_numpy()
    k_stats = k_statistics(arrays["k"])
    bias = clear_day_ratio_diagnostic(grid["ghi_wm2"], g_clear, daylight)

    # Clear-condition reference-bias decomposition (measurement only).
    bias_tables = {
        "clearest_days_train_years": clear_ratio_by_month(
            clear_day_mask_train, grid["ghi_wm2"], g_clear
        ),
        "detector_train_years": clear_ratio_by_month(
            detector_mask, grid["ghi_wm2"], g_clear
        ),
        "clearest_days_all_years": clear_ratio_by_month(
            clear_day_mask_all, grid["ghi_wm2"], g_clear
        ),
    }
    bias_bins = {
        "clearest_days_train_years": clear_ratio_by_reference_bin(
            clear_day_mask_train, grid["ghi_wm2"], g_clear
        ),
        "detector_train_years": clear_ratio_by_reference_bin(
            detector_mask, grid["ghi_wm2"], g_clear
        ),
    }

    def table_to_records(table: pd.DataFrame) -> dict:
        return json.loads(table.to_json(orient="index", double_precision=10))

    # Per-split evaluation of the calibration: the training ratios are
    # construction checks, 2024/2025 are transfer checks computed from data
    # that never entered the fit.
    ratio_cal = grid["ghi_wm2"] / g_clear_cal
    usable_cal = ratio_cal.notna() & (g_clear >= CALIBRATION_G_CLEAR_MIN_WM2)
    year = grid.index.year
    calibrated_by_split = {}
    for name, (first, last) in SPLIT_YEARS.items():
        in_split = (year >= first) & (year <= last)
        split_clear = clearest_days_mask(
            grid["ghi_wm2"][in_split], g_clear[in_split], daylight[in_split]
        )
        slots = usable_cal & split_clear
        calibrated_by_split[name] = {
            "median_ratio_clearest_days": (
                float(ratio_cal[slots].median()) if slots.any() else None
            ),
            "n_slots": int(slots.sum()),
        }
    calibrated_by_split["train"]["median_ratio_detector_slots"] = float(
        ratio_cal[usable_cal & detector_mask].median()
    )
    k_cal_stats = k_statistics(arrays["k_calibrated"])

    monthly_k_stats = {
        str(month): {key: float(value) for key, value in row.items()}
        for month, row in monthly.iterrows()
    }
    report = {
        "station": station,
        "clear_sky_model": "pvlib Ineichen, apparent zenith + Kasten-Young airmass, station altitude correction",
        "linke_turbidity": "monthly Remund et al. climatology bundled with pvlib, interpolated to daily values (approximate climatology, not measured)",
        "daylight_threshold_wm2": DAYLIGHT_G_CLEAR_THRESHOLD_WM2,
        "slot_status_encoding": {
            str(STATUS_NIGHT): "night: G_clear below threshold",
            str(STATUS_DAYLIGHT_VALID): "daylight with valid measurement (k finite; measured zero keeps k = 0)",
            str(STATUS_DAYLIGHT_MISSING): "daylight without valid measurement (k NaN)",
        },
        "splits": {name: list(map(int, years)) for name, years in SPLIT_YEARS.items()},
        "daylight_slots_per_day": {
            "min": int(inclusion["n_daylight_slots"].min()),
            "median": float(inclusion["n_daylight_slots"].median()),
            "max": int(inclusion["n_daylight_slots"].max()),
        },
        "valid_measurements_outside_daylight": int(
            np.count_nonzero(
                (arrays["status"] == STATUS_NIGHT) & np.isfinite(arrays["ghi"])
            )
        ),
        "inclusion_screens": {
            "full_day_98pct_candidates": int(full.sum()),
            "daylight_98pct_included": int(dayl.sum()),
            "both": int((full & dayl).sum()),
            "daylight_only": int((~full & dayl).sum()),
            "full_day_only": int((full & ~dayl).sum()),
            "neither": int((~full & ~dayl).sum()),
        },
        "included_days_by_split": {
            name: int((dayl & masks[name]).sum()) for name in SPLIT_YEARS
        },
        "reference_bias_diagnostic": bias,
        "reference_bias_by_month": {
            name: table_to_records(table) for name, table in bias_tables.items()
        },
        "reference_ratio_by_gclear_bin": {
            name: table_to_records(table) for name, table in bias_bins.items()
        },
        "clear_slot_detection_train_years": detector_audit,
        "reference_calibration": {
            "method": "constant multiplicative scale: median observed/clear-sky ratio on Reno & Hansen detector clear slots (pvlib detect_clearsky, infer_limits=True), training years only",
            **calibration,
            "residuals_not_fitted": "slots below the G_clear floor (low-sun dawn/dusk regime) and deep-winter Dec/Jan detector slots (medians 0.6-0.75, contamination-limited by the detector's absolute thresholds at winter magnitudes)",
        },
        "calibrated_clear_ratio_by_split": calibrated_by_split,
        "k_statistics_calibrated": k_cal_stats,
        "k_statistics": k_stats,
        "k_statistics_by_month": monthly_k_stats,
    }
    (report_dir / f"clearsky_{station_id}_report.json").write_text(
        json.dumps(report, indent=2)
    )

    # Example days: clearest and most overcast included days by daily median k.
    valid_k = np.where(arrays["status"] == STATUS_DAYLIGHT_VALID, arrays["k"], np.nan)
    daily_median_k = np.full(len(valid_k), np.nan)
    with_valid = np.isfinite(valid_k).any(axis=1)
    daily_median_k[with_valid] = np.nanmedian(valid_k[with_valid], axis=1)
    eligible = inclusion["include_day"].to_numpy() & np.isfinite(daily_median_k)
    candidates = np.flatnonzero(eligible)
    clearest = candidates[np.argmax(daily_median_k[candidates])]
    overcast = candidates[np.argmin(daily_median_k[candidates])]
    example_days = {
        "rows": [int(clearest), int(overcast)],
        "labels": [
            f"clearest {arrays['dates'][clearest]}",
            f"most overcast {arrays['dates'][overcast]}",
        ],
    }
    save_figure(
        arrays["dates"],
        inclusion,
        monthly,
        arrays,
        example_days,
        DAYLIGHT_G_CLEAR_THRESHOLD_WM2,
        root / "figures" / f"clearsky_{station_id}_diagnostics.png",
    )

    print(f"Station: {station['station_name']} ({station_id})")
    print(f"Slots: {len(grid)}; daylight: {int(daylight.sum())} "
          f"({100 * daylight.mean():.1f}%)")
    print("Daylight slots/day (min/median/max): "
          f"{inclusion['n_daylight_slots'].min()}/"
          f"{inclusion['n_daylight_slots'].median():.0f}/"
          f"{inclusion['n_daylight_slots'].max()}")
    print(f"Days included, 98% daylight screen: {int(dayl.sum())} of {len(dayl)} "
          f"(full-day screen: {int(full.sum())})")
    print("Included days by split:",
          {name: int((dayl & masks[name]).sum()) for name in SPLIT_YEARS})
    print(f"Clearest-day median G/G_clear: {bias['median_ratio']:.3f} "
          f"(reference-bias diagnostic; 1.0 = unbiased)")
    for name, table in bias_tables.items():
        medians = table["median_ratio"]
        if len(medians):
            print(f"Clear-condition median ratio by month ({name}): min {medians.min():.3f} / "
                  f"max {medians.max():.3f} (span {medians.max() - medians.min():.3f})")
    print(f"Detector (train years): {detector_audit['n_clear_slots']} clear slots on "
          f"{detector_audit['n_days_with_clear_slots']}/{detector_audit['n_days_evaluated']} days, "
          f"median alpha {detector_audit['median_alpha']}")
    print(f"Reference calibration: scale {calibration['scale']:.4f} fitted on "
          f"{calibration['n_slots']} detector slots (train years, "
          f"G_clear >= {CALIBRATION_G_CLEAR_MIN_WM2:.0f} W/m^2)")
    print("Calibrated clear-day median ratio by split:", {
        name: values["median_ratio_clearest_days"]
        for name, values in calibrated_by_split.items()
    })
    print(f"k>1: {k_stats['fraction_k_gt_1']:.3%}; k>1.05: "
          f"{k_stats['fraction_k_gt_1_05']:.3%}; median k: {k_stats['median']:.3f}")
    print(f"k_cal>1: {k_cal_stats['fraction_k_gt_1']:.3%}; k_cal>1.05: "
          f"{k_cal_stats['fraction_k_gt_1_05']:.3%}; median k_cal: {k_cal_stats['median']:.3f}")
    print(f"Outputs: {processed / f'skyflux_daily_{station_id}_{FIRST_YEAR}_{LAST_YEAR}.npz'}, "
          f"{report_dir / f'clearsky_{station_id}_report.json'}, "
          f"{root / 'figures' / f'clearsky_{station_id}_diagnostics.png'}")


if __name__ == "__main__":
    main()
