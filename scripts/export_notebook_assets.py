"""Export representative observed days for the scientific notebooks.

Reads the daily arrays written by ``prepare_daily.py`` plus the DWD
validation and clear-sky reports for provenance, and writes a compact
``notebooks/assets/representative_days.npz`` holding a handful of days that
illustrate the physical representation: measured GHI, the raw and calibrated
Ineichen clear-sky reference, the cloud factor k (raw and calibrated) and the
int8 slot status encoding (night / daylight-valid / daylight-missing).

Days are chosen by deterministic criteria over the training split
(2020-2023), each skipping days already selected: the clearest and most
overcast included days by daily median k, the most intermittent day by p95
|dG|, the day with the largest cloud-enhancement fraction (k_cal > 1.05),
the shortest and longest days (seasonal daylight geometry), and one day with
many daylight-missing slots (mask semantics; not an included day by
construction). Training-year days only, so the 2025 test year stays
exclusively associated with the final evaluation. Values are stored exactly
as packed in the daily npz (float32, NaN where invalid); nothing is
imputed, clipped or recalculated.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from skyflux.baselines import SPLIT_IDS
from skyflux.clearsky import (
    STATUS_DAYLIGHT_MISSING,
    STATUS_DAYLIGHT_VALID,
    STATUS_NIGHT,
    SPLIT_YEARS,
)
from skyflux.dwd import FIRST_YEAR, LAST_YEAR

# Same floor as the clear-condition diagnostics: days with very few valid
# daylight slots give unstable medians and are not representative.
MIN_VALID_SLOTS = 50


def daily_selection_statistics(npz) -> dict:
    """Per-day statistics used by the selection criteria.

    ``median_k`` and the ramp p95 mirror the evaluation conventions: ramps
    are |dG| between consecutive valid slots only, so pairs spanning an
    invalid slot or a day boundary never count.
    """
    status = npz["status"]
    valid = status == STATUS_DAYLIGHT_VALID
    n_valid = valid.sum(axis=1)
    k_cal = np.where(valid, npz["k_calibrated"].astype(np.float64), np.nan)

    median_k = np.full(len(status), np.nan)
    has_valid = n_valid > 0
    with np.errstate(invalid="ignore"):
        median_k[has_valid] = np.nanmedian(k_cal[has_valid], axis=1)

    ghi = np.where(valid, npz["ghi"].astype(np.float64), 0.0)
    both = valid[:, :-1] & valid[:, 1:]
    diffs = np.abs(np.diff(ghi, axis=1))
    ramp_p95 = np.full(len(status), np.nan)
    for row in np.flatnonzero(both.any(axis=1)):
        ramp_p95[row] = np.percentile(diffs[row][both[row]], 95)

    return {
        "valid": valid,
        "n_valid": n_valid,
        "n_daylight": (status != STATUS_NIGHT).sum(axis=1),
        "n_missing": (status == STATUS_DAYLIGHT_MISSING).sum(axis=1),
        "median_k": median_k,
        "ramp_p95": ramp_p95,
        "enhancement_fraction": np.where(valid, k_cal > 1.05, False).sum(axis=1)
        / np.maximum(n_valid, 1),
    }


def select_representative_rows(npz) -> tuple[list[int], list[str], list[str]]:
    """Row indices, labels and criterion descriptions for the export."""
    stats = daily_selection_statistics(npz)
    train_included = (npz["split_id"] == SPLIT_IDS["train"]) & npz["include_day"].astype(bool)
    train_any_valid = (npz["split_id"] == SPLIT_IDS["train"]) & (stats["n_valid"] > 0)
    eligible = train_included & (stats["n_valid"] >= MIN_VALID_SLOTS)

    criteria = [
        ("clearest", eligible, stats["median_k"], "max", "highest daily median k_cal"),
        ("most_overcast", eligible, stats["median_k"], "min", "lowest daily median k_cal"),
        (
            "most_intermittent",
            eligible,
            stats["ramp_p95"],
            "max",
            "highest p95 |dG| over consecutive valid slots",
        ),
        (
            "cloud_enhancement",
            eligible,
            stats["enhancement_fraction"],
            "max",
            "largest fraction of valid slots with k_cal > 1.05",
        ),
        ("shortest_day", train_included, stats["n_daylight"].astype(float), "min",
         "fewest daylight slots (winter geometry)"),
        ("longest_day", train_included, stats["n_daylight"].astype(float), "max",
         "most daylight slots (summer geometry)"),
        (
            "partial_data",
            train_any_valid,
            stats["n_missing"].astype(float),
            "max",
            "most daylight-missing slots (mask semantics; typically not an included day)",
        ),
    ]

    rows: list[int] = []
    labels: list[str] = []
    descriptions: list[str] = []
    for label, pool, values, direction, description in criteria:
        ordered = np.argsort(-values if direction == "max" else values, kind="stable")
        for row in ordered:
            if pool[row] and np.isfinite(values[row]) and int(row) not in rows:
                rows.append(int(row))
                labels.append(label)
                descriptions.append(description)
                break
        else:
            raise SystemExit(f"no eligible training day for criterion {label!r}")
    return rows, labels, descriptions


def verify_selection(npz, rows: list[int], scale: float) -> None:
    """Representation invariants on the selected rows, before writing."""
    status = npz["status"][rows]
    valid = status == STATUS_DAYLIGHT_VALID
    ghi = npz["ghi"][rows].astype(np.float64)
    g_clear = npz["g_clear"][rows].astype(np.float64)
    g_clear_cal = npz["g_clear_calibrated"][rows].astype(np.float64)
    k = npz["k"][rows].astype(np.float64)
    k_cal = npz["k_calibrated"][rows].astype(np.float64)

    years = npz["dates"][rows].astype("datetime64[Y]").astype(int) + 1970
    first, last = SPLIT_YEARS["train"]
    if not ((years >= first) & (years <= last)).all():
        raise SystemExit("selected days must lie in the training years")
    if not np.isin(status, (STATUS_NIGHT, STATUS_DAYLIGHT_VALID, STATUS_DAYLIGHT_MISSING)).all():
        raise SystemExit("unexpected slot status code")
    if not np.array_equal(np.isfinite(k), valid) or not np.array_equal(np.isfinite(k_cal), valid):
        raise SystemExit("k must be finite exactly on valid daylight slots")
    if not np.isfinite(ghi[valid]).all():
        raise SystemExit("GHI must be finite on valid daylight slots")
    # float32 tolerance: k and the references were packed from float64.
    # Divide only on the valid slots: night G_clear is exactly zero and
    # would warn even though those slots are never compared.
    if not np.allclose(k[valid], ghi[valid] / g_clear[valid], rtol=1e-6, atol=1e-9):
        raise SystemExit("k does not equal GHI / G_clear on valid slots")
    if not np.allclose(k_cal[valid], ghi[valid] / g_clear_cal[valid], rtol=1e-6, atol=1e-9):
        raise SystemExit("k_calibrated does not equal GHI / G_clear_calibrated on valid slots")
    if not np.allclose(g_clear_cal, scale * g_clear, rtol=1e-6, atol=1e-6):
        raise SystemExit("calibrated reference is not scale * G_clear")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="00342", help="Five-digit DWD station ID")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    args = parser.parse_args()
    root: Path = args.root

    npz_path = (
        root / "data/processed" / f"skyflux_daily_{args.station}_{FIRST_YEAR}_{LAST_YEAR}.npz"
    )
    validation_path = root / "reports" / f"dwd_{args.station}_validation.json"
    clearsky_path = root / "reports" / f"clearsky_{args.station}_report.json"
    for path in (npz_path, validation_path, clearsky_path):
        if not path.is_file():
            raise SystemExit(f"Missing {path}; run scripts/validate_dwd.py and scripts/prepare_daily.py first")

    station = json.loads(validation_path.read_text())["station"]
    clearsky_report = json.loads(clearsky_path.read_text())
    calibration = clearsky_report["reference_calibration"]

    npz = np.load(npz_path)
    rows, labels, descriptions = select_representative_rows(npz)
    verify_selection(npz, rows, calibration["scale"])

    stats = daily_selection_statistics(npz)
    meta = {
        "station": station,
        "source": {
            "daily_npz": npz_path.name,
            "validation_report": validation_path.name,
            "clearsky_report": clearsky_path.name,
        },
        "selection": {
            "split": "train (2020-2023); the 2025 test year is used only by the final evaluation",
            "min_valid_daylight_slots": MIN_VALID_SLOTS,
            "criteria": [
                {
                    "label": label,
                    "criterion": description,
                    "date": str(npz["dates"][row]),
                    "n_valid_slots": int(stats["n_valid"][row]),
                    "n_daylight_slots": int(stats["n_daylight"][row]),
                    "n_missing_slots": int(stats["n_missing"][row]),
                    "median_k_cal": float(stats["median_k"][row]),
                    "ramp_p95_wm2": float(stats["ramp_p95"][row]),
                    "fraction_k_cal_gt_1_05": float(stats["enhancement_fraction"][row]),
                }
                for label, description, row in zip(labels, descriptions, rows)
            ],
        },
        "encoding": {
            "slot_hours_utc": "interval midpoints 00:05..23:55, hours after UTC midnight",
            "status": {
                str(STATUS_NIGHT): "night: G_clear below the daylight threshold",
                str(STATUS_DAYLIGHT_VALID): "daylight with valid measurement (measured zero keeps k = 0)",
                str(STATUS_DAYLIGHT_MISSING): "daylight without valid measurement",
            },
            "nan": "missing/invalid slots in k, k_calibrated and GHI",
        },
        "daylight_threshold_wm2": float(clearsky_report["daylight_threshold_wm2"]),
        "reference_calibration": {
            "method": calibration["method"],
            "scale": calibration["scale"],
            "n_slots": calibration["n_slots"],
            "g_clear_min_wm2": calibration["g_clear_min_wm2"],
        },
        "clear_sky_model": clearsky_report["clear_sky_model"],
        "linke_turbidity": clearsky_report["linke_turbidity"],
    }

    target_dir = root / "notebooks" / "assets"
    target_dir.mkdir(parents=True, exist_ok=True)
    target = target_dir / "representative_days.npz"
    np.savez_compressed(
        target,
        dates=npz["dates"][rows],
        labels=np.asarray(labels),
        ghi=npz["ghi"][rows],
        g_clear=npz["g_clear"][rows],
        g_clear_calibrated=npz["g_clear_calibrated"][rows],
        k=npz["k"][rows],
        k_calibrated=npz["k_calibrated"][rows],
        status=npz["status"][rows],
        slot_hours_utc=(np.arange(144) * 10 + 5) / 60.0,
        meta_json=np.array(json.dumps(meta, indent=2)),
    )

    print(f"Station: {station['station_name']} ({args.station})")
    for label, row in zip(labels, rows):
        print(
            f"{label:18s} {str(npz['dates'][row])}  "
            f"valid {stats['n_valid'][row]:3d}  daylight {stats['n_daylight'][row]:3d}  "
            f"missing {stats['n_missing'][row]:3d}  median k_cal {stats['median_k'][row]:.3f}  "
            f"ramp p95 {stats['ramp_p95'][row]:.0f} W/m2"
        )
    print(f"Output: {target}")


if __name__ == "__main__":
    main()
