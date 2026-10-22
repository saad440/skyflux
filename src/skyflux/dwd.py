"""Utilities for loading and validating DWD 10-minute solar radiation data.

Observation timestamps indicate the end of each 10-minute interval and
are expressed in UTC for the period covered by this dataset.
"""
from __future__ import annotations

import hashlib
import json
import re
import urllib.request
import zipfile
from pathlib import Path

import pandas as pd

ARCHIVE_ROOT = (
    "https://opendata.dwd.de/climate_environment/CDC/observations_germany/"
    "climate/10_minutes/solar/historical/"
)
STATION_LIST_NAME = "zehn_min_sd_Beschreibung_Stationen.txt"
FACTOR_JCM2_TO_WM2 = 10000.0 / 600.0
FIRST_YEAR, LAST_YEAR = 2020, 2025
USER_AGENT = "skyflux/0.1"


def download(url: str, destination: Path) -> None:
    """Download ``url`` to ``destination`` atomically.

    Streams to a temporary ``.part`` file that is renamed on success, so an
    interrupted download never leaves a partial file at the target path. An
    existing destination is reused without re-validation (no checksum check).
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    if destination.is_file():
        return
    temp = destination.with_name(destination.name + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": USER_AGENT})
    try:
        with urllib.request.urlopen(request, timeout=90) as response, temp.open("wb") as out:
            import shutil
            shutil.copyfileobj(response, out)
        temp.replace(destination)
    finally:
        temp.unlink(missing_ok=True)


def extract_station_metadata(file_contents: bytes, station_id: str) -> dict:
    """Return one station's metadata parsed from the fixed-column DWD list.

    ``file_contents`` are the raw latin-1 bytes of
    ``zehn_min_sd_Beschreibung_Stationen.txt``. The returned dict contains the
    five-digit station id, station name, latitude/longitude in decimal degrees,
    elevation in metres, and the first/last record dates of the station-list
    entry as YYYYMMDD strings. Coordinates are the list's current published
    values; the file carries no per-period relocation history, so it is not
    evidence of an unchanging station position over the archive period.
    """
    decoded = file_contents.decode("latin-1")
    # Common station-list layout: ID, begin, end, elevation, lat, lon, name/state.
    pattern = re.compile(
        r"^\s*(\d{5})\s+(\d{8})\s+(\d{8})\s+(-?\d+)\s+"
        r"(-?\d+[.,]\d+)\s+(-?\d+[.,]\d+)\s+(.+)$"
    )
    # The 10-minute solar station list publishes one row per station. If DWD
    # ever publishes per-period rows (as some other DWD station lists do), the
    # first matching row is returned without checking its period against the
    # analysis window.
    for line in decoded.splitlines():
        match = pattern.match(line)
        if match and match.group(1) == station_id:
            sid, start, end, height, lat, lon, rest = match.groups()
            name = re.split(r"\s{2,}", rest.strip(), maxsplit=1)[0]
            return {
                "station_id": sid,
                "station_name": name,
                "latitude": float(lat.replace(",", ".")),
                "longitude": float(lon.replace(",", ".")),
                "elevation_m": int(height),
                "station_list_first": start,
                "station_list_last": end,
            }
    raise ValueError(f"Station {station_id} missing/unparseable in DWD station list")


def discover_archive(
    station_id: str,
    listing_html: str,
    first_year: int = FIRST_YEAR,
    last_year: int = LAST_YEAR,
) -> str:
    """Return the archive filename whose data range spans the whole window.

    ``listing_html`` is the CDC directory-listing page. An archive qualifies
    when its filename-encoded start date (YYYYMMDD) is on or before
    1 January of ``first_year`` and its end date is on or after
    31 December of ``last_year``. Among qualifying archives the
    lexicographically first filename is returned. Raises ``ValueError`` when
    no single archive covers the window.
    """
    window_start = f"{first_year:04d}0101"
    window_end = f"{last_year:04d}1231"
    names = re.findall(
        rf"10minutenwerte_SOLAR_{re.escape(station_id)}_(\d{{8}})_(\d{{8}})_hist\.zip",
        listing_html,
    )
    candidates = [
        f"10minutenwerte_SOLAR_{station_id}_{a}_{b}_hist.zip"
        for a, b in names if a <= window_start and b >= window_end
    ]
    if not candidates:
        raise ValueError(
            f"No single ZIP spanning {first_year}-{last_year} for station {station_id}"
        )
    return sorted(set(candidates))[0]


def read_solar_zip(archive: Path) -> pd.DataFrame:
    """Read the single product table from a DWD station ZIP archive.

    The product file is semicolon-separated, latin-1 encoded text with one
    header row. Returns the parsed frame with stripped column names. Raises
    ``ValueError`` if the archive does not contain exactly one ``produkt_*.txt``
    member, or if any of the required columns ``MESS_DATUM``, ``GS_10`` or
    ``QN`` is missing.
    """
    with zipfile.ZipFile(archive) as zf:
        members = [
            name for name in zf.namelist()
            if Path(name).name.lower().startswith("produkt_")
            and name.lower().endswith(".txt")
        ]
        if len(members) != 1:
            raise ValueError(f"Expected one DWD product TXT in {archive}: {members}")
        with zf.open(members[0]) as stream:
            frame = pd.read_csv(stream, sep=";", encoding="latin-1", low_memory=False)
    frame.columns = frame.columns.str.strip()
    required = {"MESS_DATUM", "GS_10", "QN"}
    if not required.issubset(frame.columns):
        raise ValueError(f"DWD product missing {required - set(frame.columns)}; got {frame.columns.tolist()}")
    return frame


def normalize_observations(raw: pd.DataFrame, first_year=FIRST_YEAR, last_year=LAST_YEAR):
    """Return a complete 10-minute UTC grid with quality flags and summaries.

    Parameters
    ----------
    raw : pandas.DataFrame
        Product rows with columns ``MESS_DATUM`` (YYYYMMDDHHMM, interval end),
        ``QN`` (quality level) and ``GS_10`` (J/cm^2 summed over the interval).
    first_year, last_year : int
        UTC years spanning the analysis window.

    Returns
    -------
    grid : pandas.DataFrame
        Indexed by interval midpoint (UTC) and reindexed over the full
        10-minute grid from ``first_year-01-01 00:05`` to
        ``last_year-12-31 23:55``. ``ghi_wm2`` holds the interval-mean
        irradiance in W/m^2 and is set only where the record is present,
        non-sentinel, non-negative and QN in (2, 3); all other slots are NaN.
        Missing values are never imputed and out-of-range values are never
        clipped; boolean flag columns record why each slot is unusable.
    daily : pandas.DataFrame
        Per-UTC-day counts of present/valid records and anomalies, plus the
        ``candidate_day`` completeness screen (>= 98% valid slots, no
        duplicated timestamps). The screen counts all 144 slots including
        night-time; it is a completeness heuristic, not a daylight-based
        inclusion rule.
    info : dict
        Aggregate counts, including the quality-code histogram.

    Notes
    -----
    Timestamps that fail to parse or fall off the 10-minute grid are dropped
    and counted. Duplicate midpoints keep the first record; affected days are
    flagged. Values above 1500 W/m^2 are flagged as suspicious but retained,
    since that threshold is an audit marker, not a physical limit.
    """
    frame = raw.copy()
    frame.columns = frame.columns.str.strip()
    for column in ("GS_10", "QN"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce")
    stamp = frame["MESS_DATUM"].astype(str).str.strip().str.replace(r"\.0$", "", regex=True)
    frame["timestamp_end_utc"] = pd.to_datetime(
        stamp, format="%Y%m%d%H%M", utc=True, errors="coerce"
    )
    bad_timestamp_count = int(frame["timestamp_end_utc"].isna().sum())
    frame = frame.dropna(subset=["timestamp_end_utc"]).copy()
    off_grid_count = int(((frame["timestamp_end_utc"].dt.minute % 10 != 0) |
                          (frame["timestamp_end_utc"].dt.second != 0)).sum())
    frame = frame[(frame["timestamp_end_utc"].dt.minute % 10 == 0) &
                  (frame["timestamp_end_utc"].dt.second == 0)]
    frame["midpoint_utc"] = frame["timestamp_end_utc"] - pd.Timedelta(minutes=5)
    duplicate_count = int(frame["midpoint_utc"].duplicated().sum())
    duplicate_days = set(frame.loc[
        frame["midpoint_utc"].duplicated(keep=False), "midpoint_utc"
    ].dt.floor("D"))
    frame = frame.drop_duplicates("midpoint_utc", keep="first").set_index("midpoint_utc")

    # Closed interval: 00:05 UTC on first Jan 1 through 23:55 UTC on last Dec 31.
    expected = pd.date_range(
        f"{first_year}-01-01 00:05", f"{last_year}-12-31 23:55",
        freq="10min", tz="UTC", name="midpoint_utc"
    )
    grid = frame.reindex(expected)
    present = grid["timestamp_end_utc"].notna()
    numeric = grid["GS_10"]
    # Sentinel and negative readings become NaN in ghi_wm2 but stay counted in
    # the flag columns; nothing is imputed or clipped, so counts stay auditable.
    sentinel = numeric == -999
    negative = numeric.notna() & (numeric < 0) & ~sentinel
    good_qn = grid["QN"].isin((2, 3))
    valid = present & numeric.notna() & ~sentinel & ~negative & good_qn
    grid["ghi_wm2"] = (numeric * FACTOR_JCM2_TO_WM2).where(valid)
    grid["record_present"] = present
    grid["measurement_valid"] = valid
    grid["negative_gs"] = negative
    grid["missing_sentinel"] = sentinel
    grid["suspicious_high"] = valid & (grid["ghi_wm2"] > 1500)

    grid.index.name = "midpoint_utc"
    daily = grid.resample("D").agg(
        records_present=("record_present", "sum"),
        valid_measurements=("measurement_valid", "sum"),
        negative_measurements=("negative_gs", "sum"),
        high_measurements=("suspicious_high", "sum"),
    )
    daily["valid_fraction"] = daily["valid_measurements"] / 144.0
    daily["has_duplicate"] = daily.index.isin(duplicate_days)
    # Raw-completeness screen over all 144 slots including night; not a
    # daylight-based inclusion rule.
    daily["candidate_day"] = (daily["valid_fraction"] >= 0.98) & ~daily["has_duplicate"]

    qn_hist = {
        str(int(k)) if pd.notna(k) else "NaN": int(v)
        for k, v in grid.loc[present, "QN"].value_counts(dropna=False).items()
    }
    info = {
        "expected_10min_intervals": len(expected),
        "observed_timestamps": int(present.sum()),
        "absent_timestamps": int((~present).sum()),
        "missing_value_sentinel": int(sentinel.sum()),
        "negative_non_sentinel": int(negative.sum()),
        "unusual_ghi_gt_1500_wm2": int(grid["suspicious_high"].sum()),
        "accepted_qn2_or_qn3_measurements": int(valid.sum()),
        "off_grid_timestamps": off_grid_count,
        "unparseable_timestamps": bad_timestamp_count,
        "duplicate_timestamps": duplicate_count,
        "quality_code_counts": qn_hist,
        "candidate_days_98pct_raw_coverage": int(daily["candidate_day"].sum()),
        "total_days": len(daily),
    }
    return grid, daily, info


def yearly_summary(grid: pd.DataFrame, daily: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-year record, validity, sentinel and anomaly counts.

    ``expected_intervals`` is the number of grid slots per year (144 per day),
    not a count of transmitted records.
    """
    grouped = grid.groupby(grid.index.year)
    years = grouped.agg(
        expected_intervals=("record_present", "size"),
        records_present=("record_present", "sum"),
        valid_measurements=("measurement_valid", "sum"),
        sentinel_values=("missing_sentinel", "sum"),
        negative_measurements=("negative_gs", "sum"),
        suspicious_high=("suspicious_high", "sum"),
    )
    years["valid_fraction"] = years["valid_measurements"] / years["expected_intervals"]
    years["candidate_days"] = daily.groupby(daily.index.year)["candidate_day"].sum()
    return years


def save_coverage_figure(daily: pd.DataFrame, target: Path) -> None:
    """Write a PNG of the monthly mean valid fraction.

    Rendering uses the non-interactive Agg backend, so the function works on
    headless systems.
    """
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    monthly = daily["valid_fraction"].resample("MS").mean()
    fig, ax = plt.subplots(figsize=(11, 3.4))
    ax.plot(monthly.index, monthly.values, marker=".", linewidth=1)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Monthly mean valid fraction")
    ax.set_xlabel("UTC month (2020–2025)")
    ax.set_title("DWD global radiation: ten-minute measurement completeness/QN≥2")
    ax.grid(alpha=0.3)
    fig.tight_layout()
    target.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(target, dpi=160)
    plt.close(fig)


def run(root: Path, station_id: str) -> dict:
    """Download and audit one DWD station archive under ``root``.

    Fetches the station list and the station's historical ZIP from the DWD
    CDC server, runs the QC pipeline and writes:

    - ``data/raw/``: untouched downloads (station list, station ZIP)
    - ``data/processed/dwd_<id>_qc_<first>_<last>.csv.gz``: time-aligned,
      QC-flagged 10-minute table (missing intervals remain NaN)
    - ``reports/dwd_<id>_validation.json``: provenance (URL, SHA-256),
      measurement conventions and aggregate QC counts
    - ``reports/dwd_<id>_yearly.csv``, ``reports/dwd_<id>_daily.csv``:
      per-year and per-day coverage tables
    - ``figures/dwd_<id>_monthly_coverage.png``: coverage figure

    Returns the validation-report dictionary. ``station_id`` must be exactly
    five digits.
    """
    if not re.fullmatch(r"\d{5}", station_id):
        raise ValueError("Station ID must contain 5 digits, e.g. 00342")
    raw_dir, report_dir, processed_dir, fig_dir = (
        root / "data/raw", root / "reports", root / "data/processed", root / "figures"
    )
    raw_dir.mkdir(parents=True, exist_ok=True)
    station_list = raw_dir / STATION_LIST_NAME
    download(ARCHIVE_ROOT + STATION_LIST_NAME, station_list)
    metadata = extract_station_metadata(station_list.read_bytes(), station_id)
    request = urllib.request.Request(ARCHIVE_ROOT, headers={"User-Agent": USER_AGENT})
    with urllib.request.urlopen(request, timeout=45) as response:
        archive_name = discover_archive(station_id, response.read().decode("utf-8", errors="replace"))
    archive = raw_dir / archive_name
    download(ARCHIVE_ROOT + archive_name, archive)
    raw = read_solar_zip(archive)
    grid, daily, stats = normalize_observations(raw)
    yearly = yearly_summary(grid, daily)
    report = {
        "station": metadata,
        "archive_url": ARCHIVE_ROOT + archive_name,
        "archive_sha256": hashlib.sha256(archive.read_bytes()).hexdigest(),
        "interval": "10 minutes; measurement timestamp is END; midpoint=end-5min UTC",
        "unit_conversion": "GS_10 J/cm^2 per 600 s -> average W/m^2: multiply by 10000/600",
        "missing_and_quality": "-999 -> missing; negative -> invalid; accept only QN=2 or QN=3",
        "screening": "98% valid intervals per UTC day (night included), no duplicate timestamps; completeness screen, not a daylight-based inclusion rule",
        "time_window": f"{FIRST_YEAR}-01-01 through {LAST_YEAR}-12-31 UTC midpoint dates",
        "counts": stats,
    }
    report_dir.mkdir(parents=True, exist_ok=True)
    processed_dir.mkdir(parents=True, exist_ok=True)
    with (report_dir / f"dwd_{station_id}_validation.json").open("w") as f:
        json.dump(report, f, indent=2)
    yearly.to_csv(report_dir / f"dwd_{station_id}_yearly.csv", index_label="year")
    daily.to_csv(report_dir / f"dwd_{station_id}_daily.csv", index_label="midpoint_day_utc")
    # Time-aligned, QC-flagged table; missing intervals remain NaN (no imputation).
    grid[["timestamp_end_utc", "GS_10", "QN", "ghi_wm2", "record_present",
          "measurement_valid", "missing_sentinel", "negative_gs", "suspicious_high"]].to_csv(
        processed_dir / f"dwd_{station_id}_qc_{FIRST_YEAR}_{LAST_YEAR}.csv.gz",
        compression="gzip", index_label="midpoint_utc"
    )
    save_coverage_figure(daily, fig_dir / f"dwd_{station_id}_monthly_coverage.png")
    return report
