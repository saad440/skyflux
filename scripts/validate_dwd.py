"""Download and audit a historical DWD 10-minute solar archive."""
from __future__ import annotations

import argparse
from pathlib import Path
from skyflux.dwd import run


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--station", default="00342", help="Five-digit DWD station ID")
    parser.add_argument("--root", type=Path, default=Path.cwd(), help="Project root")
    args = parser.parse_args()
    report = run(args.root, args.station)
    print("Station:", report["station"])
    print("Download:", report["archive_url"])
    print("QA counts:")
    for key, value in report["counts"].items():
        print(f"  {key}: {value}")
    print(f"Reports: {args.root / 'reports'}")
    print(f"Figure: {args.root / 'figures' / ('dwd_' + args.station + '_monthly_coverage.png')}")


if __name__ == "__main__":
    main()
