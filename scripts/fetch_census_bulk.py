#!/usr/bin/env python
"""Download the Census bulk Port/HS6 monthly archives the lattice builder reads.

Until now the benchmark's pipeline began with files no script in this
repository could obtain, which is why the built lattice had only ever existed
on one machine. This closes that gap.

NOT to be confused with scripts/fetch_census_ports.py, which calls the
international-trade *API*, defaults to HS2, and writes CSVs into
``data/census_ports/raw/`` (with an "s") that nothing downstream reads. This
script fetches the *bulk* distribution -- fixed-width records inside per-month
ZIPs -- which is what ``scripts/build_census_lattice.py`` actually parses.

    https://www.census.gov/trade/downloads/<YYYY>/Port/im_hs6_m/PORTHS6MM<yy><mm>.ZIP
    https://www.census.gov/trade/downloads/<YYYY>/Port/ex_hs6_m/PORTHS6XM<yy><mm>.ZIP

Free and public; no API key. The directory names and filename prefixes come
from build_census_lattice.FLOWS rather than being repeated here, so the two
cannot drift apart.

Also fetches Schedule D (district -> state), which the builder needs to map
the 2-digit customs district in each record onto a state.

A REBUILD IS NOT BIT-IDENTICAL TO AN OLDER ONE. Census revises trade data and
re-issues the monthly archives, and the Schedule D file is a dated snapshot
(the one fetched 2026-08 is stamped "ProducedAPRIL25"). Build the lattice once
and train the whole matrix against that build. Mixing a new lattice with old
checkpoints is caught rather than silently accepted -- run_input_fingerprint
covers the NPZ's size and mtime and content-hashes its sidecar -- but it costs
a retrain, so do not rebuild mid-sweep.

    python scripts/fetch_census_bulk.py --start 2010-01 --end 2025-12 \
        --out data/census_port/raw --ref data/census_port/reference
"""

import argparse
import os
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
import urllib.error
import urllib.request
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO))
from scripts.build_census_lattice import FLOWS      # noqa: E402

BULK = "https://www.census.gov/trade/downloads/{year}/Port/{folder}/{prefix}{yy}{mm}.ZIP"
SCHEDULE_D = "https://www.census.gov/foreign-trade/schedules/d/dist3.txt"
# The builder opens this exact name under --ref.
SCHEDULE_D_NAME = "scheduleD_dist3.txt"


def months(start, end):
    y0, m0 = (int(part) for part in start.split("-"))
    y1, m1 = (int(part) for part in end.split("-"))
    y, m = y0, m0
    while (y, m) <= (y1, m1):
        yield y, m
        m += 1
        if m == 13:
            y, m = y + 1, 1


# census.gov returns 403 to urllib's default User-Agent while serving the same
# URL fine to curl. Found by testing the fetcher rather than trusting a HEAD
# check made with a different client.
_UA = ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) "
       "Chrome/124.0 Safari/537.36")


def fetch(url, dest, retries=3, timeout=180):
    """Download to a temporary name and rename, so an interrupted transfer
    never leaves a truncated archive that looks complete to --resume."""
    tmp = dest.with_suffix(dest.suffix + ".part")
    request = urllib.request.Request(url, headers={"User-Agent": _UA})
    for attempt in range(1, retries + 1):
        try:
            with urllib.request.urlopen(request, timeout=timeout) as response:
                tmp.write_bytes(response.read())
            tmp.replace(dest)
            return dest.stat().st_size
        except (urllib.error.URLError, OSError) as exc:
            tmp.unlink(missing_ok=True)
            if attempt == retries:
                raise RuntimeError(f"{url}: {exc}") from exc
            time.sleep(2 * attempt)


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--start", default="2010-01", help="first month, YYYY-MM")
    ap.add_argument("--end", default="2025-12", help="last month, YYYY-MM")
    ap.add_argument("--out", default="data/census_port/raw",
                    help="root for the im_hs6_m/ and ex_hs6_m/ folders")
    ap.add_argument("--ref", default="data/census_port/reference",
                    help="where Schedule D is written")
    ap.add_argument("--force", action="store_true",
                    help="re-download archives already present")
    ap.add_argument("--dry-run", action="store_true",
                    help="list what would be fetched and exit")
    ap.add_argument("--jobs", type=int, default=8,
                    help="parallel downloads. The archives are 12-15 MB each "
                         "and 384 of them serially is about two hours, which "
                         "is long enough that an interrupted run matters; "
                         "each file is still written atomically via .part so "
                         "concurrency cannot leave a truncated archive.")
    args = ap.parse_args(argv)

    out_root = Path(args.out)
    plan = []
    for flow, (folder, prefix) in sorted(FLOWS.items()):
        for year, month in months(args.start, args.end):
            name = f"{prefix}{str(year)[2:]}{month:02d}.ZIP"
            dest = out_root / folder / name
            url = BULK.format(year=year, folder=folder, prefix=prefix,
                              yy=str(year)[2:], mm=f"{month:02d}")
            plan.append((flow, url, dest))

    todo = [item for item in plan
            if args.force or not item[2].exists() or item[2].stat().st_size == 0]
    print(f"{len(plan)} archives in range; {len(todo)} to fetch "
          f"({len(plan) - len(todo)} already present)")
    if args.dry_run:
        for _, url, dest in todo[:6]:
            print(f"  {url}\n    -> {dest}")
        if len(todo) > 6:
            print(f"  ... and {len(todo) - 6} more")
        return 0

    ref_dir = Path(args.ref)
    ref_dir.mkdir(parents=True, exist_ok=True)
    schedule_d = ref_dir / SCHEDULE_D_NAME
    if args.force or not schedule_d.exists():
        size = fetch(SCHEDULE_D, schedule_d)
        print(f"Schedule D -> {schedule_d} ({size:,} bytes)")

    for _, _, dest in todo:
        dest.parent.mkdir(parents=True, exist_ok=True)
    failures = []
    lock = threading.Lock()
    done = 0

    def one(item):
        flow, url, dest = item
        return flow, dest, fetch(url, dest)

    with ThreadPoolExecutor(max_workers=max(1, args.jobs)) as pool:
        futures = {pool.submit(one, item): item for item in todo}
        for future in as_completed(futures):
            flow, url, dest = futures[future]
            with lock:
                done += 1
                try:
                    _, _, size = future.result()
                    print(f"[{done}/{len(todo)}] {flow:6} {dest.name}  "
                          f"{size / 1e6:.1f} MB", flush=True)
                except RuntimeError as exc:
                    # A missing month is unknown data, not zero trade -- report
                    # it and let the builder refuse, rather than leaving a hole.
                    failures.append(str(exc))
                    print(f"[{done}/{len(todo)}] {flow:6} {dest.name}  FAILED",
                          flush=True)

    if failures:
        print(f"\n{len(failures)} archive(s) could not be fetched:")
        for line in failures[:10]:
            print(f"  {line}")
        print("build_census_lattice.py will refuse to build until these exist.")
        return 1
    print(f"\nAll archives present under {out_root}")
    print(f"Next: python scripts/build_census_lattice.py --base {out_root} "
          f"--ref {ref_dir} --out data/census_port/processed")
    return 0


if __name__ == "__main__":
    sys.exit(main())
