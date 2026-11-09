#!/usr/bin/env python
"""
Cut small s7k test files from large ones.

Records are copied unchanged, in file order, from the start of the source
file until the given number of distinct pings has been seen. Records of the
types given with --also are copied from anywhere in the file (for example the
7777 filler records that sit at the end). Any 7300 file catalog in the source
is dropped, and a new 7300 catalog is written for the subset, with the 7200
file header's optional-data pointer set to it.

Usage:
    python scripts/make_test_subsets.py SOURCE DEST --pings 5 [--also 7777]
    python scripts/make_test_subsets.py --defaults     # the repository's test files
"""

import argparse
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), ".."))

from PYS7K import s7k  # noqa: E402

DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data")

# (source, destination name, pings, extra record types[, start record type])
DEFAULTS = [
    ("/Users/vschmidt/Projects/RDL/2016_Red_Bird_Reef_FH_Survey/FHRAW/Bathymetry/"
     "FH_S250_400kHz_7125_512bms/Port/2016-041/2016__0411126.HSX.s7k",
     "hsx_7125_7006_7008.s7k", 5, []),
    ("/Users/vschmidt/library.local/PapersPostersPresentations/Proposals/SESAME_CollaborativeMapping/"
     "Proposal/figures/working_data/onrbedforms/20121110_153453.s7k",
     "onr_7125_7027_7028_7008.s7k", 3, [7777], 7027),
    ("/Users/vschmidt/Projects/UAM/7125_IQ/20121128_193420.s7k",
     "uam_7125_7027_7028_nav.s7k", 5, [7777]),
    ("/Users/vschmidt/Projects/UAM/7125_IQ/20121128_194708.s7k",
     "uam_7125_7038_iq.s7k", 2, [7777], 7038),
]


def cut(source, dest, pings, also=(), start_type=None):
    """
    Write the subset of ``source`` to ``dest``. Returns records written.

    :param start_type: start at the first ping that has a record of this
        type. The records before the file's first ping (7200, 7022, 7001,
        ...) are always kept.
    """
    with s7k(source) as src:
        idx = src.index_file(full=True)
        has_ping = (idx.PingNumber >= 0).to_numpy()
        first_ping_row = int(idx.index[has_ping][0]) if has_ping.any() else len(idx)
        wanted = [r for r in idx.index[:first_ping_row] if idx.RecordType[r] != 7300]
        start_row = first_ping_row
        if start_type is not None:
            start_ping = idx.PingNumber[idx.RecordType == start_type].iloc[0]
            start_row = int(idx.index[(idx.PingNumber == start_ping).to_numpy()][0])
        seen = []
        for row, rt, ping in zip(idx.index[start_row:], idx.RecordType[start_row:], idx.PingNumber[start_row:]):
            if rt == 7300:
                continue
            if ping >= 0 and ping not in seen:
                if len(seen) == pings:
                    break
                seen.append(ping)
            wanted.append(row)
        last = wanted[-1] if wanted else -1
        extra = [r for r in idx.index[idx.RecordType.isin(list(also))] if r > last]
        dst = s7k(dest)
        dst.OpenFiletoWrite()
        try:
            for row in wanted + extra:
                dst.write_raw(src.read_record_bytes(row))
        finally:
            dst.closeFile(catalog=True)
    return len(wanted) + len(extra) + 1


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("source", nargs="?")
    p.add_argument("dest", nargs="?")
    p.add_argument("--pings", type=int, default=5)
    p.add_argument("--also", type=int, action="append", default=[])
    p.add_argument("--start-type", type=int, default=None,
                   help="start at the first ping having a record of this type")
    p.add_argument("--defaults", action="store_true", help="write the standard test subsets to data/")
    args = p.parse_args(argv)
    jobs = []
    if args.defaults:
        os.makedirs(DATA_DIR, exist_ok=True)
        jobs = [(job[0], os.path.join(DATA_DIR, job[1])) + tuple(job[2:]) for job in DEFAULTS]
    elif args.source and args.dest:
        jobs = [(args.source, args.dest, args.pings, args.also, args.start_type)]
    else:
        p.error("give SOURCE and DEST, or --defaults")
    for job in jobs:
        source, dest = job[0], job[1]
        if not os.path.exists(source):
            print("skipping (not found): %s" % source)
            continue
        count = cut(source, dest, *job[2:])
        print("%s: %d records, %d bytes" % (dest, count, os.path.getsize(dest)))


if __name__ == "__main__":
    main()
