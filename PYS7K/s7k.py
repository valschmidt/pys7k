"""
pys7k: a pure-Python reader and writer for Teledyne RESON s7k files
(7k Data Format Definition, version 3.10).

Reading::

    from PYS7K import s7k
    with s7k("20121128_193420.s7k") as S:
        S.index_file()                       # pandas DataFrame in S.Index
        for rec in S.iter_records(7027):
            print(rec["RTH"]["PingNumber"], rec["RD"]["Detections"]["DetectionPoint"][:5])

Records are dictionaries holding exactly what is stored in the file (see
``PYS7K.records.base``). Converted values (degrees, two-way travel times,
datetimes, decoded bit fields) come from the batch helpers in
``PYS7K.derived``.

The file is memory-mapped. Indexing walks the record frames with one
precompiled ``struct`` unpack per record, then gathers every frame field
for every record at once with numpy.
"""

import argparse
import mmap
import os
import struct
import sys

import numpy as np
import pandas as pd

from .frame import (CHECKSUM_SIZE, DRF_DTYPE, DRF_SIZE, FLAG_CHECKSUM_VALID,
                    SYNC_BYTES, SYNC_PATTERN, S7KChecksumError, S7KError,
                    S7KFormatError, checksum, time7k_to_epoch)
from .registry import (DEVICE_IDENTIFIERS, PING_RECORD_TYPES, RECORD_CODECS,
                       RECORD_TYPE_DESCRIPTIONS, decode_record, encode_record,
                       new, record_name)

__all__ = ["s7k", "resolve_record_type", "main"]

_FRAME_HEAD = struct.Struct("<HHII")   # ProtocolVersion, Offset, SyncPattern, Size
_MIN_RECORD = DRF_SIZE + CHECKSUM_SIZE

# Records whose RTH starts with SonarId (u64) then PingNumber (u32).
_PING_NUMBER_TYPES = frozenset(PING_RECORD_TYPES | {7002, 7038, 7503})


def resolve_record_type(value):
    """
    Turn a record type given as an int, a numeric string, or a name
    ("7027", "Raw Detection Data", "raw detection data") into its integer
    identifier. ``None`` passes through.

    :raises ValueError: no record type has that name.
    """
    if value is None:
        return None
    if isinstance(value, (int, np.integer)):
        return int(value)
    text = str(value).strip()
    if text.isdigit():
        return int(text)
    key = text.lower().replace("_", " ")
    for rt, name in RECORD_TYPE_DESCRIPTIONS.items():
        if name.lower() == key:
            return rt
    raise ValueError("unknown record type %r" % value)


def _resolve_types(types):
    if types is None:
        return None
    if isinstance(types, (str, int, np.integer)):
        types = [types]
    return {resolve_record_type(t) for t in types}


class s7k:
    """
    An s7k file, for reading or writing.

    :param filename: the file's path.
    """

    def __init__(self, filename):
        self.filename = filename
        self.FID = None
        self._mmap = None
        self.file_size = None
        self.Index = None
        self.Gaps = []
        self._write_entries = None
        self._mode = None

    # ------------------------------------------------------------------
    # Opening and closing
    # ------------------------------------------------------------------

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.closeFile()

    def OpenFiletoRead(self):
        """Open the file for reading and memory-map it."""
        self.closeFile()
        self.FID = open(self.filename, "rb")
        self.file_size = os.fstat(self.FID.fileno()).st_size
        self._mmap = mmap.mmap(self.FID.fileno(), 0, access=mmap.ACCESS_READ) if self.file_size else b""
        self._mode = "r"

    def OpenFiletoWrite(self):
        """
        Open the file for writing (truncating it). Records are added with
        ``write_record()``. ``closeFile(catalog=True)`` appends a 7300 file
        catalog.
        """
        self.closeFile()
        self.FID = open(self.filename, "wb")
        self._write_entries = []
        self._mode = "w"

    def closeFile(self, catalog=False):
        """
        Close the file.

        :param catalog: when writing, append a 7300 file catalog listing
            every record written, and point the 7200 file header's optional
            data at it (if the first record is a 7200 with optional data).
        """
        if self.FID is None:
            return
        if self._mode == "w" and catalog:
            self._write_catalog()
        if isinstance(self._mmap, mmap.mmap):
            self._mmap.close()
        self._mmap = None
        self.FID.close()
        self.FID = None
        self._mode = None

    def _ensure_read(self):
        if self._mode != "r":
            self.OpenFiletoRead()

    # ------------------------------------------------------------------
    # Indexing
    # ------------------------------------------------------------------

    def _resync(self, start, protocol):
        """
        Find the next plausible record start at or after ``start``: a sync
        pattern preceded by the file's protocol version, with a size that
        fits in the file.
        """
        m, n = self._mmap, self.file_size
        pos = start + 4
        while True:
            idx = m.find(SYNC_BYTES, pos)
            if idx < 0:
                return None
            cand = idx - 4
            if cand >= start and cand + _MIN_RECORD <= n:
                pv, _, _, size = _FRAME_HEAD.unpack_from(m, cand)
                if pv == protocol and size >= _MIN_RECORD and cand + size <= n:
                    return cand
            pos = idx + 1

    def _walk(self, strict):
        m, n = self._mmap, self.file_size
        unpack = _FRAME_HEAD.unpack_from
        offsets = []
        append = offsets.append
        gaps = []
        protocol = None
        pos = 0
        while pos + _MIN_RECORD <= n:
            pv, _, sync, size = unpack(m, pos)
            if sync == SYNC_PATTERN and size >= _MIN_RECORD and pos + size <= n:
                if protocol is None:
                    protocol = pv
                append(pos)
                pos += size
                continue
            if strict:
                raise S7KFormatError(
                    "bad record frame at offset %d (sync=0x%08X, size=%d)" % (pos, sync, size))
            nxt = self._resync(pos + 1, protocol if protocol is not None else 5)
            gaps.append((pos, (nxt if nxt is not None else n) - pos))
            if nxt is None:
                pos = n
                break
            pos = nxt
        if pos < n:
            if strict:
                raise S7KFormatError("%d trailing bytes at offset %d" % (n - pos, pos))
            gaps.append((pos, n - pos))
        return np.asarray(offsets, dtype=np.int64), gaps

    def index_file(self, full=False, verify_checksums=False, strict=False):
        """
        Index every record in the file.

        :param full: also gather PingNumber and MultiPingSequence for every
            ping record (-1 where not applicable).
        :param verify_checksums: add a ``ChecksumValid`` column: True or
            False for records whose Flags bit 0 says a checksum is present,
            missing (NA) otherwise.
        :param strict: raise ``S7KFormatError`` at the first malformed frame
            instead of resynchronizing on the next sync pattern. Skipped
            byte ranges are listed in ``self.Gaps`` as (offset, length).

        :return: ``self.Index``, a pandas DataFrame with one row per record.
        """
        self._ensure_read()
        offsets, self.Gaps = self._walk(strict)
        cols = {"Offset": offsets}
        if len(offsets):
            file_bytes = np.frombuffer(self._mmap, dtype=np.uint8)
            try:
                frames = file_bytes[offsets[:, None] + np.arange(DRF_SIZE)].view(DRF_DTYPE).reshape(-1)
                cols["Size"] = frames["Size"].astype(np.int64)
                cols["RecordType"] = frames["RecordTypeIdentifier"].astype(np.int64)
                cols["DeviceId"] = frames["DeviceIdentifier"].astype(np.int64)
                cols["SystemEnumerator"] = frames["SystemEnumerator"].astype(np.int64)
                cols["Time"] = time7k_to_epoch(frames["Year"], frames["Day"], frames["Hours"],
                                               frames["Minutes"], frames["Seconds"])
                cols["ProtocolVersion"] = frames["ProtocolVersion"].astype(np.int64)
                cols["RecordVersion"] = frames["RecordVersion"].astype(np.int64)
                cols["Flags"] = frames["Flags"].astype(np.int64)
                cols["OptionalDataOffset"] = frames["OptionalDataOffset"].astype(np.int64)
                cols["OptionalDataIdentifier"] = frames["OptionalDataIdentifier"].astype(np.int64)
                if full:
                    cols.update(self._gather_ping_numbers(file_bytes, offsets, cols))
                if verify_checksums:
                    cols["ChecksumValid"] = self._verify_checksums(file_bytes, offsets, cols)
            finally:
                # Release the view before anything can close the mapping.
                del file_bytes
        else:
            for name in ("Size", "RecordType", "DeviceId", "SystemEnumerator", "ProtocolVersion",
                         "RecordVersion", "Flags", "OptionalDataOffset", "OptionalDataIdentifier"):
                cols[name] = np.zeros(0, np.int64)
            cols["Time"] = np.zeros(0)
        self.Index = pd.DataFrame(cols)
        return self.Index

    @staticmethod
    def _gather_ping_numbers(file_bytes, offsets, cols):
        rt = cols["RecordType"]
        ping = np.full(len(offsets), -1, dtype=np.int64)
        multi = np.full(len(offsets), -1, dtype=np.int64)
        rth = offsets + 4 + (file_bytes[(offsets + 2)[:, None] + np.arange(2)].view("<u2").reshape(-1))
        has_ping = np.isin(rt, list(_PING_NUMBER_TYPES)) & (rth + 14 <= offsets + cols["Size"])
        idx = np.flatnonzero(has_ping)
        if len(idx):
            ping[idx] = file_bytes[(rth[idx] + 8)[:, None] + np.arange(4)].view("<u4").reshape(-1)
        has_multi = np.isin(rt, list(PING_RECORD_TYPES)) & has_ping
        idx = np.flatnonzero(has_multi)
        if len(idx):
            multi[idx] = file_bytes[(rth[idx] + 12)[:, None] + np.arange(2)].view("<u2").reshape(-1)
        return {"PingNumber": ping, "MultiPingSequence": multi}

    @staticmethod
    def _verify_checksums(file_bytes, offsets, cols):
        # One numpy sum per record. This measured 17x faster than a single
        # np.add.reduceat over the whole file (0.26 s vs 4.6 s per GB).
        sizes = cols["Size"]
        ends = offsets + sizes - CHECKSUM_SIZE
        sums = np.fromiter((file_bytes[o:e].sum(dtype=np.uint64) for o, e in zip(offsets.tolist(), ends.tolist())),
                           dtype=np.uint64, count=len(offsets)) & np.uint64(0xFFFFFFFF)
        stored = file_bytes[ends[:, None] + np.arange(4)].view("<u4").reshape(-1)
        valid = pd.array(sums == stored, dtype="boolean")
        valid[(cols["Flags"] & FLAG_CHECKSUM_VALID) == 0] = pd.NA
        return valid

    def check_catalog(self):
        """
        Compare the file's 7300 catalog (if any) with the index.

        :return: a dictionary with ``"catalog_records"`` (entries in the
            7300), ``"indexed_records"`` (records indexed, excluding the
            7300 itself), ``"mismatches"`` (a DataFrame of entries whose
            offset, size or record type disagree with the index),
            ``"missing"`` (indexed records the catalog does not list) and
            ``"file_header_pointer"`` (the 7200 optional data, if present).
        """
        if self.Index is None:
            self.index_file()
        idx = self.Index
        cat_rows = idx[idx.RecordType == 7300]
        result = {"catalog_records": 0, "indexed_records": int((idx.RecordType != 7300).sum()),
                  "mismatches": pd.DataFrame(), "missing": pd.DataFrame(), "file_header_pointer": None}
        hdr = idx[idx.RecordType == 7200]
        if len(hdr):
            rec = self.read_record(hdr.index[0])
            result["file_header_pointer"] = rec.get("OptionalData")
        if not len(cat_rows):
            return result
        cat = self.read_record(cat_rows.index[-1])
        entries = pd.DataFrame({k: np.asarray(v) for k, v in cat["RD"]["Entries"].items()})
        result["catalog_records"] = len(entries)
        merged = entries.merge(idx[["Offset", "Size", "RecordType"]], on="Offset", how="outer",
                               suffixes=("Catalog", "Index"), indicator=True)
        merged = merged[merged.RecordTypeIndex != 7300]
        both = merged[merged._merge == "both"]
        bad = both[(both.SizeCatalog != both.SizeIndex) | (both.RecordTypeCatalog != both.RecordTypeIndex)]
        result["mismatches"] = pd.concat([bad, merged[merged._merge == "left_only"]])
        result["missing"] = merged[merged._merge == "right_only"]
        return result

    # ------------------------------------------------------------------
    # Reading records
    # ------------------------------------------------------------------

    def read_record_bytes(self, i):
        """The raw bytes of record ``i`` (a row label of ``self.Index``)."""
        if self.Index is None:
            self.index_file()
        self._ensure_read()
        off = int(self.Index.at[i, "Offset"])
        size = int(self.Index.at[i, "Size"])
        return bytearray(self._mmap[off:off + size])

    def read_record(self, i):
        """Decode record ``i`` (a row label of ``self.Index``)."""
        return decode_record(self.read_record_bytes(i))

    def iter_records(self, record_types=None, devices=None, start_time=None, end_time=None,
                     raw=False):
        """
        Iterate over records, in file order.

        :param record_types: a record type or collection of types (ints or
            names). All types if omitted.
        :param devices: a device identifier or collection of them.
        :param start_time, end_time: POSIX seconds bounds on the frame time.
        :param raw: yield ``(index_row, bytearray)`` instead of decoded
            records.

        Each decoded record owns a private copy of its bytes, so its arrays
        stay valid after the file is closed and can be modified.
        """
        if self.Index is None:
            self.index_file()
        self._ensure_read()
        idx = self.Index
        mask = np.ones(len(idx), dtype=bool)
        types = _resolve_types(record_types)
        if types is not None:
            mask &= idx.RecordType.isin(types).to_numpy()
        if devices is not None:
            devs = {int(devices)} if np.isscalar(devices) else {int(d) for d in devices}
            mask &= idx.DeviceId.isin(devs).to_numpy()
        if start_time is not None:
            mask &= (idx.Time >= start_time).to_numpy()
        if end_time is not None:
            mask &= (idx.Time <= end_time).to_numpy()
        m = self._mmap
        for row, off, size in zip(idx.index[mask], idx.Offset.to_numpy()[mask], idx.Size.to_numpy()[mask]):
            buf = bytearray(m[off:off + size])
            yield (row, buf) if raw else decode_record(buf)

    def __iter__(self):
        return self.iter_records()

    def __len__(self):
        if self.Index is None:
            self.index_file()
        return len(self.Index)

    # ------------------------------------------------------------------
    # Reports and debugging output
    # ------------------------------------------------------------------

    def report_record_types(self, verify_checksums=False):
        """
        Print, and return, a summary of the file's records: for each record
        type, how many records there are and their total, minimum and
        maximum size in bytes, with the share of the file they occupy.
        This mirrors gsfu's report_record_types() and kmall's
        report_packet_types(), and is what the -V command line flag runs.

        :return: a pandas DataFrame with one row per record type.
        """
        if self.Index is None or (verify_checksums and "ChecksumValid" not in self.Index):
            self.index_file(verify_checksums=verify_checksums)
        idx = self.Index
        grouped = idx.groupby("RecordType")
        summary = pd.DataFrame({
            "Name": [record_name(rt) for rt in grouped.groups],
            "Count": grouped.Size.count(),
            "Total Bytes": grouped.Size.sum(),
            "Min Bytes": grouped.Size.min(),
            "Max Bytes": grouped.Size.max(),
        })
        summary["% of File"] = (summary["Total Bytes"] / max(self.file_size, 1) * 100).round(2)
        summary["Decoded"] = ["yes" if rt in RECORD_CODECS else "raw" for rt in summary.index]
        if "ChecksumValid" in idx:
            cv = idx.ChecksumValid
            summary["Checksum OK"] = cv.groupby(idx.RecordType).apply(lambda s: int((s == True).sum()))  # noqa: E712
            summary["Checksum Bad"] = cv.groupby(idx.RecordType).apply(lambda s: int((s == False).sum()))  # noqa: E712
            summary["No Checksum"] = cv.groupby(idx.RecordType).apply(lambda s: int(s.isna().sum()))
        summary = summary.sort_values("Total Bytes", ascending=False)

        t = idx.Time.dropna()
        print("File: %s" % self.filename)
        print("File size: %d bytes" % self.file_size)
        print("Total records: %d" % len(idx))
        if len(idx):
            pv = sorted(idx.ProtocolVersion.unique())
            print("Protocol version(s): %s" % ", ".join(str(v) for v in pv))
            devices = idx.groupby(["DeviceId", "SystemEnumerator"]).size()
            print("Devices: %s" % ", ".join(
                "%d/%d (%s, %d records)" % (d, e, DEVICE_IDENTIFIERS.get(d, "unknown"), c)
                for (d, e), c in devices.items()))
        if len(t):
            print("Time span: %s to %s (%.1f s)" % (
                pd.to_datetime(t.min(), unit="s", utc=True).isoformat(timespec="microseconds"),
                pd.to_datetime(t.max(), unit="s", utc=True).isoformat(timespec="microseconds"), t.max() - t.min()))
        accounted = int(idx.Size.sum())
        if accounted != self.file_size:
            print("WARNING: %d bytes not accounted for by records (%d gaps)" % (
                self.file_size - accounted, len(self.Gaps)))
            for off, length in self.Gaps[:20]:
                print("    gap at offset %d, %d bytes" % (off, length))
        print()
        with pd.option_context("display.width", 200, "display.max_columns", 20):
            print(summary.to_string())
        return summary

    def print_index(self, out=None):
        """Write the index to ``out`` (default stdout) as CSV."""
        if self.Index is None:
            self.index_file()
        self.Index.to_csv(out if out is not None else sys.stdout, index_label="Record")

    def print_records(self, record_type=None, max_records=None, max_rows=None):
        """
        A debugging utility that prints records as readable text. This is
        what the -p command line flag runs.

        For each record it prints a header line, the frame fields, the
        record type header fields, the record data (scalars as
        "key : value" lines, per-beam data as tables), the optional data,
        any unparsed bytes, and the checksum with whether it verifies.
        Values are printed exactly as stored. Lines starting with "#" are
        convenience annotations (UTC time, device name, decoded text).
        A record type with no decoder is printed as ASCII text, with each
        non-printable byte shown as '.'.

        :param record_type: a record type (int or name), or a collection of
            them, to restrict the output to. All records if omitted.
        :param max_records: stop after this many records.
        :param max_rows: print at most this many rows of each table.
        """
        count = 0
        for row, buf in self.iter_records(record_type, raw=True):
            rec = decode_record(buf)
            _print_record(rec, row, int(self.Index.at[row, "Offset"]), buf, max_rows)
            print()
            count += 1
            if max_records is not None and count >= max_records:
                break

    # ------------------------------------------------------------------
    # Writing
    # ------------------------------------------------------------------

    def write_record(self, record, checksum_mode="auto"):
        """
        Encode a record dictionary and append it to the file.

        :param record: a record from ``iter_records()``/``read_record()``,
            or a template from ``PYS7K.new(record_type)``.
        :param checksum_mode: see ``registry.encode_record``.
        :return: the number of bytes written.
        """
        return self.write_raw(encode_record(record, checksum_mode))

    def write_raw(self, data):
        """Append one already-encoded record (bytes) to the file."""
        if self._mode != "w":
            raise S7KError("file is not open for writing; call OpenFiletoWrite()")
        data = bytes(data)
        if len(data) < _MIN_RECORD:
            raise S7KFormatError("record too short (%d bytes)" % len(data))
        frame = np.frombuffer(data, dtype=DRF_DTYPE, count=1)[0]
        self._write_entries.append((self.FID.tell(), len(data), frame))
        self.FID.write(data)
        return len(data)

    def _write_catalog(self):
        entries = [e for e in self._write_entries if int(e[2]["RecordTypeIdentifier"]) != 7300]
        cat = new(7300)
        cat["Frame"]["Flags"] = 0
        n = len(entries)
        cols = RECORD_CODECS[7300].entries.new(n)
        for i, (off, size, fr) in enumerate(entries):
            cols["Size"][i] = size
            cols["Offset"][i] = off
            cols["RecordType"][i] = int(fr["RecordTypeIdentifier"]) & 0xFFFF
            cols["DeviceIdentifier"][i] = int(fr["DeviceIdentifier"]) & 0xFFFF
            cols["SystemEnumerator"][i] = fr["SystemEnumerator"]
            for k in ("Year", "Day", "Seconds", "Hours", "Minutes"):
                cols[k][i] = fr[k]
            cols["RecordCount"][i] = fr["TotalRecordsInFragmentedSet"]
        if n:
            last = entries[-1][2]
            for k in ("Year", "Day", "Seconds", "Hours", "Minutes"):
                cat["Frame"][k] = last[k].item()
        cat["RTH"]["NumberOfRecords"] = n
        cat["RD"]["Entries"] = cols
        cat_offset = self.FID.tell()
        data = encode_record(cat)
        self.FID.write(data)
        self._write_entries.append((cat_offset, len(data), np.frombuffer(data, DRF_DTYPE, 1)[0]))
        # Point the 7200 file header's optional data at the catalog.
        first = self._write_entries[0] if self._write_entries else None
        if first is not None and int(first[2]["RecordTypeIdentifier"]) == 7200:
            off0, size0, fr0 = first
            odo = int(fr0["OptionalDataOffset"])
            if odo and size0 - CHECKSUM_SIZE - odo >= 12:
                self.FID.flush()
                with open(self.filename, "r+b") as fh:
                    fh.seek(off0)
                    rec_bytes = bytearray(fh.read(size0))
                    struct.pack_into("<IQ", rec_bytes, odo, len(data), cat_offset)
                    if int(fr0["Flags"]) & FLAG_CHECKSUM_VALID:
                        struct.pack_into("<I", rec_bytes, size0 - CHECKSUM_SIZE,
                                         checksum(rec_bytes[:size0 - CHECKSUM_SIZE]))
                    fh.seek(off0)
                    fh.write(rec_bytes)


# ----------------------------------------------------------------------
# Text rendering for print_records()
# ----------------------------------------------------------------------

def _ascii(data):
    return "".join(chr(b) if 32 <= b < 127 else "." for b in bytes(data))


def _fmt_value(v):
    if isinstance(v, (bytes, bytearray)):
        text = bytes(v)
        stripped = text.rstrip(b"\x00")
        if not stripped:
            return "(%d zero bytes)" % len(text)
        return '"%s"%s' % (_ascii(stripped), "" if len(stripped) == len(text) else
                           "  (+%d NUL)" % (len(text) - len(stripped)))
    if isinstance(v, np.generic):
        v = v.item()
    if isinstance(v, float):
        return repr(v)
    return str(v)


def _print_scalars(d, indent="  "):
    items = [(k, v) for k, v in d.items()
             if not isinstance(v, (dict, list, np.ndarray))]
    if not items:
        return
    width = max(len(k) for k, _ in items)
    for k, v in items:
        print("%s%-*s : %s" % (indent, width, k, _fmt_value(v)))


def _print_table(name, table, max_rows):
    cols = {k: np.asarray(v) for k, v in table.items()}
    one_d = {k: v for k, v in cols.items() if v.ndim == 1}
    lengths = {len(v) for v in one_d.values()}
    if one_d and len(lengths) == 1:
        n = lengths.pop()
        print("-- %s (%d rows) --" % (name, n))
        frame = pd.DataFrame(one_d)
        if max_rows is not None and n > max_rows:
            frame = frame.head(max_rows)
            print(frame.to_string())
            print("  ... %d more rows" % (n - max_rows))
        elif n:
            print(frame.to_string())
    else:
        for k, v in one_d.items():
            _print_array("%s.%s" % (name, k), v, max_rows)
    for k, v in cols.items():
        if v.ndim > 1:
            _print_array("%s.%s" % (name, k), v, max_rows)


def _print_array(name, arr, max_rows):
    arr = np.asarray(arr)
    if arr.ndim == 1:
        shown = arr if max_rows is None else arr[:max_rows]
        print("-- %s (%d values, %s) --" % (name, len(arr), arr.dtype))
        if len(arr):
            print(np.array2string(shown, threshold=sys.maxsize, max_line_width=120))
            if len(shown) < len(arr):
                print("  ... %d more values" % (len(arr) - len(shown)))
    else:
        print("-- %s (array %s, %s) --" % (name, "x".join(str(s) for s in arr.shape), arr.dtype))
        if arr.size:
            print("  min %s  max %s  mean %.6g" % (arr.min(), arr.max(), float(arr.mean())))
            rows = arr if max_rows is None else arr[:max_rows]
            with np.printoptions(threshold=sys.maxsize, linewidth=120):
                print(rows)
            if len(rows) < len(arr):
                print("  ... %d more rows" % (len(arr) - len(rows)))


def _print_section(title, d, max_rows):
    if d is None:
        return
    if any(not isinstance(v, (dict, list, np.ndarray)) for v in d.values()) or not d:
        print("-- %s --" % title)
        _print_scalars(d)
    for k, v in d.items():
        if isinstance(v, dict):
            _print_table("%s.%s" % (title, k), v, max_rows)
        elif isinstance(v, np.ndarray):
            _print_array("%s.%s" % (title, k), v, max_rows)
        elif isinstance(v, list):
            for i, item in enumerate(v):
                print("-- %s.%s[%d] --" % (title, k, i))
                if isinstance(item, dict):
                    _print_scalars(item, "    ")
                    for kk, vv in item.items():
                        if isinstance(vv, (dict, np.ndarray)):
                            _print_section("%s.%s[%d].%s" % (title, k, i, kk), vv if isinstance(vv, dict)
                                           else {kk: vv}, max_rows)
                else:
                    print("    %s" % _fmt_value(item))


def _print_record(rec, row, offset, buf, max_rows):
    rt = rec["RecordType"]
    print("=== %d %s  record=%s  offset=%d  size=%d ===" % (rt, record_name(rt), row, offset, len(buf)))
    frame = rec["Frame"]
    print("-- Frame --")
    _print_scalars(frame)
    t = time7k_to_epoch(frame["Year"], frame["Day"], frame["Hours"], frame["Minutes"], frame["Seconds"])
    if t == t:
        print("  # time: %s" % pd.to_datetime(t, unit="s", utc=True).isoformat(timespec="microseconds"))
    print("  # device: %s" % DEVICE_IDENTIFIERS.get(frame["DeviceIdentifier"], "unknown"))
    if "Raw" in rec:
        print("  # %s" % rec.get("Error", ""))
        print(_ascii(rec["Raw"][DRF_SIZE:-CHECKSUM_SIZE]))
        return
    if "Error" in rec:
        print("  # %s" % rec["Error"])
    if rt in RECORD_CODECS:
        _print_section("RTH", rec["RTH"], max_rows)
        if rec["RD"]:
            _print_section("RD", rec["RD"], max_rows)
    if rec["OptionalData"] is not None:
        _print_section("OptionalData", rec["OptionalData"], max_rows)
    for part, data in rec["Unparsed"].items():
        if data:
            label = "Body (no decoder for this record type)" if (
                part == "Body" and rt not in RECORD_CODECS) else part
            print("-- Unparsed %s: %d bytes --" % (label, len(data)))
            print(_ascii(data))
    stored = rec["Checksum"]
    if frame["Flags"] & FLAG_CHECKSUM_VALID:
        calc = checksum(buf[:len(buf) - CHECKSUM_SIZE])
        status = "verifies" if calc == stored else "MISMATCH (computed %d)" % calc
    else:
        status = "not flagged as valid (Flags bit 0 clear)"
    print("-- Checksum: %d  # %s" % (stored, status))


# ----------------------------------------------------------------------
# Command line
# ----------------------------------------------------------------------

def _epilog():
    lines = ["record types with field-level decoders:"]
    for rt in sorted(RECORD_CODECS):
        lines.append("  %5d  %s" % (rt, RECORD_CODECS[rt].name))
    lines.append("other record types are indexed and printed as raw ASCII text.")
    return "\n".join(lines)


def main(argv=None):
    """Command line interface: ``s7k.py -f FILE [-V] [-p [TYPE]] ...``."""
    parser = argparse.ArgumentParser(
        prog="s7k.py",
        description="Index, summarize and dump Teledyne RESON s7k files (DFD 3.10).",
        epilog=_epilog(), formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("-f", "--file", dest="files", action="append", required=True,
                        help="s7k file to read (may be repeated)")
    parser.add_argument("-V", "--verify", action="store_true",
                        help="index the file and print a table of record types, counts and sizes")
    parser.add_argument("-c", "--checksums", action="store_true",
                        help="with -V, also verify every record's checksum")
    parser.add_argument("-p", "--print", dest="print_type", nargs="?", const="ALL", default=None,
                        metavar="RECORDTYPE",
                        help="print records as text, optionally only one record type (number or name)")
    parser.add_argument("-n", "--max-records", type=int, default=None,
                        help="with -p, print at most this many records")
    parser.add_argument("-r", "--max-rows", type=int, default=None,
                        help="with -p, print at most this many rows of each table")
    parser.add_argument("-i", "--index", action="store_true",
                        help="print the full record index as CSV")
    parser.add_argument("--catalog", action="store_true",
                        help="compare the file's 7300 catalog with the index")
    parser.add_argument("-v", "--verbose", action="count", default=0)
    args = parser.parse_args(argv)

    if not (args.verify or args.print_type or args.index or args.catalog):
        args.verify = True

    status = 0
    for filename in args.files:
        try:
            with s7k(filename) as S:
                if args.verify:
                    S.index_file(full=args.verbose > 0, verify_checksums=args.checksums)
                    S.report_record_types()
                    print()
                if args.index:
                    S.index_file(full=True, verify_checksums=args.checksums)
                    S.print_index()
                if args.catalog:
                    result = S.check_catalog()
                    print("7300 catalog entries: %d; indexed records (excluding 7300): %d" % (
                        result["catalog_records"], result["indexed_records"]))
                    print("7200 catalog pointer: %s" % result["file_header_pointer"])
                    print("mismatched entries: %d; records missing from catalog: %d" % (
                        len(result["mismatches"]), len(result["missing"])))
                    if args.verbose and len(result["mismatches"]):
                        print(result["mismatches"].to_string())
                if args.print_type:
                    types = None if args.print_type.upper() == "ALL" else args.print_type
                    S.print_records(types, max_records=args.max_records, max_rows=args.max_rows)
        except (OSError, S7KError, ValueError) as exc:
            print("%s: %s" % (filename, exc), file=sys.stderr)
            status = 1
    return status


if __name__ == "__main__":
    sys.exit(main())
