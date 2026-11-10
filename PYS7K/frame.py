"""
The 7k Data Record Frame (DRF), 7KTIME, checksums, and errors.

Every s7k record is laid out as::

    DRF (64 bytes) | RTH | RD | OD | checksum (u32)

* The record type header (RTH) starts ``4 + Offset`` bytes from the start
  of the record (64 bytes for every protocol-5 file seen so far).
* The optional data (OD) starts at ``OptionalDataOffset``, or is absent when
  that field is zero.
* The checksum is the last four bytes. It is the sum of every byte before
  it, as an unsigned 32-bit integer.

Checksum flag
-------------
The DFD (3.10, Table 5) says Flags bit 0 means "valid checksum", but the
checksum row of the same table says the checksum "depends on bit 1 of the
Flags field". Bit 0 is correct. The "bit 1" wording is left over from the
protocol 2 and 3 frames (DFD 0.42 and 0.50 say "Bit 1 - Valid checksum");
DFD 0.53 (protocol 4, 2004) moved the flag to bit 0 and every later DFD
keeps both wordings. On every sample file checked (7125 data from
2012 and 2016, about 26,000 records) every record with bit 0 set has a
matching checksum. Records with bit 0 clear (7200, 7300, 7777, and HYPACK's
7008) store either zero or a value that does not match.

Older frames
------------
DFD 3.10 is protocol version 5 (DFD 0.54 and later). Earlier DFDs, from
the MB-System format archive, define the older frames:

* Protocol 2 (DFD 0.32-0.47, 2002-2003): 72 bytes, ``DRF_V2`` below.
  Bytes 0-39 and SystemEnumerator (byte 42) match protocol 5. Byte 40 is
  SubsystemIdentifier, then come DataSet, RecordCount, the Previous and
  Next record pointers, and Flags at byte 68. Read with the protocol-5
  layout, RecordCount lands where Flags is. Its checksum is the sum of the
  data section only, when flagged.
* Protocol 3 (DFD 0.48-0.50) and 4 (DFD 0.51-0.53): not seen in any file
  yet. Protocol 4 is already the 64-byte layout. In it, and in DFD
  0.54-1.00, protocol 5's Reserved2 (byte 44) is "Record count".
"""

import struct

import numpy as np

from .fields import Section

__all__ = [
    "DRF", "DRF_SIZE", "DRF_V2", "LEGACY_FRAMES", "SYNC_PATTERN", "SYNC_BYTES", "CHECKSUM_SIZE",
    "FLAG_CHECKSUM_VALID", "FLAG_RECORDED_DATA",
    "checksum", "time7k_to_epoch", "epoch_to_time7k",
    "S7KError", "S7KFormatError", "S7KChecksumError",
]

DRF_SIZE = 64
CHECKSUM_SIZE = 4
SYNC_PATTERN = 0x0000FFFF
SYNC_BYTES = struct.pack("<I", SYNC_PATTERN)
FLAG_CHECKSUM_VALID = 0x0001
FLAG_RECORDED_DATA = 0x8000

# Table 5: Data Record Frame. The 7KTIME structure (Table 3) is spelled out
# as its five fields so that it decodes without any conversion.
DRF = Section([
    ("ProtocolVersion", "u16"),
    ("Offset", "u16"),
    ("SyncPattern", "u32"),
    ("Size", "u32"),
    ("OptionalDataOffset", "u32"),
    ("OptionalDataIdentifier", "u32"),
    ("Year", "u16"),
    ("Day", "u16"),
    ("Seconds", "f32"),
    ("Hours", "u8"),
    ("Minutes", "u8"),
    ("RecordVersion", "u16"),
    ("RecordTypeIdentifier", "u32"),
    ("DeviceIdentifier", "u32"),
    ("Reserved1", "u16"),
    ("SystemEnumerator", "u16"),
    ("Reserved2", "u32"),
    ("Flags", "u16"),
    ("Reserved3", "u16"),
    ("Reserved4", "u32"),
    ("TotalRecordsInFragmentedSet", "u32"),
    ("FragmentNumber", "u32"),
])
assert DRF.size == DRF_SIZE

# Protocol version 2 Data Record Frame (DFD 0.42, section 5). Verified on
# 2003 SeaBat 8111-era snapshot files: Offset 68, DataSet counting pairs of
# records, RecordCount running from 0, both record pointers -1, Flags 0.
DRF_V2 = Section([
    ("ProtocolVersion", "u16"),
    ("Offset", "u16"),
    ("SyncPattern", "u32"),
    ("Size", "u32"),
    ("OptionalDataOffset", "u32"),
    ("OptionalDataIdentifier", "u32"),
    ("Year", "u16"),
    ("Day", "u16"),
    ("Seconds", "f32"),
    ("Hours", "u8"),
    ("Minutes", "u8"),
    ("Reserved1", "u16"),
    ("RecordTypeIdentifier", "u32"),
    ("DeviceIdentifier", "u32"),
    ("SubsystemIdentifier", "u16"),
    ("SystemEnumerator", "u16"),
    ("DataSet", "u32"),
    ("RecordCount", "u32"),
    ("PreviousRecord", "i64"),
    ("NextRecord", "i64"),
    ("Flags", "u16"),
    ("Reserved2", "u16"),
])
assert DRF_V2.size == 72

# Frames for protocol versions other than 5, by version.
LEGACY_FRAMES = {2: DRF_V2}

# Byte offsets of DRF fields, for vectorized gathers over a whole file.
DRF_OFFSETS = {}
_pos = 0
for _name, _code in DRF.fields:
    DRF_OFFSETS[_name] = _pos
    _pos += struct.calcsize("<" + {"u8": "B", "u16": "H", "u32": "I", "f32": "f"}[_code])
del _pos, _name, _code

# A structured dtype matching the DRF, for np.frombuffer gathers.
DRF_DTYPE = np.dtype([
    ("ProtocolVersion", "<u2"), ("Offset", "<u2"), ("SyncPattern", "<u4"), ("Size", "<u4"),
    ("OptionalDataOffset", "<u4"), ("OptionalDataIdentifier", "<u4"),
    ("Year", "<u2"), ("Day", "<u2"), ("Seconds", "<f4"), ("Hours", "u1"), ("Minutes", "u1"),
    ("RecordVersion", "<u2"), ("RecordTypeIdentifier", "<u4"), ("DeviceIdentifier", "<u4"),
    ("Reserved1", "<u2"), ("SystemEnumerator", "<u2"), ("Reserved2", "<u4"),
    ("Flags", "<u2"), ("Reserved3", "<u2"), ("Reserved4", "<u4"),
    ("TotalRecordsInFragmentedSet", "<u4"), ("FragmentNumber", "<u4"),
])
assert DRF_DTYPE.itemsize == DRF_SIZE


class S7KError(Exception):
    """Base class for pys7k errors."""


class S7KFormatError(S7KError):
    """A record frame is malformed (bad sync pattern, size, or offsets)."""


class S7KChecksumError(S7KError):
    """A record's stored checksum does not match its contents."""


def checksum(data):
    """
    Compute the 7k checksum of ``data``: the sum of every byte, as an
    unsigned 32-bit integer.

    :param data: a bytes-like object holding the record from the protocol
        version field to the end of the optional data (everything except the
        trailing four checksum bytes).
    """
    return int(np.frombuffer(data, dtype=np.uint8).sum(dtype=np.uint64)) & 0xFFFFFFFF


def time7k_to_epoch(year, day, hours, minutes, seconds):
    """
    Convert 7KTIME fields to POSIX seconds (UTC). Works on scalars or numpy
    arrays. A time whose fields are all zero ("no time available", Table 3)
    converts to NaN.

    :return: float seconds since 1970-01-01T00:00:00Z (a float64 array if
        any input is an array).
    """
    year = np.asarray(year, dtype=np.int64)
    day = np.asarray(day, dtype=np.int64)
    valid = year > 0
    safe_year = np.where(valid, year, 1970)
    days_to_year = (safe_year - 1970).astype("datetime64[Y]").astype("datetime64[D]").astype(np.int64)
    t = ((days_to_year + day - 1) * 86400.0
         + np.asarray(hours, dtype=np.float64) * 3600.0
         + np.asarray(minutes, dtype=np.float64) * 60.0
         + np.asarray(seconds, dtype=np.float64))
    t = np.where(valid, t, np.nan)
    return float(t) if t.ndim == 0 else t


def epoch_to_time7k(t):
    """
    Convert POSIX seconds (UTC) to a 7KTIME dictionary (Year, Day, Seconds,
    Hours, Minutes). Seconds is a float32 in the file, so sub-microsecond
    precision is lost on write.
    """
    whole = np.floor(t)
    dt64 = np.datetime64(int(whole), "s")
    year = dt64.astype("datetime64[Y]")
    day = int((dt64.astype("datetime64[D]") - year.astype("datetime64[D]")).astype(np.int64)) + 1
    sod = int((dt64 - dt64.astype("datetime64[D]")).astype(np.int64))
    return {
        "Year": int(year.astype(np.int64)) + 1970,
        "Day": day,
        "Hours": sod // 3600,
        "Minutes": (sod % 3600) // 60,
        "Seconds": float(sod % 60 + (t - whole)),
    }
