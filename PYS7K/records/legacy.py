"""
Record layouts from DFDs older than protocol version 5.

DFD 3.10 describes protocol 5. These codecs decode records in files with
an older ``ProtocolVersion``, using the layouts of the DFD that defined
that protocol (from the MB-System format archive). They are keyed by
(protocol version, record type) in ``registry.LEGACY_CODECS`` and are not
in ``RECORD_CODECS``.

Protocol 2 (DFD 0.32-0.47; layouts from DFD 0.42, 2002):

* 1003 Position: Datum, Latitude, Longitude, Height (28 bytes; no latency,
  position type or UTM fields yet).
* 1004 Attitude: FieldMask, Reserved, N, Frequency, then N f32 values.
  FieldMask bits 0-3 say which of Pitch, Roll, Heading (radians) and
  Heave (metres) are present; the values repeat in that order.

Verified against real data: 2003 Kauai snapshot files (devices 11, 21
and 31). 1003 positions fall at 22.165 N, 159.804 W. 1004 records come as
a pitch/roll/heave stream (mask 0b1011, N 3) and a heading stream (mask
0b0100, N 1), with Frequency 0.
"""

import numpy as np

from ..fields import Section
from .base import Codec, HeaderCodec

__all__ = ["CODECS"]

CODECS = {}

CODECS[(2, 1003)] = HeaderCodec(1003, "Position (protocol 2)", [
    ("Datum", "u32"),
    ("Latitude", "f64"),
    ("Longitude", "f64"),
    ("Height", "f64"),
])


class _AttitudeV2(Codec):
    """1004 Attitude, protocol 2 (DFD 0.42)."""

    record_type = 1004
    name = "Attitude (protocol 2)"
    rth = Section([
        ("FieldMask", "u8"),
        ("Reserved", "u8"),
        ("N", "u16"),
        ("Frequency", "f32"),
    ])
    FIELDS = ("Pitch", "Roll", "Heading", "Heave")

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        n = rth["N"]
        names = [f for i, f in enumerate(self.FIELDS) if rth["FieldMask"] >> i & 1]
        if not names or n % len(names) or pos + 4 * n > len(body):
            return rth, {}, pos
        values = np.frombuffer(body, "<f4", n, pos).reshape(-1, len(names))
        rd = {name: values[:, i] for i, name in enumerate(names)}
        return rth, rd, pos + 4 * n

    def encode(self, record):
        rd = record["RD"]
        names = [f for f in self.FIELDS if f in rd]
        out = self.rth.pack(record["RTH"])
        if names:
            out += np.column_stack([np.asarray(rd[f], "<f4") for f in names]).astype("<f4").tobytes()
        return out


CODECS[(2, 1004)] = _AttitudeV2()
