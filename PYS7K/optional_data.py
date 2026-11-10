"""
Optional data codecs (DFD 3.10 Appendix A, Teledyne PDS optional data, plus
the 7200 file-catalog pointer of Table 105).

Optional data is decoded to a dictionary of header fields plus, where the
layout has one, a ``"Beams"`` dictionary of numpy arrays. The number of
beams is not stored; it is however many whole beam entries fit. Bytes past
the known layout are kept as ``Unparsed["OptionalData"]``.

Appendix F notes that the 7027 optional data is "fully corrected for
motion, sound velocity and Tx-offset". It is reported here exactly as
stored.
"""

from .fields import Section, Table

__all__ = ["OptionalDataCodec", "OPTIONAL_DATA_CODECS", "recover_misplaced_optional_data"]


class OptionalDataCodec:
    """A fixed header, optionally followed by a per-beam table."""

    def __init__(self, header_fields, beam_fields=None):
        self.header = Section(header_fields)
        self.beams = Table(beam_fields) if beam_fields else None

    def new(self):
        d = self.header.new()
        if self.beams is not None:
            d["Beams"] = self.beams.new()
        return d

    def decode(self, buf):
        d, pos = self.header.unpack_prefix(buf, 0, len(buf))
        if self.beams is not None and pos == self.header.size:
            count = (len(buf) - pos) // self.beams.itemsize
            cols, used = self.beams.decode(buf, pos, count)
            d["Beams"] = cols
            pos += used
        return d, pos

    def encode(self, d):
        out = self.header.pack(d)
        if self.beams is not None and d.get("Beams"):
            out += self.beams.encode(d["Beams"])
        return out


_PDS_POSITION = [
    ("Frequency", "f32"),
    ("Latitude", "f64"),
    ("Longitude", "f64"),
    ("Heading", "f32"),
]
_PDS_BATHY_HEADER = _PDS_POSITION + [
    ("HeightSource", "u8"),
    ("Tide", "f32"),
    ("Roll", "f32"),
    ("Pitch", "f32"),
    ("Heave", "f32"),
    ("VehicleDepth", "f32"),
]
_PDS_BATHY_BEAMS = [
    ("Depth", "f32"),
    ("AlongTrackDistance", "f32"),
    ("AcrossTrackDistance", "f32"),
    ("PointingAngle", "f32"),
    ("AzimuthAngle", "f32"),
]
_PDS_WC_BEAMS = [
    ("AlongTrackDistance", "f32"),
    ("AcrossTrackDistance", "f32"),
    ("CenterSampleNumber", "u32"),
]

OPTIONAL_DATA_CODECS = {
    7006: OptionalDataCodec(_PDS_BATHY_HEADER, _PDS_BATHY_BEAMS),      # Table 217
    7007: OptionalDataCodec(_PDS_POSITION + [("Depth", "f32")]),       # Table 218
    7008: OptionalDataCodec(_PDS_POSITION, _PDS_WC_BEAMS),             # Table 219
    7027: OptionalDataCodec(_PDS_BATHY_HEADER, _PDS_BATHY_BEAMS),      # Table 220
    7028: OptionalDataCodec(_PDS_POSITION, _PDS_WC_BEAMS),             # Table 221
    7047: OptionalDataCodec(_PDS_BATHY_HEADER, _PDS_BATHY_BEAMS),      # Table 222
    7057: OptionalDataCodec(_PDS_POSITION + [("Depth", "f32")]),       # Table 223
    7058: OptionalDataCodec(_PDS_POSITION, _PDS_WC_BEAMS),             # Table 224
    # Table 105. Reson's RDR 1.2 recorder (7125 files from 2006-2011) writes
    # only 6 bytes of zeros here: Size decodes, Offset is None, and the
    # remaining 2 bytes are kept in Unparsed["OptionalData"].
    7200: OptionalDataCodec([("Size", "u32"), ("Offset", "u64")]),
    7400: OptionalDataCodec([                                          # Table 225
        ("UTCTime", "f64"),
        ("ExternalTime", "f64"),
        ("T0", "f64"),
        ("T1", "f64"),
        ("PulseLength", "f64"),
        ("Difference", "f64"),
        ("IOModuleStatus", "u16"),
    ]),
}


def recover_misplaced_optional_data(record):
    """
    Decode 7028 optional data that was written before the place the frame's
    OptionalDataOffset points to.

    Teledyne PDS2000 (3.9.1.3, 3.9.2.0 and 3.9.2.2 seen, in 2014 7125 and
    T20-P files) writes 7028 records with u16 snippets (Flags bit 0 clear)
    but sizes the record, and sets OptionalDataOffset, as if the snippets
    were u32. The real optional data follows the u16 snippets, inside what
    the frame says is record data. The region the frame calls optional data
    holds zeros or stale bytes from an earlier record. pys7k decodes such a
    record as stored: ``OptionalData`` is whatever the declared region
    holds, and ``Unparsed["Body"]``, which is exactly as long as the
    snippets, starts with the real optional data. When the snippets are
    shorter than the optional data, the rest of it is at the start of the
    declared region.

    This function returns the optional data decoded from the start of
    ``Unparsed["Body"]`` (continuing into the declared region if needed)
    when the record shows that signature (a 7028 with
    Flags bit 0 clear, optional data present, and unparsed body bytes equal
    in length to the snippets), or None otherwise. The beam count is the
    record header's ``N``. The record itself is not changed.
    """
    if record.get("RecordType") != 7028 or "RTH" not in record:
        return None
    codec = OPTIONAL_DATA_CODECS[7028]
    extra = bytes((record.get("Unparsed") or {}).get("Body", b""))
    samples = (record.get("RD") or {}).get("Samples")
    rth = record["RTH"]
    if record.get("OptionalData") is None or samples is None or rth.get("Flags", 0) & 1:
        return None
    if not extra or len(extra) != len(samples) * 2:
        return None
    size = codec.header.size + codec.beams.itemsize * (rth.get("N") or 0)
    # The real optional data starts the unparsed body; when the snippets are
    # shorter than it, it runs on into the declared optional data region.
    declared = codec.encode(record["OptionalData"]) + bytes(record["Unparsed"].get("OptionalData", b""))
    buf = extra + declared
    if len(buf) < size:
        return None
    d, _ = codec.decode(memoryview(buf)[:size])
    return d
