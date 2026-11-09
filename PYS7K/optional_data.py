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

__all__ = ["OptionalDataCodec", "OPTIONAL_DATA_CODECS"]


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
    7200: OptionalDataCodec([("Size", "u32"), ("Offset", "u64")]),     # Table 105
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
