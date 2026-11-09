"""
File, time, and environment records: 7200, 7300, 7400, 7610-7614.
"""

from ..fields import Section, Table
from .base import Codec, HeaderCodec, HeaderTableCodec

CODECS = []


def _add(codec):
    CODECS.append(codec)
    return codec


# 7200 - File Header (Tables 103-104). The optional data (Table 105) points
# at the 7300 catalog. 7k logger files leave that pointer zero.
_add(HeaderTableCodec(7200, "File Header", [
    ("FileIdentifier", "u8*16"),
    ("VersionNumber", "u16"),
    ("Reserved", "u16"),
    ("SessionIdentifier", "u8*16"),
    ("RecordDataSize", "u32"),
    ("N", "u32"),
    ("RecordingName", "c8*64"),
    ("RecordingProgramVersionNumber", "c8*16"),
    ("UserDefinedName", "c8*64"),
    ("Notes", "c8*128"),
], "N", [("DeviceIdentifier", "u32"), ("SystemEnumerator", "u16")], "Devices"))


class _FileCatalog(Codec):
    """
    7300 - File Catalog Record (Tables 106-107). One entry per record in the
    file, excluding the 7300 itself. The RTH's Size field gives the header
    length (14 bytes in every sample file). Each entry is 48 bytes.
    """

    record_type = 7300
    name = "File Catalog"
    rth = Section([("Size", "u32"), ("Version", "u16"), ("NumberOfRecords", "u32"), ("Reserved", "u32")])
    entries = Table([
        ("Size", "u32"),
        ("Offset", "u64"),
        ("RecordType", "u16"),
        ("DeviceIdentifier", "u16"),
        ("SystemEnumerator", "u16"),
        ("Year", "u16"),
        ("Day", "u16"),
        ("Seconds", "f32"),
        ("Hours", "u8"),
        ("Minutes", "u8"),
        ("RecordCount", "u32"),
        ("Reserved1", "u16"),
        ("Reserved2", "u16"),
        ("Reserved3", "u16"),
        ("Reserved4", "u16"),
        ("Reserved5", "u16"),
        ("Reserved6", "u16"),
        ("Reserved7", "u16"),
        ("Reserved8", "u16"),
    ])

    def new_rth(self):
        rth = self.rth.new()
        rth["Size"] = self.rth.size
        rth["Version"] = 1
        return rth

    def new_rd(self):
        return {"Entries": self.entries.new()}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        count = min(rth["NumberOfRecords"], (len(body) - pos) // self.entries.itemsize)
        cols, used = self.entries.decode(body, pos, count)
        return rth, {"Entries": cols}, pos + used

    def encode(self, record):
        return self.rth.pack(record["RTH"]) + self.entries.encode(record["RD"]["Entries"])


_add(_FileCatalog())

# 7400 - Time Message (Table 108)
_add(HeaderCodec(7400, "Time Message", [
    ("LeapSecondOffset", "i8"),
    ("PulseFlag", "u8"),
    ("PortIdentifier", "u16"),
    ("Reserved1", "u32"),
    ("Reserved2", "u64"),
], phase=3))

# 7610 - Sound Velocity (Table 117). Temperature and Pressure are optional.
_add(HeaderCodec(7610, "Sound Velocity", [
    ("SoundVelocity", "f32"),
    ("Temperature", "f32"),
    ("Pressure", "f32"),
]))
# 7611 - Absorption Loss (Table 118)
_add(HeaderCodec(7611, "Absorption Loss", [("AbsorptionLoss", "f32")], phase=3))
# 7612 - Spreading Loss (Table 119)
_add(HeaderCodec(7612, "Spreading Loss", [("SpreadingLoss", "f32")], phase=3))
# 7613 - Profile Average Salinity (Table 120)
_add(HeaderCodec(7613, "Profile Average Salinity", [("AverageSalinity", "f32")], phase=3))
# 7614 - Profile Average Temperature (Table 121)
_add(HeaderCodec(7614, "Profile Average Temperature", [("Temperature", "f32")], phase=3))
