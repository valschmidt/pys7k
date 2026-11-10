"""
Generic sensor records (1000-1999), 2004 and 3001.

Field names are the DFD 3.10 names in PascalCase. Units are as stored:
angles in radians, distances in meters.

Verified against real data (see ``PYS7K.verification``): 1003, 1008, 1010,
1012, 1013, 1015 and 1016, from Reson 7k, Teledyne PDS, HYPACK, NORBIT and
Fledermaus writers. Tested with synthetic records only: 1000-1002, 1004
(the 2003 protocol-2 1003 and 1004 records use DFD 0.42 layouts, decoded
by ``records.legacy``), 1005-1007, 1009, 1011, 1014, 1017, 1020, 2004 and 3001.
"""

from ..fields import Section, Table
from .base import Codec, HeaderCodec, HeaderTableCodec

CODECS = []


def _add(codec):
    CODECS.append(codec)
    return codec


# 1000 - Reference Point (Table 11)
_add(HeaderCodec(1000, "Reference Point", [
    ("VehicleXReferencePointToCOG", "f32"),
    ("VehicleYReferencePointToCOG", "f32"),
    ("VehicleZReferencePointToCOG", "f32"),
    ("WaterLevelToCOG", "f32"),
], phase=3))

_OFFSET_FIELDS = [
    ("SensorPositionXOffset", "f32"),
    ("SensorPositionYOffset", "f32"),
    ("SensorPositionZOffset", "f32"),
    ("SensorRollAngleOffset", "f32"),
    ("SensorPitchAngleOffset", "f32"),
    ("SensorYawAngleOffset", "f32"),
]
# 1001 - Sensor Offset Position (Table 12)
_add(HeaderCodec(1001, "Sensor Offset Position", _OFFSET_FIELDS, phase=3))
# 1002 - Sensor Offset Position Calibrated (Table 13)
_add(HeaderCodec(1002, "Sensor Offset Position Calibrated", _OFFSET_FIELDS, phase=3))

# 1003 - Position (Table 14). NumberOfSatellites is optional: 7125-era
# files write a 36-byte record without it. Verified against real data:
# 36 bytes (Reson 7k), 37 bytes (PDS, NORBIT), UTM grid positions
# (HYPACK) and 35 bytes ending at QualityFlag (Fledermaus, RecordVersion 0).
_add(HeaderCodec(1003, "Position", [
    ("DatumIdentifier", "u32"),
    ("Latency", "f32"),
    ("LatitudeOrNorthing", "f64"),
    ("LongitudeOrEasting", "f64"),
    ("HeightRelativeToDatumOrHeight", "f64"),
    ("PositionTypeFlag", "u8"),
    ("UTMZone", "u8"),
    ("QualityFlag", "u8"),
    ("PositioningMethod", "u8"),
    ("NumberOfSatellites", "u8"),
]))


class _CustomAttitude(Codec):
    """
    1004 - Custom Attitude Information (Tables 15-16).

    FieldMask bits 0-7 select Pitch, Roll, Heading, Heave and their rates.
    The DFD says only that the record data holds "N repeated fields". We
    read it like 1007: N samples, each holding the selected fields in bit
    order. This is an assumption (no sample file has a 1004 record).
    """

    record_type = 1004
    name = "Custom Attitude Information"
    phase = 3
    rth = Section([("FieldMask", "u8"), ("Reserved", "u8"), ("N", "u16"), ("Frequency", "f32")])
    _FIELD_NAMES = ["Pitch", "Roll", "Heading", "Heave", "PitchRate", "RollRate", "HeadingRate", "HeaveRate"]

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Samples": {}}

    def _table(self, mask):
        names = [n for bit, n in enumerate(self._FIELD_NAMES) if mask & (1 << bit)]
        return Table([(n, "f32") for n in names]) if names else None

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        table = self._table(rth["FieldMask"])
        if table is None:
            return rth, {"Samples": {}}, pos
        n = min(rth["N"], (len(body) - pos) // table.itemsize)
        cols, used = table.decode(body, pos, n)
        return rth, {"Samples": cols}, pos + used

    def encode(self, record):
        table = self._table(record["RTH"]["FieldMask"])
        cols = record["RD"]["Samples"]
        return self.rth.pack(record["RTH"]) + (table.encode(cols) if table and cols else b"")


_add(_CustomAttitude())

# 1005 - Tide (Table 17)
_add(HeaderCodec(1005, "Tide", [
    ("Tide", "f32"),
    ("Source", "u16"),
    ("Flags", "u8"),
    ("GaugeIdentifier", "u16"),
    ("DatumIdentifier", "u32"),
    ("Latency", "f32"),
    ("LatitudeOrNorthing", "f64"),
    ("LongitudeOrEasting", "f64"),
    ("HeightRelativeToDatumOrHeight", "f64"),
    ("PositionTypeFlag", "u8"),
    ("UTMZone", "u8"),
], phase=3))

# 1006 - Altitude (Table 18)
_add(HeaderCodec(1006, "Altitude", [("Distance", "f32")], phase=3))


class _MotionOverGround(Codec):
    """1007 - Motion Over Ground (Tables 19-20). Flags bit 0: XYZ speed,
    bit 1: XYZ acceleration. Each reading holds 3 or 6 f32 values."""

    record_type = 1007
    name = "Motion Over Ground"
    phase = 3
    rth = Section([("Flags", "u8"), ("Reserved", "u8"), ("N", "u16"), ("Frequency", "f32")])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Readings": {}}

    @staticmethod
    def _names(flags):
        names = []
        if flags & 1:
            names += ["SpeedX", "SpeedY", "SpeedZ"]
        if flags & 2:
            names += ["AccelerationX", "AccelerationY", "AccelerationZ"]
        return names

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        names = self._names(rth["Flags"])
        n = rth["N"]
        if not names or pos + 4 * len(names) * n > len(body):
            return rth, {"Readings": {}}, pos
        table = Table([(name, "f32") for name in names])
        cols, used = table.decode(body, pos, n)
        return rth, {"Readings": cols}, pos + used

    def encode(self, record):
        names = self._names(record["RTH"]["Flags"])
        cols = record["RD"]["Readings"]
        tail = Table([(name, "f32") for name in names]).encode(cols) if names and cols else b""
        return self.rth.pack(record["RTH"]) + tail


_add(_MotionOverGround())

# 1008 - Depth (Table 21)
_add(HeaderCodec(1008, "Depth", [
    ("DepthDescriptor", "u8"),
    ("CorrectionFlag", "u8"),
    ("Reserved", "u16"),
    ("Depth", "f32"),
], phase=3))

# 1009 - Sound Velocity Profile (Tables 22-23)
_add(HeaderTableCodec(1009, "Sound Velocity Profile", [
    ("PositionFlag", "u8"),
    ("Reserved1", "u8"),
    ("Reserved2", "u16"),
    ("Latitude", "f64"),
    ("Longitude", "f64"),
    ("N", "u32"),
], "N", [("Depth", "f32"), ("SoundVelocity", "f32")], "Samples"))

# 1010 - CTD (Tables 24-25)
_add(HeaderTableCodec(1010, "CTD", [
    ("Frequency", "f32"),
    ("SoundVelocitySourceFlag", "u8"),
    ("SoundVelocityAlgorithm", "u8"),
    ("ConductivityFlag", "u8"),
    ("PressureFlag", "u8"),
    ("PositionFlag", "u8"),
    ("SampleContentValidity", "u8"),
    ("Reserved", "u16"),
    ("Latitude", "f64"),
    ("Longitude", "f64"),
    ("SampleRate", "f32"),
    ("N", "u32"),
], "N", [
    ("ConductivityOrSalinity", "f32"),
    ("WaterTemperature", "f32"),
    ("PressureOrDepth", "f32"),
    ("SoundVelocity", "f32"),
    ("Absorption", "f32"),
], "Samples", phase=3))

# 1011 - Geodesy (Table 26)
_add(HeaderCodec(1011, "Geodesy", [
    ("SpheroidName", "c8*32"),
    ("SemiMajorAxis", "f64"),
    ("InverseFlattening", "f64"),
    ("Reserved1", "u8*16"),
    ("DatumName", "c8*32"),
    ("DataCalculationMethod", "u32"),
    ("NumberOfParameters", "u8"),
    ("DX", "f64"),
    ("DY", "f64"),
    ("DZ", "f64"),
    ("RX", "f64"),
    ("RY", "f64"),
    ("RZ", "f64"),
    ("Scale", "f64"),
    ("Reserved2", "u8*35"),
    ("GridName", "c8*32"),
    ("GridDistanceUnits", "u8"),
    ("GridAngularUnits", "u8"),
    ("LatitudeOfOrigin", "f64"),
    ("CentralMeridian", "f64"),
    ("FalseEasting", "f64"),
    ("FalseNorthing", "f64"),
    ("CentralScaleFactor", "f64"),
    ("CustomIdentifier", "i32"),
    ("Reserved3", "u8*50"),
], phase=3))

# 1012 - Roll Pitch Heave (Table 27)
_add(HeaderCodec(1012, "Roll Pitch Heave", [
    ("Roll", "f32"),
    ("Pitch", "f32"),
    ("Heave", "f32"),
]))

# 1013 - Heading (Table 28)
_add(HeaderCodec(1013, "Heading", [("Heading", "f32")]))

# 1014 - Survey Line (Tables 29-30)
_add(HeaderTableCodec(1014, "Survey Line", [
    ("N", "u16"),
    ("PositionType", "u16"),
    ("Radius", "f32"),
    ("LineName", "c8*64"),
], "N", [("LatitudeOrNorthing", "f64"), ("LongitudeOrEasting", "f64")], "Waypoints", phase=3))

# 1015 - Navigation (Table 31)
_add(HeaderCodec(1015, "Navigation", [
    ("VerticalReference", "u8"),
    ("Latitude", "f64"),
    ("Longitude", "f64"),
    ("HorizontalPositionAccuracy", "f32"),
    ("VesselHeight", "f32"),
    ("HeightAccuracy", "f32"),
    ("SpeedOverGround", "f32"),
    ("CourseOverGround", "f32"),
    ("Heading", "f32"),
]))

# 1016 - Attitude (Tables 32-33). TimeDifference is in milliseconds from
# the record time stamp.
_add(HeaderTableCodec(1016, "Attitude", [("N", "u8")], "N", [
    ("TimeDifference", "u16"),
    ("Roll", "f32"),
    ("Pitch", "f32"),
    ("Heave", "f32"),
    ("Heading", "f32"),
], "Samples"))

# 1017 - Pan Tilt (Table 34)
_add(HeaderCodec(1017, "Pan Tilt", [("Pan", "f32"), ("Tilt", "f32")], phase=3))

# 1020 - Sonar Installation Identifiers (Table 35)
_add(HeaderCodec(1020, "Sonar Installation Identifiers", [
    ("SystemIdentificationNumber", "u32"),
    ("TransmitterIdentification", "u32"),
    ("ReceiverIdentification", "u32"),
    ("StandardConfiguration", "u32"),
    ("ConfigurationFixed", "u32"),
    ("TxLength", "f32"),
    ("TxWidth", "f32"),
    ("TxHeight", "f32"),
    ("TxRadius", "f32"),
    ("SRPToTxXOffset", "f32"),
    ("SRPToTxYOffset", "f32"),
    ("SRPToTxZOffset", "f32"),
    ("TxRollOffset", "f32"),
    ("TxPitchOffset", "f32"),
    ("TxYawOffset", "f32"),
    ("RxLength", "f32"),
    ("RxWidth", "f32"),
    ("RxHeight", "f32"),
    ("RxRadius", "f32"),
    ("SRPToRxXOffset", "f32"),
    ("SRPToRxYOffset", "f32"),
    ("SRPToRxZOffset", "f32"),
    ("RxRollOffset", "f32"),
    ("RxPitchOffset", "f32"),
    ("RxYawOffset", "f32"),
    ("Frequency", "f32"),
    ("VRPToSRPXOffset", "f32"),
    ("VRPToSRPYOffset", "f32"),
    ("VRPToSRPZOffset", "f32"),
    ("CableLength", "u32"),
    ("Reserved", "u8*44"),
]))

# 2004 - Sonar Pipe Environment (Tables 36-37). The DFD labels the point
# table "7001 - Record Data"; it is the 2004 point sub record.
_add(HeaderTableCodec(2004, "Sonar Pipe Environment", [
    ("PipeNumber", "u32"),
    ("PingNumber", "u32"),
    ("MultiPingSequence", "u32"),
    ("PipeDiameter", "f32"),
    ("SoundVelocity", "f32"),
    ("SampleRate", "f32"),
    ("Finished", "u8"),
    ("NumberOfPoints", "u32"),
    ("PointSubRecordSize", "u32"),
    ("Reserved", "u32*10"),
], "NumberOfPoints", [
    ("X", "f32"),
    ("Y", "f32"),
    ("Z", "f32"),
    ("Angle", "f32"),
    ("SampleNumber", "f32"),
], "Points", itemsize_field="PointSubRecordSize", phase=3))

# 3001 - Contact Output (Table 38)
_add(HeaderCodec(3001, "Contact Output", [
    ("TargetId", "u32"),
    ("PingNumber", "u32"),
    ("OperationTime", "u8*10"),  # a 7KTIME, kept as its 10 raw bytes
    ("OperatorName", "c8*128"),
    ("ContactState", "u32"),
    ("Range", "f32"),
    ("Bearing", "f32"),
    ("InformationFlags", "u32"),
    ("Latitude", "f64"),
    ("Longitude", "f64"),
    ("Azimuth", "f32"),
    ("ContactLength", "f32"),
    ("ContactWidth", "f32"),
    ("Classification", "c8*128"),
    ("Description", "c8*128"),
], phase=3))
