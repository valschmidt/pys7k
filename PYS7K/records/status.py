"""
Status, event, logging and remote-control records added in phase 3:
7021, 7031, 7050-7055, 7059, 7500-7502, 7504, 7510, 7511.

None of the local sample files contain these records. Each layout is
transcribed from DFD 3.10 and tested by synthetic round trips. A 7500
remote control's record data depends on its command (DFD section 11) and
is kept as raw bytes in ``RD["Data"]``.
"""

import numpy as np

from ..fields import Section, Table
from .base import Codec, HeaderCodec

CODECS = []


def _add(codec):
    CODECS.append(codec)
    return codec


class _ListCodec(Codec):
    """
    A header followed by a list of variable-length entries. Each entry is a
    fixed ``entry`` Section, optionally followed by a byte block whose
    length is one of the entry's fields (``tail_field``), stored under
    ``tail_key``. RD is ``{key: [entry dict, ...]}``.
    """

    def __init__(self, record_type, name, rth_fields, count_field, entry_fields, key,
                 tail_field=None, tail_key=None, phase=3):
        self.record_type = record_type
        self.name = name
        self.rth = Section(rth_fields)
        self.entry = Section(entry_fields)
        self.count_field = count_field
        self.key = key
        self.tail_field = tail_field
        self.tail_key = tail_key
        self.phase = phase

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {self.key: []}

    def new_entry(self):
        e = self.entry.new()
        if self.tail_key:
            e[self.tail_key] = b""
        return e

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        entries = []
        for _ in range(max(int(rth[self.count_field]), 0)):
            if pos + self.entry.size > len(body):
                break
            e = self.entry.unpack(body, pos)
            end = pos + self.entry.size
            if self.tail_field:
                n = e[self.tail_field]
                if end + n > len(body):
                    break
                e[self.tail_key] = bytes(body[end:end + n])
                end += n
            entries.append(e)
            pos = end
        return rth, {self.key: entries}, pos

    def encode(self, record):
        out = [self.rth.pack(record["RTH"])]
        for e in record["RD"][self.key]:
            out.append(self.entry.pack(e))
            if self.tail_key:
                out.append(bytes(e[self.tail_key]))
        return b"".join(out)


class _BITE(Codec):
    """
    7021 - Built-In Test Environment Data (Tables 67-68). One entry per
    board: the board header, then NBF bite fields of 79 bytes each
    (Field, NameAndValueText, SensorType, Minimum, Maximum, Value). The DFD
    is not explicit that only NBF fields follow; that is assumed here.
    """

    record_type = 7021
    name = "Built-In Test Environment Data"
    phase = 3
    rth = Section([("N", "u16")])
    board = Section([
        ("SourceName", "c8*64"),
        ("SourceAddress", "u8"),
        ("Reserved1", "f32"),
        ("Reserved2", "u16"),
        ("DownlinkTimeSent", "u8*10"),
        ("UplinkTimeReceived", "u8*10"),
        ("BITETimeReceived", "u8*10"),
        ("Status", "u8"),
        ("NBF", "u16"),
        ("BITEStatusBits", "u64*4"),
    ])
    fields = Table([
        ("Field", "u16"),
        ("NameAndValueText", "c8*64"),
        ("SensorType", "u8"),
        ("Minimum", "f32"),
        ("Maximum", "f32"),
        ("Value", "f32"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Boards": []}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        boards = []
        for _ in range(rth["N"]):
            if pos + self.board.size > len(body):
                break
            b = self.board.unpack(body, pos)
            pos += self.board.size
            n = min(b["NBF"], (len(body) - pos) // self.fields.itemsize)
            b["BiteFields"], used = self.fields.decode(body, pos, n)
            pos += used
            boards.append(b)
        return rth, {"Boards": boards}, pos

    def encode(self, record):
        out = [self.rth.pack(record["RTH"])]
        for b in record["RD"]["Boards"]:
            out.append(self.board.pack(b))
            out.append(self.fields.encode(b["BiteFields"]))
        return b"".join(out)


_add(_BITE())

# 7031 - BITE Summary (Table 79)
_add(HeaderCodec(7031, "Built-In Test Environment Data (Summary)", [
    ("TotalItems", "u16"),
    ("Warnings", "u16*4"),
    ("Errors", "u16*4"),
    ("Fatals", "u16*4"),
    ("Reserved", "u32*2"),
], phase=3))

# 7050 - System Events (Tables 88-89)
_add(_ListCodec(7050, "System Events", [("SonarId", "u64"), ("N", "u32")], "N", [
    ("EventType", "u16"),
    ("EventIdentifier", "u16"),
    ("DeviceIdentifier", "u32"),
    ("SystemEnumerator", "u16"),
    ("EventMessageLength", "u16"),
    ("Time", "u8*10"),
], "Events", tail_field="EventMessageLength", tail_key="EventMessage"))


class _EventMessage(Codec):
    """7051 - System Event Message (Tables 90-91). The message text is kept
    as bytes (MessageLength of them)."""

    record_type = 7051
    name = "System Event Message"
    phase = 3
    rth = Section([("SonarId", "u64"), ("EventId", "u16"), ("MessageLength", "u16"),
                   ("EventIdentifier", "u16")])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"EventMessage": b""}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        end = min(len(body), self.rth.size + rth["MessageLength"])
        return rth, {"EventMessage": bytes(body[self.rth.size:end])}, end

    def encode(self, record):
        return self.rth.pack(record["RTH"]) + bytes(record["RD"]["EventMessage"])


_add(_EventMessage())


class _RecordingStatus(Codec):
    """7052 - RDR Recording Status (Tables 92-93). RD holds five
    count-prefixed u32 arrays."""

    record_type = 7052
    name = "RDR Recording Status"
    phase = 3
    rth = Section([
        ("Position", "u32"),
        ("DiskFree", "u8"),
        ("Mode", "u8"),
        ("FileRecords", "u32"),
        ("FileSize", "u64"),
        ("FirstTime", "u8*10"),
        ("LastTime", "u8*10"),
        ("TotalTime", "u32"),
        ("DirectoryName", "c8*256"),
        ("FileName", "c8*256"),
        ("Error", "u32"),
        ("Flags", "u32"),
        ("LoggerAddress", "u32"),
        ("SingleOrMultipleFile", "u8"),
        ("PingDataHistory", "u8"),
        ("Reserved1", "u16"),
        ("Reserved2", "u32*4"),
    ])
    _ARRAYS = ["ThresholdValues", "IncludedRecords", "ExcludedRecords", "IncludedDevices", "ExcludedDevices"]

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {k: np.zeros(0, "<u4") for k in self._ARRAYS}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        rd = {}
        for name in self._ARRAYS:
            if pos + 4 > len(body):
                break
            n = int(np.frombuffer(body, "<u4", 1, pos)[0])
            if pos + 4 + 4 * n > len(body):
                break
            rd[name] = np.frombuffer(body, "<u4", n, pos + 4)
            pos += 4 + 4 * n
        return rth, rd, pos

    def encode(self, record):
        out = [self.rth.pack(record["RTH"])]
        for name in self._ARRAYS:
            if name not in record["RD"]:
                break
            arr = np.asarray(record["RD"][name], "<u4")
            out.append(np.uint32(len(arr)).tobytes() + arr.tobytes())
        return b"".join(out)


_add(_RecordingStatus())

# 7053 - Subscriptions (Tables 94-95)
_add(_ListCodec(7053, "Subscriptions", [("N", "i32")], "N", [
    ("Address", "u32"),
    ("Port", "u16"),
    ("Type", "u16"),
    ("NumberOfRecords", "u32"),
    ("RecordList", "u32*64"),
    ("Reserved", "u32*128"),
], "Subscriptions"))

# 7054 - RDR Storage Recording Short Update (Table 96)
_add(HeaderCodec(7054, "RDR Storage Recording Short Update", [
    ("DiskFreePercentage", "u16"),
    ("NumberOfRecords", "u32"),
    ("Size", "u64"),
    ("Reserved", "u32*4"),
    ("Mode", "u8"),
    ("FileName", "c8*256"),
    ("RDRError", "u32"),
    ("DataRate", "u64"),
    ("MinutesSpaceLeft", "u32"),
], phase=3))

# 7055 - Calibration Status (Table 97)
_add(HeaderCodec(7055, "Calibration Status", [
    ("SonarId", "u64"),
    ("CalibrationStatus", "u16"),
    ("PercentComplete", "u16"),
    ("CalibrationTime", "u8*10"),
    ("StatusMessage", "c8*800"),
    ("SubStatus", "u32"),
    ("CalibrationSystem", "u8"),
    ("CalibrationSystemDone", "u8"),
    ("CurrentCalibration", "u8"),
    ("StartupCalibration", "u8"),
    ("Status", "u16*8"),
    ("Reserved", "u32*2"),
], phase=3))

_PORT = []
for _p in ("ZDA", "GGA", "SVP", "HPR", "HDT"):
    _PORT += [(_p + "PortBaudRate", "u8"), (_p + "PortParity", "u8"),
              (_p + "PortDataBits", "u8"), (_p + "PortStopBits", "u8")]

# 7059 - MB2 Specific Status (Table 102). The DFD names the third RRIO
# address "RRIO address2" twice; it is RRIOAddress3 here.
_add(HeaderCodec(7059, "MB2 Specific Status", [
    ("Directory", "c8*256"),
    ("HeaderName", "c8*256"),
    ("TrailerName", "c8*256"),
    ("PrependHeader", "u8"),
    ("AppendTrailer", "u8"),
    ("Storage", "u8"),
    ("PlaybackPathName", "c8*256"),
    ("PlaybackFileName", "c8*256"),
    ("PlaybackLoopMode", "u32"),
    ("Playback", "u8"),
    ("RRIOAddress1", "c8*256"),
    ("RRIOAddress2", "c8*256"),
    ("RRIOAddress3", "c8*256"),
    ("UseBuiltInHPR", "u8"),
    ("UseAttachedSVP", "u8"),
    ("EnableStacking", "u8"),
    ("StackingValue", "u8"),
] + _PORT + [
    ("RRIO", "u16"),
    ("PlaybackTimestamps", "u8"),
    ("Reserved1", "u8"),
    ("Reserved2", "u32"),
], phase=3))


class _RemoteControl(Codec):
    """7500 - Remote Control (Tables 109-110). The record data depends on
    RemoteControlId (DFD section 11) and is kept as bytes in RD["Data"]."""

    record_type = 7500
    name = "Remote Control"
    phase = 3
    rth = Section([("RemoteControlId", "u32"), ("Ticket", "u32"), ("TrackingNumber", "u64*2")])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Data": b""}

    def decode(self, body):
        rth, used = self.rth.unpack_prefix(body, 0, len(body))
        return rth, {"Data": bytes(body[used:])}, len(body)

    def encode(self, record):
        return self.rth.pack(record["RTH"]) + bytes(record["RD"].get("Data", b""))


_add(_RemoteControl())

# 7501 - Remote Control Acknowledge (Table 111)
_add(HeaderCodec(7501, "Remote Control Acknowledge", [("Ticket", "u32"), ("TrackingNumber", "u64*2")], phase=3))
# 7502 - Remote Control Not Acknowledge (Table 112)
_add(HeaderCodec(7502, "Remote Control Not Acknowledge", [
    ("Ticket", "u32"), ("TrackingNumber", "u64*2"), ("ErrorCode", "u32")], phase=3))

# 7504 - Common System Settings (Table 114)
_add(HeaderCodec(7504, "Common System Settings", [
    ("SonarId", "u64"),
    ("PingNumber", "u32"),
    ("SoundVelocity", "f32"),
    ("Absorption", "f32"),
    ("SpreadingLoss", "f32"),
    ("SequencerControl", "u32"),
    ("MotionSensorFormat", "u8"),
    ("MotionSensorBaudRate", "u8"),
    ("MotionSensorParity", "u8"),
    ("MotionSensorDataBits", "u8"),
    ("MotionSensorStopBits", "u8"),
    ("Orientation", "u8"),
    ("RecordVersion", "u8"),
    ("MotionLatency", "f32"),
    ("SVPFilter", "u8"),
    ("SVManualOverride", "u8"),
    ("ActiveEnumerator", "u16"),
    ("ActiveDeviceId", "u32"),
    ("SystemMode", "u32"),
    ("MasterSlaveMode", "u32"),
    ("TrackerFlags", "u32"),
    ("TrackerSwathWidth", "f32"),
    ("MultiDetectEnable", "u16"),
    ("MultiDetectObjectSize", "u16"),
    ("MultiDetectSensitivity", "u16"),
    ("MultiDetectMaxDetections", "u16"),
    ("MultiDetectReserved", "u16*2"),
    ("SlaveIPAddress", "u8*4"),
    ("SnippetControlFlags", "u32"),
    ("SnippetControlMinWindow", "u32"),
    ("SnippetControlMaxWindow", "u32"),
    ("FullRateDualHeadFlag", "u32"),
    ("DelayMultiplier", "f32"),
    ("PowerSavingMode", "u8"),
    ("Flags", "u8"),
    ("RangeBlankControl", "u16"),
    ("StartupNormalization", "u8"),
    ("RestorePingRate", "u8"),
    ("RestorePower", "u8"),
    ("SVInterlock", "u8"),
    ("IgnorePPSErrors", "u8"),
    ("Reserved1", "u8*15"),
    ("CompressedWaterColumnFlags", "u32"),
    ("DeckMode", "u8"),
    ("Reserved2", "u8"),
    ("PowerModeFlags", "u8"),
    ("PowerModeMax", "u8"),
    ("WaterTemperature", "f32"),
    ("SensorManualOverride", "u8"),
    ("SensorDataFlags", "u8"),
    ("SensorActive", "u8"),
    ("Reserved3", "u8"),
    ("TrackerMaxCoverage", "f32"),
    ("DutyCycleMethod", "u16"),
    ("Reserved4", "u16"),
    ("Reserved5", "u32*99"),
], phase=3))

# 7510 - SV Filtering (Table 115)
_add(HeaderCodec(7510, "SV Filtering", [("SensorSV", "f32"), ("FilteredSV", "f32"), ("Filter", "u8")], phase=3))
# 7511 - System Lock Status (Table 116)
_add(HeaderCodec(7511, "System Lock Status", [("SystemLock", "u16"), ("ClientIP", "u32"),
                                              ("Reserved", "u32*8")], phase=3))
