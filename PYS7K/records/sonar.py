"""
SeaBat 7k sonar records (7000-7099) and 7503.

Field names are the DFD 3.10 names in PascalCase. Values are as stored:
angles in radians, times in seconds, detections in fractional samples, and
bit fields as integers (see ``PYS7K.bitfields`` to expand them).
"""

import numpy as np

from ..fields import Section, Table
from .base import (PING_PREFIX, Codec, ColumnsCodec, HeaderCodec,
                   HeaderTableCodec)

CODECS = []


def _add(codec):
    CODECS.append(codec)
    return codec


# Fields shared by 7000 (Table 39) and 7503 (Table 113), from Frequency
# through Spreading. 7503 has no ReceiveBeamWidth (checked against 7125
# data: with it, 7503's SoundVelocity lands on the Spreading value).
_SETTINGS_FIELDS = [
    ("Frequency", "f32"),
    ("SampleRate", "f32"),
    ("ReceiverBandwidth", "f32"),
    ("TxPulseWidth", "f32"),
    ("TxPulseTypeIdentifier", "u32"),
    ("TxPulseEnvelopeIdentifier", "u32"),
    ("TxPulseEnvelopeParameter", "f32"),
    ("TxPulseMode", "u16"),
    ("TxPulseReserved", "u16"),
    ("MaxPingRate", "f32"),
    ("PingPeriod", "f32"),
    ("RangeSelection", "f32"),
    ("PowerSelection", "f32"),
    ("GainSelection", "f32"),
    ("ControlFlags", "u32"),
    ("ProjectorIdentifier", "u32"),
    ("ProjectorBeamSteeringAngleVertical", "f32"),
    ("ProjectorBeamSteeringAngleHorizontal", "f32"),
    ("ProjectorBeamWidthVertical", "f32"),
    ("ProjectorBeamWidthHorizontal", "f32"),
    ("ProjectorBeamFocalPoint", "f32"),
    ("ProjectorBeamWeightingWindowType", "u32"),
    ("ProjectorBeamWeightingWindowParameter", "f32"),
    ("TransmitFlags", "u32"),
    ("HydrophoneIdentifier", "u32"),
    ("ReceiveBeamWeightingWindow", "u32"),
    ("ReceiveBeamWeightingParameter", "f32"),
    ("ReceiveFlags", "u32"),
    ("ReceiveBeamWidth", "f32"),
    ("BottomDetectionFilterMinRange", "f32"),
    ("BottomDetectionFilterMaxRange", "f32"),
    ("BottomDetectionFilterMinDepth", "f32"),
    ("BottomDetectionFilterMaxDepth", "f32"),
    ("Absorption", "f32"),
    ("SoundVelocity", "f32"),
    ("Spreading", "f32"),
]

_SETTINGS_7503 = [f for f in _SETTINGS_FIELDS if f[0] != "ReceiveBeamWidth"]

# 7000 - Sonar Settings (Table 39). 156 bytes.
_add(HeaderCodec(7000, "Sonar Settings", PING_PREFIX + _SETTINGS_FIELDS + [
    ("Reserved", "u16"),
], ping=True))


class _Configuration(Codec):
    """
    7001 - Configuration (Tables 40-41). One entry per device. Each entry's
    Info is a variable-length block (XML on recent systems), kept as bytes.
    """

    record_type = 7001
    name = "Configuration"
    rth = Section([("SonarId", "u64"), ("N", "u32")])
    device = Section([
        ("Identifier", "u32"),
        ("Description", "c8*60"),
        ("AlphaDataCard", "u32"),
        ("SerialNumber", "u64"),
        ("InfoLength", "u32"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Devices": []}

    def new_device(self):
        d = self.device.new()
        d["Info"] = b""
        return d

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        devices = []
        for _ in range(rth["N"]):
            if pos + self.device.size > len(body):
                break
            d = self.device.unpack(body, pos)
            start = pos + self.device.size
            if start + d["InfoLength"] > len(body):
                break
            d["Info"] = bytes(body[start:start + d["InfoLength"]])
            devices.append(d)
            pos = start + d["InfoLength"]
        return rth, {"Devices": devices}, pos

    def encode(self, record):
        out = [self.rth.pack(record["RTH"])]
        for d in record["RD"]["Devices"]:
            out.append(self.device.pack(d))
            out.append(bytes(d["Info"]))
        return b"".join(out)


_add(_Configuration())

# 7002 - Match Filter (Table 42)
_add(HeaderCodec(7002, "Match Filter", [
    ("SonarId", "u64"),
    ("PingNumber", "u32"),
    ("Operation", "u32"),
    ("StartFrequency", "f32"),
    ("StopFrequency", "f32"),
    ("WindowType", "u32"),
    ("ShadingValue", "f32"),
    ("EffectivePulseWidth", "f32"),
    ("Reserved", "u32*13"),
], phase=3))


class _FirmwareHardware(Codec):
    """7003 - Firmware and Hardware Configuration (Table 43). The info
    block (XML) is kept as bytes."""

    record_type = 7003
    name = "Firmware and Hardware Configuration"
    phase = 3
    rth = Section([("Devices", "u32"), ("InfoLength", "u32")])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Info": b""}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        end = min(len(body), self.rth.size + rth["InfoLength"])
        return rth, {"Info": bytes(body[self.rth.size:end])}, end

    def encode(self, record):
        return self.rth.pack(record["RTH"]) + bytes(record["RD"]["Info"])


_add(_FirmwareHardware())

# 7004 - Beam Geometry (Tables 44-45). Older systems (e.g. 7125) omit the
# TxDelay array; only the arrays present are decoded.
_add(ColumnsCodec(7004, "Beam Geometry", [("SonarId", "u64"), ("N", "u32")], "N", [
    ("BeamVerticalDirection", "f32"),
    ("BeamHorizontalDirection", "f32"),
    ("BeamWidthY", "f32"),
    ("BeamWidthX", "f32"),
    ("TxDelay", "f32"),
]))

# 7006 - Bathymetric Data (Tables 46-47). Deprecated, superseded by 7027.
_add(ColumnsCodec(7006, "Bathymetric Data", PING_PREFIX + [
    ("N", "u32"),
    ("Flags", "u8"),
    ("SoundVelocityFlag", "u8"),
    ("SoundVelocity", "f32"),
], "N", [
    ("Range", "f32"),
    ("Quality", "u8"),
    ("Intensity", "f32"),
    ("MinFilterInfo", "f32"),
    ("MaxFilterInfo", "f32"),
], ping=True))

_UINT = {1: "<u1", 2: "<u2", 4: "<u4"}


class _SideScan(Codec):
    """7007 - Side Scan Data (Tables 48-49). Port then starboard, S samples
    of W bytes each."""

    record_type = 7007
    name = "Side Scan Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("BeamPosition", "f32"),
        ("ControlFlags", "u32"),
        ("S", "u32"),
        ("NadirDepth", "u32"),
        ("Reserved", "f32*7"),
        ("N", "u16"),
        ("CurrentBeamNumber", "u16"),
        ("W", "u8"),
        ("DataTypes", "u8"),
    ])

    def new_rth(self):
        rth = self.rth.new()
        rth["W"] = 2
        return rth

    def new_rd(self):
        return {"Port": np.zeros(0, "<u2"), "Starboard": np.zeros(0, "<u2")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        dt = _UINT.get(rth["W"])
        n = rth["S"]
        if dt is None or pos + 2 * n * rth["W"] > len(body):
            return rth, {}, pos
        port = np.frombuffer(body, dtype=dt, count=n, offset=pos)
        stbd = np.frombuffer(body, dtype=dt, count=n, offset=pos + n * rth["W"])
        return rth, {"Port": port, "Starboard": stbd}, pos + 2 * n * rth["W"]

    def encode(self, record):
        rd = record["RD"]
        if "Port" not in rd:
            return self.rth.pack(record["RTH"])
        dt = _UINT[record["RTH"]["W"]]
        return (self.rth.pack(record["RTH"]) + np.asarray(rd["Port"], dt).tobytes()
                + np.asarray(rd["Starboard"], dt).tobytes())


_add(_SideScan())


def _sample_dtype_7008(sample_type):
    """
    The per-sample structure of 7008 data from its "Data sample type" bit
    field: intensity (bits 0-3), phase (4-7), then I and Q (8-11).
    Reserved codes give ``None``.
    """
    fields = []
    mag = sample_type & 0xF
    phase = (sample_type >> 4) & 0xF
    iq = (sample_type >> 8) & 0xF
    if mag == 2:
        fields.append(("Magnitude", "<u2"))
    elif mag == 3:
        fields.append(("Magnitude", "<u4"))
    elif mag:
        return None
    if phase == 2:
        fields.append(("Phase", "<i2"))
    elif phase == 3:
        fields.append(("Phase", "<i4"))
    elif phase:
        return None
    if iq == 1:
        fields += [("I", "<i2"), ("Q", "<i2")]
    elif iq == 2:
        fields += [("I", "<i4"), ("Q", "<i4")]
    elif iq:
        return None
    if not fields:
        return None
    return np.dtype(fields)


class _GenericWaterColumn(Codec):
    """
    7008 - Generic Water Column Data (Tables 50-52). Deprecated.

    RD holds the beam descriptors (``Beams``) and the samples
    (``Samples``) as one structured column per sample component, in file
    order. With RowColumnFlag 0, beam 0's samples come first, then beam
    1's. With RowColumnFlag 1, sample 0 of every beam comes first. Use
    ``derived.read_watercolumn`` for a samples-by-beams image.
    """

    record_type = 7008
    name = "Generic Water Column Data"
    ping = True
    rth = Section(PING_PREFIX + [
        ("N", "u16"),
        ("Reserved1", "u16"),
        ("Samples", "u32"),
        ("RecordSubsetFlag", "u8"),
        ("RowColumnFlag", "u8"),
        ("Reserved2", "u16"),
        ("DataSampleType", "u32"),
    ])
    beams = Table([("Descriptor", "u16"), ("FirstSample", "u32"), ("LastSample", "u32")])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Beams": self.beams.new(), "Samples": {}}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        n = min(rth["N"], (len(body) - pos) // self.beams.itemsize)
        beams, used = self.beams.decode(body, pos, n)
        pos += used
        dt = _sample_dtype_7008(rth["DataSampleType"])
        if dt is None:
            return rth, {"Beams": beams, "Samples": {}}, pos
        counts = beams["LastSample"].astype(np.int64) - beams["FirstSample"].astype(np.int64) + 1
        total = int(np.clip(counts, 0, None).sum())
        total = min(total, (len(body) - pos) // dt.itemsize)
        arr = np.frombuffer(body, dtype=dt, count=total, offset=pos)
        return rth, {"Beams": beams, "Samples": {k: arr[k] for k in dt.names}}, pos + total * dt.itemsize

    def encode(self, record):
        rd = record["RD"]
        out = self.rth.pack(record["RTH"]) + self.beams.encode(rd["Beams"])
        samples = rd.get("Samples") or {}
        if samples:
            dt = _sample_dtype_7008(record["RTH"]["DataSampleType"])
            arr = np.zeros(len(next(iter(samples.values()))), dtype=dt)
            for k in dt.names:
                arr[k] = samples[k]
            out += arr.tobytes()
        return out


_add(_GenericWaterColumn())

# 7010 - TVG Values (Table 53, read from the DFD text)
_add(HeaderTableCodec(7010, "TVG Values", PING_PREFIX + [
    ("N", "u32"),
    ("Reserved", "u32*8"),
], "N", [("Gain", "f32")], "Samples", ping=True, phase=2))

# 7017 - Detection Data Setup (Tables 60-61). Deprecated.
_add(HeaderTableCodec(7017, "Detection Data Setup", PING_PREFIX + [
    ("N", "u32"),
    ("DataBlockSize", "u32"),
    ("DetectionAlgorithm", "u8"),
    ("Flags", "u32"),
    ("MinimumDepth", "f32"),
    ("MaximumDepth", "f32"),
    ("MinimumRange", "f32"),
    ("MaximumRange", "f32"),
    ("MinimumNadirSearch", "f32"),
    ("MaximumNadirSearch", "f32"),
    ("AutomaticFilterWindow", "u8"),
    ("AppliedRoll", "f32"),
    ("DepthGateTilt", "f32"),
    ("NadirDepth", "f32"),
    ("Reserved", "u32*13"),
], "N", [
    ("BeamDescriptor", "u16"),
    ("DetectionPoint", "f32"),
    ("Flags", "u32"),
    ("AutomaticLimitsMinSample", "f32"),
    ("AutomaticLimitsMaxSample", "f32"),
    ("UserDefinedLimitsMinSample", "f32"),
    ("UserDefinedLimitsMaxSample", "f32"),
    ("Quality", "u32"),
    ("Uncertainty", "f32"),
], "Detections", itemsize_field="DataBlockSize", ping=True))


class _Beamformed(Codec):
    """
    7018 - Beamformed Data (Tables 63-64). Amplitude (u16) and phase (i16)
    for every sample of every beam, stored sample by sample. Decoded as
    arrays of shape (Samples, Beams).
    """

    record_type = 7018
    name = "Beamformed Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [("N", "u16"), ("S", "u32"), ("Reserved", "u32*8")])
    table = Table([("Amplitude", "u16"), ("Phase", "i16")])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Amplitude": np.zeros((0, 0), "<u2"), "Phase": np.zeros((0, 0), "<i2")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        n, s = rth["N"], rth["S"]
        if pos + n * s * 4 > len(body):
            return rth, {}, pos
        cols, used = self.table.decode(body, pos, n * s)
        return rth, {k: v.reshape(s, n) for k, v in cols.items()}, pos + used

    def encode(self, record):
        rd = record["RD"]
        if "Amplitude" not in rd:
            return self.rth.pack(record["RTH"])
        return self.rth.pack(record["RTH"]) + self.table.encode(rd)


_add(_Beamformed())

# 7022 - Sonar Source Version (Table 69)
_add(HeaderCodec(7022, "Sonar Source Version", [("VersionString", "c8*32")]))
# 7023 - 8k Wet End Version (Table 70)
_add(HeaderCodec(7023, "8k Wet End Version", [("VersionString", "c8*32")], phase=3))

# 7027 - Raw Detection Data (Tables 71-72). The per-detection size
# (DataFieldSize) is 22 bytes in DFD 2.20-era files, 26 in 2.41 and 34 in
# 3.10. Only the fields that fit are decoded.
_add(HeaderTableCodec(7027, "Raw Detection Data", PING_PREFIX + [
    ("N", "u32"),
    ("DataFieldSize", "u32"),
    ("DetectionAlgorithm", "u8"),
    ("Flags", "u32"),
    ("SamplingRate", "f32"),
    ("TxAngle", "f32"),
    ("AppliedRoll", "f32"),
    ("Reserved", "u32*15"),
], "N", [
    ("BeamDescriptor", "u16"),
    ("DetectionPoint", "f32"),
    ("RxAngle", "f32"),
    ("Flags", "u32"),
    ("Quality", "u32"),
    ("Uncertainty", "f32"),
    ("Intensity", "f32"),
    ("MinLimit", "f32"),
    ("MaxLimit", "f32"),
], "Detections", itemsize_field="DataFieldSize", ping=True))


class _Snippet(Codec):
    """
    7028 - Snippet Data (Tables 74-75). Beam descriptors followed by each
    beam's snippet, SnippetEnd - SnippetStart + 1 samples long. Samples are
    u16, or u32 when Flags bit 0 is set. ``Samples`` holds every snippet
    concatenated in file order.
    """

    record_type = 7028
    name = "Snippet Data"
    ping = True
    rth = Section(PING_PREFIX + [
        ("N", "u16"),
        ("ErrorFlag", "u8"),
        ("ControlFlags", "u8"),
        ("Flags", "u32"),
        ("Reserved", "u32*6"),
    ])
    beams = Table([
        ("BeamDescriptor", "u16"),
        ("SnippetStart", "u32"),
        ("DetectionSample", "u32"),
        ("SnippetEnd", "u32"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Beams": self.beams.new(), "Samples": np.zeros(0, "<u2")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        n = min(rth["N"], (len(body) - pos) // self.beams.itemsize)
        beams, used = self.beams.decode(body, pos, n)
        pos += used
        dt = np.dtype("<u4" if rth["Flags"] & 1 else "<u2")
        counts = beams["SnippetEnd"].astype(np.int64) - beams["SnippetStart"].astype(np.int64) + 1
        total = min(int(np.clip(counts, 0, None).sum()), (len(body) - pos) // dt.itemsize)
        samples = np.frombuffer(body, dtype=dt, count=total, offset=pos)
        return rth, {"Beams": beams, "Samples": samples}, pos + total * dt.itemsize

    def encode(self, record):
        rd = record["RD"]
        dt = "<u4" if record["RTH"]["Flags"] & 1 else "<u2"
        return (self.rth.pack(record["RTH"]) + self.beams.encode(rd["Beams"])
                + np.asarray(rd["Samples"], dtype=dt).tobytes())


_add(_Snippet())

# 7030 - Sonar Installation Parameters (Table 78)
_add(HeaderCodec(7030, "Sonar Installation Parameters", [
    ("Frequency", "f32"),
    ("FirmwareVersionLength", "u16"),
    ("FirmwareVersion", "u8*128"),
    ("SoftwareVersionLength", "u16"),
    ("SoftwareVersion", "u8*128"),
    ("SevenKSoftwareVersionLength", "u16"),
    ("SevenKSoftwareVersion", "u8*128"),
    ("RecordProtocolVersionLength", "u16"),
    ("RecordProtocolVersion", "u8*128"),
    ("TransmitArrayX", "f32"),
    ("TransmitArrayY", "f32"),
    ("TransmitArrayZ", "f32"),
    ("TransmitArrayRoll", "f32"),
    ("TransmitArrayPitch", "f32"),
    ("TransmitArrayHeading", "f32"),
    ("ReceiveArrayX", "f32"),
    ("ReceiveArrayY", "f32"),
    ("ReceiveArrayZ", "f32"),
    ("ReceiveArrayRoll", "f32"),
    ("ReceiveArrayPitch", "f32"),
    ("ReceiveArrayHeading", "f32"),
    ("MotionSensorX", "f32"),
    ("MotionSensorY", "f32"),
    ("MotionSensorZ", "f32"),
    ("MotionSensorRollCalibration", "f32"),
    ("MotionSensorPitchCalibration", "f32"),
    ("MotionSensorHeadingCalibration", "f32"),
    ("MotionSensorTimeDelay", "u16"),
    ("PositionSensorX", "f32"),
    ("PositionSensorY", "f32"),
    ("PositionSensorZ", "f32"),
    ("PositionSensorTimeDelay", "u16"),
    ("WaterLineVerticalOffset", "f32"),
]))


class _RawIQ(Codec):
    """
    7038 - Raw IQ element data. Listed as "Reserved" in DFD 3.10. The layout
    follows the CCOM 7125 MATLAB reader (read7125_7038Record.m) and has been
    checked against 7125 files: a 60-byte header, the list of element
    numbers (u16), then interleaved I and Q samples for every element of
    every sample (elements vary fastest). SampleType 16 means 16-bit I and
    Q. Decoded as I and Q arrays of shape (samples, elements).
    """

    record_type = 7038
    name = "Raw IQ Data (undocumented)"
    ping = False      # bytes 12-13 are Reserved rather than MultiPingSequence
    phase = 1
    rth = Section([
        ("SonarId", "u64"),
        ("PingNumber", "u32"),
        ("Reserved1", "u16"),
        ("N", "u16"),
        ("S", "u32"),
        ("ElementsInRecord", "u16"),
        ("FirstSample", "u32"),
        ("LastSample", "u32"),
        ("SampleType", "u16"),
        ("Reserved2", "u32*7"),
    ])
    _SAMPLE = {16: "<i2", 32: "<i4", 8: "i1"}

    def new_rth(self):
        rth = self.rth.new()
        rth["SampleType"] = 16
        return rth

    def new_rd(self):
        return {"Elements": np.zeros(0, "<u2"), "I": np.zeros((0, 0), "<i2"), "Q": np.zeros((0, 0), "<i2")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        ne = rth["ElementsInRecord"]
        if pos + 2 * ne > len(body):
            return rth, {}, pos
        elements = np.frombuffer(body, "<u2", ne, pos)
        pos += 2 * ne
        dt = self._SAMPLE.get(rth["SampleType"])
        ns = rth["LastSample"] - rth["FirstSample"] + 1 if rth["LastSample"] >= rth["FirstSample"] else 0
        if dt is None:
            return rth, {"Elements": elements}, pos
        width = np.dtype(dt).itemsize
        if pos + ns * ne * 2 * width > len(body):
            return rth, {"Elements": elements}, pos
        iq = np.frombuffer(body, dt, ns * ne * 2, pos).reshape(ns, ne, 2)
        return rth, {"Elements": elements, "I": iq[:, :, 0], "Q": iq[:, :, 1]}, pos + ns * ne * 2 * width

    def encode(self, record):
        rd = record["RD"]
        out = self.rth.pack(record["RTH"]) + np.asarray(rd.get("Elements", ()), "<u2").tobytes()
        if "I" in rd:
            dt = self._SAMPLE[record["RTH"]["SampleType"]]
            iq = np.stack([np.asarray(rd["I"], dt), np.asarray(rd["Q"], dt)], axis=-1)
            out += iq.tobytes()
        return out


_add(_RawIQ())

# 7503 - Remote Control Sonar Settings (Table 113). Note there is no
# MultiPingSequence (Frequency follows PingNumber) and no ReceiveBeamWidth.
# The full 3.10 layout is 260 bytes, which is what 7125 systems write.
# Shorter or longer variants decode the fields present.
_add(HeaderCodec(7503, "Remote Control Sonar Settings", [
    ("SonarId", "u64"),
    ("PingNumber", "u32"),
] + _SETTINGS_7503 + [
    ("VernierOperationMode", "u8"),
    ("AutomaticFilterWindow", "u8"),
    ("TxArrayPositionOffsetX", "f32"),
    ("TxArrayPositionOffsetY", "f32"),
    ("TxArrayPositionOffsetZ", "f32"),
    ("HeadTiltX", "f32"),
    ("HeadTiltY", "f32"),
    ("HeadTiltZ", "f32"),
    ("PingState", "u32"),
    ("BeamSpacingMode", "u16"),
    ("SonarSourceMode", "u16"),
    ("AdaptiveGateBottomFilterMinDepth", "f32"),
    ("AdaptiveGateBottomFilterMaxDepth", "f32"),
    ("TriggerOutWidth", "f64"),
    ("TriggerOutOffset", "f64"),
    ("Series81xxProjectorSelection", "u16"),
    ("Reserved1", "u32*2"),
    ("Series81xxAlternateGain", "f32"),
    ("VernierFilter", "u8"),
    ("Reserved2", "u8"),
    ("CustomBeams", "u16"),
    ("CoverageAngle", "f32"),
    ("CoverageMode", "u8"),
    ("QualityFilterFlags", "u8"),
    ("HorizontalReceiverBeamSteeringAngle", "f32"),
    ("FlexModeSectorCoverage", "f32"),
    ("FlexModeSectorSteering", "f32"),
    ("ConstantSpacing", "f32"),
    ("BeamModeSelection", "u16"),
    ("DepthGateTilt", "f32"),
    ("AppliedFrequency", "f32"),
    ("ElementNumber", "u32"),
]))
