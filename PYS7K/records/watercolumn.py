"""
Water column, imagery and detection records added in phase 2:
7011, 7012, 7014, 7019, 7029, 7041, 7042, 7047, 7048, 7057, 7058.

(7007, 7008, 7010, 7018 and 7028 are in ``sonar.py``.)

Each layout is transcribed from DFD 3.10. Assumptions where the DFD is
incomplete are noted on each codec.

Verified against real data (see ``PYS7K.verification``): 7011 (Reson 7125,
2006-2015), 7012 (7kCenter 6.1, and 2010 files), 7041 (7kCenter 5.2,
which writes the DFD 2.20-2.31 header), 7042 (NORBIT STX and WBMS,
2023-2024), 7048 and 7057 (2010 7125 files; their 7048 is stored beam by
beam) and 7058 (T50-R, 7kCenter 6.3). Tested with synthetic records only:
7014, 7019, 7029 and 7047.
"""

import numpy as np

from ..fields import Section, Table
from .base import PING_PREFIX, Codec, HeaderTableCodec

CODECS = []


def _add(codec):
    CODECS.append(codec)
    return codec


class _Image(Codec):
    """
    7011 - Image Data (Tables 55-56). H rows of W pixels, ColorDepth bytes
    each, decoded as an (H, W) unsigned array. The RTH has no SonarId.

    Verified against real data: 7125 files from 2006-2015 (RDR 1.2 to
    7kCenter 6.0). The 1024 x 512 image matches the 7008 water column of
    the same ping in shape and orientation.
    """

    record_type = 7011
    name = "Image Data"
    phase = 2
    rth = Section([
        ("PingNumber", "u32"),
        ("MultiPingSequence", "u16"),
        ("W", "u32"),
        ("H", "u32"),
        ("ColorDepth", "u16"),
        ("Reserved1", "u16"),
        ("CompressionAlgorithm", "u16"),
        ("Samples", "u32"),
        ("Flags", "u32"),
        ("RxDelay", "f32"),
        ("Reserved2", "u32*6"),
    ])
    _DT = {1: "u1", 2: "<u2", 4: "<u4"}

    def new_rth(self):
        rth = self.rth.new()
        rth["ColorDepth"] = 1
        return rth

    def new_rd(self):
        return {"Image": np.zeros((0, 0), "u1")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        dt = self._DT.get(rth["ColorDepth"])
        n = rth["W"] * rth["H"]
        if dt is None or pos + n * np.dtype(dt).itemsize > len(body):
            return rth, {}, pos
        img = np.frombuffer(body, dt, n, pos).reshape(rth["H"], rth["W"])
        return rth, {"Image": img}, pos + img.nbytes

    def encode(self, record):
        out = self.rth.pack(record["RTH"])
        if "Image" in record["RD"]:
            out += np.asarray(record["RD"]["Image"], self._DT[record["RTH"]["ColorDepth"]]).tobytes()
        return out


_add(_Image())


class _PingMotion(Codec):
    """
    7012 - Ping Motion Data (Table 57). Flags bits 0-3 say which fields
    follow the header: Pitch (one f32), then Roll, Heading and Heave (N f32
    each). DFD 2.20 lists Flags as u32; the 2010 and 2015 files fit the
    3.10 u16 (their SamplingRate is plausible only at the 3.10 offset).

    Verified against real data: 7kCenter 6.1 files (2014-2015, 7125 and
    T20-P) with Flags 2 (roll only), N equal to the ping's sample count,
    and SamplingRate equal to the 7000 record's. 2010 RDR 1.2-era 7125
    files write the same layout, with N equal to the ping's 7018/7008
    sample count, but most of their records carry extra bytes after the
    arrays (anywhere from 4 bytes to 24 KB, varying from ping to ping) that
    hold roll-like values, apparently stale contents of a reused buffer.
    They are kept in ``Unparsed["Body"]``.
    """

    record_type = 7012
    name = "Ping Motion Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("N", "u32"),
        ("Flags", "u16"),
        ("ErrorFlags", "u32"),
        ("SamplingRate", "f32"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        n = rth["N"]
        rd = {}
        if rth["Flags"] & 1 and pos + 4 <= len(body):
            rd["Pitch"] = float(np.frombuffer(body, "<f4", 1, pos)[0])
            pos += 4
        for bit, name in ((1, "Roll"), (2, "Heading"), (3, "Heave")):
            if rth["Flags"] & (1 << bit):
                if pos + 4 * n > len(body):
                    break
                rd[name] = np.frombuffer(body, "<f4", n, pos)
                pos += 4 * n
        return rth, rd, pos

    def encode(self, record):
        rd = record["RD"]
        out = [self.rth.pack(record["RTH"])]
        if "Pitch" in rd:
            out.append(np.float32(rd["Pitch"]).tobytes())
        for name in ("Roll", "Heading", "Heave"):
            if name in rd:
                out.append(np.asarray(rd[name], "<f4").tobytes())
        return b"".join(out)


_add(_PingMotion())


class _AdaptiveGate(HeaderTableCodec):
    """
    7014 - Adaptive Gate (Tables 58-59). The header declares its own size
    (RecordHeaderSize); bytes beyond the known fields are kept in
    ``RD["HeaderExtra"]``.
    """

    def __init__(self):
        super().__init__(7014, "Adaptive Gate", [
            ("RecordHeaderSize", "u16"),
            ("SonarId", "u64"),
            ("PingNumber", "u32"),
            ("MultiPingSequence", "u16"),
            ("NGates", "u32"),
            ("GateDescriptorSize", "u16"),
        ], "NGates", [("Angle", "f32"), ("MinLimit", "f32"), ("MaxLimit", "f32")], "Gates",
            itemsize_field="GateDescriptorSize", phase=2)

    def new_rth(self):
        rth = super().new_rth()
        rth["RecordHeaderSize"] = self.rth.size
        return rth

    def new_rd(self):
        rd = super().new_rd()
        rd["HeaderExtra"] = b""
        return rd

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        hdr = max(rth["RecordHeaderSize"], self.rth.size)
        if hdr > len(body):
            return rth, {}, self.rth.size
        extra = bytes(body[self.rth.size:hdr])
        isz = rth["GateDescriptorSize"] or self.table.itemsize
        count = min(rth["NGates"], (len(body) - hdr) // isz)
        cols, used = self.table.decode(body, hdr, count, isz)
        return rth, {"Gates": cols, "HeaderExtra": extra}, hdr + used

    def encode(self, record):
        rd = record["RD"]
        if "Gates" not in rd:
            return self.rth.pack(record["RTH"])
        return self.rth.pack(record["RTH"]) + bytes(rd.get("HeaderExtra", b"")) + self.table.encode(rd["Gates"])


_add(_AdaptiveGate())


class _VernierRaw(Codec):
    """
    7019 - Vernier Processing Data (Raw) (Tables 65-66). Six arrays of
    D (decimated samples) x N (beams), stored one after another, sample
    by sample: Angle (i16), Magnitude, Coherence, CrossPower,
    QualityFactor and Reserved (u16). Angle in radians is
    Angle / 65536 * ElevationCoverage.
    """

    record_type = 7019
    name = "Vernier Processing Data (Raw)"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("ReferenceArray", "u8"),
        ("Pair1Array2", "u8"),
        ("Pair2Array2", "u8"),
        ("DecimationFactor", "u8"),
        ("N", "u16"),
        ("S", "u32"),
        ("D", "u32"),
        ("FirstSample", "u32"),
        ("Reserved1", "u32*2"),
        ("SmoothingWindowType", "u16"),
        ("SmoothingWindowLength", "u16"),
        ("Reserved2", "u32*2"),
        ("IntensityThreshold", "f32"),
        ("MinQF", "f32"),
        ("MaxQF", "f32"),
        ("MinAngle", "f32"),
        ("MaxAngle", "f32"),
        ("ElevationCoverage", "f32"),
        ("Reserved3", "u32*4"),
    ])
    _PARTS = [("Angle", "<i2"), ("Magnitude", "<u2"), ("Coherence", "<u2"), ("CrossPower", "<u2"),
              ("QualityFactor", "<u2"), ("Reserved", "<u2")]

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {name: np.zeros((0, 0), dt) for name, dt in self._PARTS}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        d, n = rth["D"], rth["N"]
        rd = {}
        for name, dt in self._PARTS:
            if pos + 2 * d * n > len(body):
                break
            rd[name] = np.frombuffer(body, dt, d * n, pos).reshape(d, n)
            pos += 2 * d * n
        return rth, rd, pos

    def encode(self, record):
        out = [self.rth.pack(record["RTH"])]
        for name, dt in self._PARTS:
            if name not in record["RD"]:
                break
            out.append(np.asarray(record["RD"][name], dt).tobytes())
        return b"".join(out)


_add(_VernierRaw())

# 7029 - Vernier Processing Data (Filtered) (Tables 76-77)
_add(HeaderTableCodec(7029, "Vernier Processing Data (Filtered)", PING_PREFIX + [
    ("S", "u16"),
    ("MinAngle", "f32"),
    ("MaxAngle", "f32"),
    ("RepeatSize", "u16"),
], "S", [
    ("BeamAngle", "f32"),
    ("Sample", "u32"),
    ("Elevation", "f32"),
    ("Reserved", "f32"),
], "Soundings", itemsize_field="RepeatSize", ping=True, phase=2))


class _CompressedBeamformed(Codec):
    """
    7041 - Compressed Beamformed Intensity Data (Tables 80-81).

    For each beam: an identifier (u16 beam number, or f32 angle when Flags
    bit 8 is set), a u32 sample count, then the samples. The DFD refers to
    a "Data Size" it never defines. The sample width (1 or 2 bytes) is
    taken from Flags bits 0-1 when that is 1 or 2, and otherwise chosen as
    the width that makes the beams fill the record exactly. ``Beams`` holds
    each beam's identifier and count; ``Samples`` holds every beam's
    samples concatenated.

    Two header forms exist, and the one that makes the beams fill the
    record exactly is used (the DFD 3.10 form is tried first):

    * DFD 2.41 and later (38 bytes): Flags is u16, with the beam
      identifier type in bit 8.
    * DFD 2.20 to 2.31 (2011-2013, 44 bytes): DownsamplingMethod (u8),
      FilteringMethod (u8), Flags (u32: bits 0-3 data encoding, bit 4 beam
      identifier type), MagnitudeEncoding (u8) and PhaseEncoding (u8) in
      place of the u16 Flags. Data encodings 0-6 are 16-bit magnitude, 8-bit
      magnitude, 16-bit magnitude and phase, 8-bit magnitude and phase,
      32-bit, 16-bit and 8-bit calibrated magnitude. Such records decode
      with those extra RTH keys.

    Verified against real data: 7125 files from 7kCenter 5.2.2.7 (2013)
    use the DFD 2.2x header (DownsamplingMethod 1, nearest neighbour;
    FilteringMethod 0; Flags 0, 16-bit magnitude by beam number). An
    earlier pys7k release read its first two bytes as a u16 Flags of 1
    followed by six "undocumented" bytes. Their SampleRate (999.5 Hz) and
    sample counts (198 per beam) match the 7000 range setting (150 m).
    For DFD 3.10 headers, Flags bits 0-1 are "always 1 for backward
    compatibility with 7111 systems", so the width is found by fitting
    the record whenever more than one width is possible.
    """

    record_type = 7041
    name = "Compressed Beamformed Intensity Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("N", "u16"),
        ("Flags", "u16"),
        ("SampleRate", "f32"),
        ("Reserved", "u32*4"),
    ])
    # DFD 2.20-2.31 (Table 68 in 2.20)
    rth_dfd22 = Section(PING_PREFIX + [
        ("N", "u16"),
        ("DownsamplingMethod", "u8"),
        ("FilteringMethod", "u8"),
        ("Flags", "u32"),
        ("MagnitudeEncoding", "u8"),
        ("PhaseEncoding", "u8"),
        ("SampleRate", "f32"),
        ("Reserved", "u32*4"),
    ])
    # DFD 2.2x data encoding (Flags bits 0-3) -> bytes per sample
    _DFD22_WIDTH = {0: 2, 1: 1, 2: 4, 3: 2, 4: 4, 5: 2, 6: 1}
    _DT = {1: "u1", 2: "<u2", 4: "<u4"}

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Beams": {"Beam": np.zeros(0, "<u2"), "Samples": np.zeros(0, "<u4")},
                "Samples": np.zeros(0, "u1"), "SampleWidth": 1}

    def _walk(self, body, pos, n, id_dt, width):
        ids, counts, starts = [], [], []
        idsize = np.dtype(id_dt).itemsize
        for _ in range(n):
            if pos + idsize + 4 > len(body):
                return None
            ids.append(np.frombuffer(body, id_dt, 1, pos)[0])
            cnt = int(np.frombuffer(body, "<u4", 1, pos + idsize)[0])
            pos += idsize + 4
            if pos + cnt * width > len(body):
                return None
            counts.append(cnt)
            starts.append(pos)
            pos += cnt * width
        return ids, counts, starts, pos

    def decode(self, body):
        best = None
        for section in (self.rth, self.rth_dfd22):
            if len(body) < section.size:
                continue
            rth = section.unpack(body, 0)
            if section is self.rth:
                id_dt = "<f4" if rth["Flags"] & 0x100 else "<u2"
                code = rth["Flags"] & 3
                widths = [code] if code in (1, 2) else []
            else:
                id_dt = "<f4" if rth["Flags"] & 0x10 else "<u2"
                code = self._DFD22_WIDTH.get(rth["Flags"] & 0xF)
                widths = [code] if code else []
            widths += [w for w in (1, 2) if w not in widths]
            for width in widths:
                walk = self._walk(body, section.size, rth["N"], id_dt, width)
                if walk is None:
                    continue
                if walk[3] == len(body):
                    best = (rth, id_dt, width, walk)
                    break
                if best is None:
                    best = (rth, id_dt, width, walk)
            if best is not None and best[3][3] == len(body):
                break
        if best is None:
            rth = self.rth.unpack(body, 0)
            return rth, {}, self.rth.size
        rth, id_dt, width, walk = best
        ids, counts, starts, end = walk
        dt = self._DT[width]
        samples = np.concatenate([np.frombuffer(body, dt, c, s) for c, s in zip(counts, starts)]) \
            if counts else np.zeros(0, dt)
        rd = {"Beams": {"Beam": np.array(ids, dtype=id_dt), "Samples": np.array(counts, dtype="<u4")},
              "Samples": samples, "SampleWidth": width}
        return rth, rd, end

    def encode(self, record):
        rd = record["RD"]
        dfd22 = record["RTH"].get("DownsamplingMethod") is not None
        section = self.rth_dfd22 if dfd22 else self.rth
        out = [section.pack(record["RTH"])]
        if "Beams" not in rd:
            return out[0]
        id_dt = "<f4" if record["RTH"]["Flags"] & (0x10 if dfd22 else 0x100) else "<u2"
        dt = self._DT[rd.get("SampleWidth", 1)]
        samples = np.asarray(rd["Samples"], dt)
        pos = 0
        for bid, cnt in zip(rd["Beams"]["Beam"], rd["Beams"]["Samples"]):
            out.append(np.asarray(bid, id_dt).tobytes() + np.uint32(cnt).tobytes())
            out.append(samples[pos:pos + int(cnt)].tobytes())
            pos += int(cnt)
        return b"".join(out)


_add(_CompressedBeamformed())


def sample_dtype_7042(flags):
    """
    The per-sample structure of 7042 data from its Flags (Table 82-83):

    ============================  ===============================
    Flags                         Sample
    ============================  ===============================
    (none)                        u16 magnitude, i16 phase
    bit 1                         u16 magnitude
    bit 2                         u8 magnitude (dB), i8 phase
    bits 1+2                      u8 magnitude (dB)
    bit 12                        u32 magnitude, i8 phase
    bits 1+12                     u32 magnitude
    ============================  ===============================
    """
    mag_only = flags & 0x2
    if flags & 0x4:
        fields = [("Magnitude", "u1")] + ([] if mag_only else [("Phase", "i1")])
    elif flags & 0x1000:
        fields = [("Magnitude", "<u4")] + ([] if mag_only else [("Phase", "i1")])
    else:
        fields = [("Magnitude", "<u2")] + ([] if mag_only else [("Phase", "<i2")])
    return np.dtype(fields)


class _CompressedWaterColumn(Codec):
    """
    7042 - Compressed Water Column Data (Tables 82-83, Appendix I).

    For each beam: Beam (u16), SegmentNumber (u8, only when Flags bit 14 is
    set), Samples (u32), then the samples (see ``sample_dtype_7042``).
    ``Beams`` holds each beam's number, segment and sample count;
    ``Samples`` holds every beam's samples concatenated, as one column per
    component.

    Verified against real data: NORBIT STX (2024) and WBMS (2023) files,
    including 2x middle-value downsampling (Flags 0x122) with FirstSample
    in downsampled units; 7027 detections line up with the bottom echo
    (see ``derived.read_watercolumn``).
    """

    record_type = 7042
    name = "Compressed Water Column Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("N", "u16"),
        ("Samples", "u32"),
        ("CompressedSamples", "u32"),
        ("Flags", "u32"),
        ("FirstSample", "u32"),
        ("SampleRate", "f32"),
        ("CompressionFactor", "f32"),
        ("Reserved", "u32"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Beams": {"Beam": np.zeros(0, "<u2"), "Samples": np.zeros(0, "<u4")}, "Samples": {}}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        flags = rth["Flags"]
        has_seg = bool(flags & 0x4000)
        dt = sample_dtype_7042(flags)
        head = 2 + (1 if has_seg else 0) + 4
        beams, segs, counts, chunks = [], [], [], []
        for _ in range(rth["N"]):
            if pos + head > len(body):
                break
            beam = int(np.frombuffer(body, "<u2", 1, pos)[0])
            seg = int(body[pos + 2]) if has_seg else None
            cnt = int(np.frombuffer(body, "<u4", 1, pos + head - 4)[0])
            if pos + head + cnt * dt.itemsize > len(body):
                break
            chunks.append(np.frombuffer(body, dt, cnt, pos + head))
            beams.append(beam)
            segs.append(seg)
            counts.append(cnt)
            pos += head + cnt * dt.itemsize
        samples = np.concatenate(chunks) if chunks else np.zeros(0, dt)
        table = {"Beam": np.array(beams, "<u2"), "Samples": np.array(counts, "<u4")}
        if has_seg:
            table["SegmentNumber"] = np.array(segs, "u1")
        return rth, {"Beams": table, "Samples": {k: samples[k] for k in dt.names}}, pos

    def encode(self, record):
        rd = record["RD"]
        flags = record["RTH"]["Flags"]
        dt = sample_dtype_7042(flags)
        has_seg = bool(flags & 0x4000)
        out = [self.rth.pack(record["RTH"])]
        beams = rd["Beams"]
        n = len(beams["Beam"])
        cols = rd.get("Samples") or {}
        total = int(np.sum(beams["Samples"])) if n else 0
        arr = np.zeros(total, dt)
        for k in dt.names:
            if k in cols:
                arr[k] = cols[k]
        pos = 0
        for i in range(n):
            cnt = int(beams["Samples"][i])
            out.append(np.uint16(beams["Beam"][i]).tobytes())
            if has_seg:
                out.append(np.uint8(beams["SegmentNumber"][i]).tobytes())
            out.append(np.uint32(cnt).tobytes())
            out.append(arr[pos:pos + cnt].tobytes())
            pos += cnt
        return b"".join(out)


_add(_CompressedWaterColumn())


class _SegmentedDetection(Codec):
    """
    7047 - Segmented Raw Detection Data (Tables 84-85). The header declares
    its own size and the size of each segment and detection entry, so
    larger entries from newer firmware decode their known fields and keep
    the rest as ``Unparsed`` columns.
    """

    record_type = 7047
    name = "Segmented Raw Detection Data"
    phase = 2
    rth = Section([
        ("RecordHeaderSize", "u16"),
        ("NSegments", "u32"),
        ("SegmentFieldSize", "u16"),
        ("NRx", "u32"),
        ("RxFieldSize", "u16"),
        ("SonarId", "u64"),
        ("PingNumber", "u32"),
        ("MultiPingSequence", "u16"),
        ("SoundVelocity", "f32"),
        ("RxDelay", "f32"),
    ])
    segments = Table([
        ("Number", "u16"),
        ("TxAngleAlong", "f32"),
        ("TxAngleCross", "f32"),
        ("TxDelay", "f32"),
        ("Frequency", "f32"),
        ("PulseType", "u32"),
        ("PulseBandwidth", "f32"),
        ("TxPulseWidth", "f32"),
        ("TxBeamWidthCross", "f32"),
        ("TxBeamWidthAlong", "f32"),
        ("TxPulseEnvelope", "u32"),
        ("TxPulseEnvelopeParameter", "f32"),
        ("TxRelativeSourceLevel", "f32"),
        ("RxBeamWidth", "f32"),
        ("DetectionAlgorithm", "u8"),
        ("Flags", "u32"),
        ("SamplingRate", "f32"),
        ("TVG", "u8"),
        ("RxBandwidth", "f32"),
    ])
    detections = Table([
        ("BeamNumber", "u16"),
        ("UsedSegment", "u16"),
        ("DetectionPoint", "f32"),
        ("RxAngleCross", "f32"),
        ("Flags", "u32"),
        ("Quality", "u32"),
        ("Uncertainty", "f32"),
        ("SignalStrength", "f32"),
        ("SNRatio", "f32"),
    ])

    def new_rth(self):
        rth = self.rth.new()
        rth.update(RecordHeaderSize=self.rth.size, SegmentFieldSize=self.segments.itemsize,
                   RxFieldSize=self.detections.itemsize)
        return rth

    def new_rd(self):
        return {"HeaderExtra": b"", "Segments": self.segments.new(), "Detections": self.detections.new()}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        hdr = max(rth["RecordHeaderSize"], self.rth.size)
        if hdr > len(body):
            return rth, {}, self.rth.size
        rd = {"HeaderExtra": bytes(body[self.rth.size:hdr])}
        pos = hdr
        for key, table, count, size in (("Segments", self.segments, rth["NSegments"], rth["SegmentFieldSize"]),
                                        ("Detections", self.detections, rth["NRx"], rth["RxFieldSize"])):
            size = size or table.itemsize
            count = min(count, (len(body) - pos) // size)
            rd[key], used = table.decode(body, pos, count, size)
            pos += used
        return rth, rd, pos

    def encode(self, record):
        rd = record["RD"]
        if "Segments" not in rd:
            return self.rth.pack(record["RTH"])
        return (self.rth.pack(record["RTH"]) + bytes(rd.get("HeaderExtra", b""))
                + self.segments.encode(rd["Segments"]) + self.detections.encode(rd["Detections"]))


_add(_SegmentedDetection())


class _CalibratedBeam(Codec):
    """
    7048 - Calibrated Beam Data (Tables 86-87): S x N f32 intensities,
    decoded as an (S, N) array ``Intensity``.

    DFD 3.10 stores them sample by sample. Verified against real data:
    RDR 1.2-era 7125 files from 2010 (ErrorFlag 5, so the values are the
    uncalibrated beamformed amplitudes) store them beam by beam instead,
    and end with N x 14 zero bytes. Stored beam by beam, they equal the
    same ping's 7018 amplitudes exactly; read sample by sample they do
    not. Records with that trailing padding are therefore decoded beam by
    beam, with ``RD["BeamMajor"]`` True (the padding stays in
    ``Unparsed["Body"]``), and are written back in the order they were
    stored. Records of the DFD size are decoded sample by sample, which
    has not been checked against real data.
    """

    record_type = 7048
    name = "Calibrated Beam Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("FirstBeam", "u16"),
        ("N", "u16"),
        ("S", "u32"),
        ("ForwardLookingSonar", "u8"),
        ("ErrorFlag", "u8"),
        ("Reserved", "u32*8"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Intensity": np.zeros((0, 0), "<f4")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        nb, ns = rth["N"], rth["S"]
        n = nb * ns
        if pos + 4 * n > len(body):
            return rth, {}, pos
        values = np.frombuffer(body, "<f4", n, pos)
        extra = bytes(body[pos + 4 * n:])
        if nb and len(extra) == 14 * nb and not any(extra):
            return rth, {"Intensity": values.reshape(nb, ns).T, "BeamMajor": True}, pos + 4 * n
        return rth, {"Intensity": values.reshape(ns, nb)}, pos + 4 * n

    def encode(self, record):
        out = self.rth.pack(record["RTH"])
        rd = record["RD"]
        if "Intensity" in rd:
            data = np.asarray(rd["Intensity"], "<f4")
            if rd.get("BeamMajor"):
                data = data.T
            out += np.ascontiguousarray(data).tobytes()
        return out


_add(_CalibratedBeam())


class _CalibratedSideScan(Codec):
    """
    7057 - Calibrated Side-Scan Data (Tables 98-99). Port then starboard
    series of S samples (f32 when W is 4, f64 when W is 8), then the port
    and starboard beam numbers (S u16 each).

    ``Reserved1`` (8 f32) held the port and starboard beam widths (y, z)
    and steering angles (y, z) in DFD 2.20-2.31; DFD 2.41 reserved them.
    They are zero in the 2010 files seen.

    Verified against real data: 2010 RDR 1.2-era 7125 files (W = 4, with
    ErrorFlag 5) decode with no bytes left over and plausible series.
    """

    record_type = 7057
    name = "Calibrated Side-Scan Data"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("BeamPosition", "f32"),
        ("ControlFlags", "u32"),
        ("S", "u32"),
        ("Reserved1", "f32*8"),
        ("N", "u16"),
        ("CurrentBeamNumber", "u16"),
        ("W", "u8"),
        ("DataTypes", "u8"),
        ("ErrorFlag", "u8"),
    ])
    _DT = {4: "<f4", 8: "<f8"}

    def new_rth(self):
        rth = self.rth.new()
        rth["W"] = 4
        return rth

    def new_rd(self):
        return {"Port": np.zeros(0, "<f4"), "Starboard": np.zeros(0, "<f4"),
                "PortBeams": np.zeros(0, "<u2"), "StarboardBeams": np.zeros(0, "<u2")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        dt = self._DT.get(rth["W"])
        s = rth["S"]
        if dt is None or pos + s * (2 * rth["W"] + 4) > len(body):
            return rth, {}, pos
        rd = {}
        for name, d in (("Port", dt), ("Starboard", dt), ("PortBeams", "<u2"), ("StarboardBeams", "<u2")):
            rd[name] = np.frombuffer(body, d, s, pos)
            pos += s * np.dtype(d).itemsize
        return rth, rd, pos

    def encode(self, record):
        rd = record["RD"]
        out = [self.rth.pack(record["RTH"])]
        if "Port" in rd:
            dt = self._DT[record["RTH"]["W"]]
            for name, d in (("Port", dt), ("Starboard", dt), ("PortBeams", "<u2"), ("StarboardBeams", "<u2")):
                out.append(np.asarray(rd[name], d).tobytes())
        return b"".join(out)


_add(_CalibratedSideScan())


class _SnippetBackscatter(Codec):
    """
    7058 - Snippet Backscattering Strength (Tables 100-101). Beam
    descriptors, then every beam's backscattering strength (f32, dB) and,
    when ControlFlags bit 6 is set, every beam's footprint areas (f32, m^2).
    A beam whose EndSample is below its BeginSample has no samples (DFD
    note). ``BackscatterStrength`` and ``Footprints`` hold all beams
    concatenated.

    Verified against real data: T50-R files from 7kCenter 6.3 (2018), with
    footprints (ControlFlags bit 6). The beams' sample counts add up to the
    stored array lengths exactly, N equals the same ping's 7027 N, and the
    median backscatter is about -31 dB.
    """

    record_type = 7058
    name = "Snippet Backscattering Strength"
    ping = True
    phase = 2
    rth = Section(PING_PREFIX + [
        ("N", "u16"),
        ("ErrorFlag", "u8"),
        ("ControlFlags", "u32"),
        ("Absorption", "f32"),
        ("Reserved", "u32*6"),
    ])
    beams = Table([
        ("BeamDescriptor", "u16"),
        ("BeginSample", "u32"),
        ("BottomDetectionSample", "u32"),
        ("EndSample", "u32"),
    ])

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {"Beams": self.beams.new(), "BackscatterStrength": np.zeros(0, "<f4")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        n = min(rth["N"], (len(body) - pos) // self.beams.itemsize)
        beams, used = self.beams.decode(body, pos, n)
        pos += used
        total = int(np.clip(beams["EndSample"].astype(np.int64) - beams["BeginSample"].astype(np.int64) + 1,
                            0, None).sum())
        rd = {"Beams": beams}
        if pos + 4 * total > len(body):
            return rth, rd, pos
        rd["BackscatterStrength"] = np.frombuffer(body, "<f4", total, pos)
        pos += 4 * total
        if rth["ControlFlags"] & 0x40 and pos + 4 * total <= len(body):
            rd["Footprints"] = np.frombuffer(body, "<f4", total, pos)
            pos += 4 * total
        return rth, rd, pos

    def encode(self, record):
        rd = record["RD"]
        out = [self.rth.pack(record["RTH"]), self.beams.encode(rd["Beams"])]
        if "BackscatterStrength" in rd:
            out.append(np.asarray(rd["BackscatterStrength"], "<f4").tobytes())
        if "Footprints" in rd:
            out.append(np.asarray(rd["Footprints"], "<f4").tobytes())
        return b"".join(out)


_add(_SnippetBackscatter())
