"""Populated synthetic round trips for the phase 2 and 3 record codecs
(no sample files contain these records)."""

import numpy as np
import pytest

from PYS7K import RECORD_CODECS, decode_record, encode_record, new
from PYS7K.records.watercolumn import sample_dtype_7042


def round_trip(rec):
    data = encode_record(rec)
    out = decode_record(bytearray(data))
    assert "Error" not in out
    assert not out["Unparsed"]["Body"], "%d unparsed bytes" % len(out["Unparsed"]["Body"])
    assert encode_record(out) == data
    return data, out


def test_7011_image():
    rec = new(7011)
    rec["RTH"].update(W=3, H=2, ColorDepth=2)
    rec["RD"]["Image"] = np.arange(6, dtype="<u2").reshape(2, 3)
    _, out = round_trip(rec)
    assert out["RD"]["Image"].shape == (2, 3) and out["RD"]["Image"][1, 2] == 5


@pytest.mark.parametrize("flags", [0b0001, 0b1110, 0b1111])
def test_7012_ping_motion(flags):
    rec = new(7012)
    rec["RTH"].update(N=3, Flags=flags)
    rd = rec["RD"]
    if flags & 1:
        rd["Pitch"] = 0.5
    for bit, name in ((1, "Roll"), (2, "Heading"), (3, "Heave")):
        if flags & (1 << bit):
            rd[name] = np.arange(3, dtype="<f4") * bit
    data, out = round_trip(rec)
    assert sorted(out["RD"]) == sorted(rd)


def test_7014_gates_with_larger_header():
    rec = new(7014)
    rec["RTH"].update(NGates=2, RecordHeaderSize=RECORD_CODECS[7014].rth.size + 2)
    rec["RD"]["HeaderExtra"] = b"\x01\x02"
    rec["RD"]["Gates"] = {"Angle": np.array([0.1, 0.2], "<f4"), "MinLimit": np.zeros(2, "<f4"),
                          "MaxLimit": np.ones(2, "<f4")}
    _, out = round_trip(rec)
    assert out["RD"]["HeaderExtra"] == b"\x01\x02"
    assert np.allclose(out["RD"]["Gates"]["Angle"], [0.1, 0.2])


def test_7019_vernier_raw():
    rec = new(7019)
    rec["RTH"].update(N=2, D=3)
    rec["RD"] = {name: np.full((3, 2), i, dtype=dt) for i, (name, dt) in enumerate(RECORD_CODECS[7019]._PARTS)}
    _, out = round_trip(rec)
    assert out["RD"]["QualityFactor"][2, 1] == 4


def test_7029_vernier_filtered():
    rec = new(7029)
    rec["RTH"].update(S=2)
    rec["RD"]["Soundings"] = {"BeamAngle": np.zeros(2, "<f4"), "Sample": np.array([5, 6], "<u4"),
                              "Elevation": np.ones(2, "<f4"), "Reserved": np.zeros(2, "<f4")}
    _, out = round_trip(rec)
    assert list(out["RD"]["Soundings"]["Sample"]) == [5, 6]


@pytest.mark.parametrize("flags,width,id_dt", [(1, 1, "<u2"), (2, 2, "<u2"), (0x101, 1, "<f4"), (0, 2, "<u2")])
def test_7041_compressed_beamformed(flags, width, id_dt):
    rec = new(7041)
    rec["RTH"].update(N=2, Flags=flags)
    rec["RD"] = {"Beams": {"Beam": np.array([3, 4], id_dt), "Samples": np.array([2, 3], "<u4")},
                 "Samples": np.arange(5, dtype="u1" if width == 1 else "<u2"), "SampleWidth": width}
    _, out = round_trip(rec)
    assert out["RD"]["SampleWidth"] == width
    assert list(out["RD"]["Samples"]) == [0, 1, 2, 3, 4]


@pytest.mark.parametrize("flags", [0, 0x2, 0x4, 0x6, 0x1000, 0x1002, 0x4000, 0x4004])
def test_7042_compressed_water_column(flags):
    dt = sample_dtype_7042(flags)
    rec = new(7042)
    rec["RTH"].update(N=2, Flags=flags)
    beams = {"Beam": np.array([0, 1], "<u2"), "Samples": np.array([3, 2], "<u4")}
    if flags & 0x4000:
        beams["SegmentNumber"] = np.array([1, 2], "u1")
    rec["RD"] = {"Beams": beams, "Samples": {k: np.arange(5) for k in dt.names}}
    data, out = round_trip(rec)
    assert sorted(out["RD"]["Samples"]) == sorted(dt.names)
    head = 2 + (1 if flags & 0x4000 else 0) + 4
    assert len(data) == 64 + RECORD_CODECS[7042].rth.size + 2 * head + 5 * dt.itemsize + 4


def test_7042_sample_types():
    assert sample_dtype_7042(0).names == ("Magnitude", "Phase")
    assert sample_dtype_7042(0x6).itemsize == 1
    assert sample_dtype_7042(0x1000).itemsize == 5


def test_7047_segmented_detections():
    rec = new(7047)
    seg = {k: np.zeros(1, v.dtype) for k, v in rec["RD"]["Segments"].items()}
    det = {k: np.arange(3).astype(v.dtype) for k, v in rec["RD"]["Detections"].items()}
    rec["RTH"].update(NSegments=1, NRx=3)
    rec["RD"]["Segments"], rec["RD"]["Detections"] = seg, det
    data, out = round_trip(rec)
    assert RECORD_CODECS[7047].segments.itemsize == 68 and RECORD_CODECS[7047].detections.itemsize == 32
    assert list(out["RD"]["Detections"]["BeamNumber"]) == [0, 1, 2]


def test_7048_calibrated_beams():
    rec = new(7048)
    rec["RTH"].update(N=2, S=3)
    rec["RD"]["Intensity"] = np.arange(6, dtype="<f4").reshape(3, 2)
    _, out = round_trip(rec)
    assert out["RD"]["Intensity"][2, 1] == 5


@pytest.mark.parametrize("w", [4, 8])
def test_7057_calibrated_sidescan(w):
    rec = new(7057)
    rec["RTH"].update(S=2, W=w)
    dt = "<f4" if w == 4 else "<f8"
    rec["RD"] = {"Port": np.array([1, 2], dt), "Starboard": np.array([3, 4], dt),
                 "PortBeams": np.array([0, 1], "<u2"), "StarboardBeams": np.array([2, 3], "<u2")}
    _, out = round_trip(rec)
    assert out["RD"]["Port"].dtype == np.dtype(dt)


@pytest.mark.parametrize("footprints", [False, True])
def test_7058_snippet_backscatter(footprints):
    rec = new(7058)
    rec["RTH"].update(N=3, ControlFlags=0x40 if footprints else 0)
    rec["RD"]["Beams"] = {"BeamDescriptor": np.array([0, 1, 2], "<u2"),
                          "BeginSample": np.array([10, 20, 30], "<u4"),
                          "BottomDetectionSample": np.array([11, 21, 30], "<u4"),
                          "EndSample": np.array([12, 21, 29], "<u4")}     # beam 2 has no samples
    rec["RD"]["BackscatterStrength"] = -np.arange(5, dtype="<f4")
    if footprints:
        rec["RD"]["Footprints"] = np.ones(5, "<f4")
    _, out = round_trip(rec)
    assert len(out["RD"]["BackscatterStrength"]) == 5
    assert ("Footprints" in out["RD"]) == footprints


def test_7021_bite():
    rec = new(7021)
    rec["RTH"]["N"] = 1
    board = RECORD_CODECS[7021].board.new()
    board.update(SourceName=b"TX1".ljust(64, b"\0"), NBF=2)
    fields = RECORD_CODECS[7021].fields.new(2)
    fields["NameAndValueText"][:] = [b"PA5V\x005.02", b"TEMP"]
    fields["Value"][:] = [5.02, 30.0]
    board["BiteFields"] = fields
    rec["RD"]["Boards"] = [board]
    data, out = round_trip(rec)
    assert out["RD"]["Boards"][0]["BiteFields"]["NameAndValueText"][0] == b"PA5V\x005.02"


def test_7050_events():
    rec = new(7050)
    codec = RECORD_CODECS[7050]
    events = []
    for msg in (b"started\0", b"stopped pinging\0"):
        e = codec.new_entry()
        e.update(EventType=1, EventMessageLength=len(msg), EventMessage=msg)
        events.append(e)
    rec["RTH"]["N"] = 2
    rec["RD"]["Events"] = events
    _, out = round_trip(rec)
    assert [e["EventMessage"] for e in out["RD"]["Events"]] == [b"started\0", b"stopped pinging\0"]


def test_7051_message():
    rec = new(7051)
    rec["RTH"]["MessageLength"] = 6
    rec["RD"]["EventMessage"] = b"hello\0"
    _, out = round_trip(rec)
    assert out["RD"]["EventMessage"] == b"hello\0"


def test_7052_arrays():
    rec = new(7052)
    rec["RD"]["IncludedRecords"] = np.array([7000, 7027], "<u4")
    _, out = round_trip(rec)
    assert list(out["RD"]["IncludedRecords"]) == [7000, 7027]


def test_7053_subscriptions():
    rec = new(7053)
    e = RECORD_CODECS[7053].new_entry()
    e.update(Port=7000, NumberOfRecords=2, RecordList=(7000, 7027) + (0,) * 62)
    rec["RTH"]["N"] = 1
    rec["RD"]["Subscriptions"] = [e]
    data, out = round_trip(rec)
    assert out["RD"]["Subscriptions"][0]["RecordList"][:2] == (7000, 7027)
    assert len(data) == 64 + 4 + 12 + 4 * 192 + 4


def test_7500_remote_control_data():
    rec = new(7500)
    rec["RTH"].update(RemoteControlId=1003, Ticket=7)
    rec["RD"]["Data"] = np.float32(50.0).tobytes()
    _, out = round_trip(rec)
    assert out["RD"]["Data"] == np.float32(50.0).tobytes()


@pytest.mark.parametrize("bps,dt", [(8, "u1"), (16, "<u2"), (32, "<u4")])
def test_10018_echogram(bps, dt):
    rec = new(10018)
    rec["RTH"].update(BitsPerSample=bps, N=8)
    rec["RD"]["Samples"] = np.arange(8, dtype=dt)
    _, out = round_trip(rec)
    assert out["RD"]["Samples"].dtype == np.dtype(dt)


def test_10027_sbes_detections():
    rec = new(10027)
    rec["RTH"]["N"] = 2
    rec["RD"]["Detections"] = {k: np.ones(2, v.dtype) for k, v in rec["RD"]["Detections"].items()}
    data, out = round_trip(rec)
    assert len(data) == 64 + RECORD_CODECS[10027].rth.size + 2 * 32 + 4


def test_every_documented_type_has_codec():
    documented = [7000, 7001, 7002, 7003, 7004, 7006, 7007, 7008, 7010, 7011, 7012, 7014, 7017, 7018, 7019,
                  7021, 7022, 7023, 7027, 7028, 7029, 7030, 7031, 7041, 7042, 7047, 7048, 7050, 7051, 7052,
                  7053, 7054, 7055, 7057, 7058, 7059, 7200, 7300, 7400, 7500, 7501, 7502, 7503, 7504, 7510,
                  7511, 7610, 7611, 7612, 7613, 7614, 1000, 1001, 1002, 1003, 1004, 1005, 1006, 1007, 1008,
                  1009, 1010, 1011, 1012, 1013, 1014, 1015, 1016, 1017, 1020, 2004, 3001, 10000, 10003,
                  10004, 10018, 10027, 10049, 10060, 10061, 10062]
    assert [rt for rt in documented if rt not in RECORD_CODECS] == []


def test_7009_vertical_depth():
    rec = new(7009)
    rec["RTH"].update(Frequency=396000.0, PingNumber=385268, Latitude=0.8788, VerticalDepth=29.74)
    data, out = round_trip(rec)
    assert len(data) == 64 + 42 + 4
    assert out["RTH"]["PingNumber"] == 385268


def test_7028_pds2000_misplaced_optional_data():
    """PDS2000 writes u16 snippets but sizes the record (and places the
    optional data offset) as if they were u32 (see
    recover_misplaced_optional_data)."""
    import struct
    from PYS7K.frame import checksum
    from PYS7K.optional_data import OPTIONAL_DATA_CODECS, recover_misplaced_optional_data
    rec = new(7028)
    rec["RTH"]["N"] = 2
    rec["RD"]["Beams"] = {"BeamDescriptor": np.array([0, 1], "<u2"), "SnippetStart": np.array([10, 40], "<u4"),
                          "DetectionSample": np.array([25, 55], "<u4"), "SnippetEnd": np.array([39, 69], "<u4")}
    rec["RD"]["Samples"] = np.arange(1, 61, dtype="<u2")           # 60 samples, 120 bytes
    od = OPTIONAL_DATA_CODECS[7028].new()
    od.update(Frequency=396000.0, Latitude=0.88)
    od["Beams"] = {"AlongTrackDistance": np.ones(2, "<f4"), "AcrossTrackDistance": np.ones(2, "<f4"),
                   "CenterSampleNumber": np.array([25, 55], "<u4")}
    good = encode_record(rec)                          # no optional data
    od_bytes = OPTIONAL_DATA_CODECS[7028].encode(od)
    extra = od_bytes + bytes(120 - len(od_bytes))      # as long as the u16 snippets
    stale = b"\x5a" * len(od_bytes)                    # declared optional data region
    body = bytearray(good[:-4]) + extra + stale
    struct.pack_into("<I", body, 8, len(body) + 4)                     # Size
    struct.pack_into("<I", body, 12, len(good) - 4 + len(extra))       # OptionalDataOffset
    struct.pack_into("<I", body, 16, 7028)
    data = bytes(body) + struct.pack("<I", checksum(body))
    out = decode_record(bytearray(data))
    assert encode_record(out) == data
    assert len(out["Unparsed"]["Body"]) == 120
    got = recover_misplaced_optional_data(out)
    assert got["Frequency"] == 396000.0 and list(got["Beams"]["CenterSampleNumber"]) == [25, 55]
    assert recover_misplaced_optional_data(decode_record(bytearray(good))) is None


def test_7028_pds2000_optional_data_spills_into_declared_region():
    """When the u16 snippets are shorter than the optional data, PDS's real
    optional data runs on past the unparsed body into the declared region."""
    import struct
    from PYS7K.frame import checksum
    from PYS7K.optional_data import OPTIONAL_DATA_CODECS, recover_misplaced_optional_data
    codec = OPTIONAL_DATA_CODECS[7028]
    rec = new(7028)
    rec["RTH"]["N"] = 2
    rec["RD"]["Beams"] = {"BeamDescriptor": np.array([0, 1], "<u2"), "SnippetStart": np.array([10, 20], "<u4"),
                          "DetectionSample": np.array([11, 21], "<u4"), "SnippetEnd": np.array([12, 22], "<u4")}
    rec["RD"]["Samples"] = np.arange(1, 7, dtype="<u2")            # 12 bytes, less than the 48-byte OD
    od = codec.new()
    od.update(Frequency=396000.0, Latitude=0.88, Longitude=-0.07, Heading=1.5)
    od["Beams"] = {"AlongTrackDistance": np.ones(2, "<f4"), "AcrossTrackDistance": np.ones(2, "<f4"),
                   "CenterSampleNumber": np.array([11, 21], "<u4")}
    good = encode_record(rec)
    od_bytes = codec.encode(od)
    body = bytearray(good[:-4]) + od_bytes + bytes(12)              # OD, then 12 bytes of padding
    struct.pack_into("<I", body, 8, len(body) + 4)
    struct.pack_into("<I", body, 12, len(good) - 4 + 12)            # declared OD 12 bytes too late
    struct.pack_into("<I", body, 16, 7028)
    data = bytes(body) + struct.pack("<I", checksum(body))
    out = decode_record(bytearray(data))
    assert encode_record(out) == data
    got = recover_misplaced_optional_data(out)
    assert got["Heading"] == pytest.approx(1.5) and list(got["Beams"]["CenterSampleNumber"]) == [11, 21]


def test_real_data_verified_types_have_codecs():
    from PYS7K.verification import REAL_DATA_VERIFIED
    assert [rt for rt in REAL_DATA_VERIFIED if rt not in RECORD_CODECS] == []


def test_7041_dfd22_header():
    """DFD 2.20-2.31 define a 44-byte 7041 header (separate down-sampling,
    filtering and encoding bytes, u32 Flags); 7kCenter 5.2 writes it."""
    rec = new(7041)
    rec["RTH"] = RECORD_CODECS[7041].rth_dfd22.new()
    rec["RTH"].update(N=3, DownsamplingMethod=1, Flags=0, SampleRate=999.5)
    rec["RD"] = {"Beams": {"Beam": np.arange(3, dtype="<u2"), "Samples": np.full(3, 4, "<u4")},
                 "Samples": np.arange(300, 312, dtype="<u2"), "SampleWidth": 2}
    data, out = round_trip(rec)
    assert len(data) == 64 + 44 + 3 * (6 + 8) + 4
    assert out["RTH"]["DownsamplingMethod"] == 1 and out["RTH"]["SampleRate"] == 999.5
    assert out["RD"]["SampleWidth"] == 2 and out["RD"]["Samples"][-1] == 311


def test_7041_dfd22_encodings():
    """DFD 2.2x Flags bits 0-3 give the sample width (8-bit magnitude here)
    and bit 4 selects f32 beam angles."""
    rec = new(7041)
    rec["RTH"] = RECORD_CODECS[7041].rth_dfd22.new()
    rec["RTH"].update(N=2, Flags=0x11, SampleRate=1000.0)
    rec["RD"] = {"Beams": {"Beam": np.array([-0.5, 0.5], "<f4"), "Samples": np.full(2, 3, "<u4")},
                 "Samples": np.arange(6, dtype="u1"), "SampleWidth": 1}
    data, out = round_trip(rec)
    assert out["RD"]["SampleWidth"] == 1 and out["RD"]["Beams"]["Beam"].dtype == np.float32


def test_7048_beam_major_with_padding():
    """2010 7125 files store 7048 beam by beam and end it with N x 14 zero
    bytes; the decoder recognizes that and keeps the stored order."""
    rec = new(7048)
    rec["RTH"].update(N=2, S=3)
    beams = np.array([[1, 2, 3], [4, 5, 6]], "<f4")         # beam by beam
    rec["RD"] = {"Intensity": beams.T, "BeamMajor": True}
    rec["Unparsed"]["Body"] = bytes(2 * 14)
    data, out = (encode_record(rec), None)
    out = decode_record(bytearray(data))
    assert out["RD"]["BeamMajor"] and out["RD"]["Intensity"].shape == (3, 2)
    assert list(out["RD"]["Intensity"][:, 1]) == [4, 5, 6]
    assert encode_record(out) == data
