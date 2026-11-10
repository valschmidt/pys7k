"""Synthetic encode/decode tests for every record codec."""

import numpy as np
import pytest

from helpers import assert_same
from PYS7K import RECORD_CODECS, decode_record, encode_record, new
from PYS7K.frame import FLAG_CHECKSUM_VALID, checksum


def round_trip(rec):
    """Encode, decode, re-encode. Returns (bytes, decoded)."""
    data = encode_record(rec)
    out = decode_record(bytearray(data))
    assert encode_record(out) == data
    return data, out


@pytest.mark.parametrize("rt", sorted(RECORD_CODECS))
def test_template_round_trips(rt):
    rec = new(rt)
    data, out = round_trip(rec)
    assert out["RecordType"] == rt
    assert out["Frame"]["Size"] == len(data)
    assert not out["Unparsed"]["Body"], "template left unparsed bytes"
    assert "Error" not in out


@pytest.mark.parametrize("rt", sorted(RECORD_CODECS))
def test_template_fields_survive(rt):
    rec = new(rt)
    data, out = round_trip(rec)
    assert_same(rec["RTH"], out["RTH"])


class TestFrameHandling:
    def test_checksum_auto_recompute_when_flagged(self):
        rec = new(1013)
        rec["RTH"]["Heading"] = 1.25
        data = encode_record(rec)
        assert int.from_bytes(data[-4:], "little") == checksum(data[:-4])

    def test_checksum_kept_when_not_flagged(self):
        rec = new(1013)
        rec["Frame"]["Flags"] = 0
        rec["Checksum"] = 12345
        data = encode_record(rec)
        assert int.from_bytes(data[-4:], "little") == 12345
        assert decode_record(data)["Checksum"] == 12345

    def test_checksum_modes(self):
        rec = new(1013)
        rec["Checksum"] = 7
        assert int.from_bytes(encode_record(rec, "keep")[-4:], "little") == 7
        rec["Frame"]["Flags"] = 0
        data = encode_record(rec, "recompute")
        assert int.from_bytes(data[-4:], "little") == checksum(data[:-4])
        with pytest.raises(ValueError):
            encode_record(rec, "bogus")

    def test_bad_stored_checksum_round_trips(self):
        # Seen in 2006 7125 files whose timestamps were edited after
        # recording without recomputing the checksum.
        data = bytearray(encode_record(new(1013)))
        data[-4:] = (int.from_bytes(data[-4:], "little") + 1).to_bytes(4, "little")
        rec = decode_record(data)
        assert rec["ChecksumValid"] is False
        assert encode_record(rec) == bytes(data)
        assert decode_record(encode_record(rec, "recompute"))["ChecksumValid"] is True
        assert decode_record(encode_record(new(1013)))["ChecksumValid"] is True

    def test_unknown_record_type_passes_through(self):
        rec = new(9999)
        rec["Unparsed"]["Body"] = b"hello world"
        data, out = round_trip(rec)
        assert out["Unparsed"]["Body"] == b"hello world"
        assert out["RTH"] == {} and out["RD"] == {}

    def test_extra_frame_bytes_preserved(self):
        rec = new(1013)
        rec["Unparsed"]["Frame"] = b"\x01\x02\x03\x04"
        data, out = round_trip(rec)
        assert out["Frame"]["Offset"] == 64
        assert out["Unparsed"]["Frame"] == b"\x01\x02\x03\x04"

    def test_trailing_body_bytes_preserved(self):
        rec = new(7000)
        rec["Unparsed"]["Body"] = b"\xaa" * 6
        data, out = round_trip(rec)
        assert out["Unparsed"]["Body"] == b"\xaa" * 6

    def test_optional_data_unknown_layout_preserved(self):
        rec = new(1012)
        rec["OptionalData"] = {}
        rec["Unparsed"]["OptionalData"] = b"xyz"
        data, out = round_trip(rec)
        assert out["Frame"]["OptionalDataOffset"] == 64 + 12
        assert out["OptionalData"] == {}
        assert out["Unparsed"]["OptionalData"] == b"xyz"

    def test_inconsistent_offsets_kept_raw(self):
        data = bytearray(encode_record(new(1013)))
        data[12:16] = (1000).to_bytes(4, "little")     # OptionalDataOffset past the end
        out = decode_record(data)
        assert "Raw" in out and "Error" in out
        assert encode_record(out) == bytes(data)

    def test_truncated_body_decode_error_is_kept(self):
        rec = new(7027)
        rec["RTH"]["N"] = 5
        data = bytearray(encode_record(rec, "keep"))
        out = decode_record(data)
        assert out["RD"]["Detections"]["BeamDescriptor"].size == 0
        assert encode_record(out, "keep") == bytes(data)


class TestSensorRecords:
    def test_1003_with_and_without_satellites(self):
        rec = new(1003)
        rec["RTH"].update(LatitudeOrNorthing=0.75, LongitudeOrEasting=-1.2, NumberOfSatellites=12)
        data, out = round_trip(rec)
        assert len(data) == 64 + 37 + 4
        assert out["RTH"]["NumberOfSatellites"] == 12
        rec["RTH"]["NumberOfSatellites"] = None
        data, out = round_trip(rec)
        assert len(data) == 64 + 36 + 4
        assert out["RTH"]["NumberOfSatellites"] is None
        assert out["RTH"]["LatitudeOrNorthing"] == 0.75

    def test_1009_profile(self):
        rec = new(1009)
        rec["RTH"]["N"] = 3
        rec["RD"]["Samples"] = {"Depth": np.array([0, 10, 20], "<f4"),
                                "SoundVelocity": np.array([1500, 1490, 1480], "<f4")}
        data, out = round_trip(rec)
        assert np.array_equal(out["RD"]["Samples"]["SoundVelocity"], [1500, 1490, 1480])

    def test_1016_attitude(self):
        rec = new(1016)
        rec["RTH"]["N"] = 2
        rec["RD"]["Samples"] = {"TimeDifference": np.array([0, 10], "<u2"), "Roll": np.array([0.1, 0.2], "<f4"),
                                "Pitch": np.zeros(2, "<f4"), "Heave": np.zeros(2, "<f4"),
                                "Heading": np.array([1, 2], "<f4")}
        data, out = round_trip(rec)
        assert len(data) == 64 + 1 + 2 * 18 + 4
        assert np.array_equal(out["RD"]["Samples"]["TimeDifference"], [0, 10])

    def test_1004_masked_fields(self):
        rec = new(1004)
        rec["RTH"].update(FieldMask=0b1001, N=2)
        rec["RD"]["Samples"] = {"Pitch": np.array([1, 2], "<f4"), "Heave": np.array([3, 4], "<f4")}
        data, out = round_trip(rec)
        assert set(out["RD"]["Samples"]) == {"Pitch", "Heave"}

    def test_1007_readings(self):
        rec = new(1007)
        rec["RTH"].update(Flags=3, N=1)
        rec["RD"]["Readings"] = {k: np.array([i], "<f4") for i, k in enumerate(
            ["SpeedX", "SpeedY", "SpeedZ", "AccelerationX", "AccelerationY", "AccelerationZ"])}
        data, out = round_trip(rec)
        assert out["RD"]["Readings"]["AccelerationZ"][0] == 5


class TestSonarRecords:
    @pytest.mark.parametrize("itemsize,fields", [
        (22, ["BeamDescriptor", "DetectionPoint", "RxAngle", "Flags", "Quality", "Uncertainty"]),
        (26, ["BeamDescriptor", "DetectionPoint", "RxAngle", "Flags", "Quality", "Uncertainty", "Intensity"]),
        (34, ["BeamDescriptor", "DetectionPoint", "RxAngle", "Flags", "Quality", "Uncertainty", "Intensity",
              "MinLimit", "MaxLimit"]),
        (38, ["BeamDescriptor", "DetectionPoint", "RxAngle", "Flags", "Quality", "Uncertainty", "Intensity",
              "MinLimit", "MaxLimit", "Unparsed"]),
    ])
    def test_7027_detection_sizes(self, itemsize, fields):
        rec = new(7027)
        n = 4
        cols = {f: np.arange(n) for f in fields if f != "Unparsed"}
        if "Unparsed" in fields:
            cols["Unparsed"] = np.full((n, itemsize - 34), 7, np.uint8)
        rec["RTH"].update(N=n, DataFieldSize=itemsize, SamplingRate=34482.76, PingNumber=99)
        rec["RD"]["Detections"] = cols
        data, out = round_trip(rec)
        assert len(data) == 64 + 99 + n * itemsize + 4
        assert sorted(out["RD"]["Detections"]) == sorted(fields)
        assert np.array_equal(out["RD"]["Detections"]["BeamDescriptor"], np.arange(n))

    @pytest.mark.parametrize("with_txdelay", [False, True])
    def test_7004_four_or_five_arrays(self, with_txdelay):
        rec = new(7004)
        n = 3
        names = ["BeamVerticalDirection", "BeamHorizontalDirection", "BeamWidthY", "BeamWidthX"]
        if with_txdelay:
            names.append("TxDelay")
        rec["RTH"]["N"] = n
        rec["RD"]["Beams"] = {k: np.linspace(0, 1, n).astype("<f4") for k in names}
        data, out = round_trip(rec)
        assert sorted(out["RD"]["Beams"]) == sorted(names)
        assert len(data) == 64 + 12 + 4 * n * len(names) + 4

    def test_7006_columns(self):
        rec = new(7006)
        rec["RTH"]["N"] = 2
        rec["RD"]["Beams"] = {"Range": np.array([0.1, 0.2], "<f4"), "Quality": np.array([3, 15], "u1"),
                              "Intensity": np.ones(2, "<f4"), "MinFilterInfo": np.zeros(2, "<f4"),
                              "MaxFilterInfo": np.ones(2, "<f4")}
        data, out = round_trip(rec)
        assert len(data) == 64 + 24 + 2 * 17 + 4
        assert np.array_equal(out["RD"]["Beams"]["Quality"], [3, 15])

    @pytest.mark.parametrize("flags,dtype", [(0, "<u2"), (1, "<u4")])
    def test_7028_snippet_widths(self, flags, dtype):
        rec = new(7028)
        rec["RTH"].update(N=2, Flags=flags)
        rec["RD"]["Beams"] = {"BeamDescriptor": np.array([5, 6], "<u2"), "SnippetStart": np.array([10, 20], "<u4"),
                              "DetectionSample": np.array([12, 21], "<u4"), "SnippetEnd": np.array([14, 22], "<u4")}
        rec["RD"]["Samples"] = np.arange(8, dtype=dtype)
        data, out = round_trip(rec)
        assert out["RD"]["Samples"].dtype == np.dtype(dtype)
        assert np.array_equal(out["RD"]["Samples"], np.arange(8))

    @pytest.mark.parametrize("sample_type,names", [
        (0x22, ["Magnitude", "Phase"]),
        (0x02, ["Magnitude"]),
        (0x100, ["I", "Q"]),
        (0x33, ["Magnitude", "Phase"]),
    ])
    def test_7008_sample_types(self, sample_type, names):
        rec = new(7008)
        rec["RTH"].update(N=2, DataSampleType=sample_type, RecordSubsetFlag=1)
        rec["RD"]["Beams"] = {"Descriptor": np.array([0, 1], "<u2"), "FirstSample": np.array([0, 5], "<u4"),
                              "LastSample": np.array([2, 6], "<u4")}
        rec["RD"]["Samples"] = {k: np.arange(5) for k in names}
        data, out = round_trip(rec)
        assert sorted(out["RD"]["Samples"]) == sorted(names)
        assert len(out["RD"]["Samples"][names[0]]) == 5

    def test_7001_devices(self):
        rec = new(7001)
        rec["RTH"]["N"] = 2
        for i in range(2):
            dev = {"Identifier": 7125 + i, "Description": b"SN123".ljust(60, b"\x00"), "AlphaDataCard": 0,
                   "SerialNumber": 123 + i, "InfoLength": 5 + i, "Info": b"<x/>" + b"!" * (1 + i)}
            rec["RD"]["Devices"].append(dev)
        data, out = round_trip(rec)
        assert [d["Info"] for d in out["RD"]["Devices"]] == [b"<x/>!", b"<x/>!!"]

    def test_7018_shape(self):
        rec = new(7018)
        rec["RTH"].update(N=3, S=4)
        rec["RD"] = {"Amplitude": np.arange(12, dtype="<u2").reshape(4, 3),
                     "Phase": -np.arange(12, dtype="<i2").reshape(4, 3)}
        data, out = round_trip(rec)
        assert out["RD"]["Amplitude"].shape == (4, 3)
        assert out["RD"]["Phase"][3, 2] == -11

    def test_7038_iq(self):
        rec = new(7038)
        rec["RTH"].update(N=4, S=3, ElementsInRecord=4, FirstSample=0, LastSample=2, SampleType=16)
        rec["RD"] = {"Elements": np.arange(4, dtype="<u2"),
                     "I": np.arange(12, dtype="<i2").reshape(3, 4), "Q": -np.arange(12, dtype="<i2").reshape(3, 4)}
        data, out = round_trip(rec)
        assert len(data) == 64 + 60 + 8 + 3 * 4 * 4 + 4
        assert out["RD"]["I"][2, 3] == 11 and out["RD"]["Q"][1, 0] == -4

    def test_7007_sidescan(self):
        rec = new(7007)
        rec["RTH"].update(S=3, W=2)
        rec["RD"] = {"Port": np.array([1, 2, 3], "<u2"), "Starboard": np.array([4, 5, 6], "<u2")}
        data, out = round_trip(rec)
        assert np.array_equal(out["RD"]["Starboard"], [4, 5, 6])

    def test_7503_short_and_full(self):
        rec = new(7503)
        rec["RTH"]["SoundVelocity"] = 1500.0
        data, out = round_trip(rec)
        full = len(data)
        names = list(rec["RTH"])
        cut = names.index("TxArrayPositionOffsetX")
        for k in names[names.index("VernierOperationMode"):]:
            rec["RTH"][k] = None
        data, out = round_trip(rec)
        assert full == 64 + 260 + 4
        spreading_end = RECORD_CODECS[7503].rth._ends[names.index("Spreading")]
        assert len(data) == 64 + spreading_end + 4
        assert out["RTH"]["SoundVelocity"] == 1500.0
        assert out["RTH"][names[cut]] is None

    def test_7000_size(self):
        data, out = round_trip(new(7000))
        assert len(data) == 64 + 156 + 4


class TestSystemRecords:
    def test_7200_with_catalog_pointer(self):
        rec = new(7200)
        rec["RTH"]["N"] = 1
        rec["RD"]["Devices"] = {"DeviceIdentifier": np.array([7125], "<u4"),
                                "SystemEnumerator": np.array([0], "<u2")}
        rec["OptionalData"].update(Size=100, Offset=5000)
        data, out = round_trip(rec)
        assert len(data) == 64 + 316 + 6 + 12 + 4
        assert out["OptionalData"] == {"Size": 100, "Offset": 5000}

    def test_7300_entries(self):
        rec = new(7300)
        rec["RTH"]["NumberOfRecords"] = 2
        cols = rec["RD"]["Entries"] = {k: np.zeros(2, v.dtype) for k, v in rec["RD"]["Entries"].items()}
        cols["Offset"][:] = [0, 402]
        cols["RecordType"][:] = [7200, 7022]
        data, out = round_trip(rec)
        assert len(data) == 64 + 14 + 2 * 48 + 4
        assert list(out["RD"]["Entries"]["RecordType"]) == [7200, 7022]

    def test_7610_optional_fields(self):
        rec = new(7610)
        rec["RTH"].update(SoundVelocity=1500.0, Temperature=None, Pressure=None)
        data, out = round_trip(rec)
        assert len(data) == 64 + 4 + 4


class TestOptionalData:
    def test_7027_pds_optional_data(self):
        rec = new(7027)
        rec["OptionalData"] = {"Frequency": 400000.0, "Latitude": 0.7, "Longitude": -1.2, "Heading": 0.1,
                               "HeightSource": 1, "Tide": 0.0, "Roll": 0.0, "Pitch": 0.0, "Heave": 0.0,
                               "VehicleDepth": 0.0,
                               "Beams": {k: np.ones(3, "<f4") for k in
                                         ["Depth", "AlongTrackDistance", "AcrossTrackDistance",
                                          "PointingAngle", "AzimuthAngle"]}}
        data, out = round_trip(rec)
        assert out["OptionalData"]["Latitude"] == 0.7
        assert len(out["OptionalData"]["Beams"]["Depth"]) == 3
        assert out["Frame"]["OptionalDataOffset"] == 64 + 99

    def test_flags_default_checksum_valid(self):
        assert new(1012)["Frame"]["Flags"] & FLAG_CHECKSUM_VALID
