"""Tests for the convenience layer, bit fields and the Kluster adapter."""

import numpy as np
import pytest

from PYS7K import derived as D
from PYS7K import new, s7k
from PYS7K.bitfields import (CONTROL_FLAGS_7000, DETECTION_FLAGS_7027,
                             decode_bits)
from PYS7K.kluster import (X7kRead, translate_detectioninfo, translate_mode,
                           translate_mode_two, translate_yawpitch)
from test_s7k import T0, make_record, synthetic_file, synthetic_records  # noqa: F401


class TestBitfields:
    def test_scalar(self):
        d = decode_bits((3 << 27) | (1 << 31) | 5, CONTROL_FLAGS_7000)
        assert d["TimestampState"] == 3 and d["Simulator"] == 1 and d["AutoRangeMethod"] == 5

    def test_array(self):
        flags = np.array([1, 2, (4 << 9) | 1 | (1 << 14)], np.uint32)
        d = decode_bits(flags, DETECTION_FLAGS_7027)
        assert list(d["MagnitudeDetect"]) == [1, 0, 1]
        assert list(d["PhaseDetect"]) == [0, 1, 0]
        assert list(d["DetectionPriority"]) == [0, 0, 4]
        assert list(d["NotUsedInSnippet"]) == [0, 0, 1]


class TestDerived:
    def test_attitude(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            a = D.read_attitude(S, degrees=True)
        assert set(a) == {"1012", "1013"}
        assert np.allclose(a["1012"]["Roll"], np.rad2deg([0.0, 0.01, 0.02]), atol=1e-5)
        assert np.allclose(a["1013"]["Heading"], 90.0)
        assert np.allclose(a["1012"]["Time"], T0 + np.arange(3), atol=1e-5)

    def test_attitude_in_degrees_warns(self, tmp_path):
        # A 2011 third-party writer stored 1013 heading in degrees.
        path = tmp_path / "deg.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        w.write_record(make_record(1013, T0, Heading=260.81))
        w.closeFile()
        with s7k(str(path)) as S:
            with pytest.warns(UserWarning, match="not radians"):
                a = D.read_attitude(S)
        assert np.allclose(a["1013"]["Heading"], 260.81, atol=1e-4)

    def test_attitude_1016(self, tmp_path):
        rec = make_record(1016, T0, N=2)
        rec["RD"]["Samples"] = {"TimeDifference": np.array([0, 250], "<u2"), "Roll": np.zeros(2, "<f4"),
                                "Pitch": np.zeros(2, "<f4"), "Heave": np.array([1, 2], "<f4"),
                                "Heading": np.full(2, np.pi, "<f4")}
        path = tmp_path / "a.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        w.write_record(rec)
        w.closeFile()
        with s7k(str(path)) as S:
            a = D.read_attitude(S, degrees=True)
        assert np.allclose(a["1016"]["Time"] - T0, [0, 0.25], atol=1e-5)
        assert np.allclose(a["1016"]["Heading"], 180.0)

    def test_navigation_geographic_and_grid(self, tmp_path):
        recs = [make_record(1003, T0, LatitudeOrNorthing=np.deg2rad(43.0), LongitudeOrEasting=np.deg2rad(-70.0),
                            NumberOfSatellites=None),
                make_record(1003, T0 + 1, LatitudeOrNorthing=4770000.0, LongitudeOrEasting=350000.0,
                            PositionTypeFlag=1, UTMZone=19, NumberOfSatellites=None)]
        path = tmp_path / "n.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        for r in recs:
            w.write_record(r)
        w.closeFile()
        with s7k(str(path)) as S:
            n = D.read_navigation(S)["1003"]
        assert np.isclose(n["Latitude"][0], 43.0) and np.isnan(n["Latitude"][1])
        assert n["LatitudeOrNorthing"][1] == 4770000.0 and n["UTMZone"][1] == 19

    def test_gather_header_mixed_lengths(self, tmp_path):
        a = make_record(1003, T0, NumberOfSatellites=7)
        b = make_record(1003, T0 + 1, NumberOfSatellites=None)
        path = tmp_path / "m.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        w.write_record(a)
        w.write_record(b)
        w.closeFile()
        with s7k(str(path)) as S:
            g = D.gather_header(S, 1003)
        assert "NumberOfSatellites" not in g and len(g["Time"]) == 2

    def test_detections(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            d = D.read_detections(S, degrees=True)
        assert d["DetectionPoint"].shape == (3, 4)
        assert np.allclose(d["TwoWayTravelTime"][0], np.linspace(100, 200, 4) / 1000.0)
        assert np.allclose(d["RxAngle"][0, 0], np.rad2deg(-1.0), atol=1e-4)
        assert list(d["PhaseDetect"][0]) == [0, 1, 1, 1]
        assert list(d["PingNumber"]) == [100, 101, 102]

    def test_runtime(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            r = D.read_runtime(S)
        assert list(r.PingNumber) == [100, 101, 102] and (r.SoundVelocity == 1500.0).all()

    def test_svp_and_sound_velocity(self, tmp_path):
        p = make_record(1009, T0, N=2, Latitude=np.deg2rad(10.0), Longitude=0.0)
        p["RD"]["Samples"] = {"Depth": np.array([0, 5], "<f4"), "SoundVelocity": np.array([1500, 1490], "<f4")}
        sv = make_record(7610, T0, SoundVelocity=1499.5, Temperature=None, Pressure=None)
        path = tmp_path / "p.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        w.write_record(p)
        w.write_record(sv)
        w.closeFile()
        with s7k(str(path)) as S:
            prof = D.read_svp(S)
            v = D.read_sound_velocity(S)
        assert np.isclose(prof[0]["Latitude"], 10.0) and list(prof[0]["SoundVelocity"]) == [1500, 1490]
        assert v["SoundVelocity"][0] == 1499.5 and np.isnan(v["Temperature"][0])

    def test_watercolumn_7008_both_orders(self):
        for rowcol in (0, 1):
            rec = new(7008)
            rec["RTH"].update(N=2, DataSampleType=2, RowColumnFlag=rowcol)
            rec["RD"]["Beams"] = {"Descriptor": np.array([0, 1], "<u2"), "FirstSample": np.array([10, 10], "<u4"),
                                  "LastSample": np.array([12, 12], "<u4")}
            rec["RD"]["Samples"] = {"Magnitude": np.arange(6, dtype="<u2")}
            img = D.read_watercolumn(rec)
            assert img["FirstSample"] == 10 and img["Magnitude"].shape == (3, 2)
            if rowcol == 0:
                assert list(img["Magnitude"][:, 1]) == [3, 4, 5]
            else:
                assert list(img["Magnitude"][1]) == [2, 3]

    def test_watercolumn_7042(self):
        rec = new(7042)
        rec["RTH"].update(N=2, Flags=0x6, FirstSample=5, CompressionFactor=0.0)
        rec["RD"] = {"Beams": {"Beam": np.array([0, 1], "<u2"), "Samples": np.array([3, 1], "<u4")},
                     "Samples": {"Magnitude": np.array([1, 2, 3, 4], "u1")}}
        img = D.read_watercolumn(rec)
        assert img["FirstSample"] == 5 and img["CompressionFactor"] == 40.0
        assert list(img["Magnitude"][:, 0]) == [1, 2, 3]
        assert img["Magnitude"][0, 1] == 4 and np.isnan(img["Magnitude"][1, 1])

    def test_watercolumn_7048(self):
        rec = new(7048)
        rec["RTH"].update(N=2, S=2, FirstBeam=10)
        rec["RD"]["Intensity"] = np.ones((2, 2), "<f4")
        img = D.read_watercolumn(rec)
        assert list(img["Beams"]) == [10, 11] and img["Intensity"].shape == (2, 2)

    def test_sidescan(self, tmp_path):
        rec = make_record(7007, T0, 5, S=2, W=1, BeamPosition=1.5)
        rec["RD"] = {"Port": np.array([1, 2], "u1"), "Starboard": np.array([3, 4], "u1")}
        path = tmp_path / "ss.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        w.write_record(rec)
        w.closeFile()
        with s7k(str(path)) as S:
            ss = D.read_sidescan(S)
        assert ss[0]["PingNumber"] == 5 and list(ss[0]["Starboard"]) == [3, 4]

    def test_watercolumn_rejects_other_records(self):
        with pytest.raises(ValueError):
            D.read_watercolumn(new(7027))

    def test_decode_text(self):
        assert D.decode_text(b"abc\x00junk") == "abc"


class TestKlusterTranslators:
    def test_translators(self):
        assert list(translate_yawpitch([0, 1, 16, 17])) == ["N", "P", "Y", "PY"]
        assert list(translate_mode([0, 1, 2, 3, 9])) == ["CW", "FM", "MP", "MP", ""]
        assert list(translate_mode_two([1, 2, 3, 4, 0])) == ["EqAn", "EqDs", "Flex", "Intr", ""]
        assert list(translate_detectioninfo([1, 2, 3, 0])) == [0, 1, 0, 0]


class TestKlusterAdapter:
    def test_structure(self, synthetic_file):
        with X7kRead(str(synthetic_file)) as K:
            r = K.sequential_read_records()
            assert K.fast_read_serial_number() == [0, 0, "7125"]
            assert K.has_datagram([1012, 1013], 300)
            start, end = K.fast_read_start_end_time()
        assert set(r) == {"attitude", "navigation", "profile", "ping", "runtime_params",
                          "installation_params", "format"}
        p = r["ping"]
        assert p["traveltime"].shape == (3, 4)
        assert np.allclose(p["beampointingangle"][0, 0], np.rad2deg(1.0), atol=1e-4)    # sign flipped
        assert np.allclose(p["tiltangle"], np.rad2deg(0.01), atol=1e-5)                  # degrees
        assert list(p["mode"]) == ["CW"] * 3 and list(p["modetwo"]) == ["EqAn"] * 3
        assert np.allclose(r["attitude"]["heave"], -0.1)
        assert np.allclose(r["navigation"]["latitude"], 43.0)
        assert np.isclose(start, T0, atol=1e-5) and np.isclose(end, T0 + 2, atol=1e-5)
        settings = r["runtime_params"]["runtime_settings"][0]
        assert settings["ReceiveBeamWidth"] == "0.458"        # from the 7000, not Tx/2
        assert settings["sound_velocity_meters_per_sec"] == "1500.0"
        assert r["installation_params"]["installation_settings"][0]["sonar_model_number"] == "7125"

    def test_requires_attitude(self, tmp_path):
        path = tmp_path / "k.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        w.write_record(make_record(7503, T0, 1))
        w.closeFile()
        with X7kRead(str(path)) as K:
            with pytest.raises(ValueError):
                K.sequential_read_records()

    def test_chunk_reads_following_nav_block(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file()
        end = int(idx[idx.RecordType == 7027].Offset.iloc[0]) + 1
        with X7kRead(str(synthetic_file), 0, end) as K:
            r = K.sequential_read_records()
        assert len(r["ping"]["time"]) == 1
        assert len(r["navigation"]["time"]) == 2    # the 1003 in the chunk and the next block

    def test_7030_installation(self, tmp_path):
        rec = make_record(7030, T0, TransmitArrayX=1.0, TransmitArrayY=2.0, TransmitArrayZ=3.0,
                          TransmitArrayHeading=np.deg2rad(10.0), MotionSensorTimeDelay=20)
        path = tmp_path / "i.s7k"
        w = s7k(str(path))
        w.OpenFiletoWrite()
        for r in [rec] + synthetic_records()[2:]:
            w.write_record(r)
        w.closeFile()
        with X7kRead(str(path)) as K:
            first = K.sequential_read_records(first_installation_rec=True)
        iset = first["installation_params"]["installation_settings"][0]
        assert iset["transducer_1_along_location"] == "2.000"
        assert iset["transducer_1_athwart_location"] == "1.000"
        assert iset["transducer_1_vertical_location"] == "-3.000"
        assert iset["transducer_1_heading_angle"] == "10.000"
        assert iset["motion_sensor_1_time_delay"] == "0.020"
