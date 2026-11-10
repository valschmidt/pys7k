"""
Tests on small samples of real files from many writers and systems
(``scripts/make_test_subsets.py --writers`` writes them to data/writers/; each test
is skipped when its file is missing). The samples are copied record for
record, unchanged, so writer quirks are preserved. See PYS7K/verification.py.
"""

import os
import warnings

import numpy as np
import pytest

from PYS7K import derived as D
from PYS7K import RECORD_CODECS, encode_record, s7k
from PYS7K.frame import DRF_OFFSETS, checksum
from PYS7K.optional_data import recover_misplaced_optional_data

WRITERS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "writers")
FILES = sorted(f for f in os.listdir(WRITERS_DIR) if f.endswith(".s7k")) if os.path.isdir(WRITERS_DIR) else []


def path_of(name):
    p = os.path.join(WRITERS_DIR, name)
    if not os.path.exists(p):
        pytest.skip("%s not present (run scripts/make_test_subsets.py --writers)" % name)
    return p


def open_file(name):
    S = s7k(path_of(name))
    S.index_file(verify_checksums=True)
    return S


def explained_leftovers(rec):
    """True when every unparsed byte in ``rec`` is a documented writer quirk."""
    un = {k: len(v) for k, v in rec["Unparsed"].items() if v is not None and len(v)}
    if not un or "Note" in rec:
        return True
    rt = rec["RecordType"]
    if rt not in RECORD_CODECS:                                      # 7777, 7026: kept raw
        return True
    if rt == 7200 and un == {"OptionalData": 2}:                     # RDR 1.2
        return True
    if rt == 7028 and set(un) == {"Body"}:                           # PDS
        return recover_misplaced_optional_data(rec) is not None
    if rt == 7048 and rec["RD"].get("BeamMajor"):                    # 2010 zero padding
        return True
    if rt == 7012 and set(un) == {"Body"}:                           # 2010 stale samples
        return True
    return False


@pytest.mark.skipif(not FILES, reason="no data/writers samples")
@pytest.mark.parametrize("name", FILES)
class TestEveryWriter:
    def test_records_round_trip_and_leftovers_explained(self, name):
        with s7k(path_of(name)) as S:
            S.index_file()
            assert S.Gaps == []
            for (row, buf), rec in zip(S.iter_records(raw=True), S.iter_records()):
                assert "Error" not in rec, (row, rec.get("Error"))
                assert explained_leftovers(rec), (row, rec["RecordType"],
                                                  {k: len(v) for k, v in rec["Unparsed"].items() if v})
                assert encode_record(rec) == bytes(buf), (row, rec["RecordType"])

    def test_whole_file_rewrite_is_identical(self, name, tmp_path):
        out = tmp_path / "copy.s7k"
        with s7k(path_of(name)) as S:
            w = s7k(str(out))
            w.OpenFiletoWrite()
            for rec in S.iter_records():
                w.write_record(rec)
            w.closeFile()
        with open(path_of(name), "rb") as f:
            assert out.read_bytes() == f.read()


def test_rdr12_2006_short_7200_optional_data():
    with open_file("rdr12_2006_7125.s7k") as S:
        hdr = next(S.iter_records(7200))
        assert hdr["OptionalData"]["Offset"] is None and len(hdr["Unparsed"]["OptionalData"]) == 2
        assert S.Index.ChecksumValid[S.Index.RecordType == 7008].isna().all()     # no checksum
        assert S.Index.ChecksumValid.dropna().all()


def test_rdr12_2009_short_7002_bite_and_image():
    with open_file("rdr12_2009_7002_7011_7021.s7k") as S:
        mf = next(S.iter_records(7002))
        assert mf["RTH"]["StartFrequency"] == 396000.0 and mf["RTH"]["WindowType"] is None
        for board in next(S.iter_records(7021))["RD"]["Boards"]:
            f = board["BiteFields"]
            limited = f["Minimum"] != f["Maximum"]          # firmware versions have no limits
            assert limited.any()
            assert np.all((f["Value"] >= f["Minimum"]) & (f["Value"] <= f["Maximum"]) | ~limited)
        img = next(S.iter_records(7011))
        assert img["RD"]["Image"].shape == (img["RTH"]["H"], img["RTH"]["W"])


def test_edited_2006_bad_checksums_kept():
    with open_file("edited_2006_bad_checksums.s7k") as S:
        assert (S.Index.ChecksumValid == False).any()          # noqa: E712
        for (row, buf), rec in zip(S.iter_records(raw=True), S.iter_records()):
            if rec.get("ChecksumValid") is False:
                assert encode_record(rec) == bytes(buf)
                assert encode_record(rec, "recompute") != bytes(buf)


def test_pds_7028_misplaced_optional_data_and_7009():
    with open_file("pds392_2014_7028_7009.s7k") as S:
        pings = set()
        for rec in S.iter_records(7028):
            od = recover_misplaced_optional_data(rec)
            assert od["Frequency"] == 396000.0
            assert np.array_equal(od["Beams"]["CenterSampleNumber"], rec["RD"]["Beams"]["DetectionSample"])
            pings.add(rec["RTH"]["PingNumber"])
        vd = [r for r in S.iter_records(7009) if r["RTH"]["PingNumber"] in pings]
        assert vd and all(0 < r["RTH"]["VerticalDepth"] < 100 for r in vd)


def test_hypack_size_minus_4_checksums_and_degrees():
    with open_file("hypack_2011_degrees.s7k") as S:
        assert 7200 not in set(S.Index.RecordType)
        for rec in S.iter_records(1013):
            assert rec["Checksum"] == rec["Frame"]["Size"] - 4
        pos = next(S.iter_records(1003))["RTH"]
        assert pos["PositionTypeFlag"] == 1 and pos["UTMZone"] == 19
        with pytest.warns(UserWarning, match="not radians"):
            D.read_attitude(S)


def test_fledermaus_records_without_checksum_field():
    with open_file("fledermaus_nav_no_checksums.s7k") as S:
        assert S.ChecksumTrailer is False
        depth = next(S.iter_records(1008))
        assert depth["Checksum"] is None and depth["RTH"]["Depth"] == pytest.approx(1487.4, abs=1)
        nav = D.read_navigation(S)["1003"]
        assert np.allclose(nav["Latitude"], 27.0, atol=0.1) and np.allclose(nav["Longitude"], -90.6, atol=0.1)


def test_norbit_stx_7042_lines_up_with_7027():
    with open_file("norbit_stx_2024_7042.s7k") as S:
        det = D.read_detections(S)
        pings = list(det["PingNumber"])
        checked = 0
        for rec in S.iter_records(7042):
            if rec["RTH"]["PingNumber"] not in pings:
                continue
            i = pings.index(rec["RTH"]["PingNumber"])
            wc = D.read_watercolumn(rec)
            img = np.nan_to_num(wc["Magnitude"])
            for b in (150, 200, 250, 300):
                if not det["Valid"][i][b] or det["TwoWayTravelTime"][i][b] <= 0:
                    continue
                row = det["TwoWayTravelTime"][i][b] * wc["SampleRate"] - wc["FirstSample"]
                peak = np.argmax(img[:int(row) + 50, b])
                assert abs(peak - row) < 25, (b, peak, row)
            checked += 1
        assert checked
        sv = next(S.iter_records(7610))["RTH"]
        assert sv["Temperature"] is None and 1400 < sv["SoundVelocity"] < 1550
        assert next(S.iter_records(1003))["RTH"]["NumberOfSatellites"] is not None


def test_norbit_wbms_7028_sizes():
    with open_file("norbit_wbms_2023.s7k") as S:
        for rec in S.iter_records(7028):
            b = rec["RD"]["Beams"]
            counts = b["SnippetEnd"].astype(np.int64) - b["SnippetStart"] + 1
            assert counts.sum() == len(rec["RD"]["Samples"]) and not rec["Unparsed"]["Body"]


def test_t50r_7058_and_34_byte_7027():
    with open_file("t50r_7kc63_7058.s7k") as S:
        assert all(r["RTH"]["DataFieldSize"] == 34 for r in S.iter_records(7027))
        for rec in S.iter_records(7058):
            b = rec["RD"]["Beams"]
            counts = np.clip(b["EndSample"].astype(np.int64) - b["BeginSample"] + 1, 0, None)
            assert counts.sum() == len(rec["RD"]["BackscatterStrength"]) == len(rec["RD"]["Footprints"])
            assert len(b["BeamDescriptor"]) == rec["RTH"]["N"]
        assert next(S.iter_records(7504))["RTH"]["RecordVersion"] == 2


def test_7kcenter52_7041_dfd22_header():
    """7kCenter 5.2 writes the DFD 2.20-2.31 7041 header."""
    with open_file("7kc52_2013_7041.s7k") as S:
        for rec in S.iter_records(7041):
            h = rec["RTH"]
            assert (h["DownsamplingMethod"], h["FilteringMethod"], h["Flags"]) == (1, 0, 0)
            assert (h["MagnitudeEncoding"], h["PhaseEncoding"]) == (0, 0)
            assert rec["RTH"]["SampleRate"] == pytest.approx(999.5, abs=0.1)
            assert rec["RD"]["SampleWidth"] == 2 and not rec["Unparsed"]["Body"]


def test_rdr12_2010_beam_major_7048_equals_7018():
    with open_file("rdr12_2010_7048_7057_7012.s7k") as S:
        bf = {r["RTH"]["PingNumber"]: r for r in S.iter_records(7018)}
        checked = 0
        for rec in S.iter_records(7048):
            assert rec["RD"]["BeamMajor"]
            ref = bf.get(rec["RTH"]["PingNumber"])
            if ref is None:
                continue
            a = D.read_watercolumn(ref)["Amplitude"]
            i = D.read_watercolumn(rec)["Intensity"]
            assert i.shape == a.shape
            assert np.corrcoef(np.log1p(a[:, 200]), np.log1p(i[:, 200]))[0, 1] > 0.99
            checked += 1
        assert checked
        for rec in S.iter_records(7012):
            if rec["RTH"]["PingNumber"] in bf:
                assert rec["RTH"]["N"] == bf[rec["RTH"]["PingNumber"]]["RTH"]["S"]
        side = next(S.iter_records(7057))
        assert side["RTH"]["W"] == 4 and not side["Unparsed"]["Body"]


def test_rdr20_7018_bottom_matches_7027():
    with open_file("rdr20_2013_7018.s7k") as S:
        det = D.read_detections(S)
        pings = list(det["PingNumber"])
        for rec in S.iter_records(7018):
            if rec["RTH"]["PingNumber"] in pings:
                i = pings.index(rec["RTH"]["PingNumber"])
                amp = D.read_watercolumn(rec)["Amplitude"]
                for b in (240, 256, 272):      # near nadir, where amplitude and phase detection agree
                    dp = det["DetectionPoint"][i][b]
                    assert abs(np.argmax(amp[int(dp) - 40:int(dp) + 40, b]) - 40) < 6
                return
        pytest.fail("no 7018 ping with 7027 detections")


def test_7kcenter61_7012_roll():
    with open_file("7kc61_2015_7012.s7k") as S:
        rates = {r["RTH"]["PingNumber"]: r["RTH"]["SampleRate"] for r in S.iter_records(7000)}
        for rec in S.iter_records(7012):
            assert rec["RTH"]["Flags"] == 2 and len(rec["RD"]["Roll"]) == rec["RTH"]["N"]
            if rec["RTH"]["PingNumber"] in rates:
                assert rec["RTH"]["SamplingRate"] == rates[rec["RTH"]["PingNumber"]]


def test_pds_7610_checksum_without_reserved4():
    with open_file("pds_2014_7610_reserved4.s7k") as S:
        bad = 0
        for (row, buf), rec in zip(S.iter_records(7610, raw=True), S.iter_records(7610)):
            if rec["ChecksumValid"] is False:
                o = DRF_OFFSETS["Reserved4"]
                assert (checksum(bytes(buf[:-4])) - sum(buf[o:o + 4])) & 0xFFFFFFFF == rec["Checksum"]
                bad += 1
        assert bad


def test_ixblue_seapix_converter():
    with open_file("ixblue_seapix_2021.s7k") as S:
        assert set(S.Index.DeviceId) == {103}
        assert S.Index.ChecksumValid.dropna().all()
        with warnings.catch_warnings():
            warnings.simplefilter("error")
            nav = D.read_navigation(S)["1003"]
        assert np.allclose(nav["Latitude"], 43.05, atol=0.05)
