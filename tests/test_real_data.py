"""
Tests against real s7k files in data/ (cut from 7125 survey files by
scripts/make_test_subsets.py). Skipped when data/ has no .s7k files.
"""

import numpy as np
import pytest

from helpers import SAMPLE_FILES, requires_sample_data, sample
from PYS7K import RECORD_CODECS, decode_record, encode_record, s7k
from PYS7K import derived as D

pytestmark = requires_sample_data
FILES = SAMPLE_FILES or [None]


@pytest.mark.parametrize("path", FILES, ids=lambda p: getattr(p, "name", "none"))
class TestEveryFile:
    def test_index_accounts_for_every_byte(self, path):
        with s7k(str(path)) as S:
            idx = S.index_file(verify_checksums=True)
            assert int(idx.Size.sum()) == S.file_size and S.Gaps == []
            assert idx.RecordType.iloc[0] == 7200
            assert idx.ChecksumValid.dropna().all()
            assert (idx.ProtocolVersion == 5).all()

    def test_every_record_round_trips(self, path):
        with s7k(str(path)) as S:
            for row, buf in S.iter_records(raw=True):
                rec = decode_record(buf)
                assert "Error" not in rec, (row, rec.get("Error"))
                if rec["RecordType"] in RECORD_CODECS:
                    assert not rec["Unparsed"]["Body"], "record %d left %d bytes unparsed" % (
                        row, len(rec["Unparsed"]["Body"]))
                assert encode_record(rec) == bytes(buf), "record %d (%d)" % (row, rec["RecordType"])

    def test_whole_file_rewrite_is_identical(self, path, tmp_path):
        out = tmp_path / "copy.s7k"
        with s7k(str(path)) as S:
            w = s7k(str(out))
            w.OpenFiletoWrite()
            for rec in S.iter_records():
                w.write_record(rec)
            w.closeFile()
        assert out.read_bytes() == path.read_bytes()

    def test_catalog_matches_index(self, path):
        with s7k(str(path)) as S:
            r = S.check_catalog()
        assert r["catalog_records"] == r["indexed_records"]
        assert len(r["mismatches"]) == 0 and len(r["missing"]) == 0

    def test_full_index_ping_numbers_match_records(self, path):
        with s7k(str(path)) as S:
            idx = S.index_file(full=True)
            for row in idx.index[idx.PingNumber >= 0]:
                rec = S.read_record(row)
                assert rec["RTH"]["PingNumber"] == idx.PingNumber[row]


class TestCrossRecordConsistency:
    """Values that must agree between records. A wrong field layout would
    still round-trip byte for byte, so these check the layouts themselves."""

    def test_7503_matches_7000(self):
        with s7k(str(sample("uam_7125_7027_7028_nav.s7k"))) as S:
            a = {r["RTH"]["PingNumber"]: r["RTH"] for r in S.iter_records(7000)}
            for b in S.iter_records(7503):
                ref = a[b["RTH"]["PingNumber"]]
                for k in ("Frequency", "SampleRate", "SoundVelocity", "Absorption", "Spreading", "ControlFlags"):
                    assert b["RTH"][k] == ref[k], k
                assert np.isclose(np.rad2deg(b["RTH"]["CoverageAngle"]), 120.0)

    def test_7027_7017_7028_agree(self):
        with s7k(str(sample("uam_7125_7027_7028_nav.s7k"))) as S:
            r27 = next(S.iter_records(7027))
            r17 = next(S.iter_records(7017))
            r28 = next(S.iter_records(7028))
            r00 = next(S.iter_records(7000))
        d27, d17 = r27["RD"]["Detections"], r17["RD"]["Detections"]
        assert r27["RTH"]["DataFieldSize"] == 22
        assert np.array_equal(d27["DetectionPoint"], d17["DetectionPoint"])
        assert r27["RTH"]["SamplingRate"] == r00["RTH"]["SampleRate"]
        b28 = r28["RD"]["Beams"]
        assert np.all(b28["SnippetStart"] <= b28["DetectionSample"])
        assert np.all(b28["DetectionSample"] <= b28["SnippetEnd"])
        assert np.allclose(b28["DetectionSample"], np.round(d27["DetectionPoint"]), atol=1)

    def test_7006_range_matches_7027(self):
        with s7k(str(sample("uam_7125_7027_7028_nav.s7k"))) as S:
            r06 = next(S.iter_records(7006))
            r27 = next(S.iter_records(7027))
            r04 = next(S.iter_records(7004))
        twtt = r27["RD"]["Detections"]["DetectionPoint"] / r27["RTH"]["SamplingRate"]
        beams = r27["RD"]["Detections"]["BeamDescriptor"]
        assert np.allclose(r06["RD"]["Beams"]["Range"][beams], twtt, atol=5e-4)
        assert np.allclose(r04["RD"]["Beams"]["BeamHorizontalDirection"][[0, -1]], np.deg2rad([-60, 60]), atol=0.01)

    def test_navigation_is_portsmouth(self):
        with s7k(str(sample("uam_7125_7027_7028_nav.s7k"))) as S:
            nav = D.read_navigation(S)["1003"]
        assert np.allclose(nav["Latitude"], 43.08, atol=0.05)
        assert np.allclose(nav["Longitude"], -70.72, atol=0.05)

    def test_7038_iq(self):
        with s7k(str(sample("uam_7125_7038_iq.s7k"))) as S:
            rec = next(S.iter_records(7038))
            ping = next(S.iter_records(7000, start_time=None))
        rth = rec["RTH"]
        assert rec["RD"]["I"].shape == (rth["LastSample"] - rth["FirstSample"] + 1, rth["ElementsInRecord"])
        assert list(rec["RD"]["Elements"][:4]) == [0, 1, 2, 3]
        assert abs(float(rec["RD"]["I"].mean())) < 50 and rec["RD"]["I"].std() > 1
        assert ping["RTH"]["PingNumber"] <= rth["PingNumber"]

    def test_7008_hsx_watercolumn(self):
        with s7k(str(sample("hsx_7125_7006_7008.s7k"))) as S:
            rec = next(S.iter_records(7008))
        assert rec["RTH"]["RecordSubsetFlag"] == 1 and rec["RTH"]["DataSampleType"] == 2
        b = rec["RD"]["Beams"]
        assert len(rec["RD"]["Samples"]["Magnitude"]) == int((b["LastSample"] - b["FirstSample"] + 1).sum())

    def test_7777_filler_kept_raw(self):
        with s7k(str(sample("uam_7125_7027_7028_nav.s7k"))) as S:
            rec = next(S.iter_records(7777))
        assert rec["RTH"] == {} and len(rec["Unparsed"]["Body"]) > 0
        assert not rec["Frame"]["Flags"] & 1


def test_kluster_adapter_on_real_data():
    from PYS7K.kluster import X7kRead
    with X7kRead(str(sample("uam_7125_7027_7028_nav.s7k"))) as K:
        r = K.sequential_read_records()
    assert r["ping"]["traveltime"].shape == (5, 512)
    assert np.allclose(r["ping"]["soundspeed"], 1481.0)
    assert len(r["attitude"]["time"]) == 23 and len(r["navigation"]["time"]) == 1
