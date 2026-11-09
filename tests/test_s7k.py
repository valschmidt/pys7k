"""Tests for the s7k class: indexing, iteration, writing, reports and CLI,
using synthetic files built with the writer."""

import numpy as np
import pandas as pd
import pytest

from PYS7K import S7KError, S7KFormatError, encode_record, main, new, s7k
from PYS7K.frame import epoch_to_time7k

T0 = 1354131260.5


def make_record(rt, t, ping=None, **rth):
    rec = new(rt)
    rec["Frame"].update(epoch_to_time7k(t))
    rec["Frame"]["DeviceIdentifier"] = 7125
    if ping is not None:
        rec["RTH"]["PingNumber"] = ping
    rec["RTH"].update(rth)
    return rec


def make_7027(t, ping, n=4):
    rec = make_record(7027, t, ping, N=n, SamplingRate=1000.0, TxAngle=0.01)
    rec["RD"]["Detections"] = {
        "BeamDescriptor": np.arange(n, dtype="<u2"),
        "DetectionPoint": np.linspace(100, 200, n).astype("<f4"),
        "RxAngle": np.linspace(-1, 1, n).astype("<f4"),
        "Flags": np.array([1, 2, 3, 2][:n], "<u4"),
        "Quality": np.zeros(n, "<u4"),
        "Uncertainty": np.full(n, 0.01, "<f4"),
        "Intensity": np.full(n, 100.0, "<f4"),
        "MinLimit": np.zeros(n, "<f4"),
        "MaxLimit": np.zeros(n, "<f4"),
    }
    return rec


def synthetic_records(pings=3):
    recs = [make_record(7200, T0), make_record(7022, T0, VersionString=b"test".ljust(32, b"\0"))]
    recs[0]["Frame"]["Flags"] = 0
    for i in range(pings):
        t = T0 + i
        recs.append(make_record(1012, t, Roll=0.01 * i, Pitch=0.02, Heave=0.1))
        recs.append(make_record(1013, t, Heading=np.deg2rad(90.0)))
        recs.append(make_record(1003, t, LatitudeOrNorthing=np.deg2rad(43.0), LongitudeOrEasting=np.deg2rad(-70.0),
                                NumberOfSatellites=None))
        recs.append(make_record(7000, t, 100 + i, SoundVelocity=1500.0, SampleRate=1000.0,
                                ProjectorBeamWidthVertical=0.0174533, ReceiveBeamWidth=0.008))
        recs.append(make_7027(t, 100 + i))
        recs.append(make_record(7503, t, 100 + i, SoundVelocity=1500.0, Frequency=400000.0, BeamSpacingMode=1))
    return recs


@pytest.fixture
def synthetic_file(tmp_path):
    path = tmp_path / "synthetic.s7k"
    w = s7k(str(path))
    w.OpenFiletoWrite()
    for rec in synthetic_records():
        w.write_record(rec)
    w.closeFile(catalog=True)
    return path


class TestWriteAndIndex:
    def test_index_covers_file(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file()
            assert int(idx.Size.sum()) == S.file_size
            assert S.Gaps == []
            assert idx.RecordType.iloc[0] == 7200 and idx.RecordType.iloc[-1] == 7300
            assert np.isclose(idx.Time.iloc[2], T0, atol=1e-5)

    def test_full_index_ping_numbers(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file(full=True)
            pings = idx[idx.RecordType == 7027].PingNumber.tolist()
            assert pings == [100, 101, 102]
            assert (idx[idx.RecordType == 1012].PingNumber == -1).all()
            assert (idx[idx.RecordType == 7503].PingNumber.tolist()) == [100, 101, 102]
            assert (idx[idx.RecordType == 7503].MultiPingSequence == -1).all()

    def test_checksums(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file(verify_checksums=True)
            cv = idx.ChecksumValid
            assert cv[idx.RecordType == 7027].all()
            assert cv[idx.RecordType == 7200].isna().all()

    def test_bad_checksum_detected(self, synthetic_file):
        data = bytearray(synthetic_file.read_bytes())
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file()
            off = int(idx[idx.RecordType == 7027].Offset.iloc[0])
        data[off + 100] ^= 0xFF
        synthetic_file.write_bytes(bytes(data))
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file(verify_checksums=True)
            assert (idx.ChecksumValid == False).sum() == 1  # noqa: E712

    def test_catalog_written_and_checked(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            result = S.check_catalog()
            assert result["catalog_records"] == result["indexed_records"] == len(S.Index) - 1
            assert len(result["mismatches"]) == 0 and len(result["missing"]) == 0
            ptr = result["file_header_pointer"]
            last = S.Index.iloc[-1]
            assert ptr == {"Size": int(last.Size), "Offset": int(last.Offset)}

    def test_round_trip_whole_file(self, synthetic_file, tmp_path):
        out = tmp_path / "copy.s7k"
        with s7k(str(synthetic_file)) as S:
            w = s7k(str(out))
            w.OpenFiletoWrite()
            for rec in S.iter_records():
                w.write_record(rec)
            w.closeFile()
        assert out.read_bytes() == synthetic_file.read_bytes()

    def test_write_requires_open(self, tmp_path):
        with pytest.raises(S7KError):
            s7k(str(tmp_path / "x.s7k")).write_record(new(1013))

    def test_empty_file(self, tmp_path):
        path = tmp_path / "empty.s7k"
        path.write_bytes(b"")
        with s7k(str(path)) as S:
            assert len(S.index_file()) == 0


class TestIterate:
    def test_filter_by_type_and_name(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            assert [r["RecordType"] for r in S.iter_records(7027)] == [7027] * 3
            assert len(list(S.iter_records("Raw Detection Data"))) == 3
            assert len(list(S.iter_records([1012, 1013]))) == 6

    def test_filter_by_time_and_device(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            recs = list(S.iter_records(7027, start_time=T0 + 0.5))
            assert [r["RTH"]["PingNumber"] for r in recs] == [101, 102]
            assert len(list(S.iter_records(devices=9999))) == 0

    def test_records_survive_close_and_are_writable(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            rec = next(S.iter_records(7027))
        dp = rec["RD"]["Detections"]["DetectionPoint"]
        dp[0] = 1.0
        assert dp[0] == 1.0

    def test_len_and_iter(self, synthetic_file):
        S = s7k(str(synthetic_file))
        assert len(S) == len(list(S))
        S.closeFile()

    def test_unknown_name(self, synthetic_file):
        with s7k(str(synthetic_file)) as S:
            with pytest.raises(ValueError):
                list(S.iter_records("no such record"))


class TestCorruption:
    def test_resync_after_garbage(self, synthetic_file):
        data = synthetic_file.read_bytes()
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file()
            cut = int(idx.Offset.iloc[5])
        junk = b"\x00\x11" * 50
        synthetic_file.write_bytes(data[:cut] + junk + data[cut:])
        with s7k(str(synthetic_file)) as S:
            idx2 = S.index_file()
            assert len(idx2) == len(idx)
            assert S.Gaps == [(cut, len(junk))]
            with pytest.raises(S7KFormatError):
                S.index_file(strict=True)

    def test_truncated_file(self, synthetic_file):
        data = synthetic_file.read_bytes()
        synthetic_file.write_bytes(data[:-10])
        with s7k(str(synthetic_file)) as S:
            idx = S.index_file()
            assert idx.RecordType.iloc[-1] != 7300
            assert S.Gaps and S.Gaps[-1][0] + S.Gaps[-1][1] == len(data) - 10


class TestReports:
    def test_report_record_types(self, synthetic_file, capsys):
        with s7k(str(synthetic_file)) as S:
            summary = S.report_record_types(verify_checksums=True)
        out = capsys.readouterr().out
        assert "Raw Detection Data" in out and "Total records: 21" in out
        assert summary.loc[7027, "Count"] == 3
        assert summary.loc[7027, "Checksum OK"] == 3
        assert summary.loc[7200, "No Checksum"] == 1

    def test_print_records(self, synthetic_file, capsys):
        with s7k(str(synthetic_file)) as S:
            S.print_records(7027, max_records=1, max_rows=2)
        out = capsys.readouterr().out
        assert "=== 7027 Raw Detection Data" in out
        assert "DataFieldSize" in out and "-- RD.Detections (4 rows) --" in out
        assert "2 more rows" in out and "# verifies" in out
        assert out.count("=== ") == 1

    def test_print_unknown_record_as_ascii(self, tmp_path, capsys):
        path = tmp_path / "u.s7k"
        rec = new(4242)
        rec["Unparsed"]["Body"] = b"hello\x00world"
        path.write_bytes(encode_record(rec))
        with s7k(str(path)) as S:
            S.print_records()
        out = capsys.readouterr().out
        assert "hello.world" in out and "no decoder" in out

    def test_print_index_csv(self, synthetic_file, capsys):
        with s7k(str(synthetic_file)) as S:
            S.index_file(full=True)
            S.print_index()
        lines = capsys.readouterr().out.splitlines()
        assert lines[0].startswith("Record,Offset,Size,RecordType")
        assert len(lines) == 22


class TestCLI:
    def test_verify(self, synthetic_file, capsys):
        assert main(["-f", str(synthetic_file), "-V", "-c"]) == 0
        out = capsys.readouterr().out
        assert "Checksum OK" in out and "Remote Control Sonar Settings" in out

    def test_default_is_verify(self, synthetic_file, capsys):
        assert main(["-f", str(synthetic_file)]) == 0
        assert "Total records" in capsys.readouterr().out

    def test_print_type(self, synthetic_file, capsys):
        assert main(["-f", str(synthetic_file), "-p", "1013", "-n", "1"]) == 0
        out = capsys.readouterr().out
        assert "=== 1013 Heading" in out and "Heading : 1.57" in out

    def test_print_all(self, synthetic_file, capsys):
        assert main(["-f", str(synthetic_file), "-p"]) == 0
        assert capsys.readouterr().out.count("=== ") == 21

    def test_index_and_catalog(self, synthetic_file, capsys):
        assert main(["-f", str(synthetic_file), "-i", "--catalog"]) == 0
        out = capsys.readouterr().out
        assert "PingNumber" in out and "mismatched entries: 0" in out

    def test_missing_file(self, tmp_path, capsys):
        assert main(["-f", str(tmp_path / "nope.s7k")]) == 1
        assert "nope.s7k" in capsys.readouterr().err

    def test_help_lists_codecs(self, capsys):
        with pytest.raises(SystemExit):
            main(["-h"])
        assert "7027  Raw Detection Data" in capsys.readouterr().out


def test_index_dataframe_types(synthetic_file):
    with s7k(str(synthetic_file)) as S:
        idx = S.index_file(full=True)
        assert isinstance(idx, pd.DataFrame)
        assert idx.Offset.dtype == np.int64 and idx.Time.dtype == np.float64
