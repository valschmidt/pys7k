"""Tests for the field-spec helpers and the record frame."""

import struct

import numpy as np
import pytest

from PYS7K.fields import Section, Table, parse_code
from PYS7K.frame import (DRF, DRF_DTYPE, DRF_SIZE, checksum, epoch_to_time7k,
                         time7k_to_epoch)


class TestParseCode:
    def test_scalars(self):
        assert parse_code("u32")[:2] == ("I", "scalar")
        assert parse_code("f64")[3] == 8
        assert parse_code("i8")[0] == "b"

    def test_bytes(self):
        assert parse_code("c8*64") == ("64s", "bytes", 1, 64)
        assert parse_code("u8*16") == ("16s", "bytes", 1, 16)

    def test_tuple(self):
        assert parse_code("u32*15") == ("15I", "tuple", 15, 60)

    def test_bad(self):
        with pytest.raises(ValueError):
            parse_code("x32")


class TestSection:
    S = Section([("A", "u16"), ("B", "f32"), ("C", "c8*4"), ("D", "u32*2"), ("E", "u8")])

    def test_size_and_template(self):
        assert self.S.size == 2 + 4 + 4 + 8 + 1
        assert self.S.new() == {"A": 0, "B": 0.0, "C": b"\x00" * 4, "D": (0, 0), "E": 0}

    def test_round_trip(self):
        values = {"A": 7, "B": 1.5, "C": b"ab\x00z", "D": (1, 2), "E": 9}
        data = self.S.pack(values)
        assert self.S.unpack(data) == values

    def test_prefix_decode_and_encode(self):
        full = self.S.pack({"A": 7, "B": 1.5, "C": b"abcd", "D": (1, 2), "E": 9})
        short = full[:10]                     # A, B and C only
        d, used = self.S.unpack_prefix(short, 0, len(short))
        assert used == 10
        assert d["C"] == b"abcd" and d["D"] is None and d["E"] is None
        assert self.S.pack(d) == short

    def test_prefix_partial_field_not_decoded(self):
        d, used = self.S.unpack_prefix(b"\x01\x00\x00", 0, 3)
        assert d["A"] == 1 and d["B"] is None and used == 2

    def test_duplicate_names_rejected(self):
        with pytest.raises(ValueError):
            Section([("A", "u8"), ("A", "u8")])


class TestTable:
    T = Table([("Beam", "u16"), ("Value", "f32"), ("Flags", "u32")])

    def test_round_trip(self):
        cols = {"Beam": np.arange(3, dtype="<u2"), "Value": np.array([1.0, 2.0, 3.0], "<f4"),
                "Flags": np.array([0, 1, 2], "<u4")}
        data = self.T.encode(cols)
        assert len(data) == 30
        out, used = self.T.decode(data, 0, 3)
        assert used == 30
        for k in cols:
            assert np.array_equal(out[k], cols[k])

    def test_smaller_itemsize_decodes_leading_fields(self):
        data = self.T.encode({"Beam": [1, 2], "Value": [0.5, 0.25], "Flags": [9, 9]})
        short = b"".join(data[i * 10:i * 10 + 6] for i in range(2))
        out, used = self.T.decode(short, 0, 2, itemsize=6)
        assert set(out) == {"Beam", "Value"}
        assert self.T.encode(out) == short

    def test_larger_itemsize_keeps_unparsed_bytes(self):
        data = b"".join(struct.pack("<HfI", i, i / 2, i) + bytes([i, i + 1]) for i in range(3))
        out, used = self.T.decode(data, 0, 3, itemsize=12)
        assert out["Unparsed"].shape == (3, 2)
        assert self.T.encode(out) == data


class TestFrame:
    def test_drf_layout(self):
        assert DRF.size == DRF_SIZE == DRF_DTYPE.itemsize == 64

    def test_checksum(self):
        assert checksum(b"\x01\x02\xff") == 0x102
        big = bytes([255]) * (2 ** 24 + 3)
        assert checksum(big) == (255 * len(big)) & 0xFFFFFFFF

    def test_time_round_trip(self):
        t = 1354130060.621952
        d = epoch_to_time7k(t)
        assert d["Year"] == 2012 and d["Day"] == 333 and d["Hours"] == 19 and d["Minutes"] == 14
        back = time7k_to_epoch(d["Year"], d["Day"], d["Hours"], d["Minutes"], np.float32(d["Seconds"]))
        assert abs(back - t) < 1e-5

    def test_time_vectorized_and_zero_is_nan(self):
        t = time7k_to_epoch(np.array([2012, 0]), np.array([1, 0]), np.array([0, 0]),
                            np.array([0, 0]), np.array([0.5, 0.0]))
        assert t[0] == 1325376000.5
        assert np.isnan(t[1])

    def test_leap_year_day(self):
        assert time7k_to_epoch(2016, 366, 0, 0, 0) == 1483142400.0   # 2016-12-31
