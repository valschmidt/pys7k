"""
Codec base classes and the record template.

A decoded record is one dictionary, built from ``new_record()``::

    {
     "RecordType": 7027,
     "Frame": {...every DRF field, including reserved ones...},
     "RTH": {...record type header fields...},
     "RD": {...record data: scalars, dictionaries of numpy arrays, lists...},
     "OptionalData": None, or a dictionary (Appendix A layouts),
     "Unparsed": {"Frame": b"", "Body": b"", "OptionalData": b""},
     "Checksum": 1234567,
     "ChecksumValid": True,   # None when Flags bit 0 says no checksum
    }

Values are exactly what the file stores. No units are converted, no bit
fields are expanded, and no fields from other records are merged in. The
``derived`` module computes converted values separately.

``Unparsed`` holds bytes the codec did not account for: frame bytes between
the 64-byte DRF and the RTH, body bytes past the known RTH/RD layout (fields
added by a newer DFD revision, or a record type with no codec), and optional
data past its known layout. The encoder writes them back unchanged, so every
record round-trips byte for byte whether or not it is fully understood.

Each codec decodes the record body (RTH and RD) from a buffer and encodes it
back from the same dictionary shape that ``new()`` returns.
"""

import numpy as np

from ..fields import Section, Table
from ..frame import DRF, FLAG_CHECKSUM_VALID, SYNC_PATTERN

__all__ = [
    "Codec", "HeaderCodec", "HeaderTableCodec", "ColumnsCodec",
    "Columns", "new_record", "new_frame", "PING_PREFIX",
]

# Most ping records begin with this prefix. Records whose RTH starts with
# it are flagged with ``ping = True`` so the full index can gather their
# ping numbers.
PING_PREFIX = [
    ("SonarId", "u64"),
    ("PingNumber", "u32"),
    ("MultiPingSequence", "u16"),
]


def new_frame(record_type=0):
    """A DRF template for a new record of ``record_type``."""
    frame = DRF.new()
    frame.update({
        "ProtocolVersion": 5,
        "Offset": 60,
        "SyncPattern": SYNC_PATTERN,
        "RecordVersion": 1,
        "RecordTypeIdentifier": record_type,
        "Flags": FLAG_CHECKSUM_VALID,
    })
    return frame


def new_record(record_type, codec=None):
    """
    Return a template record dictionary for ``record_type``.

    :param codec: the record type's codec, if it has one. Its ``new_rth()``
        and ``new_rd()`` templates fill the "RTH" and "RD" entries.
    """
    return {
        "RecordType": record_type,
        "Frame": new_frame(record_type),
        "RTH": codec.new_rth() if codec is not None else {},
        "RD": codec.new_rd() if codec is not None else {},
        "OptionalData": None,
        "Unparsed": {"Frame": b"", "Body": b"", "OptionalData": b""},
        "Checksum": 0,
        "ChecksumValid": None,
    }


class Codec:
    """
    Base codec. Subclasses set ``record_type`` and ``name`` and implement
    ``new_rth``, ``new_rd``, ``decode`` and ``encode``.
    """

    record_type = None
    name = ""
    ping = False          # RTH begins with SonarId, PingNumber, MultiPingSequence
    phase = 1

    def new_rth(self):
        return {}

    def new_rd(self):
        return {}

    def new(self):
        """A full record template (see ``new_record``)."""
        return new_record(self.record_type, self)

    def decode(self, body):
        """
        Decode the record body.

        :param body: a buffer holding the RTH and RD (not the frame, optional
            data or checksum).
        :return: ``(rth, rd, nbytes)``. Bytes past ``nbytes`` are kept as
            ``Unparsed["Body"]``.
        """
        raise NotImplementedError

    def encode(self, record):
        """Encode ``record["RTH"]`` and ``record["RD"]`` to bytes."""
        raise NotImplementedError


class HeaderCodec(Codec):
    """
    A record with only a record type header. The header is decoded with
    ``Section.unpack_prefix``, so shorter variants written by older software
    decode their leading fields and round-trip at their original length.
    """

    def __init__(self, record_type, name, rth_fields, ping=False, phase=1):
        self.record_type = record_type
        self.name = name
        self.rth = Section(rth_fields)
        self.ping = ping
        self.phase = phase

    def new_rth(self):
        return self.rth.new()

    def decode(self, body):
        rth, used = self.rth.unpack_prefix(body, 0, len(body))
        return rth, {}, used

    def field_boundaries(self):
        """Body lengths that end on a field boundary (0 included)."""
        return [0] + list(self.rth._ends)

    def encode(self, record):
        return self.rth.pack(record["RTH"])


class HeaderTableCodec(Codec):
    """
    A record type header followed by a structure repeated N times.

    :param count_field: the RTH field holding N.
    :param table_key: the RD key the repeated structure is stored under.
    :param itemsize_field: the RTH field holding the size of each item in
        bytes, for records that declare it (7027, 7017, ...). Without one the
        full DFD item size is used.
    """

    def __init__(self, record_type, name, rth_fields, count_field, table_fields,
                 table_key, itemsize_field=None, ping=False, phase=1):
        self.record_type = record_type
        self.name = name
        self.rth = Section(rth_fields)
        self.table = Table(table_fields)
        self.count_field = count_field
        self.table_key = table_key
        self.itemsize_field = itemsize_field
        self.ping = ping
        self.phase = phase

    def new_rth(self):
        rth = self.rth.new()
        if self.itemsize_field:
            rth[self.itemsize_field] = self.table.itemsize
        return rth

    def new_rd(self):
        return {self.table_key: self.table.new()}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        count = rth[self.count_field]
        itemsize = rth[self.itemsize_field] if self.itemsize_field else None
        isz = itemsize if itemsize is not None else self.table.itemsize
        if isz <= 0:
            return rth, {self.table_key: self.table.new()}, pos
        count = min(count, (len(body) - pos) // isz)
        cols, used = self.table.decode(body, pos, count, itemsize)
        return rth, {self.table_key: cols}, pos + used

    def encode(self, record):
        return self.rth.pack(record["RTH"]) + self.table.encode(record["RD"][self.table_key])


class Columns:
    """
    A set of arrays stored one after another (all N values of the first
    field, then all N of the second, ...), as in 7004 and 7006. Trailing
    arrays added in later DFD revisions are optional: only the arrays that
    fit in the record are decoded, and only the arrays present are encoded.
    """

    def __init__(self, fields):
        self.fields = list(fields)
        self.names = [n for n, _ in self.fields]
        self.dtypes = [np.dtype(Table([(n, c)]).formats[0]) for n, c in self.fields]

    def new(self, count=0):
        return {n: np.zeros(count, dtype=dt) for n, dt in zip(self.names, self.dtypes)}

    def decode(self, buf, offset, count, available):
        out = {}
        pos = offset
        for name, dt in zip(self.names, self.dtypes):
            nbytes = count * dt.itemsize
            if pos - offset + nbytes > available:
                break
            out[name] = np.frombuffer(buf, dtype=dt, count=count, offset=pos)
            pos += nbytes
        return out, pos - offset

    def encode(self, columns):
        parts = []
        for name, dt in zip(self.names, self.dtypes):
            if name not in columns or columns[name] is None:
                break
            parts.append(np.ascontiguousarray(columns[name], dtype=dt).tobytes())
        return b"".join(parts)


class ColumnsCodec(Codec):
    """A record type header followed by per-beam arrays stored column-wise."""

    def __init__(self, record_type, name, rth_fields, count_field, column_fields,
                 table_key="Beams", ping=False, phase=1):
        self.record_type = record_type
        self.name = name
        self.rth = Section(rth_fields)
        self.columns = Columns(column_fields)
        self.count_field = count_field
        self.table_key = table_key
        self.ping = ping
        self.phase = phase

    def new_rth(self):
        return self.rth.new()

    def new_rd(self):
        return {self.table_key: self.columns.new()}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        cols, used = self.columns.decode(body, pos, rth[self.count_field], len(body) - pos)
        return rth, {self.table_key: cols}, pos + used

    def encode(self, record):
        return self.rth.pack(record["RTH"]) + self.columns.encode(record["RD"][self.table_key])
