"""
Field-specification helpers shared by every record codec.

A record section is described once, as an ordered list of
``(FieldName, type_code)`` pairs transcribed from the DFD tables. From that
one list this module builds the struct format used to decode the section,
the template dictionary returned by ``new()``, and the encoder. Decode and
encode therefore can never disagree about which fields a section has.

Type codes follow the DFD's own notation:

* ``u8 u16 u32 u64 i8 i16 i32 i64 f32 f64`` are scalars.
* ``c8*N`` and ``u8*N`` are fixed-length byte strings. They are returned as
  ``bytes`` exactly as stored, including any bytes after a NUL terminator.
* ``u16*N``, ``u32*N``, ``f32*N`` (and so on) are fixed-length arrays. They
  are returned as tuples.

Two kinds of section are supported:

* ``Section``: a fixed group of fields (a record type header, for example).
  A Section can also be decoded from a shorter buffer that holds only its
  leading fields. Fields that are not present are set to ``None``, and the
  encoder stops at the first ``None``. This is how records whose header grew
  between DFD revisions (1003, 7503, ...) round-trip byte for byte.
* ``Table``: a structure repeated once per beam, detection, or sample. It
  decodes to a dictionary of numpy arrays (one per field), using the item
  size the record itself declares. A smaller item size decodes the leading
  fields that fit. A larger one keeps the extra bytes of every item in an
  ``"Unparsed"`` column of shape ``(count, extra)``.
"""

import re
import struct

import numpy as np

__all__ = ["Section", "Table", "parse_code", "RawNaN"]

_STRUCT_CODES = {
    "u8": "B", "u16": "H", "u32": "I", "u64": "Q",
    "i8": "b", "i16": "h", "i32": "i", "i64": "q",
    "f32": "f", "f64": "d",
}
_NUMPY_CODES = {
    "u8": "<u1", "u16": "<u2", "u32": "<u4", "u64": "<u8",
    "i8": "<i1", "i16": "<i2", "i32": "<i4", "i64": "<i8",
    "f32": "<f4", "f64": "<f8",
}
_CODE_RE = re.compile(r"^\s*([uicf]\d+)\s*(?:\*\s*(\d+))?\s*$")


def parse_code(code):
    """
    Parse a DFD type code.

    :param code: a type code such as ``"u32"``, ``"c8*64"`` or ``"u32*15"``.

    :return: ``(struct_format, kind, count, nbytes)``. ``kind`` is
        ``"scalar"``, ``"bytes"`` or ``"tuple"``. ``count`` is the number of
        struct values the field consumes.
    """
    m = _CODE_RE.match(code)
    if not m:
        raise ValueError("bad type code %r" % code)
    base, n = m.group(1), m.group(2)
    if base == "c8" or (base == "u8" and n is not None):
        n = int(n or 1)
        return "%ds" % n, "bytes", 1, n
    if base not in _STRUCT_CODES:
        raise ValueError("bad type code %r" % code)
    fmt = _STRUCT_CODES[base]
    size = struct.calcsize("<" + fmt)
    if n is None:
        return fmt, "scalar", 1, size
    n = int(n)
    return "%d%s" % (n, fmt), "tuple", n, size * n


class RawNaN(float):
    """
    A NaN float field that remembers its stored bit pattern.

    Converting a stored f32/f64 to a Python float turns a signaling NaN into
    a quiet one (and an f32 NaN payload can change), so writing it back
    would change a byte. Seen in the optional data of Teledyne PDS 7028
    records (2014). ``Section`` decodes NaN fields as ``RawNaN`` and
    encodes them from ``raw``, so they round-trip byte for byte. A RawNaN
    behaves as an ordinary NaN float otherwise.
    """

    def __new__(cls, raw):
        value = struct.unpack("<f" if len(raw) == 4 else "<d", raw)[0]
        obj = float.__new__(cls, value)
        obj.raw = bytes(raw)
        return obj

    def __repr__(self):
        return "nan"


def _default(kind, code, nbytes, count):
    if kind == "bytes":
        return b"\x00" * nbytes
    if kind == "tuple":
        zero = 0.0 if code.startswith("f") else 0
        return (zero,) * count
    return 0.0 if code.startswith("f") else 0


class Section:
    """
    A fixed group of fields, decoded to and encoded from a dictionary.

    :param fields: an ordered list of ``(name, type_code)`` pairs.
    """

    def __init__(self, fields):
        self.fields = list(fields)
        self.names = [name for name, _ in self.fields]
        if len(set(self.names)) != len(self.names):
            dupes = sorted({n for n in self.names if self.names.count(n) > 1})
            raise ValueError("duplicate field names: %s" % dupes)
        parsed = [parse_code(code) for _, code in self.fields]
        self._kinds = [p[1] for p in parsed]
        self._counts = [p[2] for p in parsed]
        self._fmts = [p[0] for p in parsed]
        self._codes = [code for _, code in self.fields]
        sizes = [p[3] for p in parsed]
        self._ends = []
        total = 0
        for s in sizes:
            total += s
            self._ends.append(total)
        self.struct = struct.Struct("<" + "".join(self._fmts))
        self.size = self.struct.size
        self._simple = all(k != "tuple" for k in self._kinds)
        self._prefix_structs = {len(self.fields): self.struct}
        # Byte offset and width of each float in the flat struct values,
        # for keeping NaN bit patterns (see RawNaN).
        self._float_slots = []
        flat_i = 0
        start = 0
        for fmt, kind, count, end in zip(self._fmts, self._kinds, self._counts, self._ends):
            base = fmt[-1]
            if base in "fd":
                width = 4 if base == "f" else 8
                for j in range(count):
                    self._float_slots.append((flat_i + j, start + j * width, width))
            flat_i += count
            start = end

    def new(self):
        """Return a template dictionary with every field set to zero."""
        return {name: _default(kind, code, end - (self._ends[i - 1] if i else 0), count)
                for i, (name, kind, code, end, count) in enumerate(
                    zip(self.names, self._kinds, self._codes, self._ends, self._counts))}

    def _prefix_struct(self, nfields):
        s = self._prefix_structs.get(nfields)
        if s is None:
            s = struct.Struct("<" + "".join(self._fmts[:nfields]))
            self._prefix_structs[nfields] = s
        return s

    def _group(self, values, nfields):
        if self._simple:
            return dict(zip(self.names[:nfields], values))
        out = {}
        pos = 0
        for name, kind, count in zip(self.names[:nfields], self._kinds, self._counts):
            if kind == "tuple":
                out[name] = tuple(values[pos:pos + count])
            else:
                out[name] = values[pos]
            pos += count
        return out

    def _keep_nans(self, values, buf, offset, nvalues):
        if not self._float_slots:
            return values
        nan_slots = [(i, o, w) for i, o, w in self._float_slots if i < nvalues and values[i] != values[i]]
        if not nan_slots:
            return values
        values = list(values)
        for i, o, w in nan_slots:
            values[i] = RawNaN(bytes(buf[offset + o:offset + o + w]))
        return values

    def unpack(self, buf, offset=0):
        """Decode the full section from ``buf`` at ``offset``."""
        values = self.struct.unpack_from(buf, offset)
        return self._group(self._keep_nans(values, buf, offset, len(values)), len(self.fields))

    def unpack_prefix(self, buf, offset=0, available=None):
        """
        Decode as many leading fields as fit in ``available`` bytes.

        :return: ``(fields, nbytes)``. Fields that do not fit are ``None``.
            ``nbytes`` is the number of bytes consumed.
        """
        if available is None:
            available = len(buf) - offset
        if available >= self.size:
            return self.unpack(buf, offset), self.size
        nfields = 0
        while nfields < len(self._ends) and self._ends[nfields] <= available:
            nfields += 1
        out = {}
        if nfields:
            values = self._prefix_struct(nfields).unpack_from(buf, offset)
            out = self._group(self._keep_nans(values, buf, offset, len(values)), nfields)
        for name in self.names[nfields:]:
            out[name] = None
        return out, (self._ends[nfields - 1] if nfields else 0)

    def pack(self, values):
        """
        Encode ``values`` (a dictionary). Encoding stops at the first field
        whose value is ``None``, so a section decoded with
        ``unpack_prefix()`` is written back at its original length.
        """
        flat = []
        nfields = 0
        for name, kind in zip(self.names, self._kinds):
            v = values.get(name)
            if v is None:
                break
            if kind == "tuple":
                flat.extend(v)
            elif kind == "bytes":
                flat.append(bytes(v))
            else:
                flat.append(v)
            nfields += 1
        data = self._prefix_struct(nfields).pack(*flat)
        raw = [(o, flat[i].raw) for i, o, w in self._float_slots
               if i < len(flat) and isinstance(flat[i], RawNaN) and len(flat[i].raw) == w]
        if raw:
            data = bytearray(data)
            for o, b in raw:
                data[o:o + len(b)] = b
            data = bytes(data)
        return data

    def __len__(self):
        return self.size

    def numpy_dtype(self, nbytes=None):
        """
        A packed numpy structured dtype for the leading fields that fit in
        ``nbytes`` (all fields by default), for gathering many copies of
        this section at once. Byte strings become ``S<n>`` fields and fixed
        arrays become sub-array fields.
        """
        if nbytes is None:
            nbytes = self.size
        names, formats = [], []
        for name, code, end in zip(self.names, self._codes, self._ends):
            if end > nbytes:
                break
            m = _CODE_RE.match(code)
            base, n = m.group(1), m.group(2)
            names.append(name)
            if base == "c8" or (base == "u8" and n is not None):
                formats.append("S%d" % int(n or 1))
            elif n is not None:
                formats.append((_NUMPY_CODES[base], (int(n),)))
            else:
                formats.append(_NUMPY_CODES[base])
        return np.dtype({"names": names, "formats": formats})


class Table:
    """
    A structure repeated N times, decoded to a dictionary of numpy arrays.

    :param fields: an ordered list of ``(name, type_code)`` pairs. Scalar
        codes, and byte strings (``c8*N``, ``u8*N``, stored as numpy ``S``
        fields that keep every byte), are allowed.
    """

    def __init__(self, fields):
        self.fields = list(fields)
        self.names = [name for name, _ in self.fields]
        self.formats = []
        for _, code in self.fields:
            if code in _NUMPY_CODES:
                self.formats.append(_NUMPY_CODES[code])
                continue
            fmt, kind, _, nbytes = parse_code(code)
            if kind != "bytes":
                raise ValueError("Table fields must be scalars or byte strings, got %r" % code)
            self.formats.append("S%d" % nbytes)
        self.offsets = []
        pos = 0
        for fmt in self.formats:
            self.offsets.append(pos)
            pos += np.dtype(fmt).itemsize
        self.itemsize = pos
        self.dtype = self._dtype(self.itemsize)
        self._dtypes = {self.itemsize: self.dtype}

    def _dtype(self, itemsize):
        names, formats, offsets = [], [], []
        end = 0
        for name, fmt, off in zip(self.names, self.formats, self.offsets):
            stop = off + np.dtype(fmt).itemsize
            if stop > itemsize:
                break
            names.append(name)
            formats.append(fmt)
            offsets.append(off)
            end = stop
        if itemsize > end:
            names.append("Unparsed")
            formats.append(("u1", (itemsize - end,)))
            offsets.append(end)
        return np.dtype({"names": names, "formats": formats, "offsets": offsets, "itemsize": itemsize})

    def dtype_for(self, itemsize=None):
        """The structured dtype used for items of ``itemsize`` bytes."""
        if itemsize is None:
            return self.dtype
        dt = self._dtypes.get(itemsize)
        if dt is None:
            dt = self._dtype(itemsize)
            self._dtypes[itemsize] = dt
        return dt

    def new(self, count=0, itemsize=None):
        """Return a template: a dictionary of zero-filled arrays."""
        arr = np.zeros(count, dtype=self.dtype_for(itemsize))
        return {name: arr[name] for name in arr.dtype.names}

    def decode(self, buf, offset, count, itemsize=None):
        """
        Decode ``count`` items starting at ``offset``.

        :return: ``(columns, nbytes)``. ``columns`` maps each field present
            to a numpy array (a view into ``buf``).
        """
        dt = self.dtype_for(itemsize)
        arr = np.frombuffer(buf, dtype=dt, count=count, offset=offset)
        return {name: arr[name] for name in dt.names}, count * dt.itemsize

    def encode(self, columns):
        """
        Encode a dictionary of columns. The item size is worked out from the
        columns present: the leading fields that are present, plus the width
        of an ``"Unparsed"`` column if there is one.
        """
        nfields = 0
        for name in self.names:
            if name not in columns or columns[name] is None:
                break
            nfields += 1
        end = (self.offsets[nfields - 1] + np.dtype(self.formats[nfields - 1]).itemsize) if nfields else 0
        unparsed = columns.get("Unparsed")
        extra = 0
        if unparsed is not None:
            unparsed = np.asarray(unparsed, dtype=np.uint8)
            extra = unparsed.shape[1] if unparsed.ndim == 2 else 0
        itemsize = end + extra
        if itemsize == 0:
            return b""
        count = np.size(columns[self.names[0]]) if nfields else len(unparsed)
        arr = np.zeros(count, dtype=self.dtype_for(itemsize))
        for name in self.names[:nfields]:
            # Columns may be 2-D (e.g. samples x beams); items are stored
            # in C order.
            arr[name] = np.ravel(columns[name])
        if extra:
            arr["Unparsed"] = unparsed.reshape(count, extra)
        return arr.tobytes()

    def itemsize_of(self, columns):
        """The item size ``encode()`` would use for ``columns``."""
        nfields = 0
        for name in self.names:
            if name not in columns or columns[name] is None:
                break
            nfields += 1
        end = (self.offsets[nfields - 1] + np.dtype(self.formats[nfields - 1]).itemsize) if nfields else 0
        unparsed = columns.get("Unparsed")
        if unparsed is not None and np.ndim(unparsed) == 2:
            end += np.shape(unparsed)[1]
        return end
