# pys7k

pys7k is a pure-Python library and command line utility for reading and
writing Teledyne RESON s7k sonar files. It follows the 7k Data Format
Definition (DFD) version 3.10. It needs only numpy and pandas.

pys7k has two goals.

* **Forensic accuracy.** Records are reported exactly as they are stored:
  DFD field names, native units, raw bit fields, reserved fields and all.
  Every record can be written back byte for byte, including record types
  pys7k does not understand.
* **Ease of use.** Converted values (degrees, travel times, datetimes,
  named flags, water column images) come from a separate set of batch
  functions in `PYS7K.derived`. They never change the raw records.

## Installation

```
pip install pys7k
```

## Command line

`s7k.py -f FILE -V` indexes a file and prints a table of its record types.
Add `-c` to verify every record's checksum.

```
$ s7k.py -f uam_7125_7027_7028_nav.s7k -V -c
File: uam_7125_7027_7028_nav.s7k
File size: 356131 bytes
Total records: 65
Protocol version(s): 5
Devices: 0/0 (Unspecified, 1 records), 7000/0 (7k sonar source, 4 records), 7003/0 (Teledyne PDS, 25 records), 7125/0 (SeaBat 7125, 35 records)
Time span: 2012-11-28T19:09:22.014999+00:00 to 2012-11-28T19:35:56.467998+00:00 (1594.5 s)

                                     Name  Count  Total Bytes  Min Bytes  Max Bytes  % of File Decoded  Checksum OK  Checksum Bad  No Checksum
RecordType
7028                         Snippet Data      5       113078      22598      22642      31.75     yes            5             0            0
7017                 Detection Data Setup      5        87858      17558      17592      24.67     yes            5             0            0
7027                   Raw Detection Data      5        57089      11409      11431      16.03     yes            5             0            0
...
7777                 Filler (file repair)      1          390        390        390       0.11     raw            0             0            1
```

`s7k.py -f FILE -p [RECORDTYPE]` prints records as text for debugging. The
record type may be a number or a name. `-n` limits the number of records
and `-r` limits the rows printed for each table. Lines starting with `#`
are annotations, not stored values.

```
$ s7k.py -f uam_7125_7027_7028_nav.s7k -p 1003 -n 1
=== 1003 Position  record=28  offset=143796  size=104 ===
-- Frame --
  ProtocolVersion             : 5
  Offset                      : 60
  ...
  DeviceIdentifier            : 7003
  Flags                       : 1
  ...
  # time: 2012-11-28T19:34:20.691999+00:00
  # device: Teledyne PDS
-- RTH --
  DatumIdentifier               : 0
  Latency                       : 0.0
  LatitudeOrNorthing            : 0.7518556113456243
  LongitudeOrEasting            : -1.234361627814439
  HeightRelativeToDatumOrHeight : 0.0
  PositionTypeFlag              : 0
  UTMZone                       : 0
  QualityFlag                   : 0
  PositioningMethod             : 0
  NumberOfSatellites            : None
-- Checksum: 4663  # verifies
```

`NumberOfSatellites` is `None` because this 7125 file writes the older
36-byte 1003 record, which ends before that field.

Other options:

* `-i` prints the full record index as CSV.
* `--catalog` compares the file's 7300 catalog with the index.
* Record types with no decoder are printed as ASCII text, with each
  non-printable byte shown as `.`.

## Library

### Opening and indexing

```python
from PYS7K import s7k

S = s7k("uam_7125_7027_7028_nav.s7k")
S.index_file()
print(S.Index.head(3))
```

```
   Offset  Size  RecordType  DeviceId  SystemEnumerator          Time  ProtocolVersion  RecordVersion  Flags  OptionalDataOffset  OptionalDataIdentifier
0       0   402        7200      7000                 0  1.354131e+09                5              1      0                 386                    7300
1     402   100        7022      7000                 0  1.354130e+09                5              1      1                   0                       0
2     502  3944        7001      7000                 0  1.354130e+09                5              1      1                   0                       0
```

The file is memory-mapped. Indexing walks the record frames with one
precompiled `struct` unpack per record, then gathers every frame field at
once with numpy. A 1 GB file indexes in about 0.05 s.

`index_file(full=True)` adds `PingNumber` and `MultiPingSequence` columns
(-1 where a record has none). `index_file(verify_checksums=True)` adds a
`ChecksumValid` column. It is missing (NA) for records whose frame says no
checksum is present. Both together take about 0.25 s per GB.

A damaged file does not stop indexing. pys7k skips to the next sync
pattern and lists each skipped byte range in `S.Gaps`. Use
`index_file(strict=True)` to raise an error instead.

### Reading records

```python
for rec in S.iter_records(7027):
    ...
```

`iter_records` accepts record types (numbers or names), devices, and a
time range. Each record is a dictionary:

```python
rec = next(S.iter_records(7027))
{k: type(v).__name__ for k, v in rec.items()}
```

```
{'RecordType': 'int', 'Frame': 'dict', 'RTH': 'dict', 'RD': 'dict',
 'OptionalData': 'NoneType', 'Unparsed': 'dict', 'Checksum': 'int'}
```

* `rec["Frame"]` holds every field of the 64-byte Data Record Frame.
* `rec["RTH"]` holds the record type header.
* `rec["RD"]` holds the record data. Repeated structures are dictionaries
  of numpy arrays, for example `rec["RD"]["Detections"]["DetectionPoint"]`.
* `rec["OptionalData"]` holds the optional data (DFD Appendix A), or `None`.
* `rec["Unparsed"]` holds any bytes past the known layout. For record types
  with no decoder, `rec["Unparsed"]["Body"]` holds the whole body.

Field sets follow the sizes the record declares. This 2012 file writes
22-byte 7027 detections, so `Intensity`, `MinLimit` and `MaxLimit` are
absent:

```python
rec["RTH"]["DataFieldSize"], sorted(rec["RD"]["Detections"])
```

```
(22, ['BeamDescriptor', 'DetectionPoint', 'Flags', 'Quality', 'RxAngle', 'Uncertainty'])
```

Every record type documented in DFD 3.10 has a decoder. 7038 (raw IQ
element data) also has one, although DFD 3.10 lists it as reserved. Its
layout comes from the CCOM 7125 MATLAB reader and was checked against 7125
files.

### Converted values

`PYS7K.derived` computes values for many records at once, which is much
faster than converting each record as it is read.

```python
from PYS7K import derived as D

d = D.read_detections(S, degrees=True)
d["TwoWayTravelTime"].shape, d["TwoWayTravelTime"][0, :3], d["RxAngle"][0, :3]
```

```
((5, 512), array([0.02279599, 0.02266362, 0.02254943]), array([-60.30000407, -60.20000327, -60.10000247]))
```

Available functions:

| Function | Records | Returns |
|---|---|---|
| `read_attitude` | 1012, 1013, 1016 | Attitude and heading arrays by source |
| `read_navigation` | 1003, 1015 | Positions (degrees only where geographic) |
| `read_detections` | 7027 | Ping-by-detection arrays, travel times, flag fields |
| `read_bathymetry_7006` | 7006 | Ping-by-beam arrays |
| `read_beam_geometry` | 7004 | Ping-by-beam angles |
| `read_runtime` | 7503 or 7000 | A DataFrame of settings |
| `read_svp`, `read_sound_velocity` | 1009, 7610 | Profiles and surface sound velocity |
| `read_installation` | 7030 | Installation parameters |
| `read_snippets` | 7028 | Each beam's snippet |
| `read_sidescan` | 7007, 7057 | Port and starboard series |
| `read_watercolumn` | 7008, 7018, 7042, 7048 | A samples-by-beams image |
| `gather_header` | any header-only record | Every header field, read straight from the file |
| `record_times` | any | Frame times as seconds and datetimes |

`PYS7K.bitfields.decode_bits` splits bit fields into named parts, for
example `decode_bits(rec["RTH"]["ControlFlags"], CONTROL_FLAGS_7000)`.

### Writing

```python
from PYS7K import new, s7k

out = s7k("copy.s7k")
out.OpenFiletoWrite()
for rec in S.iter_records():
    out.write_record(rec)
out.closeFile(catalog=True)
```

`new(record_type)` returns a template record with every field set to zero.
Fill it in and pass it to `write_record`. The encoder recomputes the
frame's Size and offsets. The checksum is computed when Flags bit 0 is set
and written as stored otherwise, so unmodified records come back byte for
byte. `closeFile(catalog=True)` appends a 7300 file catalog and points the
7200 file header at it.

## Kluster

`PYS7K.kluster.X7kRead` can stand in for NOAA's
`HSTB.drivers.prr3.X7kRead`. It returns the same dictionary from
`sequential_read_records()` and keeps prr3's record preferences and
conversions. On the sample data its ping, attitude and navigation arrays
match prr3's to float precision. Its module docstring lists nine places
where it deliberately differs from prr3, for example converting the 7027
TxAngle to degrees like every other angle.

## Notes on the DFD

These points were checked against 7125 data from 2012 and 2016.

* The checksum flag is Flags bit 0. The DFD's checksum row says bit 1,
  which is a typo.
* The 7200, 7300, 7777 and 7038 records carry no checksum, and neither do
  7008 records written by HYPACK.
* The 7503 record has no ReceiveBeamWidth field. Its full 3.10 layout is
  260 bytes.
* 7k logger files leave the 7200 pointer to the 7300 catalog at zero.

## Tests

```
python scripts/make_test_subsets.py --defaults   # optional: cut real-data test files into data/
pytest
```

Most tests build synthetic records with the writer and always run. Tests
on real data use the small files in `data/` and are skipped when that
folder is empty.

## License

BSD 3-Clause. See LICENSE.
