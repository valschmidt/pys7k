"""
Which record decoders have been checked against real data, and how.

"Verified" here means real records of that type, from the writers listed,
decode with no bytes left over (or with left-over bytes explained below),
re-encode byte for byte, and give plausible values. For many types the
values were also cross-checked against another record (positions against
known survey areas, water column against 7027 detections, and so on).
Record types not listed have only been tested with synthetic records.

Sources (November 2026):

* Local 7125 files from 2012 and 2016 (7k logger, RDR 2.0; HYPACK 7008).
* A survey of an institutional sonar data archive: over 25,000 s7k files
  (about 15 TB). Up to two files per directory (the smallest and the
  largest), about 3,800 in all, were probed: every record type in the
  first 1,500 records, plus four windows later in the file, was decoded,
  re-encoded and checked. Files showing anything unusual were copied and
  examined in full.

Writers and systems seen:

* Reson RDR 1.2 (2005-2011), RDR 2.0 (2011-2014), and 7kCenter 3.7, 3.15,
  5.2, 5.3, 5.4, 6.0, 6.1 and 6.3.
* SeaBat 7101, 7111, 7125 (HF and LF), T20-P, T50-P and T50-R.
* Teledyne PDS 2.3.3.19 (2005), 3.5.0.6 (2010), 3.6.0.22 (2011), 3.9.1.3,
  3.9.2.0 and 3.9.2.2 (2014), and 3.9.6.3 (2015).
* HYPACK (2011, 2013), with no 7200 file header.
* NORBIT WBMS (device 13003, 2023) and NORBIT STX (13016, 2024).
* iXblue SeapiX data converted by "SeapiX_to_s7k" 0.1 (device 103, 2021):
  1003, 1012, 1013, 7000, 7004, 7006 and 7200. Its 7200 text fields hold
  uninitialized memory.
* Fledermaus navigation exports (1003/1008 only, no checksum trailers).
* Protocol version 2 snapshot files from 2003. These predate DFD 3.10 and
  have other layouts, so their record bodies are kept raw.

Writer quirks found, and how pys7k handles them (each is also noted where
the code deals with it):

* Bad stored checksums (timestamps edited after recording, 2006; HYPACK
  writes Size - 4; PDS 7610 records checksummed before the frame's
  Reserved4 was filled in). Records are written back with the stored
  checksum, and ``s7k.py -p`` says which pattern matches.
* No checksum trailer at all (Fledermaus). Detected per file
  (``registry.detect_checksum_trailer``).
* PDS 7028: optional data placed before OptionalDataOffset
  (``optional_data.recover_misplaced_optional_data``), and signaling NaNs
  in it (kept bit for bit by ``fields.RawNaN``).
* RDR 1.2 7200 optional data is 6 bytes, too short for the Offset field.
* Angles in degrees in a 2011 HYPACK-style file (``derived.read_attitude``
  warns).
* Layouts from older DFDs: 7kCenter 5.2 writes the 44-byte DFD 2.20-2.31
  7041 header. Shorter records are earlier versions of the record: 7002
  in 2009 is the DFD 1.00 record, the 4-byte 7610 is the record before
  DFD 2.31, the 36-byte 1003 the record before DFD 2.41, and 22-byte and 26-byte 7027 detections
  are the DFD 2.20-2.22 and 2.31-2.41 forms. These decode as prefixes,
  with absent fields None. The 35-byte Fledermaus 1003 matches no DFD.
* Departures no DFD explains: 2010 files store 7048 beam by beam with
  N x 14 zero bytes (every DFD from 2.20 on says sample by sample), and
  pad 7012 with stale samples. Each is detected from the record itself.

DFDs consulted: DFD 3.10 (protocol 5), plus the archive kept with
MB-System (https://www.mbari.org/technology/mb-system/documentation/
data-format-specifications/): DFD 0.35-0.54 (2002-2004), 1.00 (2006),
2.20-2.41 (2011-2015), 3.07 and 3.09. What they did not resolve: 7026
("7k detection data", in 2010 files) never had a published layout and was
removed in DFD 2.11, so it is kept raw; 7777 is "reserved for filler
records (used when repairing corrupt files)" from DFD 3.05 on, with no
body defined; no DFD defines 7200 optional data shorter than 12 bytes (RDR
1.2 writes 6) or the PDS 7028 placement; and 7038 (raw IQ) is listed only
as reserved.

``REAL_DATA_VERIFIED`` maps each verified record type to a short summary
of the evidence. ``s7k.py -V`` uses it for its "Real Data" column.
"""

__all__ = ["REAL_DATA_VERIFIED"]

REAL_DATA_VERIFIED = {
    # Sensors
    1003: "Reson 7k (36/37 B), PDS 2.3-3.9, HYPACK (UTM grid), NORBIT (37 B), Fledermaus (35 B); "
          "positions match survey areas",
    1008: "PDS 3.6-3.9, Fledermaus (AUV depth 1487 m)",
    1010: "PDS 2.3-3.9; sound speed and absorption plausible",
    1012: "Reson 7k, PDS, HYPACK, NORBIT",
    1013: "Reson 7k, PDS, HYPACK, NORBIT (one 2011 writer stored degrees)",
    1015: "PDS 2.3-3.9, NORBIT, 2006 merged navigation (device 180)",
    1016: "PDS 2.3-3.9, NORBIT, 2006 merged attitude; TimeDifference in ms",
    # Sonar
    7000: "every writer seen; 7125, 7101, 7111, T20-P, T50-R, NORBIT",
    7001: "RDR 1.2 to 7kCenter 6.3",
    7002: "RDR 1.2 (24-byte DFD 1.00 form, 2009), 7kCenter 6.0-6.3",
    7004: "every writer seen",
    7006: "Reson 7k, PDS, HYPACK, iXblue SeapiX converter",
    7007: "RDR 1.2, 7kCenter 3.7-6.1, PDS, NORBIT, HYPACK",
    7008: "RDR 1.2 (no checksum), 7kCenter 3.7-6.0, HYPACK, PDS 2.3; RowColumnFlag 0 and 1",
    7009: "PDS 2.3-3.9 (layout from older DFDs; position and ping match 7028 optional data)",
    7010: "7kCenter 6.0-6.3; TVG gains plausible",
    7011: "RDR 1.2 to 7kCenter 6.0; image matches the 7008 water column of the same ping",
    7012: "7kCenter 6.1 (roll only, per sample); 2010 files (trailing stale bytes after N samples)",
    7017: "7kCenter 3.7-6.1",
    7018: "7kCenter 3.7-6.1 (no checksum); 7027 detections land on the bottom echo",
    7021: "RDR 1.2, 7kCenter 6.0-6.3; BITE values inside their limits, board 7KTIMEs match the frame",
    7022: "RDR 2.0 to 7kCenter 6.3",
    7027: "22-byte (DFD 2.2x: 7kCenter 3.x-5.x, PDS), 26-byte (DFD 2.31-2.41: 7kCenter 6.1, NORBIT), "
          "34-byte (DFD 3.x: 7kCenter 6.3)",
    7028: "7kCenter 3.7-6.1, NORBIT; PDS (optional data misplaced, see optional_data)",
    7030: "PDS 2.3-3.9, NORBIT",
    7038: "7125 raw IQ, 2012 (layout from the CCOM MATLAB reader)",
    7041: "7kCenter 5.2 (DFD 2.20-2.31 44-byte header: nearest-neighbour down-sampling, 16-bit "
          "magnitude; seafloor arc at the 7000 range)",
    7042: "NORBIT STX and WBMS; 2x downsampled samples line up with 7027 travel times",
    7048: "2010 RDR 1.2-era 7125 (beam-major with N x 14 zero bytes, equal to the 7018 amplitudes)",
    7057: "2010 RDR 1.2-era 7125 (W = 4); port and starboard series plausible",
    7058: "7kCenter 6.3 (T50-R); sample counts, footprints flag and N agree with 7027",
    # System
    7200: "every writer seen (RDR 1.2 optional data is 6 bytes)",
    7300: "RDR 1.2 to 7kCenter 6.1",
    7501: "files with no 7200 header (HYPACK-style, 2011; SatMon calibration, 2013)",
    7503: "RDR 1.2 to 7kCenter 6.3 (no ReceiveBeamWidth field)",
    7504: "7kCenter 6.0-6.3 (RecordVersion 2; later fields are 0xFB fill)",
    7610: "4-byte pre-DFD 2.31 form (NORBIT, 7kCenter 5.3, PDS) and 12-byte full form (PDS)",
}
