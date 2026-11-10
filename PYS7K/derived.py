"""
Convenience values computed from raw records, in batches.

The record decoders report what is stored. The functions here convert and
combine those values for analysis. Every function takes an ``s7k`` object,
indexes it if needed, and returns plain dictionaries of numpy arrays (or
DataFrames for tabular settings). The raw records are never modified.

Each function documents exactly which conversion it applies. In summary:

* Times are POSIX seconds (UTC) computed from the record frame's 7KTIME.
* Angles stay in radians unless ``degrees=True`` is passed.
* Two-way travel time is DetectionPoint / SamplingRate (DFD Appendix F).
* Bit fields are split with the specs in ``PYS7K.bitfields``.
* No sign conventions are changed. The DFD's own conventions apply (roll
  positive port up, pitch positive bow up, heave positive up, Section 2.2).

Fixed-size records (1003, 1012, 1013, 1015, 7610, ...) are gathered
directly from the memory-mapped file for every record at once, without
decoding each record in Python.
"""

import warnings

import numpy as np
import pandas as pd

from .bitfields import DETECTION_FLAGS_7027, decode_bits
from .registry import MIN_DECODED_PROTOCOL, RECORD_CODECS, decode_record

__all__ = [
    "gather_header", "record_times", "decode_text",
    "read_attitude", "read_navigation", "read_detections", "read_bathymetry_7006",
    "read_runtime", "read_svp", "read_sound_velocity", "read_installation",
    "read_beam_geometry", "read_snippets", "read_watercolumn", "read_sidescan",
]

_GATHER_CHUNK = 1_000_000


def _index(S):
    if S.Index is None:
        S.index_file()
    S._ensure_read()
    return S.Index


def decode_text(value):
    """Decode a stored c8 string: up to the first NUL, as ASCII, with
    undecodable bytes replaced."""
    if isinstance(value, np.ndarray):
        value = value.tobytes()
    return bytes(value).split(b"\x00", 1)[0].decode("ascii", "replace")


def record_times(S, record_types=None):
    """
    Frame times for records.

    :param record_types: an int or collection of ints. All records if
        omitted.
    :return: a DataFrame with RecordType, Time (POSIX seconds) and Datetime
        (UTC), indexed like ``S.Index``.
    """
    idx = _index(S)
    if record_types is not None:
        types = [record_types] if np.isscalar(record_types) else list(record_types)
        idx = idx[idx.RecordType.isin(types)]
    return pd.DataFrame({"RecordType": idx.RecordType, "Time": idx.Time,
                         "Datetime": pd.to_datetime(idx.Time, unit="s", utc=True)})


def gather_header(S, record_type):
    """
    Gather the record type header of every record of ``record_type`` at
    once, straight from the memory-mapped file. Only for record types whose
    codec is a header-only record (1003, 1012, 1013, 1015, 7000, 7503,
    7610, ...). When records of different lengths occur (1003 with and
    without NumberOfSatellites, for example), the fields present in every
    record are gathered.

    :return: a dictionary of arrays, one per field, plus ``Time`` and
        ``Record`` (the index row). Empty if there are no such records.
    """
    idx = _index(S)
    # Records older than protocol 5 have other layouts.
    rows = idx[(idx.RecordType == record_type) & (idx.ProtocolVersion >= MIN_DECODED_PROTOCOL)]
    codec = RECORD_CODECS[record_type]
    section = codec.rth
    if not len(rows):
        dt = section.numpy_dtype()
        out = {name: np.zeros(0, dt[name]) for name in dt.names}
        out.update(Time=np.zeros(0), Record=np.zeros(0, np.int64))
        return out
    offsets = rows.Offset.to_numpy()
    file_bytes = np.frombuffer(S._mmap, dtype=np.uint8)
    try:
        rth_offset = file_bytes[(offsets + 2)[:, None] + np.arange(2)].view("<u2").reshape(-1).astype(np.int64) + 4
        body_end = np.where(rows.OptionalDataOffset.to_numpy() > 0, rows.OptionalDataOffset.to_numpy(),
                            rows.Size.to_numpy() - (4 if getattr(S, "ChecksumTrailer", True) else 0))
        nbytes = int(min(section.size, (body_end - rth_offset).min()))
        dt = section.numpy_dtype(nbytes)
        starts = offsets + rth_offset
        parts = []
        for i in range(0, len(starts), _GATHER_CHUNK):
            chunk = starts[i:i + _GATHER_CHUNK]
            parts.append(file_bytes[chunk[:, None] + np.arange(dt.itemsize)].view(dt).reshape(-1))
        arr = np.concatenate(parts) if len(parts) > 1 else parts[0]
    finally:
        del file_bytes
    out = {name: arr[name] for name in dt.names}
    out["Time"] = rows.Time.to_numpy()
    out["Record"] = rows.index.to_numpy()
    return out


def read_attitude(S, degrees=False):
    """
    Attitude and heading from 1012 (roll, pitch, heave), 1013 (heading) and
    1016 (attitude sets). The sources are returned separately, not merged
    or interpolated.

    1016 sample times are the record time plus TimeDifference / 1000
    (TimeDifference is in milliseconds, Table 33).

    The DFD stores angles in radians. Some third-party writers store
    degrees instead: a 2011 7125 file with no 7200 header (its 1003/1012/
    1013 records carry Size - 4 in the checksum field, as HYPACK writes
    them) has 1013 headings such as 260.8. HYPACK files from 2013 store
    radians. A ``UserWarning`` is issued when heading magnitudes
    exceed 2*pi or roll/pitch magnitudes exceed pi, since such values cannot
    be radians. The values are still returned (and converted, if asked) as
    if they were radians.

    :param degrees: convert roll, pitch and heading from radians to degrees.
    :return: ``{"1012": {...}, "1013": {...}, "1016": {...}}``, with only
        the sources present in the file. Each holds Time and the measured
        values as arrays.
    """
    idx = _index(S)
    stored = {"Roll": [], "Pitch": [], "Heading": []}

    def conv(a, name):
        a = a.astype(np.float64)
        stored[name].append(a)
        return np.rad2deg(a) if degrees else a
    out = {}
    if (idx.RecordType == 1012).any():
        g = gather_header(S, 1012)
        out["1012"] = {"Time": g["Time"], "Roll": conv(g["Roll"], "Roll"), "Pitch": conv(g["Pitch"], "Pitch"),
                       "Heave": g["Heave"].astype(np.float64)}
    if (idx.RecordType == 1013).any():
        g = gather_header(S, 1013)
        out["1013"] = {"Time": g["Time"], "Heading": conv(g["Heading"], "Heading")}
    if (idx.RecordType == 1016).any():
        cols = {k: [] for k in ("TimeDifference", "Roll", "Pitch", "Heave", "Heading")}
        for rec in S.iter_records(1016):
            samples = rec["RD"]["Samples"]
            for k in cols:
                cols[k].append(np.asarray(samples[k]))
        rec_times = idx.Time[idx.RecordType == 1016].to_numpy()
        counts = np.array([len(c) for c in cols["Roll"]])
        td = np.concatenate(cols["TimeDifference"]).astype(np.float64)
        out["1016"] = {
            "Time": np.repeat(rec_times, counts) + td / 1000.0,
            "TimeDifference": td,
            "Roll": conv(np.concatenate(cols["Roll"]), "Roll"),
            "Pitch": conv(np.concatenate(cols["Pitch"]), "Pitch"),
            "Heave": np.concatenate(cols["Heave"]).astype(np.float64),
            "Heading": conv(np.concatenate(cols["Heading"]), "Heading"),
        }
    for name, limit in (("Roll", np.pi), ("Pitch", np.pi), ("Heading", 2 * np.pi)):
        vals = np.concatenate(stored[name]) if stored[name] else np.zeros(0)
        if len(vals) and np.nanmax(np.abs(vals)) > limit + 1e-6:
            warnings.warn("stored %s values up to %.6g exceed %.4g, so they are not radians "
                          "(the writer probably stored degrees)" % (name, np.nanmax(np.abs(vals)), limit))
    return out


def read_navigation(S, degrees=True):
    """
    Positions from 1003 and 1015.

    For 1003, ``Latitude``/``Longitude`` are filled only where
    PositionTypeFlag is 0 (geographic, radians). Grid positions
    (PositionTypeFlag 1) are left NaN there. The raw LatitudeOrNorthing and
    LongitudeOrEasting are always included, unconverted, together with
    DatumIdentifier, PositionTypeFlag and UTMZone.

    :param degrees: return geographic coordinates in degrees (default) or
        radians.
    :return: ``{"1003": {...}, "1015": {...}}`` with the sources present.
    """
    idx = _index(S)
    conv = np.rad2deg if degrees else (lambda a: a)
    out = {}
    g = gather_header(S, 1003) if (idx.RecordType == 1003).any() else {"Record": []}
    if len(g["Record"]):
        geo = g["PositionTypeFlag"] == 0
        d = {k: v for k, v in g.items() if k != "Record"}
        d["Latitude"] = np.where(geo, conv(g["LatitudeOrNorthing"]), np.nan)
        d["Longitude"] = np.where(geo, conv(g["LongitudeOrEasting"]), np.nan)
        out["1003"] = d
    if (idx.RecordType == 1015).any():
        g = gather_header(S, 1015)
        d = {k: v for k, v in g.items() if k != "Record"}
        d["Latitude"] = conv(g["Latitude"])
        d["Longitude"] = conv(g["Longitude"])
        if degrees:
            d["CourseOverGround"] = np.rad2deg(g["CourseOverGround"])
            d["Heading"] = np.rad2deg(g["Heading"])
        out["1015"] = d
    return out


def _pad(arrays, fill, dtype):
    n = max((len(a) for a in arrays), default=0)
    out = np.full((len(arrays), n), fill, dtype=dtype)
    for i, a in enumerate(arrays):
        out[i, :len(a)] = a
    return out


def read_detections(S, degrees=False, flags=True):
    """
    7027 raw detections for every ping, as ping-by-detection arrays padded
    with NaN (or 0 for integer fields, with ``Valid`` marking real entries).

    Adds:

    * ``TwoWayTravelTime`` = DetectionPoint / SamplingRate, in seconds
      (Appendix F). This is uncorrected for the Tx/Rx offset.
    * ``TxAngle`` per ping, and ``RxAngle`` per detection, in radians (or
      degrees with ``degrees=True``). Signs are as stored.
    * With ``flags=True``, the per-detection Flags split into
      MagnitudeDetect, PhaseDetect, QualityType, DetectionPriority and
      NotUsedInSnippet (Table 72).

    Fields missing from older layouts (Intensity, MinLimit, MaxLimit in
    22-byte detections) are NaN.

    :return: a dictionary. Per-ping arrays have shape (P,), per-detection
        arrays (P, Nmax).
    """
    pings = {k: [] for k in ("Time", "PingNumber", "MultiPingSequence", "SonarId", "N", "DataFieldSize",
                             "DetectionAlgorithm", "Flags", "SamplingRate", "TxAngle", "AppliedRoll",
                             "DeviceId", "SystemEnumerator")}
    det_names = ["BeamDescriptor", "DetectionPoint", "RxAngle", "Flags", "Quality", "Uncertainty",
                 "Intensity", "MinLimit", "MaxLimit"]
    dets = {k: [] for k in det_names}
    idx = _index(S)
    for row, buf in S.iter_records(7027, raw=True):
        rec = decode_record(buf, checksum_trailer=getattr(S, "ChecksumTrailer", True))
        rth = rec["RTH"]
        pings["Time"].append(idx.at[row, "Time"])
        pings["DeviceId"].append(rec["Frame"]["DeviceIdentifier"])
        pings["SystemEnumerator"].append(rec["Frame"]["SystemEnumerator"])
        for k in pings:
            if k in rth:
                pings[k].append(rth[k])
        d = rec["RD"]["Detections"]
        n = len(d["BeamDescriptor"])
        for k in det_names:
            dets[k].append(np.asarray(d[k]) if k in d else np.full(n, np.nan, np.float32))
    out = {k: np.asarray(v) for k, v in pings.items()}
    valid = [np.ones(len(a), bool) for a in dets["BeamDescriptor"]]
    out["Valid"] = _pad(valid, False, bool)
    for k in det_names:
        if k in ("BeamDescriptor", "Flags", "Quality"):
            out[k] = _pad(dets[k], 0, np.uint32 if k != "BeamDescriptor" else np.uint16)
        else:
            out[k] = _pad(dets[k], np.nan, np.float64)
    if len(out["Time"]):
        out["TwoWayTravelTime"] = out["DetectionPoint"] / out["SamplingRate"][:, None]
    else:
        out["TwoWayTravelTime"] = np.zeros((0, 0))
    if degrees:
        out["RxAngle"] = np.rad2deg(out["RxAngle"])
        out["TxAngle"] = np.rad2deg(out["TxAngle"].astype(np.float64))
        out["AppliedRoll"] = np.rad2deg(out["AppliedRoll"].astype(np.float64))
    if flags:
        for k, v in decode_bits(out["Flags"], DETECTION_FLAGS_7027).items():
            out[k] = v
    return out


def read_bathymetry_7006(S):
    """
    7006 bathymetry for every ping as ping-by-beam arrays. ``Range`` is
    two-way travel time in seconds, as stored. Use ``read_beam_geometry``
    for the matching 7004 beam angles.
    """
    rows = list(S.iter_records(7006))
    out = {
        "PingNumber": np.array([r["RTH"]["PingNumber"] for r in rows], dtype=np.int64),
        "SoundVelocity": np.array([r["RTH"]["SoundVelocity"] for r in rows]),
        "Flags": np.array([r["RTH"]["Flags"] for r in rows], dtype=np.uint8),
    }
    out["Time"] = S.Index.Time[S.Index.RecordType == 7006].to_numpy()
    for k in ("Range", "Quality", "Intensity", "MinFilterInfo", "MaxFilterInfo"):
        arrays = [np.asarray(r["RD"]["Beams"].get(k, np.zeros(0))) for r in rows]
        out[k] = _pad(arrays, 0 if k == "Quality" else np.nan, np.uint8 if k == "Quality" else np.float64)
    return out


def read_runtime(S, record_type=7503):
    """
    Sonar settings, one row per record, as a DataFrame of the raw RTH
    fields (7503 by default, or 7000). Bit fields stay integers. Use
    ``bitfields.decode_bits`` to expand them.
    """
    g = gather_header(S, record_type)
    frame = pd.DataFrame({k: v for k, v in g.items() if np.ndim(v) == 1})
    frame.insert(0, "Datetime", pd.to_datetime(frame.Time, unit="s", utc=True))
    return frame


def read_svp(S, degrees=True):
    """
    Sound velocity profiles from 1009 records.

    :return: a list of dictionaries, one per profile, with Time, the RTH
        fields (Latitude and Longitude in degrees unless ``degrees=False``)
        and the Depth and SoundVelocity arrays.
    """
    idx = _index(S)
    out = []
    for row, buf in S.iter_records(1009, raw=True):
        rec = decode_record(buf, checksum_trailer=getattr(S, "ChecksumTrailer", True))
        d = dict(rec["RTH"])
        d["Time"] = float(idx.at[row, "Time"])
        if degrees:
            d["Latitude"] = float(np.rad2deg(d["Latitude"]))
            d["Longitude"] = float(np.rad2deg(d["Longitude"]))
        d["Depth"] = np.array(rec["RD"]["Samples"]["Depth"], dtype=np.float64)
        d["SoundVelocity"] = np.array(rec["RD"]["Samples"]["SoundVelocity"], dtype=np.float64)
        out.append(d)
    return out


def read_sound_velocity(S):
    """Surface sound velocity from 7610 records: Time, SoundVelocity,
    Temperature and Pressure (NaN where a record omits them)."""
    g = gather_header(S, 7610)
    out = {"Time": g["Time"], "SoundVelocity": g["SoundVelocity"].astype(np.float64)}
    for k in ("Temperature", "Pressure"):
        out[k] = g[k].astype(np.float64) if k in g else np.full(len(g["Time"]), np.nan)
    return out


def read_installation(S):
    """
    7030 installation parameters, one dictionary per record. The four
    version-info byte arrays are decoded to text using their stored
    lengths (as ``*Text`` entries); every raw field is kept.
    """
    out = []
    for rec in S.iter_records(7030):
        d = dict(rec["RTH"])
        for name in ("FirmwareVersion", "SoftwareVersion", "SevenKSoftwareVersion", "RecordProtocolVersion"):
            n = d.get(name + "Length")
            if n is not None and d.get(name) is not None:
                d[name + "Text"] = decode_text(d[name][:n])
        d["DeviceId"] = rec["Frame"]["DeviceIdentifier"]
        out.append(d)
    return out


def read_beam_geometry(S, degrees=False):
    """7004 beam geometry: per-record arrays padded to the largest N."""
    recs = list(S.iter_records(7004))
    names = ["BeamVerticalDirection", "BeamHorizontalDirection", "BeamWidthY", "BeamWidthX", "TxDelay"]
    out = {"Time": S.Index.Time[S.Index.RecordType == 7004].to_numpy(),
           "N": np.array([r["RTH"]["N"] for r in recs], dtype=np.int64)}
    for k in names:
        arrays = [np.asarray(r["RD"]["Beams"].get(k, np.zeros(0)), dtype=np.float64) for r in recs]
        a = _pad(arrays, np.nan, np.float64)
        out[k] = np.rad2deg(a) if (degrees and k != "TxDelay") else a
    return out


def read_snippets(S):
    """
    7028 snippets, one dictionary per ping: PingNumber, Time, the beam
    descriptor table, and ``Snippets``, a list holding each beam's samples
    as its own array (SnippetEnd - SnippetStart + 1 samples).
    """
    idx = _index(S)
    out = []
    for row, buf in S.iter_records(7028, raw=True):
        rec = decode_record(buf, checksum_trailer=getattr(S, "ChecksumTrailer", True))
        beams = rec["RD"]["Beams"]
        counts = (beams["SnippetEnd"].astype(np.int64) - beams["SnippetStart"].astype(np.int64) + 1)
        splits = np.cumsum(np.clip(counts, 0, None))[:-1]
        out.append({"PingNumber": rec["RTH"]["PingNumber"], "Time": float(idx.at[row, "Time"]),
                    "Beams": beams, "Snippets": np.split(rec["RD"]["Samples"], splits)})
    return out


def read_watercolumn(record):
    """
    Turn one water column record (7008, 7018, 7042 or 7048) into a
    samples-by-beams image for each sample component.

    * 7018: the stored (S, N) Amplitude and Phase arrays, as float64.
    * 7048: the stored (S, N) calibrated Intensity, as float64.
    * 7042: each beam's samples from row 0, NaN past its sample count.
      ``FirstSample`` is the RTH FirstSample. Values are as stored: 8-bit
      magnitudes (Flags bit 2) are ``factor * log10(magnitude)`` with
      factor CompressionFactor (Flags bit 13) or 40 (Appendix I), so
      ``20 * value / factor`` gives dB. ``SampleRate`` is the RTH sample
      rate, which already accounts for downsampling, and ``FirstSample`` is
      in those downsampled units: row ``i`` is at two-way travel time
      ``(FirstSample + i) / SampleRate``. Verified on NORBIT STX (device
      13016) data from 2024 with 2x downsampling (Flags 0x122): 7027
      detections land within a few rows of the bottom echo this way. Do
      not use the 7027 sample numbers directly, since its SamplingRate is
      the undecimated rate.
    * 7008: each beam's samples placed at rows FirstSample..LastSample
      (relative to the smallest FirstSample), NaN elsewhere. RowColumnFlag
      0 (all samples of a beam together) and 1 (sample-interleaved, which
      requires every beam to have the same sample range) are handled.

    :param record: a decoded 7008 or 7018 record.
    :return: a dictionary with ``FirstSample`` (the sample number of row
        0), ``Beams`` (the beam numbers, one per column) and one 2-D array
        per component.
    """
    rt = record["RecordType"]
    rd = record["RD"]
    if rt == 7018:
        n = record["RTH"]["N"]
        return {"FirstSample": 0, "Beams": np.arange(n),
                "Amplitude": np.asarray(rd["Amplitude"], np.float64),
                "Phase": np.asarray(rd["Phase"], np.float64)}
    if rt == 7048:
        rth = record["RTH"]
        return {"FirstSample": 0, "Beams": rth["FirstBeam"] + np.arange(rth["N"]),
                "Intensity": np.asarray(rd["Intensity"], np.float64)}
    if rt == 7042:
        rth = record["RTH"]
        counts = np.asarray(rd["Beams"]["Samples"], np.int64)
        nrows = int(counts.max()) if len(counts) else 0
        out = {"FirstSample": rth["FirstSample"], "Beams": np.asarray(rd["Beams"]["Beam"]),
               "SampleRate": rth["SampleRate"]}
        if rth["Flags"] & 0x4:
            out["CompressionFactor"] = rth["CompressionFactor"] if rth["Flags"] & 0x2000 else 40.0
        starts = np.concatenate([[0], np.cumsum(counts)[:-1]]) if len(counts) else counts
        for name, values in rd["Samples"].items():
            img = np.full((nrows, len(counts)), np.nan)
            values = np.asarray(values, np.float64)
            for b, (s0, c) in enumerate(zip(starts, counts)):
                img[:c, b] = values[s0:s0 + c]
            out[name] = img
        return out
    if rt != 7008:
        raise ValueError("read_watercolumn handles 7008, 7018, 7042 and 7048 records, not %d" % rt)
    beams = rd["Beams"]
    first = beams["FirstSample"].astype(np.int64)
    last = beams["LastSample"].astype(np.int64)
    counts = np.clip(last - first + 1, 0, None)
    nb = len(first)
    if nb == 0 or not rd["Samples"]:
        return {"FirstSample": 0, "Beams": np.asarray(beams["Descriptor"])}
    s0 = int(first.min())
    nrows = int((last.max() - s0 + 1))
    out = {"FirstSample": s0, "Beams": np.asarray(beams["Descriptor"])}
    for name, values in rd["Samples"].items():
        img = np.full((nrows, nb), np.nan)
        values = np.asarray(values, dtype=np.float64)
        if record["RTH"]["RowColumnFlag"] == 1 and len(set(counts.tolist())) == 1:
            ns = int(counts[0])
            block = values[:ns * nb].reshape(ns, nb)
            img[first[0] - s0:first[0] - s0 + ns, :] = block
        else:
            pos = 0
            for b in range(nb):
                c = int(counts[b])
                img[first[b] - s0:first[b] - s0 + c, b] = values[pos:pos + c]
                pos += c
        out[name] = img
    return out


def read_sidescan(S, record_type=7007):
    """
    Side scan from 7007 (raw) or 7057 (calibrated) records, one dictionary
    per record: PingNumber, Time, BeamPosition, CurrentBeamNumber and the
    Port and Starboard sample arrays as stored (first sample at range 0).
    7057 also carries PortBeams and StarboardBeams.
    """
    idx = _index(S)
    out = []
    for row, buf in S.iter_records(record_type, raw=True):
        rec = decode_record(buf, checksum_trailer=getattr(S, "ChecksumTrailer", True))
        d = {"PingNumber": rec["RTH"]["PingNumber"], "Time": float(idx.at[row, "Time"]),
             "BeamPosition": rec["RTH"]["BeamPosition"], "CurrentBeamNumber": rec["RTH"]["CurrentBeamNumber"]}
        d.update(rec["RD"])
        out.append(d)
    return out
