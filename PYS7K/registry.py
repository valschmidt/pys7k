"""
The record-type registry, and decoding and encoding of whole records.

``RECORD_CODECS`` maps each record type identifier to its codec.
``RECORD_TYPE_DESCRIPTIONS`` names every record type DFD 3.10 lists,
including ones with no codec yet, so reports can label them.
"""

import struct

import numpy as np

from .frame import (CHECKSUM_SIZE, DRF, DRF_SIZE, FLAG_CHECKSUM_VALID,
                    LEGACY_FRAMES, S7KFormatError, checksum)
from .optional_data import OPTIONAL_DATA_CODECS
from .records import ALL_CODECS
from .records.base import new_record
from .records.legacy import CODECS as LEGACY_CODECS

__all__ = [
    "RECORD_CODECS", "RECORD_TYPE_DESCRIPTIONS", "DEVICE_IDENTIFIERS",
    "PING_RECORD_TYPES", "record_name", "new", "decode_record", "encode_record",
    "detect_checksum_trailer", "MIN_DECODED_PROTOCOL", "LEGACY_CODECS",
]

RECORD_CODECS = {c.record_type: c for c in ALL_CODECS}

# Table 8: Record Type Definitions (names for every listed type).
RECORD_TYPE_DESCRIPTIONS = {
    1000: "Reference Point", 1001: "Sensor Offset Position",
    1002: "Sensor Offset Position Calibrated", 1003: "Position",
    1004: "Custom Attitude Information", 1005: "Tide", 1006: "Altitude",
    1007: "Motion Over Ground", 1008: "Depth", 1009: "Sound Velocity Profile",
    1010: "CTD", 1011: "Geodesy", 1012: "Roll Pitch Heave", 1013: "Heading",
    1014: "Survey Line", 1015: "Navigation", 1016: "Attitude", 1017: "Pan Tilt",
    1020: "Sonar Installation Identifiers", 1021: "Reserved (IDH)",
    1022: "Reserved (IOP motion)", 1050: "Reserved (sensor calibration)",
    1200: "Reserved (generic side-scan)", 2004: "Sonar Pipe Environment",
    3001: "Contact Output", 7000: "Sonar Settings", 7001: "Configuration",
    7002: "Match Filter", 7003: "Firmware and Hardware Configuration",
    7004: "Beam Geometry", 7005: "Reserved (calibration data)",
    7006: "Bathymetric Data", 7007: "Side Scan Data",
    7008: "Generic Water Column Data", 7009: "Vertical Depth",
    7010: "TVG Values", 7011: "Image Data", 7012: "Ping Motion Data",
    7014: "Adaptive Gate", 7016: "Reserved (extended bottom detect)",
    7017: "Detection Data Setup", 7018: "Beamformed Data",
    7019: "Vernier Processing Data (Raw)", 7021: "Built-In Test Environment Data",
    7022: "Sonar Source Version", 7023: "8k Wet End Version",
    7024: "Reserved (license)", 7026: "Detection Data (no layout published; dropped in DFD 2.11)",
    7027: "Raw Detection Data", 7028: "Snippet Data",
    7029: "Vernier Processing Data (Filtered)", 7030: "Sonar Installation Parameters",
    7031: "Built-In Test Environment Data (Summary)", 7037: "Reserved",
    7038: "Raw IQ Data (undocumented)", 7040: "Reserved (Tx configuration)",
    7041: "Compressed Beamformed Intensity Data", 7042: "Compressed Water Column Data",
    7047: "Segmented Raw Detection Data", 7048: "Calibrated Beam Data",
    7049: "Reserved", 7050: "System Events", 7051: "System Event Message",
    7052: "RDR Recording Status", 7053: "Subscriptions",
    7054: "RDR Storage Recording Short Update", 7055: "Calibration Status",
    7057: "Calibrated Side-Scan Data", 7058: "Snippet Backscattering Strength",
    7059: "MB2 Specific Status", 7060: "Reserved (target data)", 7068: "Reserved",
    7200: "File Header", 7300: "File Catalog", 7310: "Reserved (trigger)",
    7311: "Reserved (trigger sequence setup)", 7312: "Reserved (trigger sequence done)",
    7400: "Time Message", 7500: "Remote Control",
    7501: "Remote Control Acknowledge", 7502: "Remote Control Not Acknowledge",
    7503: "Remote Control Sonar Settings", 7504: "Common System Settings",
    7505: "Reserved", 7510: "SV Filtering", 7511: "System Lock Status",
    7515: "Timestamp (undocumented)", 7610: "Sound Velocity", 7611: "Absorption Loss",
    7612: "Spreading Loss", 7613: "Profile Average Salinity",
    7614: "Profile Average Temperature", 7777: "Filler (file repair)",
    8012: "Pitch Yaw Heave Flag (undocumented)", 8100: "8k Series Sonar Data (undocumented)",
    10000: "SBES Channel Settings", 10003: "SBES Channel Status",
    10004: "SBES System Settings", 10018: "SBES Echogram Water Column Data",
    10027: "SBES Raw Detection Data", 10049: "SBES Depth Reading",
    10060: "SBES Port Configuration", 10061: "SBES Port Status", 10062: "SBES Port QC",
}
for _rt, _codec in RECORD_CODECS.items():
    RECORD_TYPE_DESCRIPTIONS.setdefault(_rt, _codec.name)

# Appendix B: Device Identifiers.
DEVICE_IDENTIFIERS = {
    0: "Unspecified", 20: "SeaBat T20-P/T20-S", 22: "SeaBat T20-R (dual head)", 30: "SeaBat F30",
    50: "SeaBat T50-P/T50-R", 52: "SeaBat T50-R (dual head)",
    100: "Generic Position Sensor", 101: "Generic Heading Sensor", 102: "Generic Attitude Sensor",
    103: "Generic MBES", 104: "Generic Side-scan Sonar", 105: "Generic Sub-bottom Profiler",
    1000: "Odom MB1", 1001: "TrueTime PCISG", 1002: "Odom MB2", 2000: "CDC SMCG", 2001: "CDC SPG",
    2002: "Empire Magnetics YS2000", 4013: "RESON TC4013", 6000: "RESON DiverDat",
    7000: "7k sonar source", 7001: "7k User Interface", 7003: "Teledyne PDS", 7004: "7k Logger",
    7005: "BlueView ProScan", 7012: "SeaBat 7012", 7100: "SeaBat 7100", 7101: "SeaBat 7101",
    7102: "SeaBat 7102", 7111: "SeaBat 7111", 7112: "SeaBat 7112", 7123: "SeaBat 7123",
    7125: "SeaBat 7125", 7128: "SeaBat 7128", 7130: "SeaBat 7130", 7150: "SeaBat 7150",
    7160: "SeaBat 7160", 8100: "SeaBat 8100", 8101: "SeaBat 8101", 8102: "SeaBat 8102",
    8111: "SeaBat 8111", 8123: "SeaBat 8123", 8124: "SeaBat 8124", 8125: "SeaBat 8125",
    8128: "SeaBat 8128", 8150: "SeaBat 8150", 8160: "SeaBat 8160", 9000: "Echotrac E20",
    10000: "TSS DMS 05", 10001: "TSS 335B", 10002: "TSS 332B", 10010: "SeaBird SBE37",
    10200: "Litton 200", 11000: "EdgeTech FS-DW SBP", 11001: "EdgeTech FS-DW LFSSS",
    11002: "EdgeTech FS-DW HFSSS", 11100: "BlueFin vehicle controller", 11200: "iFremer Techsas",
    12000: "Simrad RPT319", 13002: "NORBIT WBMS FLS 400kHz", 13003: "NORBIT WBMS Bathy 400kHz",
    13004: "NORBIT iWBMS", 13005: "NORBIT Bathy 400kHz Compact", 13007: "NORBIT WBMS Bathy 200kHz",
    13008: "NORBIT Bathy 400kHz", 13009: "NORBIT FLS DeepSea 400k", 13010: "NORBIT Bathy DeepSea 400k",
    13011: "NORBIT Bathy DeepSea 200k", 13012: "NORBIT iLIDAR", 13016: "NORBIT Bathy STX 400kHz",
    13017: "NORBIT Bathy STX 200kHz", 13018: "NORBIT iWBMSe", 14000: "HydroSweep 3 DS",
    14001: "HydroSweep 3 MD50", 14002: "HydroSweep 3 MD30",
}

PING_RECORD_TYPES = frozenset(rt for rt, c in RECORD_CODECS.items() if c.ping)

_U32 = struct.Struct("<I")

# Record bodies, optional data and the Flags checksum bit are interpreted
# with DFD 3.10 layouts for this protocol version and later (DFD 3.10 is
# protocol 5). Older protocols use frame.LEGACY_FRAMES and LEGACY_CODECS
# (keyed by (protocol, record type)) where those are known, and are
# otherwise kept raw.
MIN_DECODED_PROTOCOL = 5


def record_name(record_type):
    """A human-readable name for ``record_type``."""
    return RECORD_TYPE_DESCRIPTIONS.get(int(record_type), "Unknown")


def new(record_type):
    """
    Return a template record dictionary for ``record_type``, ready to fill in
    and pass to ``encode_record()``.
    """
    codec = RECORD_CODECS.get(record_type)
    rec = new_record(record_type, codec)
    od = OPTIONAL_DATA_CODECS.get(record_type)
    if od is not None and record_type == 7200:
        rec["OptionalData"] = od.new()
    return rec


def decode_record(data, offset=0, checksum_trailer=True):
    """
    Decode one record.

    :param data: a buffer holding the record. It is not copied: arrays in
        the result are views into it.
    :param offset: where the record starts in ``data``.
    :param checksum_trailer: False for records written without the 4-byte
        checksum the DFD requires (Fledermaus-written navigation files do
        this: their Size is 64 + body). The record then decodes with
        ``Checksum`` None, and ``encode_record`` writes it back without a
        trailer. ``s7k.index_file`` detects this per file (see
        ``detect_checksum_trailer``).

    :return: the record dictionary (see ``records.base``). A record whose
        frame is self-inconsistent (optional data offset outside the record,
        for example) is returned with its bytes in ``"Raw"`` and an
        ``"Error"`` message, and is written back verbatim.

    :raises S7KFormatError: the buffer does not hold a whole record frame.
    """
    mv = memoryview(data)
    tail = CHECKSUM_SIZE if checksum_trailer else 0
    if len(mv) - offset < DRF_SIZE + tail:
        raise S7KFormatError("buffer too short for a record at offset %d" % offset)
    frame = DRF.unpack(mv, offset)
    size = frame["Size"]
    rt = frame["RecordTypeIdentifier"]
    if size < DRF_SIZE + tail or offset + size > len(mv):
        raise S7KFormatError("record at offset %d has bad size %d" % (offset, size))
    rec_mv = mv[offset:offset + size]
    rth_start = 4 + frame["Offset"]
    odo = frame["OptionalDataOffset"]
    body_end = odo if odo else size - tail
    if rth_start < DRF_SIZE or rth_start > body_end or body_end > size - tail:
        return {
            "RecordType": rt, "Frame": frame, "Raw": bytes(rec_mv),
            "Error": "inconsistent frame offsets (Offset=%d, OptionalDataOffset=%d, Size=%d)"
                     % (frame["Offset"], odo, size),
        }
    # DFD 3.10 describes protocol version 5. Older protocols have their
    # own frame (frame.LEGACY_FRAMES) and record layouts (LEGACY_CODECS);
    # bodies with no known layout for their protocol are kept raw.
    pv = frame["ProtocolVersion"]
    modern = pv >= MIN_DECODED_PROTOCOL
    frame_size = DRF_SIZE
    if not modern and pv in LEGACY_FRAMES and rth_start >= LEGACY_FRAMES[pv].size:
        frame = LEGACY_FRAMES[pv].unpack(mv, offset)
        frame_size = LEGACY_FRAMES[pv].size
    if modern:
        codec = RECORD_CODECS.get(rt)
    else:   # legacy layouts only inside a legacy frame
        codec = LEGACY_CODECS.get((pv, rt)) if frame_size != DRF_SIZE else None
    rec = new_record(rt, codec)
    if not modern:
        rec["Note"] = "protocol version %d predates DFD 3.10 (protocol 5); %s" % (
            pv, "decoded with the layouts of its own DFD" if codec is not None
            else "the body is kept raw")
    rec["Frame"] = frame
    rec["Unparsed"]["Frame"] = bytes(rec_mv[frame_size:rth_start])
    body = rec_mv[rth_start:body_end]
    used = 0
    if codec is not None:
        try:
            rec["RTH"], rec["RD"], used = codec.decode(body)
        except Exception as exc:  # a malformed body; keep it raw
            rec["RTH"], rec["RD"], used = codec.new_rth(), {}, 0
            rec["Error"] = "decode failed: %s" % exc
    rec["Unparsed"]["Body"] = bytes(body[used:])
    if odo:
        od_buf = rec_mv[odo:size - tail]
        od_codec = OPTIONAL_DATA_CODECS.get(rt) if modern else None
        od_used = 0
        if od_codec is not None:
            rec["OptionalData"], od_used = od_codec.decode(od_buf)
        else:
            rec["OptionalData"] = {}
        rec["Unparsed"]["OptionalData"] = bytes(od_buf[od_used:])
    if not checksum_trailer:
        rec["Checksum"] = None
        return rec
    rec["Checksum"] = _U32.unpack_from(rec_mv, size - CHECKSUM_SIZE)[0]
    if modern and frame["Flags"] & FLAG_CHECKSUM_VALID:
        rec["ChecksumValid"] = checksum(rec_mv[:size - CHECKSUM_SIZE]) == rec["Checksum"]
    return rec


def encode_record(record, checksum_mode="auto"):
    """
    Encode a record dictionary to bytes.

    The frame's Size, Offset and OptionalDataOffset are recomputed from the
    encoded contents. Every other frame field is written as given.

    :param checksum_mode:
        * ``"auto"`` (default): compute the checksum when Flags bit 0 (valid
          checksum) is set (protocol 5 and later only), otherwise write
          ``record["Checksum"]`` as stored. The exception is a record that was read with a checksum
          that did not verify (``record["ChecksumValid"] is False``): its
          stored checksum is written back, so damaged or hand-edited
          records are preserved as found. Unmodified records from a file
          therefore always come back byte for byte, and edited records with
          a good checksum get a correct one. To repair a bad checksum, use
          ``"recompute"``.
        * ``"recompute"``: always compute it.
        * ``"keep"``: always write ``record["Checksum"]``.

        A record whose ``Checksum`` is None (read from a file written
        without checksum trailers) is written without one, unless its Flags
        bit 0 is set or ``checksum_mode`` is ``"recompute"``.
    """
    if "Raw" in record:
        return bytes(record["Raw"])
    rt = record["RecordType"]
    pv = record["Frame"]["ProtocolVersion"]
    modern = pv >= MIN_DECODED_PROTOCOL
    codec = RECORD_CODECS.get(rt) if modern else None
    frame_section = DRF
    if not modern and pv in LEGACY_FRAMES and all(k in record["Frame"] for k in LEGACY_FRAMES[pv].names):
        frame_section = LEGACY_FRAMES[pv]
        codec = LEGACY_CODECS.get((pv, rt))
    unparsed = record.get("Unparsed") or {}
    frame_extra = bytes(unparsed.get("Frame", b""))
    body = (codec.encode(record) if codec is not None else b"") + bytes(unparsed.get("Body", b""))
    od = b""
    if record.get("OptionalData") is not None:
        od_codec = OPTIONAL_DATA_CODECS.get(rt) if modern else None
        if od_codec is not None and record["OptionalData"]:
            od = od_codec.encode(record["OptionalData"])
        od += bytes(unparsed.get("OptionalData", b""))
    frame = dict(record["Frame"])
    head = frame_section.size + len(frame_extra)
    frame["Offset"] = head - 4
    frame["RecordTypeIdentifier"] = rt
    frame["OptionalDataOffset"] = head + len(body) if record.get("OptionalData") is not None else 0
    flagged = frame["ProtocolVersion"] >= MIN_DECODED_PROTOCOL and frame["Flags"] & FLAG_CHECKSUM_VALID
    no_trailer = record.get("Checksum", 0) is None and checksum_mode != "recompute" and not flagged
    frame["Size"] = head + len(body) + len(od) + (0 if no_trailer else CHECKSUM_SIZE)
    out = frame_section.pack(frame) + frame_extra + body + od
    if no_trailer:
        return out
    if checksum_mode == "recompute" or (checksum_mode == "auto" and flagged
                                        and record.get("ChecksumValid") is not False):
        cs = checksum(out)
    elif checksum_mode in ("auto", "keep"):
        cs = int(record.get("Checksum") or 0) & 0xFFFFFFFF
    else:
        raise ValueError("checksum_mode must be 'auto', 'recompute' or 'keep'")
    return out + _U32.pack(cs)


def detect_checksum_trailer(record_types, sizes, flags):
    """
    Decide whether a file's records end with the 4-byte checksum.

    The DFD requires it, and every Reson, PDS, HYPACK and NORBIT file seen
    has it, but Fledermaus-written navigation files (1003/1008 records,
    RecordVersion 0) leave it out. Any record with Flags bit 0 set means
    the trailer is present. Otherwise each record of a type with a
    header-only layout votes: its body length with and without a trailer
    either does or does not end on a field boundary. The trailer is assumed
    absent only when some records vote for that and none vote against.

    :param record_types, sizes, flags: index columns (array-likes).
    :return: True if the records carry checksums.
    """
    record_types = np.asarray(record_types)
    sizes = np.asarray(sizes, dtype=np.int64)
    flags = np.asarray(flags)
    if len(sizes) == 0 or (flags & FLAG_CHECKSUM_VALID).any():
        return True
    without = with_ = 0
    for rt in np.unique(record_types):
        codec = RECORD_CODECS.get(int(rt))
        bounds = getattr(codec, "field_boundaries", None)
        if bounds is None:
            continue
        bounds = bounds()
        body = sizes[record_types == rt] - DRF_SIZE
        a = np.isin(body, bounds)                    # no trailer
        b = np.isin(body - CHECKSUM_SIZE, bounds)    # with trailer
        without += int((a & ~b).sum())
        with_ += int((b & ~a).sum())
    return not (without > 0 and with_ == 0)
