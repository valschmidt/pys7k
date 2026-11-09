"""
Named bit-field specifications from DFD 3.10, and a vectorized decoder.

Records keep bit fields as the integers stored in the file. These specs let
you expand them when you want named values::

    from PYS7K.bitfields import decode_bits, CONTROL_FLAGS_7000
    decode_bits(rec["RTH"]["ControlFlags"], CONTROL_FLAGS_7000)
    {'AutoRangeMethod': 0, 'AutoBottomDetectionFilterMethod': 0, ...}

Each spec is a list of ``(name, first_bit, width)``. ``decode_bits`` works
on a scalar or on a numpy array (for example a column of per-detection
flags), returning a dictionary of the same shape.
"""

import numpy as np

__all__ = [
    "decode_bits",
    "CONTROL_FLAGS_7000", "TRANSMIT_FLAGS_7000", "RECEIVE_FLAGS_7000",
    "FLAGS_7006", "QUALITY_7006",
    "FLAGS_7017", "DETECTION_FLAGS_7017",
    "FLAGS_7027", "DETECTION_FLAGS_7027", "QUALITY_7027",
    "FLAGS_7028", "DATA_SAMPLE_TYPE_7008", "FLAGS_7042",
    "FRAME_FLAGS", "BIT_FIELDS",
]


def decode_bits(value, spec):
    """
    Split an integer bit field (or an array of them) into named parts.

    :param value: an int or an integer numpy array.
    :param spec: a list of ``(name, first_bit, width)``.
    :return: a dictionary mapping each name to an int, or to an array the
        shape of ``value``.
    """
    scalar = np.isscalar(value)
    v = np.asarray(value).astype(np.uint64)
    out = {}
    for name, bit, width in spec:
        part = (v >> np.uint64(bit)) & np.uint64((1 << width) - 1)
        out[name] = int(part) if scalar else part.astype(np.uint32 if width > 8 else np.uint8)
    return out


# Table 5: DRF Flags. Bit 0 is the checksum flag (verified on real data;
# the DFD's checksum row says "bit 1", which is a typo).
FRAME_FLAGS = [
    ("ChecksumValid", 0, 1),
    ("RecordedData", 15, 1),
]

# Table 39: 7000 Control flags (also used by 7503).
CONTROL_FLAGS_7000 = [
    ("AutoRangeMethod", 0, 4),
    ("AutoBottomDetectionFilterMethod", 4, 4),
    ("BottomDetectionRangeFilterEnabled", 8, 1),
    ("BottomDetectionDepthFilterEnabled", 9, 1),
    ("ReceiverGainAutoGain", 10, 1),
    ("ReceiverGainFixedGain", 11, 1),
    ("ReceiverGainReserved", 12, 1),
    ("Reserved13", 13, 1),
    ("TriggerOutHighForRxDuration", 14, 1),
    ("SystemActive", 15, 1),
    ("ReservedBottomDetection", 16, 4),
    ("PipeGatingFilter", 20, 1),
    ("AdaptiveGateDepthFilterFixed", 21, 1),
    ("AdaptiveGate", 22, 1),
    ("AdaptiveGateDepthFilter", 23, 1),
    ("TriggerOut", 24, 1),
    ("TriggerInEdgeNegative", 25, 1),
    ("PPSEdgeNegative", 26, 1),
    ("TimestampState", 27, 2),          # 0 n/a, 1 error, 2 warning, 3 ok
    ("DepthFilterFollowsSeafloor", 29, 1),
    ("ReducedCoverageForConstantSpacing", 30, 1),
    ("Simulator", 31, 1),
]

# Table 39: Transmit flags.
TRANSMIT_FLAGS_7000 = [
    ("PitchStabilizationMethod", 0, 4),
    ("YawStabilizationMethod", 4, 4),
]

# Table 39: Receive flags.
RECEIVE_FLAGS_7000 = [
    ("RollCompensation", 0, 1),
    ("HeaveCompensation", 2, 1),
    ("DynamicFocusingMethod", 4, 4),
    ("DopplerCompensationMethod", 8, 4),
    ("MatchFilteringMethod", 12, 4),
    ("TVGMethod", 16, 4),
    ("MultiPingMode", 20, 4),
]

# Table 46: 7006 Flags, and Table 47: 7006 Quality.
FLAGS_7006 = [
    ("LayerCompensation", 0, 1),
    ("XYZCompensation", 1, 1),
]
QUALITY_7006 = [
    ("BrightnessPass", 0, 1),
    ("ColinearityPass", 1, 1),
    ("IntensityDetectionUsed", 2, 1),
    ("PhaseDetectionUsed", 3, 1),
    ("InternalUse", 4, 1),
    ("PDSNadirFilterFail", 5, 1),
]

# Table 60: 7017 Flags, and Table 61: per-detection Flags.
FLAGS_7017 = [
    ("UserDepthFilterEnabled", 0, 1),
    ("UserRangeFilterEnabled", 1, 1),
    ("AutomaticFilterEnabled", 2, 1),
    ("NadirSearchLimitsEnabled", 3, 1),
    ("AutomaticWindowLimitsEnabled", 4, 1),
    ("QualityFilterEnabled", 5, 1),
    ("MultiDetectionEnabled", 6, 1),
]
DETECTION_FLAGS_7017 = [
    ("AutomaticLimitsValid", 0, 1),
    ("UserDefinedLimitsValid", 1, 1),
    ("QualityType", 2, 7),
    ("QualityPassesUserCriteria", 9, 1),
    ("MagnitudeDetect", 10, 1),
    ("PhaseDetect", 11, 1),
    ("DetectionPriority", 16, 4),
]

# Table 71: 7027 Flags, Table 72: per-detection Flags, Table 73: Quality
# (when the quality type is 1).
FLAGS_7027 = [
    ("UncertaintyMethod", 0, 4),        # 0 none, 1 Rob Hare, 2 Ifremer
    ("MultiDetectionEnabled", 4, 1),
    ("HasSnippetsDetectionPointFlag", 6, 1),
]
DETECTION_FLAGS_7027 = [
    ("MagnitudeDetect", 0, 1),
    ("PhaseDetect", 1, 1),
    ("QualityType", 2, 7),
    ("DetectionPriority", 9, 4),
    ("NotUsedInSnippet", 14, 1),
]
QUALITY_7027 = [
    ("BrightnessPass", 0, 1),
    ("ColinearityPass", 1, 1),
]

# Table 74: 7028 Flags.
FLAGS_7028 = [("Samples32Bit", 0, 1)]

# Table 50: 7008 Data sample type.
DATA_SAMPLE_TYPE_7008 = [
    ("Magnitude", 0, 4),                # 0 none, 2 16-bit, 3 32-bit
    ("Phase", 4, 4),                    # 0 none, 2 16-bit, 3 32-bit
    ("IQ", 8, 4),                       # 0 none, 1 16-bit, 2 32-bit
    ("ElementData", 12, 3),             # 0 beamformed, 1 element data
]

# Table 82: 7042 Flags.
FLAGS_7042 = [
    ("UseMaximumBottomDetection", 0, 1),
    ("MagnitudeOnly", 1, 1),
    ("Magnitude8BitdB", 2, 1),
    ("Reserved3", 3, 1),
    ("DownsamplingDivisor", 4, 4),
    ("DownsamplingType", 8, 4),         # 0 none, 1 middle, 2 peak, 3 average
    ("Data32Bit", 12, 1),
    ("CompressionFactorAvailable", 13, 1),
    ("SegmentNumbersAvailable", 14, 1),
    ("FirstSampleContainsRxDelay", 15, 1),
]

# Bit fields indexed by (record type, section, field name), for tools that
# want to expand every known bit field in a record.
BIT_FIELDS = {
    (7000, "RTH", "ControlFlags"): CONTROL_FLAGS_7000,
    (7000, "RTH", "TransmitFlags"): TRANSMIT_FLAGS_7000,
    (7000, "RTH", "ReceiveFlags"): RECEIVE_FLAGS_7000,
    (7503, "RTH", "ControlFlags"): CONTROL_FLAGS_7000,
    (7503, "RTH", "TransmitFlags"): TRANSMIT_FLAGS_7000,
    (7503, "RTH", "ReceiveFlags"): RECEIVE_FLAGS_7000,
    (7006, "RTH", "Flags"): FLAGS_7006,
    (7006, "Beams", "Quality"): QUALITY_7006,
    (7008, "RTH", "DataSampleType"): DATA_SAMPLE_TYPE_7008,
    (7017, "RTH", "Flags"): FLAGS_7017,
    (7017, "Detections", "Flags"): DETECTION_FLAGS_7017,
    (7027, "RTH", "Flags"): FLAGS_7027,
    (7027, "Detections", "Flags"): DETECTION_FLAGS_7027,
    (7028, "RTH", "Flags"): FLAGS_7028,
    (7042, "RTH", "Flags"): FLAGS_7042,
}
