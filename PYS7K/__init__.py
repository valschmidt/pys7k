"""
pys7k: a pure-Python reader and writer for Teledyne RESON s7k sonar files
(7k Data Format Definition 3.10).
"""

from .frame import (FLAG_CHECKSUM_VALID, FLAG_RECORDED_DATA, S7KChecksumError,
                    S7KError, S7KFormatError, checksum, epoch_to_time7k,
                    time7k_to_epoch)
from .registry import (DEVICE_IDENTIFIERS, RECORD_CODECS,
                       RECORD_TYPE_DESCRIPTIONS, decode_record, encode_record,
                       new, record_name)
from .s7k import main, resolve_record_type, s7k

__version__ = "0.1.0"

__all__ = [
    "s7k", "main", "new", "decode_record", "encode_record", "record_name",
    "resolve_record_type", "checksum", "time7k_to_epoch", "epoch_to_time7k",
    "RECORD_CODECS", "RECORD_TYPE_DESCRIPTIONS", "DEVICE_IDENTIFIERS",
    "FLAG_CHECKSUM_VALID", "FLAG_RECORDED_DATA",
    "S7KError", "S7KFormatError", "S7KChecksumError",
]
