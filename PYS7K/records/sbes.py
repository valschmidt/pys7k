"""
Single-beam echo sounder records (10000-10062), added in phase 3.

No local sample file contains SBES records. Layouts are transcribed from DFD
3.10 Tables 122-132. Header-only records decode the leading fields present,
so the extensions the DFD announces (for example 8 reserved u32 fields to
be added to 10000) round-trip as unparsed bytes until they are defined.
"""

import numpy as np

from ..fields import Section
from .base import Codec, HeaderCodec, HeaderTableCodec

CODECS = []


def _add(codec):
    CODECS.append(codec)
    return codec


# 10000 - SBES Channel Settings (Table 122)
_add(HeaderCodec(10000, "SBES Channel Settings", [
    ("PingNumber", "u32"),
    ("MultiPingSequence", "u32"),
    ("Reserved1", "u32"),
    ("TransportLatency", "u32"),
    ("FrequencyCenter", "f32"),
    ("SweepWidth", "f32"),
    ("SampleRate", "f32"),
    ("ReceiverBandwidth", "f32"),
    ("Absorption", "f32"),
    ("SpreadingLoss", "f32"),
    ("InitialGain", "f32"),
    ("Range", "f32"),
    ("Reserved2", "f32"),
    ("Power", "f32"),
    ("PulseLength", "f32"),
    ("PulseType", "u32"),
    ("PulseEnvelopeType", "u32"),
    ("PulseEnvelopeParameter", "f32"),
    ("MultiPingCount", "u32"),
    ("Reserved3", "u32"),
    ("MaxPingRate", "f32"),
    ("PingPeriod", "f32"),
    ("RxMinGate", "f32"),
    ("ProjectorId", "u32"),
    ("TransmitBeamWidth", "f32"),
    ("TransmitBeamWidthAlong", "f32"),
    ("ReceiverId", "u32"),
    ("ReceiverBeamWidth", "f32"),
    ("ReceiverBeamWidthAlong", "f32"),
    ("CorrectedTransducerDraft", "f32"),
    ("AbsorptionTweak", "f32"),
    ("PulseLengthTweak", "f32"),
    ("SpreadingLossTweak", "f32"),
    ("InitialGainTweak", "f32"),
    ("RangeTweak", "f32"),
    ("PowerTweak", "f32"),
], phase=3))

# 10003 - SBES Channel Status (Table 123)
_add(HeaderCodec(10003, "SBES Channel Status", [("PingNumber", "u32"), ("FilteredPingRate", "f32")], phase=3))

# 10004 - SBES System Settings (Table 124)
_add(HeaderCodec(10004, "SBES System Settings", [
    ("Channels", "u32"),
    ("PingPong", "u32"),
    ("MeanSoundVelocity", "f32"),
    ("TriggerOutPolarity", "u32"),
    ("TriggerOutMode", "u32"),
    ("TriggerInEnabled", "u32"),
    ("TriggerInDelay", "f32"),
    ("TriggerInActiveEdge", "u32"),
    ("TriggerOutDuration", "f32"),
    ("TriggerOutDelay", "f32"),
    ("Reserved1", "u32*2"),
    ("TrackerMode", "u32"),
    ("OperationMode", "u32"),
    ("TrackerModes1", "u32"),
    ("Reserved2", "u32"),
], phase=3))


class _Echogram(Codec):
    """
    10018 - SBES Echogram Water Column Data (Tables 125-126). n samples of
    BitsPerSample (8, 16 or 32) bits, packed little-endian into u32 words.
    Padding to a whole word is kept as unparsed bytes.
    """

    record_type = 10018
    name = "SBES Echogram Water Column Data"
    phase = 3
    rth = Section([
        ("PingNumber", "u32"),
        ("MultiPingSequence", "u32"),
        ("MeanSoundVelocity", "f32"),
        ("NotAppliedCorrectedTransducerDraft", "f32"),
        ("SampleRate", "f32"),
        ("EffectivePulseLength", "f32"),
        ("StartSampleDelay", "f32"),
        ("BitsPerSample", "u32"),
        ("FullScale", "u32"),
        ("N", "u32"),
    ])
    _DT = {8: "u1", 16: "<u2", 32: "<u4"}

    def new_rth(self):
        rth = self.rth.new()
        rth["BitsPerSample"] = 16
        return rth

    def new_rd(self):
        return {"Samples": np.zeros(0, "<u2")}

    def decode(self, body):
        rth = self.rth.unpack(body, 0)
        pos = self.rth.size
        dt = self._DT.get(rth["BitsPerSample"])
        if dt is None or pos + rth["N"] * np.dtype(dt).itemsize > len(body):
            return rth, {}, pos
        s = np.frombuffer(body, dt, rth["N"], pos)
        return rth, {"Samples": s}, pos + s.nbytes

    def encode(self, record):
        out = self.rth.pack(record["RTH"])
        if "Samples" in record["RD"]:
            out += np.asarray(record["RD"]["Samples"], self._DT[record["RTH"]["BitsPerSample"]]).tobytes()
        return out


_add(_Echogram())

# 10027 - SBES Raw Detection Data (Tables 127-128; the DFD labels the header
# table "10018").
_add(HeaderTableCodec(10027, "SBES Raw Detection Data", [
    ("PingNumber", "u32"),
    ("MultiPingSequence", "u32"),
    ("N", "u32"),
    ("DataFieldSize", "u32"),
    ("MeanSoundVelocity", "f32"),
    ("SampleRate", "f32"),
    ("Reserved", "f32*4"),
    ("NotAppliedCorrectedTransducerDraft", "f32"),
], "N", [
    ("DetectionMethod", "u32"),
    ("Range", "f32"),
    ("NotAppliedHeave", "f32"),
    ("Amplitude", "f32"),
    ("SNR", "f32"),
    ("SeabedConfidence", "f32"),
    ("CoLinearity", "f32"),
    ("Classifier", "u32"),
], "Detections", itemsize_field="DataFieldSize", phase=3))

# 10049 - SBES Depth Reading (Table 129)
_add(HeaderCodec(10049, "SBES Depth Reading", [("DepthMeters", "f32"), ("DepthFeet", "f32")], phase=3))

# 10060 - SBES Port Configuration (Table 130)
_add(HeaderCodec(10060, "SBES Port Configuration", [
    ("PortIndex", "u32"),
    ("PortType", "u32"),
    ("SystemFunction", "u64"),
    ("PhySetting", "u32"),
    ("BaudRate", "u32"),
    ("DataBits", "u32"),
    ("StopBits", "u32"),
    ("Parity", "u32"),
    ("SentenceIndex0", "u32"),
    ("Reserved", "u32*3"),
    ("LabelOnBox", "c8*32"),
], phase=3))

# 10061 - SBES Port Status (Table 131)
_add(HeaderCodec(10061, "SBES Port Status", [("PortIndex", "u32"), ("CurrentStatus", "u32")], phase=3))

# 10062 - SBES Port QC (Table 132; labelled "10061" in the DFD)
_add(HeaderCodec(10062, "SBES Port QC", [
    ("PortIndex", "u32"),
    ("ByteCount", "u32"),
    ("MessageCount", "u32"),
    ("ErrorCount", "u32"),
    ("CurrentStatus", "u32"),
    ("SentenceStartTime", "u64"),
    ("SentenceEndTime", "u64"),
    ("SentenceByteCount", "u32"),
    ("SentenceDirection", "u32"),
    ("SentenceContent", "c8*512"),
], phase=3))
