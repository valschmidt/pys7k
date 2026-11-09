"""
A Kluster-compatible reader built on pys7k.

Kluster reads s7k files through NOAA's ``HSTB.drivers.prr3.X7kRead``. This
module provides ``X7kRead`` with the methods Kluster calls
(``sequential_read_records``, ``fast_read_start_end_time``,
``fast_read_serial_number``, ``has_datagram``,
``return_empty_installparams``) returning the same dictionary structure, so
it can stand in for prr3.

Record choices follow prr3 and the Kluster documentation
(https://kluster.readthedocs.io/en/latest/requirements/index.html):

* Pings come from 7027 Raw Detection Data.
* Runtime settings come from 7503, or from 7000 when the file has no 7503.
  One of them is required.
* Attitude comes from 1012 (roll, pitch, heave) with 1013 (heading), or from
  1016 when 1012 and 1013 are absent.
* Navigation comes from 1003. When reading a chunk (``end_ptr``), the run of
  1003 records that follows the chunk is also read, because 7k systems log
  1003 in blocks rather than with each ping.
* Installation parameters come from 7030, or are built from the 7503 Tx
  offsets when there is no 7030.
* Serial numbers come from the 7001 device descriptions. The model comes
  from the 7000/7027 frame DeviceIdentifier.
* Sound velocity profiles come from 1009.

Conversions applied for Kluster (kept from prr3, because Kluster expects
them):

* Angles in degrees. Latitude and longitude in degrees.
* ``beampointingangle`` = -RxAngle (prr3: "beam angles are always negative
  port up, contrary to the rest of the +port up convention").
* ``traveltime`` = DetectionPoint / SamplingRate.
* Heave negated (Kluster wants positive down). Heading taken modulo 360.
* ``reflectivity`` = 20 log10(Intensity), dropped when every value is zero.
* ``detectioninfo``: 0 amplitude, 1 phase, 2 rejected (padding beams).
* 7030 offsets in the Kongsberg convention: along = Y, athwart = X,
  vertical = -Z.
* 7503 ReceiveBeamWidth: half the Tx beam width when no 7000 gives the real
  value (prr3's assumption, doubled back for the T20).

Corrections relative to prr3 (each changes Kluster's input):

1. ``tiltangle`` (7027 TxAngle) is converted to degrees. prr3 left it in
   radians while every other angle was in degrees.
2. 7030 ``transducer_1_heading_angle`` is converted to degrees. prr3
   converted every other 7030 angle but not this one.
3. 7030 time delays are stored in milliseconds (Table 78) and are divided
   by 1000 to give the seconds Kluster expects. prr3 multiplied by 1000.
4. Runtime settings are matched to pings by ping number, falling back to
   the most recent 7503/7000 at or before the ping. prr3 used the next
   record at or after the ping time.
5. ``ReceiveBeamWidth`` is taken from the matching 7000 record when the
   file has one, rather than assumed to be half the Tx beam width.
6. 1009 sound velocity profiles are read. prr3's 1009 mapping did not match
   its decoder.
7. Heading for each 1012 time is the nearest 1013 sample in time. prr3 took
   the next sample.
8. Zero intensities give a NaN reflectivity instead of -inf.
9. No artificial +10 microsecond shift is applied to the first ping time.
   Duplicate times are still removed and records sorted.
"""

import numpy as np

from .bitfields import CONTROL_FLAGS_7000, RECEIVE_FLAGS_7000, decode_bits
from .derived import decode_text, gather_header, read_svp
from .registry import RECORD_CODECS, decode_record
from .s7k import s7k

__all__ = ["X7kRead", "KLUSTER_DEVICE_IDENTIFIERS", "translate_yawpitch", "translate_mode",
           "translate_mode_two", "translate_detectioninfo"]

# prr3's model-name lookup (Appendix B, short names as Kluster uses them).
KLUSTER_DEVICE_IDENTIFIERS = {
    20: 'T20', 22: 'T20Dual', 30: 'F30', 50: 'T50', 51: 'T51', 52: 'T50Dual', 103: 'GenericMBES',
    1000: 'OdomMB1', 1002: 'OdomMB2', 4013: 'TC4013', 7003: 'PDS', 7005: 'ProScan', 7012: '7012', 7100: '7100',
    7101: '7101', 7102: '7102', 7111: '7111', 7112: '7112', 7123: '7123', 7125: '7125', 7128: '7128',
    7130: '7130', 7150: '7150', 7160: '7160', 8100: '8100', 8101: '8101', 8102: '8102', 8111: '8111',
    8123: '8123', 8124: '8124', 8125: '8125', 8128: '8128', 8150: '8150', 8160: '8160', 9000: 'E20',
    9001: 'Deso5', 9002: 'Deso5DS', 10000: 'DMS05', 10001: '335B', 10002: '332B', 10010: 'SBE37',
    10200: 'Litton200', 11000: 'FS-DW-SBP', 11001: 'FS-DW-LFSSS', 11002: 'FS-DW-HFSSS',
    12000: 'RPT319', 13002: 'NorbitFLS', 13003: 'NorbitBathy', 13004: 'NorbitiWBMS', 13005: 'NorbitBathyCompact',
    13007: 'NorbitBathy', 13008: 'NorbitBathy', 13009: 'NorbitDeepSea', 13010: 'NorbitDeepSea',
    13011: 'NorbitDeepSea', 13012: 'NorbitiLidar', 13016: 'NorbitBathySTX', 13017: 'NorbitBathySTX',
    13018: 'NorbitiWBMSe', 14000: 'HydroSweep3DS', 14001: 'HydroSweep3MD50', 14002: 'HydroSweep3MD30',
}

# pys7k field name -> prr3/Kluster runtime_settings key.
_SETTINGS_KEYS = {
    'SonarId': 'sonar_id', 'SampleRate': 'sample_rate_hertz',
    'ReceiverBandwidth': 'receiver_bandwidth_3db_hertz', 'TxPulseWidth': 'tx_pulse_width_seconds',
    'TxPulseTypeIdentifier': 'tx_pulse_type_id', 'TxPulseEnvelopeIdentifier': 'tx_pulse_envelope_identifier',
    'TxPulseEnvelopeParameter': 'tx_pulse_envelope_parameter', 'TxPulseMode': 'tx_single_multiping_mode',
    'MaxPingRate': 'maximum_ping_rate_per_second', 'PingPeriod': 'ping_period_seconds',
    'RangeSelection': 'range_selection_meters', 'PowerSelection': 'power_selection_db_re_1micropascal',
    'GainSelection': 'gain_selection_db', 'ProjectorIdentifier': 'projector_selection',
    'ProjectorBeamWeightingWindowType': 'projector_weighting_window_type',
    'HydrophoneIdentifier': 'hydrophone_identifier', 'ReceiveBeamWeightingWindow': 'receiver_weighting_window_type',
    'ReceiveBeamWidth': 'ReceiverBeamwidthHorizontal',
    'BottomDetectionFilterMinRange': 'bottom_detect_filter_min_range',
    'BottomDetectionFilterMaxRange': 'bottom_detect_filter_max_range',
    'BottomDetectionFilterMinDepth': 'bottom_detect_filter_min_depth',
    'BottomDetectionFilterMaxDepth': 'bottom_detect_filter_max_depth',
    'Absorption': 'absorption_db_km', 'SoundVelocity': 'sound_velocity_meters_per_sec',
    'Spreading': 'spreading_loss_db', 'VernierOperationMode': 'vernier_operation_mode',
    'AutomaticFilterWindow': 'automatic_filter_window_size_percent_depth',
    'HeadTiltX': 'head_tilt_acrosstrack', 'HeadTiltY': 'head_tilt_alongtrack',
    'PingState': 'ping_state', 'BeamSpacingMode': 'beam_spacing_mode', 'SonarSourceMode': 'sonar_source_mode',
    'AdaptiveGateBottomFilterMinDepth': 'adaptive_gate_min_depth',
    'AdaptiveGateBottomFilterMaxDepth': 'adaptive_gate_max_depth',
    'TriggerOutWidth': 'trigger_out_width', 'TriggerOutOffset': 'trigger_out_offset',
    'Series81xxProjectorSelection': 'eight_series_projector_selection',
    'Series81xxAlternateGain': 'eight_series_alternate_gain_db', 'VernierFilter': 'vernier_filter_settings',
    'CustomBeams': 'custom_beams', 'CoverageAngle': 'coverage_angle_radians', 'CoverageMode': 'coverage_mode',
    'QualityFilterFlags': 'quality_filter_enabled',
    'HorizontalReceiverBeamSteeringAngle': 'receiver_beam_steering_angle_radians',
    'FlexModeSectorCoverage': 'flexmode_coverage_sector_radians',
    'FlexModeSectorSteering': 'flexmode_steering_angle_radians',
    'ConstantSpacing': 'constant_beam_spacing_meters', 'BeamModeSelection': 'sonar_xml_beam_mode_index',
    'DepthGateTilt': 'depth_gate_tilt_angle_radians', 'AppliedFrequency': 'transmit_frequency_slider',
}
_SETTINGS_VALUES = {
    'TxPulseTypeIdentifier': {0: 'CW', 1: 'FM'},
    'TxPulseEnvelopeIdentifier': {0: 'tapered_rectangular', 1: 'tukey', 2: 'hamming', 3: 'han', 4: 'rectangular'},
    'TxPulseMode': {1: 'single_ping', 2: 'multiping2', 3: 'multiping3', 4: 'multiping4'},
    'ProjectorBeamWeightingWindowType': {0: 'rectangular', 1: 'chebychev'},
    'ReceiveBeamWeightingWindow': {0: 'chebychev', 1: 'kaiser'},
    'PingState': {0: 'disabled', 1: 'enabled', 2: 'externally_triggered'},
    'BeamSpacingMode': {1: 'equiangle', 2: 'equidistant', 3: 'flex', 4: 'intermediate'},
    'SonarSourceMode': {0: 'normal', 1: 'autopilot', 2: 'calibration'},
    'Series81xxProjectorSelection': {0: 'stick', 1: 'main_array', 2: 'extended_range'},
    'CoverageMode': {0: 'reduce_spacing', 1: 'reduce_beams'},
    'QualityFilterFlags': {0: 'disabled', 1: 'enabled'},
}
_CONTROL_KEYS = {
    'AutoRangeMethod': 'auto_range_method',
    'AutoBottomDetectionFilterMethod': 'auto_bottom_detection_filter_method',
    'BottomDetectionRangeFilterEnabled': 'bottom_detection_range_filter_enabled',
    'BottomDetectionDepthFilterEnabled': 'bottom_detection_depth_filter_enabled',
    'ReceiverGainAutoGain': 'receiver_gain_autogain', 'ReceiverGainFixedGain': 'receiver_gain_fixedgain',
    'TriggerOutHighForRxDuration': 'trigger_out_high', 'SystemActive': 'system_active',
    'PipeGatingFilter': 'pipe_gating_filter_enabled',
    'AdaptiveGateDepthFilterFixed': 'adaptive_gate_depth_filter_fixed', 'AdaptiveGate': 'adaptive_gate_enabled',
    'AdaptiveGateDepthFilter': 'adaptive_gate_depth_filter_enabled', 'TriggerOut': 'trigger_out_enabled',
    'TriggerInEdgeNegative': 'trigger_in_edge_negative', 'PPSEdgeNegative': 'pps_edge_negative',
    'DepthFilterFollowsSeafloor': 'depth_filter_follow_seafloor',
    'ReducedCoverageForConstantSpacing': 'reduced_coverage_for_constant_spacing', 'Simulator': 'is_simulator',
}
_RECEIVE_KEYS = {
    'RollCompensation': 'roll_compensation_indicator', 'HeaveCompensation': 'heave_compensation_indicator',
    'DynamicFocusingMethod': 'dynamic_focusing_method', 'DopplerCompensationMethod': 'doppler_compensation_method',
    'MatchFilteringMethod': 'match_filtering_method', 'TVGMethod': 'tvg_method',
    'MultiPingMode': 'multi_ping_number_of_pings',
}
_METHOD_FIELDS = {'AutoRangeMethod', 'AutoBottomDetectionFilterMethod', 'DynamicFocusingMethod',
                  'DopplerCompensationMethod', 'MatchFilteringMethod', 'TVGMethod', 'MultiPingMode'}


def translate_yawpitch(arr):
    """TransmitFlags to 'N', 'P', 'Y' or 'PY' (prr3)."""
    arr = np.asarray(arr, dtype=np.int64)
    rslt = np.full(arr.shape, 'N', dtype='U2')
    pstab = (arr & 15).astype(bool)
    ystab = (arr & 240).astype(bool)
    rslt[pstab] = 'P'
    rslt[ystab] = 'Y'
    rslt[pstab & ystab] = 'PY'
    return rslt


def translate_mode(arr):
    """TxPulseTypeIdentifier to 'CW', 'FM' or 'MP' (prr3)."""
    arr = np.asarray(arr)
    rslt = np.full(arr.shape, '', dtype='U2')
    rslt[arr == 0] = 'CW'
    rslt[arr == 1] = 'FM'
    rslt[(arr == 2) | (arr == 3)] = 'MP'
    return rslt


def translate_mode_two(arr):
    """BeamSpacingMode to 'EqAn', 'EqDs', 'Flex' or 'Intr' (prr3)."""
    arr = np.asarray(arr)
    rslt = np.full(arr.shape, '', dtype='U4')
    for code, name in ((1, 'EqAn'), (2, 'EqDs'), (3, 'Flex'), (4, 'Intr')):
        rslt[arr == code] = name
    return rslt


def translate_detectioninfo(arr):
    """7027 detection Flags to 0 (amplitude) or 1 (phase); amplitude wins
    when both bits are set (prr3)."""
    arr = np.asarray(arr, dtype=np.int64)
    rslt = np.zeros(arr.shape, dtype=int)
    rslt[(arr & 2).astype(bool)] = 1
    rslt[(arr & 1).astype(bool)] = 0
    return rslt


def _fmt(value):
    value = float(value)
    return '0.000' if value == 0 else "{:.3f}".format(round(value, 3))


def _serial_from_description(description):
    """prr3's rule: strip a leading 'SN' and keep the leading digits."""
    text = decode_text(description) if isinstance(description, (bytes, bytearray)) else str(description)
    if text[:2] == 'SN':
        text = text[2:]
    digits = ''
    for ch in text:
        if not ch.isdigit():
            break
        digits += ch
    return int(digits) if digits else 0


def _settings_dict(rth, record_type, rx_beamwidth=None):
    """The prr3 ``full_settings`` dictionary for a 7503 or 7000 header."""
    settings = {}
    if record_type == 7503:
        if rth.get('TxArrayPositionOffsetY') is not None:
            settings['transducer_1_along_refpt_offset'] = '0.0'
            settings['transducer_2_along_refpt_offset'] = str(round(-float(rth['TxArrayPositionOffsetY']), 3))
        if rth.get('TxArrayPositionOffsetX') is not None:
            settings['transducer_1_athwart_refpt_offset'] = str(round(float(rth['TxArrayPositionOffsetX']), 3))
            settings['transducer_2_athwart_refpt_offset'] = '0.000'
        if rth.get('TxArrayPositionOffsetZ') is not None:
            settings['transducer_1_vertical_refpt_offset'] = str(round(-float(rth['TxArrayPositionOffsetZ']), 3))
            settings['transducer_2_vertical_refpt_offset'] = '0.000'
    tx_bw = float(np.rad2deg(rth['ProjectorBeamWidthVertical']))
    settings['TransmitBeamWidth'] = str(round(tx_bw, 3))
    if record_type == 7000:
        settings['ReceiveBeamWidth'] = str(round(float(np.rad2deg(rth['ReceiveBeamWidth'])), 3))
    elif rx_beamwidth is not None:
        settings['ReceiveBeamWidth'] = str(round(float(np.rad2deg(rx_beamwidth)), 3))
    else:
        settings['ReceiveBeamWidth'] = str(round(tx_bw / 2, 3))
    # prr3 also reports bit 19 (within "reserved for bottom detection").
    settings['adaptive_search_window_passive'] = str(bool((int(rth['ControlFlags']) >> 19) & 1))
    for k, v in decode_bits(rth['ControlFlags'], CONTROL_FLAGS_7000).items():
        if k == 'TimestampState':
            settings['timestamp_state_ok'] = str(v == 3)
        elif k in _CONTROL_KEYS:
            settings[_CONTROL_KEYS[k]] = str(v) if k in _METHOD_FIELDS else str(bool(v))
    for k, v in decode_bits(rth['ReceiveFlags'], RECEIVE_FLAGS_7000).items():
        settings[_RECEIVE_KEYS[k]] = str(v) if k in _METHOD_FIELDS else str(bool(v))
    codes = dict(RECORD_CODECS[record_type].rth.fields)
    for name, value in rth.items():
        if value is None or name.startswith('Reserved'):
            continue
        if name in _SETTINGS_VALUES:
            value = _SETTINGS_VALUES[name].get(value, value)
        elif codes.get(name) == 'f32':
            value = np.float32(value)      # prints as prr3 does ('34482.758')
        settings[_SETTINGS_KEYS.get(name, name)] = str(value)
    return settings


def _install_settings_7030(rth, device_id):
    """prr3's ``translated_settings`` for a 7030 header, with the three
    corrections noted in the module docstring."""
    def text(name):
        n = rth.get(name + 'Length') or 0
        return decode_text(rth[name][:n]) if rth.get(name) is not None else ''
    deg = lambda v: _fmt(np.rad2deg(v))  # noqa: E731
    return {
        'sonar_model_number': KLUSTER_DEVICE_IDENTIFIERS.get(device_id, 'Unknown'),
        'transducer_1_vertical_location': _fmt(-rth['TransmitArrayZ']),
        'transducer_1_along_location': _fmt(rth['TransmitArrayY']),
        'transducer_1_athwart_location': _fmt(rth['TransmitArrayX']),
        'transducer_1_heading_angle': deg(rth['TransmitArrayHeading']),
        'transducer_1_roll_angle': deg(rth['TransmitArrayRoll']),
        'transducer_1_pitch_angle': deg(rth['TransmitArrayPitch']),
        'transducer_2_vertical_location': _fmt(-rth['ReceiveArrayZ']),
        'transducer_2_along_location': _fmt(rth['ReceiveArrayY']),
        'transducer_2_athwart_location': _fmt(rth['ReceiveArrayX']),
        'transducer_2_heading_angle': deg(rth['ReceiveArrayHeading']),
        'transducer_2_roll_angle': deg(rth['ReceiveArrayRoll']),
        'transducer_2_pitch_angle': deg(rth['ReceiveArrayPitch']),
        'position_1_time_delay': _fmt(rth['PositionSensorTimeDelay'] / 1000.0),
        'position_1_vertical_location': _fmt(-rth['PositionSensorZ']),
        'position_1_along_location': _fmt(rth['PositionSensorY']),
        'position_1_athwart_location': _fmt(rth['PositionSensorX']),
        'motion_sensor_1_time_delay': _fmt(rth['MotionSensorTimeDelay'] / 1000.0),
        'motion_sensor_1_vertical_location': _fmt(-rth['MotionSensorZ']),
        'motion_sensor_1_along_location': _fmt(rth['MotionSensorY']),
        'motion_sensor_1_athwart_location': _fmt(rth['MotionSensorX']),
        'motion_sensor_1_roll_angle': deg(rth['MotionSensorRollCalibration']),
        'motion_sensor_1_pitch_angle': deg(rth['MotionSensorPitchCalibration']),
        'motion_sensor_1_heading_angle': deg(rth['MotionSensorHeadingCalibration']),
        'waterline_vertical_location': _fmt(-rth['WaterLineVerticalOffset']),
        'system_main_head_serial_number': '0', 'active_position_system_number': '1',
        'active_heading_sensor': 'motion_1', 'position_1_datum': 'WGS84',
        'tx_serial_number': '0', 'tx_2_serial_number': '0',
        'firmware_version': text('FirmwareVersion'), 'software_version': text('SoftwareVersion'),
        'sevenk_version': text('SevenKSoftwareVersion'), 'protocol_version': text('RecordProtocolVersion'),
    }


class X7kRead:
    """
    A prr3-compatible reader for Kluster.

    :param infilename: the s7k file.
    :param start_ptr: read records starting at or after this byte offset.
    :param end_ptr: read records starting before this byte offset (0 or
        None for the end of the file).
    """

    def __init__(self, infilename, start_ptr=0, end_ptr=0):
        self.infilename = infilename
        self.start_ptr = int(start_ptr or 0)
        self.end_ptr = int(end_ptr or 0)
        self.S = s7k(infilename)
        self.S.index_file(full=True)
        self.Index = self.S.Index
        self.max_filelen = self.S.file_size

    def close(self):
        self.S.closeFile()

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.close()

    # -- quick queries ------------------------------------------------------

    def has_datagram(self, datagramnum, max_records=50):
        """True if every given record type occurs within the first
        ``max_records`` records (all records if ``max_records`` is falsy)."""
        types = datagramnum if isinstance(datagramnum, (list, tuple, set)) else [datagramnum]
        head = self.Index.RecordType if not max_records else self.Index.RecordType.iloc[:max_records]
        present = set(head.tolist())
        return all(int(t) in present for t in types)

    def fast_read_start_end_time(self, only_start=False):
        """[start, end] POSIX times of the first and last 1003, 1012, 1013,
        1016, 7000, 7027, 7030 or 7503 record."""
        idx = self.Index[self.Index.RecordType.isin([1003, 1012, 1013, 1016, 7000, 7027, 7030, 7503])]
        if not len(idx):
            raise ValueError('pys7k: unable to find a suitable record to read the start time of the file')
        start = float(idx.Time.iloc[0])
        if only_start:
            return start, None
        return [start, float(idx.Time.iloc[-1])]

    def _serials(self):
        rows = self.Index.index[self.Index.RecordType == 7001]
        if not len(rows):
            return 0, 0
        devices = self.S.read_record(rows[0])["RD"]["Devices"]
        serials = [_serial_from_description(d["Description"]) for d in devices[:2]]
        serials += [0] * (2 - len(serials))
        return serials[0], serials[1]

    def _model(self):
        rows = self.Index[self.Index.RecordType.isin([7000, 7027])]
        if not len(rows):
            return ''
        return KLUSTER_DEVICE_IDENTIFIERS.get(int(rows.DeviceId.iloc[0]), '')

    def fast_read_serial_number(self):
        """[serial_one, serial_two, model] from 7001 and the 7000/7027 frame."""
        one, two = self._serials()
        return [one, two, self._model()]

    def return_empty_installparams(self, model_num='', serial_num_one=0, serial_num_two=0, runtime=None):
        """Installation parameters built from 7503 offsets when there is no
        7030 (all other values zero), as prr3 does."""
        starttime, _ = self.fast_read_start_end_time(only_start=True)
        keys = ['transducer_1_vertical_location', 'transducer_1_along_location', 'transducer_1_athwart_location',
                'transducer_1_heading_angle', 'transducer_1_roll_angle', 'transducer_1_pitch_angle',
                'transducer_2_vertical_location', 'transducer_2_along_location', 'transducer_2_athwart_location',
                'transducer_2_heading_angle', 'transducer_2_roll_angle', 'transducer_2_pitch_angle',
                'position_1_time_delay', 'position_1_vertical_location', 'position_1_along_location',
                'position_1_athwart_location', 'motion_sensor_1_time_delay', 'motion_sensor_1_vertical_location',
                'motion_sensor_1_along_location', 'motion_sensor_1_athwart_location', 'motion_sensor_1_roll_angle',
                'motion_sensor_1_pitch_angle', 'motion_sensor_1_heading_angle', 'waterline_vertical_location']
        isets = {'sonar_model_number': model_num}
        isets.update({k: '0.000' for k in keys})
        isets.update({'system_main_head_serial_number': '0', 'tx_serial_number': '0', 'tx_2_serial_number': '0',
                      'firmware_version': '', 'active_position_system_number': '1',
                      'active_heading_sensor': 'motion_1', 'position_1_datum': 'WGS84',
                      'software_version': '', 'sevenk_version': '', 'protocol_version': ''})
        if runtime and 'transducer_1_along_refpt_offset' in runtime:
            for t in ('transducer_1', 'transducer_2'):
                for axis in ('along', 'athwart', 'vertical'):
                    isets['%s_%s_location' % (t, axis)] = runtime['%s_%s_refpt_offset' % (t, axis)]
        return {'installation_params': {
            'time': np.array([starttime], dtype=float),
            'serial_one': np.array([serial_num_one], dtype=np.uint64),
            'serial_two': np.array([serial_num_two], dtype=np.uint64),
            'installation_settings': np.array([isets], dtype=object)}}

    # -- the main read ------------------------------------------------------

    def _chunk_rows(self):
        """Index rows in [start_ptr, end_ptr), plus the 1003 block after."""
        idx = self.Index
        end = self.end_ptr if self.end_ptr else self.max_filelen + 1
        inside = (idx.Offset >= self.start_ptr) & (idx.Offset < end)
        rows = idx.index[inside.to_numpy()]
        if self.end_ptr and self.end_ptr < self.max_filelen:
            after = idx[idx.Offset >= end]
            is_nav = (after.RecordType == 1003).to_numpy()
            if is_nav.any():
                first = int(np.argmax(is_nav))
                stop = first
                while stop < len(is_nav) and is_nav[stop]:
                    stop += 1
                rows = rows.append(after.index[first:stop])
        return rows

    def sequential_read_records(self, first_installation_rec=False):
        """
        Read the chunk and return prr3's dictionary of 'attitude',
        'installation_params', 'ping', 'runtime_params', 'profile' and
        'navigation' arrays, plus 'format': 's7k'.
        """
        serial_one, serial_two = self._serials()
        model = self._model()
        types_head = set(self.Index.RecordType.iloc[:300].tolist())
        if {1012, 1013} <= types_head:
            use_1016 = False
        elif 1016 in types_head:
            use_1016 = True
        else:
            raise ValueError('pys7k: unable to find either 1012/1013 or 1016 attitude records')
        settings_type = 7503 if 7503 in types_head else 7000
        if not (self.Index.RecordType == settings_type).any():
            raise ValueError('pys7k: runtime parameters record 7503 or 7000 is required, but not found in this file.')

        rows = self._chunk_rows()
        sub = self.Index.loc[rows]
        has_7030 = bool((self.Index.RecordType.iloc[:20] == 7030).any())

        if first_installation_rec:
            r7030 = self.Index.index[self.Index.RecordType == 7030]
            if len(r7030):
                return {'installation_params': self._installation(r7030[:1], serial_one, serial_two)}
            r_set = self.Index.index[self.Index.RecordType == settings_type]
            runtime = _settings_dict(self.S.read_record(r_set[0])["RTH"], settings_type)
            return self.return_empty_installparams(model, serial_one, serial_two, runtime=runtime)

        out = {
            'attitude': self._attitude(sub, use_1016),
            'navigation': self._navigation(sub),
            'profile': self._profile(sub),
        }
        ping, runtime = self._ping_and_runtime(sub, settings_type, model, serial_one)
        out['ping'] = ping
        out['runtime_params'] = runtime
        r7030 = sub.index[sub.RecordType == 7030]
        if len(r7030):
            out['installation_params'] = self._installation(r7030, serial_one, serial_two)
        else:
            rs = runtime['runtime_settings'][0] if len(runtime['runtime_settings']) else None
            if has_7030:
                out['installation_params'] = {'time': np.zeros(0), 'serial_one': np.zeros(0),
                                              'serial_two': np.zeros(0), 'installation_settings': np.zeros(0)}
            else:
                out['installation_params'] = self.return_empty_installparams(
                    model, serial_one, serial_two, runtime=rs)['installation_params']
        for name in ('attitude', 'navigation', 'ping'):
            out[name] = _unique_sorted(out[name])
        out['format'] = 's7k'
        return out

    def _installation(self, rows, serial_one, serial_two):
        recs = [self.S.read_record(r) for r in rows]
        return {
            'time': self.Index.Time.loc[rows].to_numpy(dtype=float),
            'serial_one': np.full(len(recs), serial_one, dtype=np.uint64),
            'serial_two': np.full(len(recs), serial_two, dtype=np.uint64),
            'installation_settings': np.array(
                [_install_settings_7030(r["RTH"], r["Frame"]["DeviceIdentifier"]) for r in recs], dtype=object),
        }

    def _gather(self, record_type, rows):
        g = gather_header(self.S, record_type)
        keep = np.isin(g["Record"], np.asarray(rows))
        return {k: v[keep] for k, v in g.items()}

    def _attitude(self, sub, use_1016):
        if use_1016:
            rows = sub.index[sub.RecordType == 1016]
            parts = {k: [] for k in ('time', 'roll', 'pitch', 'heave', 'heading')}
            for r in rows:
                rec = self.S.read_record(r)
                s = rec["RD"]["Samples"]
                parts['time'].append(float(self.Index.Time[r]) + np.asarray(s["TimeDifference"], float) / 1000.0)
                parts['roll'].append(np.rad2deg(np.asarray(s["Roll"], float)))
                parts['pitch'].append(np.rad2deg(np.asarray(s["Pitch"], float)))
                parts['heave'].append(np.asarray(s["Heave"], float))
                parts['heading'].append(np.rad2deg(np.asarray(s["Heading"], float)))
            att = {k: (np.concatenate(v) if v else np.zeros(0)) for k, v in parts.items()}
        else:
            g12 = self._gather(1012, sub.index[sub.RecordType == 1012])
            g13 = self._gather(1013, sub.index[sub.RecordType == 1013])
            att = {'time': g12["Time"], 'roll': np.rad2deg(g12["Roll"].astype(float)),
                   'pitch': np.rad2deg(g12["Pitch"].astype(float)), 'heave': g12["Heave"].astype(float)}
            heading = np.rad2deg(g13["Heading"].astype(float))
            if len(heading) and len(att['time']):
                if len(heading) == len(att['time']):
                    att['heading'] = heading
                else:
                    att['heading'] = heading[_nearest(g13["Time"], att['time'])]
            else:
                att['heading'] = np.full(len(att['time']), np.nan)
        att['heave'] = att['heave'] * -1
        att['heading'] = att['heading'] % 360
        return att

    def _navigation(self, sub):
        g = self._gather(1003, sub.index[sub.RecordType == 1003])
        nav = {'time': g["Time"], 'latitude': np.rad2deg(g["LatitudeOrNorthing"]),
               'longitude': np.rad2deg(g["LongitudeOrEasting"])}
        if len(g["Time"]):
            nav['altitude'] = g["HeightRelativeToDatumOrHeight"].astype(np.float32)
        return nav

    def _profile(self, sub):
        if not (sub.RecordType == 1009).any():
            return {'time': np.zeros(0), 'depth': np.zeros(0), 'soundspeed': np.zeros(0)}
        profiles = [p for p, r in zip(read_svp(self.S), self.Index.index[self.Index.RecordType == 1009])
                    if r in set(sub.index)]
        return {'time': np.array([p['Time'] for p in profiles], dtype=float),
                'depth': np.array([p['Depth'] for p in profiles], dtype=object),
                'soundspeed': np.array([p['SoundVelocity'] for p in profiles], dtype=object)}

    def _ping_and_runtime(self, sub, settings_type, model, serial_one):
        rows = sub.index[sub.RecordType == 7027]
        recs = [decode_record(self.S.read_record_bytes(r)) for r in rows]
        times = self.Index.Time.loc[rows].to_numpy(dtype=float)
        counters = np.array([r["RTH"]["PingNumber"] for r in recs], dtype=np.int64)
        dets = [r["RD"]["Detections"] for r in recs]
        nmax = max((len(d["BeamDescriptor"]) for d in dets), default=0)
        P = len(recs)

        def pad(values, fill, dtype):
            arr = np.full((P, nmax), fill, dtype=dtype)
            for i, v in enumerate(values):
                arr[i, :len(v)] = v
            return arr

        valid = pad([np.ones(len(d["BeamDescriptor"]), bool) for d in dets], False, bool)
        traveltime = pad([np.asarray(d["DetectionPoint"], np.float64) / r["RTH"]["SamplingRate"]
                          for d, r in zip(dets, recs)], np.nan, np.float32)
        rx = pad([-np.rad2deg(np.asarray(d["RxAngle"], np.float64)) for d in dets], 0, np.float32)
        tilt = pad([np.full(len(d["BeamDescriptor"]), np.rad2deg(r["RTH"]["TxAngle"])) for d, r in zip(dets, recs)],
                   0, np.float32)
        quality = pad([np.asarray(d["Uncertainty"]) for d in dets], 0, np.float32)
        flags = pad([np.asarray(d["Flags"]) for d in dets], 0, np.uint32)
        intensity = pad([np.asarray(d["Intensity"]) if "Intensity" in d else np.zeros(len(d["BeamDescriptor"]))
                         for d in dets], 0, np.float64)

        # Runtime settings for every ping: by ping number, else the latest
        # settings record at or before the ping time.
        set_rows_all = self.Index.index[self.Index.RecordType == settings_type]
        set_rows = sub.index[sub.RecordType == settings_type]
        if not len(set_rows):
            set_rows = set_rows_all[:1]
        g = self._gather(settings_type, set_rows)
        set_times = g["Time"]
        set_pings = g["PingNumber"].astype(np.int64)
        by_ping = {int(p): i for i, p in enumerate(set_pings)}
        prev = np.clip(np.searchsorted(set_times, times, side='right') - 1, 0, max(len(set_times) - 1, 0))
        rindex = np.array([by_ping.get(int(c), int(j)) for c, j in zip(counters, prev)], dtype=np.int64)

        rx_beamwidth = None
        if settings_type == 7503:
            g7000 = gather_header(self.S, 7000)
            if len(g7000["Time"]):
                rx_beamwidth = float(g7000["ReceiveBeamWidth"][0])
        first_rth = self.S.read_record(set_rows[0])["RTH"]
        full = _settings_dict(first_rth, settings_type, rx_beamwidth)
        if model == 'T20' and settings_type == 7503 and rx_beamwidth is None:
            full['ReceiveBeamWidth'] = str(float(full['ReceiveBeamWidth']) * 2)

        def take(arr):
            return arr[rindex] if len(rindex) else arr[:0]

        beam_spacing = g["BeamSpacingMode"] if "BeamSpacingMode" in g else np.zeros(len(set_times), np.uint16)
        ping = {
            'time': times,
            'counter': counters,
            'tiltangle': tilt,
            'beampointingangle': rx,
            'traveltime': traveltime,
            'qualityfactor': quality,
            'detectioninfo': np.where(valid, translate_detectioninfo(flags), 2),
            'txsector_beam': np.zeros((P, nmax), dtype=np.uint8),
            'frequency': (np.ones((P, nmax)) * take(g["Frequency"].astype(np.float64))[:, None]).astype(np.int32),
            'soundspeed': take(g["SoundVelocity"].astype(np.float64)),
            'mode': take(translate_mode(g["TxPulseTypeIdentifier"])),
            'modetwo': take(translate_mode_two(beam_spacing)),
            'yawpitchstab': take(translate_yawpitch(g["TransmitFlags"])),
            'serial_num': np.full(P, serial_one, dtype=np.uint64),
            'delay': np.zeros((P, nmax), dtype=np.float32),
            'processing_status': np.zeros((P, nmax), dtype=np.uint8),
        }
        if P and (intensity[valid] != 0).any():
            with np.errstate(divide='ignore', invalid='ignore'):
                refl = 20 * np.log10(intensity)
            refl[~np.isfinite(refl)] = np.nan
            ping['reflectivity'] = refl.astype(np.float32)
        runtime = {
            'time': np.array([set_times[0]], dtype=float) if len(set_times) else np.zeros(0),
            'runtime_settings': np.array([full], dtype=object),
        }
        return ping, runtime


def _nearest(source_times, target_times):
    """Index of the nearest source time for each target time."""
    source_times = np.asarray(source_times)
    pos = np.clip(np.searchsorted(source_times, target_times), 1, len(source_times) - 1) if len(source_times) > 1 \
        else np.zeros(len(target_times), dtype=np.int64)
    if len(source_times) <= 1:
        return pos
    left = source_times[pos - 1]
    right = source_times[pos]
    return np.where(np.abs(target_times - left) <= np.abs(right - target_times), pos - 1, pos)


def _unique_sorted(dset):
    """Drop records with duplicate times and sort by time (as prr3)."""
    if 'time' not in dset or not len(dset['time']):
        return dset
    _, index = np.unique(dset['time'], return_index=True)
    if index.size != dset['time'].size or not np.all(np.diff(dset['time']) >= 0):
        index = np.sort(index)
        order = index[np.argsort(dset['time'][index], kind='stable')]
        dset = {k: (v[order] if hasattr(v, '__len__') and len(v) == len(dset['time']) else v)
                for k, v in dset.items()}
    return dset
