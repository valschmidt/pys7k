"""Record codecs, grouped by record family."""

from . import sbes, sensors, sonar, status, system, watercolumn

__all__ = ["ALL_CODECS"]

ALL_CODECS = (sensors.CODECS + sonar.CODECS + watercolumn.CODECS + system.CODECS
              + status.CODECS + sbes.CODECS)
