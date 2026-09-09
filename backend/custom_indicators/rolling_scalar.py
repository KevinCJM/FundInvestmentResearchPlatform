"""Public metadata for the active rolling-scalar-draft API.

Both rolling derivation routes use rolling_series.py. This module contains no
alternate transformer; the narrower bounds belong to the existing API schema.
"""
from .rolling_series import ROLLING_TRANSFORM_VERSION as ROLLING_SCALAR_TRANSFORM_VERSION

MIN_ROLLING_WINDOW_OBSERVATIONS = 2
MAX_ROLLING_WINDOW_OBSERVATIONS = 5000

__all__ = [
    "ROLLING_SCALAR_TRANSFORM_VERSION",
    "MIN_ROLLING_WINDOW_OBSERVATIONS",
    "MAX_ROLLING_WINDOW_OBSERVATIONS",
]
