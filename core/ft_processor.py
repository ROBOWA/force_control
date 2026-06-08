"""Shared FT data processor: sign convention, bias removal, low-pass filter.

Reusable by both MuJoCo sim and real-hardware backends.  No sensor-specific
or backend-specific logic lives here.
"""

from __future__ import annotations
import numpy as np


class FTProcessor:
    """Processes raw wrench arrays: sign flip → bias subtract → EWA low-pass.

    All operations are in-place on an internal state vector; process() returns
    a fresh copy each call.

    Typical usage::

        proc = FTProcessor(sign=-1.0, lowpass_alpha=0.2)
        proc.tare(raw_wrench)            # zero at current reading
        out = proc.process(raw_wrench)   # each control tick
    """

    def __init__(
        self,
        sign: float = 1.0,
        lowpass_alpha: float = 1.0,
    ):
        """
        Args:
            sign:           Applied as a scalar multiplier before bias subtraction.
                            Use -1.0 to flip the sensor sign convention.
            lowpass_alpha:  EWA coefficient in (0, 1].  1.0 = no filtering;
                            smaller values → heavier smoothing.
        """
        if not (0.0 < lowpass_alpha <= 1.0):
            raise ValueError(f"lowpass_alpha must be in (0, 1], got {lowpass_alpha}")

        self._sign = float(sign)
        self._alpha = float(lowpass_alpha)
        self._bias = np.zeros(6)
        self._filtered = np.zeros(6)
        self._initialized = False

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    def tare(self, raw_wrench: np.ndarray) -> None:
        """Capture the current raw reading as the bias.

        After tare(), process() returns zero for the same input and deviations
        from that baseline for subsequent readings.
        """
        self._bias = raw_wrench.copy() * self._sign

    def process(self, raw_wrench: np.ndarray) -> np.ndarray:
        """Return a processed (6,) wrench array.

        Pipeline: sign → bias subtract → EWA low-pass filter.
        """
        signed = raw_wrench * self._sign
        unbiased = signed - self._bias

        if not self._initialized:
            self._filtered = unbiased.copy()
            self._initialized = True
        else:
            self._filtered = (
                self._alpha * unbiased + (1.0 - self._alpha) * self._filtered
            )

        return self._filtered.copy()

    def reset(self) -> None:
        """Clear bias and filter state (e.g. before a new trial)."""
        self._bias[:] = 0.0
        self._filtered[:] = 0.0
        self._initialized = False
