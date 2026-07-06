"""FT processor for the Panda flange F_ext estimate: sign, bias, real low-pass.

The flange external-wrench estimate (sensors.ft_flange.FTFlangeSource) is
noisier than a dedicated load cell — it is reconstructed from the joint-torque
sensors — so the light single-pole EWA used for the 400 Hz Bota stream
(core.ft_processor.FTProcessor) is not enough here.  This variant keeps the
exact same public interface (tare / process / reset) so it is a drop-in for the
backend and the state-machine TARE, but replaces the smoother with a proper
**2nd-order Butterworth low-pass** (with an optional median despike stage to
kill the isolated torque-estimate spikes).

Pipeline per tick:  sign  ->  bias subtract  ->  despike  ->  low-pass

Design goals:
    * real-time safe: all buffers preallocated, no per-tick heap allocation
      beyond the small median temporaries (only when despike is enabled).
    * flat passband, DC gain exactly 1 (a constant force in -> same force out),
      so a tared baseline stays tared.
    * priming on tare: the filter is seeded to the current reading so the first
      processed output snaps to ~0 instead of ramping up from a cold start.
"""

from __future__ import annotations
import math
import numpy as np


class _Butter2LowPass:
    """Vectorized 2nd-order Butterworth low-pass (RBJ biquad, Direct Form II-T).

    One independent filter per channel (n channels), Q = 1/sqrt(2) so the
    magnitude response is maximally flat (Butterworth).  DC gain == 1.
    step() does no allocation; the state (z1, z2) is updated in place.
    """

    def __init__(self, cutoff_hz: float, sample_rate_hz: float, n: int = 6):
        fc = float(cutoff_hz)
        fs = float(sample_rate_hz)
        if not (0.0 < fc < fs / 2.0):
            raise ValueError(
                f"cutoff_hz must be in (0, fs/2 = {fs / 2.0:.1f} Hz), got {fc}"
            )

        # RBJ audio-EQ-cookbook low-pass coefficients with Q = 1/sqrt(2).
        w0 = 2.0 * math.pi * fc / fs
        cos_w0 = math.cos(w0)
        alpha = math.sin(w0) / (2.0 * (1.0 / math.sqrt(2.0)))
        a0 = 1.0 + alpha

        # Normalized (a0 = 1) feed-forward / feed-back taps.
        self._b0 = ((1.0 - cos_w0) / 2.0) / a0
        self._b1 = (1.0 - cos_w0) / a0
        self._b2 = ((1.0 - cos_w0) / 2.0) / a0
        self._a1 = (-2.0 * cos_w0) / a0
        self._a2 = (1.0 - alpha) / a0

        self._z1 = np.zeros(n)
        self._z2 = np.zeros(n)

    def reset(self, x0: np.ndarray | None = None) -> None:
        """Clear the filter, optionally priming it to steady state at x0.

        With DC gain 1 the steady-state output for a constant input x0 is x0
        itself; seeding the delay line to that state means the first step(x0)
        returns x0 with no startup transient.
        """
        if x0 is None:
            self._z1[:] = 0.0
            self._z2[:] = 0.0
            return
        x0 = np.asarray(x0, dtype=float)
        # Solve the DF2-T recurrence for constant input/output y = x0:
        #   z2 = b2*x0 - a2*x0 ; z1 = b1*x0 - a1*x0 + z2
        self._z2[:] = self._b2 * x0 - self._a2 * x0
        self._z1[:] = self._b1 * x0 - self._a1 * x0 + self._z2

    def step(self, x: np.ndarray) -> np.ndarray:
        """Advance one sample; returns the filtered (n,) vector (a fresh copy)."""
        y = self._b0 * x + self._z1
        self._z1[:] = self._b1 * x - self._a1 * y + self._z2
        self._z2[:] = self._b2 * x - self._a2 * y
        return y


class _MedianDespike:
    """Per-channel sliding-window median filter to reject isolated spikes.

    Window must be odd and >= 3.  Keeps a small ring buffer of the last N
    samples; the returned value is the per-channel median, which is immune to a
    single outlier in the window.  Cheap for the small N used in practice.
    """

    def __init__(self, window: int, n: int = 6):
        if window < 3 or window % 2 == 0:
            raise ValueError(f"despike window must be an odd int >= 3, got {window}")
        self._w = int(window)
        self._buf = np.zeros((self._w, n))
        self._count = 0

    def reset(self, x0: np.ndarray | None = None) -> None:
        if x0 is None:
            self._buf[:] = 0.0
            self._count = 0
        else:
            self._buf[:] = np.asarray(x0, dtype=float)
            self._count = self._w

    def step(self, x: np.ndarray) -> np.ndarray:
        # Shift the ring by one and insert the newest sample at the end.
        self._buf[:-1] = self._buf[1:]
        self._buf[-1] = x
        if self._count < self._w:
            self._count += 1
            # Not enough history yet — median over what we have.
            return np.median(self._buf[self._w - self._count:], axis=0)
        return np.median(self._buf, axis=0)


class FlangeFTProcessor:
    """Processes raw flange wrenches: sign -> bias -> despike -> low-pass.

    Drop-in replacement for core.ft_processor.FTProcessor: exposes the same
    tare() / process() / reset() surface so the FrankaBackend and the state
    machine's TARE state use it unchanged.

    Typical usage::

        proc = FlangeFTProcessor(sign=1.0, filter_type="butterworth",
                                 cutoff_hz=20.0, sample_rate_hz=1000.0)
        proc.tare(mean_raw_wrench)         # zero the payload baseline at a pose
        out = proc.process(raw_wrench)     # each control tick -> (6,) contact wrench
    """

    def __init__(
        self,
        sign=1.0,
        filter_type: str = "butterworth",
        cutoff_hz: float = 20.0,
        sample_rate_hz: float = 1000.0,
        ewa_alpha: float = 0.1,
        despike_window: int = 1,
    ):
        """
        Args:
            sign:           Multiplier applied per component before bias
                            subtraction. Either a scalar (global flip) or a
                            length-6 vector to flip individual [Fx,Fy,Fz,Tx,Ty,Tz]
                            axes. Choose it so the response is right-handed —
                            pushing the tip UP reads +Fz (a hanging tool reads
                            -Fz), which is the +Fz-on-contact the state machine
                            expects. libfranka's F_ext global sign varies by
                            setup: if EVERY axis reads opposite to the push, use
                            -1.0; a per-axis vector flips a single reversed axis.
            filter_type:    "butterworth" (2nd-order low-pass, default),
                            "ewa" (single-pole exponential), or "none".
            cutoff_hz:      Low-pass cutoff [Hz] (butterworth). Must be < fs/2.
            sample_rate_hz: Control-loop rate [Hz] (Franka torque loop = 1000).
            ewa_alpha:      EWA coefficient in (0, 1] (only when filter_type="ewa";
                            smaller = heavier smoothing).
            despike_window: Median despike window (odd, >= 3 to enable; 1 = off).
                            Applied before the low-pass to remove single-sample
                            spikes common in torque-derived F_ext.
        """
        sign_arr = np.asarray(sign, dtype=float)
        if sign_arr.ndim == 0:
            sign_arr = np.full(6, float(sign_arr))
        elif sign_arr.shape != (6,):
            raise ValueError(f"sign must be a scalar or length-6, got shape {sign_arr.shape}")
        self._sign = sign_arr
        self._bias = np.zeros(6)

        self._filter_type = str(filter_type).lower()
        if self._filter_type == "butterworth":
            self._lp: _Butter2LowPass | None = _Butter2LowPass(cutoff_hz, sample_rate_hz, n=6)
            self._alpha = None
        elif self._filter_type == "ewa":
            if not (0.0 < ewa_alpha <= 1.0):
                raise ValueError(f"ewa_alpha must be in (0, 1], got {ewa_alpha}")
            self._lp = None
            self._alpha = float(ewa_alpha)
        elif self._filter_type == "none":
            self._lp = None
            self._alpha = None
        else:
            raise ValueError(
                f"filter_type must be 'butterworth', 'ewa', or 'none', "
                f"got {filter_type!r}"
            )

        self._despike: _MedianDespike | None = (
            _MedianDespike(despike_window, n=6) if int(despike_window) >= 3 else None
        )

        self._ewa_state = np.zeros(6)
        self._primed = False

    # ------------------------------------------------------------------
    # Public API (identical surface to FTProcessor)
    # ------------------------------------------------------------------

    def tare(self, raw_wrench: np.ndarray, reset_filter: bool = True) -> None:
        """Capture a raw reading (e.g. a multi-sample average) as the bias.

        After tare(), process() returns ~zero for that same input and only
        deviations from the baseline afterwards — so once the payload's gravity
        wrench at a fixed pose is tared, any further reading is the external
        (contact) wrench.

        reset_filter (default True) primes the filter to the current reading so
        the first processed output snaps to ~0 instead of decaying from the
        pre-tare payload level.
        """
        self._bias = np.asarray(raw_wrench, dtype=float).copy() * self._sign
        if reset_filter:
            self._reset_filter_state()

    def process(self, raw_wrench: np.ndarray) -> np.ndarray:
        """Return a processed (6,) wrench.

        Pipeline: sign -> bias subtract -> despike -> low-pass filter.
        """
        signed = np.asarray(raw_wrench, dtype=float) * self._sign
        x = signed - self._bias

        if self._despike is not None:
            x = self._despike.step(x)

        return self._filter(x).copy()

    def reset(self) -> None:
        """Clear bias and all filter state (e.g. before a new trial)."""
        self._bias[:] = 0.0
        self._reset_filter_state()

    # ------------------------------------------------------------------
    # Internals
    # ------------------------------------------------------------------

    def _filter(self, x: np.ndarray) -> np.ndarray:
        """Apply the configured low-pass, priming on the first post-reset sample."""
        if self._filter_type == "none":
            return x

        if not self._primed:
            # Seed the filter to the current reading so output starts at x.
            if self._lp is not None:
                self._lp.reset(x)
            self._ewa_state[:] = x
            self._primed = True

        if self._lp is not None:                 # butterworth
            return self._lp.step(x)
        # ewa
        self._ewa_state = self._alpha * x + (1.0 - self._alpha) * self._ewa_state
        return self._ewa_state

    def _reset_filter_state(self) -> None:
        self._primed = False
        self._ewa_state[:] = 0.0
        if self._lp is not None:
            self._lp.reset()
        if self._despike is not None:
            self._despike.reset()
