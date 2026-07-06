"""Sensing-only bring-up test for the Panda flange FT + filter.

Reads the Panda's own external-wrench estimate O_F_ext_hat_K in a read-only
libfranka loop — robot.read(), which commands NO torque, so the arm just holds
its own pose — pushes it through the SAME sensors.ft_flange.FTFlangeSource and
core.ft_processor_flange.FlangeFTProcessor the real controller uses, and
live-plots UNFILTERED vs FILTERED Fx / Fy / Fz so you can verify, before any
torque test:

    * SIGN   — the response should be RIGHT-HANDED: push the tip UP -> +Fz,
               push +X -> +Fx, push +Y -> +Fy (and a freely hanging tool -> -Fz).
               +Fz is the contact/"compression" sign the state machine expects.
               libfranka's F_ext global sign varies by setup: if EVERY axis reads
               OPPOSITE to your push (a "left-handed" response), set ft.sign: -1.0
               — that restores right-handedness AND the +Fz-on-contact sign. Use a
               per-axis ft.sign (e.g. [1,1,-1,1,1,1]) only for a single reversed axis.
    * FILTER — watch the thin "unfiltered" trace (noisy) vs the bold "filtered"
               trace (smooth). Tune filter.cutoff_hz / despike_window until the
               noise band is acceptable without too much lag.

No motion is commanded and no payload-gravity torque is applied — this is a
pure sensing/DSP check.  Run from the repo root (force_control/):

    python scripts/check_ft_flange.py                       # uses configs/real_flange.yaml
    python scripts/check_ft_flange.py --tare-seconds 2      # zero the baseline at start
    python scripts/check_ft_flange.py --cutoff 10           # try a heavier low-pass
    python scripts/check_ft_flange.py --show-torque         # also plot Tx/Ty/Tz
    python scripts/check_ft_flange.py --simulate            # no robot: synthetic noisy data (UI check)

Hotkeys (plot window):  t = tare (zero baseline)   r = reset/untare   q = quit
"""

from __future__ import annotations

import argparse
import sys
import threading
import time
from collections import deque
from pathlib import Path

import numpy as np
import yaml

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from sensors.ft_flange import FTFlangeSource            # noqa: E402
from core.ft_processor_flange import FlangeFTProcessor  # noqa: E402

_LABELS = ["Fx", "Fy", "Fz", "Tx", "Ty", "Tz"]
_UNITS = ["N", "N", "N", "Nm", "Nm", "Nm"]


class _SimRobotState:
    """Mimics a pylibfranka RobotState for --simulate (only the wrench field)."""

    def __init__(self, wrench6):
        self.O_F_ext_hat_K = list(wrench6)
        self.K_F_ext_hat_K = list(wrench6)


class FlangeFTMonitor:
    """Owns the source + processor, the reader thread, and the shared buffers."""

    def __init__(self, cfg: dict, args):
        ft_cfg = cfg.get("ft", {})
        filt = dict(ft_cfg.get("filter", {}))

        # CLI overrides win over the config file.
        self._frame = (args.frame or ft_cfg.get("frame", "base")).lower()
        sign_cfg = args.sign if args.sign is not None else ft_cfg.get("sign", 1.0)
        self._sign = np.asarray(sign_cfg, dtype=float)   # scalar or (6,); broadcasts
        filter_type = args.filter_type or filt.get("type", "butterworth")
        cutoff = float(args.cutoff if args.cutoff is not None else filt.get("cutoff_hz", 20.0))
        self._fs = float(filt.get("sample_rate_hz", 1000.0))
        despike = int(args.despike if args.despike is not None else filt.get("despike_window", 1))

        self._source = FTFlangeSource(frame=self._frame)
        self._proc = FlangeFTProcessor(
            sign=self._sign,
            filter_type=filter_type,
            cutoff_hz=cutoff,
            sample_rate_hz=self._fs,
            despike_window=despike,
        )
        sign_str = (f"{self._sign.flat[0]:+.0f}"
                    if np.all(self._sign == self._sign.flat[0])
                    else np.array2string(self._sign.astype(int)))
        self._status = (
            f"frame={self._frame}  sign={sign_str}  "
            f"filter={filter_type}@{cutoff:g}Hz  despike={despike}  fs={self._fs:g}Hz"
        )

        # Shared, lock-protected ring buffers (single writer = reader thread).
        cap = int(args.window * self._fs * 1.3) + 500
        self._lock = threading.Lock()
        self._t = deque(maxlen=cap)
        self._raw = deque(maxlen=cap)     # sign+bias corrected, UNFILTERED
        self._filt = deque(maxlen=cap)    # full FlangeFTProcessor output

        # Reader-thread-owned tare state (never mutated from the main thread).
        self._bias = np.zeros(6)          # mirrors the processor's internal bias
        self._tare_seconds = float(args.tare_seconds)
        self._tare_active = False
        self._tare_t0 = 0.0
        self._tare_sum = np.zeros(6)
        self._tare_n = 0
        # Requests set by main-thread hotkeys, consumed by the reader thread.
        self._req_tare = args.tare_seconds > 0.0   # optional auto-tare at start
        self._req_untare = False

        # Rate meter.
        self._count = 0
        self._t_start = time.perf_counter()
        self._rate_hz = 0.0
        self._rate_count0 = 0
        self._rate_t0 = self._t_start

        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._simulate = bool(args.simulate)
        self._robot_ip = args.ip or cfg.get("robot_ip", "172.16.0.2")

    # ------------------------------------------------------------------
    # Reader thread
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._thread = threading.Thread(target=self._run, daemon=True, name="flange_read")
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=2.0)

    def _run(self) -> None:
        if self._simulate:
            self._run_sim()
        else:
            self._run_robot()

    def _run_robot(self) -> None:
        try:
            from pylibfranka import Robot  # noqa: PLC0415
        except Exception as e:  # pragma: no cover - hardware import
            print(f"\nERROR importing pylibfranka: {e}\n"
                  "Run on the Ubuntu Panda PC, or use --simulate for a UI check.")
            self._stop.set()
            return

        try:
            print(f"Connecting to Franka at {self._robot_ip} (read-only, no motion) …")
            robot = Robot(self._robot_ip)
            s0 = robot.read_once()
            print(f"Connected. q[deg]={np.degrees(np.asarray(s0.q)).round(1).tolist()}")
            print(f"Configured load m_load = {float(getattr(s0, 'm_load', 0.0)):.3f} kg  "
                  f"(if ~0, F_ext includes the tool weight — tare it out with 't').")
            print(f"Start O_F_ext_hat_K = {np.asarray(s0.O_F_ext_hat_K).round(2).tolist()}")
            print(">>> Reading at ~1 kHz. Push the tip to check sign; 't' tares, 'q' quits.\n")
            robot.read(self._on_state)  # blocks until the callback returns False
        except Exception as e:  # pragma: no cover - hardware runtime
            print(f"\nRobot read loop ended: {e}")
        finally:
            self._stop.set()

    def _on_state(self, robot_state) -> bool:
        """libfranka read callback (~1 kHz). Return False to stop reading."""
        self._handle_raw(robot_state)
        return not self._stop.is_set()

    def _run_sim(self) -> None:
        """Synthetic noisy wrench (no robot): slow drift + white noise + a bump."""
        print(">>> SIMULATE mode: synthetic noisy data, no robot. 't' tares, 'q' quits.\n")
        rng = np.random.default_rng(0)
        dt = 1.0 / self._fs
        k = 0
        base = np.array([0.5, -0.3, -8.0, 0.02, -0.05, 0.01])  # payload-like offset
        next_t = time.perf_counter()
        while not self._stop.is_set():
            tt = k * dt
            drift = 0.4 * np.sin(2 * np.pi * 0.2 * tt)
            bump = 5.0 if (5.0 < tt % 10.0 < 6.0) else 0.0     # periodic +Fz "contact"
            w = base + rng.normal(0, [0.8, 0.8, 1.2, 0.03, 0.03, 0.03])
            w[2] += drift + bump
            self._handle_raw(_SimRobotState(w))
            k += 1
            next_t += dt
            sleep = next_t - time.perf_counter()
            if sleep > 0:
                time.sleep(sleep)

    # ------------------------------------------------------------------
    # Shared processing (reader thread only)
    # ------------------------------------------------------------------

    def _handle_raw(self, robot_state) -> None:
        now = time.perf_counter() - self._t_start

        # Extract via the REAL source (tests the actual code path), then filter.
        sample = self._source.update(robot_state, t=now)
        filt = self._proc.process(sample.wrench)

        # ---- tare state machine (single-writer, race-free) ----
        if self._req_untare:
            self._proc.reset()
            self._bias[:] = 0.0
            self._req_untare = False
        if self._req_tare and not self._tare_active:
            self._tare_active = True
            self._tare_t0 = now
            self._tare_sum[:] = 0.0
            self._tare_n = 0
            self._req_tare = False
        if self._tare_active:
            self._tare_sum += sample.wrench
            self._tare_n += 1
            if (now - self._tare_t0) >= max(self._tare_seconds, 0.3):
                mean_raw = self._tare_sum / max(self._tare_n, 1)
                self._proc.tare(mean_raw)               # zero the processor baseline
                self._bias[:] = mean_raw * self._sign   # mirror it for the raw trace
                self._tare_active = False
                print(f"Tared over {self._tare_n} samples: "
                      f"bias(raw)={mean_raw.round(3).tolist()}")

        corrected = self._sign * sample.wrench - self._bias   # unfiltered, same baseline

        with self._lock:
            self._t.append(now)
            self._raw.append(corrected)
            self._filt.append(filt)

        # rate meter (once/sec)
        self._count += 1
        if now - (self._rate_t0 - self._t_start) >= 1.0:
            n = self._count - self._rate_count0
            self._rate_hz = n / (time.perf_counter() - self._rate_t0)
            self._rate_t0 = time.perf_counter()
            self._rate_count0 = self._count

    # ------------------------------------------------------------------
    # Main-thread API (hotkeys / snapshot)
    # ------------------------------------------------------------------

    def request_tare(self) -> None:
        self._req_tare = True

    def request_untare(self) -> None:
        self._req_untare = True

    def snapshot(self):
        with self._lock:
            if not self._t:
                return None, None, None
            return (np.fromiter(self._t, float),
                    np.array(self._raw),
                    np.array(self._filt))

    @property
    def status(self) -> str:
        return self._status

    @property
    def rate_hz(self) -> float:
        return self._rate_hz

    @property
    def stopped(self) -> bool:
        return self._stop.is_set()


# ----------------------------------------------------------------------
# Live plot
# ----------------------------------------------------------------------

def run_plot(mon: FlangeFTMonitor, channels, window: float) -> None:
    import matplotlib.pyplot as plt
    from matplotlib.animation import FuncAnimation

    n = len(channels)
    fig, axes = plt.subplots(n, 1, sharex=True, figsize=(10, 2.0 * n + 1))
    if n == 1:
        axes = [axes]
    colors = ["tab:blue", "tab:green", "tab:red", "tab:purple", "tab:olive", "tab:cyan"]

    raw_lines, filt_lines = [], []
    for ax, ch in zip(axes, channels):
        ax.axhline(0.0, color="k", lw=0.8, ls="--", alpha=0.4)
        lr, = ax.plot([], [], color="silver", lw=0.8, alpha=0.9, label="unfiltered")
        lf, = ax.plot([], [], color=colors[ch], lw=2.0, label="filtered")
        raw_lines.append(lr)
        filt_lines.append(lf)
        ax.set_ylabel(f"{_LABELS[ch]} [{_UNITS[ch]}]")
        ax.grid(True, alpha=0.25)
        ax.legend(loc="upper left", fontsize=8, ncol=2)
    axes[-1].set_xlabel("time [s]")

    hint = ("right-handed check:  push UP→+Fz  +X→+Fx  +Y→+Fy   "
            "(ALL opposite ⇒ sign -1)    [t]tare [r]reset [q]quit")
    fig.suptitle(f"Flange FT sensing check   |   {mon.status}\n{hint}", fontsize=10)
    txt = fig.text(0.5, 0.005, "", ha="center", va="bottom", fontsize=9, family="monospace")

    def on_key(event):
        if event.key == "t":
            mon.request_tare()
        elif event.key == "r":
            mon.request_untare()
        elif event.key in ("q", "escape"):
            mon.stop()
            plt.close(fig)

    fig.canvas.mpl_connect("key_press_event", on_key)
    fig.canvas.mpl_connect("close_event", lambda e: mon.stop())

    def update(_frame):
        t, raw, filt = mon.snapshot()
        if t is None:
            return raw_lines + filt_lines + [txt]
        t_now = t[-1]
        vis = t >= (t_now - window)
        tv = t[vis]
        cur = []
        for i, ch in enumerate(channels):
            rv = raw[vis, ch]
            fv = filt[vis, ch]
            raw_lines[i].set_data(tv, rv)
            filt_lines[i].set_data(tv, fv)
            lo = min(rv.min(), fv.min())
            hi = max(rv.max(), fv.max())
            pad = max(0.5, 0.1 * (hi - lo))
            axes[i].set_ylim(lo - pad, hi + pad)
            cur.append(f"{_LABELS[ch]}={fv[-1]:+7.2f}")
        axes[0].set_xlim(max(0.0, t_now - window), t_now + 0.05)
        txt.set_text(f"filtered:  {'   '.join(cur)}     |     "
                     f"read rate ~{mon.rate_hz:5.0f} Hz")
        if mon.stopped:
            plt.close(fig)
        return raw_lines + filt_lines + [txt]

    ani = FuncAnimation(fig, update, interval=50, blit=False, cache_frame_data=False)
    fig.tight_layout(rect=(0, 0.03, 1, 0.92))
    plt.show()
    mon.stop()
    return ani


def run_headless(mon: FlangeFTMonitor, channels) -> None:
    """No-display fallback: print filtered vs unfiltered at ~10 Hz until Ctrl+C."""
    print("(headless: no plot — Ctrl+C to quit)")
    try:
        while not mon.stopped:
            time.sleep(0.1)
            t, raw, filt = mon.snapshot()
            if t is None:
                continue
            r = raw[-1]
            f = filt[-1]
            cells = [f"{_LABELS[c]} {f[c]:+7.2f}({r[c]:+6.2f})" for c in channels]
            print(f"\r~{mon.rate_hz:4.0f}Hz  " + "  ".join(cells) +
                  "   filt(raw)   ", end="", flush=True)
    except KeyboardInterrupt:
        print()
    finally:
        mon.stop()


def main() -> None:
    p = argparse.ArgumentParser(
        description="Sensing-only live check of the Panda flange FT + filter "
                    "(no motion, no torque).")
    p.add_argument("--config", default=str(PROJECT_ROOT / "configs" / "real_flange.yaml"))
    p.add_argument("--ip", default=None, help="robot IP (overrides config robot_ip)")
    p.add_argument("--frame", choices=["base", "stiffness"], default=None)
    p.add_argument("--sign", type=float, default=None, help="override ft.sign (+1 / -1)")
    p.add_argument("--filter-type", choices=["butterworth", "ewa", "none"], default=None)
    p.add_argument("--cutoff", type=float, default=None, help="override low-pass cutoff [Hz]")
    p.add_argument("--despike", type=int, default=None, help="median despike window (odd>=3, 1=off)")
    p.add_argument("--window", type=float, default=10.0, help="plot time window [s]")
    p.add_argument("--tare-seconds", type=float, default=0.0,
                   help="average this many seconds at start and zero the baseline")
    p.add_argument("--show-torque", action="store_true", help="also plot Tx/Ty/Tz")
    p.add_argument("--no-plot", action="store_true", help="headless numeric readout")
    p.add_argument("--simulate", action="store_true",
                   help="no robot: feed synthetic noisy data to check the UI/filter")
    args = p.parse_args()

    cfg_path = Path(args.config)
    cfg = yaml.safe_load(cfg_path.read_text()) if cfg_path.exists() else {}
    if not cfg_path.exists():
        print(f"(config {cfg_path} not found — using defaults / CLI overrides)")

    channels = [0, 1, 2] + ([3, 4, 5] if args.show_torque else [])

    mon = FlangeFTMonitor(cfg, args)
    print(f"Flange FT check — {mon.status}")
    mon.start()

    import os
    have_display = bool(os.environ.get("DISPLAY")) or sys.platform == "darwin"
    if args.no_plot or not have_display:
        if not have_display and not args.no_plot:
            print("(no DISPLAY detected — falling back to headless readout)")
        run_headless(mon, channels)
    else:
        try:
            run_plot(mon, channels, args.window)
        except Exception as e:
            print(f"(plot unavailable: {e} — falling back to headless)")
            run_headless(mon, channels)

    mon.stop()
    print("Done.")


if __name__ == "__main__":
    main()
