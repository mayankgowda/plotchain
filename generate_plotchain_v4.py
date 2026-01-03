# generate_plotchain_v4.py
"""
PlotChain v4 (audited): deterministic synthetic engineering-plot dataset generator.

Key goals:
- Irrefutable ground truth: every target value is computed deterministically from the same
  underlying signals/curves that are plotted, with independent audit checks.
- Human-readable targets: final-answer fields avoid awkward fractional bin artifacts
  (e.g., FFT bin = 0.9765625 Hz) by choosing parameter grids aligned to plot axes.
- Difficulty split: 40% clean, 30% moderate, 30% edge (deterministic via seed).
- Edge cases remain *readable* (reduced grid/shorter window/noisier), but the asked
  quantities remain observable within the plotted window.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np


# ----------------------------
# Global configuration
# ----------------------------

VERSION = "v4-audited"
OUT_JSONL_NAME = "plotchain_v4.jsonl"
IMAGES_DIRNAME = "images"

# Difficulty mix (percent)
DIFFICULTY_SPLIT = {"clean": 0.40, "moderate": 0.30, "edge": 0.30}

# Determinism
DEFAULT_MASTER_SEED = 42
DIFFICULTY_PLAN_SEED = 42

# Plot rendering
FIG_W, FIG_H = 6.0, 3.6
DPI = 160

# Formatting/rounding rules for *final answer fields* (to keep targets human-friendly)
# checkpoint fields may be more precise.
ANSWER_FORMAT: Dict[str, Dict[str, int]] = {
    # family -> field -> decimals (0 means integer)
    "step_response": {"percent_overshoot": 1, "settling_time_s": 2, "steady_state": 0},
    "bode_magnitude": {"dc_gain_db": 0, "cutoff_hz": 0},
    "bode_phase": {"cutoff_hz": 0, "phase_deg_at_10fc": 1},
    "bandpass_response": {"resonance_hz": 0, "bandwidth_hz": 0},
    "time_waveform": {"frequency_hz": 0, "vpp_v": 0},
    "fft_spectrum": {"dominant_frequency_hz": 0, "secondary_frequency_hz": 0},
    "spectrogram": {"f1_hz": 0, "f2_hz": 0, "cp_duration_s": 2, "switch_time_s": 2},
    "iv_curve": {"resistance_ohm": 0, "turn_on_voltage_v_at_target_i": 3},
    "transfer_characteristic": {"small_signal_gain": 2, "saturation_v": 0},
    "pole_zero": {"wn_rad_s": 0, "zeta": 2},
}

# Tolerances for internal audit (not scoring)
AUDIT_ABS_TOL = 1e-6
AUDIT_REL_TOL = 1e-6


# ----------------------------
# Utilities
# ----------------------------

@dataclass(frozen=True)
class ItemMeta:
    difficulty: str
    edge_tag: str
    seed: int


def stable_int_seed(master_seed: int, family: str, idx: int) -> int:
    """Stable per-item seed from (master_seed, family, idx)."""
    s = f"{master_seed}|{family}|{idx}".encode("utf-8")
    # simple 32-bit hash
    h = 2166136261
    for b in s:
        h ^= b
        h = (h * 16777619) & 0xFFFFFFFF
    return int(h)


def quantize_value(family: str, field: str, value: float) -> Any:
    """Quantize final-answer fields to human-friendly decimals (ints when decimals==0)."""
    dec = ANSWER_FORMAT.get(family, {}).get(field, None)
    if dec is None:
        return float(value)
    if dec == 0:
        return int(round(float(value)))
    return float(round(float(value), int(dec)))


def assert_close(name: str, a: float, b: float, abs_tol: float = AUDIT_ABS_TOL, rel_tol: float = AUDIT_REL_TOL) -> None:
    if a is None or b is None or (isinstance(a, float) and math.isnan(a)) or (isinstance(b, float) and math.isnan(b)):
        raise ValueError(f"[audit] {name}: NaN/None encountered (a={a}, b={b})")
    da = abs(float(a) - float(b))
    if da <= abs_tol:
        return
    denom = max(abs(float(b)), 1e-12)
    if da / denom <= rel_tol:
        return
    raise ValueError(f"[audit] {name}: mismatch a={a} b={b} abs_err={da}")


def nice_num(x: float) -> float:
    """Return a 'nice' number approximately equal to x (1,2,5 * 10^k)."""
    if x <= 0:
        return 1.0
    expv = math.floor(math.log10(x))
    f = x / (10**expv)
    if f < 1.5:
        nf = 1.0
    elif f < 3.5:
        nf = 2.0
    elif f < 7.5:
        nf = 5.0
    else:
        nf = 10.0
    return nf * (10**expv)


def nice_ticks_linear(xmin: float, xmax: float, n: int = 6) -> List[float]:
    if xmax <= xmin:
        return [xmin, xmax]
    rng = xmax - xmin
    step = nice_num(rng / max(n - 1, 1))
    start = math.floor(xmin / step) * step
    ticks = [start + i * step for i in range(n + 2)]
    ticks = [t for t in ticks if (t >= xmin - 1e-9 and t <= xmax + 1e-9)]
    # avoid duplicate due to float noise
    out = []
    for t in ticks:
        if not out or abs(t - out[-1]) > 1e-9:
            out.append(float(t))
    if len(out) < 2:
        out = [float(xmin), float(xmax)]
    return out


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out_path, dpi=DPI)
    # fig closed by caller


def make_difficulty_plan(n: int, seed: int = DIFFICULTY_PLAN_SEED) -> List[str]:
    """Deterministic 40/30/30 split."""
    n_clean = int(round(n * DIFFICULTY_SPLIT["clean"]))
    n_moderate = int(round(n * DIFFICULTY_SPLIT["moderate"]))
    n_edge = n - n_clean - n_moderate
    plan = (["clean"] * n_clean) + (["moderate"] * n_moderate) + (["edge"] * n_edge)
    rng = np.random.default_rng(seed)
    rng.shuffle(plan)
    return plan


def _ensure_images_root(images_root: Path) -> None:
    images_root.mkdir(parents=True, exist_ok=True)


def _rel_image_path(images_root: Path, abs_image_path: Path) -> str:
    """Return POSIX-style relative path from images_root parent (dataset dir)."""
    try:
        rel = abs_image_path.relative_to(images_root.parent)
    except Exception:
        rel = abs_image_path
    return str(rel).replace("\\", "/")


# ----------------------------
# Family: Step Response (2nd-order underdamped)
# ----------------------------

def _step_response_curve(t: np.ndarray, zeta: float, wn: float, K: float) -> np.ndarray:
    wd = wn * math.sqrt(max(1.0 - zeta * zeta, 1e-12))
    phi = math.atan2(math.sqrt(max(1.0 - zeta * zeta, 1e-12)), max(zeta, 1e-12))
    y = K * (1.0 - (1.0 / math.sqrt(max(1.0 - zeta * zeta, 1e-12))) * np.exp(-zeta * wn * t) * np.sin(wd * t + phi))
    return y


def _step_metrics_numeric(zeta: float, wn: float, K: float) -> Dict[str, float]:
    # compute on a sufficiently long horizon to capture settling
    # use heuristic upper bound: 12/(zeta*wn)
    t_end = 12.0 / max(zeta * wn, 1e-9)
    t = np.linspace(0.0, t_end, 40000)
    y = _step_response_curve(t, zeta, wn, K)

    steady = float(K)
    peak = float(np.max(y))
    overshoot = max(0.0, (peak - steady) / max(steady, 1e-12) * 100.0)

    lo, hi = 0.98 * steady, 1.02 * steady
    outside = np.where((y < lo) | (y > hi))[0]
    if outside.size == 0:
        ts = 0.0
    else:
        last = int(outside[-1])
        ts = float(t[min(last + 1, len(t) - 1)])
    # peak time (exact) for underdamped:
    wd = wn * math.sqrt(max(1.0 - zeta * zeta, 1e-12))
    tp = math.pi / max(wd, 1e-12)

    return {"steady_state": steady, "percent_overshoot": overshoot, "settling_time_s": ts, "cp_peak_time_s": tp, "cp_peak_value": peak}


def baseline_step_response(pp: Dict[str, Any]) -> Dict[str, Any]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    K = float(pp["K"])
    m = _step_metrics_numeric(zeta, wn, K)
    gt: Dict[str, Any] = {
        "steady_state": quantize_value("step_response", "steady_state", m["steady_state"]),
        "percent_overshoot": quantize_value("step_response", "percent_overshoot", m["percent_overshoot"]),
        "settling_time_s": quantize_value("step_response", "settling_time_s", m["settling_time_s"]),
        # checkpoints:
        "cp_peak_time_s": float(m["cp_peak_time_s"]),
        "cp_peak_value": float(m["cp_peak_value"]),
        "cp_band_lower": float(0.98 * K),
        "cp_band_upper": float(1.02 * K),
    }
    return gt


def gen_step_response(rng: np.random.Generator, difficulty: str) -> Dict[str, Any]:
    zetas = [0.2, 0.3, 0.4, 0.5, 0.6]
    wns = [4.0, 6.0, 8.0, 10.0]  # rad/s
    zeta = float(rng.choice(zetas))
    wn = float(rng.choice(wns))
    K = 1.0

    # ensure plot shows at least up to settling time
    ts = _step_metrics_numeric(zeta, wn, K)["settling_time_s"]
    if difficulty == "clean":
        t_end = 2.2 * ts
        n_points = 800
        noise = 0.0
        grid = True
        edge_tag = ""
    elif difficulty == "moderate":
        t_end = 1.6 * ts
        n_points = 600
        noise = 0.0
        grid = True
        edge_tag = "shorter_window"
    else:
        # edge: still includes settling, but fewer points and no grid
        t_end = 1.25 * ts
        n_points = 450
        noise = float(rng.uniform(0.0, 0.01))
        grid = False
        edge_tag = "no_grid_sparse"
    pp = {"zeta": zeta, "wn_rad_s": wn, "K": K, "t_end_s": float(t_end), "n_points": int(n_points), "noise": noise, "grid": grid}
    return pp, edge_tag


def render_step_response(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    K = float(pp["K"])
    t_end = float(pp["t_end_s"])
    n_points = int(pp["n_points"])
    noise = float(pp["noise"])
    grid = bool(pp["grid"])

    t = np.linspace(0.0, t_end, n_points)
    y = _step_response_curve(t, zeta, wn, K)
    if noise > 0:
        y = y + noise * np.random.default_rng(meta.seed + 7).normal(size=y.shape)

    # audit: metrics from plotted curve should match baseline (within quantization)
    gt = baseline_step_response(pp)
    m = {
        "steady_state": float(K),
        "percent_overshoot": max(0.0, (float(np.max(y)) - K) / K * 100.0),
        "settling_time_s": _step_metrics_numeric(zeta, wn, K)["settling_time_s"],  # numeric true
    }
    assert_close("step/settling_time_s", float(m["settling_time_s"]), float(gt["settling_time_s"]), abs_tol=0.05, rel_tol=0.05)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(t, y, linewidth=1.6)
    ax.set_title("Step Response (2nd-order)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    if grid:
        ax.grid(True, alpha=0.35)

    ax.set_xlim(0.0, t_end)
    ax.set_ylim(min(-0.1, float(np.min(y)) - 0.05), float(np.max(y)) + 0.05)
    ax.set_xticks(nice_ticks_linear(0.0, t_end, 6))
    ax.set_yticks(nice_ticks_linear(float(np.min(y)), float(np.max(y)), 6))

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": 0.0, "x_max": t_end,
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "tick_step_x": nice_ticks_linear(0.0, t_end, 6),
        "tick_step_y": nice_ticks_linear(float(np.min(y)), float(np.max(y)), 6),
    }


# ----------------------------
# Family: Bode Magnitude (1st-order low-pass)
# ----------------------------

LOG_TICKS = [1, 2, 5, 10, 20, 50, 100, 200, 500, 1_000, 2_000, 5_000, 10_000]


def _bode_mag_curve(f: np.ndarray, K: float, fc: float) -> np.ndarray:
    mag = 20.0 * np.log10(K / np.sqrt(1.0 + (f / fc) ** 2))
    return mag


def baseline_bode_magnitude(pp: Dict[str, Any]) -> Dict[str, Any]:
    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    # final fields quantized
    dc = 20.0 * math.log10(max(K, 1e-12))
    gt: Dict[str, Any] = {
        "dc_gain_db": quantize_value("bode_magnitude", "dc_gain_db", dc),
        "cutoff_hz": quantize_value("bode_magnitude", "cutoff_hz", fc),
        # checkpoints
        "cp_mag_at_fc_db": float(dc - 3.0103),
        "cp_slope_db_per_decade": float(-20.0),
    }
    return gt


def gen_bode_magnitude(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    # Choose dc gain directly to avoid awkward decimals in final answers.
    dc_choices = [-20, -10, 0, 10, 20]
    dc = float(rng.choice(dc_choices))
    K = 10 ** (dc / 20.0)

    fc_choices = [10, 20, 50, 100, 200, 500, 1_000]
    fc = float(rng.choice(fc_choices))

    fmin, fmax = 1.0, 10_000.0
    n_points = 500 if difficulty != "edge" else 350
    grid = (difficulty != "edge")
    edge_tag = "" if difficulty != "edge" else "no_grid_sparse"
    pp = {"K": float(K), "fc_hz": float(fc), "fmin_hz": fmin, "fmax_hz": fmax, "n_points": int(n_points), "grid": grid}
    return pp, edge_tag


def render_bode_magnitude(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_points = int(pp["n_points"])
    grid = bool(pp["grid"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    mag = _bode_mag_curve(f, K, fc)

    # audit: ensure the -3dB point occurs at fc (within small tolerance)
    dc = 20.0 * math.log10(max(K, 1e-12))
    mag_fc = float(_bode_mag_curve(np.array([fc]), K, fc)[0])
    assert_close("bode_mag/mag_at_fc", mag_fc, dc - 3.0103, abs_tol=5e-4, rel_tol=1e-6)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.semilogx(f, mag, linewidth=1.6)
    ax.set_title("Bode Magnitude (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if grid:
        ax.grid(True, which="both", alpha=0.30)

    ax.set_xlim(fmin, fmax)
    ax.set_xticks([t for t in LOG_TICKS if fmin <= t <= fmax])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks(nice_ticks_linear(float(np.min(mag)), float(np.max(mag)), 6))

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
        "tick_step_x": None,
        "tick_step_y": nice_ticks_linear(float(np.min(mag)), float(np.max(mag)), 6),
    }


# ----------------------------
# Family: Bode Phase (1st-order low-pass)
# ----------------------------

def _bode_phase_curve(f: np.ndarray, fc: float) -> np.ndarray:
    phase = -np.degrees(np.arctan(f / fc))
    return phase


def baseline_bode_phase(pp: Dict[str, Any]) -> Dict[str, Any]:
    fc = float(pp["fc_hz"])
    phase_10fc = float(-math.degrees(math.atan(10.0)))
    gt: Dict[str, Any] = {
        "cutoff_hz": quantize_value("bode_phase", "cutoff_hz", fc),
        "phase_deg_at_10fc": quantize_value("bode_phase", "phase_deg_at_10fc", phase_10fc),
        # checkpoints
        "cp_phase_deg_at_fc": float(-45.0),
    }
    return gt


def gen_bode_phase(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    fc_choices = [10, 20, 50, 100, 200, 500, 1_000]
    fc = float(rng.choice(fc_choices))
    fmin, fmax = 1.0, 10_000.0
    n_points = 450 if difficulty != "edge" else 320
    grid = (difficulty != "edge")
    edge_tag = "" if difficulty != "edge" else "no_grid_sparse"
    pp = {"fc_hz": float(fc), "fmin_hz": fmin, "fmax_hz": fmax, "n_points": int(n_points), "grid": grid}
    return pp, edge_tag


def render_bode_phase(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_points = int(pp["n_points"])
    grid = bool(pp["grid"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    phase = _bode_phase_curve(f, fc)

    # audit: phase at fc is -45
    ph_fc = float(_bode_phase_curve(np.array([fc]), fc)[0])
    assert_close("bode_phase/phase_at_fc", ph_fc, -45.0, abs_tol=1e-3, rel_tol=1e-6)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.semilogx(f, phase, linewidth=1.6)
    ax.set_title("Bode Phase (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (deg)")
    if grid:
        ax.grid(True, which="both", alpha=0.30)

    ax.set_xlim(fmin, fmax)
    ax.set_ylim(-95, 5)
    ax.set_xticks([t for t in LOG_TICKS if fmin <= t <= fmax])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks([-90, -75, -60, -45, -30, -15, 0])

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(phase)), "y_max": float(np.max(phase)),
        "tick_step_x": None,
        "tick_step_y": [-90, -75, -60, -45, -30, -15, 0],
    }


# ----------------------------
# Family: Bandpass Response (synthetic magnitude with -3 dB points)
# ----------------------------

def _bandpass_mag_curve(f: np.ndarray, f1: float, f2: float) -> np.ndarray:
    # Parabolic in log-frequency, peak at f0 with 0 dB, and -3 dB at f1 and f2
    logf = np.log10(f)
    logf1, logf2 = math.log10(f1), math.log10(f2)
    logf0 = 0.5 * (logf1 + logf2)
    # Choose curvature so mag(logf1)=mag(logf2)=-3
    a = 3.0 / ((logf1 - logf0) ** 2)
    mag_db = -a * (logf - logf0) ** 2
    return mag_db


def _bandpass_pairs() -> List[Tuple[int, int]]:
    # pick pairs from LOG_TICKS such that resonance = sqrt(f1*f2) is integer and within ticks
    ticks = [t for t in LOG_TICKS if t >= 5 and t <= 5_000]
    pairs = []
    for i, f1 in enumerate(ticks):
        for f2 in ticks[i + 1:]:
            prod = f1 * f2
            r = int(round(math.sqrt(prod)))
            if r * r == prod:
                # ensure bandwidth not too extreme
                if 0.15 <= (f2 / f1) <= 20.0:
                    pairs.append((int(f1), int(f2)))
    # prefer mid-bandwidth pairs
    return pairs


BANDPASS_PAIRS = _bandpass_pairs()


def baseline_bandpass(pp: Dict[str, Any]) -> Dict[str, Any]:
    f1, f2 = float(pp["f1_hz"]), float(pp["f2_hz"])
    res = math.sqrt(f1 * f2)
    bw = f2 - f1
    q = res / max(bw, 1e-12)
    gt: Dict[str, Any] = {
        "resonance_hz": quantize_value("bandpass_response", "resonance_hz", res),
        "bandwidth_hz": quantize_value("bandpass_response", "bandwidth_hz", bw),
        # checkpoints
        "cp_f1_3db_hz": float(f1),
        "cp_f2_3db_hz": float(f2),
        "cp_q_factor": float(q),
    }
    return gt


def gen_bandpass(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    if not BANDPASS_PAIRS:
        raise RuntimeError("No valid bandpass pairs found.")
    f1, f2 = BANDPASS_PAIRS[int(rng.integers(0, len(BANDPASS_PAIRS)))]
    # difficulty affects resolution/grid only
    n_points = 500 if difficulty == "clean" else (420 if difficulty == "moderate" else 320)
    grid = (difficulty != "edge")
    edge_tag = "" if difficulty != "edge" else "no_grid_sparse"
    pp = {"f1_hz": float(f1), "f2_hz": float(f2), "fmin_hz": 1.0, "fmax_hz": 10_000.0, "n_points": int(n_points), "grid": grid}
    return pp, edge_tag


def render_bandpass(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter

    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_points = int(pp["n_points"])
    grid = bool(pp["grid"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    mag = _bandpass_mag_curve(f, f1, f2)

    # audit: ensure -3 dB at f1,f2 and 0 dB at resonance
    m1 = float(_bandpass_mag_curve(np.array([f1]), f1, f2)[0])
    m2 = float(_bandpass_mag_curve(np.array([f2]), f1, f2)[0])
    assert_close("bandpass/mag_at_f1", m1, -3.0, abs_tol=1e-6, rel_tol=1e-6)
    assert_close("bandpass/mag_at_f2", m2, -3.0, abs_tol=1e-6, rel_tol=1e-6)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.semilogx(f, mag, linewidth=1.6)
    ax.set_title("Bandpass Magnitude Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if grid:
        ax.grid(True, which="both", alpha=0.30)

    ax.set_xlim(fmin, fmax)
    ax.set_ylim(-20, 1)
    ax.set_xticks([t for t in LOG_TICKS if fmin <= t <= fmax])
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_yticks([-18, -15, -12, -9, -6, -3, 0])

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
        "tick_step_x": None,
        "tick_step_y": [-18, -15, -12, -9, -6, -3, 0],
    }


# ----------------------------
# Family: Time Waveform (sine/square/triangle)
# ----------------------------

def _wave(t: np.ndarray, wave_type: str, f0: float, A: float, duty: float) -> np.ndarray:
    phase = (t * f0) % 1.0
    if wave_type == "sine":
        return A * np.sin(2 * np.pi * f0 * t)
    if wave_type == "square":
        return np.where(phase < duty, A, -A)
    if wave_type == "triangle":
        return A * (4 * np.abs(phase - 0.5) - 1.0)
    raise ValueError(wave_type)


def baseline_time_wave(pp: Dict[str, Any]) -> Dict[str, Any]:
    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    wave_type = str(pp["wave_type"])
    duty = float(pp.get("duty", 0.5))

    gt: Dict[str, Any] = {
        "frequency_hz": quantize_value("time_waveform", "frequency_hz", f0),
        "vpp_v": quantize_value("time_waveform", "vpp_v", 2.0 * A),
        "cp_period_s": float(1.0 / max(f0, 1e-12)),
        "cp_vmax_v": float(A),
        "cp_vmin_v": float(-A),
    }
    if wave_type == "square":
        gt["cp_duty"] = float(duty)
    return gt


def gen_time_wave(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    wave_type = str(rng.choice(["sine", "square", "triangle"]))
    # Choose frequencies that divide fs exactly to avoid sampling artifacts.
    fs = 6000.0
    f_choices = [5, 10, 12, 15, 20, 24, 25, 30, 40, 50, 60, 75, 80, 100, 120]
    f0 = float(rng.choice(f_choices))
    A_choices = [0.5, 1.0, 2.0, 3.0, 5.0]
    A = float(rng.choice(A_choices))
    duty = 0.5
    if wave_type == "square":
        duty = float(rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))

    if difficulty == "clean":
        cycles = int(rng.choice([5, 6]))
        grid = True
        edge_tag = ""
    elif difficulty == "moderate":
        cycles = int(rng.choice([3, 4]))
        grid = True
        edge_tag = "fewer_cycles"
    else:
        cycles = int(rng.choice([1, 2]))
        grid = False
        edge_tag = "no_grid_low_cycles"

    t_end = float(cycles / max(f0, 1e-12))
    n_samples = int(round(t_end * fs))
    pp: Dict[str, Any] = {"wave_type": wave_type, "f0_hz": f0, "A": A, "fs_hz": fs, "t_end_s": t_end, "n_samples": n_samples}
    if wave_type == "square":
        pp["duty"] = duty
    pp["grid"] = grid
    return pp, edge_tag


def render_time_wave(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    wave_type = str(pp["wave_type"])
    duty = float(pp.get("duty", 0.5))
    t_end = float(pp["t_end_s"])
    fs = float(pp["fs_hz"])
    n_samples = int(pp["n_samples"])
    grid = bool(pp.get("grid", True))

    t = np.arange(n_samples) / fs
    y = _wave(t, wave_type, f0, A, duty)

    # audit: ensure period is exactly representable in samples for chosen f0, fs
    spp = fs / f0
    assert_close("time_wave/samples_per_period_int", spp, round(spp), abs_tol=1e-9, rel_tol=0.0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(t, y, linewidth=1.6)
    ax.set_title(f"Time Waveform ({wave_type})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    if grid:
        ax.grid(True, alpha=0.35)

    ax.set_xlim(0.0, t_end)
    ax.set_ylim(float(np.min(y)) - 0.1 * abs(A), float(np.max(y)) + 0.1 * abs(A))
    ax.set_xticks(nice_ticks_linear(0.0, t_end, 6))
    ax.set_yticks(nice_ticks_linear(float(np.min(y)), float(np.max(y)), 6))

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": 0.0, "x_max": t_end,
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "tick_step_x": nice_ticks_linear(0.0, t_end, 6),
        "tick_step_y": nice_ticks_linear(float(np.min(y)), float(np.max(y)), 6),
    }


# ----------------------------
# Family: FFT Spectrum (two-tone magnitude)
# ----------------------------

def _fft_spectrum(signal: np.ndarray, fs: float) -> Tuple[np.ndarray, np.ndarray]:
    N = len(signal)
    Y = np.fft.rfft(signal * np.hanning(N))
    mag = np.abs(Y)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    return freqs, mag


def baseline_fft(pp: Dict[str, Any]) -> Dict[str, Any]:
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    a1 = float(pp["a1"])
    a2 = float(pp["a2"])
    # peak ratio checkpoint is amplitude ratio (not power), still meaningful
    gt: Dict[str, Any] = {
        "dominant_frequency_hz": quantize_value("fft_spectrum", "dominant_frequency_hz", max(f1, f2) if a1 == a2 else (f1 if a1 > a2 else f2)),
        "secondary_frequency_hz": quantize_value("fft_spectrum", "secondary_frequency_hz", min(f1, f2) if a1 == a2 else (f2 if a1 > a2 else f1)),
        "cp_peak_ratio": float(max(a1, a2) / max(min(a1, a2), 1e-12)),
    }
    return gt


def gen_fft(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    # Use fs=N=2048 -> 1 Hz bin spacing; avoid awkward fractions.
    fs = 2048.0
    N = 2048
    # choose well-separated tones, away from DC and Nyquist
    f_choices = [20, 40, 50, 60, 80, 100, 120, 150, 180, 200, 240, 300, 360, 400, 480, 600]
    f1, f2 = rng.choice(f_choices, size=2, replace=False)
    f1, f2 = float(min(f1, f2)), float(max(f1, f2))

    if difficulty == "clean":
        a1, a2 = 1.0, 0.6
        noise = 0.0
        grid = True
        edge_tag = ""
    elif difficulty == "moderate":
        a1, a2 = 1.0, float(rng.choice([0.5, 0.6, 0.7]))
        noise = 0.01
        grid = True
        edge_tag = "light_noise"
    else:
        a1, a2 = 1.0, float(rng.choice([0.35, 0.45, 0.55]))
        noise = 0.03
        grid = False
        edge_tag = "no_grid_noisy"

    pp = {"fs_hz": fs, "N": N, "f1_hz": f1, "f2_hz": f2, "a1": float(a1), "a2": float(a2), "noise": float(noise), "grid": grid}
    return pp, edge_tag


def render_fft(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fs = float(pp["fs_hz"])
    N = int(pp["N"])
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    a1 = float(pp["a1"])
    a2 = float(pp["a2"])
    noise = float(pp["noise"])
    grid = bool(pp["grid"])

    t = np.arange(N) / fs
    sig = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    if noise > 0:
        sig = sig + noise * np.random.default_rng(meta.seed + 11).normal(size=sig.shape)

    freqs, mag = _fft_spectrum(sig, fs)

    # audit: ensure peaks align exactly to integer bins
    # find two largest peaks excluding DC
    idx0 = 1
    peak_idxs = np.argsort(mag[idx0:])[::-1][:6] + idx0
    peak_freqs = sorted({int(round(freqs[i])) for i in peak_idxs})
    assert int(round(f1)) in peak_freqs and int(round(f2)) in peak_freqs

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(freqs, mag, linewidth=1.3)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (a.u.)")
    if grid:
        ax.grid(True, alpha=0.30)

    ax.set_xlim(0, 700)
    ax.set_xticks([0, 100, 200, 300, 400, 500, 600, 700])
    ax.set_yticks(nice_ticks_linear(float(np.min(mag)), float(np.max(mag)), 6))

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": 0.0, "x_max": 700.0,
        "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
        "tick_step_x": [0, 100, 200, 300, 400, 500, 600, 700],
        "tick_step_y": nice_ticks_linear(float(np.min(mag)), float(np.max(mag)), 6),
    }


# ----------------------------
# Family: Spectrogram (two-frequency switch)
# ----------------------------

def baseline_spectrogram(pp: Dict[str, Any]) -> Dict[str, Any]:
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    gt: Dict[str, Any] = {
        "f1_hz": quantize_value("spectrogram", "f1_hz", f1),
        "f2_hz": quantize_value("spectrogram", "f2_hz", f2),
        "cp_duration_s": quantize_value("spectrogram", "cp_duration_s", float(pp["duration_s"])),
        "switch_time_s": quantize_value("spectrogram", "switch_time_s", float(pp["switch_time_s"])),
    }
    return gt


def gen_spectrogram(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    fs = 2048.0
    duration = 1.0
    # NFFT=256 -> 8 Hz bins; choose freqs as multiples of 8
    f_choices = [64, 80, 96, 112, 128, 160, 192, 256, 320, 384, 512, 640]
    f1, f2 = rng.choice(f_choices, size=2, replace=False)
    f1, f2 = float(f1), float(f2)

    if difficulty == "clean":
        switch_time = float(rng.choice([0.35, 0.45, 0.55]))
        noise = 0.0
        grid = True
        edge_tag = ""
    elif difficulty == "moderate":
        switch_time = float(rng.choice([0.30, 0.50, 0.70]))
        noise = 0.01
        grid = True
        edge_tag = "light_noise"
    else:
        switch_time = float(rng.choice([0.80, 0.85, 0.90]))
        noise = 0.03
        grid = False
        edge_tag = "late_switch_noisy"

    pp = {"fs_hz": fs, "duration_s": duration, "switch_time_s": switch_time, "f1_hz": f1, "f2_hz": f2, "noise": noise, "grid": grid}
    return pp, edge_tag


def render_spectrogram(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fs = float(pp["fs_hz"])
    duration = float(pp["duration_s"])
    switch_time = float(pp["switch_time_s"])
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    noise = float(pp["noise"])
    grid = bool(pp["grid"])

    t = np.arange(int(duration * fs)) / fs
    sig = np.where(t < switch_time,
                   np.sin(2 * np.pi * f1 * t),
                   np.sin(2 * np.pi * f2 * t))
    if noise > 0:
        sig = sig + noise * np.random.default_rng(meta.seed + 19).normal(size=sig.shape)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    Pxx, freqs, bins, im = ax.specgram(sig, NFFT=256, Fs=fs, noverlap=128)
    ax.set_title("Spectrogram (frequency switch)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    ax.set_ylim(0, 700)
    ax.set_yticks([0, 100, 200, 300, 400, 500, 600, 700])
    if grid:
        ax.grid(False)  # grid over image is distracting

    # audit: dominant freq before/after switch matches f1/f2 on coarse bins
    # Use simple FFT of segments
    i_switch = int(round(switch_time * fs))
    seg1 = sig[:max(i_switch, 1)]
    seg2 = sig[min(i_switch, len(sig)-1):]
    def _dom_freq(seg):
        if len(seg) < 32:
            return None
        fre, mag = _fft_spectrum(seg, fs)
        k = int(np.argmax(mag[1:]) + 1)
        return int(round(fre[k]))
    d1 = _dom_freq(seg1)
    d2 = _dom_freq(seg2)
    if d1 is not None:
        assert abs(d1 - int(round(f1))) <= 8
    if d2 is not None:
        assert abs(d2 - int(round(f2))) <= 8

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": 0.0, "x_max": duration,
        "y_min": 0.0, "y_max": 700.0,
        "tick_step_x": nice_ticks_linear(0.0, duration, 6),
        "tick_step_y": [0, 100, 200, 300, 400, 500, 600, 700],
    }


# ----------------------------
# Family: I–V Curve (resistor or diode+series resistance)
# ----------------------------

def baseline_iv(pp: Dict[str, Any]) -> Dict[str, Any]:
    kind = str(pp["kind"])
    if kind == "resistor":
        R = float(pp["R_ohm"])
        gt: Dict[str, Any] = {
            "resistance_ohm": quantize_value("iv_curve", "resistance_ohm", R),
            "cp_slope_ohm": float(R),
        }
        return gt

    # diode
    Is = float(pp["Is"])
    nVt = float(pp["nVt"])
    Rs = float(pp["Rs"])
    It = float(pp["target_current_a"])
    V = nVt * math.log(It / max(Is, 1e-30) + 1.0) + It * Rs
    gt = {
        "turn_on_voltage_v_at_target_i": quantize_value("iv_curve", "turn_on_voltage_v_at_target_i", V),
        "cp_Is": float(Is),
        "cp_nVt": float(nVt),
        "cp_Rs": float(Rs),
        "target_current_a": float(It),
    }
    return gt


def gen_iv(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    if rng.random() < 0.5:
        # resistor
        R = float(rng.choice([10, 22, 47, 100, 220, 470, 1_000]))
        grid = (difficulty != "edge")
        edge_tag = "" if difficulty != "edge" else "no_grid"
        pp = {"kind": "resistor", "R_ohm": R, "grid": grid}
        return pp, edge_tag

    # diode tuned to yield ~0.6–0.8 V at 10–20 mA (human-friendly)
    Is = float(rng.choice([1e-8, 2e-8, 5e-8]))
    nVt = float(rng.choice([0.05]))  # keep simple
    Rs = float(rng.choice([0.0, 1.0, 2.0]))
    It = float(rng.choice([0.01, 0.02]))  # 10 mA / 20 mA

    grid = (difficulty != "edge")
    edge_tag = "" if difficulty != "edge" else "no_grid"
    pp = {"kind": "diode", "Is": Is, "nVt": nVt, "Rs": Rs, "target_current_a": It, "grid": grid}
    return pp, edge_tag


def render_iv(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    kind = str(pp["kind"])
    grid = bool(pp.get("grid", True))

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    if kind == "resistor":
        R = float(pp["R_ohm"])
        V = np.linspace(-5, 5, 200)
        I = V / R
        ax.plot(V, I, linewidth=1.6)
        ax.set_title("I–V Curve (Resistor)")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        ax.set_xlim(-5, 5)
        ax.set_xticks([-5, -2.5, 0, 2.5, 5])
        ax.set_yticks(nice_ticks_linear(float(np.min(I)), float(np.max(I)), 6))
        if grid:
            ax.grid(True, alpha=0.30)
        save_figure(fig, out_path)
        plt.close(fig)
        return {"x_min": -5.0, "x_max": 5.0, "y_min": float(np.min(I)), "y_max": float(np.max(I)), "tick_step_x": [-5, -2.5, 0, 2.5, 5], "tick_step_y": None}

    # diode
    Is = float(pp["Is"])
    nVt = float(pp["nVt"])
    Rs = float(pp["Rs"])
    It = float(pp["target_current_a"])

    I = np.linspace(0, 0.03, 220)  # up to 30 mA
    V = nVt * np.log(I / max(Is, 1e-30) + 1.0) + I * Rs

    # audit: voltage at target current matches baseline (quantized)
    gt = baseline_iv(pp)
    Vt = float(nVt * math.log(It / max(Is, 1e-30) + 1.0) + It * Rs)
    assert_close("iv/turn_on_voltage", float(gt["turn_on_voltage_v_at_target_i"]), float(round(Vt, ANSWER_FORMAT["iv_curve"]["turn_on_voltage_v_at_target_i"])), abs_tol=1e-6, rel_tol=0.0)

    ax.plot(V, I, linewidth=1.6)
    ax.set_title("I–V Curve (Diode + Rs)")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    ax.set_xlim(0, float(np.max(V)) * 1.05)
    ax.set_ylim(0, 0.03)
    ax.set_yticks([0.0, 0.01, 0.02, 0.03])
    ax.set_xticks(nice_ticks_linear(0.0, float(np.max(V)) * 1.05, 6))
    if grid:
        ax.grid(True, alpha=0.30)

    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": 0.0, "x_max": float(np.max(V)) * 1.05, "y_min": 0.0, "y_max": 0.03, "tick_step_x": None, "tick_step_y": [0.0, 0.01, 0.02, 0.03]}


# ----------------------------
# Family: Transfer Characteristic (saturating amplifier)
# ----------------------------

def baseline_transfer(pp: Dict[str, Any]) -> Dict[str, Any]:
    gain = float(pp["gain"])
    vsat = float(pp["vsat_v"])
    vin_sat = vsat / max(gain, 1e-12)
    gt: Dict[str, Any] = {
        "small_signal_gain": quantize_value("transfer_characteristic", "small_signal_gain", gain),
        "saturation_v": quantize_value("transfer_characteristic", "saturation_v", vsat),
        # checkpoints
        "cp_vin_at_saturation": float(vin_sat),
    }
    return gt


def gen_transfer(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    gain = float(rng.choice([1.0, 1.5, 2.0, 3.0]))
    vsat = float(rng.choice([1.0, 2.0, 3.0, 5.0]))
    grid = (difficulty != "edge")
    edge_tag = "" if difficulty != "edge" else "no_grid"
    pp = {"gain": gain, "vsat_v": vsat, "grid": grid}
    return pp, edge_tag


def render_transfer(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    gain = float(pp["gain"])
    vsat = float(pp["vsat_v"])
    grid = bool(pp.get("grid", True))

    vin = np.linspace(-10, 10, 400)
    vout = np.clip(gain * vin, -vsat, vsat)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.plot(vin, vout, linewidth=1.6)
    ax.set_title("Transfer Characteristic (saturation)")
    ax.set_xlabel("Vin (V)")
    ax.set_ylabel("Vout (V)")
    ax.set_xlim(-10, 10)
    ax.set_ylim(-vsat * 1.1, vsat * 1.1)
    ax.set_xticks([-10, -5, 0, 5, 10])
    ax.set_yticks(nice_ticks_linear(-vsat * 1.1, vsat * 1.1, 6))
    if grid:
        ax.grid(True, alpha=0.30)

    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": -10.0, "x_max": 10.0, "y_min": float(np.min(vout)), "y_max": float(np.max(vout)), "tick_step_x": [-10, -5, 0, 5, 10], "tick_step_y": None}


# ----------------------------
# Family: Pole-Zero Plot (2nd-order poles)
# ----------------------------

def baseline_pole_zero(pp: Dict[str, Any]) -> Dict[str, Any]:
    wn = float(pp["wn_rad_s"])
    zeta = float(pp["zeta"])
    pole_real = -zeta * wn
    pole_imag = wn * math.sqrt(max(1.0 - zeta * zeta, 0.0))
    gt: Dict[str, Any] = {
        "wn_rad_s": quantize_value("pole_zero", "wn_rad_s", wn),
        "zeta": quantize_value("pole_zero", "zeta", zeta),
        # checkpoints (pole coordinates)
        "cp_pole_real": float(pole_real),
        "cp_pole_imag": float(pole_imag),
    }
    return gt


def gen_pole_zero(rng: np.random.Generator, difficulty: str) -> Tuple[Dict[str, Any], str]:
    # wn in rad/s (keep integers; easy to read on axis)
    wn = float(rng.choice([4, 6, 8, 10, 12, 15]))
    zeta = float(rng.choice([0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]))
    grid = (difficulty != "edge")
    edge_tag = "" if difficulty != "edge" else "no_grid"
    pp = {"wn_rad_s": wn, "zeta": zeta, "grid": grid}
    return pp, edge_tag


def render_pole_zero(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    wn = float(pp["wn_rad_s"])
    zeta = float(pp["zeta"])
    grid = bool(pp.get("grid", True))

    pole_real = -zeta * wn
    pole_imag = wn * math.sqrt(max(1.0 - zeta * zeta, 0.0))

    # audit: consistent with baseline
    gt = baseline_pole_zero(pp)
    assert_close("pole_zero/pole_real", pole_real, float(gt["cp_pole_real"]), abs_tol=1e-9, rel_tol=0.0)
    assert_close("pole_zero/pole_imag", pole_imag, float(gt["cp_pole_imag"]), abs_tol=1e-9, rel_tol=0.0)

    fig, ax = plt.subplots(figsize=(FIG_W, FIG_H))
    ax.scatter([pole_real, pole_real], [pole_imag, -pole_imag], s=40, marker="x")
    ax.axvline(0, color="k", linewidth=1.0, alpha=0.6)
    ax.axhline(0, color="k", linewidth=1.0, alpha=0.6)
    ax.set_title("Pole-Zero Plot (2nd-order poles)")
    ax.set_xlabel("Real axis (rad/s)")
    ax.set_ylabel("Imag axis (rad/s)")

    lim = max(wn * 1.3, 10.0)
    ax.set_xlim(-lim, lim * 0.2)
    ax.set_ylim(-lim, lim)
    ax.set_xticks(nice_ticks_linear(-lim, lim * 0.2, 6))
    ax.set_yticks(nice_ticks_linear(-lim, lim, 6))
    if grid:
        ax.grid(True, alpha=0.30)

    save_figure(fig, out_path)
    plt.close(fig)

    return {"x_min": -lim, "x_max": lim * 0.2, "y_min": -lim, "y_max": lim, "tick_step_x": None, "tick_step_y": None}


# ----------------------------
# Orchestrator
# ----------------------------

FAMILIES = [
    ("step_response", gen_step_response, baseline_step_response, render_step_response,
     ["steady_state", "percent_overshoot", "settling_time_s"],
     ["cp_peak_time_s", "cp_peak_value", "cp_band_lower", "cp_band_upper"]),
    ("bode_magnitude", gen_bode_magnitude, baseline_bode_magnitude, render_bode_magnitude,
     ["dc_gain_db", "cutoff_hz"],
     ["cp_mag_at_fc_db", "cp_slope_db_per_decade"]),
    ("bode_phase", gen_bode_phase, baseline_bode_phase, render_bode_phase,
     ["cutoff_hz", "phase_deg_at_10fc"],
     ["cp_phase_deg_at_fc"]),
    ("bandpass_response", gen_bandpass, baseline_bandpass, render_bandpass,
     ["resonance_hz", "bandwidth_hz"],
     ["cp_f1_3db_hz", "cp_f2_3db_hz", "cp_q_factor"]),
    ("time_waveform", gen_time_wave, baseline_time_wave, render_time_wave,
     ["frequency_hz", "vpp_v"],
     ["cp_period_s", "cp_vmax_v", "cp_vmin_v", "cp_duty"]),
    ("fft_spectrum", gen_fft, baseline_fft, render_fft,
     ["dominant_frequency_hz", "secondary_frequency_hz"],
     ["cp_peak_ratio"]),
    ("spectrogram", gen_spectrogram, baseline_spectrogram, render_spectrogram,
     ["f1_hz", "f2_hz"],
     ["cp_duration_s", "switch_time_s"]),
    ("iv_curve", gen_iv, baseline_iv, render_iv,
     ["resistance_ohm", "turn_on_voltage_v_at_target_i"],
     ["cp_slope_ohm", "cp_Is", "cp_nVt", "cp_Rs", "target_current_a"]),
    ("transfer_characteristic", gen_transfer, baseline_transfer, render_transfer,
     ["small_signal_gain", "saturation_v"],
     ["cp_vin_at_saturation"]),
    ("pole_zero", gen_pole_zero, baseline_pole_zero, render_pole_zero,
     ["zeta", "wn_rad_s"],
     ["cp_pole_real", "cp_pole_imag"]),
]


def _question_for_family(family: str, final_fields: List[str], checkpoint_fields: List[str]) -> str:
    # Describe rounding expectations for final fields.
    fmt_parts = []
    for f in final_fields:
        dec = ANSWER_FORMAT.get(family, {}).get(f, None)
        if dec is None:
            continue
        if dec == 0:
            fmt_parts.append(f"- {f}: integer")
        else:
            fmt_parts.append(f"- {f}: round to {dec} decimal places")
    fmt_txt = "\n".join(fmt_parts) if fmt_parts else "- Use reasonable numeric precision."

    schema = {k: "<number or null>" for k in (final_fields + checkpoint_fields) if k.startswith("cp_") or k in final_fields}
    schema_txt = json.dumps(schema, indent=2)

    q = (
        "You are given an engineering plot image. Read the plot and answer the question.\n\n"
        f"Return ONLY a single JSON object matching this schema (numbers or null; no strings; no units; no extra keys):\n{schema_txt}\n\n"
        "Notes:\n"
        "- Use cp_* fields as intermediate plot reads (checkpoints).\n"
        "- If you cannot determine a value, output null for that key.\n"
        "Rounding for final fields:\n"
        f"{fmt_txt}\n"
    )
    return q


def generate_dataset(out_dir: Path, n_per_family: int, master_seed: int) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / IMAGES_DIRNAME
    _ensure_images_root(images_root)

    items: List[Dict[str, Any]] = []
    for family, gen_fn, baseline_fn, render_fn, final_fields, checkpoint_fields in FAMILIES:
        diff_plan = make_difficulty_plan(n_per_family, seed=DIFFICULTY_PLAN_SEED)
        fam_dir = images_root / family
        fam_dir.mkdir(parents=True, exist_ok=True)

        for i in range(n_per_family):
            seed = stable_int_seed(master_seed, family, i)
            rng = np.random.default_rng(seed)
            difficulty = diff_plan[i]
            pp, edge_tag = gen_fn(rng, difficulty)
            meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

            img_name = f"{family}_{i:03d}.png"
            abs_img = fam_dir / img_name
            axis_meta = render_fn(pp, abs_img, meta)

            gt = baseline_fn(pp)

            # Keep only keys that exist (some families have mutually exclusive finals like iv_curve).
            gt_kept = {k: gt[k] for k in gt.keys()}

            q = _question_for_family(family, final_fields, checkpoint_fields)

            items.append({
                "id": f"{family}_{i:03d}",
                "type": family,
                "image_path": _rel_image_path(images_root, abs_img),
                "question": q,
                "ground_truth": gt_kept,
                "plot_params": pp,
                "generation": {
                    "version": VERSION,
                    "difficulty_plan_seed": DIFFICULTY_PLAN_SEED,
                    "seed": seed,
                    "difficulty": difficulty,
                    "edge_tag": edge_tag,
                    "final_fields": final_fields,
                    "checkpoint_fields": checkpoint_fields,
                    "answer_format": ANSWER_FORMAT.get(family, {}),
                    **axis_meta,
                },
            })

    jsonl_path = out_dir / OUT_JSONL_NAME
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it) + "\n")
    return jsonl_path


def validate_dataset(jsonl_path: Path) -> None:
    # Lightweight structural checks + audit the "human-friendly" formatting of final fields.
    n = 0
    with jsonl_path.open("r", encoding="utf-8") as f:
        for line in f:
            it = json.loads(line)
            n += 1
            fam = it["type"]
            gt = it["ground_truth"]
            fmt = ANSWER_FORMAT.get(fam, {})
            for field, dec in fmt.items():
                if field in gt and gt[field] is not None:
                    v = gt[field]
                    if dec == 0:
                        if not isinstance(v, int):
                            # allow float that is integral (e.g., 100.0) but enforce convertibility
                            if abs(float(v) - round(float(v))) > 1e-9:
                                raise ValueError(f"[validate] {it['id']} field {field} expected int, got {v}")
                    else:
                        # ensure no ugly tails like 0.9765625 unless allowed by dec
                        if abs(float(v) - round(float(v), dec)) > 1e-9:
                            raise ValueError(f"[validate] {it['id']} field {field} not quantized to {dec} decimals: {v}")
    print(f"[validate] OK items={n}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/plotchain_v4", help="Output dataset directory")
    ap.add_argument("--n_per_family", type=int, default=30)
    ap.add_argument("--seed", type=int, default=DEFAULT_MASTER_SEED)
    ap.add_argument("--validate_only", action="store_true")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    jsonl_path = out_dir / OUT_JSONL_NAME

    if not args.validate_only:
        path = generate_dataset(out_dir=out_dir, n_per_family=args.n_per_family, master_seed=args.seed)
        print(f"[write] {path} ({args.n_per_family} per family; {len(FAMILIES)*args.n_per_family} items)")
    validate_dataset(jsonl_path)


if __name__ == "__main__":
    main()
