#!/usr/bin/env python3
"""
generate_plotchain_v4_FROZEN.py

PlotChain v4 (FROZEN) — deterministic, synthetic, verifiable engineering-plot benchmark.

Key properties:
- 100% reproducible from (master_seed, family, index)
- Ground-truth computed from plot_params via explicit baselines (no OCR)
- Human-friendly targets (tick-aligned; avoids weird fractions in expected outputs)
- Difficulty split per family: 40% clean / 30% moderate / 30% edge
- 15 plot families (includes IV split into iv_resistor + iv_diode)
- Bandpass resonance stored with 1 decimal (per user request)

Output layout:
  <out_dir>/
    plotchain_v4.jsonl
    images/<type>/*.png
    validation_rows.csv
    validation_summary.csv

Run:
  python3 generate_plotchain_v4_FROZEN.py --out_dir data/plotchain_v4 --n_per_family 30 --seed 0
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple, Optional

import numpy as np


# ---------------------------
# Utilities
# ---------------------------

def stable_int_seed(master_seed: int, *parts: Any) -> int:
    h = hashlib.sha256()
    h.update(str(master_seed).encode("utf-8"))
    for p in parts:
        h.update(b"|")
        h.update(str(p).encode("utf-8"))
    return int.from_bytes(h.digest()[:8], "big") & 0x7FFFFFFF


def float_close(a: float, b: float, abs_tol: float = 1e-12, rel_tol: float = 1e-12) -> bool:
    return abs(a - b) <= abs_tol or abs(a - b) <= rel_tol * max(abs(b), 1e-12)


def axis_ticks_linear(xmin: float, xmax: float, n_ticks: int) -> float:
    # Return a "nice" tick step for linear axes
    if n_ticks <= 1:
        return 1.0
    span = max(xmax - xmin, 1e-12)
    raw = span / (n_ticks - 1)
    # snap to {1,2,5} * 10^k
    k = 10 ** math.floor(math.log10(raw))
    for m in (1, 2, 5, 10):
        if m * k >= raw:
            return float(m * k)
    return float(10 * k)


def save_figure(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=160, bbox_inches="tight")
    fig.clf()


@dataclass(frozen=True)
class ItemMeta:
    difficulty: str  # "clean" | "moderate" | "edge"
    edge_tag: str
    seed: int


def make_difficulty_plan(n: int, seed: int) -> List[str]:
    # 40/30/30 split
    n_clean = int(round(0.40 * n))
    n_mod = int(round(0.30 * n))
    n_edge = n - n_clean - n_mod
    plan = (["clean"] * n_clean) + (["moderate"] * n_mod) + (["edge"] * n_edge)
    rng = np.random.default_rng(seed)
    rng.shuffle(plan)
    return plan


# ---------------------------
# Quantization (human-friendly outputs)
# ---------------------------

# decimals = number of decimal places to keep
ANSWER_FORMAT: Dict[str, Dict[str, int]] = {
    # step_response
    "step_response": {
        "percent_overshoot": 1,
        "settling_time_s": 2,
        "steady_state": 2,
        "cp_peak_time_s": 2,
        "cp_peak_value": 2,
        "cp_band_lower": 2,
        "cp_band_upper": 2,
    },
    # bode magnitude
    "bode_magnitude": {
        "dc_gain_db": 1,
        "cutoff_hz": 0,
        "cp_mag_at_fc_db": 1,
        "cp_slope_db_per_decade": 0,
    },
    # bode phase
    "bode_phase": {
        "cutoff_hz": 0,
        "phase_deg_at_fq": 1,
        "cp_phase_deg_at_fc": 1,
    },
    # bandpass
    "bandpass_response": {
        "resonance_hz": 1,         # <-- per user request
        "bandwidth_hz": 1,
        "cp_f1_3db_hz": 1,
        "cp_f2_3db_hz": 1,
        "cp_q_factor": 2,
    },
    # time waveform
    "time_waveform": {
        "frequency_hz": 0,
        "vpp_v": 1,
        "cp_period_s": 3,
        "cp_vmax_v": 1,
        "cp_vmin_v": 1,
        "cp_duty": 2,
    },
    # fft
    "fft_spectrum": {
        "dominant_frequency_hz": 0,
        "secondary_frequency_hz": 0,
        "cp_peak_ratio": 1,
    },
    # spectrogram
    "spectrogram": {
        "f1_hz": 0,
        "f2_hz": 0,
        "switch_time_s": 2,
        "cp_duration_s": 2,
    },
    # iv resistor
    "iv_resistor": {
        "resistance_ohm": 0,
        "cp_slope_ohm": 0,
    },
    # iv diode
    "iv_diode": {
        "target_current_a": 3,
        "turn_on_voltage_v_at_target_i": 2,
    },
    # transfer characteristic
    "transfer_characteristic": {
        "small_signal_gain": 1,
        "saturation_v": 1,
        "cp_vin_at_saturation": 2,
    },
    # pole-zero plot (single pole + single zero)
    "pole_zero": {
        "pole_real": 0,
        "pole_imag": 0,
        "zero_real": 0,
        "zero_imag": 0,
    },
    "stress_strain": {
        "yield_strength_mpa": 0,
        "uts_mpa": 0,
        "fracture_strain": 3
    },
    "torque_speed": {
        "stall_torque_nm": 1,
        "no_load_speed_rpm": 0
    },
    "pump_curve": {
        "head_at_qop_m": 1,
        "q_at_half_head_m3h": 0
    },
    "sn_curve": {
        "stress_at_1e5_mpa": 0,
        "endurance_limit_mpa": 0
    },
}


def quantize_value(family: str, field: str, v: float) -> float:
    dec = ANSWER_FORMAT.get(family, {}).get(field, 6)
    return float(np.round(float(v), dec))


# ---------------------------
# Family 1: Step response
# ---------------------------

def _step_response_y(t: np.ndarray, zeta: float, wn: float) -> np.ndarray:
    # standard 2nd-order underdamped step response (unit step, final=1)
    # y(t) = 1 - (1/sqrt(1-z^2))*exp(-z w_n t)*sin(w_d t + phi)
    if zeta >= 1.0:
        # overdamped fallback (simple first-order-ish)
        return 1.0 - np.exp(-wn * t)
    wd = wn * math.sqrt(1.0 - zeta**2)
    phi = math.atan2(math.sqrt(1.0 - zeta**2), zeta)
    return 1.0 - (1.0 / math.sqrt(1.0 - zeta**2)) * np.exp(-zeta * wn * t) * np.sin(wd * t + phi)


def _settling_time_2pct(t: np.ndarray, y: np.ndarray, final: float = 1.0, tol: float = 0.02) -> float:
    band = tol * max(abs(final), 1e-12)
    lo, hi = final - band, final + band
    inside = (y >= lo) & (y <= hi)
    for i in range(len(t)):
        if inside[i] and inside[i:].all():
            return float(t[i])
    return float(t[-1])


def baseline_step_response(pp: Dict[str, Any]) -> Dict[str, Any]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    t_end = float(pp["t_end_s"])
    n = int(pp["n_points"])
    t = np.linspace(0.0, t_end, n)
    y = _step_response_y(t, zeta, wn)

    steady = float(y[-1])
    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    peak_t = float(t[peak_idx])

    overshoot = max(0.0, (peak_val - steady) / max(abs(steady), 1e-12)) * 100.0
    ts = _settling_time_2pct(t, y, final=steady, tol=0.02)
    band = 0.02 * max(abs(steady), 1e-12)
    band_lo, band_hi = steady - band, steady + band

    gt = {
        "percent_overshoot": overshoot,
        "settling_time_s": ts,
        "steady_state": steady,
        "cp_peak_time_s": peak_t,
        "cp_peak_value": peak_val,
        "cp_band_lower": band_lo,
        "cp_band_upper": band_hi,
    }
    return {k: quantize_value("step_response", k, v) for k, v in gt.items()}


def render_step_response(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    t_end = float(pp["t_end_s"])
    n = int(pp["n_points"])
    t = np.linspace(0.0, t_end, n)
    y = _step_response_y(t, zeta, wn)

    rng = np.random.default_rng(meta.seed + 17)
    if meta.difficulty == "moderate":
        y = y + rng.normal(0.0, 0.003, size=y.shape)
    elif meta.difficulty == "edge":
        y = y + rng.normal(0.0, 0.006, size=y.shape)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, y, linewidth=1.4)
    ax.set_title("Step Response (2nd-order)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": float(t.min()), "x_max": float(t.max()),
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "tick_step_x": axis_ticks_linear(float(t.min()), float(t.max()), 6),
        "tick_step_y": axis_ticks_linear(float(np.min(y)), float(np.max(y)), 6),
    }


# ---------------------------
# Family 2: Bode magnitude (1st order low-pass)
# ---------------------------

def baseline_bode_magnitude(pp: Dict[str, Any]) -> Dict[str, Any]:
    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    dc_gain_db = 20.0 * math.log10(max(K, 1e-12))
    mag_at_fc = dc_gain_db - 3.0103
    slope = -20.0
    gt = {
        "dc_gain_db": dc_gain_db,
        "cutoff_hz": fc,
        "cp_mag_at_fc_db": mag_at_fc,
        "cp_slope_db_per_decade": slope,
    }
    return {k: quantize_value("bode_magnitude", k, v) for k, v in gt.items()}


def render_bode_magnitude(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    fq = float(pp["fq_hz"])
    n_pts = int(pp["n_points"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_pts)
    mag_db = 20.0 * np.log10(max(K, 1e-12) / np.sqrt(1.0 + (f / max(fc, 1e-12)) ** 2))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogx(f, mag_db, linewidth=1.4)
    ax.set_title("Bode Magnitude (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    b = baseline_bode_magnitude(pp)
    if meta.difficulty != "edge":
        ax.axvline(float(b["cutoff_hz"]), linestyle="--", linewidth=1.0, alpha=0.6)
        ax.scatter([float(b["cutoff_hz"])], [float(b["cp_mag_at_fc_db"])], s=22)
        ax.text(float(b["cutoff_hz"]), float(np.max(mag_db)), "fc", fontsize=9, va="bottom", ha="center")

    ax.axvline(fq, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(fq, float(np.min(mag_db)), " f_q", rotation=90, va="bottom", ha="left", fontsize=9)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(mag_db)), "y_max": float(np.max(mag_db)),
        "tick_step_x": None,
        "tick_step_y": None,
    }


# ---------------------------
# Family 3: Bode phase (1st order low-pass) — phase at per-item f_q
# ---------------------------

def baseline_bode_phase(pp: Dict[str, Any]) -> Dict[str, Any]:
    fc = float(pp["fc_hz"])
    fq = float(pp["fq_hz"])
    phase_fq = -math.degrees(math.atan(fq / max(fc, 1e-12)))
    phase_fc = -45.0
    gt = {
        "cutoff_hz": fc,
        "phase_deg_at_fq": phase_fq,
        "cp_phase_deg_at_fc": phase_fc,
    }
    return {k: quantize_value("bode_phase", k, v) for k, v in gt.items()}


def render_bode_phase(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    fq = float(pp["fq_hz"])
    n_pts = int(pp["n_points"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_pts)
    phase_deg = -np.degrees(np.arctan(f / max(fc, 1e-12)))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogx(f, phase_deg, linewidth=1.4)
    ax.set_title("Bode Phase (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (deg)")
    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    if meta.difficulty != "edge":
        ax.axvline(fc, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.scatter([fc], [-45.0], s=22)
        ax.text(fc, float(np.max(phase_deg)), "fc", fontsize=9, va="bottom", ha="center")

    ax.axvline(fq, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(fq, float(np.min(phase_deg)), " f_q", rotation=90, va="bottom", ha="left", fontsize=9)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(phase_deg)), "y_max": float(np.max(phase_deg)),
        "tick_step_x": None,
        "tick_step_y": None,
    }


# ---------------------------
# Family 4: Bandpass response (2nd order band-pass magnitude)
# ---------------------------

def _bandpass_mag(f: np.ndarray, f0: float, Q: float) -> np.ndarray:
    # Normalized 2nd-order bandpass (peak ~ 1 at resonance)
    w = f / max(f0, 1e-12)
    num = w / max(Q, 1e-12)
    den = np.sqrt((1.0 - w**2)**2 + (w / max(Q, 1e-12))**2)
    return num / np.maximum(den, 1e-12)


def _bandpass_3db_points(f0: float, Q: float) -> Tuple[float, float]:
    # Solve |H|^2 = 1/2 for normalized bandpass:
    # w^2 = ((2Q^2+1) ± sqrt((2Q^2+1)^2 - 4Q^2)) / (2Q^2)
    Q2 = Q * Q
    a = 2.0 * Q2 + 1.0
    disc = max(a * a - 4.0 * Q2, 1e-12)
    w2_1 = (a - math.sqrt(disc)) / (2.0 * Q2)
    w2_2 = (a + math.sqrt(disc)) / (2.0 * Q2)
    w1 = math.sqrt(max(w2_1, 1e-12))
    w2 = math.sqrt(max(w2_2, 1e-12))
    return float(f0 * w1), float(f0 * w2)


def baseline_bandpass(pp: Dict[str, Any]) -> Dict[str, Any]:
    f0 = float(pp["f0_hz"])
    Q = float(pp["Q"])
    f1, f2 = _bandpass_3db_points(f0, Q)
    bw = f2 - f1
    gt = {
        "resonance_hz": f0,
        "cp_f1_3db_hz": f1,
        "cp_f2_3db_hz": f2,
        "bandwidth_hz": bw,
        "cp_q_factor": f0 / max(bw, 1e-12),
    }
    return {k: quantize_value("bandpass_response", k, v) for k, v in gt.items()}


def render_bandpass(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    f0 = float(pp["f0_hz"])
    Q = float(pp["Q"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_pts = int(pp["n_points"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_pts)
    mag = _bandpass_mag(f, f0, Q)
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogx(f, mag_db, linewidth=1.4)
    ax.set_title("Bandpass Magnitude (2nd-order)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    b = baseline_bandpass(pp)
    if meta.difficulty != "edge":
        # helper line at -3 dB from peak (~0 dB peak)
        ax.axhline(-3.0103, linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(float(b["cp_f1_3db_hz"]), linestyle="--", linewidth=1.0, alpha=0.6)
        ax.axvline(float(b["cp_f2_3db_hz"]), linestyle="--", linewidth=1.0, alpha=0.6)
        ax.text(float(b["resonance_hz"]), float(np.max(mag_db)), "f0", fontsize=9, va="bottom", ha="center")

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(mag_db)), "y_max": float(np.max(mag_db)),
        "tick_step_x": None,
        "tick_step_y": None,
    }


# ---------------------------
# Family 5: Time waveform
# ---------------------------

def _wave(t: np.ndarray, wave_type: str, f0: float, A: float, duty: float) -> np.ndarray:
    phase = (t * f0) % 1.0
    if wave_type == "sine":
        return A * np.sin(2 * np.pi * f0 * t)
    if wave_type == "square":
        return np.where(phase < duty, A, -A)
    if wave_type == "triangle":
        return A * (4 * np.abs(phase - 0.5) - 1.0)
    raise ValueError(wave_type)


def baseline_time_waveform(pp: Dict[str, Any]) -> Dict[str, Any]:
    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    gt: Dict[str, float] = {
        "frequency_hz": f0,
        "vpp_v": 2.0 * A,
        "cp_period_s": 1.0 / max(f0, 1e-12),
        "cp_vmax_v": A,
        "cp_vmin_v": -A,
    }
    if str(pp.get("wave_type", "")) == "square":
        gt["cp_duty"] = float(pp.get("duty", 0.5))
    return {k: quantize_value("time_waveform", k, v) for k, v in gt.items()}


def render_time_waveform(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    wave_type = str(pp["wave_type"])
    duty = float(pp.get("duty", 0.5))
    t_end = float(pp["t_end_s"])
    fs = float(pp["fs_hz"])

    t = np.arange(0.0, t_end, 1.0 / fs)
    y = _wave(t, wave_type, f0, A, duty)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, y, linewidth=1.3)
    ax.set_title(f"Time Waveform ({wave_type})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    x_min, x_max = 0.0, t_end
    y_min, y_max = float(np.min(y)), float(np.max(y))
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "tick_step_x": axis_ticks_linear(x_min, x_max, 6),
        "tick_step_y": axis_ticks_linear(y_min, y_max, 6),
    }


# ---------------------------
# Family 6: FFT spectrum (two-tone) — normalized magnitudes & clean ratios
# ---------------------------

def baseline_fft(pp: Dict[str, Any]) -> Dict[str, Any]:
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    a1 = float(pp["a1"])
    a2 = float(pp["a2"])

    if a1 == a2:
        dom_f = max(f1, f2)
        sec_f = min(f1, f2)
        ratio = 1.0
    elif a1 > a2:
        dom_f, sec_f = f1, f2
        ratio = a1 / max(a2, 1e-12)
    else:
        dom_f, sec_f = f2, f1
        ratio = a2 / max(a1, 1e-12)

    gt = {
        "dominant_frequency_hz": dom_f,
        "secondary_frequency_hz": sec_f,
        "cp_peak_ratio": ratio,
    }
    return {k: quantize_value("fft_spectrum", k, v) for k, v in gt.items()}


def render_fft(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fs = float(pp["fs_hz"])
    N = int(pp["N"])
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    a1 = float(pp["a1"])
    a2 = float(pp["a2"])
    noise = float(pp["noise"])

    t = np.arange(N) / fs
    rng = np.random.default_rng(meta.seed + 31)
    x = a1 * np.sin(2 * np.pi * f1 * t) + a2 * np.sin(2 * np.pi * f2 * t)
    if noise > 0:
        x = x + rng.normal(0.0, noise, size=x.shape)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = np.abs(X)
    mag = mag / max(float(np.max(mag)), 1e-12)  # normalize to 1.0 peak
    mag_db = 20.0 * np.log10(np.maximum(mag, 1e-12))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(freqs, mag_db, linewidth=1.2)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB, normalized)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.3)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": float(freqs.min()), "x_max": float(freqs.max()),
        "y_min": float(np.min(mag_db)), "y_max": float(np.max(mag_db)),
        "tick_step_x": axis_ticks_linear(float(freqs.min()), float(freqs.max()), 7),
        "tick_step_y": axis_ticks_linear(float(np.min(mag_db)), float(np.max(mag_db)), 6),
    }


# ---------------------------
# Family 7: Spectrogram (tone switch)
# ---------------------------

def baseline_spectrogram(pp: Dict[str, Any]) -> Dict[str, Any]:
    gt = {
        "f1_hz": float(pp["f1_hz"]),
        "f2_hz": float(pp["f2_hz"]),
        "switch_time_s": float(pp["switch_time_s"]),
        "cp_duration_s": float(pp["duration_s"]),
    }
    return {k: quantize_value("spectrogram", k, v) for k, v in gt.items()}


def render_spectrogram(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fs = float(pp["fs_hz"])
    dur = float(pp["duration_s"])
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    ts = float(pp["switch_time_s"])
    noise = float(pp["noise"])

    t = np.arange(int(dur * fs)) / fs
    x = np.where(t < ts, np.sin(2 * np.pi * f1 * t), np.sin(2 * np.pi * f2 * t))
    rng = np.random.default_rng(meta.seed + 43)
    if noise > 0:
        x = x + rng.normal(0.0, noise, size=x.shape)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.specgram(x, NFFT=256, Fs=fs, noverlap=128)
    ax.set_title("Spectrogram (Tone Switch)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if meta.difficulty == "edge":
        # keep it harder: no grid
        pass

    save_figure(fig, out_path)
    plt.close(fig)

    return {"x_min": 0.0, "x_max": dur, "y_min": 0.0, "y_max": fs / 2.0,
            "tick_step_x": axis_ticks_linear(0.0, dur, 6),
            "tick_step_y": axis_ticks_linear(0.0, fs / 2.0, 6)}


# ---------------------------
# Family 8: IV Resistor (V vs I, slope = R)
# ---------------------------

def baseline_iv_resistor(pp: Dict[str, Any]) -> Dict[str, Any]:
    R = float(pp["R_ohm"])
    gt = {"resistance_ohm": R, "cp_slope_ohm": R}
    return {k: quantize_value("iv_resistor", k, v) for k, v in gt.items()}


def render_iv_resistor(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    R = float(pp["R_ohm"])
    i_max = float(pp["i_max_a"])
    n = int(pp["n_points"])

    I = np.linspace(0.0, i_max, n)
    V = R * I

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(I, V, linewidth=1.4)
    ax.set_title("IV Curve (Resistor) — V vs I")
    ax.set_xlabel("Current (A)")
    ax.set_ylabel("Voltage (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": float(I.min()), "x_max": float(I.max()),
        "y_min": float(V.min()), "y_max": float(V.max()),
        "tick_step_x": axis_ticks_linear(float(I.min()), float(I.max()), 6),
        "tick_step_y": axis_ticks_linear(float(V.min()), float(V.max()), 6),
    }


# ---------------------------
# Family 9: IV Diode (I vs V, includes target_current_a)
# ---------------------------

def baseline_iv_diode(pp: Dict[str, Any]) -> Dict[str, Any]:
    # I = I0 * exp((V - Vt)/k)
    Vt = float(pp["Vt"])
    k = float(pp["k"])
    I0 = float(pp["I0"])
    It = float(pp["target_current_a"])
    # Solve for V at target current
    v_at = Vt + k * math.log(It / max(I0, 1e-30))
    gt = {"target_current_a": It, "turn_on_voltage_v_at_target_i": v_at}
    return {k2: quantize_value("iv_diode", k2, v) for k2, v in gt.items()}


def render_iv_diode(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    Vt = float(pp["Vt"])
    k = float(pp["k"])
    I0 = float(pp["I0"])
    It = float(pp["target_current_a"])
    vmin = float(pp["v_min"])
    vmax = float(pp["v_max"])
    n = int(pp["n_points"])
    noise = float(pp["noise"])

    V = np.linspace(vmin, vmax, n)
    I = I0 * np.exp((V - Vt) / max(k, 1e-12))
    rng = np.random.default_rng(meta.seed + 59)
    if noise > 0:
        I = I * (1.0 + rng.normal(0.0, noise, size=I.shape))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(V, I, linewidth=1.4)
    ax.set_title("IV Curve (Diode) — I vs V")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    # Always show target current line for diode family (critical fairness)
    ax.axhline(It, linestyle="--", linewidth=1.0, alpha=0.8)
    ax.text(vmin, It, f"I_target={It:.3f} A", fontsize=9, va="bottom", ha="left")

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": float(V.min()), "x_max": float(V.max()),
        "y_min": float(np.min(I)), "y_max": float(np.max(I)),
        "tick_step_x": axis_ticks_linear(float(V.min()), float(V.max()), 6),
        "tick_step_y": axis_ticks_linear(float(np.min(I)), float(np.max(I)), 6),
    }


# ---------------------------
# Family 10: Transfer characteristic (saturating amp)
# ---------------------------

def baseline_transfer(pp: Dict[str, Any]) -> Dict[str, Any]:
    gain = float(pp["gain"])
    vsat = float(pp["Vsat"])
    vin_sat = vsat / max(gain, 1e-12)
    gt = {"small_signal_gain": gain, "saturation_v": vsat, "cp_vin_at_saturation": vin_sat}
    return {k: quantize_value("transfer_characteristic", k, v) for k, v in gt.items()}


def render_transfer(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    gain = float(pp["gain"])
    vsat = float(pp["Vsat"])
    vin_max = float(pp["vin_max"])
    n = int(pp["n_points"])

    Vin = np.linspace(-vin_max, vin_max, n)
    Vout = np.clip(gain * Vin, -vsat, vsat)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(Vin, Vout, linewidth=1.4)
    ax.set_title("Transfer Characteristic (Saturating Amplifier)")
    ax.set_xlabel("Vin (V)")
    ax.set_ylabel("Vout (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": float(Vin.min()), "x_max": float(Vin.max()),
        "y_min": float(Vout.min()), "y_max": float(Vout.max()),
        "tick_step_x": axis_ticks_linear(float(Vin.min()), float(Vin.max()), 7),
        "tick_step_y": axis_ticks_linear(float(Vout.min()), float(Vout.max()), 7),
    }


# ---------------------------
# Family 11: Pole-zero plot (single pole + single zero) — tick steps adjusted
# ---------------------------

def baseline_pole_zero(pp: Dict[str, Any]) -> Dict[str, Any]:
    gt = {
        "pole_real": float(pp["pole_real"]),
        "pole_imag": float(pp["pole_imag"]),
        "zero_real": float(pp["zero_real"]),
        "zero_imag": float(pp["zero_imag"]),
    }
    return {k: quantize_value("pole_zero", k, v) for k, v in gt.items()}


def render_pole_zero(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    pr, pi = float(pp["pole_real"]), float(pp["pole_imag"])
    zr, zi = float(pp["zero_real"]), float(pp["zero_imag"])
    lim = float(pp["axis_lim"])

    fig, ax = plt.subplots(figsize=(5.2, 5.2))
    ax.axhline(0.0, linewidth=1.0)
    ax.axvline(0.0, linewidth=1.0)

    ax.scatter([zr], [zi], marker="o", s=70, facecolors="none", edgecolors="black", linewidths=1.6)
    ax.scatter([pr], [pi], marker="x", s=70, color="black", linewidths=1.6)

    ax.set_title("Pole-Zero Plot")
    ax.set_xlabel("Real")
    ax.set_ylabel("Imag")
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    # Tick policy: 1–2 for clean/moderate; 5 for edge
    if meta.difficulty == "clean":
        step = 1.0
    elif meta.difficulty == "moderate":
        step = 2.0
    else:
        step = 5.0

    ticks = np.arange(-lim, lim + 1e-9, step)
    ax.set_xticks(ticks)
    ax.set_yticks(ticks)
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    return {"x_min": -lim, "x_max": lim, "y_min": -lim, "y_max": lim,
            "tick_step_x": step, "tick_step_y": step}

# ----------------------------
# Family: Stress–Strain Curve
# ----------------------------
def baseline_stress_strain(pp: Dict[str, Any]) -> Dict[str, float]:
    E_mpa = float(pp["E_gpa"]) * 1_000.0
    ys = float(pp["yield_strength_mpa"])
    uts = float(pp["uts_mpa"])
    eps_f = float(pp["fracture_strain"])
    eps_y = ys / E_mpa
    eps_uts = float(pp["uts_strain"])
    return {
        "yield_strength_mpa": float(quantize_value("stress_strain", "yield_strength_mpa", ys)),
        "uts_mpa": float(quantize_value("stress_strain", "uts_mpa", uts)),
        "fracture_strain": float(quantize_value("stress_strain", "fracture_strain", eps_f)),
        "cp_yield_strain": eps_y,
        "cp_uts_strain": eps_uts,
    }


def render_stress_strain(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    E_mpa = float(pp["E_gpa"]) * 1_000.0
    ys = float(pp["yield_strength_mpa"])
    uts = float(pp["uts_mpa"])
    eps_f = float(pp["fracture_strain"])
    eps_y = ys / E_mpa
    eps_uts = float(pp["uts_strain"])
    fs = float(pp["fracture_stress_mpa"])
    noise = float(pp.get("noise", 0.0))

    strain = np.linspace(0.0, eps_f, 500)
    stress = np.zeros_like(strain)

    m1 = strain <= eps_y
    stress[m1] = E_mpa * strain[m1]

    m2 = (strain > eps_y) & (strain <= eps_uts)
    stress[m2] = ys + (uts - ys) * (strain[m2] - eps_y) / max(1e-9, (eps_uts - eps_y))

    m3 = strain > eps_uts
    stress[m3] = uts + (fs - uts) * (strain[m3] - eps_uts) / max(1e-9, (eps_f - eps_uts))

    if noise > 0:
        stress = stress * (1.0 + noise * np.random.default_rng(meta.seed).normal(0, 0.6, size=stress.shape))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(strain, stress, linewidth=1.6)
    ax.set_title("Stress–Strain Curve")
    ax.set_xlabel("Strain")
    ax.set_ylabel("Stress (MPa)")
    if pp.get("show_markers", False):
        ax.plot([eps_y], [ys], "o", ms=5)
        ax.plot([eps_uts], [uts], "o", ms=5)

    x_max = eps_f * 1.05
    y_max = uts * 1.10
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)

    xt = 0.05 if meta.difficulty != "edge" else 0.10
    yt = 50.0 if meta.difficulty != "edge" else 100.0
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, xt))
    ax.set_yticks(np.arange(0.0, y_max + 1e-9, yt))
    if pp.get("grid", True):
        ax.grid(True, alpha=0.30)

    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": 0.0, "x_max": x_max, "y_min": 0.0, "y_max": y_max, "tick_step_x": xt, "tick_step_y": yt}


# ----------------------------
# Family: Torque–Speed Curve
# ----------------------------
def baseline_torque_speed(pp: Dict[str, Any]) -> Dict[str, float]:
    w0 = float(pp["no_load_speed_rpm"])
    Ts = float(pp["stall_torque_nm"])
    wq = float(pp["speed_q_rpm"])
    Tq = Ts * (1.0 - wq / max(1e-9, w0))
    return {
        "stall_torque_nm": float(quantize_value("torque_speed", "stall_torque_nm", Ts)),
        "no_load_speed_rpm": float(quantize_value("torque_speed", "no_load_speed_rpm", w0)),
        "cp_torque_at_speed_q_nm": Tq,
    }


def render_torque_speed(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    w0 = float(pp["no_load_speed_rpm"])
    Ts = float(pp["stall_torque_nm"])
    wq = float(pp["speed_q_rpm"])
    w = np.linspace(0.0, w0, 200)
    T = Ts * (1.0 - w / max(1e-9, w0))
    Tq = Ts * (1.0 - wq / max(1e-9, w0))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(w, T, lw=1.6)
    ax.plot([wq], [Tq], "o", ms=5)
    ax.set_title("Torque–Speed Curve")
    ax.set_xlabel("Speed (rpm)")
    ax.set_ylabel("Torque (N·m)")
    x_max = w0 * 1.05
    y_max = Ts * 1.20
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, 500.0))
    ax.set_yticks(np.arange(0.0, y_max + 1e-9, 2.0))
    if pp.get("grid", True):
        ax.grid(True, alpha=0.30)
    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": 0.0, "x_max": x_max, "y_min": 0.0, "y_max": y_max, "tick_step_x": 500.0, "tick_step_y": 2.0}


# ----------------------------
# Family: Pump Curve (Head vs Flow)
# ----------------------------
def baseline_pump_curve(pp: Dict[str, Any]) -> Dict[str, float]:
    H0 = float(pp["shutoff_head_m"])
    q_half = float(pp["q_at_half_head_m3h"])
    q_zero = float(pp["q_zero_head_m3h"])
    q_op = float(pp["q_op_m3h"])
    a = float(pp["a"])
    H_op = H0 - a * (q_op ** 2)
    return {
        "head_at_qop_m": float(quantize_value("pump_curve", "head_at_qop_m", H_op)),
        "q_at_half_head_m3h": float(quantize_value("pump_curve", "q_at_half_head_m3h", q_half)),
        "cp_shutoff_head_m": H0,
        "cp_qmax_m3h": q_zero,
    }


def render_pump_curve(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    H0 = float(pp["shutoff_head_m"])
    q_zero = float(pp["q_zero_head_m3h"])
    q_op = float(pp["q_op_m3h"])
    a = float(pp["a"])
    q = np.linspace(0.0, q_zero, 400)
    H = np.clip(H0 - a * (q ** 2), 0.0, None)
    H_op = float(H0 - a * (q_op ** 2))
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(q, H, lw=1.6)
    ax.plot([q_op], [H_op], "o", ms=5)
    ax.set_title("Pump Curve (Head vs Flow)")
    ax.set_xlabel("Flow Q (m³/h)")
    ax.set_ylabel("Head H (m)")
    x_max = q_zero * 1.05
    y_max = H0 * 1.10
    ax.set_xlim(0.0, x_max)
    ax.set_ylim(0.0, y_max)
    ax.set_xticks(np.arange(0.0, x_max + 1e-9, 10.0))
    ax.set_yticks(np.arange(0.0, y_max + 1e-9, 5.0))
    if pp.get("grid", True):
        ax.grid(True, alpha=0.30)
    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": 0.0, "x_max": x_max, "y_min": 0.0, "y_max": y_max, "tick_step_x": 10.0, "tick_step_y": 5.0}


# ----------------------------
# Family: S–N Curve (log-x)
# ----------------------------
def baseline_sn_curve(pp: Dict[str, Any]) -> Dict[str, float]:
    S1 = float(pp["stress_at_1e3_mpa"])
    Se = float(pp["endurance_limit_mpa"])
    b = float(np.log(Se / S1) / np.log(1e6 / 1e3))
    S_1e5 = float(S1 * ((1e5 / 1e3) ** b))
    return {
        "stress_at_1e5_mpa": float(quantize_value("sn_curve", "stress_at_1e5_mpa", S_1e5)),
        "endurance_limit_mpa": float(quantize_value("sn_curve", "endurance_limit_mpa", Se)),
        "cp_stress_at_1e3_mpa": S1,
    }


def render_sn_curve(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt
    from matplotlib.ticker import ScalarFormatter
    S1 = float(pp["stress_at_1e3_mpa"])
    Se = float(pp["endurance_limit_mpa"])
    b = float(np.log(Se / S1) / np.log(1e6 / 1e3))
    N = np.logspace(3, 7, 300)
    S = S1 * ((N / 1e3) ** b)
    S = np.where(N >= 1e6, Se, S)
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(N, S, lw=1.6)
    ax.set_xscale("log")
    ax.set_title("S–N Curve (Fatigue)")
    ax.set_xlabel("Cycles N (log scale)")
    ax.set_ylabel("Stress (MPa)")
    ax.set_xlim(1e3, 1e7)
    y_max = float(max(S1, Se) * 1.10)
    ax.set_ylim(0.0, y_max)
    xt = [1e3, 1e4, 1e5, 1e6, 1e7]
    ax.set_xticks(xt)
    ax.get_xaxis().set_major_formatter(ScalarFormatter())
    ax.set_xticklabels([f"1e{int(np.log10(t))}" for t in xt])
    yt = 50.0 if meta.difficulty != "edge" else 100.0
    ax.set_yticks(np.arange(0.0, y_max + 1e-9, yt))
    if pp.get("grid", True):
        ax.grid(True, which="both", alpha=0.30)
    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": 1e3, "x_max": 1e7, "y_min": 0.0, "y_max": y_max, "tick_step_x": None, "tick_step_y": yt}

# ---------------------------
# Orchestrator for all families
# ---------------------------

FAMILIES = [
    "step_response",
    "bode_magnitude",
    "bode_phase",
    "bandpass_response",
    "time_waveform",
    "fft_spectrum",
    "spectrogram",
    "iv_resistor",
    "iv_diode",
    "transfer_characteristic",
    "pole_zero",
    "stress_strain",
    "torque_speed",
    "pump_curve",
    "sn_curve",
]


def _mk_item(
    typ: str,
    idx: int,
    images_root: Path,
    out_dir: Path,
    meta: ItemMeta,
    pp: Dict[str, Any],
    gt: Dict[str, Any],
    question: str,
    axis_meta: Dict[str, Any],
) -> Dict[str, Any]:
    img_name = f"{typ}_{idx:03d}.png"
    rel_img = Path("images") / typ / img_name
    abs_img = images_root / typ / img_name

    # Render based on type
    if typ == "step_response":
        axis_meta = render_step_response(pp, abs_img, meta)
    elif typ == "bode_magnitude":
        axis_meta = render_bode_magnitude(pp, abs_img, meta)
    elif typ == "bode_phase":
        axis_meta = render_bode_phase(pp, abs_img, meta)
    elif typ == "bandpass_response":
        axis_meta = render_bandpass(pp, abs_img, meta)
    elif typ == "time_waveform":
        axis_meta = render_time_waveform(pp, abs_img, meta)
    elif typ == "fft_spectrum":
        axis_meta = render_fft(pp, abs_img, meta)
    elif typ == "spectrogram":
        axis_meta = render_spectrogram(pp, abs_img, meta)
    elif typ == "iv_resistor":
        axis_meta = render_iv_resistor(pp, abs_img, meta)
    elif typ == "iv_diode":
        axis_meta = render_iv_diode(pp, abs_img, meta)
    elif typ == "transfer_characteristic":
        axis_meta = render_transfer(pp, abs_img, meta)
    elif typ == "pole_zero":
        axis_meta = render_pole_zero(pp, abs_img, meta)
    elif typ == "stress_strain":
        axis_meta = render_stress_strain(pp, abs_img, meta)
    elif typ == "torque_speed":
        axis_meta = render_torque_speed(pp, abs_img, meta)
    elif typ == "pump_curve":
        axis_meta = render_pump_curve(pp, abs_img, meta)
    elif typ == "sn_curve":
        axis_meta = render_sn_curve(pp, abs_img, meta)
    else:
        raise ValueError(f"Unknown family: {typ}")

    # Separate final fields from checkpoint fields
    final_fields = [k for k in gt.keys() if not k.startswith("cp_")]
    checkpoint_fields = [k for k in gt.keys() if k.startswith("cp_")]
    
    return {
        "id": f"{typ}_{idx:03d}",
        "type": typ,
        "image_path": str(rel_img).replace("\\", "/"),
        "question": question,
        "ground_truth": gt,
        "plot_params": pp,
        "generation": {
            "seed": meta.seed,
            "difficulty": meta.difficulty,
            "edge_tag": meta.edge_tag,
            "final_fields": final_fields,
            "checkpoint_fields": checkpoint_fields,
            **axis_meta,
        },
    }


def generate_family(typ: str, out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    diff_seed = stable_int_seed(master_seed, "difficulty_plan", typ)
    diff_plan = make_difficulty_plan(n, diff_seed)

    # per-family unique pools where needed
    used_bandpass_f0: set[float] = set()

    for i in range(n):
        seed = stable_int_seed(master_seed, typ, i)
        rng = np.random.default_rng(seed)
        difficulty = diff_plan[i]
        edge_tag = ""

        if typ == "step_response":
            zeta_choices = [0.2, 0.3, 0.4, 0.5, 0.7]
            wn_choices = [4.0, 6.0, 8.0, 10.0, 12.0]
            zeta = float(rng.choice(zeta_choices))
            wn = float(rng.choice(wn_choices))
            # show enough time to settle
            t_end = float(6.0 / max(zeta * wn, 1e-12))
            n_points = 700 if difficulty == "clean" else (550 if difficulty == "moderate" else 450)
            pp = {"zeta": zeta, "wn_rad_s": wn, "t_end_s": t_end, "n_points": int(n_points)}
            gt = baseline_step_response(pp)
            q = (
                "From the step response plot, estimate:\n"
                "1) percent_overshoot (%)\n"
                "2) settling_time_s (2% criterion, seconds)\n"
                "3) steady_state (final value)\n"
                "Return numeric JSON."
            )

        elif typ == "bode_magnitude":
            K = float(rng.choice([0.1, 1.0, 10.0, 100.0]))  # tick-friendly dB
            fc = float(rng.choice([5, 10, 20, 50, 100, 200, 500]))
            fq = float(rng.choice([fc / 2, fc, fc * 2]))
            fmin, fmax = float(fc / 50), float(fc * 50)
            pp = {"K": K, "fc_hz": fc, "fq_hz": fq, "fmin_hz": fmin, "fmax_hz": fmax, "n_points": 600}
            gt = baseline_bode_magnitude(pp)
            q = (
                "From the Bode magnitude plot, estimate:\n"
                "1) dc_gain_db (dB)\n"
                "2) cutoff_hz (Hz) (−3 dB point)\n"
                "Return numeric JSON."
            )

        elif typ == "bode_phase":
            fc = float(rng.choice([5, 10, 20, 50, 100, 200, 500]))
            # Choose a target phase angle (human-friendly) and set fq so phase(fq)= -theta exactly.
            theta = float(rng.choice([15, 30, 45, 60, 75]))
            fq = float(fc * math.tan(math.radians(theta)))
            fmin, fmax = float(fc / 50), float(fc * 50)
            pp = {"fc_hz": fc, "fq_hz": fq, "theta_deg": theta, "fmin_hz": fmin, "fmax_hz": fmax, "n_points": 600}
            gt = baseline_bode_phase(pp)
            q = (
                "From the Bode phase plot, estimate:\n"
                "1) cutoff_hz (Hz)\n"
                "2) phase_deg_at_fq (deg) at the marked f_q\n"
                "Return numeric JSON."
            )

        elif typ == "bandpass_response":
            Q = float(rng.choice([1.5, 2.0, 3.0, 5.0, 8.0, 10.0]))
            # sample resonance without replacement (rounded to 0.1)
            while True:
                f0 = float(10 ** rng.uniform(math.log10(12.0), math.log10(1500.0)))
                f0 = float(np.round(f0, 1))  # 1 decimal resonance GT
                if f0 not in used_bandpass_f0:
                    used_bandpass_f0.add(f0)
                    break

            # frequency span around resonance
            fmin, fmax = float(max(1.0, f0 / 20.0)), float(f0 * 20.0)
            pp = {"f0_hz": f0, "Q": Q, "fmin_hz": fmin, "fmax_hz": fmax, "n_points": 700}
            gt = baseline_bandpass(pp)
            if difficulty == "edge":
                edge_tag = "no_grid_no_helpers"
            q = (
                "From the bandpass magnitude plot, estimate:\n"
                "1) resonance_hz (Hz)\n"
                "2) bandwidth_hz (Hz) using −3 dB points\n"
                "Return numeric JSON."
            )

        elif typ == "time_waveform":
            wave_type = str(rng.choice(["sine", "square", "triangle"]))
            f0 = float(rng.choice([5, 10, 20, 25, 40, 50, 80, 100, 120]))
            A = float(rng.choice([1, 2, 3, 4, 5]))
            duty = 0.5
            if wave_type == "square":
                duty = float(rng.choice([0.25, 0.33, 0.50, 0.66, 0.75]))

            if difficulty == "clean":
                cycles, fs = 4.0, 4000.0
            elif difficulty == "moderate":
                cycles, fs = 2.5, 3200.0
            else:
                cycles, fs = 1.1, 2600.0
                edge_tag = "short_window"

            t_end = float(cycles / max(f0, 1e-12))
            pp = {"wave_type": wave_type, "f0_hz": f0, "A": A, "fs_hz": fs, "t_end_s": t_end}
            if wave_type == "square":
                pp["duty"] = duty

            gt = baseline_time_waveform(pp)
            q = (
                "From the time-domain waveform plot, estimate:\n"
                "1) frequency_hz (Hz)\n"
                "2) vpp_v (V)\n"
                "Return numeric JSON."
            )

        elif typ == "fft_spectrum":
            fs = 2048.0
            N = 2048
            f_choices = [20, 40, 50, 60, 80, 100, 120, 150, 180, 200, 240, 300, 360, 400, 480, 600]
            f1, f2 = rng.choice(f_choices, size=2, replace=False)
            f1, f2 = float(min(f1, f2)), float(max(f1, f2))

            # choose clean integer-ish ratios to reduce "a/b" formatting
            if difficulty == "clean":
                a1, a2, noise = 1.0, 0.5, 0.0
            elif difficulty == "moderate":
                a1, a2, noise = 1.0, float(rng.choice([0.5, 0.25])), 0.01
                edge_tag = "light_noise"
            else:
                a1, a2, noise = 1.0, float(rng.choice([0.5, 0.25, 0.2])), 0.03
                edge_tag = "no_grid_noisy"

            pp = {"fs_hz": fs, "N": N, "f1_hz": f1, "f2_hz": f2, "a1": float(a1), "a2": float(a2), "noise": float(noise)}
            gt = baseline_fft(pp)
            q = (
                "From the FFT magnitude plot, estimate:\n"
                "1) dominant_frequency_hz (Hz)\n"
                "2) secondary_frequency_hz (Hz)\n"
                "Return numeric JSON."
            )

        elif typ == "spectrogram":
            fs = 2000.0 if difficulty != "edge" else 1600.0
            duration = 2.0 if difficulty != "edge" else 1.6
            switch_time = 1.0 if difficulty != "edge" else 0.8
            f_choices = [50, 80, 100, 120, 150, 200, 240, 300, 360]
            f1, f2 = rng.choice(f_choices, size=2, replace=False)
            noise = 0.0 if difficulty == "clean" else (0.03 if difficulty == "moderate" else 0.06)
            if difficulty == "edge":
                edge_tag = "shorter_noisier"
            pp = {"fs_hz": fs, "duration_s": duration, "switch_time_s": switch_time, "f1_hz": float(f1), "f2_hz": float(f2), "noise": float(noise)}
            gt = baseline_spectrogram(pp)
            q = (
                "From the spectrogram, estimate:\n"
                "1) f1_hz (Hz) before the switch\n"
                "2) f2_hz (Hz) after the switch\n"
                "3) switch_time_s (s)\n"
                "Return numeric JSON."
            )

        elif typ == "iv_resistor":
            R = float(rng.choice([50, 100, 200, 300, 500]))
            i_max = 0.02
            n_points = 60
            if difficulty == "edge":
                edge_tag = "no_grid"
            pp = {"R_ohm": R, "i_max_a": float(i_max), "n_points": int(n_points)}
            gt = baseline_iv_resistor(pp)
            q = (
                "From the resistor IV plot (Voltage vs Current), estimate:\n"
                "1) resistance_ohm (Ohm)\n"
                "Return numeric JSON."
            )

        elif typ == "iv_diode":
            # Choose target I and a human-friendly V at that current; derive I0 so it lands exactly.
            target_I = float(rng.choice([0.002, 0.005, 0.010, 0.020]))
            v_at_target = float(rng.choice([0.55, 0.60, 0.65, 0.70]))
            Vt = 0.40
            k = float(rng.choice([0.05, 0.06]))
            I0 = target_I / math.exp((v_at_target - Vt) / k)
            noise = 0.0 if difficulty == "clean" else (0.03 if difficulty == "moderate" else 0.06)
            if difficulty == "edge":
                edge_tag = "no_grid_more_noise"
            pp = {"Vt": float(Vt), "k": float(k), "I0": float(I0),
                  "target_current_a": float(target_I),
                  "v_min": 0.0, "v_max": 0.9, "n_points": 220, "noise": float(noise)}
            gt = baseline_iv_diode(pp)
            q = (
                f"From the diode IV plot, a horizontal line shows I_target = {target_I:.3f} A.\n"
                "Estimate the turn-on voltage at that current:\n"
                "1) target_current_a (A)\n"
                "2) turn_on_voltage_v_at_target_i (V)\n"
                "Return numeric JSON."
            )

        elif typ == "transfer_characteristic":
            gain = float(rng.choice([2, 3, 5, 10]))
            vsat = float(rng.choice([2, 3, 5, 8]))
            vin_max = float(max(1.2, (vsat / gain) * 2.5))
            n_points = 200
            if difficulty == "edge":
                edge_tag = "no_grid"
            pp = {"gain": gain, "Vsat": vsat, "vin_max": vin_max, "n_points": int(n_points)}
            gt = baseline_transfer(pp)
            q = (
                "From the transfer characteristic plot, estimate:\n"
                "1) small_signal_gain\n"
                "2) saturation_v (V)\n"
                "Return numeric JSON."
            )

        elif typ == "pole_zero":
            # Single pole + single zero, integer coordinates
            pole_real = float(rng.choice([-8, -6, -5, -4, -3, -2, -1]))
            pole_imag = float(rng.choice([0, 1, -1, 2, -2, 3, -3, 4, -4]))
            zero_real = float(rng.choice([-8, -6, -5, -4, -3, -2, -1]))
            zero_imag = float(rng.choice([0, 1, -1, 2, -2, 3, -3, 4, -4]))
            axis_lim = 10.0
            if difficulty == "edge":
                edge_tag = "coarse_ticks"
            pp = {"pole_real": pole_real, "pole_imag": pole_imag, "zero_real": zero_real, "zero_imag": zero_imag, "axis_lim": axis_lim}
            gt = baseline_pole_zero(pp)
            q = (
                "From the pole-zero plot, read the coordinates of the single pole (x) and single zero (o):\n"
                "1) pole_real\n"
                "2) pole_imag\n"
                "3) zero_real\n"
                "4) zero_imag\n"
                "Return numeric JSON."
            )

        elif typ == "stress_strain":
            E_gpa = 200.0
            ys = float(rng.choice([200.0, 250.0, 300.0, 350.0, 400.0, 450.0, 500.0]))
            uts = float(rng.choice([ys + 100.0, ys + 150.0, ys + 200.0]))
            eps_f = float(rng.choice([0.15, 0.20, 0.25, 0.30]))
            eps_uts = float(rng.choice([0.08, 0.10, 0.12, 0.15]))
            if eps_uts >= eps_f - 0.03:
                eps_uts = max(0.06, eps_f - 0.05)
            fs = float(round((0.75 * uts) / 25.0) * 25.0)
            if difficulty == "edge":
                edge_tag = "no_grid_noise"
            pp = {
                "E_gpa": E_gpa, "yield_strength_mpa": ys, "uts_mpa": uts,
                "fracture_strain": eps_f, "uts_strain": eps_uts, "fracture_stress_mpa": fs,
                "grid": (difficulty != "edge"), "show_markers": (difficulty == "clean"),
                "noise": (0.0 if difficulty != "edge" else 0.01),
            }
            gt = baseline_stress_strain(pp)
            q = (
                "From the stress-strain curve, estimate:\n"
                "1) yield_strength_mpa (MPa)\n"
                "2) uts_mpa (MPa, ultimate tensile strength)\n"
                "3) fracture_strain\n"
                "Return numeric JSON."
            )

        elif typ == "torque_speed":
            w0 = float(rng.choice([1000.0, 1200.0, 1500.0, 1800.0, 2000.0, 2200.0, 2500.0, 2800.0, 3000.0]))
            Ts = float(rng.choice([2.0, 3.0, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0]))
            wq = float(round((0.6 * w0) / 100.0) * 100.0)
            if difficulty == "edge":
                edge_tag = "no_grid"
            pp = {"no_load_speed_rpm": w0, "stall_torque_nm": Ts, "speed_q_rpm": wq, "grid": (difficulty != "edge")}
            gt = baseline_torque_speed(pp)
            q = (
                "From the torque-speed curve, estimate:\n"
                "1) stall_torque_nm (N·m)\n"
                "2) no_load_speed_rpm (rpm)\n"
                "Return numeric JSON."
            )

        elif typ == "pump_curve":
            H0 = float(rng.choice([35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0, 70.0, 75.0]))
            q_half = float(rng.choice([25.0, 30.0, 35.0, 40.0, 45.0, 50.0, 55.0, 60.0, 65.0]))
            a = (H0 / 2.0) / max(1e-9, (q_half ** 2))
            q_zero = float(np.sqrt(H0 / a))
            q_op = float(q_half / 2.0)
            if difficulty == "edge":
                edge_tag = "no_grid"
            pp = {"shutoff_head_m": H0, "q_at_half_head_m3h": q_half, "q_zero_head_m3h": q_zero, "q_op_m3h": q_op, "a": a, "grid": (difficulty != "edge")}
            gt = baseline_pump_curve(pp)
            q = (
                "From the pump curve (head vs flow), estimate:\n"
                "1) head_at_qop_m (m) at the operating point\n"
                "2) q_at_half_head_m3h (m³/h) flow at half shutoff head\n"
                "Return numeric JSON."
            )

        elif typ == "sn_curve":
            S1 = float(rng.choice([450.0, 500.0, 550.0, 600.0]))
            Se = float(rng.choice([200.0, 225.0, 250.0, 275.0, 300.0]))
            if difficulty == "edge":
                edge_tag = "no_grid"
            pp = {"stress_at_1e3_mpa": S1, "endurance_limit_mpa": Se, "knee_cycles": 1e6, "grid": (difficulty != "edge")}
            gt = baseline_sn_curve(pp)
            q = (
                "From the S-N curve (fatigue), estimate:\n"
                "1) stress_at_1e5_mpa (MPa) at 10^5 cycles\n"
                "2) endurance_limit_mpa (MPa)\n"
                "Return numeric JSON."
            )

        else:
            raise ValueError(f"Unhandled family: {typ}")

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        axis_meta: Dict[str, Any] = {}
        item = _mk_item(typ, i, images_root, out_dir, meta, pp, gt, q, axis_meta)
        items.append(item)

    return items


def baseline_for_item(it: Dict[str, Any]) -> Dict[str, Any]:
    typ = str(it["type"])
    pp = dict(it["plot_params"])
    if typ == "step_response":
        return baseline_step_response(pp)
    if typ == "bode_magnitude":
        return baseline_bode_magnitude(pp)
    if typ == "bode_phase":
        return baseline_bode_phase(pp)
    if typ == "bandpass_response":
        return baseline_bandpass(pp)
    if typ == "time_waveform":
        return baseline_time_waveform(pp)
    if typ == "fft_spectrum":
        return baseline_fft(pp)
    if typ == "spectrogram":
        return baseline_spectrogram(pp)
    if typ == "iv_resistor":
        return baseline_iv_resistor(pp)
    if typ == "iv_diode":
        return baseline_iv_diode(pp)
    if typ == "transfer_characteristic":
        return baseline_transfer(pp)
    if typ == "pole_zero":
        return baseline_pole_zero(pp)
    if typ == "stress_strain":
        return baseline_stress_strain(pp)
    if typ == "torque_speed":
        return baseline_torque_speed(pp)
    if typ == "pump_curve":
        return baseline_pump_curve(pp)
    if typ == "sn_curve":
        return baseline_sn_curve(pp)
    raise ValueError(f"Unknown type: {typ}")


def validate_items(items: List[Dict[str, Any]], out_dir: Path) -> None:
    rows: List[Dict[str, Any]] = []
    for it in items:
        typ = str(it["type"])
        iid = str(it["id"])
        gt = dict(it["ground_truth"])
        pred = baseline_for_item(it)
        for k, gv in gt.items():
            pv = pred.get(k, None)
            ok = False
            ae = None
            if pv is not None:
                ae = float(abs(float(pv) - float(gv)))
                ok = float_close(float(pv), float(gv), abs_tol=1e-9, rel_tol=1e-9)
            rows.append({"type": typ, "id": iid, "field": k, "pass": ok, "abs_err": ae})

    rows_path = out_dir / "validation_rows.csv"
    with rows_path.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "id", "field", "pass", "abs_err"])
        w.writeheader()
        w.writerows(rows)

    # summary
    import pandas as pd
    df = pd.DataFrame(rows)
    summ = (
        df.groupby(["type", "field"])
          .agg(n=("pass", "size"), pass_rate=("pass", "mean"), max_abs_err=("abs_err", "max"))
          .reset_index()
          .sort_values(["type", "field"])
    )
    summ_path = out_dir / "validation_summary.csv"
    summ.to_csv(summ_path, index=False)
    overall = float(df["pass"].mean()) if len(df) else 0.0
    print(f"[validate] pass_rate={overall*100:.1f}%  rows={len(df)}")
    print(f"[validate] wrote: {rows_path}")
    print(f"[validate] wrote: {summ_path}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/plotchain_v4", help="Output directory")
    ap.add_argument("--n_per_family", type=int, default=30, help="Items per plot family")
    ap.add_argument("--seed", type=int, default=0, help="Master seed")
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    images_root = out_dir / "images"
    images_root.mkdir(parents=True, exist_ok=True)

    all_items: List[Dict[str, Any]] = []
    for typ in FAMILIES:
        items = generate_family(typ, out_dir=out_dir, images_root=images_root, master_seed=args.seed, n=args.n_per_family)
        print(f"[gen] {typ}: n={len(items)}")
        all_items.extend(items)

    jsonl_path = out_dir / "plotchain_v4.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it in all_items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")
    print(f"[write] {jsonl_path} ({len(all_items)} items)")

    validate_items(all_items, out_dir)


if __name__ == "__main__":
    main()
