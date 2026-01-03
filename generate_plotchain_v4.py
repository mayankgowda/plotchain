#!/usr/bin/env python3
# generate_plotchain_v4.py
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

# --- matplotlib (headless safe) ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt


# =========================
# Global style (fairness)
# =========================
FIGSIZE = (6.0, 3.6)
DPI = 200
LINEWIDTH = 2.0
FONTSIZE = 10


def set_mpl_style() -> None:
    plt.rcParams.update({
        "figure.dpi": DPI,
        "savefig.dpi": DPI,
        "font.size": FONTSIZE,
        "axes.titlesize": FONTSIZE + 1,
        "axes.labelsize": FONTSIZE,
        "xtick.labelsize": FONTSIZE - 1,
        "ytick.labelsize": FONTSIZE - 1,
        "lines.linewidth": LINEWIDTH,
    })


# =========================
# Common utilities
# =========================
def stable_int_seed(master_seed: int, family: str, idx: int) -> int:
    s = f"{master_seed}|{family}|{idx}".encode("utf-8")
    h = hashlib.sha256(s).hexdigest()
    return int(h[:16], 16)  # 64-bit int


def difficulty_plan(n: int) -> List[str]:
    # 60% clean, 30% moderate, 10% edge (deterministic ordering)
    n_clean = int(round(0.6 * n))
    n_mod = int(round(0.3 * n))
    n_edge = max(0, n - n_clean - n_mod)
    return (["clean"] * n_clean) + (["moderate"] * n_mod) + (["edge"] * n_edge)


def axis_ticks_linear(xmin: float, xmax: float, n: int = 6) -> List[float]:
    if n < 2:
        return [xmin, xmax]
    step = (xmax - xmin) / (n - 1)
    return [xmin + i * step for i in range(n)]


def write_jsonl(path: Path, items: List[Dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")


def float_close(a: float, b: float, abs_tol: float = 1e-12, rel_tol: float = 1e-12) -> bool:
    return abs(a - b) <= max(abs_tol, rel_tol * max(abs(a), abs(b), 1e-12))


def save_fig(fig, out_path: Path) -> None:
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, bbox_inches="tight")
    plt.close(fig)


@dataclass(frozen=True)
class ItemMeta:
    difficulty: str
    edge_tag: str
    seed: int


# =========================
# Family registry
# =========================
FAMILIES = [
    "step_response",
    "bode_magnitude",
    "bode_phase",
    "bandpass_response",
    "time_waveform",
    "fft_spectrum",
    "spectrogram",
    "iv_curve",
    "transfer_characteristic",
    "pole_zero",
]


# =========================
# 1) Step response
# =========================
def step_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    K = float(pp["K"])

    # Overshoot (standard formula)
    if zeta >= 1.0:
        os = 0.0
    else:
        os = math.exp(-zeta * math.pi / math.sqrt(1.0 - zeta**2)) * 100.0

    # 2% settling time approximation (deterministic, canonical)
    ts = 4.0 / (max(zeta * wn, 1e-12))

    # Peak time and peak value (for checkpoints)
    if zeta >= 1.0:
        tp = float("nan")
        ypk = K
    else:
        wd = wn * math.sqrt(1.0 - zeta**2)
        tp = math.pi / max(wd, 1e-12)
        ypk = K * (1.0 + os / 100.0)

    return {
        "percent_overshoot": float(os),
        "settling_time_s": float(ts),
        "steady_state": float(K),
        "cp_peak_time_s": float(tp),
        "cp_peak_value": float(ypk),
        "cp_band_lower": float(0.98 * K),
        "cp_band_upper": float(1.02 * K),
    }


def step_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    K = float(pp["K"])

    # Analytic step response for underdamped case
    t_end = float(pp["t_end_s"])
    n = int(pp["n_points"])
    t = np.linspace(0.0, t_end, n)

    if zeta >= 1.0:
        # overdamped-ish placeholder (still deterministic)
        y = K * (1.0 - np.exp(-wn * t))
    else:
        wd = wn * math.sqrt(1.0 - zeta**2)
        phi = math.atan2(math.sqrt(1.0 - zeta**2), zeta)
        y = K * (1.0 - (1.0 / math.sqrt(1.0 - zeta**2)) * np.exp(-zeta * wn * t) * np.sin(wd * t + phi))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t, y)
    ax.set_title("2nd-Order Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)
    else:
        ax.grid(False)

    save_fig(fig, out_path)
    return {
        "x_min": 0.0, "x_max": t_end,
        "y_min": float(np.min(y)), "y_max": float(np.max(y)),
        "tick_step_x": axis_ticks_linear(0.0, t_end, 6),
        "tick_step_y": None,
    }


def gen_step(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    items = []
    plan = difficulty_plan(n)

    zetas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    wns = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]

    fam = "step_response"
    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        zeta = float(rng.choice(zetas))
        wn = float(rng.choice(wns))
        K = 1.0

        ts = 4.0 / (zeta * wn)
        if difficulty == "clean":
            t_end = 6.0 * ts
            n_points = 600
        elif difficulty == "moderate":
            t_end = 4.5 * ts
            n_points = 450
        else:
            edge_tag = "short_window"
            t_end = 3.0 * ts
            n_points = 320

        pp = {"zeta": zeta, "wn_rad_s": wn, "K": K, "t_end_s": float(t_end), "n_points": int(n_points)}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = step_render(pp, abs_img, meta)
        gt = step_baseline(pp)

        q = (
            "From the step response plot, estimate:\n"
            "1) percent_overshoot (%)\n"
            "2) settling_time_s (2% criterion, seconds)\n"
            "3) steady_state (final value)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["percent_overshoot", "settling_time_s", "steady_state"],
                           "checkpoint_fields": ["cp_peak_time_s", "cp_peak_value", "cp_band_lower", "cp_band_upper"],
                           **axis_meta},
        })
    return items


# =========================
# 2) Bode magnitude
# =========================
LOG_TICKS = [10, 20, 50, 100, 200, 500, 1000, 2000, 5000, 10000]


def bode_mag_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    dc = 20.0 * math.log10(max(K, 1e-12))
    return {
        "dc_gain_db": float(dc),
        "cutoff_hz": float(fc),
        "cp_mag_at_fc_db": float(dc - 3.0103),
        "cp_slope_db_per_decade": float(-20.0),
    }


def bode_mag_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_points = int(pp["n_points"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    mag = 20.0 * np.log10(K / np.sqrt(1.0 + (f / fc) ** 2))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(f, mag)
    ax.set_title("Bode Magnitude (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")

    ax.set_xticks([x for x in LOG_TICKS if fmin <= x <= fmax])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, out_path)
    return {"x_min": fmin, "x_max": fmax, "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
            "tick_step_x": None, "tick_step_y": None}


def gen_bode_mag(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "bode_magnitude"
    items = []
    plan = difficulty_plan(n)

    Ks = [0.5, 1.0, 2.0, 5.0, 10.0]
    fcs = [20, 50, 100, 200, 500, 1000, 2000, 5000]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        K = float(rng.choice(Ks))
        fc = float(rng.choice(fcs))

        if difficulty == "clean":
            fmin, fmax = fc / 10.0, fc * 10.0
            n_points = 600
        elif difficulty == "moderate":
            fmin, fmax = fc / 7.0, fc * 12.0
            n_points = 450
        else:
            edge_tag = "narrow_span"
            fmin, fmax = fc / 4.0, fc * 4.0
            n_points = 300

        pp = {"K": K, "fc_hz": fc, "fmin_hz": float(fmin), "fmax_hz": float(fmax), "n_points": int(n_points)}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = bode_mag_render(pp, abs_img, meta)
        gt = bode_mag_baseline(pp)

        q = (
            "From the Bode magnitude plot, estimate:\n"
            "1) dc_gain_db (dB)\n"
            "2) cutoff_hz (Hz) (−3 dB point)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["dc_gain_db", "cutoff_hz"],
                           "checkpoint_fields": ["cp_mag_at_fc_db", "cp_slope_db_per_decade"],
                           **axis_meta},
        })
    return items


# =========================
# 3) Bode phase
# =========================
def bode_phase_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    fc = float(pp["fc_hz"])
    phase_10 = -math.degrees(math.atan(10.0))
    return {
        "cutoff_hz": float(fc),
        "phase_deg_at_10fc": float(phase_10),
        "cp_phase_deg_at_fc": float(-45.0),
    }


def bode_phase_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_points = int(pp["n_points"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    phase = -np.degrees(np.arctan(f / fc))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(f, phase)
    ax.set_title("Bode Phase (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (deg)")

    ax.set_xticks([x for x in LOG_TICKS if fmin <= x <= fmax])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, out_path)
    return {"x_min": fmin, "x_max": fmax, "y_min": float(np.min(phase)), "y_max": float(np.max(phase)),
            "tick_step_x": None, "tick_step_y": None}


def gen_bode_phase(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "bode_phase"
    items = []
    plan = difficulty_plan(n)
    fcs = [20, 50, 100, 200, 500, 1000, 2000, 5000]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        fc = float(rng.choice(fcs))
        if difficulty == "clean":
            fmin, fmax = fc / 10.0, fc * 10.0
            n_points = 600
        elif difficulty == "moderate":
            fmin, fmax = fc / 7.0, fc * 12.0
            n_points = 450
        else:
            edge_tag = "narrow_span"
            fmin, fmax = fc / 4.0, fc * 4.0
            n_points = 300

        pp = {"fc_hz": fc, "fmin_hz": float(fmin), "fmax_hz": float(fmax), "n_points": int(n_points)}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = bode_phase_render(pp, abs_img, meta)
        gt = bode_phase_baseline(pp)

        q = (
            "From the Bode phase plot, estimate:\n"
            "1) cutoff_hz (Hz)\n"
            "2) phase_deg_at_10fc (deg)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["cutoff_hz", "phase_deg_at_10fc"],
                           "checkpoint_fields": ["cp_phase_deg_at_fc"],
                           **axis_meta},
        })
    return items


# =========================
# 4) Bandpass response (log-symmetric parabola)
# =========================
def bandpass_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    f1 = float(pp["f1_3db_hz"])
    f2 = float(pp["f2_3db_hz"])
    f0 = math.sqrt(f1 * f2)
    bw = f2 - f1
    q = f0 / max(bw, 1e-12)
    return {
        "resonance_hz": float(f0),
        "bandwidth_hz": float(bw),
        "cp_f1_3db_hz": float(f1),
        "cp_f2_3db_hz": float(f2),
        "cp_q_factor": float(q),
    }


def bandpass_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    f1 = float(pp["f1_3db_hz"])
    f2 = float(pp["f2_3db_hz"])
    f0 = math.sqrt(f1 * f2)

    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    n_points = int(pp["n_points"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), n_points)
    logf = np.log10(f)
    logf0 = math.log10(f0)
    logf1 = math.log10(f1)

    # log-parabola: mag_db(f1)= -3, mag_db(f0)=0, symmetric -> mag_db(f2)=-3
    a = 3.0 / max((logf1 - logf0) ** 2, 1e-12)
    mag_db = -a * (logf - logf0) ** 2

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.semilogx(f, mag_db)
    ax.set_title("Bandpass Magnitude Response (synthetic)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")

    ax.set_xticks([x for x in LOG_TICKS if fmin <= x <= fmax])
    ax.get_xaxis().set_major_formatter(matplotlib.ticker.ScalarFormatter())

    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    save_fig(fig, out_path)
    return {"x_min": fmin, "x_max": fmax, "y_min": float(np.min(mag_db)), "y_max": float(np.max(mag_db)),
            "tick_step_x": None, "tick_step_y": None}


def gen_bandpass(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "bandpass_response"
    items = []
    plan = difficulty_plan(n)

    tick = [20, 30, 50, 80, 100, 150, 200, 300, 500, 800, 1000, 1500, 2000, 3000, 5000]
    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        f1 = float(rng.choice(tick))
        # pick f2 > f1 with reasonable ratio
        candidates = [x for x in tick if x > f1 and (x / f1) <= 8.0]
        if not candidates:
            candidates = [x for x in tick if x > f1]
        f2 = float(rng.choice(candidates))

        if difficulty == "clean":
            fmin, fmax = f1 / 2.0, f2 * 2.0
            n_points = 700
        elif difficulty == "moderate":
            fmin, fmax = f1 / 1.7, f2 * 1.7
            n_points = 520
        else:
            edge_tag = "tight_span"
            fmin, fmax = f1 / 1.3, f2 * 1.3
            n_points = 380

        pp = {"f1_3db_hz": f1, "f2_3db_hz": f2, "fmin_hz": float(fmin), "fmax_hz": float(fmax), "n_points": int(n_points)}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = bandpass_render(pp, abs_img, meta)
        gt = bandpass_baseline(pp)

        q = (
            "From the bandpass magnitude plot, estimate:\n"
            "1) resonance_hz (Hz)\n"
            "2) bandwidth_hz (Hz)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["resonance_hz", "bandwidth_hz"],
                           "checkpoint_fields": ["cp_f1_3db_hz", "cp_f2_3db_hz", "cp_q_factor"],
                           **axis_meta},
        })
    return items


# =========================
# 5) Time waveform
# =========================
def time_wave_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    wave = str(pp["wave_type"])
    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    gt = {
        "frequency_hz": float(f0),
        "vpp_v": float(2.0 * A),
        "cp_period_s": float(1.0 / max(f0, 1e-12)),
        "cp_vmax_v": float(A),
        "cp_vmin_v": float(-A),
    }
    if wave == "square":
        gt["cp_duty"] = float(pp["duty"])
    return gt


def _wave(t: np.ndarray, wave_type: str, f0: float, A: float, duty: float) -> np.ndarray:
    phase = (t * f0) % 1.0
    if wave_type == "sine":
        return A * np.sin(2 * np.pi * f0 * t)
    if wave_type == "square":
        return np.where(phase < duty, A, -A)
    if wave_type == "triangle":
        return A * (4 * np.abs(phase - 0.5) - 1.0)
    raise ValueError(wave_type)


def time_wave_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    wave = str(pp["wave_type"])
    duty = float(pp.get("duty", 0.5))
    t_end = float(pp["t_end_s"])
    fs = float(pp["fs_hz"])

    t = np.arange(0.0, t_end, 1.0 / fs)
    y = _wave(t, wave, f0, A, duty)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(t, y)
    ax.set_title(f"Time Waveform ({wave})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_fig(fig, out_path)
    return {"x_min": 0.0, "x_max": t_end, "y_min": float(np.min(y)), "y_max": float(np.max(y)),
            "tick_step_x": axis_ticks_linear(0.0, t_end, 6),
            "tick_step_y": axis_ticks_linear(float(np.min(y)), float(np.max(y)), 6)}


def gen_time_wave(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "time_waveform"
    items = []
    plan = difficulty_plan(n)

    f0s = [2, 5, 10, 20, 40, 60, 80, 100, 120]
    As = [0.5, 1.0, 2.0, 3.0, 5.0]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        wave = str(rng.choice(["sine", "square", "triangle"]))
        f0 = float(rng.choice(f0s))
        A = float(rng.choice(As))
        duty = 0.5

        if difficulty == "clean":
            cycles = float(rng.uniform(2.5, 6.0))
            fs = 4000.0
        elif difficulty == "moderate":
            cycles = float(rng.uniform(1.5, 3.5))
            fs = 3500.0
        else:
            edge_tag = "short_window"
            cycles = float(rng.uniform(0.6, 1.1))
            fs = 3000.0

        if wave == "square":
            duty = float(rng.uniform(0.2, 0.8))

        t_end = float(cycles / max(f0, 1e-12))

        pp = {"wave_type": wave, "f0_hz": f0, "A": A, "fs_hz": fs, "t_end_s": t_end}
        if wave == "square":
            pp["duty"] = duty

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = time_wave_render(pp, abs_img, meta)
        gt = time_wave_baseline(pp)

        # IMPORTANT: checkpoint fields per-item (cp_duty only for square)
        cp_fields = ["cp_period_s", "cp_vmax_v", "cp_vmin_v"]
        if wave == "square":
            cp_fields.append("cp_duty")

        q = (
            "From the time-domain waveform plot, estimate:\n"
            "1) frequency_hz (Hz)\n"
            "2) vpp_v (V)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["frequency_hz", "vpp_v"],
                           "checkpoint_fields": cp_fields,
                           **axis_meta},
        })
    return items


# =========================
# 6) FFT spectrum (bin-aligned)
# =========================
def fft_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    A1 = float(pp["A1"])
    A2 = float(pp["A2"])
    # dominant is higher amplitude
    if A1 >= A2:
        dom, sec = f1, f2
        ratio = A1 / max(A2, 1e-12)
    else:
        dom, sec = f2, f1
        ratio = A2 / max(A1, 1e-12)
    return {
        "dominant_frequency_hz": float(dom),
        "secondary_frequency_hz": float(sec),
        "cp_peak_ratio": float(ratio),
    }


def fft_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    fs = float(pp["fs_hz"])
    N = int(pp["N"])
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    A1 = float(pp["A1"])
    A2 = float(pp["A2"])

    n = np.arange(N)
    t = n / fs
    x = A1 * np.sin(2 * np.pi * f1 * t) + A2 * np.sin(2 * np.pi * f2 * t)

    X = np.fft.rfft(x)
    freqs = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = np.abs(X)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(freqs, mag)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("|X(f)|")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.3)

    save_fig(fig, out_path)
    return {"x_min": float(freqs[0]), "x_max": float(freqs[-1]), "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
            "tick_step_x": None, "tick_step_y": None}


def gen_fft(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "fft_spectrum"
    items = []
    plan = difficulty_plan(n)

    fs = 2000.0
    N = 2048

    # choose integer bins so f aligns exactly -> no leakage
    bins = list(range(10, 250, 10))  # 10,20,...,240
    amps = [(1.0, 0.6), (1.0, 0.4), (0.8, 0.5)]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        k1 = int(rng.choice(bins))
        k2 = int(rng.choice([k for k in bins if abs(k - k1) >= 20]))
        f1 = k1 * fs / N
        f2 = k2 * fs / N
        A1, A2 = rng.choice(amps)

        if difficulty == "edge":
            edge_tag = "close_peaks"
            # force closer peaks
            k2 = k1 + int(rng.choice([10, 20]))
            f2 = k2 * fs / N

        pp = {"fs_hz": fs, "N": N, "f1_hz": float(f1), "f2_hz": float(f2), "A1": float(A1), "A2": float(A2)}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = fft_render(pp, abs_img, meta)
        gt = fft_baseline(pp)

        q = (
            "From the FFT magnitude plot, estimate:\n"
            "1) dominant_frequency_hz (Hz)\n"
            "2) secondary_frequency_hz (Hz)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["dominant_frequency_hz", "secondary_frequency_hz"],
                           "checkpoint_fields": ["cp_peak_ratio"],
                           **axis_meta},
        })
    return items


# =========================
# 7) Spectrogram (two-tone switch)
# =========================
def spec_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    return {
        "f1_hz": float(pp["f1_hz"]),
        "f2_hz": float(pp["f2_hz"]),
        "switch_time_s": float(pp["switch_time_s"]),
        "cp_duration_s": float(pp["duration_s"]),
    }


def spec_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    fs = float(pp["fs_hz"])
    duration = float(pp["duration_s"])
    t0 = float(pp["switch_time_s"])
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])

    t = np.arange(0.0, duration, 1.0 / fs)
    x = np.where(t < t0, np.sin(2 * np.pi * f1 * t), np.sin(2 * np.pi * f2 * t))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.specgram(x, NFFT=256, Fs=fs, noverlap=128)
    ax.set_title("Spectrogram (two-tone switch)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    if meta.difficulty != "edge":
        ax.grid(False)

    save_fig(fig, out_path)
    return {"x_min": 0.0, "x_max": duration, "y_min": 0.0, "y_max": fs / 2.0,
            "tick_step_x": axis_ticks_linear(0.0, duration, 6),
            "tick_step_y": None}


def gen_spectrogram(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "spectrogram"
    items = []
    plan = difficulty_plan(n)

    fs = 2000.0
    duration = 1.0
    freqs = [50, 80, 100, 150, 200, 300, 400, 500, 600]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        f1 = float(rng.choice(freqs))
        f2 = float(rng.choice([x for x in freqs if x != f1]))
        if difficulty == "edge":
            edge_tag = "late_switch"
            t0 = float(rng.uniform(0.75, 0.9))
        else:
            t0 = float(rng.uniform(0.25, 0.6))

        pp = {"fs_hz": fs, "duration_s": duration, "switch_time_s": t0, "f1_hz": f1, "f2_hz": f2}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = spec_render(pp, abs_img, meta)
        gt = spec_baseline(pp)

        q = (
            "From the spectrogram, estimate:\n"
            "1) f1_hz (Hz) (first tone)\n"
            "2) f2_hz (Hz) (second tone)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["f1_hz", "f2_hz"],
                           "checkpoint_fields": ["switch_time_s", "cp_duration_s"],
                           **axis_meta},
        })
    return items


# =========================
# 8) I–V curve
# =========================
def iv_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    kind = str(pp["kind"])
    if kind == "resistor":
        R = float(pp["R_ohm"])
        return {"resistance_ohm": float(R), "cp_slope_ohm": float(R)}
    # diode
    Is = float(pp["Is"])
    nVt = float(pp["nVt"])
    Rs = float(pp["Rs"])
    It = float(pp["target_current_a"])
    Vt = nVt * math.log(It / Is + 1.0) + It * Rs
    return {
        "target_current_a": float(It),
        "turn_on_voltage_v_at_target_i": float(Vt),
        "cp_Is": float(Is),
        "cp_nVt": float(nVt),
        "cp_Rs": float(Rs),
    }


def iv_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    kind = str(pp["kind"])
    fig, ax = plt.subplots(figsize=FIGSIZE)

    if kind == "resistor":
        R = float(pp["R_ohm"])
        v = np.linspace(-5.0, 5.0, 600)
        i = v / R
        ax.plot(v, i)
        ax.set_title("I–V Curve (Resistor)")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (A)")
        if meta.difficulty != "edge":
            ax.grid(True, alpha=0.35)
        save_fig(fig, out_path)
        return {"x_min": -5.0, "x_max": 5.0, "y_min": float(np.min(i)), "y_max": float(np.max(i)),
                "tick_step_x": None, "tick_step_y": None}

    # diode: parametric curve V(I)
    Is = float(pp["Is"])
    nVt = float(pp["nVt"])
    Rs = float(pp["Rs"])

    I = np.linspace(0.0, 0.05, 700)
    V = nVt * np.log(I / Is + 1.0) + I * Rs

    ax.plot(V, I)
    ax.set_title("I–V Curve (Diode)")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_fig(fig, out_path)
    return {"x_min": float(np.min(V)), "x_max": float(np.max(V)), "y_min": 0.0, "y_max": float(np.max(I)),
            "tick_step_x": None, "tick_step_y": None}


def gen_iv(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "iv_curve"
    items = []
    plan = difficulty_plan(n)

    Rs = [10.0, 22.0, 47.0, 100.0, 220.0, 470.0, 1000.0]
    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        kind = "resistor" if (i % 2 == 0) else "diode"
        if difficulty == "edge":
            edge_tag = "tight_scale"

        if kind == "resistor":
            R = float(rng.choice(Rs))
            pp = {"kind": "resistor", "R_ohm": R}
            final_fields = ["resistance_ohm"]
            cp_fields = ["cp_slope_ohm"]
        else:
            pp = {
                "kind": "diode",
                "Is": 1e-12,
                "nVt": float(rng.choice([0.035, 0.050, 0.060])),
                "Rs": float(rng.choice([1.0, 2.0, 5.0])),
                "target_current_a": float(rng.choice([0.005, 0.01, 0.02])),
            }
            final_fields = ["turn_on_voltage_v_at_target_i"]
            cp_fields = ["target_current_a", "cp_Is", "cp_nVt", "cp_Rs"]

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = iv_render(pp, abs_img, meta)
        gt = iv_baseline(pp)

        q = (
            "From the I–V curve, estimate the requested value(s) for this plot type.\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": final_fields,
                           "checkpoint_fields": cp_fields,
                           **axis_meta},
        })
    return items


# =========================
# 9) Transfer characteristic
# =========================
def transfer_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    g = float(pp["gain"])
    vs = float(pp["Vsat"])
    vin_sat = vs / max(g, 1e-12)
    return {"small_signal_gain": float(g), "saturation_v": float(vs), "cp_vin_at_saturation": float(vin_sat)}


def transfer_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    g = float(pp["gain"])
    vs = float(pp["Vsat"])
    vin = np.linspace(-2.0 * (vs / g), 2.0 * (vs / g), 600)
    vout = np.clip(g * vin, -vs, vs)

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.plot(vin, vout)
    ax.set_title("Transfer Characteristic (gain + saturation)")
    ax.set_xlabel("Vin (V)")
    ax.set_ylabel("Vout (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_fig(fig, out_path)
    return {"x_min": float(np.min(vin)), "x_max": float(np.max(vin)), "y_min": float(np.min(vout)), "y_max": float(np.max(vout)),
            "tick_step_x": None, "tick_step_y": None}


def gen_transfer(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "transfer_characteristic"
    items = []
    plan = difficulty_plan(n)

    gains = [0.5, 1.0, 2.0, 5.0]
    vsats = [1.0, 2.0, 3.0, 5.0]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        g = float(rng.choice(gains))
        vs = float(rng.choice(vsats))
        if difficulty == "edge":
            edge_tag = "small_linear_region"
            g = float(rng.choice([5.0]))
            vs = float(rng.choice([1.0, 2.0]))

        pp = {"gain": g, "Vsat": vs}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = transfer_render(pp, abs_img, meta)
        gt = transfer_baseline(pp)

        q = (
            "From the transfer characteristic, estimate:\n"
            "1) small_signal_gain\n"
            "2) saturation_v\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["small_signal_gain", "saturation_v"],
                           "checkpoint_fields": ["cp_vin_at_saturation"],
                           **axis_meta},
        })
    return items


# =========================
# 10) Pole-zero plot (2nd-order poles)
# =========================
def pz_baseline(pp: Dict[str, Any]) -> Dict[str, float]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    real = -zeta * wn
    imag = wn * math.sqrt(max(1.0 - zeta**2, 0.0))
    return {"zeta": float(zeta), "wn_rad_s": float(wn), "cp_pole_real": float(real), "cp_pole_imag": float(imag)}


def pz_render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])

    real = -zeta * wn
    imag = wn * math.sqrt(max(1.0 - zeta**2, 0.0))

    fig, ax = plt.subplots(figsize=FIGSIZE)
    ax.scatter([real, real], [imag, -imag], marker="x", s=80)
    ax.axhline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.axvline(0.0, color="black", linewidth=1.0, alpha=0.4)
    ax.set_title("Pole-Zero Plot (2nd-order poles)")
    ax.set_xlabel("Real (rad/s)")
    ax.set_ylabel("Imag (rad/s)")
    ax.set_aspect("equal", adjustable="box")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    # set symmetric limits for readability
    lim = max(abs(real), abs(imag)) * 1.6 + 0.5
    ax.set_xlim(-lim, lim)
    ax.set_ylim(-lim, lim)

    save_fig(fig, out_path)
    return {"x_min": -lim, "x_max": lim, "y_min": -lim, "y_max": lim, "tick_step_x": None, "tick_step_y": None}


def gen_pole_zero(out_dir: Path, images_root: Path, master_seed: int, n: int) -> List[Dict[str, Any]]:
    fam = "pole_zero"
    items = []
    plan = difficulty_plan(n)

    zetas = [0.2, 0.3, 0.4, 0.5, 0.6, 0.7]
    wns = [4.0, 6.0, 8.0, 10.0, 12.0, 16.0]

    for i in range(n):
        seed = stable_int_seed(master_seed, fam, i)
        rng = np.random.default_rng(seed)
        difficulty = plan[i]
        edge_tag = ""

        zeta = float(rng.choice(zetas))
        wn = float(rng.choice(wns))
        if difficulty == "edge":
            edge_tag = "near_critical"
            zeta = float(rng.choice([0.7, 0.8, 0.9]))
            wn = float(rng.choice([6.0, 8.0, 10.0]))

        pp = {"zeta": zeta, "wn_rad_s": wn}
        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)

        img_name = f"{fam}_{i:03d}.png"
        rel_img = Path("images") / fam / img_name
        abs_img = images_root / fam / img_name

        axis_meta = pz_render(pp, abs_img, meta)
        gt = pz_baseline(pp)

        q = (
            "From the pole plot, estimate:\n"
            "1) zeta (damping ratio)\n"
            "2) wn_rad_s (natural frequency, rad/s)\n"
            "Return numeric JSON."
        )

        items.append({
            "id": f"{fam}_{i:03d}",
            "type": fam,
            "image_path": str(rel_img).replace("\\", "/"),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": ["zeta", "wn_rad_s"],
                           "checkpoint_fields": ["cp_pole_real", "cp_pole_imag"],
                           **axis_meta},
        })
    return items


# =========================
# Orchestrator + validator
# =========================
GEN_MAP = {
    "step_response": gen_step,
    "bode_magnitude": gen_bode_mag,
    "bode_phase": gen_bode_phase,
    "bandpass_response": gen_bandpass,
    "time_waveform": gen_time_wave,
    "fft_spectrum": gen_fft,
    "spectrogram": gen_spectrogram,
    "iv_curve": gen_iv,
    "transfer_characteristic": gen_transfer,
    "pole_zero": gen_pole_zero,
}

BASELINE_MAP = {
    "step_response": step_baseline,
    "bode_magnitude": bode_mag_baseline,
    "bode_phase": bode_phase_baseline,
    "bandpass_response": bandpass_baseline,
    "time_waveform": time_wave_baseline,
    "fft_spectrum": fft_baseline,
    "spectrogram": spec_baseline,
    "iv_curve": iv_baseline,
    "transfer_characteristic": transfer_baseline,
    "pole_zero": pz_baseline,
}


def validate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows = []
    for it in items:
        typ = it["type"]
        pp = it["plot_params"]
        gt = it["ground_truth"]
        pred = BASELINE_MAP[typ](pp)
        for k in gt.keys():
            pv = float(pred[k])
            gv = float(gt[k])
            ok = float_close(pv, gv, abs_tol=1e-12, rel_tol=1e-12)
            rows.append({"type": typ, "id": it["id"], "field": k, "pass": ok, "abs_err": abs(pv - gv)})
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="data/plotchain_v4", help="Output dataset directory")
    ap.add_argument("--seed", type=int, default=0, help="Master seed")
    ap.add_argument("--n_per_family", type=int, default=20, help="Items per family (recommended 20 or 30)")
    args = ap.parse_args()

    set_mpl_style()

    out_dir = Path(args.out_dir)
    images_root = out_dir / "images"
    out_dir.mkdir(parents=True, exist_ok=True)

    all_items: List[Dict[str, Any]] = []
    for fam in FAMILIES:
        gen = GEN_MAP[fam]
        items = gen(out_dir=out_dir, images_root=images_root, master_seed=args.seed, n=args.n_per_family)
        print(f"[gen] {fam}: n={len(items)}")
        all_items.extend(items)

    jsonl_path = out_dir / "plotchain_v4.jsonl"
    write_jsonl(jsonl_path, all_items)
    print(f"[write] {jsonl_path} ({len(all_items)} items)")

    rows = validate(all_items)
    pass_rate = sum(1 for r in rows if r["pass"]) / max(len(rows), 1)
    print(f"[validate] pass_rate={pass_rate*100:.1f}% rows={len(rows)}")

    report_csv = out_dir / "baseline_report_v4.csv"
    with report_csv.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["type", "id", "field", "pass", "abs_err"])
        w.writeheader()
        w.writerows(rows)
    print(f"[write] {report_csv}")


if __name__ == "__main__":
    main()
