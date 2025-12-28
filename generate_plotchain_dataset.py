#!/usr/bin/env python3
"""
PlotChain v2 generator (Option B: deterministic noise per item)

Key properties:
- Each item has a stable item_seed derived from (base_seed, item_id).
- All randomness (params + noise) uses sub-seeded numpy RNGs derived from item_seed.
- Ground truth is computed deterministically from the same arrays used to render the plot.
- v2 schema includes: version, item_seed, generation metadata.

Usage:
  python generate_plotchain_v2_dataset.py --out_dir plotchain_v2 --n_per_type 10 --base_seed 42
"""

import argparse
import hashlib
import json
import math
import os
import textwrap
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal


# -----------------------------
# Seeding utilities (stable)
# -----------------------------
def _u64_from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], "little", signed=False)

def item_seed_from_id(base_seed: int, item_id: str) -> int:
    """Stable seed from base_seed + item_id (independent of generation order)."""
    h = hashlib.sha256(f"{base_seed}|{item_id}".encode("utf-8")).digest()
    return _u64_from_bytes(h)

def sub_seed(item_seed: int, salt: str) -> int:
    """Derive stable sub-seed for independent RNG streams (params vs noise etc.)."""
    h = hashlib.sha256(f"{item_seed}|{salt}".encode("utf-8")).digest()
    return _u64_from_bytes(h)

def rng_for(item_seed: int, salt: str) -> np.random.Generator:
    return np.random.default_rng(sub_seed(item_seed, salt))


# -----------------------------
# Small numeric helpers
# -----------------------------
def save_plot(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def settling_time_2pct(t: np.ndarray, y: np.ndarray, final: float, tol: float = 0.02) -> float:
    """2% settling time: first time after which response stays within ±2% of final value."""
    band = tol * abs(final) if abs(final) > 0 else tol
    lo, hi = final - band, final + band
    inside = (y >= lo) & (y <= hi)
    for i in range(len(t)):
        if inside[i] and inside[i:].all():
            return float(t[i])
    return float(t[-1])

def interp_x_at_y(x: np.ndarray, y: np.ndarray, target: float) -> float:
    """Find x value where y equals target using local linear interpolation."""
    idx = int(np.argmin(np.abs(y - target)))
    if y[idx] == target:
        return float(x[idx])
    if idx == 0:
        j = 1
    elif idx == len(y) - 1:
        j = len(y) - 2
    else:
        if (y[idx - 1] - target) * (y[idx] - target) <= 0:
            j = idx - 1
        else:
            j = idx + 1
    x1, x2 = float(x[idx]), float(x[j])
    y1, y2 = float(y[idx]), float(y[j])
    if y2 == y1:
        return float(x1)
    return float(x1 + (target - y1) * (x2 - x1) / (y2 - y1))


# -----------------------------
# Plot generators (v2)
# -----------------------------
def make_step_response(item_id: str, item_seed: int, img_dir: str) -> Dict[str, Any]:
    rp = rng_for(item_seed, "params")
    zeta = float(rp.uniform(0.15, 0.85))
    wn = float(rp.uniform(2 * math.pi * 0.8, 2 * math.pi * 3.0))  # rad/s

    # Store grid for exact re-generation
    t_end = float(rp.uniform(2.5, 6.0))
    n = 1200
    t = np.linspace(0, t_end, n)

    sys = signal.TransferFunction([wn**2], [1, 2*zeta*wn, wn**2])
    tout, y = signal.step(sys, T=t)

    final = float(y[-1])
    peak = float(np.max(y))
    overshoot = 0.0 if final == 0 else max(0.0, (peak - final) / abs(final) * 100.0)
    ts = settling_time_2pct(tout, y, final, tol=0.02)

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(tout, y, linewidth=2)
    ax.set_title("Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (unitless)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"{item_id}.png")
    save_plot(fig, img_path)

    return {
        "id": item_id,
        "version": "v2",
        "type": "step_response",
        "item_seed": int(item_seed),
        "image_path": f"images/{item_id}.png",
        "question": "From the step response, estimate (a) percent overshoot (%) and (b) 2% settling time (s). Show your reasoning steps.",
        "gold_chain": [
            "Identify the steady-state value from the final plateau of the response.",
            "Identify the peak value (maximum amplitude) from the curve.",
            "Compute percent overshoot = (peak − steady_state) / steady_state × 100%.",
            "Compute 2% settling time as the first time after which the response stays within ±2% of steady-state.",
        ],
        "ground_truth": {
            "percent_overshoot": float(np.round(overshoot, 6)),
            "settling_time_s": float(np.round(ts, 6)),
            "steady_state": float(np.round(final, 6)),
        },
        "plot_params": {"zeta": zeta, "wn_rad_s": wn},
        "generation": {"t_end_s": t_end, "n_samples": n},
    }


def make_bode(item_id: str, item_seed: int, img_dir: str) -> Dict[str, Any]:
    rp = rng_for(item_seed, "params")
    K = float(rp.uniform(0.5, 2.0))
    fc = float(rp.uniform(20, 400))  # Hz
    tau = 1.0 / (2 * math.pi * fc)

    f_min, f_max, n = 1.0, 2000.0, 600
    f = np.logspace(math.log10(f_min), math.log10(f_max), n)
    w = 2 * math.pi * f

    sys = signal.TransferFunction([K], [tau, 1.0])
    _, mag_db, _ = signal.bode(sys, w=w)

    dc_gain_db = float(20 * math.log10(K))
    target = dc_gain_db - 3.0
    f_cut = interp_x_at_y(f, mag_db, target)

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.semilogx(f, mag_db, linewidth=2)
    ax.set_title("Bode Magnitude Plot (Low-Pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", alpha=0.3)
    img_path = os.path.join(img_dir, f"{item_id}.png")
    save_plot(fig, img_path)

    return {
        "id": item_id,
        "version": "v2",
        "type": "bode_magnitude",
        "item_seed": int(item_seed),
        "image_path": f"images/{item_id}.png",
        "question": "From the Bode magnitude plot, estimate (a) DC gain (dB) and (b) the -3 dB cutoff frequency (Hz). Show your reasoning steps.",
        "gold_chain": [
            "Read the low-frequency (left-side) magnitude level as the DC gain in dB.",
            "Compute the target level for cutoff: DC gain − 3 dB.",
            "Find the frequency where the curve crosses the target level and report it as the cutoff frequency.",
        ],
        "ground_truth": {
            "dc_gain_db": float(np.round(dc_gain_db, 6)),
            "cutoff_hz": float(np.round(float(f_cut), 6)),
        },
        "plot_params": {"K": K, "fc_hz": fc, "tau_s": tau},
        "generation": {"f_min_hz": f_min, "f_max_hz": f_max, "n_points": n},
    }


def make_fft(item_id: str, item_seed: int, img_dir: str) -> Dict[str, Any]:
    rp = rng_for(item_seed, "params")
    rn = rng_for(item_seed, "noise_fft")

    fs = int(rp.choice([500, 1000, 2000]))
    duration_s = float(rp.uniform(0.5, 1.5))
    N = int(round(duration_s * fs))
    duration_s = N / fs  # snap to exact grid

    t = np.arange(N, dtype=float) / fs

    n_tones = int(rp.choice([2, 3]))
    # choose tones below ~200Hz and below Nyquist
    max_f = int(min(200, fs // 2 - 10))
    tones = sorted(rp.choice(np.arange(10, max_f), size=n_tones, replace=False).tolist())
    amps = rp.uniform(0.6, 1.6, size=n_tones).astype(float)

    x = np.zeros_like(t)
    for a, f0 in zip(amps, tones):
        x += float(a) * np.sin(2 * math.pi * float(f0) * t)

    noise_std = 0.15
    x_noisy = x + rn.normal(0.0, noise_std, size=N)

    X = np.fft.rfft(x_noisy)
    f_axis = np.fft.rfftfreq(N, d=1.0 / fs)
    mag = np.abs(X) / (N / 2)

    idx = int(np.argmax(mag[1:]) + 1)  # exclude DC
    dom_f = float(f_axis[idx])

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(f_axis, mag, linewidth=2)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.set_xlim(0, min(250, fs / 2))
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"{item_id}.png")
    save_plot(fig, img_path)

    return {
        "id": item_id,
        "version": "v2",
        "type": "fft_spectrum",
        "item_seed": int(item_seed),
        "image_path": f"images/{item_id}.png",
        "question": "From the FFT magnitude spectrum, identify the dominant frequency component (Hz). Show your reasoning steps.",
        "gold_chain": [
            "Ignore the DC component at 0 Hz.",
            "Find the tallest peak in the magnitude spectrum.",
            "Read the corresponding frequency on the x-axis and report it as the dominant frequency.",
        ],
        "ground_truth": {"dominant_frequency_hz": float(np.round(dom_f, 6))},
        "plot_params": {
            "fs_hz": fs,
            "duration_s": duration_s,
            "tones_hz": tones,
            "amps": [float(a) for a in amps],
            "noise_std": noise_std,
        },
        "generation": {"N": N},
    }


def make_waveform(item_id: str, item_seed: int, img_dir: str) -> Dict[str, Any]:
    rp = rng_for(item_seed, "params")
    rn = rng_for(item_seed, "noise_wave")

    fs = 2000
    duration_s = 0.02
    N = int(round(duration_s * fs))
    t = np.arange(N, dtype=float) / fs

    wave_type = str(rp.choice(["sine", "square", "triangle", "trapezoid"]))
    f0 = float(rp.choice([50, 60, 100, 120, 200, 250]))
    A = float(rp.uniform(0.8, 2.5))
    duty = None

    if wave_type == "sine":
        y = A * np.sin(2 * math.pi * f0 * t)

    elif wave_type == "square":
        duty = float(rp.uniform(0.3, 0.7))
        y = A * signal.square(2 * math.pi * f0 * t, duty=duty)

    elif wave_type == "triangle":
        y = A * signal.sawtooth(2 * math.pi * f0 * t, width=0.5)

    else:
        tri = signal.sawtooth(2 * math.pi * f0 * t, width=0.5)
        y = A * np.clip(tri * 1.4, -1, 1)

    noise_std = 0.05
    y_noisy = y + rn.normal(0.0, noise_std, size=N)
    vpp = float(np.max(y_noisy) - np.min(y_noisy))

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(t * 1000, y_noisy, linewidth=2)
    ax.set_title(f"Time-Domain Waveform ({wave_type})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"{item_id}.png")
    save_plot(fig, img_path)
    plot_params = {"wave_type": wave_type, "f0_hz": f0, "A": A, "fs_hz": fs, "noise_std": noise_std}
    if duty is not None:
        plot_params["duty"] = float(np.round(duty, 6))

    return {
        "id": item_id,
        "version": "v2",
        "type": "time_waveform",
        "item_seed": int(item_seed),
        "image_path": f"images/{item_id}.png",
        "question": "From the time-domain waveform, estimate (a) the fundamental frequency (Hz) and (b) the peak-to-peak amplitude (Vpp). Show your reasoning steps.",
        "gold_chain": [
            "Estimate the period by measuring the time between successive similar points (e.g., peaks).",
            "Compute frequency = 1 / period.",
            "Estimate peak-to-peak amplitude as (maximum − minimum) on the y-axis.",
        ],
        # Frequency is the latent generating frequency (deterministic), Vpp is computed from the noisy plotted waveform.
        "ground_truth": {"frequency_hz": float(np.round(f0, 6)), "vpp_v": float(np.round(vpp, 6))},
        "plot_params": plot_params,
        "generation": {"duration_s": duration_s, "N": N},
    }


def make_iv(item_id: str, item_seed: int, img_dir: str) -> Dict[str, Any]:
    rp = rng_for(item_seed, "params")
    rn = rng_for(item_seed, "noise_iv")

    kind = str(rp.choice(["resistor", "diode"]))
    v = np.linspace(0.0, 1.2, 300)

    if kind == "resistor":
        R = float(rp.uniform(50, 600))
        noise_std = 0.0002  # A
        i_a = v / R + rn.normal(0.0, noise_std, size=len(v))

        fig = plt.figure(figsize=(6.5, 4.4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(v, i_a * 1000, linewidth=2)
        ax.set_title("I–V Curve (Resistive Load)")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (mA)")
        ax.grid(True, alpha=0.3)
        img_path = os.path.join(img_dir, f"{item_id}.png")
        save_plot(fig, img_path)

        return {
            "id": item_id,
            "version": "v2",
            "type": "iv_curve",
            "item_seed": int(item_seed),
            "image_path": f"images/{item_id}.png",
            "question": "From the I–V curve, estimate the resistance (Ohms). Show your reasoning steps.",
            "gold_chain": [
                "For a resistor, the I–V relationship is linear: I = V / R.",
                "Estimate the slope dI/dV from two points on the line.",
                "Compute R = 1 / slope and report it.",
            ],
            # Ground truth is the latent R (deterministic), despite noise in samples.
            "ground_truth": {"resistance_ohm": float(np.round(R, 6))},
            "plot_params": {"kind": "resistor", "R_ohm": R, "noise_std": noise_std},
            "generation": {"v_min": 0.0, "v_max": 1.2, "n_points": 300},
        }

    else:
        # Shockley diode with series resistance; compute I(V) via fixed-point per V
        Is = 1e-12
        nVt = float(rp.uniform(0.03, 0.06))
        Rs = float(rp.uniform(2, 20))

        i_a = np.zeros_like(v)
        for k in range(len(v)):
            Vk = float(v[k])
            Ik = 0.0
            for _ in range(40):
                Ik = Is * (math.exp((Vk - Ik * Rs) / nVt) - 1.0)
                Ik = max(Ik, 0.0)
            i_a[k] = Ik

        noise_std = 0.00005  # A
        i_noisy = i_a + rn.normal(0.0, noise_std, size=len(v))

        target_I = 0.01
        if float(np.max(i_noisy)) < target_I:
            target_I = 0.001
        v_on = interp_x_at_y(v, i_noisy, target_I)

        fig = plt.figure(figsize=(6.5, 4.4))
        ax = fig.add_subplot(1, 1, 1)
        ax.plot(v, i_noisy * 1000, linewidth=2)
        ax.set_title("I–V Curve (Diode)")
        ax.set_xlabel("Voltage (V)")
        ax.set_ylabel("Current (mA)")
        ax.grid(True, alpha=0.3)
        img_path = os.path.join(img_dir, f"{item_id}.png")
        save_plot(fig, img_path)

        return {
            "id": item_id,
            "version": "v2",
            "type": "iv_curve",
            "item_seed": int(item_seed),
            "image_path": f"images/{item_id}.png",
            "question": f"From the diode I–V curve, estimate the turn-on voltage (V) at I = {target_I*1000:.1f} mA. Show your reasoning steps.",
            "gold_chain": [
                "Identify the target current level on the y-axis.",
                "Find where the I–V curve reaches that current.",
                "Read the corresponding voltage on the x-axis and report it as the turn-on voltage at that current.",
            ],
            # Ground truth is computed from the EXACT noisy curve used to plot (deterministic via per-item seed).
            "ground_truth": {
                "turn_on_voltage_v_at_target_i": float(np.round(float(v_on), 6)),
                "target_current_a": float(target_I),
            },
            "plot_params": {"kind": "diode", "Is": Is, "nVt": nVt, "Rs": Rs, "noise_std": noise_std},
            "generation": {"v_min": 0.0, "v_max": 1.2, "n_points": 300, "fixed_point_iters": 40},
        }


def make_transfer(item_id: str, item_seed: int, img_dir: str) -> Dict[str, Any]:
    rp = rng_for(item_seed, "params")
    rn = rng_for(item_seed, "noise_transfer")

    gain = float(rp.uniform(1.2, 6.0))
    Vsat = float(rp.uniform(1.0, 4.5))
    offset = float(rp.uniform(-0.4, 0.4))

    x = np.linspace(-2.0, 2.0, 400)
    y = np.clip(gain * x + offset, -Vsat, Vsat)

    noise_std = 0.03
    y_noisy = y + rn.normal(0.0, noise_std, size=len(x))

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1, 1, 1)
    ax.plot(x, y_noisy, linewidth=2)
    ax.set_title("Transfer Characteristic (Saturating Amplifier)")
    ax.set_xlabel("Input (V)")
    ax.set_ylabel("Output (V)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"{item_id}.png")
    save_plot(fig, img_path)

    return {
        "id": item_id,
        "version": "v2",
        "type": "transfer_characteristic",
        "item_seed": int(item_seed),
        "image_path": f"images/{item_id}.png",
        "question": "From the transfer characteristic, estimate (a) the small-signal gain (slope in the linear region) and (b) the saturation voltage magnitude (V). Show your reasoning steps.",
        "gold_chain": [
            "Identify the central linear region where output changes proportionally with input.",
            "Estimate slope in that region using two points: gain ≈ Δoutput / Δinput.",
            "Identify the flat regions where the output saturates and read the saturation voltage magnitude.",
        ],
        # Ground truth is the latent generating parameters (deterministic).
        "ground_truth": {"small_signal_gain": float(np.round(gain, 6)), "saturation_v": float(np.round(Vsat, 6))},
        "plot_params": {"gain": gain, "Vsat": Vsat, "offset": offset, "noise_std": noise_std},
        "generation": {"x_min": -2.0, "x_max": 2.0, "n_points": 400},
    }


# -----------------------------
# Main: build dataset
# -----------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="plotchain_v2")
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--base_seed", type=int, default=42)
    args = ap.parse_args()

    out_dir = Path(args.out_dir)
    img_dir = out_dir / "images"
    img_dir.mkdir(parents=True, exist_ok=True)

    items: List[Dict[str, Any]] = []

    # Deterministic ids (stable). Seeds are derived from id, so ordering is irrelevant.
    for i in range(args.n_per_type):
        item_id = f"step_{i:02d}"
        items.append(make_step_response(item_id, item_seed_from_id(args.base_seed, item_id), str(img_dir)))

    for i in range(args.n_per_type):
        item_id = f"bode_{i:02d}"
        items.append(make_bode(item_id, item_seed_from_id(args.base_seed, item_id), str(img_dir)))

    for i in range(args.n_per_type):
        item_id = f"fft_{i:02d}"
        items.append(make_fft(item_id, item_seed_from_id(args.base_seed, item_id), str(img_dir)))

    for i in range(args.n_per_type):
        item_id = f"wave_{i:02d}"
        items.append(make_waveform(item_id, item_seed_from_id(args.base_seed, item_id), str(img_dir)))

    for i in range(args.n_per_type):
        item_id = f"iv_{i:02d}"
        items.append(make_iv(item_id, item_seed_from_id(args.base_seed, item_id), str(img_dir)))

    for i in range(args.n_per_type):
        item_id = f"transfer_{i:02d}"
        items.append(make_transfer(item_id, item_seed_from_id(args.base_seed, item_id), str(img_dir)))

    jsonl_path = out_dir / "plotchain_v2.jsonl"
    with jsonl_path.open("w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    rows = []
    for it in items:
        rows.append({
            "id": it["id"],
            "version": it["version"],
            "type": it["type"],
            "item_seed": it["item_seed"],
            "image_path": it["image_path"],
            "question": it["question"],
            "ground_truth_json": json.dumps(it["ground_truth"], ensure_ascii=False),
            "gold_chain_json": json.dumps(it["gold_chain"], ensure_ascii=False),
            "plot_params_json": json.dumps(it["plot_params"], ensure_ascii=False),
            "generation_json": json.dumps(it["generation"], ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(out_dir / "plotchain_v2.csv", index=False)

    readme = textwrap.dedent(f"""\
    # PlotChain v2

    Option B: deterministic noise per item.

    - Canonical file: plotchain_v2.jsonl
    - Images: images/
    - Base seed: {args.base_seed}
    - Items per type: {args.n_per_type}
    - Total items: {len(items)}

    Regenerate:
      python generate_plotchain_v2_dataset.py --out_dir plotchain_v2 --n_per_type {args.n_per_type} --base_seed {args.base_seed}
    """)
    (out_dir / "README.md").write_text(readme, encoding="utf-8")

    print(f"Wrote {len(items)} items to {out_dir}/")
    print(f"JSONL: {jsonl_path}")


if __name__ == "__main__":
    main()
