#!/usr/bin/env python3
"""
baseline_solver_v2.py

PlotChain v2 verification checker (exact, deterministic).

Strategy:
- For each record, rebuild the underlying arrays using item_seed-derived RNG streams.
- Compute the same ground truth fields the generator wrote.
- Compare to stored ground_truth.

Tolerances:
- Tight but not brittle across machines: default abs_tol = 1e-5.
- You can set it to 1e-6 if you're pinning numpy/scipy/matplotlib versions.
"""

import argparse
import json
import math
import hashlib
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
from scipy import signal


def _u64_from_bytes(b: bytes) -> int:
    return int.from_bytes(b[:8], "little", signed=False)

def sub_seed(item_seed: int, salt: str) -> int:
    h = hashlib.sha256(f"{item_seed}|{salt}".encode("utf-8")).digest()
    return _u64_from_bytes(h)

def rng_for(item_seed: int, salt: str) -> np.random.Generator:
    return np.random.default_rng(sub_seed(item_seed, salt))

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def settling_time_2pct(t: np.ndarray, y: np.ndarray, final: float, tol: float = 0.02) -> float:
    band = tol * abs(final) if abs(final) > 0 else tol
    lo, hi = final - band, final + band
    inside = (y >= lo) & (y <= hi)
    for i in range(len(t)):
        if inside[i] and inside[i:].all():
            return float(t[i])
    return float(t[-1])

def interp_x_at_y(x: np.ndarray, y: np.ndarray, target: float) -> float:
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

def compare(a: float, b: float) -> float:
    return float(abs(a - b))

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--jsonl", type=str, default="plotchain_v2.jsonl")
    ap.add_argument("--out_csv", type=str, default="")
    ap.add_argument("--abs_tol", type=float, default=1e-5)
    args = ap.parse_args()

    items = load_jsonl(Path(args.jsonl))
    rows = []

    for it in items:
        typ = it["type"]
        item_seed = int(it["item_seed"])
        gt = it["ground_truth"]
        pp = it["plot_params"]
        gen = it.get("generation", {})

        pred: Dict[str, float] = {}

        if typ == "step_response":
            zeta = float(pp["zeta"])
            wn = float(pp["wn_rad_s"])
            t_end = float(gen["t_end_s"])
            n = int(gen["n_samples"])
            t = np.linspace(0, t_end, n)
            sys = signal.TransferFunction([wn**2], [1, 2*zeta*wn, wn**2])
            tout, y = signal.step(sys, T=t)
            final = float(y[-1])
            peak = float(np.max(y))
            overshoot = 0.0 if final == 0 else max(0.0, (peak - final) / abs(final) * 100.0)
            ts = settling_time_2pct(tout, y, final, tol=0.02)
            pred["percent_overshoot"] = float(np.round(overshoot, 6))
            pred["settling_time_s"] = float(np.round(ts, 6))
            pred["steady_state"] = float(np.round(final, 6))

        elif typ == "bode_magnitude":
            K = float(pp["K"])
            f_min = float(gen["f_min_hz"])
            f_max = float(gen["f_max_hz"])
            n = int(gen["n_points"])
            f = np.logspace(math.log10(f_min), math.log10(f_max), n)
            w = 2 * math.pi * f
            tau = float(pp["tau_s"])
            sys = signal.TransferFunction([K], [tau, 1.0])
            _, mag_db, _ = signal.bode(sys, w=w)
            dc_gain_db = float(20 * math.log10(K))
            target = dc_gain_db - 3.0
            f_cut = interp_x_at_y(f, mag_db, target)
            pred["dc_gain_db"] = float(np.round(dc_gain_db, 6))
            pred["cutoff_hz"] = float(np.round(float(f_cut), 6))

        elif typ == "fft_spectrum":
            rp = rng_for(item_seed, "params")
            rn = rng_for(item_seed, "noise_fft")
            fs = int(pp["fs_hz"])
            duration_s = float(pp["duration_s"])
            N = int(gen["N"])
            # recreate exact grid
            t = np.arange(N, dtype=float) / fs

            tones = list(pp["tones_hz"])
            amps = np.array(pp["amps"], dtype=float)
            x = np.zeros_like(t)
            for a, f0 in zip(amps, tones):
                x += float(a) * np.sin(2 * math.pi * float(f0) * t)

            noise_std = float(pp["noise_std"])
            x_noisy = x + rn.normal(0.0, noise_std, size=N)

            X = np.fft.rfft(x_noisy)
            f_axis = np.fft.rfftfreq(N, d=1.0 / fs)
            mag = np.abs(X) / (N / 2)
            idx = int(np.argmax(mag[1:]) + 1)
            dom_f = float(f_axis[idx])
            pred["dominant_frequency_hz"] = float(np.round(dom_f, 6))

        elif typ == "time_waveform":
            rp = rng_for(item_seed, "params")
            rn = rng_for(item_seed, "noise_wave")
            fs = int(pp["fs_hz"])
            duration_s = float(gen["duration_s"])
            N = int(gen["N"])
            t = np.arange(N, dtype=float) / fs

            wave_type = str(pp["wave_type"])
            f0 = float(pp["f0_hz"])
            A = float(pp["A"])
            if wave_type == "sine":
                y = A * np.sin(2 * math.pi * f0 * t)
            elif wave_type == "square":
                duty = float(pp["duty"])
                y = A * signal.square(2 * math.pi * f0 * t, duty=duty)
            elif wave_type == "triangle":
                y = A * signal.sawtooth(2 * math.pi * f0 * t, width=0.5)
            else:
                tri = signal.sawtooth(2 * math.pi * f0 * t, width=0.5)
                y = A * np.clip(tri * 1.4, -1, 1)

            noise_std = float(pp["noise_std"])
            y_noisy = y + rn.normal(0.0, noise_std, size=N)
            vpp = float(np.max(y_noisy) - np.min(y_noisy))
            pred["frequency_hz"] = float(np.round(f0, 6))
            pred["vpp_v"] = float(np.round(vpp, 6))

        elif typ == "iv_curve":
            kind = str(pp["kind"])
            v_min = float(gen["v_min"])
            v_max = float(gen["v_max"])
            npts = int(gen["n_points"])
            v = np.linspace(v_min, v_max, npts)

            rn = rng_for(item_seed, "noise_iv")

            if kind == "resistor":
                pred["resistance_ohm"] = float(np.round(float(pp["R_ohm"]), 6))
            else:
                Is = float(pp["Is"])
                nVt = float(pp["nVt"])
                Rs = float(pp["Rs"])
                iters = int(gen["fixed_point_iters"])
                i_a = np.zeros_like(v)
                for k in range(len(v)):
                    Vk = float(v[k])
                    Ik = 0.0
                    for _ in range(iters):
                        Ik = Is * (math.exp((Vk - Ik * Rs) / nVt) - 1.0)
                        Ik = max(Ik, 0.0)
                    i_a[k] = Ik

                noise_std = float(pp["noise_std"])
                i_noisy = i_a + rn.normal(0.0, noise_std, size=len(v))

                target_I = float(gt["target_current_a"])
                v_on = interp_x_at_y(v, i_noisy, target_I)
                pred["turn_on_voltage_v_at_target_i"] = float(np.round(float(v_on), 6))
                pred["target_current_a"] = float(target_I)

        elif typ == "transfer_characteristic":
            pred["small_signal_gain"] = float(np.round(float(pp["gain"]), 6))
            pred["saturation_v"] = float(np.round(float(pp["Vsat"]), 6))

        # Compare all fields present in gt that we predicted
        for k, pv in pred.items():
            gv = float(gt[k])
            abs_err = compare(pv, gv)
            passed = abs_err <= args.abs_tol
            rows.append({
                "id": it["id"],
                "type": typ,
                "field": k,
                "ground_truth": gv,
                "baseline_pred": pv,
                "abs_err": abs_err,
                "abs_tol": args.abs_tol,
                "pass": passed,
            })

    df = pd.DataFrame(rows)
    if df.empty:
        print("No comparisons generated.")
        return

    summary = (
        df.groupby(["type", "field"])
          .agg(n=("pass", "size"), pass_rate=("pass", "mean"),
               mean_abs_err=("abs_err", "mean"), max_abs_err=("abs_err", "max"))
          .reset_index()
          .sort_values(["type", "field"])
    )

    print("\n=== PlotChain v2 Baseline Validity Report ===\n")
    print(summary.to_string(index=False))
    overall = float(df["pass"].mean())
    print(f"\nOverall pass rate: {overall*100:.1f}% ({df['pass'].sum()}/{len(df)})")

    fails = df[df["pass"] == False]
    if not fails.empty:
        print("\n=== FAIL DETAILS ===\n")
        print(fails.to_string(index=False))

    if args.out_csv:
        out = Path(args.out_csv)
        out.parent.mkdir(parents=True, exist_ok=True)
        df.to_csv(out, index=False)
        print(f"\nWrote per-item comparisons to: {out}")

if __name__ == "__main__":
    main()
