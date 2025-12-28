# generate_plotchain_dataset.py
# Recreate PlotChain v1 quickly.
# Usage:
#   python generate_plotchain_dataset.py --out_dir plotchain_v1 --n_per_type 5 --seed 42
#
# This script is intentionally minimal; customize plot types or questions as needed.

import os, json, math, random, textwrap
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy import signal
import argparse

def set_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)

def save_plot(fig, path: str):
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)

def settling_time_2pct(t, y, final, tol=0.02):
    band = tol * abs(final) if final != 0 else tol
    lo, hi = final - band, final + band
    inside = (y >= lo) & (y <= hi)
    for i in range(len(t)):
        if inside[i] and inside[i:].all():
            return float(t[i])
    return float(t[-1])

def interp_x_at_y(x, y, target):
    idx = int(np.argmin(np.abs(y - target)))
    if y[idx] == target:
        return float(x[idx])
    if idx == 0:
        j = 1
    elif idx == len(y)-1:
        j = len(y)-2
    else:
        if (y[idx-1]-target)*(y[idx]-target) <= 0:
            j = idx-1
        else:
            j = idx+1
    x1, x2 = x[idx], x[j]
    y1, y2 = y[idx], y[j]
    if y2 == y1:
        return float(x1)
    return float(x1 + (target - y1) * (x2 - x1) / (y2 - y1))

def make_step_response_item(i, img_dir):
    zeta = float(np.random.uniform(0.15, 0.85))
    wn = float(np.random.uniform(2*math.pi*0.8, 2*math.pi*3.0))
    sys = signal.TransferFunction([wn**2], [1, 2*zeta*wn, wn**2])
    t = np.linspace(0, np.random.uniform(2.5, 6.0), 1200)
    tout, y = signal.step(sys, T=t)
    final = float(y[-1])
    peak = float(np.max(y))
    overshoot = 0.0 if final == 0 else max(0.0, (peak - final)/abs(final)*100.0)
    ts = settling_time_2pct(tout, y, final, tol=0.02)

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(tout, y, linewidth=2)
    ax.set_title("Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Amplitude (unitless)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"step_{i:02d}.png")
    save_plot(fig, img_path)

    return {
        "id": f"step_{i:02d}",
        "type": "step_response",
        "image_path": f"images/step_{i:02d}.png",
        "question": "From the step response, estimate (a) percent overshoot (%) and (b) 2% settling time (s). Show your reasoning steps.",
        "gold_chain": [
            "Identify the steady-state value from the final plateau of the response.",
            "Identify the peak value (maximum amplitude) from the curve.",
            "Compute percent overshoot = (peak − steady_state) / steady_state × 100%.",
            "Compute 2% settling time as the first time after which the response stays within ±2% of steady-state.",
        ],
        "ground_truth": {
            "percent_overshoot": round(overshoot, 3),
            "settling_time_s": round(ts, 3),
            "steady_state": round(final, 3),
        },
        "plot_params": {"zeta": zeta, "wn_rad_s": wn}
    }

def make_bode_item(i, img_dir):
    K = float(np.random.uniform(0.5, 2.0))
    fc = float(np.random.uniform(20, 400))
    tau = 1.0/(2*math.pi*fc)
    sys = signal.TransferFunction([K], [tau, 1.0])
    f = np.logspace(math.log10(1), math.log10(2000), 600)
    w = 2*math.pi*f
    _, mag_db, _ = signal.bode(sys, w=w)
    dc_gain_db = float(20*math.log10(K))
    target = dc_gain_db - 3.0
    f_cut = interp_x_at_y(f, mag_db, target)

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1,1,1)
    ax.semilogx(f, mag_db, linewidth=2)
    ax.set_title("Bode Magnitude Plot (Low-Pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    ax.grid(True, which="both", alpha=0.3)
    img_path = os.path.join(img_dir, f"bode_{i:02d}.png")
    save_plot(fig, img_path)

    return {
        "id": f"bode_{i:02d}",
        "type": "bode_magnitude",
        "image_path": f"images/bode_{i:02d}.png",
        "question": "From the Bode magnitude plot, estimate (a) DC gain (dB) and (b) the -3 dB cutoff frequency (Hz). Show your reasoning steps.",
        "gold_chain": [
            "Read the low-frequency (left-side) magnitude level as the DC gain in dB.",
            "Compute the target level for cutoff: DC gain − 3 dB.",
            "Find the frequency where the curve crosses the target level and report it as the cutoff frequency.",
        ],
        "ground_truth": {"dc_gain_db": round(dc_gain_db, 3), "cutoff_hz": round(float(f_cut), 3)},
        "plot_params": {"K": K, "fc_hz": fc, "tau_s": tau}
    }

def make_fft_item(i, img_dir):
    fs = int(np.random.choice([500, 1000, 2000]))
    duration = float(np.random.uniform(0.5, 1.5))
    t = np.arange(0, duration, 1/fs)
    n_tones = int(np.random.choice([2,3]))
    freqs = sorted(np.random.choice(np.arange(10, min(200, fs//2 - 10)), size=n_tones, replace=False).tolist())
    amps = np.random.uniform(0.6, 1.6, size=n_tones)
    x = np.zeros_like(t)
    for a, f0 in zip(amps, freqs):
        x += a*np.sin(2*np.pi*f0*t)
    x_noisy = x + np.random.normal(0, 0.15, size=len(t))

    N = len(t)
    X = np.fft.rfft(x_noisy)
    f_axis = np.fft.rfftfreq(N, d=1/fs)
    mag = np.abs(X) / (N/2)
    idx = int(np.argmax(mag[1:]) + 1)
    dom_f = float(f_axis[idx])

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(f_axis, mag, linewidth=2)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (a.u.)")
    ax.set_xlim(0, min(250, fs/2))
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"fft_{i:02d}.png")
    save_plot(fig, img_path)

    return {
        "id": f"fft_{i:02d}",
        "type": "fft_spectrum",
        "image_path": f"images/fft_{i:02d}.png",
        "question": "From the FFT magnitude spectrum, identify the dominant frequency component (Hz). Show your reasoning steps.",
        "gold_chain": [
            "Ignore the DC component at 0 Hz.",
            "Find the tallest peak in the magnitude spectrum.",
            "Read the corresponding frequency on the x-axis and report it as the dominant frequency.",
        ],
        "ground_truth": {"dominant_frequency_hz": round(dom_f, 3)},
        "plot_params": {"fs_hz": fs, "duration_s": duration, "tones_hz": freqs, "amps": [float(a) for a in amps]}
    }

def make_waveform_item(i, img_dir):
    fs = 2000
    duration = 0.02
    t = np.arange(0, duration, 1/fs)
    wave_type = random.choice(["sine", "square", "triangle", "trapezoid"])
    f0 = float(random.choice([50, 60, 100, 120, 200, 250]))
    A = float(np.random.uniform(0.8, 2.5))
    if wave_type == "sine":
        y = A*np.sin(2*np.pi*f0*t)
    elif wave_type == "square":
        duty = float(np.random.uniform(0.3, 0.7))
        y = A*signal.square(2*np.pi*f0*t, duty=duty)
    elif wave_type == "triangle":
        y = A*signal.sawtooth(2*np.pi*f0*t, width=0.5)
    else:
        tri = signal.sawtooth(2*np.pi*f0*t, width=0.5)
        y = A*np.clip(tri*1.4, -1, 1)
    y = y + np.random.normal(0, 0.05, size=len(t))
    vpp = float(np.max(y) - np.min(y))

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(t*1000, y, linewidth=2)
    ax.set_title(f"Time-Domain Waveform ({wave_type})")
    ax.set_xlabel("Time (ms)")
    ax.set_ylabel("Voltage (V)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"wave_{i:02d}.png")
    save_plot(fig, img_path)

    return {
        "id": f"wave_{i:02d}",
        "type": "time_waveform",
        "image_path": f"images/wave_{i:02d}.png",
        "question": "From the time-domain waveform, estimate (a) the fundamental frequency (Hz) and (b) the peak-to-peak amplitude (Vpp). Show your reasoning steps.",
        "gold_chain": [
            "Estimate the period by measuring the time between two successive similar points (e.g., peaks) on the waveform.",
            "Compute frequency = 1 / period.",
            "Estimate peak-to-peak amplitude as (maximum − minimum) value on the y-axis.",
        ],
        "ground_truth": {"frequency_hz": round(f0, 3), "vpp_v": round(vpp, 3)},
        "plot_params": {"wave_type": wave_type, "f0_hz": f0, "A": A, "fs_hz": fs}
    }

def make_iv_item(i, img_dir):
    kind = random.choice(["resistor", "diode"])
    v = np.linspace(0, 1.2, 300)
    if kind == "resistor":
        R = float(np.random.uniform(50, 600))
        i_a = v / R + np.random.normal(0, 0.0002, size=len(v))
        gt = {"resistance_ohm": round(R, 3)}
        question = "From the I–V curve, estimate the resistance (Ohms). Show your reasoning steps."
        gold_chain = [
            "For a resistor, the I–V relationship is linear: I = V / R.",
            "Estimate the slope dI/dV from two points on the line.",
            "Compute R = 1 / slope and report it.",
        ]
        title = "I–V Curve (Resistive Load)"
        params = {"kind": kind, "R_ohm": R}
    else:
        Is = 1e-12
        nVt = float(np.random.uniform(0.03, 0.06))
        Rs = float(np.random.uniform(2, 20))
        i_a = np.zeros_like(v)
        for k in range(len(v)):
            Vk = v[k]
            Ik = 0.0
            for _ in range(40):
                Ik = Is*(np.exp((Vk - Ik*Rs)/nVt) - 1.0)
                Ik = max(Ik, 0.0)
            i_a[k] = Ik
        i_a = i_a + np.random.normal(0, 0.00005, size=len(v))
        target_I = 0.01
        if np.max(i_a) < target_I:
            target_I = 0.001
        v_on = interp_x_at_y(v, i_a, target_I)
        gt = {"turn_on_voltage_v_at_target_i": round(float(v_on), 3), "target_current_a": float(target_I)}
        question = f"From the diode I–V curve, estimate the turn-on voltage (V) at I = {target_I*1000:.1f} mA. Show your reasoning steps."
        gold_chain = [
            "Identify the target current level on the y-axis.",
            "Find where the I–V curve reaches that current.",
            "Read the corresponding voltage on the x-axis and report it as the turn-on voltage at that current.",
        ]
        title = "I–V Curve (Diode)"
        params = {"kind": kind, "Is": Is, "nVt": nVt, "Rs": Rs}

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(v, i_a*1000, linewidth=2)
    ax.set_title(title)
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (mA)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"iv_{i:02d}.png")
    save_plot(fig, img_path)

    return {
        "id": f"iv_{i:02d}",
        "type": "iv_curve",
        "image_path": f"images/iv_{i:02d}.png",
        "question": question,
        "gold_chain": gold_chain,
        "ground_truth": gt,
        "plot_params": params
    }

def make_transfer_char_item(i, img_dir):
    gain = float(np.random.uniform(1.2, 6.0))
    Vsat = float(np.random.uniform(1.0, 4.5))
    offset = float(np.random.uniform(-0.4, 0.4))
    x = np.linspace(-2.0, 2.0, 400)
    y = np.clip(gain*x + offset, -Vsat, Vsat) + np.random.normal(0, 0.03, size=len(x))

    fig = plt.figure(figsize=(6.5, 4.4))
    ax = fig.add_subplot(1,1,1)
    ax.plot(x, y, linewidth=2)
    ax.set_title("Transfer Characteristic (Saturating Amplifier)")
    ax.set_xlabel("Input (V)")
    ax.set_ylabel("Output (V)")
    ax.grid(True, alpha=0.3)
    img_path = os.path.join(img_dir, f"transfer_{i:02d}.png")
    save_plot(fig, img_path)

    return {
        "id": f"transfer_{i:02d}",
        "type": "transfer_characteristic",
        "image_path": f"images/transfer_{i:02d}.png",
        "question": "From the transfer characteristic, estimate (a) the small-signal gain (slope in the linear region) and (b) the saturation voltage magnitude (V). Show your reasoning steps.",
        "gold_chain": [
            "Identify the central linear region where output changes proportionally with input.",
            "Estimate slope in that region using two points: gain ≈ Δoutput / Δinput.",
            "Identify the flat regions where the output saturates and read the saturation voltage magnitude.",
        ],
        "ground_truth": {"small_signal_gain": round(gain, 3), "saturation_v": round(Vsat, 3)},
        "plot_params": {"gain": gain, "Vsat": Vsat, "offset": offset}
    }

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out_dir", type=str, default="plotchain_v1")
    ap.add_argument("--n_per_type", type=int, default=5)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    set_seed(args.seed)
    out_dir = args.out_dir
    img_dir = os.path.join(out_dir, "images")
    os.makedirs(img_dir, exist_ok=True)

    items = []
    for i in range(args.n_per_type): items.append(make_step_response_item(i, img_dir))
    for i in range(args.n_per_type): items.append(make_bode_item(i, img_dir))
    for i in range(args.n_per_type): items.append(make_fft_item(i, img_dir))
    for i in range(args.n_per_type): items.append(make_waveform_item(i, img_dir))
    for i in range(args.n_per_type): items.append(make_iv_item(i, img_dir))
    for i in range(args.n_per_type): items.append(make_transfer_char_item(i, img_dir))

    with open(os.path.join(out_dir, "plotchain_v1.jsonl"), "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

    rows = []
    for it in items:
        rows.append({
            "id": it["id"],
            "type": it["type"],
            "image_path": it["image_path"],
            "question": it["question"],
            "ground_truth_json": json.dumps(it["ground_truth"], ensure_ascii=False),
            "gold_chain_json": json.dumps(it["gold_chain"], ensure_ascii=False),
            "plot_params_json": json.dumps(it["plot_params"], ensure_ascii=False),
        })
    pd.DataFrame(rows).to_csv(os.path.join(out_dir, "plotchain_v1.csv"), index=False)

    with open(os.path.join(out_dir, "README.md"), "w", encoding="utf-8") as f:
        f.write("PlotChain v1 generated. See plotchain_v1.jsonl for records.\n")

    print(f"Wrote {len(items)} items to {out_dir}/")

if __name__ == "__main__":
    main()
