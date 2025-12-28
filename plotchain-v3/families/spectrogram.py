# families/spectrogram.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np

from common import ItemMeta, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "spectrogram"
DEFAULT_N = 15

FINAL_FIELDS = ["f1_hz", "f2_hz", "switch_time_s"]
CHECKPOINT_FIELDS = ["cp_duration_s"]


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    return {
        "f1_hz": float(pp["f1_hz"]),
        "f2_hz": float(pp["f2_hz"]),
        "switch_time_s": float(pp["switch_time_s"]),
        "cp_duration_s": float(pp["duration_s"]),
    }


def _make_signal(pp: Dict[str, Any]) -> np.ndarray:
    fs = float(pp["fs_hz"])
    dur = float(pp["duration_s"])
    t = np.arange(0.0, dur, 1.0 / fs)
    f1 = float(pp["f1_hz"])
    f2 = float(pp["f2_hz"])
    ts = float(pp["switch_time_s"])

    y = np.zeros_like(t)
    y[t < ts] = np.sin(2 * np.pi * f1 * t[t < ts])
    y[t >= ts] = np.sin(2 * np.pi * f2 * t[t >= ts])
    return y


def _simple_spectrogram(y: np.ndarray, fs: float, nfft: int, hop: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    y = np.asarray(y, dtype=float)
    win = np.hanning(nfft)
    frames = []
    times = []
    for start in range(0, max(len(y) - nfft, 1), hop):
        seg = y[start:start + nfft]
        if len(seg) < nfft:
            seg = np.pad(seg, (0, nfft - len(seg)))
        spec = np.fft.rfft(seg * win)
        frames.append(np.abs(spec))
        times.append((start + nfft / 2) / fs)
    S = np.stack(frames, axis=1)
    freqs = np.fft.rfftfreq(nfft, d=1.0 / fs)
    return S, freqs, np.array(times)


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fs = float(pp["fs_hz"])
    y = _make_signal(pp)
    nfft = int(pp["nfft"])
    hop = int(pp["hop"])

    S, freqs, times = _simple_spectrogram(y, fs, nfft, hop)
    SdB = 20 * np.log10(np.maximum(S, 1e-12))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    im = ax.imshow(
        SdB,
        aspect="auto",
        origin="lower",
        extent=[float(times.min()), float(times.max()), float(freqs.min()), float(freqs.max())],
    )
    ax.set_title("Spectrogram (Tone Switch)")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Frequency (Hz)")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)

    save_figure(fig, out_path)
    plt.close(fig)

    return {"x_min": float(times.min()), "x_max": float(times.max()),
            "y_min": float(freqs.min()), "y_max": float(freqs.max()),
            "tick_step_x": None, "tick_step_y": None}


def generate(out_dir: Path, master_seed: int, n: int, images_root: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    diff_plan = make_difficulty_plan(n)
    fam_dir = images_root / TYPE
    fam_dir.mkdir(parents=True, exist_ok=True)

    for i in range(n):
        seed = stable_int_seed(master_seed, TYPE, i)
        rng = np.random.default_rng(seed)
        difficulty = diff_plan[i]
        edge_tag = ""

        duration = float(rng.uniform(1.8, 3.5))
        fs = 2000.0

        if difficulty == "clean":
            f1 = float(rng.uniform(80.0, 400.0))
            f2 = float(rng.uniform(600.0, 1200.0))
            switch = float(rng.uniform(0.7, duration - 0.7))
            nfft, hop = 256, 64
        elif difficulty == "moderate":
            f1 = float(rng.uniform(60.0, 500.0))
            f2 = float(rng.uniform(500.0, 1400.0))
            switch = float(rng.uniform(0.6, duration - 0.6))
            nfft, hop = 256, 96
        else:
            edge_tag = "close_freq_switch"
            f1 = float(rng.uniform(200.0, 600.0))
            f2 = float(f1 + rng.uniform(60.0, 180.0))
            switch = float(rng.uniform(0.5, duration - 0.5))
            nfft, hop = 256, 128

        pp = {"duration_s": float(duration), "fs_hz": float(fs), "f1_hz": float(f1), "f2_hz": float(f2),
              "switch_time_s": float(switch), "nfft": int(nfft), "hop": int(hop)}

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the spectrogram plot (tone switch), estimate:\\n"
            "1) f1_hz (Hz) (dominant frequency before the switch)\\n"
            "2) f2_hz (Hz) (dominant frequency after the switch)\\n"
            "3) switch_time_s (s)\\n"
            "Return JSON numeric values."
        )

        items.append({
            "id": f"{TYPE}_{i:03d}",
            "type": TYPE,
            "image_path": str(rel_img).replace('\\\\', '/'),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": FINAL_FIELDS, "checkpoint_fields": CHECKPOINT_FIELDS, **axis_meta},
        })

    return items


def validate(items: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    rows: List[Dict[str, Any]] = []
    for it in items:
        pred = baseline_from_params(it["plot_params"])
        gt = it["ground_truth"]
        for k in gt.keys():
            pv, gv = float(pred[k]), float(gt[k])
            ae = abs(pv - gv)
            ok = float_close(pv, gv, abs_tol=1e-12, rel_tol=1e-12)
            rows.append({"type": it["type"], "id": it["id"], "field": k, "pass": ok, "abs_err": ae})
    return rows
