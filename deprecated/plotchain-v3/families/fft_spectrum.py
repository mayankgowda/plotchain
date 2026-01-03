# families/fft_spectrum.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "fft_spectrum"
DEFAULT_N = 15

FINAL_FIELDS = ["dominant_frequency_hz", "secondary_frequency_hz"]
CHECKPOINT_FIELDS = ["cp_peak_ratio"]


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    tones = list(map(float, pp["tones_hz"]))
    amps = list(map(float, pp["amps"]))
    idx = int(np.argmax(amps))
    sidx = int(np.argsort(amps)[-2]) if len(amps) >= 2 else idx
    ratio = float(amps[idx] / max(amps[sidx], 1e-12)) if len(amps) >= 2 else float("inf")
    return {
        "dominant_frequency_hz": float(tones[idx]),
        "secondary_frequency_hz": float(tones[sidx]),
        "cp_peak_ratio": float(ratio),
    }


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fmax = float(pp["fmax_hz"])
    tones = np.array(pp["tones_hz"], dtype=float)
    amps = np.array(pp["amps"], dtype=float)

    f = np.linspace(0.0, fmax, int(pp["n_points"]))
    mag = np.full_like(f, float(pp["noise_floor"]), dtype=float)

    for tone, amp in zip(tones, amps):
        sigma = float(pp["peak_sigma_hz"])
        mag += amp * np.exp(-0.5 * ((f - tone) / sigma) ** 2)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(f, mag)
    ax.set_title("FFT Magnitude Spectrum")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (arb.)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.3)

    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": 0.0, "x_max": fmax, "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
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

        fmax = float(rng.uniform(200.0, 2500.0))
        n_tones = int(rng.integers(2, 4))

        if difficulty == "edge":
            edge_tag = "close_peaks"
            base = float(rng.uniform(50.0, fmax * 0.6))
            tones = [base, base + float(rng.uniform(5.0, 25.0))]
            if n_tones == 3:
                tones.append(base + float(rng.uniform(30.0, 60.0)))
        else:
            tones = sorted(list(rng.uniform(30.0, fmax * 0.9, size=n_tones)))

        amps = list(rng.uniform(0.4, 2.0, size=n_tones))

        if difficulty == "clean":
            amps[int(np.argmax(amps))] *= 1.8
            noise_floor = 0.02
            sigma = 6.0
            n_points = 1200
        elif difficulty == "moderate":
            noise_floor = 0.05
            sigma = 8.0
            n_points = 900
        else:
            noise_floor = 0.08
            sigma = 10.0
            n_points = 800

        pp = {"fmax_hz": float(fmax), "tones_hz": [float(x) for x in tones], "amps": [float(a) for a in amps],
              "noise_floor": float(noise_floor), "peak_sigma_hz": float(sigma), "n_points": int(n_points)}

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the FFT magnitude spectrum plot, estimate:\\n"
            "1) dominant_frequency_hz (Hz)\\n"
            "2) secondary_frequency_hz (Hz) (second-largest peak)\\n"
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
