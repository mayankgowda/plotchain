# families/bandpass_response.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "bandpass_response"
DEFAULT_N = 15

FINAL_FIELDS = ["resonance_hz", "bandwidth_hz"]
CHECKPOINT_FIELDS = ["cp_f1_3db_hz", "cp_f2_3db_hz", "cp_q_factor"]

def _mag_bandpass(f: np.ndarray, f0: float, Q: float, G: float) -> np.ndarray:
    x = f / f0
    denom = np.sqrt((1 - x**2)**2 + (x / Q)**2)
    return G * (x / np.maximum(denom, 1e-12))

def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    f0 = float(pp["f0_hz"])
    Q = float(pp["Q"])
    term = math.sqrt(1.0 + 1.0 / (4.0 * Q * Q))
    f1 = f0 * (term - 1.0 / (2.0 * Q))
    f2 = f0 * (term + 1.0 / (2.0 * Q))
    bw = f2 - f1
    return {
        "resonance_hz": float(f0),
        "bandwidth_hz": float(bw),
        "cp_f1_3db_hz": float(f1),
        "cp_f2_3db_hz": float(f2),
        "cp_q_factor": float(Q),
    }

def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    f0 = float(pp["f0_hz"])
    Q = float(pp["Q"])
    G = float(pp["G"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), int(pp["n_points"]))
    mag_db = 20 * np.log10(np.maximum(_mag_bandpass(f, f0, Q, G), 1e-12))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogx(f, mag_db)
    ax.set_title("Band-pass Magnitude Response")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    save_figure(fig, out_path)
    plt.close(fig)
    return {"x_min": fmin, "x_max": fmax, "y_min": float(np.min(mag_db)), "y_max": float(np.max(mag_db)),
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

        if difficulty == "clean":
            f0 = float(rng.uniform(80.0, 2500.0))
            Q = float(rng.uniform(1.5, 8.0))
            G = float(rng.uniform(0.8, 2.0))
            n_points = 520
            fmin, fmax = f0 / 30.0, f0 * 30.0
        elif difficulty == "moderate":
            f0 = float(rng.uniform(60.0, 4000.0))
            Q = float(rng.uniform(1.2, 12.0))
            G = float(rng.uniform(0.7, 2.5))
            n_points = 450
            fmin, fmax = f0 / 50.0, f0 * 50.0
        else:
            edge_tag = "high_Q_narrow_band_sparse"
            f0 = float(rng.uniform(120.0, 2500.0))
            Q = float(rng.uniform(10.0, 20.0))
            G = float(rng.uniform(0.9, 1.8))
            n_points = 240
            fmin, fmax = f0 / 120.0, f0 * 120.0

        pp = {"f0_hz": float(f0), "Q": float(Q), "G": float(G), "fmin_hz": float(fmin), "fmax_hz": float(fmax),
              "n_points": int(n_points)}

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the band-pass magnitude response plot, estimate:\\n"
            "1) resonance_hz (Hz) (frequency of peak)\\n"
            "2) bandwidth_hz (Hz) (difference between the two -3 dB frequencies)\\n"
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
