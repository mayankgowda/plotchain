# families/bode_magnitude.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "bode_magnitude"
DEFAULT_N = 15

FINAL_FIELDS = ["dc_gain_db", "cutoff_hz"]
CHECKPOINT_FIELDS = ["cp_mag_at_fc_db", "cp_slope_db_per_decade"]


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    dc_gain_db = 20.0 * math.log10(max(K, 1e-12))
    mag_fc = dc_gain_db - 20.0 * math.log10(math.sqrt(2.0))
    return {
        "dc_gain_db": float(dc_gain_db),
        "cutoff_hz": float(fc),
        "cp_mag_at_fc_db": float(mag_fc),
        "cp_slope_db_per_decade": -20.0,
    }


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    K = float(pp["K"])
    fc = float(pp["fc_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])

    f = np.logspace(np.log10(fmin), np.log10(fmax), int(pp["n_points"]))
    mag = 20.0 * np.log10(K / np.sqrt(1.0 + (f / fc) ** 2))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogx(f, mag)
    ax.set_title("Bode Magnitude (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Magnitude (dB)")
    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(mag)), "y_max": float(np.max(mag)),
        "tick_step_x": None,
        "tick_step_y": None,
    }


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
            K = float(rng.uniform(0.7, 5.0))
            fc = float(rng.uniform(40.0, 800.0))
            n_points = 500
            fmin, fmax = fc / 20.0, fc * 200.0
        elif difficulty == "moderate":
            K = float(rng.uniform(0.5, 8.0))
            fc = float(rng.uniform(20.0, 1500.0))
            n_points = 450
            fmin, fmax = fc / 30.0, fc * 300.0
        else:
            edge_tag = "sparse_log_ticks_wide_range"
            K = float(rng.uniform(0.6, 6.0))
            fc = float(rng.uniform(30.0, 1200.0))
            n_points = 220
            fmin, fmax = fc / 60.0, fc * 800.0

        pp = {
            "K": float(K),
            "fc_hz": float(fc),
            "fmin_hz": float(fmin),
            "fmax_hz": float(fmax),
            "n_points": int(n_points),
        }

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the Bode magnitude plot (1st-order low-pass), estimate:\\n"
            "1) dc_gain_db (dB)\\n"
            "2) cutoff_hz (Hz) at -3 dB\\n"
            "Return JSON numeric values."
        )

        items.append({
            "id": f"{TYPE}_{i:03d}",
            "type": TYPE,
            "image_path": str(rel_img).replace('\\\\', '/'),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {
                "seed": seed,
                "difficulty": difficulty,
                "edge_tag": edge_tag,
                "final_fields": FINAL_FIELDS,
                "checkpoint_fields": CHECKPOINT_FIELDS,
                **axis_meta,
            },
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
