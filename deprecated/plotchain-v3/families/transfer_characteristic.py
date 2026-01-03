# families/transfer_characteristic.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, axis_ticks_linear, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "transfer_characteristic"
DEFAULT_N = 15

FINAL_FIELDS = ["small_signal_gain", "saturation_v"]
CHECKPOINT_FIELDS = ["cp_vin_at_saturation"]


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    gain = float(pp["gain"])
    Vsat = float(pp["Vsat"])
    vin_sat = Vsat / max(gain, 1e-12)
    return {"small_signal_gain": float(gain), "saturation_v": float(Vsat), "cp_vin_at_saturation": float(vin_sat)}


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    gain = float(pp["gain"])
    Vsat = float(pp["Vsat"])
    vin_min = float(pp["vin_min"])
    vin_max = float(pp["vin_max"])
    n_points = int(pp["n_points"])

    vin = np.linspace(vin_min, vin_max, n_points)
    vout = np.clip(gain * vin, -Vsat, Vsat)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(vin, vout)
    ax.set_title("Transfer Characteristic (Saturation)")
    ax.set_xlabel("Vin (V)")
    ax.set_ylabel("Vout (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    x_min, x_max = float(np.min(vin)), float(np.max(vin))
    y_min, y_max = float(np.min(vout)), float(np.max(vout))
    return {"x_min": x_min, "x_max": x_max, "y_min": y_min, "y_max": y_max,
            "tick_step_x": axis_ticks_linear(x_min, x_max, 6),
            "tick_step_y": axis_ticks_linear(y_min, y_max, 6)}


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
            gain = float(rng.uniform(0.8, 5.0))
            Vsat = float(rng.uniform(1.5, 6.0))
            span = float(rng.uniform(1.2, 2.0))
            n_points = 500
        elif difficulty == "moderate":
            gain = float(rng.uniform(0.6, 8.0))
            Vsat = float(rng.uniform(1.0, 7.0))
            span = float(rng.uniform(1.0, 2.5))
            n_points = 450
        else:
            edge_tag = "high_gain_early_saturation"
            gain = float(rng.uniform(6.0, 15.0))
            Vsat = float(rng.uniform(1.0, 3.0))
            span = float(rng.uniform(0.6, 1.2))
            n_points = 300

        vin_min = -span * (Vsat / max(gain, 1e-12)) * 2.5
        vin_max =  span * (Vsat / max(gain, 1e-12)) * 2.5

        pp = {"gain": float(gain), "Vsat": float(Vsat), "vin_min": float(vin_min), "vin_max": float(vin_max),
              "n_points": int(n_points)}

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the transfer characteristic plot, estimate:\\n"
            "1) small_signal_gain (slope in the linear region)\\n"
            "2) saturation_v (V)\\n"
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
