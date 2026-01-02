# families/iv_curve.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, axis_ticks_linear, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "iv_curve"
DEFAULT_N = 15

FINAL_FIELDS_RES = ["resistance_ohm"]
FINAL_FIELDS_DIODE = ["turn_on_voltage_v_at_target_i"]
CHECKPOINT_FIELDS_RES = ["cp_slope_ohm"]
CHECKPOINT_FIELDS_DIODE: List[str] = []


def _diode_voltage(Is: float, nVt: float, Rs: float, I: float) -> float:
    return float(nVt * math.log(I / Is + 1.0) + I * Rs)


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    kind = str(pp["kind"])
    if kind == "resistor":
        R = float(pp["R_ohm"])
        return {"resistance_ohm": float(R), "cp_slope_ohm": float(R)}
    Is = float(pp["Is"])
    nVt = float(pp["nVt"])
    Rs = float(pp["Rs"])
    target_I = float(pp["target_current_a"])
    Vt = _diode_voltage(Is, nVt, Rs, target_I)
    # Target current is provided in the question text; do not treat it as a prediction target.
    return {"turn_on_voltage_v_at_target_i": float(Vt)}


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    kind = str(pp["kind"])
    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.set_title("I–V Curve")
    ax.set_xlabel("Voltage (V)")
    ax.set_ylabel("Current (A)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    if kind == "resistor":
        R = float(pp["R_ohm"])
        vmax = float(pp["vmax_v"])
        v = np.linspace(0.0, vmax, int(pp["n_points"]))
        i = v / R
        ax.plot(v, i)
        x_min, x_max = 0.0, vmax
        y_min, y_max = 0.0, float(np.max(i))
    else:
        Is = float(pp["Is"])
        nVt = float(pp["nVt"])
        Rs = float(pp["Rs"])
        vmax = float(pp["vmax_v"])
        v = np.linspace(0.0, vmax, int(pp["n_points"]))
        I = np.zeros_like(v)
        for idx, V in enumerate(v):
            cur = 0.0
            for _ in range(25):
                cur = Is * (math.exp((V - cur * Rs) / max(nVt, 1e-12)) - 1.0)
            I[idx] = cur
        ax.plot(v, I)

        # Measurement aids: read V at a target current (disabled for edge cases)
        b = baseline_from_params(pp)
        target_I = float(b.get("target_current_a", pp.get("target_current_a", 0.0)))
        Vt = float(b.get("turn_on_voltage_v_at_target_i", 0.0))
        if getattr(meta, "difficulty", "clean") != "edge":
            ax.axhline(target_I, linestyle="--", linewidth=1.0, alpha=0.6)
            ax.axvline(Vt, linestyle="--", linewidth=1.0, alpha=0.6)
            ax.scatter([Vt], [target_I], s=24)
            ax.text(Vt, target_I, "  target", fontsize=9, va="bottom", ha="left")

        x_min, x_max = 0.0, vmax
        y_min, y_max = 0.0, float(np.max(I))

    save_figure(fig, out_path)
    plt.close(fig)
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

        kind = "resistor" if (i % 3 != 0) else "diode"

        if kind == "resistor":
            if difficulty == "clean":
                R = float(rng.uniform(50.0, 1000.0))
                vmax = float(rng.uniform(2.0, 10.0))
                n_points = 400
            elif difficulty == "moderate":
                R = float(rng.uniform(20.0, 2000.0))
                vmax = float(rng.uniform(1.5, 12.0))
                n_points = 350
            else:
                edge_tag = "shallow_slope_high_R"
                R = float(rng.uniform(1200.0, 4000.0))
                vmax = float(rng.uniform(2.0, 8.0))
                n_points = 250

            pp = {"kind": "resistor", "R_ohm": float(R), "vmax_v": float(vmax), "n_points": int(n_points)}
            gt = baseline_from_params(pp)
            final_fields = FINAL_FIELDS_RES
            checkpoint_fields = CHECKPOINT_FIELDS_RES
            q = "From the resistor I–V curve, estimate resistance_ohm (Ohms). Return JSON numeric values."
        else:
            if difficulty == "clean":
                Is = float(10 ** rng.uniform(-12.0, -9.0))
                nVt = float(rng.uniform(0.020, 0.030))
                Rs = float(rng.uniform(0.5, 8.0))
                target_I = float(rng.uniform(0.005, 0.03))
                vmax = 0.9
                n_points = 420
            elif difficulty == "moderate":
                Is = float(10 ** rng.uniform(-13.0, -9.0))
                nVt = float(rng.uniform(0.018, 0.032))
                Rs = float(rng.uniform(0.5, 15.0))
                target_I = float(rng.uniform(0.003, 0.04))
                vmax = 1.0
                n_points = 380
            else:
                edge_tag = "high_series_resistance"
                Is = float(10 ** rng.uniform(-12.0, -10.0))
                nVt = float(rng.uniform(0.020, 0.032))
                Rs = float(rng.uniform(10.0, 40.0))
                target_I = float(rng.uniform(0.01, 0.05))
                vmax = 1.2
                n_points = 280

            pp = {"kind": "diode", "Is": float(Is), "nVt": float(nVt), "Rs": float(Rs),
                  "target_current_a": float(target_I), "vmax_v": float(vmax), "n_points": int(n_points)}
            gt = baseline_from_params(pp)
            final_fields = FINAL_FIELDS_DIODE
            checkpoint_fields = CHECKPOINT_FIELDS_DIODE
            q = (
                f"From the diode I–V curve, at the given target current I = {target_I:.6f} A, "
                "estimate turn_on_voltage_v_at_target_i (V) (the voltage at that current). "
                "Return JSON numeric values."
            )

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name
        axis_meta = _render(pp, abs_img, meta)

        items.append({
            "id": f"{TYPE}_{i:03d}",
            "type": TYPE,
            "image_path": str(rel_img).replace('\\\\', '/'),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": final_fields, "checkpoint_fields": checkpoint_fields, **axis_meta},
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
