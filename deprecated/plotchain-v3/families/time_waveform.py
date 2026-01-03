# families/time_waveform.py
from __future__ import annotations

from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, axis_ticks_linear, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "time_waveform"
DEFAULT_N = 15

FINAL_FIELDS = ["frequency_hz", "vpp_v"]
CHECKPOINT_FIELDS = ["cp_period_s", "cp_vmax_v", "cp_vmin_v"]


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    f0 = float(pp["f0_hz"])
    A = float(pp["A"])

    gt: Dict[str, float] = {
        "frequency_hz": float(f0),
        "vpp_v": float(2.0 * A),
        "cp_period_s": float(1.0 / max(f0, 1e-12)),
        "cp_vmax_v": float(A),
        "cp_vmin_v": float(-A),
    }

    # Duty cycle is only meaningful for square waves. Keep it out of non-square items
    # to avoid scoring "undefined" quantities.
    if str(pp.get("wave_type", "")) == "square":
        gt["cp_duty"] = float(pp.get("duty", 0.5))

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


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    f0 = float(pp["f0_hz"])
    A = float(pp["A"])
    wave_type = str(pp["wave_type"])
    duty = float(pp.get("duty", 0.5))
    t_end = float(pp["t_end_s"])
    fs = float(pp["fs_hz"])

    t = np.arange(0.0, t_end, 1.0 / fs)
    y = _wave(t, wave_type, f0, A, duty)

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, y)
    ax.set_title(f"Time Waveform ({wave_type})")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Voltage (V)")
    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    save_figure(fig, out_path)
    plt.close(fig)

    x_min, x_max = 0.0, t_end
    y_min, y_max = float(np.min(y)), float(np.max(y))
    return {
        "x_min": x_min, "x_max": x_max,
        "y_min": y_min, "y_max": y_max,
        "tick_step_x": axis_ticks_linear(x_min, x_max, 6),
        "tick_step_y": axis_ticks_linear(y_min, y_max, 6),
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

        wave_type = str(rng.choice(["sine", "square", "triangle"]))
        f0 = float(rng.uniform(2.0, 120.0))
        A = float(rng.uniform(0.5, 5.0))
        duty = 0.5

        if difficulty == "clean":
            cycles = float(rng.uniform(2.5, 6.0))
            fs = 4000.0
        elif difficulty == "moderate":
            cycles = float(rng.uniform(1.5, 3.5))
            fs = 3500.0
        else:
            edge_tag = "short_window_low_cycles"
            cycles = float(rng.uniform(0.6, 1.1))
            fs = 3000.0

        if wave_type == "square":
            duty = float(rng.uniform(0.2, 0.8))  # stored for determinism

        t_end = float(cycles / max(f0, 1e-12))

        pp = {"wave_type": wave_type, "f0_hz": float(f0), "A": float(A), "fs_hz": float(fs), "t_end_s": float(t_end)}
        if wave_type == "square":
            pp["duty"] = float(duty)

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the time-domain waveform plot, estimate:\\n"
            "1) frequency_hz (Hz)\\n"
            "2) vpp_v (V)\\n"
            "Return JSON numeric values."
        )

        cp_fields = list(CHECKPOINT_FIELDS)
        if wave_type == "square":
            cp_fields.append("cp_duty")

        items.append({
            "id": f"{TYPE}_{i:03d}",
            "type": TYPE,
            "image_path": str(rel_img).replace('\\\\', '/'),
            "question": q,
            "ground_truth": gt,
            "plot_params": pp,
            "generation": {"seed": seed, "difficulty": difficulty, "edge_tag": edge_tag,
                           "final_fields": FINAL_FIELDS, "checkpoint_fields": cp_fields, **axis_meta},
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
