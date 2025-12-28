# families/step_response.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import (
    ItemMeta,
    axis_ticks_linear,
    float_close,
    make_difficulty_plan,
    save_figure,
    stable_int_seed,
)

TYPE = "step_response"
DEFAULT_N = 15

FINAL_FIELDS = ["percent_overshoot", "settling_time_s", "steady_state"]
CHECKPOINT_FIELDS = ["cp_peak_value", "cp_peak_time_s", "cp_band_upper", "cp_band_lower"]


def _step_response_underdamped(t: np.ndarray, zeta: float, wn: float, final: float) -> np.ndarray:
    wd = wn * math.sqrt(max(1.0 - zeta**2, 1e-12))
    phi = math.atan2(math.sqrt(max(1.0 - zeta**2, 1e-12)), zeta)
    y = 1.0 - (1.0 / math.sqrt(max(1.0 - zeta**2, 1e-12))) * np.exp(-zeta * wn * t) * np.sin(wd * t + phi)
    return final * y


def _settling_time_2pct(t: np.ndarray, y: np.ndarray, final: float, tol: float = 0.02) -> float:
    band = tol * abs(final) if abs(final) > 1e-12 else tol * max(float(np.max(np.abs(y))), 1e-12)
    lo, hi = final - band, final + band
    inside = (y >= lo) & (y <= hi)
    for i in range(len(t)):
        if inside[i] and bool(np.all(inside[i:])):
            return float(t[i])
    return float(t[-1])


def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    zeta = float(pp["zeta"])
    wn = float(pp["wn_rad_s"])
    final = float(pp["final_value"])
    t_end = float(pp["t_end_s"])
    n = int(pp["n_samples"])

    t = np.linspace(0.0, t_end, n)
    y = _step_response_underdamped(t, zeta, wn, final)

    peak_idx = int(np.argmax(y))
    peak_val = float(y[peak_idx])
    peak_t = float(t[peak_idx])

    overshoot = max((peak_val - final) / max(abs(final), 1e-12) * 100.0, 0.0)
    st = _settling_time_2pct(t, y, final, tol=0.02)

    band = 0.02 * abs(final) if abs(final) > 1e-12 else 0.02 * max(float(np.max(np.abs(y))), 1e-12)
    return {
        "percent_overshoot": float(overshoot),
        "settling_time_s": float(st),
        "steady_state": float(final),
        "cp_peak_value": float(peak_val),
        "cp_peak_time_s": float(peak_t),
        "cp_band_upper": float(final + band),
        "cp_band_lower": float(final - band),
    }


def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    gt = baseline_from_params(pp)
    t_end = float(pp["t_end_s"])
    n = int(pp["n_samples"])
    t = np.linspace(0.0, t_end, n)
    y = _step_response_underdamped(t, float(pp["zeta"]), float(pp["wn_rad_s"]), float(pp["final_value"]))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.plot(t, y)
    ax.set_title("Step Response")
    ax.set_xlabel("Time (s)")
    ax.set_ylabel("Output")

    if meta.difficulty != "edge":
        ax.grid(True, alpha=0.35)

    ax.axhline(gt["cp_band_upper"], linestyle="--", linewidth=1)
    ax.axhline(gt["cp_band_lower"], linestyle="--", linewidth=1)

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

        if difficulty == "clean":
            zeta = float(rng.uniform(0.25, 0.75))
            wn = float(rng.uniform(3.0, 10.0))
            final = float(rng.choice([1.0, 2.0, 3.0]))
        elif difficulty == "moderate":
            zeta = float(rng.uniform(0.20, 0.85))
            wn = float(rng.uniform(2.5, 12.0))
            final = float(rng.choice([1.0, 2.0, 4.0]))
        else:
            if rng.random() < 0.5:
                zeta = float(rng.uniform(0.88, 0.97))
                edge_tag = "near_critical_damping_small_overshoot"
            else:
                zeta = float(rng.uniform(0.12, 0.20))
                edge_tag = "high_overshoot_low_damping"
            wn = float(rng.uniform(2.0, 9.0))
            final = float(rng.choice([1.0, 2.0]))

        approx_ts = 4.0 / max(zeta * wn, 1e-6)
        t_end = float(max(2.0, 1.6 * approx_ts))
        n_samples = int(900 if difficulty != "edge" else 1100)

        pp = {
            "zeta": float(zeta),
            "wn_rad_s": float(wn),
            "final_value": float(final),
            "t_end_s": float(t_end),
            "n_samples": int(n_samples),
        }

        meta = ItemMeta(difficulty=difficulty, edge_tag=edge_tag, seed=seed)
        img_name = f"{TYPE}_{i:03d}.png"
        rel_img = Path("images") / TYPE / img_name
        abs_img = fam_dir / img_name

        axis_meta = _render(pp, abs_img, meta)
        gt = baseline_from_params(pp)

        q = (
            "From the step response plot, estimate:\\n"
            "1) percent_overshoot (%)\\n"
            "2) settling_time_s (2% criterion, seconds)\\n"
            "3) steady_state value\\n"
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
        pp = it["plot_params"]
        gt = it["ground_truth"]
        pred = baseline_from_params(pp)
        for k, gv in gt.items():
            pv = pred.get(k)
            ae = abs(float(pv) - float(gv))
            ok = float_close(float(pv), float(gv), abs_tol=1e-12, rel_tol=1e-12)
            rows.append({
                "type": it["type"],
                "id": it["id"],
                "field": k,
                "pass": bool(ok),
                "abs_err": float(ae),
            })
    return rows
