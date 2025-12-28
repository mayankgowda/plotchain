# families/bode_phase.py
from __future__ import annotations

import math
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

from common import ItemMeta, float_close, make_difficulty_plan, save_figure, stable_int_seed

TYPE = "bode_phase"
DEFAULT_N = 15

FINAL_FIELDS = ["cutoff_hz", "phase_deg_at_fq"]
CHECKPOINT_FIELDS = ["cp_phase_deg_at_fc"]

def baseline_from_params(pp: Dict[str, Any]) -> Dict[str, float]:
    fc = float(pp["fc_hz"])
    fq = float(pp["fq_hz"])
    phase_fq = -math.degrees(math.atan(fq / fc))
    return {
        "cutoff_hz": float(fc),
        "phase_deg_at_fq": float(phase_fq),
        "cp_phase_deg_at_fc": -45.0,
    }

def _render(pp: Dict[str, Any], out_path: Path, meta: ItemMeta) -> Dict[str, Any]:
    import matplotlib.pyplot as plt

    fc = float(pp["fc_hz"])
    fq = float(pp["fq_hz"])
    fmin = float(pp["fmin_hz"])
    fmax = float(pp["fmax_hz"])
    f = np.logspace(np.log10(fmin), np.log10(fmax), int(pp["n_points"]))
    phase = -np.degrees(np.arctan(f / fc))

    fig, ax = plt.subplots(figsize=(6.0, 3.6))
    ax.semilogx(f, phase)
    ax.set_title("Bode Phase (1st-order Low-pass)")
    ax.set_xlabel("Frequency (Hz)")
    ax.set_ylabel("Phase (deg)")
    if meta.difficulty != "edge":
        ax.grid(True, which="both", alpha=0.3)

    # Query frequency marker (f_q) for a checkpoint read.
    ax.axvline(fq, linestyle=":", linewidth=1.2, alpha=0.9)
    ax.text(fq, float(np.min(phase)), " f_q", rotation=90, va="bottom", ha="left", fontsize=9)

    save_figure(fig, out_path)
    plt.close(fig)

    return {
        "x_min": fmin, "x_max": fmax,
        "y_min": float(np.min(phase)), "y_max": float(np.max(phase)),
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
            fc = float(rng.uniform(40.0, 900.0))
            n_points = 520
            fmin, fmax = fc / 30.0, fc * 300.0
        elif difficulty == "moderate":
            fc = float(rng.uniform(20.0, 1500.0))
            n_points = 450
            fmin, fmax = fc / 40.0, fc * 500.0
        else:
            edge_tag = "sparse_log_phase_region"
            fc = float(rng.uniform(30.0, 1200.0))
            n_points = 220
            fmin, fmax = fc / 80.0, fc * 900.0

        # Query frequency marker (f_q) for reading a specific point on the curve.
        fq_mult = float(rng.choice([2.0, 5.0, 10.0, 20.0]))
        fq = float(fc * fq_mult)
        fq = float(min(max(fq, fmin * 1.1), fmax * 0.9))

        pp = {
            "fc_hz": float(fc),
            "fq_hz": float(fq),
            "fq_mult": float(fq_mult),
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
            "From the Bode phase plot (1st-order low-pass):\\n"
            "1) Find cutoff_hz (Hz) as the frequency where phase equals -45 deg.\\n"
            "2) A vertical dotted line marks f_q. Report phase_deg_at_fq (deg), the phase at f_q.\\n"
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
