#!/usr/bin/env python3
"""
run_plotchain_eval.py

Run multimodal LLMs on PlotChain (image + question) and score vs ground truth.

Modes:
  - --mode run   : call models, write raw_<provider>_<model>*.jsonl + scored CSVs
  - --mode score : read raw_*.jsonl and (re)score to produce CSVs

Outputs (in --out_dir):
  - raw_<provider>_<model>.jsonl  (or raw_<...>_<timestamp>.jsonl if not --overwrite)
  - per_item.csv                  (one row per (item, field))
  - item_level.csv                (one row per item; all-final/all-cp flags)
  - overall.csv                   (aggregate by model)
  - summary.csv                   (aggregate by model/type/scope/field)

Notes:
- This script is dataset-driven: it reads each item's generation.final_fields + generation.checkpoint_fields.
- It supports new families without hardcoding FIELDS_BY_TYPE.
- It preserves raw model outputs for audit / IEEE artifacts.
"""

from __future__ import annotations

import argparse
import base64
import json
import math
import os
import re
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import numpy as np
import pandas as pd
from tqdm import tqdm

# Import concurrent evaluation
from concurrent_eval import run_concurrent_evaluation, PROVIDER_RATE_LIMITS


# -------------------------
# JSONL helpers
# -------------------------

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items: List[Dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def iter_jsonl(path: Path) -> Iterable[Dict[str, Any]]:
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)


def get_item_id(it: Dict[str, Any]) -> str:
    return str(it.get("id", it.get("uid", "")))


def get_item_type(it: Dict[str, Any]) -> str:
    return str(it.get("type", it.get("plot_type", "")))


def get_question(it: Dict[str, Any]) -> str:
    for k in ("question", "prompt", "query", "text_question"):
        if k in it:
            return str(it[k])
    raise KeyError("Could not find question field (expected one of question/prompt/query/text_question).")


def get_ground_truth(it: Dict[str, Any]) -> Dict[str, Any]:
    gt = it.get("ground_truth")
    if gt is None:
        raise KeyError("Missing ground_truth")
    if not isinstance(gt, dict):
        raise TypeError("ground_truth must be a dict")
    return gt


def resolve_image_path(it: Dict[str, Any], images_root: Optional[Path], jsonl_path: Path) -> Path:
    for k in ("image_path", "image_file", "plot_path", "image"):
        if k in it:
            p = Path(str(it[k]))
            if p.is_absolute():
                return p
            root = images_root if images_root is not None else jsonl_path.parent
            return (root / p).resolve()
    raise KeyError("Could not find image path field (expected one of image_path/image_file/plot_path/image).")


def guess_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"


# -------------------------
# Expected fields from dataset metadata
# -------------------------

def expected_fields(it: Dict[str, Any]) -> List[str]:
    """
    Prefer generator-provided ordering (final_fields + checkpoint_fields).
    Fallback: ground_truth keys (sorted).
    """
    gen = it.get("generation", {}) or {}
    ff = gen.get("final_fields")
    cf = gen.get("checkpoint_fields")
    fields: List[str] = []
    if isinstance(ff, list):
        fields.extend([str(x) for x in ff])
    if isinstance(cf, list):
        fields.extend([str(x) for x in cf])

    if fields:
        seen = set()
        out: List[str] = []
        for k in fields:
            if k not in seen:
                seen.add(k)
                out.append(k)
        return out

    return sorted([str(k) for k in get_ground_truth(it).keys()])


def is_checkpoint_field(field: str) -> bool:
    return str(field).startswith("cp_")


# -------------------------
# Prompt helpers
# -------------------------

UNITS_HINT: Dict[str, str] = {
    # Common
    "frequency_hz": "Hz",
    "dominant_frequency_hz": "Hz",
    "secondary_frequency_hz": "Hz",
    "cutoff_hz": "Hz",
    "resonance_hz": "Hz",
    "bandwidth_hz": "Hz",
    "dc_gain_db": "dB",
    "cp_gain_db_at_res": "dB",
    "cp_mag_at_fc_db": "dB",
    "cp_slope_db_per_decade": "dB/dec",
    "vpp_v": "V",
    "steady_state": "unitless",
    "percent_overshoot": "%",
    "settling_time_s": "s",
    "cp_period_s": "s",
    "cp_peak_time_s": "s",
    "cp_peak_value": "unitless",
    "cp_band_lower": "unitless",
    "cp_band_upper": "unitless",
    "cp_vmax_v": "V",
    "cp_vmin_v": "V",
    "cp_duty": "unitless",
    "phase_deg_at_10fc": "deg",
    "phase_deg_at_fq_hz": "deg",
    "cp_phase_deg_at_fc": "deg",
    "cp_q_factor": "unitless",
    "cp_peak_ratio": "unitless",

    # IV
    "resistance_ohm": "Ohm",
    "turn_on_voltage_v_at_target_i": "V",
    "target_current_a": "A",
    "cp_slope_ohm": "Ohm",
    "cp_Is": "A",
    "cp_nVt": "V",
    "cp_Rs": "Ohm",

    # Transfer
    "small_signal_gain": "unitless",
    "saturation_v": "V",
    "cp_vin_at_saturation": "V",

    # ROC-ish
    "auc": "unitless",
    "tpr": "unitless",
    "fpr": "unitless",
    "precision": "unitless",
    "recall": "unitless",
    "eer": "unitless",

    # Constellation-ish
    "snr_db": "dB",
    "evm_percent": "%",
    "ber": "unitless",

    # Learning curve-ish
    "best_epoch": "epoch",
    "early_stop_epoch": "epoch",
    "min_val_loss": "unitless",
    "final_val_loss": "unitless",
    "final_train_loss": "unitless",
    "generalization_gap": "unitless",

    # Stress-strain curve
    "yield_strength_mpa": "MPa",
    "uts_mpa": "MPa",
    "fracture_strain": "unitless",
    "cp_yield_strain": "unitless",
    "cp_uts_strain": "unitless",

    # Torque-speed curve
    "stall_torque_nm": "N·m",
    "no_load_speed_rpm": "rpm",
    "cp_torque_at_speed_q_nm": "N·m",

    # Pump curve
    "head_at_qop_m": "m",
    "q_at_half_head_m3h": "m³/h",
    "cp_shutoff_head_m": "m",
    "cp_qmax_m3h": "m³/h",

    # S-N curve
    "stress_at_1e5_mpa": "MPa",
    "stress_at_1e3_mpa": "MPa",
    "endurance_limit_mpa": "MPa",
    "cp_stress_at_1e3_mpa": "MPa",
}


def build_schema_comment(field: str) -> str:
    u = UNITS_HINT.get(field, "")
    return f"  \"{field}\": <number or null>{('  // ' + u) if u else ''}"


def build_prompt(it: Dict[str, Any]) -> str:
    fields = expected_fields(it)
    schema = "{\n" + ",\n".join(build_schema_comment(f) for f in fields) + "\n}"

    return (
        "You are given an engineering plot image. Read the plot and answer the question.\n\n"
        f"Question:\n{get_question(it)}\n\n"
        "Return ONLY a single JSON object matching this schema (numbers or null; no strings; no units; no extra keys):\n"
        f"{schema}\n\n"
        "Notes:\n"
        "- Use cp_* fields as intermediate plot reads (checkpoints) that help verify your understanding of the plot.\n"
        "- If you cannot determine a value, output null for that key.\n"
        "- IMPORTANT: do NOT output arithmetic expressions like 1025/615; output a decimal number.\n"
    )


# -------------------------
# Robust JSON extraction (+ fraction sanitization)
# -------------------------

_JSON_RE = re.compile(r"\{.*\}", re.DOTALL)
_FENCE_RE = re.compile(r"```(?:json)?\s*([\s\S]*?)\s*```", re.IGNORECASE)

# Replace simple fractions like 1025/615 inside JSON.
_FRAC_RE = re.compile(r"(?<![0-9.])(\d+(?:\.\d+)?)\s*/\s*(\d+(?:\.\d+)?)(?![0-9.])")
_TRAILING_COMMA_RE = re.compile(r",\s*([}\]])")


def _sanitize_json_candidate(s: str) -> str:
    s = s.strip()

    def _frac_to_float(m: re.Match) -> str:
        a = float(m.group(1))
        b = float(m.group(2))
        if b == 0.0:
            return "null"
        val = a / b
        return f"{val:.12g}"

    s = _FRAC_RE.sub(_frac_to_float, s)
    s = _TRAILING_COMMA_RE.sub(r"\1", s)
    return s


def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    # 1) Try direct parse
    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    candidates: List[str] = []

    # 2) Try fenced block
    m_f = _FENCE_RE.search(text)
    if m_f:
        candidates.append(m_f.group(1))

    # 3) Try first {...} blob
    m = _JSON_RE.search(text)
    if m:
        candidates.append(m.group(0))

    for cand in candidates:
        s = _sanitize_json_candidate(cand)
        try:
            obj = json.loads(s)
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue

    return None


# -------------------------
# Scoring primitives
# -------------------------

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        if not s:
            return None
        try:
            return float(s)
        except Exception:
            return None
    return None


def abs_err(pred: Optional[float], gold: Optional[float]) -> float:
    if pred is None or gold is None:
        return float("nan")
    return float(abs(pred - gold))


def rel_err(pred: Optional[float], gold: Optional[float]) -> float:
    if pred is None or gold is None:
        return float("nan")
    denom = max(abs(gold), 1e-12)
    return float(abs(pred - gold) / denom)


# -------------------------
# Tolerances (explicit for known families + robust fallbacks for new)
# -------------------------

def tolerances_plotread() -> Dict[Tuple[str, str], Tuple[float, float]]:
    """
    Fair "human plot-read" tolerances.

    IMPORTANT: For new families, we include common field-name patterns and also
    keep a heuristic fallback so the evaluator doesn't break.
    """
    T: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # --- Step response ---
    T[("step_response", "percent_overshoot")] = (2.5, 0.07)
    T[("step_response", "settling_time_s")] = (0.35, 0.20)
    T[("step_response", "steady_state")] = (0.05, 0.02)
    T[("step_response", "cp_peak_time_s")] = (0.10, 0.20)
    T[("step_response", "cp_peak_value")] = (0.08, 0.10)

    # --- Bode magnitude ---
    T[("bode_magnitude", "dc_gain_db")] = (0.5, 0.03)
    T[("bode_magnitude", "cutoff_hz")] = (5.0, 0.05)
    T[("bode_magnitude", "cp_mag_at_fc_db")] = (0.8, 0.05)
    T[("bode_magnitude", "cp_slope_db_per_decade")] = (1.5, 0.10)

    # --- Bode phase ---
    T[("bode_phase", "cutoff_hz")] = (5.0, 0.05)
    T[("bode_phase", "cp_phase_deg_at_fc")] = (5.0, 0.08)
    T[("bode_phase", "phase_deg_at_10fc")] = (6.0, 0.10)
    T[("bode_phase", "phase_deg_at_fq_hz")] = (6.0, 0.10)

    # --- Bandpass response ---
    T[("bandpass_response", "resonance_hz")] = (0.0, 0.05)
    T[("bandpass_response", "bandwidth_hz")] = (0.0, 0.08)
    T[("bandpass_response", "cp_q_factor")] = (0.25, 0.12)
    T[("bandpass_response", "cp_f1_3db_hz")] = (0.0, 0.08)
    T[("bandpass_response", "cp_f2_3db_hz")] = (0.0, 0.08)
    T[("bandpass_response", "cp_gain_db_at_res")] = (1.0, 0.08)

    # --- Time waveform ---
    T[("time_waveform", "frequency_hz")] = (1.0, 0.02)
    T[("time_waveform", "vpp_v")] = (0.20, 0.10)
    T[("time_waveform", "cp_period_s")] = (0.01, 0.05)
    T[("time_waveform", "cp_vmax_v")] = (0.10, 0.08)
    T[("time_waveform", "cp_vmin_v")] = (0.10, 0.08)
    T[("time_waveform", "cp_duty")] = (0.05, 0.10)

    # --- FFT spectrum ---
    T[("fft_spectrum", "dominant_frequency_hz")] = (2.0, 0.03)
    T[("fft_spectrum", "secondary_frequency_hz")] = (3.0, 0.05)
    T[("fft_spectrum", "cp_peak_ratio")] = (0.20, 0.15)

    # --- Spectrogram ---
    T[("spectrogram", "f1_hz")] = (5.0, 0.05)
    T[("spectrogram", "f2_hz")] = (5.0, 0.05)
    T[("spectrogram", "switch_time_s")] = (0.10, 0.20)
    T[("spectrogram", "cp_duration_s")] = (0.10, 0.20)

    # --- Transfer characteristic ---
    T[("transfer_characteristic", "small_signal_gain")] = (0.20, 0.10)
    T[("transfer_characteristic", "saturation_v")] = (0.10, 0.10)
    T[("transfer_characteristic", "cp_vin_at_saturation")] = (0.15, 0.12)

    # --- IV curve (legacy combined) ---
    T[("iv_curve", "resistance_ohm")] = (10.0, 0.05)
    T[("iv_curve", "turn_on_voltage_v_at_target_i")] = (0.08, 0.15)
    T[("iv_curve", "target_current_a")] = (0.01, 0.10)
    T[("iv_curve", "cp_slope_ohm")] = (15.0, 0.10)
    # Diode checkpoints
    T[("iv_curve", "cp_Is")] = (0.0, 2.0)
    T[("iv_curve", "cp_nVt")] = (0.02, 0.30)
    T[("iv_curve", "cp_Rs")] = (5.0, 0.30)

    # --- IV split families ---
    T[("iv_resistor", "resistance_ohm")] = (10.0, 0.05)
    T[("iv_resistor", "cp_slope_ohm")] = (15.0, 0.10)
    T[("iv_diode", "turn_on_voltage_v_at_target_i")] = (0.08, 0.15)
    T[("iv_diode", "target_current_a")] = (0.01, 0.10)
    T[("iv_diode", "cp_Is")] = (0.0, 2.0)
    T[("iv_diode", "cp_nVt")] = (0.02, 0.30)
    T[("iv_diode", "cp_Rs")] = (5.0, 0.30)

    # --- Stress-strain curve ---
    T[("stress_strain", "yield_strength_mpa")] = (25.0, 0.07)
    T[("stress_strain", "uts_mpa")] = (25.0, 0.07)
    T[("stress_strain", "fracture_strain")] = (0.02, 0.10)
    T[("stress_strain", "cp_yield_strain")] = (0.0006, 0.15)
    T[("stress_strain", "cp_uts_strain")] = (0.02, 0.15)

    # --- Torque-speed curve ---
    T[("torque_speed", "stall_torque_nm")] = (0.7, 0.12)
    T[("torque_speed", "no_load_speed_rpm")] = (120.0, 0.07)
    T[("torque_speed", "cp_torque_at_speed_q_nm")] = (0.7, 0.12)

    # --- Pump curve ---
    T[("pump_curve", "head_at_qop_m")] = (2.0, 0.07)
    T[("pump_curve", "q_at_half_head_m3h")] = (5.0, 0.10)
    T[("pump_curve", "cp_shutoff_head_m")] = (2.0, 0.07)
    T[("pump_curve", "cp_qmax_m3h")] = (7.0, 0.12)

    # --- S-N curve ---
    T[("sn_curve", "stress_at_1e5_mpa")] = (25.0, 0.10)
    T[("sn_curve", "endurance_limit_mpa")] = (25.0, 0.10)
    T[("sn_curve", "cp_stress_at_1e3_mpa")] = (25.0, 0.10)

    return T


def tolerances_strict() -> Dict[Tuple[str, str], Tuple[float, float]]:
    T: Dict[Tuple[str, str], Tuple[float, float]] = {}

    # Step response
    T[("step_response", "percent_overshoot")] = (2.0, 0.05)
    T[("step_response", "settling_time_s")] = (0.20, 0.10)
    T[("step_response", "steady_state")] = (0.03, 0.02)
    T[("step_response", "cp_peak_time_s")] = (0.07, 0.15)
    T[("step_response", "cp_peak_value")] = (0.06, 0.08)

    # Bode magnitude
    T[("bode_magnitude", "dc_gain_db")] = (0.25, 0.02)
    T[("bode_magnitude", "cutoff_hz")] = (2.0, 0.03)
    T[("bode_magnitude", "cp_mag_at_fc_db")] = (0.5, 0.03)
    T[("bode_magnitude", "cp_slope_db_per_decade")] = (1.0, 0.07)

    # Bode phase
    T[("bode_phase", "cutoff_hz")] = (2.0, 0.03)
    T[("bode_phase", "cp_phase_deg_at_fc")] = (3.0, 0.06)
    T[("bode_phase", "phase_deg_at_10fc")] = (4.0, 0.08)
    T[("bode_phase", "phase_deg_at_fq_hz")] = (4.0, 0.08)

    # Bandpass response
    T[("bandpass_response", "resonance_hz")] = (0.0, 0.03)
    T[("bandpass_response", "bandwidth_hz")] = (0.0, 0.05)
    T[("bandpass_response", "cp_q_factor")] = (0.20, 0.10)
    T[("bandpass_response", "cp_f1_3db_hz")] = (0.0, 0.05)
    T[("bandpass_response", "cp_f2_3db_hz")] = (0.0, 0.05)
    T[("bandpass_response", "cp_gain_db_at_res")] = (0.7, 0.05)

    # Time waveform
    T[("time_waveform", "frequency_hz")] = (0.5, 0.01)
    T[("time_waveform", "vpp_v")] = (0.12, 0.07)
    T[("time_waveform", "cp_period_s")] = (0.006, 0.03)
    T[("time_waveform", "cp_vmax_v")] = (0.06, 0.05)
    T[("time_waveform", "cp_vmin_v")] = (0.06, 0.05)
    T[("time_waveform", "cp_duty")] = (0.03, 0.06)

    # FFT spectrum
    T[("fft_spectrum", "dominant_frequency_hz")] = (1.0, 0.02)
    T[("fft_spectrum", "secondary_frequency_hz")] = (2.0, 0.03)
    T[("fft_spectrum", "cp_peak_ratio")] = (0.12, 0.10)

    # Spectrogram
    T[("spectrogram", "f1_hz")] = (3.0, 0.03)
    T[("spectrogram", "f2_hz")] = (3.0, 0.03)
    T[("spectrogram", "switch_time_s")] = (0.06, 0.12)
    T[("spectrogram", "cp_duration_s")] = (0.06, 0.12)

    # Transfer characteristic
    T[("transfer_characteristic", "small_signal_gain")] = (0.12, 0.06)
    T[("transfer_characteristic", "saturation_v")] = (0.06, 0.06)
    T[("transfer_characteristic", "cp_vin_at_saturation")] = (0.10, 0.08)

    # IV curve (legacy)
    T[("iv_curve", "resistance_ohm")] = (5.0, 0.03)
    T[("iv_curve", "turn_on_voltage_v_at_target_i")] = (0.05, 0.10)
    T[("iv_curve", "target_current_a")] = (0.005, 0.07)
    T[("iv_curve", "cp_slope_ohm")] = (10.0, 0.07)
    T[("iv_curve", "cp_Is")] = (0.0, 1.0)
    T[("iv_curve", "cp_nVt")] = (0.015, 0.20)
    T[("iv_curve", "cp_Rs")] = (3.0, 0.20)

    # IV split
    T[("iv_resistor", "resistance_ohm")] = (5.0, 0.03)
    T[("iv_resistor", "cp_slope_ohm")] = (10.0, 0.07)
    T[("iv_diode", "turn_on_voltage_v_at_target_i")] = (0.05, 0.10)
    T[("iv_diode", "target_current_a")] = (0.005, 0.07)
    T[("iv_diode", "cp_Is")] = (0.0, 1.0)
    T[("iv_diode", "cp_nVt")] = (0.015, 0.20)
    T[("iv_diode", "cp_Rs")] = (3.0, 0.20)

    # --- Stress-strain curve ---
    T[("stress_strain", "yield_strength_mpa")] = (25.0, 0.07)
    T[("stress_strain", "uts_mpa")] = (25.0, 0.07)
    T[("stress_strain", "fracture_strain")] = (0.02, 0.10)
    T[("stress_strain", "cp_yield_strain")] = (0.0006, 0.15)
    T[("stress_strain", "cp_uts_strain")] = (0.02, 0.15)

    # --- Torque-speed curve ---
    T[("torque_speed", "stall_torque_nm")] = (0.7, 0.12)
    T[("torque_speed", "no_load_speed_rpm")] = (120.0, 0.07)
    T[("torque_speed", "cp_torque_at_speed_q_nm")] = (0.7, 0.12)

    # --- Pump curve ---
    T[("pump_curve", "head_at_qop_m")] = (2.0, 0.07)
    T[("pump_curve", "q_at_half_head_m3h")] = (5.0, 0.10)
    T[("pump_curve", "cp_shutoff_head_m")] = (2.0, 0.07)
    T[("pump_curve", "cp_qmax_m3h")] = (7.0, 0.12)

    # --- S-N curve ---
    T[("sn_curve", "stress_at_1e5_mpa")] = (25.0, 0.10)
    T[("sn_curve", "endurance_limit_mpa")] = (25.0, 0.10)
    T[("sn_curve", "cp_stress_at_1e3_mpa")] = (25.0, 0.10)

    return T


def _infer_tol(field: str, gold: float, policy: str) -> Tuple[float, float]:
    """
    Heuristic tolerances for unseen (type, field).
    We make it smarter for new families by using field-name semantics.
    """
    g = abs(float(gold))
    f = field.lower()

    if policy == "strict":
        scale = 0.6
    else:
        scale = 1.0

    # ---- ROC-ish (0..1) ----
    if "auc" in f:
        return (0.01 if policy == "strict" else 0.02, 0.05)
    if any(x in f for x in ("tpr", "fpr", "precision", "recall")):
        return (0.03 if policy == "strict" else 0.05, 0.20)
    if "eer" in f:
        return (0.02 if policy == "strict" else 0.03, 0.25)

    # ---- Learning curve-ish ----
    if "epoch" in f:
        # epochs are integers -> allow +/-1 in plotread, exact in strict (if you designed ticks to be integers)
        return (0.0 if policy == "strict" else 1.0, 0.0)
    if "loss" in f:
        return (0.03 if policy == "strict" else 0.05, 0.12)
    if "gap" in f:
        return (0.04 if policy == "strict" else 0.06, 0.15)
    if any(x in f for x in ("acc", "accuracy")):
        return (0.02 if policy == "strict" else 0.03, 0.10)

    # ---- Constellation-ish ----
    if "snr" in f and f.endswith("_db"):
        return (0.7 if policy == "strict" else 1.0, 0.10)
    if "evm" in f:
        return (1.2 if policy == "strict" else 2.0, 0.25)
    if "ber" in f:
        return (0.005 if policy == "strict" else 0.01, 0.35)

    # ---- Nyquist-ish margins ----
    if "margin" in f and f.endswith("_db"):
        return (0.7 if policy == "strict" else 1.0, 0.12)
    if "margin" in f and ("deg" in f or f.endswith("_deg")):
        return (3.0 if policy == "strict" else 5.0, 0.15)

    # ---- Generic suffix-based ----
    if f.endswith("_hz"):
        abs_tol = max(1.0, scale * 0.03 * max(g, 1.0))
        return (abs_tol, 0.06 if policy != "strict" else 0.03)
    if f.endswith("_db"):
        abs_tol = scale * (0.7 if policy == "strict" else 1.0)
        return (abs_tol, 0.08 if policy != "strict" else 0.05)
    if f.endswith("_deg"):
        abs_tol = scale * (3.0 if policy == "strict" else 5.0)
        return (abs_tol, 0.12 if policy != "strict" else 0.08)
    if f.endswith("_s"):
        abs_tol = max(0.02, scale * 0.12 * max(g, 0.1))
        return (abs_tol, 0.20 if policy != "strict" else 0.12)
    if f.endswith("_v"):
        abs_tol = max(0.02, scale * 0.08 * max(g, 0.2))
        return (abs_tol, 0.15 if policy != "strict" else 0.10)
    if f.endswith("_ohm"):
        abs_tol = max(1.0, scale * 0.08 * max(g, 10.0))
        return (abs_tol, 0.12 if policy != "strict" else 0.08)
    if f.endswith("_a"):
        abs_tol = max(0.001, scale * 0.15 * max(g, 0.01))
        return (abs_tol, 0.20 if policy != "strict" else 0.12)

    abs_tol = max(0.05, scale * 0.08 * max(g, 1.0))
    return (abs_tol, 0.10 if policy != "strict" else 0.06)


def get_tol(
    tol_map: Dict[Tuple[str, str], Tuple[float, float]],
    policy: str,
    typ: str,
    field: str,
    gold: float,
) -> Tuple[float, float]:
    if (typ, field) in tol_map:
        return tol_map[(typ, field)]
    return _infer_tol(field, gold, policy)


def score_item_fields(
    it: Dict[str, Any],
    parsed_json: Optional[Dict[str, Any]],
    provider: str,
    model: str,
    policy: str,
    latency_s: float,
    error: Optional[str],
) -> List[Dict[str, Any]]:
    typ = get_item_type(it)
    iid = get_item_id(it)
    gt = get_ground_truth(it)

    fields = [f for f in expected_fields(it) if f in gt]
    tol_map = tolerances_plotread() if policy == "plotread" else tolerances_strict()

    rows: List[Dict[str, Any]] = []
    for field in fields:
        gold = _to_float(gt.get(field))
        pred = _to_float(parsed_json.get(field)) if isinstance(parsed_json, dict) else None

        ae = abs_err(pred, gold) if gold is not None else float("nan")
        re_ = rel_err(pred, gold) if gold is not None else float("nan")

        abs_tol, rel_tol = (float("nan"), float("nan"))
        passed = False
        if gold is not None:
            abs_tol, rel_tol = get_tol(tol_map, policy, typ, field, gold)
            if pred is not None and not (math.isnan(ae) or math.isnan(re_)):
                passed = (ae <= abs_tol) or (re_ <= rel_tol)

        rows.append(
            {
                "provider": provider,
                "model": model,
                "id": iid,
                "type": typ,
                "field": field,
                "is_checkpoint": is_checkpoint_field(field),
                "pred": pred,
                "gold": gold,
                "abs_err": ae,
                "rel_err": re_,
                "abs_tol": abs_tol,
                "rel_tol": rel_tol,
                "pass": bool(passed),
                "latency_s": float(latency_s),
                "error": error,
            }
        )
    return rows


# -------------------------
# Model adapters
# -------------------------

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str


def parse_models(models: List[str]) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for s in models:
        if ":" not in s:
            raise ValueError(f"Model must be provider:model, got: {s}")
        p, m = s.split(":", 1)
        out.append(ModelSpec(p.strip().lower(), m.strip()))
    return out


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode("utf-8")


def openai_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()

    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)
    data_url = f"data:{mime};base64,{b64}"

    resp = client.responses.create(
        model=model,
        input=[{
            "role": "user",
            "content": [
                {"type": "input_image", "image_url": data_url},
                {"type": "input_text", "text": prompt},
            ],
        }],
        max_output_tokens=max_output_tokens,
        temperature=0,  # Deterministic for reproducibility
    )
    return getattr(resp, "output_text", "") or ""


def gemini_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    image_bytes = image_path.read_bytes()
    mime = guess_mime(image_path)

    # Gemini requires 8192 tokens for temperature=0 to work properly
    # Other models can use lower values, but Gemini needs the full allocation
    gemini_max_tokens = 8192

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
        config=types.GenerateContentConfig(
            temperature=0,  # Deterministic for reproducibility
            max_output_tokens=gemini_max_tokens,  # Gemini-specific: needs 8192 for temperature=0
        ),
    )
    
    # DEBUG: Log full response structure to file
    debug_log_path = Path("gemini_api_responses_debug.jsonl")
    try:
        resp_debug = {
            "has_text_attr": hasattr(resp, 'text'),
            "text_value": getattr(resp, 'text', None),
            "has_candidates": hasattr(resp, 'candidates'),
            "candidates_count": len(resp.candidates) if hasattr(resp, 'candidates') and resp.candidates else 0,
            "response_type": str(type(resp)),
            "response_dir": [x for x in dir(resp) if not x.startswith('_')][:20],  # First 20 attributes
        }
        
        # Try to extract candidate details
        if hasattr(resp, 'candidates') and resp.candidates:
            candidates_debug = []
            for i, candidate in enumerate(resp.candidates):
                cand_info = {
                    "index": i,
                    "type": str(type(candidate)),
                    "has_content": hasattr(candidate, 'content'),
                    "has_finish_reason": hasattr(candidate, 'finish_reason'),
                    "finish_reason": getattr(candidate, 'finish_reason', None),
                }
                if hasattr(candidate, 'content') and candidate.content:
                    cand_info["content_type"] = str(type(candidate.content))
                    cand_info["has_parts"] = hasattr(candidate.content, 'parts')
                    if hasattr(candidate.content, 'parts'):
                        parts_info = []
                        for j, part in enumerate(candidate.content.parts):
                            part_info = {
                                "index": j,
                                "type": str(type(part)),
                                "has_text": hasattr(part, 'text'),
                                "text_length": len(part.text) if hasattr(part, 'text') and part.text else 0,
                            }
                            parts_info.append(part_info)
                        cand_info["parts"] = parts_info
                    elif hasattr(candidate.content, 'text'):
                        cand_info["content_text_length"] = len(candidate.content.text) if candidate.content.text else 0
                candidates_debug.append(cand_info)
            resp_debug["candidates_detail"] = candidates_debug
        
        # Write debug info (append mode)
        with debug_log_path.open("a", encoding="utf-8") as f:
            f.write(json.dumps(resp_debug, default=str) + "\n")
    except Exception as e:
        # Don't fail if logging fails
        pass
    
    # Try multiple ways to extract text from Gemini API response
    text = ""
    
    # Method 1: Direct text attribute (most common)
    if hasattr(resp, 'text') and resp.text:
        text = resp.text
    # Method 2: Candidates -> content -> parts (fallback)
    elif hasattr(resp, 'candidates') and resp.candidates:
        for candidate in resp.candidates:
            if hasattr(candidate, 'content') and candidate.content:
                if hasattr(candidate.content, 'parts') and candidate.content.parts is not None:
                    for part in candidate.content.parts:
                        if hasattr(part, 'text') and part.text:
                            text += part.text
                elif hasattr(candidate.content, 'text'):
                    text += candidate.content.text
                # Also try direct text on content
                if hasattr(candidate.content, 'text'):
                    text += candidate.content.text
    
    return text.strip()


def anthropic_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)

    msg = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        temperature=0,  # Deterministic for reproducibility
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    out: List[str] = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            out.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "\n".join(out).strip()


def call_model(spec: ModelSpec, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    if spec.provider == "openai":
        return openai_call(spec.model, image_path, prompt, max_output_tokens)
    if spec.provider == "gemini":
        return gemini_call(spec.model, image_path, prompt, max_output_tokens)
    if spec.provider == "anthropic":
        return anthropic_call(spec.model, image_path, prompt, max_output_tokens)
    raise ValueError(f"Unknown provider: {spec.provider}")


# -------------------------
# Reporting
# -------------------------

def write_reports(df: pd.DataFrame, out_dir: Path) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    per_item_csv = out_dir / "per_item.csv"
    df.to_csv(per_item_csv, index=False)

    df2 = df.copy()
    if "is_checkpoint" not in df2.columns:
        df2["is_checkpoint"] = df2["field"].astype(str).str.startswith("cp_")

    final_mask = ~df2["is_checkpoint"]
    cp_mask = df2["is_checkpoint"]

    item_level = (
        df2.groupby(["provider", "model", "id", "type"])
        .agg(
            n_fields=("pass", "size"),
            final_n=("pass", lambda s: int(final_mask.loc[s.index].sum())),
            checkpoint_n=("pass", lambda s: int(cp_mask.loc[s.index].sum())),
            all_final_pass=("pass", lambda s: bool((s[final_mask.loc[s.index]]).all()) if final_mask.loc[s.index].any() else True),
            all_checkpoint_pass=("pass", lambda s: bool((s[cp_mask.loc[s.index]]).all()) if cp_mask.loc[s.index].any() else True),
        )
        .reset_index()
    )
    item_level_csv = out_dir / "item_level.csv"
    item_level.to_csv(item_level_csv, index=False)

    df2["scope"] = np.where(df2["is_checkpoint"], "checkpoint", "final")
    summary = (
        df2.groupby(["provider", "model", "type", "scope", "field"])
        .agg(
            n=("pass", "size"),
            pass_rate=("pass", "mean"),
            mean_abs_err=("abs_err", "mean"),
            median_abs_err=("abs_err", "median"),
            mean_rel_err=("rel_err", "mean"),
            p95_abs_err=("abs_err", lambda s: float(np.nanpercentile(pd.to_numeric(s, errors="coerce").dropna(), 95)) if pd.to_numeric(s, errors="coerce").dropna().size else np.nan),
            mean_latency_s=("latency_s", "mean"),
        )
        .reset_index()
        .sort_values(["provider", "model", "type", "scope", "field"])
    )
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    overall_all = df2.groupby(["provider", "model"]).agg(
        n=("pass", "size"),
        overall_pass_rate=("pass", "mean"),
        mean_abs_err=("abs_err", "mean"),
        mean_rel_err=("rel_err", "mean"),
        mean_latency_s=("latency_s", "mean"),
    )

    overall_final = df2[~df2["is_checkpoint"]].groupby(["provider", "model"]).agg(
        final_n=("pass", "size"),
        final_pass_rate=("pass", "mean"),
    )

    overall_cp = df2[df2["is_checkpoint"]].groupby(["provider", "model"]).agg(
        checkpoint_n=("pass", "size"),
        checkpoint_pass_rate=("pass", "mean"),
    )

    overall = (
        overall_all.join(overall_final, how="left")
        .join(overall_cp, how="left")
        .reset_index()
        .sort_values(["overall_pass_rate", "final_pass_rate", "mean_abs_err"], ascending=[False, False, True])
    )
    overall_csv = out_dir / "overall.csv"
    overall.to_csv(overall_csv, index=False)

    print(f"[write] {per_item_csv}")
    print(f"[write] {item_level_csv}")
    print(f"[write] {summary_csv}")
    print(f"[write] {overall_csv}")


# -------------------------
# Modes
# -------------------------

def run_mode_run(
    jsonl_path: Path,
    images_root: Optional[Path],
    specs: List[ModelSpec],
    out_dir: Path,
    policy: str,
    limit: int,
    max_output_tokens: int,
    seed: int,
    overwrite: bool,
    sleep_s: float,
    concurrent: Optional[int] = None,
    sequential: bool = False,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(jsonl_path)
    if limit and limit > 0:
        items = items[:limit]

    # Deterministic shuffle
    if seed:
        rng = np.random.default_rng(seed)
        idx = list(range(len(items)))
        rng.shuffle(idx)
        items = [items[i] for i in idx]

    all_rows: List[Dict[str, Any]] = []

    for spec in specs:
        base_name = f"raw_{spec.provider}_{spec.model.replace('/','_')}.jsonl"
        raw_path = out_dir / base_name

        # Preserve every run (unless --overwrite)
        if raw_path.exists() and not overwrite:
            ts = time.strftime("%Y%m%d_%H%M%S")
            raw_path = out_dir / f"raw_{spec.provider}_{spec.model.replace('/','_')}_{ts}_{uuid.uuid4().hex[:6]}.jsonl"

        print(f"[run] writing raw: {raw_path}")

        # Use concurrent evaluation unless --sequential is specified
        if not sequential:
            try:
                all_rows.extend(
                    run_concurrent_evaluation(
                        items=items,
                        spec=spec,
                        images_root=images_root,
                        jsonl_path=jsonl_path,
                        max_output_tokens=max_output_tokens,
                        call_model_fn=call_model,
                        build_prompt_fn=build_prompt,
                        resolve_image_path_fn=resolve_image_path,
                        get_item_id_fn=get_item_id,
                        get_item_type_fn=get_item_type,
                        extract_first_json_fn=extract_first_json,
                        score_item_fields_fn=score_item_fields,
                        policy=policy,
                        raw_file_path=raw_path,
                        max_workers=concurrent,
                    )
                )
                continue  # Skip sequential processing
            except Exception as e:
                print(f"[warning] Concurrent evaluation failed: {e}, falling back to sequential")
                # Fall through to sequential processing
        
        # Sequential processing (original code)
        with raw_path.open("w", encoding="utf-8") as raw_f:
            for it in tqdm(items, desc=f"{spec.provider}:{spec.model}"):
                iid = get_item_id(it)
                typ = get_item_type(it)
                img_path = resolve_image_path(it, images_root, jsonl_path)
                prompt = build_prompt(it)

                t0 = time.time()
                err: Optional[str] = None
                txt = ""
                try:
                    if not img_path.exists():
                        raise FileNotFoundError(f"Image not found: {img_path}")
                    txt = call_model(spec, img_path, prompt, max_output_tokens=max_output_tokens)
                except Exception as e:
                    err = f"{type(e).__name__}: {e}"
                dt = time.time() - t0

                parsed = extract_first_json(txt) if txt else None

                rows = score_item_fields(
                    it=it,
                    parsed_json=parsed,
                    provider=spec.provider,
                    model=spec.model,
                    policy=policy,
                    latency_s=dt,
                    error=err,
                )
                all_rows.extend(rows)

                raw_f.write(
                    json.dumps(
                        {
                            "provider": spec.provider,
                            "model": spec.model,
                            "id": iid,
                            "type": typ,
                            "image_path": str(img_path),
                            "prompt": prompt,
                            "raw_text": txt,
                            "parsed_json": parsed,
                            "latency_s": dt,
                            "error": err,
                            "raw_text_length": len(txt) if txt else 0,  # Debug: track length
                        },
                        ensure_ascii=False,
                    )
                    + "\n"
                )

                if sleep_s and sleep_s > 0:
                    time.sleep(float(sleep_s))

    if not all_rows:
        print("[run] No rows produced.")
        return

    df = pd.DataFrame(all_rows)
    write_reports(df, out_dir)


def run_mode_score(
    out_dir: Path,
    raw_glob: str,
    jsonl_path: Path,
    images_root: Optional[Path],
    policy: str,
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)

    raw_paths = sorted(out_dir.glob(raw_glob))
    if not raw_paths:
        raise FileNotFoundError(f"No raw files matched {raw_glob} in {out_dir}")

    items_by_id: Dict[str, Dict[str, Any]] = {get_item_id(it): it for it in load_jsonl(jsonl_path)}

    all_rows: List[Dict[str, Any]] = []
    for rp in raw_paths:
        print(f"[score] reading: {rp}")
        for rec in iter_jsonl(rp):
            iid = str(rec.get("id", ""))
            provider = str(rec.get("provider", ""))
            model = str(rec.get("model", ""))
            parsed = rec.get("parsed_json")
            latency_s = float(rec.get("latency_s", float("nan")))
            error = rec.get("error")

            it = items_by_id.get(iid)
            if it is None:
                raise RuntimeError(f"Item id not found in dataset jsonl: {iid}")

            rows = score_item_fields(
                it=it,
                parsed_json=parsed if isinstance(parsed, dict) else None,
                provider=provider,
                model=model,
                policy=policy,
                latency_s=latency_s,
                error=error,
            )
            all_rows.extend(rows)

    df = pd.DataFrame(all_rows)
    write_reports(df, out_dir)


def main() -> None:
    ap = argparse.ArgumentParser()

    ap.add_argument("--jsonl", type=str, required=True, help="Path to plotchain.jsonl (or subset jsonl)")
    ap.add_argument("--images_root", type=str, default="", help="Root directory for relative image paths (default: jsonl parent)")
    ap.add_argument("--out_dir", type=str, default="results", help="Output directory")

    ap.add_argument("--mode", type=str, default="run", choices=["run", "score", "eval"], help="run=model calls; score=rescore raw files")
    ap.add_argument("--policy", type=str, default="plotread", choices=["plotread", "strict"], help="Tolerance policy")
    ap.add_argument("--tol_policy", type=str, default=None, help="Alias for --policy (plotread|strict)")

    ap.add_argument("--raw_glob", type=str, default="raw_*.jsonl", help="Glob for raw files when scoring")
    ap.add_argument("--models", type=str, nargs="*", default=[], help="List like: openai:gpt-4.1 gemini:... anthropic:...")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of items")
    ap.add_argument("--max_output_tokens", type=int, default=2000, help="Max tokens for model output (default: 2000; Gemini automatically uses 8192)")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic shuffle seed (0 = no shuffle)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite raw_<provider>_<model>.jsonl if it exists")
    ap.add_argument("--sleep_s", type=float, default=0.0, help="Sleep between model calls (seconds)")
    ap.add_argument("--concurrent", type=int, default=None, help="Number of concurrent requests (None = use provider default)")
    ap.add_argument("--sequential", action="store_true", help="Use sequential processing instead of concurrent")

    args = ap.parse_args()

    mode = args.mode
    if mode == "eval":
        mode = "run"

    policy = args.policy
    if args.tol_policy:
        policy = str(args.tol_policy).strip().lower()

    jsonl_path = Path(args.jsonl)
    if not jsonl_path.exists():
        raise FileNotFoundError(f"--jsonl not found: {jsonl_path}")

    images_root = Path(args.images_root).resolve() if args.images_root else None
    out_dir = Path(args.out_dir)

    if mode == "run":
        if not args.models:
            raise ValueError("--models is required in --mode run")
        specs = parse_models(args.models)
        run_mode_run(
            jsonl_path=jsonl_path,
            images_root=images_root,
            specs=specs,
            out_dir=out_dir,
            policy=policy,
            limit=args.limit,
            max_output_tokens=args.max_output_tokens,
            seed=args.seed,
            overwrite=args.overwrite,
            sleep_s=args.sleep_s,
            concurrent=args.concurrent,
            sequential=args.sequential,
        )
    else:
        run_mode_score(
            out_dir=out_dir,
            raw_glob=args.raw_glob,
            jsonl_path=jsonl_path,
            images_root=images_root,
            policy=policy,
        )


if __name__ == "__main__":
    main()
