#!/usr/bin/env python3
"""
run_plotchain_eval.py

Run multimodal LLMs on PlotChain (image + question) and score vs ground truth.

Modes:
  --mode run   : call LLMs, write raw_<provider>_<model>.jsonl, and also write scores (per_item/summary/overall)
  --mode score : DO NOT call LLMs; recompute scores from existing raw_*.jsonl (lets you change tolerances freely)

Writes (in out_dir):
  raw_<provider>_<model>.jsonl   (full responses for audit)
  per_item.csv
  summary.csv
  overall.csv
"""

import argparse
import base64
import json
import math
import os
import re
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from PIL import Image  # ok if unused; kept to avoid unnecessary churn
from tqdm import tqdm

# ---------- Schema helpers (edit only if needed) ----------

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
    gt = it.get("ground_truth", None)
    if gt is None:
        raise KeyError("Missing ground_truth")
    return dict(gt)

def get_image_path(it: Dict[str, Any], images_root: Optional[Path]) -> Path:
    for k in ("image_path", "image_file", "plot_path", "image"):
        if k in it:
            p = Path(it[k])
            return (images_root / p).resolve() if images_root and not p.is_absolute() else p.resolve()
    raise KeyError("Could not find image path field (expected one of image_path/image_file/plot_path/image).")

def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items

def guess_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in (".jpg", ".jpeg"):
        return "image/jpeg"
    if ext == ".webp":
        return "image/webp"
    if ext == ".gif":
        return "image/gif"
    return "image/png"

# ---------- Output schema (what we ask models to return) ----------

FIELDS_BY_TYPE: Dict[str, List[str]] = {
    "step_response": ["percent_overshoot", "settling_time_s", "steady_state"],
    "bode_magnitude": ["dc_gain_db", "cutoff_hz"],
    "fft_spectrum": ["dominant_frequency_hz"],
    "time_waveform": ["frequency_hz", "vpp_v"],
    "iv_curve": ["resistance_ohm", "turn_on_voltage_v_at_target_i"],  # scorer will pick relevant keys present in gt
    "transfer_characteristic": ["small_signal_gain", "saturation_v"],
}

UNITS_HINT = {
    "percent_overshoot": "%",
    "settling_time_s": "s",
    "steady_state": "unitless",
    "dc_gain_db": "dB",
    "cutoff_hz": "Hz",
    "dominant_frequency_hz": "Hz",
    "frequency_hz": "Hz",
    "vpp_v": "V",
    "resistance_ohm": "Ohm",
    "turn_on_voltage_v_at_target_i": "V",
    "small_signal_gain": "unitless",
    "saturation_v": "V",
}

def build_json_schema_hint(fields: List[str]) -> str:
    lines = []
    for f in fields:
        lines.append(f'  "{f}": <number>  // {UNITS_HINT.get(f,"")}')
    return "{\n" + ",\n".join(lines) + "\n}"

def build_prompt(it: Dict[str, Any]) -> str:
    typ = get_item_type(it)
    gt = get_ground_truth(it)

    # Only request fields that are actually in this item's ground truth (keeps it tight + deterministic)
    candidate = FIELDS_BY_TYPE.get(typ, list(gt.keys()))
    fields = [k for k in candidate if k in gt]

    schema = build_json_schema_hint(fields)

    return f"""
You are given an engineering plot image. Answer the question by reading the plot.

Question:
{get_question(it)}

Return ONLY a single JSON object matching this schema (numbers only, no strings, no units):
{schema}

If a value cannot be determined, still output the key with value null.
""".strip()

# ---------- Robust JSON extraction ----------

_JSON_CANDIDATE_RE = re.compile(r"\{.*\}", re.DOTALL)

def extract_first_json(text: str) -> Optional[Dict[str, Any]]:
    if not text:
        return None

    try:
        obj = json.loads(text)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    m = _JSON_CANDIDATE_RE.search(text)
    if not m:
        return None
    blob = m.group(0)

    try:
        obj = json.loads(blob)
        return obj if isinstance(obj, dict) else None
    except Exception:
        pass

    for end in range(len(blob), 1, -1):
        try:
            obj = json.loads(blob[:end])
            if isinstance(obj, dict):
                return obj
        except Exception:
            continue
    return None

# ---------- Scoring ----------

def _to_float(x: Any) -> Optional[float]:
    if x is None:
        return None
    if isinstance(x, (int, float)) and not (isinstance(x, float) and math.isnan(x)):
        return float(x)
    if isinstance(x, str):
        s = x.strip()
        try:
            return float(s)
        except Exception:
            return None
    return None

def abs_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(abs(a - b))

def rel_err(a: Optional[float], b: Optional[float]) -> Optional[float]:
    if a is None or b is None:
        return None
    return float(abs(a - b) / max(abs(b), 1e-12))

def tolerances_strict():
    return {
        ("step_response", "percent_overshoot"): (2.0, 0.05),
        ("step_response", "settling_time_s"):   (0.20, 0.10),
        ("step_response", "steady_state"):      (0.03, 0.02),

        ("bode_magnitude", "dc_gain_db"):       (0.25, 0.02),
        ("bode_magnitude", "cutoff_hz"):        (2.0, 0.03),

        ("fft_spectrum", "dominant_frequency_hz"): (1.0, 0.02),

        ("time_waveform", "frequency_hz"):      (0.5, 0.01),
        ("time_waveform", "vpp_v"):             (0.12, 0.07),

        ("iv_curve", "resistance_ohm"):         (5.0, 0.03),
        ("iv_curve", "turn_on_voltage_v_at_target_i"): (0.05, 0.10),

        ("transfer_characteristic", "small_signal_gain"): (0.12, 0.06),
        ("transfer_characteristic", "saturation_v"):       (0.06, 0.06),
    }

def tolerances_plotread():
    return {
        ("step_response", "percent_overshoot"): (2.5, 0.07),
        ("step_response", "settling_time_s"):   (0.35, 0.20),
        ("step_response", "steady_state"):      (0.05, 0.02),

        ("bode_magnitude", "dc_gain_db"):       (0.5, 0.03),
        ("bode_magnitude", "cutoff_hz"):        (5.0, 0.05),

        ("fft_spectrum", "dominant_frequency_hz"): (2.0, 0.03),

        ("time_waveform", "frequency_hz"):      (1.0, 0.02),
        ("time_waveform", "vpp_v"):             (0.20, 0.10),

        ("iv_curve", "resistance_ohm"):         (10.0, 0.05),
        ("iv_curve", "turn_on_voltage_v_at_target_i"): (0.08, 0.15),

        ("transfer_characteristic", "small_signal_gain"): (0.20, 0.10),
        ("transfer_characteristic", "saturation_v"):       (0.10, 0.10),
    }

def get_tolerances(policy: str) -> Dict[Tuple[str, str], Tuple[float, float]]:
    return tolerances_strict() if policy == "strict" else tolerances_plotread()

def failure_tags(pred: Optional[float], gold: Optional[float]) -> List[str]:
    tags = []
    if pred is None:
        tags.append("missing_value")
        return tags
    if gold is None:
        tags.append("missing_gold")
        return tags
    if gold != 0:
        ratio = pred / gold
        if abs(ratio - 1000) < 50 or abs(ratio - 0.001) < 0.0001:
            tags.append("likely_unit_scale_error_x1000")
        if abs(ratio - 2 * math.pi) < 0.25 or abs(ratio - 1 / (2 * math.pi)) < 0.05:
            tags.append("hz_vs_rad_s")
    return tags

# ---------- Model adapters (OpenAI / Gemini / Anthropic) ----------

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str

def encode_image_base64(image_path: Path) -> str:
    with image_path.open("rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")

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
    )
    return getattr(resp, "output_text", "") or ""

def gemini_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    from google import genai
    from google.genai import types

    client = genai.Client()
    image_bytes = image_path.read_bytes()
    mime = guess_mime(image_path)

    resp = client.models.generate_content(
        model=model,
        contents=[
            types.Part.from_bytes(data=image_bytes, mime_type=mime),
            prompt,
        ],
    )
    return getattr(resp, "text", "") or ""

def anthropic_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    import anthropic

    api_key = os.environ.get("ANTHROPIC_API_KEY", "")
    client = anthropic.Anthropic(api_key=api_key)

    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)

    msg = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        messages=[{
            "role": "user",
            "content": [
                {"type": "image", "source": {"type": "base64", "media_type": mime, "data": b64}},
                {"type": "text", "text": prompt},
            ],
        }],
    )

    out = []
    for block in getattr(msg, "content", []) or []:
        if getattr(block, "type", None) == "text":
            out.append(getattr(block, "text", ""))
        elif isinstance(block, dict) and block.get("type") == "text":
            out.append(block.get("text", ""))
    return "\n".join(out).strip()

def call_model(spec: ModelSpec, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    if spec.provider == "openai":
        return openai_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    if spec.provider == "gemini":
        return gemini_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    if spec.provider == "anthropic":
        return anthropic_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    raise ValueError(f"Unknown provider: {spec.provider}")

# ---------- Scoring from raw (no LLM calls) ----------

def score_from_raw(items: List[Dict[str, Any]],
                   raw_paths: List[Path],
                   tols: Dict[Tuple[str, str], Tuple[float, float]],
                   out_dir: Path) -> None:
    ds_map = {get_item_id(it): it for it in items}
    rows: List[Dict[str, Any]] = []

    for rp in raw_paths:
        with rp.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                r = json.loads(line)

                iid = str(r.get("id", ""))
                it = ds_map.get(iid)
                if it is None:
                    continue

                typ = str(r.get("type", get_item_type(it)))
                gt = get_ground_truth(it)
                parsed = r.get("parsed_json", None)

                candidate = FIELDS_BY_TYPE.get(typ, list(gt.keys()))
                keys = [k for k in candidate if k in gt]

                for k in keys:
                    gold = _to_float(gt.get(k))
                    pred = _to_float(parsed.get(k)) if isinstance(parsed, dict) else None

                    ae = abs_err(pred, gold)
                    re_ = rel_err(pred, gold)

                    abs_tol, rel_tol = tols.get((typ, k), (0.0, 0.0))
                    passed = False
                    if ae is not None and re_ is not None:
                        passed = (ae <= abs_tol) or (re_ <= rel_tol)

                    tags = failure_tags(pred, gold)

                    rows.append({
                        "provider": r.get("provider", ""),
                        "model": r.get("model", ""),
                        "id": iid,
                        "type": typ,
                        "field": k,
                        "pred": pred,
                        "gold": gold,
                        "abs_err": ae,
                        "rel_err": re_,
                        "abs_tol": abs_tol,
                        "rel_tol": rel_tol,
                        "pass": bool(passed),
                        "latency_s": r.get("latency_s", None),
                        "error": r.get("error", None),
                        "tags": ",".join(tags),
                    })

    if not rows:
        print("No rows scored (no raw files, or ids did not match dataset).")
        return

    df = pd.DataFrame(rows)
    per_item_csv = out_dir / "per_item.csv"
    df.to_csv(per_item_csv, index=False)

    summary = (
        df.groupby(["provider", "model", "type", "field"])
          .agg(
              n=("pass", "size"),
              pass_rate=("pass", "mean"),
              mean_abs_err=("abs_err", "mean"),
              median_abs_err=("abs_err", "median"),
              mean_rel_err=("rel_err", "mean"),
              p95_abs_err=("abs_err", lambda s: float(np.nanpercentile(s.dropna(), 95)) if s.dropna().size else np.nan),
              mean_latency_s=("latency_s", "mean"),
          )
          .reset_index()
          .sort_values(["provider", "model", "type", "field"])
    )
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    overall = (
        df.groupby(["provider", "model"])
          .agg(
              n=("pass", "size"),
              overall_pass_rate=("pass", "mean"),
              mean_abs_err=("abs_err", "mean"),
              mean_rel_err=("rel_err", "mean"),
              mean_latency_s=("latency_s", "mean"),
          )
          .reset_index()
          .sort_values(["overall_pass_rate", "mean_abs_err"], ascending=[False, True])
    )
    overall_csv = out_dir / "overall.csv"
    overall.to_csv(overall_csv, index=False)

    print(f"\nWrote:\n- {per_item_csv}\n- {summary_csv}\n- {overall_csv}\n")

# ---------- Main ----------

def parse_models(args_models: List[str]) -> List[ModelSpec]:
    out = []
    for s in args_models:
        if ":" not in s:
            raise ValueError(f"Model must be provider:model, got: {s}")
        p, m = s.split(":", 1)
        out.append(ModelSpec(provider=p.strip().lower(), model=m.strip()))
    return out

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", choices=["run", "score"], default="run",
                    help="run=call LLMs; score=recompute metrics from existing raw_*.jsonl (no LLM calls)")
    ap.add_argument("--policy", choices=["plotread", "strict"], default="plotread",
                    help="Tolerance policy used for scoring")
    ap.add_argument("--raw_glob", type=str, default="raw_*.jsonl",
                    help="Pattern in out_dir for finding raw files in --mode score")

    ap.add_argument("--jsonl", type=str, required=True, help="Path to plotchain_v2.jsonl")
    ap.add_argument("--images_root", type=str, default="", help="Optional root directory for relative image paths")

    # NOTE: models are only required for --mode run
    ap.add_argument("--models", type=str, nargs="*", default=[],
                    help='List like: openai:gpt-4.1 gemini:gemini-2.5-flash anthropic:claude-sonnet-4-5')

    ap.add_argument("--out_dir", type=str, default="results_v2", help="Output directory")
    ap.add_argument("--limit", type=int, default=0, help="Optional: limit number of items (run mode only)")
    ap.add_argument("--max_output_tokens", type=int, default=400, help="Max tokens for model output (run mode only)")
    ap.add_argument("--seed", type=int, default=0, help="Deterministic ordering only (no effect on APIs)")
    ap.add_argument("--overwrite", action="store_true", help="Overwrite existing raw files (run mode only)")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    images_root = Path(args.images_root).resolve() if args.images_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(jsonl_path)
    if args.mode == "run" and args.limit and args.limit > 0:
        items = items[:args.limit]

    tols = get_tolerances(args.policy)

    if args.mode == "score":
        raw_paths = sorted(out_dir.glob(args.raw_glob))
        if not raw_paths:
            print(f"No raw files found in {out_dir} matching '{args.raw_glob}'.")
            return
        score_from_raw(items, raw_paths, tols, out_dir)
        return

    # ---- run mode ----
    if not args.models:
        raise SystemExit("ERROR: --models is required in --mode run.")

    specs = parse_models(args.models)

    # We still compute per-item scores during run (for immediate feedback),
    # but scoring can always be recomputed later from raw files via --mode score.
    all_rows = []
    for spec in specs:
        raw_path = out_dir / f"raw_{spec.provider}_{spec.model.replace('/','_')}.jsonl"
        if raw_path.exists() and not args.overwrite:
            print(f"[skip] raw exists: {raw_path} (use --overwrite to regenerate)")
            continue

        with raw_path.open("w", encoding="utf-8") as raw_f:
            for it in tqdm(items, desc=f"{spec.provider}:{spec.model}"):
                iid = get_item_id(it)
                typ = get_item_type(it)
                gt = get_ground_truth(it)
                img_path = get_image_path(it, images_root)
                prompt = build_prompt(it)

                t0 = time.time()
                try:
                    txt = call_model(spec, img_path, prompt, max_output_tokens=args.max_output_tokens)
                    err = None
                except Exception as e:
                    txt = ""
                    err = f"{type(e).__name__}: {e}"
                dt = time.time() - t0

                parsed = extract_first_json(txt) if txt else None

                candidate = FIELDS_BY_TYPE.get(typ, list(gt.keys()))
                keys = [k for k in candidate if k in gt]

                for k in keys:
                    gold = _to_float(gt.get(k))
                    pred = _to_float(parsed.get(k)) if isinstance(parsed, dict) else None

                    ae = abs_err(pred, gold)
                    re_ = rel_err(pred, gold)

                    abs_tol, rel_tol = tols.get((typ, k), (0.0, 0.0))
                    passed = False
                    if ae is not None and re_ is not None:
                        passed = (ae <= abs_tol) or (re_ <= rel_tol)

                    tags = failure_tags(pred, gold)

                    all_rows.append({
                        "provider": spec.provider,
                        "model": spec.model,
                        "id": iid,
                        "type": typ,
                        "field": k,
                        "pred": pred,
                        "gold": gold,
                        "abs_err": ae,
                        "rel_err": re_,
                        "abs_tol": abs_tol,
                        "rel_tol": rel_tol,
                        "pass": bool(passed),
                        "latency_s": dt,
                        "error": err,
                        "tags": ",".join(tags),
                    })

                raw_f.write(json.dumps({
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
                }, ensure_ascii=False) + "\n")

    if not all_rows:
        print("No results produced (maybe everything was skipped due to existing raw files).")
        print("Tip: use --mode score to recompute metrics from existing raw_*.jsonl.")
        return

    df = pd.DataFrame(all_rows)
    per_item_csv = out_dir / "per_item.csv"
    df.to_csv(per_item_csv, index=False)

    summary = (
        df.groupby(["provider", "model", "type", "field"])
          .agg(
              n=("pass", "size"),
              pass_rate=("pass", "mean"),
              mean_abs_err=("abs_err", "mean"),
              median_abs_err=("abs_err", "median"),
              mean_rel_err=("rel_err", "mean"),
              p95_abs_err=("abs_err", lambda s: float(np.nanpercentile(s.dropna(), 95)) if s.dropna().size else np.nan),
              mean_latency_s=("latency_s", "mean"),
          )
          .reset_index()
          .sort_values(["provider", "model", "type", "field"])
    )
    summary_csv = out_dir / "summary.csv"
    summary.to_csv(summary_csv, index=False)

    overall = (
        df.groupby(["provider", "model"])
          .agg(
              n=("pass", "size"),
              overall_pass_rate=("pass", "mean"),
              mean_abs_err=("abs_err", "mean"),
              mean_rel_err=("rel_err", "mean"),
              mean_latency_s=("latency_s", "mean"),
          )
          .reset_index()
          .sort_values(["overall_pass_rate", "mean_abs_err"], ascending=[False, True])
    )
    overall_csv = out_dir / "overall.csv"
    overall.to_csv(overall_csv, index=False)

    print(f"\nWrote:\n- {per_item_csv}\n- {summary_csv}\n- {overall_csv}\n")
    print("You can rescore later without LLM calls via:")
    print(f"  python3 run_plotchain_eval.py --mode score --policy {args.policy} --jsonl {args.jsonl} --out_dir {args.out_dir}")

if __name__ == "__main__":
    main()
