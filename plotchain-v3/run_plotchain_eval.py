#!/usr/bin/env python3
\"\"\"run_plotchain_eval.py

Run multimodal LLMs on PlotChain v3 (image + question) and score vs ground truth.

Writes (in out_dir):
  - per_item.csv
  - summary.csv
  - overall.csv
  - raw_<provider>_<model>.jsonl  (full responses for audit)

Notes:
- Requests STRUCTURED \"reasoning checkpoints\" (cp_*) as additional numeric fields.
- Disable checkpoint prompting with --final_only.
\"\"\"

from __future__ import annotations

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
from tqdm import tqdm


def load_jsonl(path: Path) -> List[Dict[str, Any]]:
    items = []
    with path.open(\"r\", encoding=\"utf-8\") as f:
        for line in f:
            line = line.strip()
            if line:
                items.append(json.loads(line))
    return items


def get_image_path(it: Dict[str, Any], images_root: Optional[Path]) -> Path:
    p = Path(it[\"image_path\"])
    return (images_root / p).resolve() if images_root and not p.is_absolute() else p.resolve()


def guess_mime(image_path: Path) -> str:
    ext = image_path.suffix.lower()
    if ext in (\".jpg\", \".jpeg\"):
        return \"image/jpeg\"
    if ext == \".webp\":
        return \"image/webp\"
    if ext == \".gif\":
        return \"image/gif\"
    return \"image/png\"


def build_schema_hint(keys: List[str]) -> str:
    lines = [f'  \"{k}\": <number or null>' for k in keys]
    return \"{\\n\" + \",\\n\".join(lines) + \"\\n}\"


def build_prompt(it: Dict[str, Any], final_only: bool) -> str:
    gen = it.get(\"generation\", {}) or {}
    finals = list(gen.get(\"final_fields\", []))
    cps = list(gen.get(\"checkpoint_fields\", []))
    keys = finals[:] if final_only else (finals + cps)

    schema = build_schema_hint(keys)
    q = it[\"question\"]

    return f\"\"\"You are given an engineering plot image. Answer the question by reading the plot.

Question:
{q}

Return ONLY a single JSON object matching this schema (numbers only; no strings; no units):
{schema}

Rules:
- Use null if a value cannot be determined from the plot.
- Do NOT include any extra keys.
\"\"\".strip()


_JSON_CANDIDATE_RE = re.compile(r\"\\{.*\\}\", re.DOTALL)


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


def tolerances_plotread() -> Dict[Tuple[str, str], Tuple[float, float]]:
    return {
        (\"step_response\", \"percent_overshoot\"): (2.5, 0.07),
        (\"step_response\", \"settling_time_s\"):   (0.35, 0.20),
        (\"step_response\", \"steady_state\"):      (0.05, 0.02),
        (\"step_response\", \"cp_peak_value\"):     (0.08, 0.04),
        (\"step_response\", \"cp_peak_time_s\"):    (0.20, 0.20),
        (\"step_response\", \"cp_band_upper\"):     (0.05, 0.02),
        (\"step_response\", \"cp_band_lower\"):     (0.05, 0.02),

        (\"bode_magnitude\", \"dc_gain_db\"):       (0.50, 0.03),
        (\"bode_magnitude\", \"cutoff_hz\"):        (5.0, 0.05),
        (\"bode_magnitude\", \"cp_mag_at_fc_db\"):  (0.60, 0.04),
        (\"bode_magnitude\", \"cp_slope_db_per_decade\"): (4.0, 0.25),

        (\"bode_phase\", \"cutoff_hz\"):            (6.0, 0.06),
        (\"bode_phase\", \"phase_deg_at_10fc\"):    (4.0, 0.08),
        (\"bode_phase\", \"cp_phase_deg_at_fc\"):   (3.0, 0.07),

        (\"bandpass_response\", \"resonance_hz\"):  (12.0, 0.05),
        (\"bandpass_response\", \"bandwidth_hz\"):  (18.0, 0.10),
        (\"bandpass_response\", \"cp_f1_3db_hz\"):  (18.0, 0.10),
        (\"bandpass_response\", \"cp_f2_3db_hz\"):  (18.0, 0.10),
        (\"bandpass_response\", \"cp_q_factor\"):   (0.8, 0.10),

        (\"time_waveform\", \"frequency_hz\"):      (1.0, 0.02),
        (\"time_waveform\", \"vpp_v\"):             (0.25, 0.10),
        (\"time_waveform\", \"cp_period_s\"):       (0.02, 0.05),
        (\"time_waveform\", \"cp_vmax_v\"):         (0.15, 0.10),
        (\"time_waveform\", \"cp_vmin_v\"):         (0.15, 0.10),
        (\"time_waveform\", \"cp_duty\"):           (0.08, 0.15),

        (\"fft_spectrum\", \"dominant_frequency_hz\"):   (2.0, 0.03),
        (\"fft_spectrum\", \"secondary_frequency_hz\"):  (3.0, 0.04),
        (\"fft_spectrum\", \"cp_peak_ratio\"):           (0.8, 0.25),

        (\"spectrogram\", \"f1_hz\"):               (25.0, 0.08),
        (\"spectrogram\", \"f2_hz\"):               (25.0, 0.08),
        (\"spectrogram\", \"switch_time_s\"):       (0.25, 0.15),
        (\"spectrogram\", \"cp_duration_s\"):       (0.10, 0.05),

        (\"iv_curve\", \"resistance_ohm\"):         (12.0, 0.06),
        (\"iv_curve\", \"target_current_a\"):       (1e-9, 1e-9),
        (\"iv_curve\", \"turn_on_voltage_v_at_target_i\"): (0.06, 0.12),
        (\"iv_curve\", \"cp_Is\"):                  (0.0, 0.20),
        (\"iv_curve\", \"cp_nVt\"):                 (0.0, 0.20),
        (\"iv_curve\", \"cp_Rs\"):                  (0.0, 0.25),

        (\"transfer_characteristic\", \"small_signal_gain\"): (0.20, 0.10),
        (\"transfer_characteristic\", \"saturation_v\"):       (0.10, 0.10),
        (\"transfer_characteristic\", \"cp_vin_at_saturation\"): (0.12, 0.15),
    }


def failure_tags(pred: Optional[float], gold: Optional[float]) -> List[str]:
    tags = []
    if pred is None:
        return [\"missing_value\"]
    if gold is None:
        return [\"missing_gold\"]
    if gold != 0:
        ratio = pred / gold
        if abs(ratio - 1000) < 50 or abs(ratio - 0.001) < 0.0001:
            tags.append(\"likely_unit_scale_error_x1000\")
        if abs(ratio - 2*math.pi) < 0.25 or abs(ratio - 1/(2*math.pi)) < 0.05:
            tags.append(\"hz_vs_rad_s\")
    if abs(pred) > 1e9:
        tags.append(\"implausible_magnitude\")
    return tags


@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str


def encode_image_base64(image_path: Path) -> str:
    return base64.b64encode(image_path.read_bytes()).decode(\"utf-8\")


def openai_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    from openai import OpenAI
    client = OpenAI()
    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)
    data_url = f\"data:{mime};base64,{b64}\"
    resp = client.responses.create(
        model=model,
        input=[{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"input_image\", \"image_url\": data_url},
                {\"type\": \"input_text\", \"text\": prompt},
            ],
        }],
        max_output_tokens=max_output_tokens,
    )
    return getattr(resp, \"output_text\", \"\") or \"\"


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
    return getattr(resp, \"text\", \"\") or \"\"


def anthropic_call(model: str, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    import anthropic
    api_key = os.environ.get(\"ANTHROPIC_API_KEY\", \"\")
    client = anthropic.Anthropic(api_key=api_key)
    b64 = encode_image_base64(image_path)
    mime = guess_mime(image_path)
    msg = client.messages.create(
        model=model,
        max_tokens=max_output_tokens,
        messages=[{
            \"role\": \"user\",
            \"content\": [
                {\"type\": \"image\", \"source\": {\"type\": \"base64\", \"media_type\": mime, \"data\": b64}},
                {\"type\": \"text\", \"text\": prompt},
            ],
        }],
    )
    out = []
    for block in getattr(msg, \"content\", []) or []:
        if getattr(block, \"type\", None) == \"text\":
            out.append(getattr(block, \"text\", \"\"))
        elif isinstance(block, dict) and block.get(\"type\") == \"text\":
            out.append(block.get(\"text\", \"\"))
    return \"\\n\".join(out).strip()


def call_model(spec: ModelSpec, image_path: Path, prompt: str, max_output_tokens: int) -> str:
    if spec.provider == \"openai\":
        return openai_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    if spec.provider == \"gemini\":
        return gemini_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    if spec.provider == \"anthropic\":
        return anthropic_call(spec.model, image_path, prompt, max_output_tokens=max_output_tokens)
    raise ValueError(f\"Unknown provider: {spec.provider}\")


def parse_models(args_models: List[str]) -> List[ModelSpec]:
    out = []
    for s in args_models:
        if \":\" not in s:
            raise ValueError(f\"Model must be provider:model, got: {s}\")
        p, m = s.split(\":\", 1)
        out.append(ModelSpec(provider=p.strip().lower(), model=m.strip()))
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument(\"--jsonl\", type=str, required=True, help=\"Path to plotchain_v3.jsonl\")
    ap.add_argument(\"--images_root\", type=str, default=\"\", help=\"Optional root directory for relative image paths\")
    ap.add_argument(\"--models\", type=str, nargs=\"+\", required=True,
                    help='List like: openai:gpt-4.1 gemini:gemini-2.5-flash anthropic:claude-sonnet-4-5')
    ap.add_argument(\"--out_dir\", type=str, default=\"results_v3\", help=\"Output directory\")
    ap.add_argument(\"--limit\", type=int, default=0, help=\"Optional: limit number of items\")
    ap.add_argument(\"--max_output_tokens\", type=int, default=500, help=\"Max tokens for model output\")
    ap.add_argument(\"--overwrite\", action=\"store_true\", help=\"Overwrite existing raw files\")
    ap.add_argument(\"--final_only\", action=\"store_true\", help=\"Prompt and score only final_fields (no checkpoints)\")
    args = ap.parse_args()

    jsonl_path = Path(args.jsonl)
    images_root = Path(args.images_root).resolve() if args.images_root else None
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    items = load_jsonl(jsonl_path)
    if args.limit and args.limit > 0:
        items = items[:args.limit]

    specs = parse_models(args.models)
    tols = tolerances_plotread()

    all_rows = []

    for spec in specs:
        raw_path = out_dir / f\"raw_{spec.provider}_{spec.model.replace('/', '_')}.jsonl\"
        if raw_path.exists() and not args.overwrite:
            print(f\"[skip] raw exists: {raw_path} (use --overwrite to regenerate)\")
            continue

        with raw_path.open(\"w\", encoding=\"utf-8\") as raw_f:
            for it in tqdm(items, desc=f\"{spec.provider}:{spec.model}\"):
                iid = str(it.get(\"id\", \"\"))
                typ = str(it.get(\"type\", \"\"))
                gt = dict(it.get(\"ground_truth\", {}) or {})
                img_path = get_image_path(it, images_root)
                prompt = build_prompt(it, final_only=args.final_only)

                t0 = time.time()
                try:
                    txt = call_model(spec, img_path, prompt, max_output_tokens=args.max_output_tokens)
                    err = None
                except Exception as e:
                    txt = \"\"
                    err = f\"{type(e).__name__}: {e}\"
                dt = time.time() - t0

                parsed = extract_first_json(txt) if txt else None

                gen = it.get(\"generation\", {}) or {}
                finals = list(gen.get(\"final_fields\", []))
                cps = list(gen.get(\"checkpoint_fields\", []))
                keys = finals[:] if args.final_only else (finals + cps)

                for k in keys:
                    gold = _to_float(gt.get(k))
                    pred = _to_float(parsed.get(k)) if isinstance(parsed, dict) else None

                    ae = abs_err(pred, gold)
                    re_ = rel_err(pred, gold)
                    abs_tol, rel_tol = tols.get((typ, k), (0.0, 0.0))

                    passed = False
                    if ae is not None and re_ is not None:
                        passed = (ae <= abs_tol) or (re_ <= rel_tol)

                    all_rows.append({
                        \"provider\": spec.provider,
                        \"model\": spec.model,
                        \"id\": iid,
                        \"type\": typ,
                        \"field\": k,
                        \"is_checkpoint\": k.startswith(\"cp_\"),
                        \"pred\": pred,
                        \"gold\": gold,
                        \"abs_err\": ae,
                        \"rel_err\": re_,
                        \"abs_tol\": abs_tol,
                        \"rel_tol\": rel_tol,
                        \"pass\": bool(passed),
                        \"latency_s\": dt,
                        \"error\": err,
                        \"tags\": \",\".join(failure_tags(pred, gold)),
                    })

                raw_f.write(json.dumps({
                    \"provider\": spec.provider,
                    \"model\": spec.model,
                    \"id\": iid,
                    \"type\": typ,
                    \"image_path\": str(img_path),
                    \"prompt\": prompt,
                    \"raw_text\": txt,
                    \"parsed_json\": parsed,
                    \"latency_s\": dt,
                    \"error\": err,
                }, ensure_ascii=False) + \"\\n\")

    if not all_rows:
        print(\"No results produced (maybe skipped due to existing raw files).\")
        return

    df = pd.DataFrame(all_rows)
    per_item_csv = out_dir / \"per_item.csv\"
    df.to_csv(per_item_csv, index=False)

    summary = (
        df.groupby([\"provider\", \"model\", \"type\", \"field\", \"is_checkpoint\"])
          .agg(
              n=(\"pass\", \"size\"),
              pass_rate=(\"pass\", \"mean\"),
              mean_abs_err=(\"abs_err\", \"mean\"),
              median_abs_err=(\"abs_err\", \"median\"),
              mean_rel_err=(\"rel_err\", \"mean\"),
              p95_abs_err=(\"abs_err\", lambda s: float(np.nanpercentile(s.dropna(), 95)) if s.dropna().size else np.nan),
              mean_latency_s=(\"latency_s\", \"mean\"),
          )
          .reset_index()
          .sort_values([\"provider\", \"model\", \"type\", \"is_checkpoint\", \"field\"])
    )
    summary_csv = out_dir / \"summary.csv\"
    summary.to_csv(summary_csv, index=False)

    overall = (
        df.groupby([\"provider\", \"model\", \"is_checkpoint\"])
          .agg(
              n=(\"pass\", \"size\"),
              overall_pass_rate=(\"pass\", \"mean\"),
              mean_abs_err=(\"abs_err\", \"mean\"),
              mean_rel_err=(\"rel_err\", \"mean\"),
              mean_latency_s=(\"latency_s\", \"mean\"),
          )
          .reset_index()
          .sort_values([\"provider\", \"model\", \"is_checkpoint\"])
    )
    overall_csv = out_dir / \"overall.csv\"
    overall.to_csv(overall_csv, index=False)

    print(f\"\\nWrote:\\n- {per_item_csv}\\n- {summary_csv}\\n- {overall_csv}\\n\")


if __name__ == \"__main__\":
    main()
