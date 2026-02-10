#!/usr/bin/env python3
"""
Standalone script to generate tolerance table for PlotChain paper appendix.

Usage:
    python3 generate_tolerance_table.py --output_dir analysis_output

This script extracts tolerance values from run_plotchain_eval.py and generates:
- CSV table: analysis_output/tables/tolerance_table.csv
- LaTeX table: analysis_output/tables/tolerance_table.tex
"""

import argparse
import pandas as pd
from pathlib import Path
import importlib.util
import sys


def generate_tolerance_table(output_dir: Path):
    """Generate families description table for paper."""
    # Import tolerance function from run_plotchain_eval
    eval_script_path = Path(__file__).parent / "run_plotchain_eval.py"
    module_name = "run_plotchain_eval"
    
    # Properly register the module in sys.modules before executing
    spec = importlib.util.spec_from_file_location(module_name, eval_script_path)
    eval_module = importlib.util.module_from_spec(spec)
    sys.modules[module_name] = eval_module  # Register before exec
    spec.loader.exec_module(eval_module)
    
    tol_map = eval_module.tolerances_plotread()
    
    # Extract final fields (non-checkpoint) for each family
    families_final_fields = {}
    for (family, field), _ in sorted(tol_map.items()):
        if not field.startswith("cp_"):  # Only final fields
            if family not in families_final_fields:
                families_final_fields[family] = []
            families_final_fields[family].append(field)
    
    # Add pole_zero manually (it uses heuristic tolerances)
    families_final_fields["pole_zero"] = ["pole_real", "pole_imag", "zero_real", "zero_imag"]
    
    # Family descriptions mapping
    family_descriptions = {
        "step_response": ("Controls; time vs response", ["percent_overshoot", "settling_time_s", "steady_state"]),
        "bode_magnitude": ("Circuits/controls; log-$f$ vs dB", ["dc_gain_db", "cutoff_hz"]),
        "bode_phase": ("Circuits/controls; log-$f$ vs deg", ["cutoff_hz", "phase_deg_at_fq"]),
        "bandpass_response": ("Filters; log-$f$ vs dB", ["resonance_hz", "bandwidth_hz"]),
        "time_waveform": ("Signals; time vs voltage", ["frequency_hz", "vpp_v"]),
        "fft_spectrum": ("Signals; $f$ vs magnitude", ["dominant_frequency_hz", "secondary_frequency_hz"]),
        "spectrogram": ("Signals; time--freq heatmap", ["f1_hz", "f2_hz", "switch_time_s"]),
        "iv_resistor": ("Circuits; $V$--$I$ linear", ["resistance_ohm"]),
        "iv_diode": ("Circuits; $V$--$I$ exponential", ["target_current_a", "turn_on_voltage_v_at_target_i"]),
        "transfer_characteristic": ("Nonlinear blocks; $V_{in}$--$V_{out}$", ["small_signal_gain", "saturation_v"]),
        "pole_zero": ("Systems; complex plane", ["pole_real", "pole_imag", "zero_real", "zero_imag"]),
        "stress_strain": ("Materials; strain vs stress", ["yield_strength_mpa", "uts_mpa", "fracture_strain"]),
        "torque_speed": ("Motors; speed vs torque", ["stall_torque_nm", "no_load_speed_rpm"]),
        "pump_curve": ("Fluids; flow vs head", ["head_at_qop_m", "q_at_half_head_m3h"]),
        "sn_curve": ("Fatigue; log--log cycles vs stress", ["stress_at_1e5_mpa", "endurance_limit_mpa"]),
    }
    
    # Create output directory
    tables_dir = output_dir / "tables"
    tables_dir.mkdir(parents=True, exist_ok=True)
    
    # Generate LaTeX table
    latex = "\\begin{table}[t]\n\\centering\n"
    latex += "\\caption{PlotChain plot families and representative target variables (final fields; checkpoint fields are prefixed \\texttt{cp\\_} and omitted for brevity).}\n"
    latex += "\\label{tab:families}\n\\footnotesize\n"
    latex += "\\begin{tabular}{p{0.28\\columnwidth} p{0.22\\columnwidth} p{0.34\\columnwidth}}\n"
    latex += "\\hline\n"
    latex += "\\textbf{Family} & \\textbf{Domain / Axes} & \\textbf{Representative final outputs} \\\\\n"
    latex += "\\hline\n"
    
    # Sort families in the order shown in the example
    family_order = [
        "step_response", "bode_magnitude", "bode_phase", "bandpass_response",
        "time_waveform", "fft_spectrum", "spectrogram", "iv_resistor", "iv_diode",
        "transfer_characteristic", "pole_zero", "stress_strain", "torque_speed",
        "pump_curve", "sn_curve"
    ]
    
    for family in family_order:
        if family in family_descriptions:
            domain_axes, representative_fields = family_descriptions[family]
            family_tex = f"\\texttt{{{family.replace('_', '\\_')}}}"
            
            # Format fields as comma-separated with \texttt
            fields_tex = ", ".join([f"\\texttt{{{f.replace('_', '\\_')}}}" for f in representative_fields])
            
            latex += f"{family_tex} & {domain_axes} & {fields_tex} \\\\\n"
    
    latex += "\\hline\n\\end{tabular}\n\\end{table}\n"
    
    # Save LaTeX
    tex_path = tables_dir / "tolerance_table.tex"
    with open(tex_path, "w") as f:
        f.write(latex)
    print(f"✅ Generated LaTeX: {tex_path}")
    print(f"\n📁 Families: {len(family_order)}")


def main():
    parser = argparse.ArgumentParser(
        description="Generate tolerance table for PlotChain paper appendix"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="analysis_output",
        help="Output directory for tables (default: analysis_output)"
    )
    
    args = parser.parse_args()
    output_dir = Path(args.output_dir)
    
    print("=" * 70)
    print("PlotChain Tolerance Table Generator")
    print("=" * 70)
    print()
    
    generate_tolerance_table(output_dir)
    
    print()
    print("=" * 70)
    print("Done!")
    print("=" * 70)


if __name__ == "__main__":
    main()
