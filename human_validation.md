Below is a human-validation checklist you can copy into your repo. It’s written for a verifier who doesn’t know the domain concepts—you just need to read axes, ticks, and approximate values.

Use this same workflow for every family:

Universal workflow (do this for every plot)

Open the plot image and the corresponding JSONL item (same id).

In the JSONL item, locate:

ground_truth (the numbers you must verify)

generation.final_fields and generation.checkpoint_fields (what you must be able to read)

Check the plot has:

readable axis labels/units

visible tick marks (or grid in clean/moderate)

enough resolution (not blurred)

For each field in final_fields + checkpoint_fields:

read the value from the plot using ticks/grid

confirm you can get close to the ground truth

If you can’t read a value reasonably (even with careful zoom), record:

id, field, and what made it unreadable (ticks too coarse, overlap, no marker, etc.)

Family-by-family human validation steps
1) step_response

Goal: From a step response curve, read steady-state, overshoot, and settling-ish timing.

Verify you can read:

steady_state

percent_overshoot

settling_time_s

checkpoints like cp_peak_time_s, cp_peak_value, cp_band_lower, cp_band_upper

Steps

Steady state: look at the tail end (right side). Estimate the final flat value on Y-axis → compare to steady_state.

Peak value: find the highest point of the curve → compare to cp_peak_value.

Peak time: read the X location of the peak → compare to cp_peak_time_s.

Percent overshoot: compute visually:

overshoot ≈ (peak − steady)/steady × 100

You don’t need exact math; verify it’s consistent with GT.

Settling time:

Identify the band around steady state (if band lines exist in clean/moderate, use them).

Find the first time after which the curve stays within that band → compare to settling_time_s.

Pass if: you can get these from the plot without guessing wildly.

2) bode_magnitude

Goal: Read DC gain, cutoff frequency, slope, and magnitude at cutoff.

Verify you can read:

dc_gain_db

cutoff_hz

cp_mag_at_fc_db

cp_slope_db_per_decade

Steps

DC gain (left plateau): On the low-frequency (left) side, read the flat magnitude → dc_gain_db.

Cutoff frequency: where magnitude drops by ~3 dB from DC plateau:

Either a dashed 3 dB reference exists, or you do it visually.

Read the X-value at that point → cutoff_hz.

Magnitude at cutoff: read Y at cutoff_hz → cp_mag_at_fc_db.

Slope (high frequency):

Pick two points one decade apart (e.g., 10× in frequency).

Compare their magnitudes: slope ≈ ΔdB per decade → cp_slope_db_per_decade.

You’re checking it’s readable and consistent, not doing a precise controls derivation.

3) bode_phase

Goal: Read phase at a specific frequency and (now) phase at fq (not fixed 10fc if you changed it).

Verify you can read (depending on dataset):

cutoff_hz

cp_phase_deg_at_fc

phase_deg_at_fq_hz (or phase_deg_at_10fc if still present)

Steps

Cutoff frequency:

If a marker/vertical line exists at fc, read its X → cutoff_hz.

Phase at fc:

At fc, read Y-phase value → cp_phase_deg_at_fc.

Phase at query frequency:

Find the vertical fq line (often dotted).

Read phase where curve intersects that line → phase_deg_at_fq_hz.

Check: values should vary across items (not constant across all plots unless that’s intentionally fixed).

4) bandpass_response

Goal: Read resonance frequency, 3 dB bandwidth, Q, and gain at resonance.

Verify you can read:

resonance_hz (now can be 1 decimal)

bandwidth_hz

checkpoints: cp_f1_3db_hz, cp_f2_3db_hz, cp_q_factor, cp_gain_db_at_res

Steps

Resonance frequency:

Find the peak of the curve.

Read X at peak → resonance_hz (expect 1 decimal).

Peak gain:

Read Y at peak → cp_gain_db_at_res.

3 dB points:

3 dB down from peak = (peak gain − 3 dB).

Find the two frequencies where the curve crosses that level:

left crossing → cp_f1_3db_hz

right crossing → cp_f2_3db_hz

Bandwidth:

bandwidth = f2 − f1 → bandwidth_hz (you can approximate; GT will match).

Q factor:

Q = resonance / bandwidth → cp_q_factor.

Edge case expectation: In edge plots the 3 dB guide line may be missing, but the peak and crossings must still be readable.

5) time_waveform

Goal: Read frequency and Vpp from a time-domain waveform.

Verify you can read:

frequency_hz

vpp_v

checkpoints: cp_period_s, cp_vmax_v, cp_vmin_v, cp_duty (square only)

Steps

Vmax and Vmin:

Read top and bottom of waveform → cp_vmax_v, cp_vmin_v.

Vpp:

Vpp = Vmax − Vmin → vpp_v.

Period:

Measure time between repeating features (peak-to-peak or rising edges) → cp_period_s.

Frequency:

f = 1/period → frequency_hz.

Duty (square wave only):

One cycle: measure high-time / period → cp_duty.

If it’s sine/triangle, duty may be absent (that’s okay).

6) fft_spectrum

Goal: Read dominant & secondary frequency peaks + their amplitude ratio.

Verify you can read:

dominant_frequency_hz

secondary_frequency_hz

cp_peak_ratio

Steps

Identify the highest peak → read X → dominant_frequency_hz.

Identify the 2nd peak → read X → secondary_frequency_hz.

Read the heights (Y values) of both peaks using ticks/grid.

Compute ratio visually: (dominant amplitude / secondary amplitude) → cp_peak_ratio.

The key is you can estimate ratio without getting fractional weirdness.

If ticks are coarse, ratio must still be reasonably determinable.

Sanity check: peaks should not be ambiguous (overlapping or too close) unless it’s intentionally an edge case.

7) spectrogram

Goal: Read two frequencies (before/after a switch) and switch time / duration.

Verify you can read:

f1_hz, f2_hz

switch_time_s

cp_duration_s (if present)

Steps

Identify the strong band before the switch → read frequency (Y) → f1_hz.

Identify the strong band after the switch → read frequency → f2_hz.

Find the time when the band changes → switch_time_s.

If duration is asked:

read total time span of plot or band span (depending on your generator) → cp_duration_s.

8) transfer_characteristic

Goal: Read small-signal gain and saturation.

Verify you can read:

small_signal_gain

saturation_v

cp_vin_at_saturation

Steps

Look at the initial linear region: measure slope ΔY/ΔX from two points → small_signal_gain.

Find where curve flattens (output saturates) → read Y plateau → saturation_v.

Find the input X where it first reaches saturation plateau → cp_vin_at_saturation.

9) pole_zero (if present in your 10/15 families)

Goal: Read pole/zero locations from a step-like magnitude response (or similar).

Verify you can read: whatever your GT defines (often intercepts/corners/pole counts).

Steps

Identify each “corner” / step transition point.

Read X locations of transitions.

Read level differences between plateaus if required.

Confirm tick spacing: clean/moderate should be fine-grained (1–2), edge can be coarse (5).

10) iv_resistor (split family)

Goal: Read resistance from a straight line I–V.

Verify you can read:

resistance_ohm

cp_slope_ohm

Steps

Pick two clear points on the line (preferably at tick intersections).

Compute slope:

If plot is V on x and I on y: slope = ΔI/ΔV, resistance = 1/slope.

If plot is I on x and V on y: slope = ΔV/ΔI, resistance = slope.

Compare to resistance_ohm and cp_slope_ohm accordingly (your GT should match your axis convention).

Confirm line is not too thick / too few ticks.

11) iv_diode (split family)

Goal: Read turn-on voltage at a specified target current and diode checkpoints.

Verify you can read:

target_current_a (should be in question and GT)

turn_on_voltage_v_at_target_i

checkpoints: cp_Is, cp_nVt, cp_Rs (if included)

Steps

Locate the horizontal line at target_current_a (or find that current on Y axis).

Move horizontally to the curve, then down to X to read voltage → turn_on_voltage_v_at_target_i.

For checkpoints (if asked):

These are harder to “read” directly unless the plot includes annotations.

Your goal is just to confirm the dataset makes them possible if guides exist, or that you’re okay treating them as “model-only checkpoints”.

If you want strict human readability, ensure the plot includes enough structure/markers.

12) nyquist (new)

Goal: Read margins/crossover values if your GT asks them.

Typical GT fields (example):

gain_margin_db

phase_margin_deg

crossover_freq_hz

Steps

Identify the critical point (usually -1+0j on the real axis).

Find where the Nyquist curve crosses the unit circle / real axis (depends on design).

Use any marked crossover frequency indicators if included.

If no explicit markers exist, confirm the fields chosen are actually human-readable—otherwise you’ll need markers or simpler GT fields.

13) roc / roc_curve (new)

Goal: Read AUC and/or a point on the curve.

Common GT:

auc OR tpr_at_fpr OR eer

Steps

Confirm axes are labeled 0–1 with readable ticks.

If verifying AUC:

confirm the curve is not too close to random (diagonal) unless intended

AUC should be provided by GT; human check is “is it plausible” not computing exact integral

If you require true human computability, don’t use AUC as a must-read value unless annotated.

If verifying TPR at FPR=x:

find x on FPR axis, go up to curve, read TPR on y-axis.

If verifying EER:

find where FPR ≈ 1-TPR (or where curve intersects the diagonal in DET-like views).

Again: if you want human verification, include a guide/marker.

14) constellation (new)

Goal: Read SNR-ish / EVM-ish metrics only if annotated; otherwise validate cluster geometry.

Possible GT:

snr_db, evm_percent, ber

Steps

Check points form a clear constellation (QPSK/16QAM etc).

Verify clusters are visually distinct and not smeared too much except in edge cases.

If GT includes snr_db/evm_percent:

You need an annotation/legend or a way to read it.

Otherwise these are not “human plot reads” from scatter alone.

If you don’t annotate, you should validate only geometry-based fields (e.g., number of clusters, centroid spacing).

15) learning_curve (new)

Goal: Read best epoch / min val loss / generalization gap.

Common GT:

best_epoch, min_val_loss, final_val_loss, final_train_loss, generalization_gap

Steps

Identify the validation loss curve (ensure legend is clear).

Find the minimum point on validation curve:

read Y value → min_val_loss

read X epoch at that minimum → best_epoch

Read final values (last epoch):

final val loss and final train loss

Generalization gap:

gap = final val loss − final train loss (or at best epoch if defined)

Edge plots may have noisier curves; still must have readable ticks.