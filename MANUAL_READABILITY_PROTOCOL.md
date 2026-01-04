# Manual Readability Protocol

## Purpose

This protocol provides a quick human validation check for each plot family to ensure plots are readable and ground truth values are verifiable.

## Protocol

For each family, perform 2-3 quick checks:

---

## 1. Step Response

- [ ] Can visually identify peak overshoot (highest point above steady state)
- [ ] Can read settling time (when curve enters ±2% band)
- [ ] Can verify steady state (final value)

**Expected**: All values should be readable from plot axes and grid.

---

## 2. Bode Magnitude

- [ ] Can identify DC gain (low-frequency magnitude in dB)
- [ ] Can locate -3dB point (where magnitude drops 3dB from DC)
- [ ] Can verify cutoff frequency matches vertical marker

**Expected**: Cutoff frequency should align with vertical line and -3dB point.

---

## 3. Bode Phase

- [ ] Can read phase at marked frequency f_q
- [ ] Can verify -45° at cutoff frequency
- [ ] Can locate cutoff frequency from vertical marker

**Expected**: Phase at f_q should match question, -45° at fc.

---

## 4. Bandpass Response

- [ ] Can identify resonance frequency (peak of curve)
- [ ] Can locate -3dB horizontal line (3dB below peak)
- [ ] Can verify f1 and f2 (where -3dB line intersects curve)
- [ ] Can compute bandwidth = f2 - f1

**Expected**: Vertical markers at f1 and f2 should align with -3dB line intersections.

---

## 5. Time Waveform

- [ ] Can identify frequency (from period: f = 1/T)
- [ ] Can read Vmax and Vmin from plot
- [ ] Can compute Vpp = Vmax - Vmin

**Expected**: Frequency and Vpp should match ground truth.

---

## 6. FFT Spectrum

- [ ] Can identify dominant frequency (highest peak)
- [ ] Can identify secondary frequency (second highest peak)
- [ ] Can estimate amplitude ratio from peak heights (in dB)

**Expected**: Frequencies should match ground truth, ratio should be computable from dB values.

---

## 7. Spectrogram

- [ ] Can identify f1 (frequency before switch)
- [ ] Can identify f2 (frequency after switch)
- [ ] Can locate switch time (when frequency changes)

**Expected**: All values should be readable from time and frequency axes.

---

## 8. IV Resistor

- [ ] Can verify linear relationship (V = I × R)
- [ ] Can read resistance from slope

**Expected**: Resistance should match ground truth.

---

## 9. IV Diode

- [ ] Can identify turn-on voltage at target current
- [ ] Can verify exponential relationship

**Expected**: Turn-on voltage should match ground truth at target current.

---

## 10. Transfer Characteristic

- [ ] Can identify small-signal gain (slope in linear region)
- [ ] Can identify saturation voltage (where curve flattens)

**Expected**: Gain and saturation should match ground truth.

---

## 11. Pole-Zero Plot

- [ ] Can read pole coordinates (marked with ×)
- [ ] Can read zero coordinates (marked with ○)

**Expected**: Coordinates should match ground truth.

---

## 12. Stress-Strain Curve

- [ ] Can identify yield strength (end of linear region)
- [ ] Can identify UTS (ultimate tensile strength, maximum stress)
- [ ] Can read fracture strain (end of curve)

**Expected**: All values should be readable from axes.

---

## 13. Torque-Speed Curve

- [ ] Can identify stall torque (torque at speed=0)
- [ ] Can identify no-load speed (speed at torque=0)

**Expected**: Both values should be readable from axes.

---

## 14. Pump Curve

- [ ] Can identify shutoff head (head at flow=0)
- [ ] Can read head at operating point
- [ ] Can identify flow at half head

**Expected**: All values should be readable from axes.

---

## 15. S-N Curve

- [ ] Can read stress at 10^5 cycles
- [ ] Can identify endurance limit (horizontal line at high cycles)

**Expected**: Values should be readable from log-scale x-axis.

---

## Validation Procedure

1. **Sample Check**: For each family, manually verify 3-5 random items
2. **Ground Truth Check**: Verify ground truth matches visual reading
3. **Tolerance Check**: Verify tolerances allow reasonable human error
4. **Edge Case Check**: Verify edge cases are still readable (even if harder)

## Expected Outcome

- ✅ All plots should be readable by human engineers
- ✅ Ground truth should match visual reading within tolerance
- ✅ Edge cases should be challenging but not impossible
- ✅ Checkpoint fields should help verify understanding

---

**Protocol Version**: 1.0
**Last Updated**: January 2026

