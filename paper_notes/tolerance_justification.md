# Tolerance Policy Justification

## Suggested Text for Methodology Section

**Location**: Methodology section, after describing the evaluation protocol

**Text**:

> **Tolerance Policy**: We use a "plotread" tolerance policy designed to reflect fair human plot-reading precision. Tolerances are specified as (absolute_tolerance, relative_tolerance) pairs for each (plot_family, field) combination, where a prediction passes if either the absolute error ≤ absolute_tolerance OR the relative error ≤ relative_tolerance. These thresholds were set through expert judgment based on typical human measurement uncertainty when reading values from axes and identifying features. For example, frequency measurements allow ±5 Hz absolute tolerance or 5% relative tolerance, gain measurements allow ±0.5 dB absolute tolerance or 3% relative tolerance, and time measurements scale with the magnitude of the value. While not derived from formal human studies, these tolerances are designed to distinguish between reasonable reading precision (which should pass) and systematic errors or misreadings (which should fail). A "strict" policy with tighter tolerances (approximately 60% of plotread values) is also available for more stringent evaluation, but we report results using the plotread policy as it better reflects real-world usage scenarios. Complete tolerance values for all families and fields are provided in the Appendix (Table X).

**Alternative Shorter Version**:

> **Tolerance Policy**: We use a "plotread" tolerance policy with (absolute, relative) tolerance pairs for each (family, field) combination. A prediction passes if either absolute error ≤ absolute_tolerance OR relative error ≤ relative_tolerance. Thresholds were set through expert judgment to reflect typical human measurement uncertainty (e.g., ±5 Hz for frequencies, ±0.5 dB for gains). While not derived from formal human studies, these tolerances distinguish reasonable reading precision from systematic errors. Complete tolerance values are provided in the Appendix.
