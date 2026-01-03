# families/__init__.py
from __future__ import annotations

from . import (
    step_response,
    bode_magnitude,
    bode_phase,
    bandpass_response,
    time_waveform,
    fft_spectrum,
    spectrogram,
    iv_curve,
    transfer_characteristic,
)

FAMILY_MODULES = [
    step_response,
    bode_magnitude,
    bode_phase,
    bandpass_response,
    time_waveform,
    fft_spectrum,
    spectrogram,
    iv_curve,
    transfer_characteristic,
]

REGISTRY = {m.TYPE: m for m in FAMILY_MODULES}
