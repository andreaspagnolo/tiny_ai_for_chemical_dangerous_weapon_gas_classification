"""Project-wide constants."""

from __future__ import annotations

CLASS_NAMES = ("TEP", "DIMP", "DMMP")
CLASS_TO_INDEX = {name: index for index, name in enumerate(CLASS_NAMES)}
RAW_FILES = {
    "TEP": "TEP_raw_Raman_spectra.xlsx",
    "DIMP": "DIMP_raw_Raman_spectra.xlsx",
    "DMMP": "DMMP_raw_Raman_spectra.xlsx",
}
SPECTRAL_LENGTH = 512
MODEL_INPUT_SHAPE = (1, SPECTRAL_LENGTH, 1)

