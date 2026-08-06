from __future__ import annotations

import numpy as np
import pandas as pd

from raman_stm32.data import assign_splits, choose_group_splits, snv_clip_scale


def test_snv_clip_scale_is_finite_bounded_and_deterministic():
    spectra = np.asarray([[1, 2, 4, 8], [10, 9, 5, 2]], dtype=np.float32)
    first = snv_clip_scale(spectra)
    second = snv_clip_scale(spectra)
    assert first.dtype == np.float32
    assert np.array_equal(first, second)
    assert np.isfinite(first).all()
    assert np.max(np.abs(first)) <= 1.0


def test_concentration_groups_are_globally_disjoint():
    concentrations = [0.005, 0.01, 0.02, 0.04, 0.06, 0.08, 0.1, 0.25, 0.5, 0.75, 1.0]
    rows = []
    for label in range(3):
        for concentration in concentrations:
            rows.extend({"label": label, "concentration_fraction": concentration} for _ in range(5 + label))
    metadata = pd.DataFrame(rows)
    groups = choose_group_splits(metadata, seed=20260805)
    assert [len(groups[name]) for name in ("train", "validation", "test")] == [7, 2, 2]
    flattened = [value for values in groups.values() for value in values]
    assert len(flattened) == len(set(flattened)) == 11
    assigned = assign_splits(metadata, groups)
    check = metadata.assign(split=assigned).groupby("concentration_fraction")["split"].nunique()
    assert (check == 1).all()

