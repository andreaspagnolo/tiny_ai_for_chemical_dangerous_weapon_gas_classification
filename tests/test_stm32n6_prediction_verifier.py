from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.verify_stm32n6_prediction import MAX_SCORE_DELTA, verify_stm32n6_prediction


EXPECTED = ["TEP", "TEP", "DIMP", "DIMP", "DMMP", "DMMP"]
CLASS_NAMES = ["TEP", "DIMP", "DMMP"]


def _write(path: Path, target: str, score_shift: float = 0.0, last_class: str | None = None):
    predictions = []
    for row, class_name in enumerate(EXPECTED):
        predicted_class = last_class if row == 5 and last_class else class_name
        scores = {name: 0.0 for name in CLASS_NAMES}
        scores[predicted_class] = 1.0 - score_shift
        predictions.append(
            {
                "row": row,
                "ground_truth_index": CLASS_NAMES.index(class_name),
                "ground_truth_class": class_name,
                "predicted_index": CLASS_NAMES.index(predicted_class),
                "predicted_class": predicted_class,
                "correct": predicted_class == class_name,
                "scores": scores,
            }
        )
    path.write_text(
        json.dumps({"target": target, "predictions": predictions}), encoding="utf-8"
    )


def test_n6_verifier_accepts_matching_predictions(tmp_path):
    host = tmp_path / "host.json"
    n6 = tmp_path / "n6.json"
    _write(host, "host")
    _write(n6, "stedgeai_n6", score_shift=1.0 / 256.0)
    assert verify_stm32n6_prediction(host, n6) == pytest.approx(1.0 / 256.0)


def test_n6_verifier_uses_32_quantization_step_limit(tmp_path):
    host = tmp_path / "host.json"
    n6 = tmp_path / "n6.json"
    _write(host, "host")
    _write(n6, "stedgeai_n6", score_shift=MAX_SCORE_DELTA)
    assert verify_stm32n6_prediction(host, n6) == pytest.approx(MAX_SCORE_DELTA)

    _write(n6, "stedgeai_n6", score_shift=MAX_SCORE_DELTA + 1.0 / 256.0)
    with pytest.raises(AssertionError, match="exceeds 32 INT8 output steps"):
        verify_stm32n6_prediction(host, n6)


def test_n6_verifier_rejects_class_disagreement(tmp_path):
    host = tmp_path / "host.json"
    n6 = tmp_path / "n6.json"
    _write(host, "host")
    _write(n6, "stedgeai_n6", last_class="TEP")
    with pytest.raises(AssertionError, match="differ from host"):
        verify_stm32n6_prediction(host, n6)
