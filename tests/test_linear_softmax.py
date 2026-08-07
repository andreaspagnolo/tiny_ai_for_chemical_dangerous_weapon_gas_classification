from __future__ import annotations

import yaml

from raman_stm32.linear_softmax_experiment import build_raman_linear_softmax


def _experiment():
    return {"model": {"l2_regularization": 1.0e-4}}


def test_linear_softmax_is_tiny_static_and_uses_standard_layers():
    model = build_raman_linear_softmax(_experiment(), batch_size=1)
    assert model.input_shape == (1, 1, 512, 1)
    assert model.output_shape == (1, 3)
    assert model.count_params() == 1539
    assert [layer.__class__.__name__ for layer in model.layers] == [
        "InputLayer",
        "Flatten",
        "Dense",
    ]


def test_linear_experiment_explicitly_uses_random_training_configuration():
    with open("configs/project/linear_softmax_experiment.yaml", encoding="utf-8") as stream:
        config = yaml.safe_load(stream)
    assert config["training"]["initialization"] == "random_glorot_uniform"
    assert config["model"]["l2_regularization"] == 1.0e-4

