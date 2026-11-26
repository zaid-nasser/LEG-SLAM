from pathlib import Path

import pytest

from eval.semantic.predictors.factory import (
    load_feature_model_registry,
    build_semantic_predictor,
)

REGISTRY = Path(__file__).resolve().parents[3] / "cfg/feature_models.json"


def test_registry_loads():
    models = load_feature_model_registry(REGISTRY)
    assert set(models) == {"radio_v4_so400m"}
    model = models["radio_v4_so400m"]
    assert model["encoder"]["type"] == "RadioTorchScript"
    assert model["compressor"]["type"] == "TorchPCA"
    assert model["text_alignment"]["type"] == "radio_siglip"


def test_unknown_feature_model_rejected():
    with pytest.raises(ValueError, match="Unknown feature model"):
        build_semantic_predictor(
            class_names=["wall"],
            feature_model_id="unsupported_model",
            registry_path=REGISTRY,
        )
