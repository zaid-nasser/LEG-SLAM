from pathlib import Path
from typing import Any, Dict, Sequence
import json

import torch

from .base import SemanticPredictor
from .radio_v4 import (
    RadioDualSemanticPredictor,
    RadioSiglipSemanticPredictor,
)
from .radio_probe import RadioLinearProbePredictor


def load_feature_model_registry(
    registry_path: Path,
) -> Dict[str, Dict[str, Any]]:
    """Load cfg/feature_models.json and return the `models` dictionary."""
    registry_path = Path(registry_path)

    if not registry_path.is_file():
        raise FileNotFoundError(
            f"Feature-model registry not found: {registry_path}"
        )

    with registry_path.open("r") as f:
        registry = json.load(f)

    models = registry.get("models")
    if not isinstance(models, dict):
        raise ValueError(
            f"Registry has no valid 'models' object: {registry_path}"
        )

    return models


def build_semantic_predictor(
    class_names: Sequence[str],
    feature_model_id: str,
    registry_path: Path,
    radio_chunk_size: int = 32768,
    device: str | torch.device = "cuda",
    radio_inference: str = "open_vocab",
    probe_checkpoint: Path | None = None,
    prototype_path: Path | None = None,
    fusion_beta: float = 1.0,
    fusion_tau_ov: float = 1.0,
    fusion_tau_cs: float = 1.0,
    fusion_use_query_coverage: bool = True,
) -> SemanticPredictor:
    """Build the semantic predictor associated with a feature model."""

    models = load_feature_model_registry(registry_path)
    registry_path = Path(registry_path).resolve()
    repo_root = registry_path.parent.parent

    if feature_model_id not in models:
        raise ValueError(
            f"Unknown feature model {feature_model_id!r}. "
            f"Available models: {sorted(models.keys())}"
        )

    model_cfg = models[feature_model_id]
    alignment_cfg = model_cfg.get("text_alignment")

    if not isinstance(alignment_cfg, dict):
        raise ValueError(
            f"Feature model {feature_model_id!r} has no valid "
            f"'text_alignment' configuration in {registry_path}."
        )

    alignment_type = alignment_cfg.get("type")
    if alignment_type is None:
        raise ValueError(
            f"Feature model {feature_model_id!r} has no "
            f"'text_alignment.type' in {registry_path}."
        )

    if alignment_type == "radio_siglip":
        if radio_inference == "open_vocab":
            return RadioSiglipSemanticPredictor(
                class_names=class_names,
                feature_model_id=feature_model_id,
                model_cfg=model_cfg,
                chunk_size=radio_chunk_size,
                device=device,
                repo_root=repo_root,
            )

        if radio_inference == "dual":
            if probe_checkpoint is None:
                raise ValueError(
                    "--probe-checkpoint is required for --radio-inference dual"
                )
            if prototype_path is None:
                raise ValueError(
                    "--prototype-file is required for --radio-inference dual"
                )

            return RadioDualSemanticPredictor(
                class_names=class_names,
                feature_model_id=feature_model_id,
                model_cfg=model_cfg,
                chunk_size=radio_chunk_size,
                device=device,
                repo_root=repo_root,
                probe_checkpoint=probe_checkpoint,
                prototype_path=prototype_path,
                fusion_beta=fusion_beta,
                tau_ov=fusion_tau_ov,
                tau_cs=fusion_tau_cs,
                use_query_coverage=fusion_use_query_coverage,
            )

        raise ValueError(
            f"Unsupported RADIO inference mode: {radio_inference!r}"
        )

    raise ValueError(
        f"Unsupported text alignment type {alignment_type!r} "
        f"for feature model {feature_model_id!r}."
    )


def build_radio_probe_predictor(
    *,
    feature_model_id: str,
    registry_path: Path,
    checkpoint_path: Path,
    device: torch.device,
    chunk_size: int = 32768,
) -> RadioLinearProbePredictor:
    registry_path = Path(registry_path)

    if not registry_path.is_file():
        raise FileNotFoundError(
            "Feature-model registry not found: "
            f"{registry_path}"
        )

    with registry_path.open("r") as f:
        registry = json.load(f)

    models = registry.get("models")
    if not isinstance(models, dict):
        raise RuntimeError(
            "Feature-model registry has no valid 'models' object: "
            f"{registry_path}"
        )

    if feature_model_id not in models:
        raise ValueError(
            f"Unknown feature model {feature_model_id!r}. "
            f"Available: {sorted(models)}"
        )

    model_cfg = models[feature_model_id]

    return RadioLinearProbePredictor(
        feature_model_id=feature_model_id,
        model_cfg=model_cfg,
        checkpoint_path=checkpoint_path,
        device=device,
        chunk_size=chunk_size,
        expected_num_classes=37,
    )