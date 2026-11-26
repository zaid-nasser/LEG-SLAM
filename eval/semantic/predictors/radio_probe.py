from __future__ import annotations

from pathlib import Path
from typing import Any, Mapping, Optional, Sequence

import torch

from .base import SemanticPrediction
from .radio_closedset import (
    NUM_INDOOR37_CLASSES,
    LinearProbe,
    RadioLinearProbeHead,
)
from .radio_features import RadioPCAReconstructor


class RadioLinearProbePredictor:
    """
    Closed-set Indoor37 predictor operating on rendered compressed
    RADIO Gaussian features.

    This diagnostic predictor shares the exact same reusable linear head used
    by the dual closed-set/open-vocabulary fusion path.
    """

    def __init__(
        self,
        *,
        feature_model_id: str,
        model_cfg: Mapping[str, Any],
        checkpoint_path: Path,
        device: torch.device,
        chunk_size: int = 32768,
        expected_num_classes: int = NUM_INDOOR37_CLASSES,
    ) -> None:
        self.feature_model_id = feature_model_id
        self.device = torch.device(device)
        self.chunk_size = int(chunk_size)
        self.num_classes = int(expected_num_classes)
        self.checkpoint_path = Path(checkpoint_path).resolve()

        if self.chunk_size <= 0:
            raise ValueError("chunk_size must be > 0")

        self.reconstructor = RadioPCAReconstructor.from_model_config(
            model_cfg=model_cfg,
            device=self.device,
        )

        self.head = RadioLinearProbeHead(
            checkpoint_path=self.checkpoint_path,
            device=self.device,
            expected_feature_dim=self.reconstructor.full_dim,
            expected_num_classes=self.num_classes,
        )

        self.feature_dim = self.head.feature_dim
        self.checkpoint_epoch = self.head.checkpoint_epoch

        self.prediction_rule = (
            "rendered compressed RADIO features -> PCA inverse -> "
            "Indoor37 linear probe -> argmax logits"
        )

    @torch.inference_mode()
    def predict_logits(
        self,
        rendered_lf: torch.Tensor,
    ) -> torch.Tensor:
        """Return full Indoor37 logits [37,H,W]."""
        if rendered_lf.ndim != 3:
            raise ValueError(
                "Expected rendered_lf [K,H,W], "
                f"got {tuple(rendered_lf.shape)}"
            )

        _, height, width = rendered_lf.shape
        logit_chunks = []

        for _, _, full_chunk in self.reconstructor.iter_reconstructed_chunks(
            rendered_lf,
            chunk_size=self.chunk_size,
        ):
            logit_chunks.append(
                self.head.score_features(full_chunk)
            )

        if not logit_chunks:
            raise RuntimeError("No pixels were produced from rendered_lf.")

        logits_flat = torch.cat(logit_chunks, dim=0)
        expected_pixels = height * width

        if logits_flat.shape != (expected_pixels, self.num_classes):
            raise RuntimeError(
                "Unexpected probe-logit shape: "
                f"got {tuple(logits_flat.shape)}, expected "
                f"({expected_pixels}, {self.num_classes})"
            )

        return (
            logits_flat
            .reshape(height, width, self.num_classes)
            .permute(2, 0, 1)
            .contiguous()
        )

    def _validate_candidate_ids(
        self,
        candidate_ids: Sequence[int],
    ) -> list[int]:
        ids = [int(class_id) for class_id in candidate_ids]

        if not ids:
            raise ValueError("candidate_ids must not be empty")
        if len(set(ids)) != len(ids):
            raise ValueError("candidate_ids must be unique")

        invalid = [
            class_id
            for class_id in ids
            if class_id < 0 or class_id >= self.num_classes
        ]
        if invalid:
            raise ValueError(
                "Indoor37 candidate ids outside "
                f"[0,{self.num_classes - 1}]: {invalid}"
            )
        return ids

    def labels_from_scores(
        self,
        class_scores: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        if class_scores.ndim != 3:
            raise ValueError(
                "Expected class_scores [C,H,W], "
                f"got {tuple(class_scores.shape)}"
            )

        ids = self._validate_candidate_ids(candidate_ids)
        if class_scores.shape[0] != len(ids):
            raise RuntimeError(
                "Score/candidate mismatch: "
                f"scores have {class_scores.shape[0]} channels, "
                f"candidate_ids has {len(ids)} entries."
            )

        local_labels = torch.argmax(class_scores, dim=0)
        candidate_ids_t = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=class_scores.device,
        )
        return candidate_ids_t[local_labels]

    @torch.inference_mode()
    def predict(
        self,
        rendered_lf: torch.Tensor,
        candidate_ids: Optional[Sequence[int]] = None,
    ) -> SemanticPrediction:
        full_logits = self.predict_logits(rendered_lf)

        if candidate_ids is None:
            candidate_ids = list(range(self.num_classes))

        ids = self._validate_candidate_ids(candidate_ids)
        candidate_ids_t = torch.as_tensor(
            ids,
            dtype=torch.long,
            device=full_logits.device,
        )
        class_scores = full_logits.index_select(0, candidate_ids_t)
        labels = self.labels_from_scores(class_scores, ids)
        local_labels = torch.argmax(class_scores, dim=0)
        selected_score = (
            class_scores
            .gather(0, local_labels.unsqueeze(0))
            .squeeze(0)
        )

        return SemanticPrediction(
            labels=labels,
            scores=selected_score,
            class_scores=class_scores,
        )

    def metadata(self) -> dict[str, Any]:
        result = {
            "predictor": "radio_linear_probe",
            "feature_model": self.feature_model_id,
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint_epoch,
            "num_classes": self.num_classes,
            "feature_dim": self.feature_dim,
            "chunk_size": self.chunk_size,
            "prediction_rule": self.prediction_rule,
        }
        result.update(
            {
                f"pca_{key}": value
                for key, value in self.reconstructor.metadata().items()
            }
        )
        return result