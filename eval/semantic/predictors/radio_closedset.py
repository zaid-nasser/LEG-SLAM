from __future__ import annotations

from dataclasses import dataclass
from math import log
from pathlib import Path
from typing import Any, Optional

import torch
import torch.nn as nn
import torch.nn.functional as F


INDOOR37_CLASS_NAMES = (
    "wall", "floor", "cabinet", "bed", "chair", "sofa", "table",
    "door", "window", "bookshelf", "picture", "counter", "blinds",
    "desk", "shelves", "curtain", "dresser", "pillow", "mirror",
    "floor_mat", "clothes", "ceiling", "books", "fridge", "tv",
    "paper", "towel", "shower_curtain", "box", "whiteboard",
    "person", "night_stand", "toilet", "sink", "lamp", "bathtub", "bag",
)

NUM_INDOOR37_CLASSES = len(INDOOR37_CLASS_NAMES)


class LinearProbe(nn.Module):
    """Exact 1x1-convolution architecture used to train the Indoor37 probe."""

    def __init__(
        self,
        in_channels: int = 1152,
        num_classes: int = NUM_INDOOR37_CLASSES,
    ) -> None:
        super().__init__()
        self.classifier = nn.Conv2d(
            in_channels,
            num_classes,
            kernel_size=1,
            bias=True,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.classifier(x)


@dataclass
class RadioClosedSetOutput:
    """Closed-set outputs for one flattened RADIO feature chunk."""

    logits: torch.Tensor              # [N, 37]
    probabilities: torch.Tensor       # [N, 37]
    head_labels: torch.Tensor         # [N]
    prototype_similarities: torch.Tensor  # [N, 37]
    prototype_labels: torch.Tensor    # [N]
    knownness: torch.Tensor           # [N]
    confidence: torch.Tensor          # [N], entropy-based M(x)


class RadioLinearProbeHead:
    """Reusable Indoor37 linear head that operates directly on [N,D] RADIO features."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        device: str | torch.device,
        expected_feature_dim: int,
        expected_num_classes: int = NUM_INDOOR37_CLASSES,
    ) -> None:
        self.device = torch.device(device)
        self.checkpoint_path = Path(checkpoint_path).resolve()
        self.feature_dim = int(expected_feature_dim)
        self.num_classes = int(expected_num_classes)

        if not self.checkpoint_path.is_file():
            raise FileNotFoundError(
                f"Linear-probe checkpoint not found: {self.checkpoint_path}"
            )

        print("Loading Indoor37 RADIO probe:", self.checkpoint_path)

        try:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            checkpoint = torch.load(
                self.checkpoint_path,
                map_location="cpu",
            )

        if not isinstance(checkpoint, dict):
            raise RuntimeError("Expected probe checkpoint to be a dictionary.")

        checkpoint_feature_dim = int(
            checkpoint.get("feature_dim", self.feature_dim)
        )
        if checkpoint_feature_dim != self.feature_dim:
            raise RuntimeError(
                "Probe/RADIO feature-dimension mismatch: "
                f"probe={checkpoint_feature_dim}, expected={self.feature_dim}"
            )

        checkpoint_num_classes = checkpoint.get("num_classes")
        if (
            checkpoint_num_classes is not None
            and int(checkpoint_num_classes) != self.num_classes
        ):
            raise RuntimeError(
                "Probe checkpoint class-count mismatch: "
                f"checkpoint={checkpoint_num_classes}, expected={self.num_classes}"
            )

        probe = LinearProbe(
            in_channels=self.feature_dim,
            num_classes=self.num_classes,
        )

        state = checkpoint["model"] if "model" in checkpoint else checkpoint
        probe.load_state_dict(state)
        probe = probe.to(self.device).eval()

        for parameter in probe.parameters():
            parameter.requires_grad_(False)

        self.classifier_weight = (
            probe.classifier.weight.detach()
            .reshape(self.num_classes, self.feature_dim)
            .float()
            .contiguous()
        )

        classifier_bias = probe.classifier.bias
        self.classifier_bias = (
            None
            if classifier_bias is None
            else classifier_bias.detach().float().contiguous()
        )

        self.checkpoint_epoch = checkpoint.get("epoch", None)
        del probe

        print(
            f"Indoor37 probe loaded: {self.feature_dim} -> {self.num_classes}, "
            f"epoch={self.checkpoint_epoch if self.checkpoint_epoch is not None else 'unknown'}"
        )

    @torch.inference_mode()
    def score_features(self, features: torch.Tensor) -> torch.Tensor:
        """Return raw Indoor37 logits [N,37] for reconstructed RADIO features [N,D]."""
        if features.ndim != 2:
            raise ValueError(
                f"Expected RADIO features [N,D], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                "RADIO feature dimension does not match probe: "
                f"features={features.shape[1]}, probe={self.feature_dim}"
            )

        return F.linear(
            features.float(),
            self.classifier_weight,
            self.classifier_bias,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "checkpoint": str(self.checkpoint_path),
            "checkpoint_epoch": self.checkpoint_epoch,
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
        }


class RadioPrototypeKnownness:
    """Prototype cosine similarity / knownness for the fixed Indoor37 vocabulary."""

    def __init__(
        self,
        *,
        prototype_path: Path,
        device: str | torch.device,
        expected_feature_dim: int,
        expected_num_classes: int = NUM_INDOOR37_CLASSES,
    ) -> None:
        self.device = torch.device(device)
        self.prototype_path = Path(prototype_path).resolve()
        self.feature_dim = int(expected_feature_dim)
        self.num_classes = int(expected_num_classes)

        if not self.prototype_path.is_file():
            raise FileNotFoundError(
                f"RADIO Indoor37 prototype file not found: {self.prototype_path}"
            )

        print("Loading Indoor37 RADIO prototypes:", self.prototype_path)

        try:
            data = torch.load(
                self.prototype_path,
                map_location="cpu",
                weights_only=False,
            )
        except TypeError:
            data = torch.load(
                self.prototype_path,
                map_location="cpu",
            )

        if not isinstance(data, dict):
            raise RuntimeError("Expected prototype file to contain a dictionary.")

        prototypes = data.get("prototypes_l2")
        if prototypes is None:
            prototypes = data.get("prototypes")
        if prototypes is None:
            raise RuntimeError(
                "Prototype file must contain 'prototypes_l2' or 'prototypes'."
            )

        prototypes = torch.as_tensor(prototypes).detach().float().cpu()

        # The current file stores [1152,37].  Accept [37,1152] too so the
        # loader is explicit and robust to future serialization changes.
        if prototypes.shape == (self.feature_dim, self.num_classes):
            prototypes = prototypes.transpose(0, 1)
        elif prototypes.shape != (self.num_classes, self.feature_dim):
            raise RuntimeError(
                "Unexpected prototype shape: "
                f"got {tuple(prototypes.shape)}, expected "
                f"({self.feature_dim},{self.num_classes}) or "
                f"({self.num_classes},{self.feature_dim})"
            )

        # Normalize again even when prototypes_l2 is supplied.  This is cheap
        # and guarantees the cosine-similarity invariant used by the fusion.
        self.prototypes = F.normalize(
            prototypes,
            dim=-1,
        ).to(self.device, non_blocking=True)

        counts = data.get("counts")
        self.counts: Optional[torch.Tensor]
        if counts is None:
            self.counts = None
        else:
            counts = torch.as_tensor(counts).detach().cpu().reshape(-1)
            if counts.numel() != self.num_classes:
                raise RuntimeError(
                    "Prototype count vector has wrong length: "
                    f"got {counts.numel()}, expected {self.num_classes}"
                )
            self.counts = counts

        print(
            "Indoor37 prototypes loaded: "
            f"{self.num_classes} x {self.feature_dim}"
        )

    @torch.inference_mode()
    def score_features(
        self,
        features: torch.Tensor,
    ) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """
        Return:
            similarities      [N,37]
            prototype_labels [N]
            knownness         [N] = max_c cosine(f, mu_c)
        """
        if features.ndim != 2:
            raise ValueError(
                f"Expected RADIO features [N,D], got {tuple(features.shape)}"
            )
        if features.shape[1] != self.feature_dim:
            raise RuntimeError(
                "RADIO feature dimension does not match prototypes: "
                f"features={features.shape[1]}, prototypes={self.feature_dim}"
            )

        normalized_features = F.normalize(
            features.float(),
            dim=-1,
        )
        similarities = torch.matmul(
            normalized_features,
            self.prototypes.transpose(0, 1),
        )
        knownness, prototype_labels = torch.max(
            similarities,
            dim=-1,
        )
        return similarities, prototype_labels, knownness

    def metadata(self) -> dict[str, Any]:
        result = {
            "prototype_path": str(self.prototype_path),
            "feature_dim": self.feature_dim,
            "num_classes": self.num_classes,
        }
        if self.counts is not None:
            result["prototype_counts"] = [int(v) for v in self.counts.tolist()]
        return result


class RadioIndoor37KnownnessBranch:
    """Complete fixed Indoor37 branch used by dual semantic fusion."""

    def __init__(
        self,
        *,
        checkpoint_path: Path,
        prototype_path: Path,
        device: str | torch.device,
        feature_dim: int,
    ) -> None:
        self.device = torch.device(device)
        self.feature_dim = int(feature_dim)
        self.num_classes = NUM_INDOOR37_CLASSES

        self.head = RadioLinearProbeHead(
            checkpoint_path=checkpoint_path,
            device=self.device,
            expected_feature_dim=self.feature_dim,
            expected_num_classes=self.num_classes,
        )
        self.prototype_model = RadioPrototypeKnownness(
            prototype_path=prototype_path,
            device=self.device,
            expected_feature_dim=self.feature_dim,
            expected_num_classes=self.num_classes,
        )

        self._entropy_normalizer = float(log(self.num_classes))

    @torch.inference_mode()
    def infer_features(self, features: torch.Tensor) -> RadioClosedSetOutput:
        """Implement H(f), prototype knownness K(x), and entropy confidence M(x)."""
        logits = self.head.score_features(features)
        probabilities = torch.softmax(logits, dim=-1)
        head_labels = torch.argmax(probabilities, dim=-1)

        prototype_similarities, prototype_labels, knownness = (
            self.prototype_model.score_features(features)
        )

        safe_probabilities = probabilities.clamp_min(1e-12)
        entropy = -torch.sum(
            probabilities * torch.log(safe_probabilities),
            dim=-1,
        )
        confidence = 1.0 - entropy / self._entropy_normalizer
        confidence = confidence.clamp(0.0, 1.0)

        return RadioClosedSetOutput(
            logits=logits,
            probabilities=probabilities,
            head_labels=head_labels,
            prototype_similarities=prototype_similarities,
            prototype_labels=prototype_labels,
            knownness=knownness,
            confidence=confidence,
        )

    def metadata(self) -> dict[str, Any]:
        return {
            "indoor37_class_names": list(INDOOR37_CLASS_NAMES),
            "head": self.head.metadata(),
            "prototypes": self.prototype_model.metadata(),
        }