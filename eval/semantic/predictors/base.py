from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional, Sequence

import torch


@dataclass
class SemanticPrediction:
    labels: torch.Tensor       # [H, W], dense taxonomy IDs
    scores: torch.Tensor       # [H, W], selected class score/confidence
    class_scores: Optional[torch.Tensor]
    aux: Optional[Dict[str, torch.Tensor]] = None


class SemanticPredictor(ABC):
    """
    Base interface for semantic prediction backends.

    A predictor receives semantic class candidates and returns dense semantic
    labels in the taxonomy used by the evaluation protocol.
    """

    def __init__(
        self,
        class_names: Sequence[str],
        device: str | torch.device,
    ):
        self.class_names = list(class_names)
        self.num_classes = len(self.class_names)
        self.device = torch.device(device)

    @abstractmethod
    def predict(
        self,
        rendered_lf: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> SemanticPrediction:
        """
        Predict semantic labels from a rendered semantic feature map.

        Args:
            rendered_lf:
                Semantic feature map with shape [K, H, W].

            candidate_ids:
                Dense taxonomy IDs allowed as prediction candidates.

        Returns:
            SemanticPrediction:
                labels: [H, W]
                scores: [H, W]
                class_scores: [C, H, W] or None
                aux: optional predictor-specific diagnostic maps
        """
        pass

    @abstractmethod
    def metadata(self) -> Dict[str, Any]:
        pass

    def labels_from_scores(
        self,
        class_scores: torch.Tensor,
        candidate_ids: Sequence[int],
    ) -> torch.Tensor:
        ...