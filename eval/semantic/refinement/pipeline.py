from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping

import torch

from .iterative import IterativeScoreRefiner
from .types import AffinityProvider


@dataclass
class RefinementPipeline:
    """
    Compose an affinity provider with a continuous score refiner.

    The pipeline is intentionally independent of the resource
    used to construct the affinity.
    """

    affinity_provider: AffinityProvider
    score_refiner: IterativeScoreRefiner

    def refine(
        self,
        class_scores: torch.Tensor,
        resources: Mapping[str, torch.Tensor],
    ) -> torch.Tensor:

        affinity = self.affinity_provider.compute(
            resources
        )

        return self.score_refiner.refine(
            class_scores,
            affinity,
        )