from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Optional, Sequence

import numpy as np


@dataclass(frozen=True)
class ReconstructionFrame:
    scene: str
    frame_id: int
    rgb_path: Path
    depth_path: Optional[Path] = None


class ReconstructionDatasetAdapter(ABC):
    def __init__(self, root: Path):
        self.root = Path(root)

    @abstractmethod
    def list_frames(
        self,
        scene: str,
        every_n_frame: int = 1,
    ) -> Sequence[ReconstructionFrame]:
        pass

    @abstractmethod
    def load_rgb_gt(
        self,
        frame: ReconstructionFrame,
    ) -> np.ndarray:
        """
        Return RGB float32 image in [0,1], shape [H,W,3].
        """
        pass

    def load_depth_gt_m(
        self,
        frame: ReconstructionFrame,
    ) -> Optional[np.ndarray]:
        """
        Return depth in meters, or None if this evaluation protocol
        does not provide/use depth.
        """
        return None

    def load_gt_poses(
        self,
        scene: str,
    ) -> Optional[Mapping[int, np.ndarray]]:
        """
        Return frame_id -> 4x4 GT pose, or None if trajectory evaluation
        is not defined for this adapter.
        """
        return None

    def prepare_depth_prediction_m(
        self,
        pred_depth_m: np.ndarray,
    ) -> np.ndarray:
        """
        Convert rendered depth into the representation used for evaluation.

        Default behavior keeps metric depth in meters unchanged.
        """
        return np.asarray(
            pred_depth_m,
            dtype=np.float32,
        )