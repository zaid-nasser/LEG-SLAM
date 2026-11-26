from abc import ABC, abstractmethod
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np


@dataclass(frozen=True)
class FrameRecord:
    scene: str
    frame_id: int
    rgb_path: Path


class DatasetAdapter(ABC):
    def __init__(self, root: Path):
        self.root = Path(root)

    @abstractmethod
    def list_frames(
        self,
        scene: str,
        every_n_frame: int = 1,
    ) -> Sequence[FrameRecord]:
        pass

    @abstractmethod
    def load_semantic_gt(
        self,
        frame: FrameRecord,
    ) -> np.ndarray:
        """Return dense benchmark IDs, with ignored pixels = 255."""
        pass

    def load_depth_gt(
        self,
        frame: FrameRecord,
    ) -> np.ndarray:
        """
        Load ground-truth depth in meters.

        Returns:
            depth: float32 array [H, W]
        """
        raise NotImplementedError