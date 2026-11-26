from __future__ import annotations

import re
from pathlib import Path
from typing import Mapping, Sequence

import cv2
import numpy as np

from .base import (
    ReconstructionDatasetAdapter,
    ReconstructionFrame,
)


REPLICA_DEPTH_SCALE = 6553.5


def _frame_id_from_path(path: Path) -> int:
    matches = re.findall(r"\d+", path.stem)

    if not matches:
        raise ValueError(
            f"Cannot recover frame ID from {path}"
        )

    return int(matches[-1])


class ReplicaReconstructionDataset(
    ReconstructionDatasetAdapter
):
    def list_frames(
        self,
        scene: str,
        every_n_frame: int = 1,
    ) -> Sequence[ReconstructionFrame]:

        if every_n_frame <= 0:
            raise ValueError(
                "every_n_frame must be > 0"
            )

        results_dir = (
            self.root
            / scene
            / "results"
        )

        if not results_dir.is_dir():
            raise FileNotFoundError(
                f"Missing Replica results directory: "
                f"{results_dir}"
            )

        rgb_paths = []

        for pattern in (
            "frame*.jpg",
            "frame*.jpeg",
            "frame*.png",
        ):
            rgb_paths.extend(
                results_dir.glob(pattern)
            )

        rgb_paths = sorted(
            rgb_paths,
            key=_frame_id_from_path,
        )

        frames = []

        for index, rgb_path in enumerate(rgb_paths):

            if index % every_n_frame != 0:
                continue

            frame_id = _frame_id_from_path(
                rgb_path
            )

            depth_path = (
                results_dir
                / f"depth{frame_id:06d}.png"
            )

            frames.append(
                ReconstructionFrame(
                    scene=scene,
                    frame_id=frame_id,
                    rgb_path=rgb_path,
                    depth_path=depth_path,
                )
            )

        return frames

    def load_rgb_gt(
        self,
        frame: ReconstructionFrame,
    ) -> np.ndarray:

        image = cv2.imread(
            str(frame.rgb_path),
            cv2.IMREAD_COLOR,
        )

        if image is None:
            raise FileNotFoundError(
                f"Could not read RGB image: "
                f"{frame.rgb_path}"
            )

        image = cv2.cvtColor(
            image,
            cv2.COLOR_BGR2RGB,
        )

        return (
            image.astype(np.float32)
            / 255.0
        )

    def load_depth_gt_m(
        self,
        frame: ReconstructionFrame,
    ) -> np.ndarray:

        if frame.depth_path is None:
            raise ValueError(
                f"No depth path for frame "
                f"{frame.frame_id}"
            )

        depth = cv2.imread(
            str(frame.depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            raise FileNotFoundError(
                f"Could not read depth image: "
                f"{frame.depth_path}"
            )

        return (
            depth.astype(np.float32)
            / REPLICA_DEPTH_SCALE
        )

    def load_gt_poses(
        self,
        scene: str,
    ) -> Mapping[int, np.ndarray]:

        trajectory_path = (
            self.root
            / scene
            / "traj.txt"
        )

        if not trajectory_path.is_file():
            raise FileNotFoundError(
                f"Missing Replica trajectory: "
                f"{trajectory_path}"
            )

        raw = np.loadtxt(
            trajectory_path
        )

        if raw.ndim == 1:
            raw = raw[None, :]

        if raw.shape[1] != 16:
            raise ValueError(
                f"Expected Replica trajectory rows "
                f"with 16 values, got {raw.shape}"
            )

        return {
            frame_id: row.reshape(4, 4)
            for frame_id, row
            in enumerate(raw)
        }
        
    def prepare_depth_prediction_m(
        self,
        pred_depth_m: np.ndarray,
    ) -> np.ndarray:
        """
        Preserve the legacy Replica evaluator's uint16 depth round-trip.
        """
        encoded = np.uint16(
            np.clip(
                pred_depth_m * REPLICA_DEPTH_SCALE,
                0,
                65535,
            )
        )

        return (
            encoded.astype(np.float32)
            / REPLICA_DEPTH_SCALE
        )