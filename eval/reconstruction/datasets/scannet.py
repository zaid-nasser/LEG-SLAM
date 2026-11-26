from __future__ import annotations

from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .base import (
    ReconstructionDatasetAdapter,
    ReconstructionFrame,
)


class ScanNetReconstructionDataset(
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

        color_dir = (
            self.root
            / scene
            / "color"
        )

        if not color_dir.is_dir():
            raise FileNotFoundError(
                f"Missing ScanNet color directory: "
                f"{color_dir}"
            )

        rgb_paths = (
            list(color_dir.glob("*.jpg"))
            + list(color_dir.glob("*.jpeg"))
            + list(color_dir.glob("*.png"))
        )

        def frame_id(path: Path) -> int:
            try:
                return int(path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"Expected numeric ScanNet frame name, "
                    f"got {path.name}"
                ) from exc

        rgb_paths = sorted(
            rgb_paths,
            key=frame_id,
        )

        frames = []

        for index, rgb_path in enumerate(rgb_paths):

            if index % every_n_frame != 0:
                continue

            frames.append(
                ReconstructionFrame(
                    scene=scene,
                    frame_id=frame_id(rgb_path),
                    rgb_path=rgb_path,
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
        """
        Load ScanNet depth in meters.

        Standard ScanNet exported depth PNGs are stored in millimeters.
        """

        depth_path = (
            self.root
            / frame.scene
            / "depth"
            / f"{frame.frame_id}.png"
        )

        depth = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth is None:
            raise FileNotFoundError(
                f"Could not read ScanNet depth image: "
                f"{depth_path}"
            )

        if depth.ndim != 2:
            raise ValueError(
                f"Expected single-channel ScanNet depth, "
                f"got shape {depth.shape} from {depth_path}"
            )

        return (
            depth.astype(np.float32)
            / 1000.0
        )

    def load_gt_poses(
        self,
        scene: str,
    ) -> dict[int, np.ndarray]:
        """
        Load ScanNet camera-to-world poses indexed by dataset frame ID.

        Invalid/non-finite ScanNet poses are skipped.
        """

        pose_dir = (
            self.root
            / scene
            / "pose"
        )

        if not pose_dir.is_dir():
            raise FileNotFoundError(
                f"Missing ScanNet pose directory: "
                f"{pose_dir}"
            )

        pose_paths = list(
            pose_dir.glob("*.txt")
        )

        def frame_id(path: Path) -> int:
            try:
                return int(path.stem)
            except ValueError as exc:
                raise ValueError(
                    f"Expected numeric ScanNet pose filename, "
                    f"got {path.name}"
                ) from exc

        pose_paths = sorted(
            pose_paths,
            key=frame_id,
        )

        poses: dict[int, np.ndarray] = {}

        for pose_path in pose_paths:
            pose = np.loadtxt(
                pose_path,
                dtype=np.float64,
            )

            if pose.shape != (4, 4):
                raise ValueError(
                    f"Expected 4x4 ScanNet pose in "
                    f"{pose_path}, got {pose.shape}"
                )

            # ScanNet can contain invalid poses containing inf/nan.
            if not np.all(np.isfinite(pose)):
                continue

            poses[
                frame_id(pose_path)
            ] = pose

        return poses