import re
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .base import DatasetAdapter, FrameRecord
from ..taxonomies import RAW_ID_TO_DENSE51


IGNORE_INDEX = 255


def parse_frame_index(path: Path) -> int:
    """
    Extract frame index from names such as:
        frame000123.jpg -> 123
    """
    matches = re.findall(r"\d+", path.stem)

    if not matches:
        raise ValueError(f"Could not extract frame index from: {path}")

    return int(matches[-1])


class ReplicaDataset(DatasetAdapter):

    def list_frames(
        self,
        scene: str,
        every_n_frame: int = 1,
    ) -> Sequence[FrameRecord]:

        if every_n_frame <= 0:
            raise ValueError("every_n_frame must be >= 1")

        rgb_dir = self.root / scene / "results"

        if not rgb_dir.is_dir():
            raise FileNotFoundError(
                f"Replica RGB directory not found: {rgb_dir}"
            )

        image_paths = [
            path
            for path in rgb_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
            and path.stem.startswith("frame")
        ]

        image_paths = sorted(
            image_paths,
            key=parse_frame_index,
        )

        # Apply evaluation frame stride.
        image_paths = image_paths[::every_n_frame]

        return [
            FrameRecord(
                scene=scene,
                frame_id=parse_frame_index(path),
                rgb_path=path,
            )
            for path in image_paths
        ]

    def load_semantic_gt(
        self,
        frame: FrameRecord,
    ) -> np.ndarray:
        """
        Load Replica semantic ground truth and convert raw Replica semantic IDs
        to dense Replica-51 benchmark IDs [0, 50].

        Pixels that are not part of Replica-51, including raw Replica ID 0
        ("other"), are assigned IGNORE_INDEX=255.
        """

        semantic_dir = (
            self.root
            / frame.scene
            / "semantic_ids"
        )

        semantic_path = (
            semantic_dir
            / f"semantic_id{frame.frame_id:06d}.png"
        )

        if not semantic_path.is_file():
            raise FileNotFoundError(
                f"Replica semantic GT not found: {semantic_path}"
            )

        raw_gt = cv2.imread(
            str(semantic_path),
            cv2.IMREAD_UNCHANGED,
        )

        if raw_gt is None:
            raise RuntimeError(
                f"Failed to read semantic GT: {semantic_path}"
            )

        # Semantic IDs should normally be single-channel.
        # Allow replicated RGB channels only if all channels are identical.
        if raw_gt.ndim == 3:
            if (
                raw_gt.shape[2] >= 3
                and np.array_equal(raw_gt[..., 0], raw_gt[..., 1])
                and np.array_equal(raw_gt[..., 0], raw_gt[..., 2])
            ):
                raw_gt = raw_gt[..., 0]
            else:
                raise ValueError(
                    f"Expected single-channel semantic IDs, "
                    f"got shape {raw_gt.shape}: {semantic_path}"
                )

        raw_gt = raw_gt.astype(np.int32, copy=False)

        dense_gt = np.full(
            raw_gt.shape,
            IGNORE_INDEX,
            dtype=np.uint8,
        )

        for raw_id, dense_id in RAW_ID_TO_DENSE51.items():
            dense_gt[raw_gt == raw_id] = dense_id

        return dense_gt

    def load_depth_gt(
        self,
        frame: FrameRecord,
    ) -> np.ndarray:

        depth_path = (
            self.root
            / frame.scene
            / "results"
            / f"depth{frame.frame_id:06d}.png"
        )

        depth_raw = cv2.imread(
            str(depth_path),
            cv2.IMREAD_UNCHANGED,
        )

        if depth_raw is None:
            raise FileNotFoundError(
                f"Could not read Replica depth: "
                f"{depth_path}"
            )

        if depth_raw.ndim != 2:
            raise RuntimeError(
                "Expected single-channel Replica depth, "
                f"got shape {depth_raw.shape} "
                f"for {depth_path}"
            )

        depth_m = (
            depth_raw.astype(np.float32)
            / 6553.5
        )

        return depth_m