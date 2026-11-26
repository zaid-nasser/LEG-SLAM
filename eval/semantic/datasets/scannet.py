import csv
from pathlib import Path
from typing import Sequence

import cv2
import numpy as np

from .base import DatasetAdapter, FrameRecord
from ..taxonomies import SCANNET20_NYU40_TO_DENSE


IGNORE_INDEX = 255


def parse_frame_index(path: Path) -> int:
    """
    ScanNet filenames are typically:
        0.jpg
        1.jpg
        ...
    """
    try:
        return int(path.stem)
    except ValueError as exc:
        raise ValueError(
            f"Could not extract ScanNet frame index from: {path}"
        ) from exc


def read_scannet_label_mapping(
    label_file: Path,
) -> dict[int, int]:
    """
    Read ScanNet TSV mapping:

        raw ScanNet semantic ID -> NYU40 ID

    Expected TSV columns:
        id
        nyu40id
    """
    label_file = Path(label_file)

    if not label_file.is_file():
        raise FileNotFoundError(
            f"ScanNet label mapping file not found: {label_file}"
        )

    mapping: dict[int, int] = {}

    with label_file.open(newline="") as f:
        reader = csv.DictReader(f, delimiter="\t")

        if reader.fieldnames is None:
            raise ValueError(
                f"ScanNet TSV has no header: {label_file}"
            )

        required_columns = {"id", "nyu40id"}

        if not required_columns.issubset(reader.fieldnames):
            raise ValueError(
                f"ScanNet TSV must contain columns {required_columns}. "
                f"Found: {reader.fieldnames}"
            )

        for row in reader:
            try:
                raw_id = int(row["id"])
                nyu40_id = int(row["nyu40id"])
            except (ValueError, TypeError):
                continue

            mapping[raw_id] = nyu40_id

    return mapping


class ScanNetDataset(DatasetAdapter):

    def __init__(
        self,
        root: Path,
        label_file: Path,
    ):
        super().__init__(root)

        self.label_file = Path(label_file)
        self.raw_to_nyu40 = read_scannet_label_mapping(
            self.label_file
        )

    def list_frames(
        self,
        scene: str,
        every_n_frame: int = 1,
    ) -> Sequence[FrameRecord]:

        if every_n_frame <= 0:
            raise ValueError("every_n_frame must be >= 1")

        rgb_dir = self.root / scene / "color"

        if not rgb_dir.is_dir():
            raise FileNotFoundError(
                f"ScanNet RGB directory not found: {rgb_dir}"
            )

        image_paths = [
            path
            for path in rgb_dir.iterdir()
            if path.is_file()
            and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
        ]

        image_paths = sorted(
            image_paths,
            key=parse_frame_index,
        )

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
        Load ScanNet 2D semantic GT and convert:

            raw ScanNet ID
                -> NYU40 ID
                -> dense ScanNet20 ID [0, 19]

        Anything outside ScanNet20 becomes IGNORE_INDEX=255.
        """

        label_dir = (
            self.root
            / frame.scene
            / f"{frame.scene}_2d-label-filt"
            / "label-filt"
        )

        label_path = label_dir / f"{frame.frame_id}.png"

        if not label_path.is_file():
            raise FileNotFoundError(
                f"ScanNet semantic GT not found: {label_path}"
            )

        raw_gt = cv2.imread(
            str(label_path),
            cv2.IMREAD_UNCHANGED,
        )

        if raw_gt is None:
            raise RuntimeError(
                f"Failed to read ScanNet semantic GT: {label_path}"
            )

        if raw_gt.ndim == 3:
            if (
                raw_gt.shape[2] >= 3
                and np.array_equal(raw_gt[..., 0], raw_gt[..., 1])
                and np.array_equal(raw_gt[..., 0], raw_gt[..., 2])
            ):
                raw_gt = raw_gt[..., 0]
            else:
                raise ValueError(
                    f"Expected single-channel ScanNet semantic IDs, "
                    f"got shape {raw_gt.shape}: {label_path}"
                )

        raw_gt = raw_gt.astype(np.int32, copy=False)

        dense_gt = np.full(
            raw_gt.shape,
            IGNORE_INDEX,
            dtype=np.uint8,
        )

        # Only iterate over IDs actually present in this frame.
        for raw_id in np.unique(raw_gt):
            raw_id = int(raw_id)

            nyu40_id = self.raw_to_nyu40.get(raw_id)

            if nyu40_id is None:
                continue

            dense_id = SCANNET20_NYU40_TO_DENSE.get(
                nyu40_id
            )

            if dense_id is None:
                continue

            dense_gt[raw_gt == raw_id] = dense_id

        return dense_gt