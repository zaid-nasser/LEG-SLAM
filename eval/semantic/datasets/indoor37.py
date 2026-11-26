from __future__ import annotations

import re
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import FrameRecord
from .replica import ReplicaDataset
from .scannet import ScanNetDataset

from ..indoor37 import (
    map_nyu40_to_indoor37,
    map_replica_to_indoor37,
    map_scannet_raw_to_indoor37,
    read_scannet_id_to_nyu40,
)


def _frame_scene_name(
    frame: FrameRecord,
) -> str:
    """
    Support the current FrameRecord naming while also being tolerant
    of older versions that used `scene`.
    """

    if hasattr(
        frame,
        "scene_name",
    ):
        return str(
            frame.scene_name
        )

    if hasattr(
        frame,
        "scene",
    ):
        return str(
            frame.scene
        )

    raise AttributeError(
        "FrameRecord has neither "
        "'scene_name' nor 'scene'."
    )


def _numeric_id(
    path: Path,
) -> Optional[int]:

    matches = re.findall(
        r"\d+",
        path.stem,
    )

    if not matches:
        return None

    return int(
        matches[-1]
    )


def _indexed_pngs(
    folder: Path,
) -> dict[int, Path]:

    if not folder.is_dir():
        raise FileNotFoundError(
            folder
        )

    result: dict[int, Path] = {}

    for path in sorted(
        folder.iterdir()
    ):
        if (
            not path.is_file()
            or path.suffix.lower()
            != ".png"
        ):
            continue

        frame_id = _numeric_id(
            path
        )

        if frame_id is None:
            continue

        result[frame_id] = path

    return result


# ======================================================================
# Replica
# ======================================================================


class ReplicaIndoor37Dataset(
    ReplicaDataset
):
    """
    Replica input/frame discovery is identical to ReplicaDataset.

    Only semantic-GT decoding changes:

        Replica raw IDs
            -> NYU40
            -> Indoor37 [0..36]

    Unsupported Replica categories become 255.
    """

    def load_semantic_gt(
        self,
        frame: FrameRecord,
    ) -> np.ndarray:

        scene_name = (
            _frame_scene_name(
                frame
            )
        )

        semantic_dir = (
            self.root
            / scene_name
            / "semantic_ids"
        )

        semantic_path = (
            semantic_dir
            / (
                "semantic_id"
                f"{frame.frame_id:06d}"
                ".png"
            )
        )

        # Normally Replica uses semantic_idXXXXXX.png.
        # Keep a numeric fallback for robustness.
        if not semantic_path.is_file():

            indexed = _indexed_pngs(
                semantic_dir
            )

            semantic_path = indexed.get(
                frame.frame_id
            )

            if semantic_path is None:
                raise FileNotFoundError(
                    "Missing Replica semantic "
                    f"GT for {scene_name} "
                    f"frame {frame.frame_id}"
                )

        raw = cv2.imread(
            str(semantic_path),
            cv2.IMREAD_UNCHANGED,
        )

        if raw is None:
            raise RuntimeError(
                "Could not read Replica "
                f"semantic GT: {semantic_path}"
            )

        if raw.ndim != 2:
            raise RuntimeError(
                "Expected 2D Replica semantic "
                f"map, got {raw.shape}: "
                f"{semantic_path}"
            )

        return (
            map_replica_to_indoor37(
                raw
            )
        )


# ======================================================================
# ScanNet
# ======================================================================


class ScanNetIndoor37Dataset(
    ScanNetDataset
):
    """
    ScanNet input/frame discovery remains identical to ScanNetDataset.

    Semantic labels may be either:

        1. already in NYU40 space, or
        2. raw ScanNet category IDs.

    In case (2):

        ScanNet raw
            -> NYU40 using TSV
            -> Indoor37

    Semantic GT is NOT resized here.

    The common semantic evaluator remains responsible for matching
    GT to rendered resolution using nearest-neighbor interpolation.
    """

    NYU40_LABEL_DIR_NAMES = (
        "label_40",
        "labels_40",
        "nyu40",
        "label_nyu40",
        "labels_nyu40",
        "semantic_nyu40",
        "semantic_40",
        "semantic_class_nyu40",
    )

    RAW_LABEL_DIR_NAMES = (
        "label-filt",
        "label_filt",
        "scannet_labels",
        "semantic_scannet",
        "semantic_class",
        "semantic_classes",
    )

    AMBIGUOUS_LABEL_DIR_NAMES = (
        "label",
        "labels",
        "semantic",
        "semantics",
        "semantic_ids",
        "semantic_labels",
    )

    def __init__(
        self,
        root: Path,
        label_file: Path,
        label_space: str = "auto",
    ) -> None:

        super().__init__(
            root=root,
            label_file=label_file,
        )

        if label_space not in (
            "auto",
            "nyu40",
            "scannet",
        ):
            raise ValueError(
                "label_space must be one of "
                "{'auto', 'nyu40', 'scannet'}, "
                f"got {label_space!r}"
            )

        self.label_space = (
            label_space
        )

        self.scannet_to_nyu40 = (
            read_scannet_id_to_nyu40(
                label_file
            )
        )

        self._semantic_cache: dict[
            str,
            tuple[
                Path,
                str,
                dict[int, Path],
            ],
        ] = {}

    def _scene_root(
        self,
        scene_name: str,
    ) -> Path:

        direct = (
            self.root
            / scene_name
        )

        if direct.is_dir():
            return direct

        under_scans = (
            self.root
            / "scans"
            / scene_name
        )

        if under_scans.is_dir():
            return under_scans

        raise FileNotFoundError(
            "Could not find ScanNet scene "
            f"{scene_name!r} under "
            f"{self.root}"
        )

    def _find_named_dir(
        self,
        scene_root: Path,
        names: tuple[str, ...],
    ) -> Optional[Path]:

        # First prefer direct children.
        for name in names:

            path = (
                scene_root
                / name
            )

            if path.is_dir():
                return path

        # Then support slightly nested preprocessed layouts.
        wanted = {
            Path(name).name
            for name in names
        }

        root_depth = len(
            scene_root.parts
        )

        candidates = []

        for path in scene_root.rglob(
            "*"
        ):
            if not path.is_dir():
                continue

            depth = (
                len(path.parts)
                - root_depth
            )

            if depth > 3:
                continue

            if path.name in wanted:
                candidates.append(
                    path
                )

        if not candidates:
            return None

        candidates.sort(
            key=lambda path: (
                len(path.parts)
                - root_depth,
                str(path),
            )
        )

        return candidates[0]

    def _resolve_semantics(
        self,
        scene_name: str,
    ) -> tuple[
        Path,
        str,
        dict[int, Path],
    ]:

        cached = self._semantic_cache.get(
            scene_name
        )

        if cached is not None:
            return cached

        scene_root = self._scene_root(
            scene_name
        )

        label_dir = self._find_named_dir(
            scene_root,
            self.NYU40_LABEL_DIR_NAMES,
        )

        inferred_space = None

        if label_dir is not None:
            inferred_space = "nyu40"

        if label_dir is None:

            label_dir = self._find_named_dir(
                scene_root,
                self.RAW_LABEL_DIR_NAMES,
            )

            if label_dir is not None:
                inferred_space = (
                    "scannet"
                )

        if label_dir is None:

            label_dir = self._find_named_dir(
                scene_root,
                self.AMBIGUOUS_LABEL_DIR_NAMES,
            )

            if label_dir is not None:
                inferred_space = (
                    "ambiguous"
                )

        if label_dir is None:
            raise RuntimeError(
                "Could not locate ScanNet "
                f"semantic labels for "
                f"{scene_name}"
            )

        indexed = _indexed_pngs(
            label_dir
        )

        if not indexed:
            raise RuntimeError(
                "No semantic PNG files in "
                f"{label_dir}"
            )

        if self.label_space != "auto":

            resolved_space = (
                self.label_space
            )

        elif inferred_space in (
            "nyu40",
            "scannet",
        ):

            resolved_space = (
                inferred_space
            )

        else:
            # Ambiguous folder name.
            #
            # Raw ScanNet IDs may be > 40.
            # If every sampled ID is <=40 we do NOT guess,
            # because both spaces are then possible.

            observed_max = -1

            for path in list(
                indexed.values()
            )[:20]:

                arr = cv2.imread(
                    str(path),
                    cv2.IMREAD_UNCHANGED,
                )

                if (
                    arr is not None
                    and arr.size > 0
                ):
                    observed_max = max(
                        observed_max,
                        int(arr.max()),
                    )

            if observed_max > 40:
                resolved_space = (
                    "scannet"
                )

            else:
                raise RuntimeError(
                    "Ambiguous ScanNet semantic "
                    f"label space in {label_dir}. "
                    f"Observed max ID="
                    f"{observed_max}. "
                    "Specify label_space='nyu40' "
                    "or label_space='scannet'."
                )

        result = (
            label_dir,
            resolved_space,
            indexed,
        )

        self._semantic_cache[
            scene_name
        ] = result

        print(
            f"{scene_name}: Indoor37 GT "
            f"from {label_dir} "
            f"(source space="
            f"{resolved_space})"
        )

        return result

    def load_semantic_gt(
        self,
        frame: FrameRecord,
    ) -> np.ndarray:

        scene_name = (
            _frame_scene_name(
                frame
            )
        )

        (
            label_dir,
            label_space,
            indexed,
        ) = self._resolve_semantics(
            scene_name
        )

        label_path = indexed.get(
            frame.frame_id
        )

        if label_path is None:
            raise FileNotFoundError(
                "Missing ScanNet semantic "
                f"GT for {scene_name} "
                f"frame {frame.frame_id} "
                f"in {label_dir}"
            )

        raw = cv2.imread(
            str(label_path),
            cv2.IMREAD_UNCHANGED,
        )

        if raw is None:
            raise RuntimeError(
                "Could not read ScanNet "
                f"semantic GT: {label_path}"
            )

        if raw.ndim != 2:
            raise RuntimeError(
                "Expected 2D ScanNet semantic "
                f"map, got {raw.shape}: "
                f"{label_path}"
            )

        if label_space == "nyu40":

            return (
                map_nyu40_to_indoor37(
                    raw
                )
            )

        if label_space == "scannet":

            return (
                map_scannet_raw_to_indoor37(
                    raw,
                    self.scannet_to_nyu40,
                )
            )

        raise RuntimeError(
            "Unexpected ScanNet "
            f"label space: {label_space}"
        )