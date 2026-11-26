from __future__ import annotations

import csv
from pathlib import Path
from typing import Mapping

import numpy as np


IGNORE_INDEX = 255


INDOOR37_CLASS_NAMES = (
    "wall",
    "floor",
    "cabinet",
    "bed",
    "chair",
    "sofa",
    "table",
    "door",
    "window",
    "bookshelf",
    "picture",
    "counter",
    "blinds",
    "desk",
    "shelves",
    "curtain",
    "dresser",
    "pillow",
    "mirror",
    "floor_mat",
    "clothes",
    "ceiling",
    "books",
    "fridge",
    "tv",
    "paper",
    "towel",
    "shower_curtain",
    "box",
    "whiteboard",
    "person",
    "night_stand",
    "toilet",
    "sink",
    "lamp",
    "bathtub",
    "bag",
)


NUM_INDOOR37_CLASSES = len(
    INDOOR37_CLASS_NAMES
)


if NUM_INDOOR37_CLASSES != 37:
    raise RuntimeError(
        "Indoor37 taxonomy must contain exactly "
        f"37 classes, got {NUM_INDOOR37_CLASSES}"
    )


# ----------------------------------------------------------------------
# Replica -> NYU40
# ----------------------------------------------------------------------
#
# Conservative explicit overlap used by the original direct-probe
# evaluator.
#
# Values are NYU40 IDs.
#
# Anything not present here is ignored during Indoor37 evaluation.
#
# In particular:
#
#     Replica 98 "rug"
#
# is deliberately NOT mapped to floor/floor_mat. Its LabelMaker/NYU40
# correspondence is otherprop (40), outside Indoor37.
# ----------------------------------------------------------------------

REPLICA_TO_NYU40 = {
    2: 3,      # base_cabinet -> cabinet
    4: 36,     # bathtub -> bathtub
    7: 4,      # bed -> bed
    12: 13,    # blinds -> blinds
    13: 23,    # book -> books
    15: 29,    # box -> box
    18: 3,     # cabinet -> cabinet
    20: 5,     # chair -> chair
    24: 21,    # clothing -> clothes
    29: 18,    # cushion -> pillow
    30: 16,    # curtain -> curtain
    31: 22,    # ceiling -> ceiling
    33: 12,    # countertop -> counter
    34: 14,    # desk -> desk
    37: 8,     # door -> door
    40: 2,     # floor -> floor
    41: 37,    # handbag -> bag
    47: 35,    # lamp -> lamp
    50: 20,    # mat -> floor_mat
    54: 32,    # nightstand -> night_stand
    59: 11,    # picture -> picture
    61: 18,    # pillow -> pillow
    67: 24,    # refrigerator -> fridge
    71: 15,    # shelf -> shelves
    74: 34,    # sink -> sink
    76: 6,     # sofa -> sofa
    80: 7,     # table -> table
    84: 33,    # toilet -> toilet
    86: 27,    # towel -> towel
    87: 25,    # tv_screen -> tv
    93: 1,     # wall -> wall
    94: 3,     # wall_cabinet -> cabinet
    97: 9,     # window -> window
    100: 37,   # bag -> bag
    101: 21,   # set_of_clothing -> clothes
}


def map_nyu40_to_indoor37(
    nyu40: np.ndarray,
) -> np.ndarray:
    """
    Map NYU40 semantic IDs into the probe label space.

        NYU40 1  -> Indoor37 0
        NYU40 2  -> Indoor37 1
        ...
        NYU40 37 -> Indoor37 36

    NYU40 IDs:
        0, 38, 39, 40
    and any unexpected values are ignored.
    """

    nyu40 = np.asarray(
        nyu40,
        dtype=np.int64,
    )

    out = np.full(
        nyu40.shape,
        IGNORE_INDEX,
        dtype=np.uint8,
    )

    valid = (
        (nyu40 >= 1)
        & (nyu40 <= 37)
    )

    out[valid] = (
        nyu40[valid] - 1
    ).astype(np.uint8)

    return out


def _map_raw_to_nyu40(
    raw: np.ndarray,
    mapping: Mapping[int, int],
) -> np.ndarray:
    """
    Generic integer-ID -> NYU40 mapping using a LUT.

    Unknown raw IDs map to NYU40 0 and are therefore ignored when
    converted to Indoor37.
    """

    raw = np.asarray(
        raw,
        dtype=np.int64,
    )

    if raw.size == 0:
        return np.zeros(
            raw.shape,
            dtype=np.int16,
        )

    max_raw = int(raw.max())
    max_mapping_id = max(
        mapping.keys(),
        default=0,
    )

    lut_size = (
        max(
            max_raw,
            max_mapping_id,
        )
        + 1
    )

    nyu40_lut = np.zeros(
        lut_size,
        dtype=np.int16,
    )

    for raw_id, nyu40_id in mapping.items():
        if 0 <= raw_id < lut_size:
            nyu40_lut[raw_id] = int(
                nyu40_id
            )

    nyu40 = np.zeros(
        raw.shape,
        dtype=np.int16,
    )

    valid_raw = (
        (raw >= 0)
        & (raw < lut_size)
    )

    nyu40[valid_raw] = nyu40_lut[
        raw[valid_raw]
    ]

    return nyu40


def map_replica_to_indoor37(
    raw_replica: np.ndarray,
) -> np.ndarray:
    """
    Replica native semantic IDs -> Indoor37 IDs.
    """

    nyu40 = _map_raw_to_nyu40(
        raw_replica,
        REPLICA_TO_NYU40,
    )

    return map_nyu40_to_indoor37(
        nyu40
    )


def read_scannet_id_to_nyu40(
    tsv_path: Path,
) -> dict[int, int]:
    """
    Read:

        ScanNet raw category ID -> NYU40 ID

    from scannetv2-labels.combined.tsv.
    """

    tsv_path = Path(tsv_path)

    if not tsv_path.is_file():
        raise FileNotFoundError(
            tsv_path
        )

    mapping: dict[int, int] = {}

    with tsv_path.open(
        "r",
        newline="",
        encoding="utf-8-sig",
    ) as f:

        reader = csv.DictReader(
            f,
            delimiter="\t",
        )

        fieldnames = set(
            reader.fieldnames or []
        )

        if "id" not in fieldnames:
            raise RuntimeError(
                f"No 'id' column in {tsv_path}"
            )

        if "nyu40id" not in fieldnames:
            raise RuntimeError(
                f"No 'nyu40id' column in {tsv_path}"
            )

        for row in reader:
            try:
                raw_id = int(
                    row["id"]
                )

                nyu40_id = int(
                    row["nyu40id"]
                )

            except (
                TypeError,
                ValueError,
            ):
                continue

            mapping[raw_id] = (
                nyu40_id
            )

    if not mapping:
        raise RuntimeError(
            "No ScanNet ID mappings were read "
            f"from {tsv_path}"
        )

    return mapping


def map_scannet_raw_to_indoor37(
    raw_scannet: np.ndarray,
    scannet_to_nyu40: Mapping[int, int],
) -> np.ndarray:
    """
    Raw ScanNet category IDs -> NYU40 -> Indoor37.
    """

    nyu40 = _map_raw_to_nyu40(
        raw_scannet,
        scannet_to_nyu40,
    )

    return map_nyu40_to_indoor37(
        nyu40
    )