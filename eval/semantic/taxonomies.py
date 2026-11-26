from dataclasses import dataclass
from typing import Sequence, Mapping, List, Dict
import numpy as np

from .indoor37 import (
    INDOOR37_CLASS_NAMES,
)


REPLICA_RAW_CLASSES: List[str] = [
    "other",
    "backpack",
    "base-cabinet",
    "basket",
    "bathtub",
    "beam",
    "beanbag",
    "bed",
    "bench",
    "bike",
    "bin",
    "blanket",
    "blinds",
    "book",
    "bottle",
    "box",
    "bowl",
    "camera",
    "cabinet",
    "candle",
    "chair",
    "chopping-board",
    "clock",
    "cloth",
    "clothing",
    "coaster",
    "comforter",
    "computer-keyboard",
    "cup",
    "cushion",
    "curtain",
    "ceiling",
    "cooktop",
    "countertop",
    "desk",
    "desk-organizer",
    "desktop-computer",
    "door",
    "exercise-ball",
    "faucet",
    "floor",
    "handbag",
    "hair-dryer",
    "handrail",
    "indoor-plant",
    "knife-block",
    "kitchen-utensil",
    "lamp",
    "laptop",
    "major-appliance",
    "mat",
    "microwave",
    "monitor",
    "mouse",
    "nightstand",
    "pan",
    "panel",
    "paper-towel",
    "phone",
    "picture",
    "pillar",
    "pillow",
    "pipe",
    "plant-stand",
    "plate",
    "pot",
    "rack",
    "refrigerator",
    "remote-control",
    "scarf",
    "sculpture",
    "shelf",
    "shoe",
    "shower-stall",
    "sink",
    "small-appliance",
    "sofa",
    "stair",
    "stool",
    "switch",
    "table",
    "table-runner",
    "tablet",
    "tissue-paper",
    "toilet",
    "toothbrush",
    "towel",
    "tv-screen",
    "tv-stand",
    "umbrella",
    "utensil-holder",
    "vase",
    "vent",
    "wall",
    "wall-cabinet",
    "wall-plug",
    "wardrobe",
    "window",
    "rug",
    "logo",
    "bag",
    "set-of-clothing",
]


REPLICA51_CLASSES: List[str] = [
    # Head (17)
    "wall",
    "ceiling",
    "floor",
    "chair",
    "blinds",
    "sofa",
    "table",
    "rug",
    "window",
    "lamp",
    "door",
    "pillow",
    "bench",
    "tv-screen",
    "cabinet",
    "pillar",
    "blanket",
    # Common (17)
    "tv-stand",
    "cushion",
    "bin",
    "vent",
    "bed",
    "stool",
    "picture",
    "indoor-plant",
    "desk",
    "comforter",
    "nightstand",
    "shelf",
    "vase",
    "plant-stand",
    "basket",
    "plate",
    "monitor",
    # Tail (17)
    "pipe",
    "panel",
    "desk-organizer",
    "wall-plug",
    "book",
    "box",
    "clock",
    "sculpture",
    "tissue-paper",
    "camera",
    "tablet",
    "pot",
    "bottle",
    "candle",
    "bowl",
    "cloth",
    "switch",
]

REPLICA51_SPLITS = {
    "head": np.arange(0, 17, dtype=np.int64),
    "common": np.arange(17, 34, dtype=np.int64),
    "tail": np.arange(34, 51, dtype=np.int64),
}

SCANNET20_CLASSES = [
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
    "desk",
    "curtain",
    "refrigerator",
    "shower curtain",
    "toilet",
    "sink",
    "bathtub",
    "otherfurniture",
]

SCANNET20_NYU40_IDS = [
    1,   # wall
    2,   # floor
    3,   # cabinet
    4,   # bed
    5,   # chair
    6,   # sofa
    7,   # table
    8,   # door
    9,   # window
    10,  # bookshelf
    11,  # picture
    12,  # counter
    14,  # desk
    16,  # curtain
    24,  # refrigerator
    28,  # shower curtain
    33,  # toilet
    34,  # sink
    36,  # bathtub
    39,  # otherfurniture
]


@dataclass(frozen=True)
class Taxonomy:
    name: str
    class_names: Sequence[str]
    ignore_index: int = 255

REPLICA51_TAXONOMY = Taxonomy(
    name="replica51",
    class_names=REPLICA51_CLASSES,
)

RAW_ID_TO_DENSE51: Dict[int, int] = {
    REPLICA_RAW_CLASSES.index(name): dense_id
    for dense_id, name in enumerate(REPLICA51_CLASSES)
}
DENSE51_TO_RAW_ID: List[int] = [REPLICA_RAW_CLASSES.index(name) for name in REPLICA51_CLASSES]

SCANNET20_TAXONOMY = Taxonomy(
    name="scannet20",
    class_names=SCANNET20_CLASSES,
)

INDOOR37_TAXONOMY = Taxonomy(
    name="indoor37",
    class_names=INDOOR37_CLASS_NAMES,
    ignore_index=255,
)

TAXONOMIES = {
    REPLICA51_TAXONOMY.name: REPLICA51_TAXONOMY,
    SCANNET20_TAXONOMY.name: SCANNET20_TAXONOMY,
    INDOOR37_TAXONOMY.name: INDOOR37_TAXONOMY,
}

SCANNET20_NYU40_TO_DENSE = {
    nyu40_id: dense_id
    for dense_id, nyu40_id in enumerate(SCANNET20_NYU40_IDS)
}