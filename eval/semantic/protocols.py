from dataclasses import dataclass
from typing import Sequence


REPLICA_SCENES = [
    "office0",
    "office1",
    "office2",
    "office3",
    "office4",
    "room0",
    "room1",
    "room2",
]


SCANNET_SCENES = [
    "scene0050_02",
    "scene0144_01",
    "scene0221_01",
    "scene0300_01",
    "scene0354_00",
    "scene0389_00",
    "scene0423_02",
    "scene0427_00",
    "scene0494_00",
    "scene0616_00",
    "scene0645_02",
    "scene0693_00",
]


@dataclass(frozen=True)
class EvaluationProtocol:
    name: str
    dataset: str
    scenes: Sequence[str]
    num_classes: int
    frame_stride: int

REPLICA51 = EvaluationProtocol(
    name = "replica51",
    dataset = "replica",
    scenes = REPLICA_SCENES,
    num_classes = 51,
    frame_stride = 1,
)
# , "office1", "office2", "office3", "office4",
#         "room0", "room1", "room2",
SCANNET12_20 = EvaluationProtocol(
    name = "scannet12_20",
    dataset = "scannet",
    scenes = SCANNET_SCENES,
    num_classes = 20,
    frame_stride = 1,
)


INDOOR37_REPLICA = EvaluationProtocol(
    name = "indoor37_replica",
    dataset = "replica",
    scenes = REPLICA_SCENES,
    num_classes = 37,
    frame_stride = 1,
)


INDOOR37_SCANNET = EvaluationProtocol(
    name = "indoor37_scannet",
    dataset = "scannet",
    scenes = SCANNET_SCENES,
    num_classes = 37,
    frame_stride = 1,
)


PROTOCOLS = {
    REPLICA51.name: REPLICA51,
    SCANNET12_20.name: SCANNET12_20,
    INDOOR37_REPLICA.name: INDOOR37_REPLICA,
    INDOOR37_SCANNET.name: INDOOR37_SCANNET,
}