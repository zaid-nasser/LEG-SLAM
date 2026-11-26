from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence


@dataclass(frozen=True)
class ReconstructionProtocol:
    name: str
    dataset: str
    scenes: Sequence[str]
    frame_stride: int = 1


REPLICA8 = ReconstructionProtocol(
    name="replica8",
    dataset="replica",
    scenes=(
        "office0",
        "office1",
        "office2",
        "office3",
        "office4",
        "room0",
        "room1",
        "room2",
    ),
)


SCANNET12 = ReconstructionProtocol(
    name="scannet12",
    dataset="scannet",
    scenes=(
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
    ),
)


RECONSTRUCTION_PROTOCOLS = {
    REPLICA8.name: REPLICA8,
    SCANNET12.name: SCANNET12,
}