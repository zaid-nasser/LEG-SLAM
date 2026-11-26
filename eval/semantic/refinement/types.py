from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Protocol

import torch


@dataclass(frozen=True)
class AffinityEdge:
    """
    Affinity for one undirected spatial offset.

    The offset (dy, dx) connects:

        (y, x) <-> (y + dy, x + dx)

    Only one direction of each undirected edge is stored.
    The score refiner propagates through it in both directions.
    """

    dy: int
    dx: int
    weights: torch.Tensor

    def validate(
        self,
        height: int,
        width: int,
    ) -> None:

        if self.dy == 0 and self.dx == 0:
            raise ValueError(
                "Affinity edge cannot have zero offset"
            )

        if self.weights.ndim != 2:
            raise ValueError(
                "Affinity weights must be 2-D, "
                f"got {tuple(self.weights.shape)}"
            )

        expected_height = (
            height - abs(self.dy)
        )

        expected_width = (
            width - abs(self.dx)
        )

        if (
            expected_height <= 0
            or expected_width <= 0
        ):
            raise ValueError(
                "Affinity offset does not fit image: "
                f"offset=({self.dy}, {self.dx}), "
                f"image=({height}, {width})"
            )

        expected_shape = (
            expected_height,
            expected_width,
        )

        if tuple(self.weights.shape) != expected_shape:
            raise ValueError(
                "Invalid affinity shape for "
                f"offset=({self.dy}, {self.dx}): "
                f"expected {expected_shape}, "
                f"got {tuple(self.weights.shape)}"
            )


@dataclass(frozen=True)
class LocalAffinity:
    """
    Collection of undirected local affinity edges.

    Each edge stores only one spatial direction.
    The refiner automatically applies it in both directions.
    """

    edges: tuple[AffinityEdge, ...]

    def validate(
        self,
        height: int,
        width: int,
    ) -> None:

        seen_offsets = set()

        for edge in self.edges:

            offset = (
                edge.dy,
                edge.dx,
            )

            if offset in seen_offsets:
                raise ValueError(
                    f"Duplicate affinity offset: {offset}"
                )

            seen_offsets.add(offset)

            edge.validate(
                height=height,
                width=width,
            )


class AffinityProvider(Protocol):

    def compute(
        self,
        resources: Mapping[str, torch.Tensor],
    ) -> LocalAffinity:
        ...