from __future__ import annotations

from typing import Mapping, Sequence

import torch

from .types import (
    AffinityEdge,
    LocalAffinity,
)


class DepthAffinityProvider:
    """
    Build local affinities from a depth map.

    For every dilation d we use one undirected half
    of the corresponding 3x3 neighborhood:

        (0, +d)
        (+d, -d)
        (+d,  0)
        (+d, +d)

    Since the refiner applies every edge in both
    directions, this gives 8 neighbors per dilation.

    Example:
        dilations=(1, 2, 4)

    gives up to:
        8 * 3 = 24 neighbors per interior pixel.
    """

    def __init__(
        self,
        sigma_d: float,
        dilations: Sequence[int] = (
            1,
            2,
            4,
        ),
        depth_key: str = "depth",
    ) -> None:

        if sigma_d <= 0:
            raise ValueError(
                "sigma_d must be > 0"
            )

        if not dilations:
            raise ValueError(
                "dilations must not be empty"
            )

        parsed_dilations = tuple(
            int(d)
            for d in dilations
        )

        if any(
            d <= 0
            for d in parsed_dilations
        ):
            raise ValueError(
                "All dilations must be > 0"
            )

        if len(set(parsed_dilations)) != len(
            parsed_dilations
        ):
            raise ValueError(
                "Dilations must be unique"
            )

        self.sigma_d = float(
            sigma_d
        )

        self.dilations = (
            parsed_dilations
        )

        self.depth_key = depth_key

    @staticmethod
    def _paired_slices(
        height: int,
        width: int,
        dy: int,
        dx: int,
    ):
        """
        Return source and destination slices for:

            dst = src + (dy, dx)
        """

        if dy >= 0:
            src_y = slice(
                0,
                height - dy,
            )

            dst_y = slice(
                dy,
                height,
            )

        else:
            src_y = slice(
                -dy,
                height,
            )

            dst_y = slice(
                0,
                height + dy,
            )

        if dx >= 0:
            src_x = slice(
                0,
                width - dx,
            )

            dst_x = slice(
                dx,
                width,
            )

        else:
            src_x = slice(
                -dx,
                width,
            )

            dst_x = slice(
                0,
                width + dx,
            )

        return (
            src_y,
            src_x,
            dst_y,
            dst_x,
        )

    def _offsets(
        self,
    ) -> tuple[tuple[int, int], ...]:

        offsets = []

        for dilation in self.dilations:

            offsets.extend(
                [
                    (0, dilation),
                    (
                        dilation,
                        -dilation,
                    ),
                    (
                        dilation,
                        0,
                    ),
                    (
                        dilation,
                        dilation,
                    ),
                ]
            )

        return tuple(offsets)

    def compute(
        self,
        resources: Mapping[str, torch.Tensor],
    ) -> LocalAffinity:

        if self.depth_key not in resources:
            raise KeyError(
                f"Missing affinity resource: "
                f"{self.depth_key!r}"
            )

        depth = resources[
            self.depth_key
        ]

        if depth.ndim != 2:
            raise ValueError(
                "DepthAffinityProvider expects "
                f"depth [H,W], got "
                f"{tuple(depth.shape)}"
            )

        if not torch.is_floating_point(
            depth
        ):
            depth = depth.float()

        height, width = depth.shape

        valid = (
            torch.isfinite(depth)
            & (depth > 0)
        )

        edges = []

        for dy, dx in self._offsets():

            # A large dilation may not fit a tiny
            # synthetic/test image. Skip that edge.
            if (
                abs(dy) >= height
                or abs(dx) >= width
            ):
                continue

            (
                src_y,
                src_x,
                dst_y,
                dst_x,
            ) = self._paired_slices(
                height=height,
                width=width,
                dy=dy,
                dx=dx,
            )

            depth_src = depth[
                src_y,
                src_x,
            ]

            depth_dst = depth[
                dst_y,
                dst_x,
            ]

            edge_valid = (
                valid[
                    src_y,
                    src_x,
                ]
                & valid[
                    dst_y,
                    dst_x,
                ]
            )

            depth_diff = (
                depth_src
                - depth_dst
            )

            weights = torch.exp(
                -0.5
                * (
                    depth_diff
                    / self.sigma_d
                ).square()
            )

            weights = torch.where(
                edge_valid,
                weights,
                torch.zeros_like(
                    weights
                ),
            )

            edges.append(
                AffinityEdge(
                    dy=dy,
                    dx=dx,
                    weights=weights,
                )
            )

        affinity = LocalAffinity(
            edges=tuple(edges)
        )

        affinity.validate(
            height=height,
            width=width,
        )

        return affinity