from __future__ import annotations

import torch

from .types import LocalAffinity


def _paired_slices(
    height: int,
    width: int,
    dy: int,
    dx: int,
):
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


class IterativeScoreRefiner:

    def __init__(
        self,
        lambda_refine: float = 1.0,
        num_iterations: int = 10,
    ) -> None:

        if lambda_refine < 0:
            raise ValueError(
                "lambda_refine must be >= 0"
            )

        if num_iterations < 0:
            raise ValueError(
                "num_iterations must be >= 0"
            )

        self.lambda_refine = float(
            lambda_refine
        )

        self.num_iterations = int(
            num_iterations
        )

    def refine(
        self,
        scores: torch.Tensor,
        affinity: LocalAffinity,
    ) -> torch.Tensor:

        if scores.ndim not in (2, 3):
            raise ValueError(
                "Expected scores [H,W] or "
                f"[C,H,W], got "
                f"{tuple(scores.shape)}"
            )

        if not torch.is_floating_point(
            scores
        ):
            raise TypeError(
                "Semantic scores must be "
                "floating-point tensors"
            )

        single_channel = (
            scores.ndim == 2
        )

        if single_channel:
            scores_c = (
                scores.unsqueeze(0)
            )
        else:
            scores_c = scores

        _, height, width = (
            scores_c.shape
        )

        affinity.validate(
            height=height,
            width=width,
        )

        original = scores_c

        if (
            self.num_iterations == 0
            or self.lambda_refine == 0.0
        ):
            result = original.clone()

            return (
                result.squeeze(0)
                if single_channel
                else result
            )

        weight_sum = torch.zeros(
            (height, width),
            dtype=scores.dtype,
            device=scores.device,
        )

        prepared_edges = []

        # ----------------------------------------------------------
        # Prepare every edge once.
        # ----------------------------------------------------------

        for edge in affinity.edges:

            weights = edge.weights.to(
                device=scores.device,
                dtype=scores.dtype,
            )

            (
                src_y,
                src_x,
                dst_y,
                dst_x,
            ) = _paired_slices(
                height=height,
                width=width,
                dy=edge.dy,
                dx=edge.dx,
            )

            # Every undirected edge contributes to
            # the denominator of both endpoints.
            weight_sum[
                src_y,
                src_x,
            ] += weights

            weight_sum[
                dst_y,
                dst_x,
            ] += weights

            prepared_edges.append(
                (
                    weights,
                    src_y,
                    src_x,
                    dst_y,
                    dst_x,
                )
            )

        denominator = (
            1.0
            + self.lambda_refine
            * weight_sum
        )

        x = original.clone()

        # ----------------------------------------------------------
        # Jacobi iterations
        # ----------------------------------------------------------

        for _ in range(
            self.num_iterations
        ):

            weighted_sum = (
                torch.zeros_like(x)
            )

            for (
                weights,
                src_y,
                src_x,
                dst_y,
                dst_x,
            ) in prepared_edges:

                weights_c = (
                    weights.unsqueeze(0)
                )

                # Source receives destination.
                weighted_sum[
                    :,
                    src_y,
                    src_x,
                ] += (
                    weights_c
                    * x[
                        :,
                        dst_y,
                        dst_x,
                    ]
                )

                # Destination receives source.
                weighted_sum[
                    :,
                    dst_y,
                    dst_x,
                ] += (
                    weights_c
                    * x[
                        :,
                        src_y,
                        src_x,
                    ]
                )

            x = (
                original
                + self.lambda_refine
                * weighted_sum
            ) / denominator.unsqueeze(0)

        if single_channel:
            return x.squeeze(0)

        return x