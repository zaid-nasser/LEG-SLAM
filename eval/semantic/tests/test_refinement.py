import torch

from eval.semantic.refinement import (
    DepthAffinityProvider,
    IterativeScoreRefiner,
    LocalAffinity,
    AffinityEdge,
)


def test_multiscale_depth_affinity_has_expected_offsets():

    depth = torch.ones(
        10,
        12,
    )

    provider = DepthAffinityProvider(
        sigma_d=0.05,
        dilations=(1, 2, 4),
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    offsets = {
        (edge.dy, edge.dx)
        for edge in affinity.edges
    }

    assert offsets == {
        (0, 1),
        (1, -1),
        (1, 0),
        (1, 1),

        (0, 2),
        (2, -2),
        (2, 0),
        (2, 2),

        (0, 4),
        (4, -4),
        (4, 0),
        (4, 4),
    }

    assert len(
        affinity.edges
    ) == 12

def test_diagonal_neighbor_propagation():

    scores = torch.zeros(
        1,
        3,
        3,
    )

    scores[
        0,
        0,
        0,
    ] = 1.0

    depth = torch.ones(
        3,
        3,
    )

    provider = DepthAffinityProvider(
        sigma_d=0.05,
        dilations=(1,),
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=1,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert (
        refined[
            0,
            1,
            1,
        ]
        > 0
    )

def test_dilated_neighbor_directly_propagates():

    scores = torch.zeros(
        1,
        1,
        9,
    )

    scores[
        0,
        0,
        0,
    ] = 1.0

    depth = torch.ones(
        1,
        9,
    )

    provider = DepthAffinityProvider(
        sigma_d=0.05,
        dilations=(4,),
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=1,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    # Pixel 4 is directly connected to pixel 0.
    assert (
        refined[
            0,
            0,
            4,
        ]
        > 0
    )

    # Pixel 1 has no radius-4 edge to pixel 0.
    assert (
        refined[
            0,
            0,
            1,
        ]
        == 0
    )

def test_constant_depth_has_unit_affinity():

    depth = torch.full(
        (3, 4),
        2.0,
    )

    provider = DepthAffinityProvider(
        sigma_d=0.05
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    assert torch.allclose(
        affinity.horizontal,
        torch.ones(3, 3),
    )

    assert torch.allclose(
        affinity.vertical,
        torch.ones(2, 4),
    )


def test_depth_discontinuity_reduces_affinity():

    depth = torch.tensor(
        [
            [1.0, 1.0, 2.0],
        ]
    )

    provider = DepthAffinityProvider(
        sigma_d=0.05
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    # Equal-depth edge.
    assert torch.isclose(
        affinity.horizontal[0, 0],
        torch.tensor(1.0),
    )

    # 1-meter discontinuity with sigma=5cm
    # should be effectively disconnected.
    assert affinity.horizontal[0, 1] < 1e-6


def test_invalid_depth_has_zero_affinity():

    depth = torch.tensor(
        [
            [1.0, 0.0, 1.0],
            [1.0, float("nan"), 1.0],
        ]
    )

    provider = DepthAffinityProvider(
        sigma_d=0.05
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    assert torch.all(
        affinity.horizontal == 0
    )

    # Left/right vertical edges remain valid.
    assert affinity.vertical[0, 0] == 1
    assert affinity.vertical[0, 2] == 1

    # Middle vertical edge contains invalid depth.
    assert affinity.vertical[0, 1] == 0


def test_lambda_zero_returns_original():

    scores = torch.randn(
        4,
        5,
        6,
    )

    affinity = LocalAffinity(
        horizontal=torch.ones(5, 5),
        vertical=torch.ones(4, 6),
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=0.0,
        num_iterations=10,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert torch.equal(
        refined,
        scores,
    )


def test_zero_affinity_returns_original():

    scores = torch.randn(
        3,
        4,
        5,
    )

    affinity = LocalAffinity(
        horizontal=torch.zeros(4, 4),
        vertical=torch.zeros(3, 5),
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=10,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert torch.allclose(
        refined,
        scores,
    )


def test_constant_scores_remain_constant():

    scores = torch.full(
        (2, 4, 5),
        0.7,
    )

    affinity = LocalAffinity(
        horizontal=torch.ones(4, 4),
        vertical=torch.ones(3, 5),
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=10,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert torch.allclose(
        refined,
        scores,
        atol=1e-6,
    )


def test_score_spreads_inside_connected_region():

    scores = torch.zeros(
        1,
        1,
        5,
    )

    scores[0, 0, 2] = 1.0

    affinity = LocalAffinity(
        horizontal=torch.ones(1, 4),
        vertical=torch.empty(0, 5),
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=10,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert refined[0, 0, 1] > 0
    assert refined[0, 0, 3] > 0


def test_depth_edge_blocks_propagation():

    scores = torch.zeros(
        1,
        1,
        4,
    )

    scores[0, 0, 0] = 1.0

    depth = torch.tensor(
        [
            [1.0, 1.0, 2.0, 2.0],
        ]
    )

    provider = DepthAffinityProvider(
        sigma_d=0.01
    )

    affinity = provider.compute(
        {"depth": depth}
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=20,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    # Propagates within first depth surface.
    assert refined[0, 0, 1] > 0

    # Does not cross the 1m depth discontinuity.
    assert refined[0, 0, 2] < 1e-6
    assert refined[0, 0, 3] < 1e-6


def test_channels_are_refined_independently():

    scores = torch.zeros(
        2,
        1,
        3,
    )

    scores[0, 0, 0] = 1.0
    scores[1, 0, 2] = 1.0

    affinity = LocalAffinity(
        horizontal=torch.ones(1, 2),
        vertical=torch.empty(0, 3),
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=5,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert refined.shape == scores.shape

    assert refined[0, 0, 1] > 0
    assert refined[1, 0, 1] > 0

    # Refinement must not mix semantic channels.
    assert refined[0, 0, 0] > refined[1, 0, 0]
    assert refined[1, 0, 2] > refined[0, 0, 2]


def test_dtype_is_preserved():

    scores = torch.randn(
        2,
        3,
        4,
        dtype=torch.float64,
    )

    affinity = LocalAffinity(
        horizontal=torch.ones(
            3,
            3,
            dtype=torch.float32,
        ),
        vertical=torch.ones(
            2,
            4,
            dtype=torch.float32,
        ),
    )

    refiner = IterativeScoreRefiner(
        lambda_refine=1.0,
        num_iterations=2,
    )

    refined = refiner.refine(
        scores,
        affinity,
    )

    assert refined.dtype == scores.dtype