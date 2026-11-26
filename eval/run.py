from __future__ import annotations

import argparse
from dataclasses import replace
from pathlib import Path

import torch

from .semantic.datasets.replica import ReplicaDataset
from .semantic.datasets.scannet import ScanNetDataset
from .semantic.evaluator import evaluate_protocol as evaluate_semantic_protocol
from .semantic.predictors.factory import build_semantic_predictor
from .semantic.protocols import PROTOCOLS as SEMANTIC_PROTOCOLS
from .semantic.taxonomies import (
    REPLICA51_SPLITS,
    REPLICA51_TAXONOMY,
    SCANNET20_TAXONOMY,
)

from .reconstruction.datasets.replica import ReplicaReconstructionDataset
from .reconstruction.datasets.scannet import ScanNetReconstructionDataset
from .reconstruction.evaluator import evaluate_dataset
from .reconstruction.protocols import RECONSTRUCTION_PROTOCOLS
from .semantic.refinement import (
    DepthAffinityProvider,
    IterativeScoreRefiner,
    RefinementPipeline,
)

from .semantic.datasets.indoor37 import (
    ReplicaIndoor37Dataset,
    ScanNetIndoor37Dataset,
)

from .semantic.taxonomies import (
    INDOOR37_TAXONOMY,
)

from .semantic.predictors.factory import (
    build_radio_probe_predictor,
)

def _default_device() -> str:
    return (
        "cuda"
        if torch.cuda.is_available()
        else "cpu"
    )


def _format_metric(
    value: float | None,
    digits: int = 4,
) -> str:
    if value is None:
        return "N/A"

    return f"{value:.{digits}f}"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m eval",
        description="LEG-SLAM evaluation",
    )

    subparsers = parser.add_subparsers(
        dest="task",
        required=True,
    )

    # ==============================================================
    # Semantic evaluation
    # ==============================================================

    semantic = subparsers.add_parser(
        "semantic",
        help="Evaluate open-vocabulary semantic segmentation",
    )

    semantic.add_argument(
        "--protocol",
        required=True,
        choices=sorted(
            SEMANTIC_PROTOCOLS.keys()
        ),
    )

    semantic.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    semantic.add_argument(
        "--recon-root",
        type=Path,
        required=True,
    )

    semantic.add_argument(
        "--feature-model",
        required=True,
    )

    semantic.add_argument(
        "--feature-registry",
        type=Path,
        default=Path(
            "cfg/feature_models.json"
        ),
    )

    semantic.add_argument(
        "--scannet-label-file",
        type=Path,
        default=None,
    )

    semantic.add_argument(
        "--out-root",
        type=Path,
        required=True,
    )

    semantic.add_argument(
        "--device",
        default=_default_device(),
    )

    semantic.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Optional scene override. "
            "By default the protocol scene list is used."
        ),
    )

    semantic.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "Optional frame-stride override. "
            "By default the protocol value is used."
        ),
    )

    semantic.add_argument(
        "--save-masks",
        action="store_true",
        help=(
            "Save predicted and ground-truth semantic ID maps "
            "and colored visualizations."
        ),
    )


    semantic.add_argument(
        "--semantic-refinement",
        choices=[
            "none",
            "iterative",
            "components",
        ],
        default="none",
        help=(
            "Optional continuous semantic-score refinement."
        ),
    )

    semantic.add_argument(
        "--affinity-source",
        choices=[
            "depth",
        ],
        default="depth",
        help=(
            "Resource used to construct local refinement affinity."
        ),
    )

    semantic.add_argument(
        "--refine-iterations",
        type=int,
        default=10,
    )

    semantic.add_argument(
        "--refine-lambda",
        type=float,
        default=0.5,
    )

    semantic.add_argument(
        "--depth-sigma",
        type=float,
        default=0.05,
        help=(
            "Depth-affinity Gaussian sigma in meters."
        ),
    )

    semantic.add_argument(
        "--affinity-dilations",
        type=int,
        nargs="+",
        default=[
            1,
            2,
            4,
        ],
        help=(
            "Pixel dilations used to construct the "
            "local affinity neighborhood. "
            "Each dilation contributes 8 neighbors."
        ),
    )

    semantic.add_argument(
        "--probe-checkpoint",
        type=Path,
        default=None,
        help=(
            "Checkpoint for the RADIO Indoor37 linear probe. "
            "Required by indoor37_* protocols and by "
            "--radio-inference dual."
        ),
    )

    semantic.add_argument(
        "--radio-inference",
        choices=(
            "open_vocab",
            "dual",
        ),
        default="open_vocab",
        help=(
            "RADIO semantic inference mode. open_vocab preserves the "
            "existing SigLIP query-only predictor. dual fuses the open-"
            "vocabulary distribution with Indoor37 closed-set support."
        ),
    )

    semantic.add_argument(
        "--prototype-file",
        type=Path,
        default=None,
        help=(
            "Indoor37 RADIO prototype file. Required by "
            "--radio-inference dual."
        ),
    )

    semantic.add_argument(
        "--fusion-beta",
        type=float,
        default=1.0,
        help="Global multiplier applied to the fusion reliability alpha.",
    )

    semantic.add_argument(
        "--fusion-tau-ov",
        type=float,
        default=1.0,
        help="Softmax temperature for open-vocabulary query scores.",
    )

    semantic.add_argument(
        "--fusion-tau-cs",
        type=float,
        default=1.0,
        help="Softmax temperature for closed-derived query scores.",
    )

    semantic.add_argument(
        "--fusion-use-query-coverage",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "Multiply closed-derived query support B(x,q) by global "
            "query coverage R(q). Enabled by default."
        ),
    )

    semantic.add_argument(
        "--radio-chunk-size",
        type=int,
        default=32768,
    )

    semantic.add_argument(
        "--scannet-label-space",
        choices=(
            "auto",
            "nyu40",
            "scannet",
        ),
        default="auto",
    )
    # ==============================================================
    # Reconstruction evaluation
    # ==============================================================

    reconstruction = subparsers.add_parser(
        "reconstruction",
        help="Evaluate reconstruction quality",
    )

    reconstruction.add_argument(
        "--protocol",
        required=True,
        choices=sorted(
            RECONSTRUCTION_PROTOCOLS.keys()
        ),
    )

    reconstruction.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    reconstruction.add_argument(
        "--recon-root",
        type=Path,
        required=True,
    )

    reconstruction.add_argument(
        "--out-root",
        type=Path,
        required=True,
    )

    reconstruction.add_argument(
        "--device",
        default=_default_device(),
    )

    reconstruction.add_argument(
        "--scenes",
        nargs="+",
        default=None,
        help=(
            "Optional scene override. "
            "By default the protocol scene list is used."
        ),
    )

    reconstruction.add_argument(
        "--frame-stride",
        type=int,
        default=None,
        help=(
            "Optional frame-stride override. "
            "By default the protocol value is used."
        ),
    )

    reconstruction.add_argument(
        "--save-renderings",
        action="store_true",
        help="Save rendered RGB/depth predictions.",
    )

    return parser


def _run_semantic(
    args: argparse.Namespace,
) -> None:
    device = torch.device(
        args.device
    )

    protocol = SEMANTIC_PROTOCOLS[
        args.protocol
    ]

    is_indoor37 = (
        protocol.name in {"indoor37_replica","indoor37_scannet",}
        )

    if args.scenes is not None:
        protocol = replace(
            protocol,
            scenes=tuple(args.scenes),
        )

    if args.frame_stride is not None:
        if args.frame_stride <= 0:
            raise ValueError(
                "--frame-stride must be > 0"
            )

        protocol = replace(
            protocol,
            frame_stride=args.frame_stride,
        )

    # --------------------------------------------------------------
    # Dataset / taxonomy
    # --------------------------------------------------------------
    if protocol.name == "indoor37_replica":
        dataset = ReplicaIndoor37Dataset(root=args.data_root,)
        taxonomy = INDOOR37_TAXONOMY
        splits = None

    elif protocol.name == "indoor37_scannet":
        if args.scannet_label_file is None:
            raise ValueError(
                "--scannet-label-file is "
                "required for "
                "indoor37_scannet"
            )

        dataset = ScanNetIndoor37Dataset(root=args.data_root, label_file=(args.scannet_label_file),
                label_space=args.scannet_label_space,)
        taxonomy = INDOOR37_TAXONOMY
        splits = None

    elif protocol.dataset == "replica":

        dataset = ReplicaDataset(root=args.data_root,)
        taxonomy = REPLICA51_TAXONOMY
        splits = REPLICA51_SPLITS

    elif protocol.dataset == "scannet":

        if args.scannet_label_file is None:
            raise ValueError(
                "--scannet-label-file is required "
                "for ScanNet semantic evaluation"
            )

        dataset = ScanNetDataset(root=args.data_root,label_file=args.scannet_label_file,)
        taxonomy = SCANNET20_TAXONOMY
        splits = None

    else:
        raise ValueError(
            f"Unsupported semantic dataset: "
            f"{protocol.dataset}"
        )

    if is_indoor37:

        if args.radio_inference != "open_vocab":
            raise ValueError(
                "Indoor37 diagnostic protocols use the dedicated linear "
                "probe predictor and cannot be combined with "
                "--radio-inference dual."
            )

        if args.probe_checkpoint is None:
            raise ValueError(
                "--probe-checkpoint is "
                "required for Indoor37 "
                "probe evaluation"
            )

        predictor = build_radio_probe_predictor(
                feature_model_id=args.feature_model,
                registry_path=args.feature_registry,
                checkpoint_path=args.probe_checkpoint,
                device=device,
                chunk_size=args.radio_chunk_size,
            )
        
    else:
        if args.radio_inference == "dual":
            if args.probe_checkpoint is None:
                raise ValueError(
                    "--probe-checkpoint is required for "
                    "--radio-inference dual"
                )
            if args.prototype_file is None:
                raise ValueError(
                    "--prototype-file is required for "
                    "--radio-inference dual"
                )
            if args.fusion_beta < 0:
                raise ValueError("--fusion-beta must be >= 0")
            if args.fusion_tau_ov <= 0:
                raise ValueError("--fusion-tau-ov must be > 0")
            if args.fusion_tau_cs <= 0:
                raise ValueError("--fusion-tau-cs must be > 0")

        predictor = build_semantic_predictor(
            class_names=taxonomy.class_names,
            feature_model_id=args.feature_model,
            registry_path=args.feature_registry,
            radio_chunk_size=args.radio_chunk_size,
            device=device,
            radio_inference=args.radio_inference,
            probe_checkpoint=args.probe_checkpoint,
            prototype_path=args.prototype_file,
            fusion_beta=args.fusion_beta,
            fusion_tau_ov=args.fusion_tau_ov,
            fusion_tau_cs=args.fusion_tau_cs,
            fusion_use_query_coverage=args.fusion_use_query_coverage,
        )

    refinement = None

    if args.semantic_refinement == "iterative":

        if args.refine_iterations < 0:
            raise ValueError(
                "--refine-iterations must be >= 0"
            )

        if args.refine_lambda < 0:
            raise ValueError(
                "--refine-lambda must be >= 0"
            )

        if args.depth_sigma <= 0:
            raise ValueError(
                "--depth-sigma must be > 0"
            )

        if not args.affinity_dilations:
            raise ValueError(
                "--affinity-dilations must not be empty"
            )

        if any(
            dilation <= 0
            for dilation in args.affinity_dilations
        ):
            raise ValueError(
                "--affinity-dilations values must be > 0"
            )

        if args.affinity_source == "depth":

            if protocol.dataset != "replica":
                raise ValueError(
                    "GT-depth refinement is currently "
                    "supported only for Replica"
                )

            affinity_provider = DepthAffinityProvider(
                sigma_d=args.depth_sigma,
                dilations=args.affinity_dilations,
            )

        else:
            raise ValueError(
                "Unsupported affinity source: "
                f"{args.affinity_source}"
            )

        score_refiner = IterativeScoreRefiner(
            lambda_refine=args.refine_lambda,
            num_iterations=args.refine_iterations,
        )

        refinement = RefinementPipeline(
            affinity_provider=affinity_provider,
            score_refiner=score_refiner,
        )

    print()
    print("Semantic evaluation")
    print("-------------------")
    print(f"Protocol:      {protocol.name}")
    print(f"Dataset:       {protocol.dataset}")
    print(f"Taxonomy:      {taxonomy.name}")
    print(f"Classes:       {len(taxonomy.class_names)}")
    print(f"Feature model: {args.feature_model}")
    if not is_indoor37:
        print(f"RADIO mode:    {args.radio_inference}")
        if args.radio_inference == "dual":
            print(f"Fusion beta:   {args.fusion_beta}")
            print(f"Tau OV:        {args.fusion_tau_ov}")
            print(f"Tau CS:        {args.fusion_tau_cs}")
            print(
                "Use R(q):      "
                f"{args.fusion_use_query_coverage}"
            )
    print(f"Scenes:        {len(protocol.scenes)}")
    print(f"Frame stride:  {protocol.frame_stride}")
    print(f"Device:        {device}")
    print(f"Save masks:    {args.save_masks}")
    print(f"Refinement:    "f"{args.semantic_refinement}")
    if refinement is not None:
        print(f"Affinity:      "f"{args.affinity_source}")
        print(f"Iterations:    "f"{args.refine_iterations}")
        print(f"Lambda:        "f"{args.refine_lambda}")
        print(f"Depth sigma:   "f"{args.depth_sigma} m")
        print(f"Dilations:     "f"{args.affinity_dilations}")
    print()

    summary = evaluate_semantic_protocol(
        protocol=protocol,
        dataset=dataset,
        taxonomy=taxonomy,
        semantic_predictor=predictor,
        recon_root=args.recon_root,
        device=device,
        splits=splits,
        out_root=args.out_root,
        save_masks=args.save_masks,
        refinement=refinement,
        compute_cross_entropy=is_indoor37,
        component_refinement=(args.semantic_refinement == "components"),
    )

    pooled = summary[
        "pooled"
    ]["splits"]["all"]

    scene_mean = summary[
        "scene_mean"
    ]

    print()
    print("=" * 80)
    print("SEMANTIC RESULTS")
    print("=" * 80)

    print("Pooled/global:")
    print(
        "  mIoU:           "
        + _format_metric(
            pooled["mIoU"]
        )
    )
    print(
        "  mAcc:           "
        + _format_metric(
            pooled["mAcc"]
        )
    )
    print(
        "  FmIoU:          "
        + _format_metric(
            pooled["FmIoU"]
        )
    )
    print(
        "  Pixel accuracy: "
        + _format_metric(
            summary["pooled"][
                "pixel_accuracy"
            ]
        )
    )

    print()
    print("Scene mean:")
    print(
        "  mIoU:           "
        + _format_metric(
            scene_mean["mIoU"]
        )
    )
    print(
        "  mAcc:           "
        + _format_metric(
            scene_mean["mAcc"]
        )
    )
    print(
        "  FmIoU:          "
        + _format_metric(
            scene_mean["FmIoU"]
        )
    )
    print(
        "  Pixel accuracy: "
        + _format_metric(
            scene_mean[
                "pixel_accuracy"
            ]
        )
    )

    if is_indoor37:

        pooled = summary["pooled"]
        all_metrics = pooled[
            "splits"
        ]["all"]

        print()
        print("Indoor37 closed-set probe")
        print("-------------------------")

        print(
            f"Cross entropy: "
            f"{pooled['cross_entropy']:.6f}"
        )

        print(
            f"Pixel accuracy: "
            f"{pooled['pixel_accuracy']:.4f}"
        )

        print(
            f"mAcc: "
            f"{all_metrics['mAcc']:.4f}"
        )

        print(
            f"mIoU: "
            f"{all_metrics['mIoU']:.4f}"
        )

        print(
            f"fwIoU: "
            f"{all_metrics['FmIoU']:.4f}"
        )


def _run_reconstruction(
    args: argparse.Namespace,
) -> None:
    device = torch.device(
        args.device
    )

    protocol = RECONSTRUCTION_PROTOCOLS[
        args.protocol
    ]

    scenes = tuple(
        args.scenes
        if args.scenes is not None
        else protocol.scenes
    )

    frame_stride = (
        args.frame_stride
        if args.frame_stride is not None
        else protocol.frame_stride
    )

    if frame_stride <= 0:
        raise ValueError(
            "--frame-stride must be > 0"
        )

    if protocol.dataset == "replica":

        dataset = (
            ReplicaReconstructionDataset(
                root=args.data_root
            )
        )

    elif protocol.dataset == "scannet":

        dataset = (
            ScanNetReconstructionDataset(
                root=args.data_root
            )
        )

    else:
        raise ValueError(
            f"Unsupported reconstruction dataset: "
            f"{protocol.dataset}"
        )

    print()
    print("Reconstruction evaluation")
    print("-------------------------")
    print(f"Protocol:     {protocol.name}")
    print(f"Dataset:      {protocol.dataset}")
    print(f"Scenes:       {len(scenes)}")
    print(f"Frame stride: {frame_stride}")
    print(f"Device:       {device}")
    print()

    summary = evaluate_dataset(
        scenes=scenes,
        dataset=dataset,
        recon_root=args.recon_root,
        device=device,
        frame_stride=frame_stride,
        out_root=args.out_root,
        save_renderings=args.save_renderings,
    )

    mean = summary[
        "scene_mean"
    ]

    print()
    print("=" * 80)
    print("RECONSTRUCTION RESULTS")
    print("=" * 80)

    print("RGB:")
    print(
        "  PSNR:  "
        + _format_metric(
            mean["rgb"]["psnr"]
        )
    )
    print(
        "  SSIM:  "
        + _format_metric(
            mean["rgb"]["ssim"]
        )
    )
    print(
        "  LPIPS: "
        + _format_metric(
            mean["rgb"]["lpips"]
        )
    )

    if (
        mean["depth"]["l1_cm"]
        is not None
    ):
        print()
        print("Depth:")
        print(
            "  L1 [cm]: "
            + _format_metric(
                mean["depth"]["l1_cm"]
            )
        )

    if (
        mean["trajectory"]["rmse_cm"]
        is not None
    ):
        print()
        print("Trajectory:")
        print(
            "  RMSE [cm]: "
            + _format_metric(
                mean["trajectory"][
                    "rmse_cm"
                ]
            )
        )
        print(
            "  Mean [cm]: "
            + _format_metric(
                mean["trajectory"][
                    "mean_cm"
                ]
            )
        )


def main() -> None:
    parser = build_parser()
    args = parser.parse_args()

    if args.task == "semantic":
        _run_semantic(args)

    elif args.task == "reconstruction":
        _run_reconstruction(args)

    else:
        parser.error(
            f"Unknown task: {args.task}"
        )


if __name__ == "__main__":
    main()