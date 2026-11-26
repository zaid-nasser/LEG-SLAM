from __future__ import annotations

import argparse
from pathlib import Path

import torch

from .datasets.replica import ReplicaDataset
from .datasets.scannet import ScanNetDataset

from .evaluator import evaluate_protocol

from .predictors.factory import (
    build_semantic_predictor,
)

from .protocols import PROTOCOLS

from .taxonomies import (
    REPLICA51_TAXONOMY,
    SCANNET20_TAXONOMY,
    REPLICA51_SPLITS,
)

def parse_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--protocol",
        required=True,
        choices=PROTOCOLS.keys(),
    )

    parser.add_argument(
        "--data-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--recon-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--feature-model",
        required=True,
    )

    parser.add_argument(
        "--feature-registry",
        type=Path,
        default=Path("cfg/feature_models.json"),
    )

    parser.add_argument(
        "--scannet-label-file",
        type=Path,
        default=None,
    )

    parser.add_argument(
        "--out-root",
        type=Path,
        required=True,
    )

    parser.add_argument(
        "--device",
        default=(
            "cuda"
            if torch.cuda.is_available()
            else "cpu"
        ),
    )

    parser.add_argument(
        "--save-masks",
        action="store_true",
        help=(
            "Save predicted and ground-truth semantic ID maps "
            "and colored visualizations."
        ),
    )

    return parser.parse_args()


def main():

    args = parse_args()

    device = torch.device(args.device)

    protocol = PROTOCOLS[args.protocol]

    # --------------------------------------------------------------
    # Dataset + taxonomy
    # --------------------------------------------------------------

    if protocol.dataset == "replica":

        dataset = ReplicaDataset(
            root=args.data_root,
        )

        taxonomy = REPLICA51_TAXONOMY
        splits = REPLICA51_SPLITS

    elif protocol.dataset == "scannet":

        if args.scannet_label_file is None:
            raise ValueError(
                "--scannet-label-file is required "
                "for ScanNet evaluation"
            )

        dataset = ScanNetDataset(
            root=args.data_root,
            label_file=args.scannet_label_file,
        )

        taxonomy = SCANNET20_TAXONOMY
        splits = None

    else:
        raise ValueError(
            f"Unsupported dataset: {protocol.dataset}"
        )

    # --------------------------------------------------------------
    # Predictor
    # --------------------------------------------------------------

    predictor = build_semantic_predictor(
        class_names=taxonomy.class_names,
        feature_model_id=args.feature_model,
        registry_path=args.feature_registry,
        device=device,
    )

    print()
    print("Evaluation configuration")
    print("------------------------")
    print(f"Protocol:      {protocol.name}")
    print(f"Dataset:       {protocol.dataset}")
    print(f"Taxonomy:      {taxonomy.name}")
    print(f"Classes:       {len(taxonomy.class_names)}")
    print(f"Feature model: {args.feature_model}")
    print(f"Device:        {device}")
    print()

    # --------------------------------------------------------------
    # Evaluation
    # --------------------------------------------------------------

    summary = evaluate_protocol(
        protocol=protocol,
        dataset=dataset,
        taxonomy=taxonomy,
        semantic_predictor=predictor,
        recon_root=args.recon_root,
        device=device,
        splits=splits,
        out_root=args.out_root,
        save_masks=args.save_masks,
    )

    print()
    print(f"Save masks:    {args.save_masks}")
    print("=" * 80)
    print("FINAL RESULTS")
    print("=" * 80)

    pooled = summary["pooled"]["splits"]["all"]
    scene_mean = summary["scene_mean"]

    print("Pooled/global:")
    print(f"  mIoU:  {pooled['mIoU']:.4f}")
    print(f"  mAcc:  {pooled['mAcc']:.4f}")
    print(f"  FmIoU: {pooled['FmIoU']:.4f}")
    print(
        f"  Pixel accuracy: "
        f"{summary['pooled']['pixel_accuracy']:.4f}"
    )

    print()
    print("Scene mean:")
    print(f"  mIoU:  {scene_mean['mIoU']:.4f}")
    print(f"  mAcc:  {scene_mean['mAcc']:.4f}")
    print(f"  FmIoU: {scene_mean['FmIoU']:.4f}")
    print(
        f"  Pixel accuracy: "
        f"{scene_mean['pixel_accuracy']:.4f}"
    )


if __name__ == "__main__":
    main()