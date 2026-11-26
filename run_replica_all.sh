#!/usr/bin/env bash

set -euo pipefail

DATA_ROOT="/workspace/data/Replica"
OUTPUT_ROOT="/workspace/results/results_replica_radiov4_128"

ORB_CONFIG_ROOT="cfg/ORB_SLAM3/RGB-D/Replica"

ENCODER_CONFIG="cfg/feature_models.json"
GAUSSIAN_CONFIG="cfg/gaussian_mapper/RGB-D/Replica/replica_rgbd.yaml"
FEATURE_MODEL="radio_v4_so400m"


SCENES=(
    office0
    office1
    office2
    office3
    office4
    room0
    room1
    room2
)


for SCENE in "${SCENES[@]}"; do

    echo
    echo "================================================================================"
    echo "Running Replica scene: ${SCENE}"
    echo "================================================================================"
    echo

    SEQUENCE="${DATA_ROOT}/${SCENE}"
    OUTPUT="${OUTPUT_ROOT}/${SCENE}"
    ORB_CONFIG="${ORB_CONFIG_ROOT}/${SCENE}.yaml"

    # ------------------------------------------------------------
    # Basic checks
    # ------------------------------------------------------------

    if [[ ! -d "${SEQUENCE}" ]]; then
        echo "ERROR: Dataset scene does not exist:"
        echo "  ${SEQUENCE}"
        exit 1
    fi

    if [[ ! -f "${ORB_CONFIG}" ]]; then
        echo "ERROR: ORB config does not exist:"
        echo "  ${ORB_CONFIG}"
        exit 1
    fi

    # ------------------------------------------------------------
    # Run LEG-SLAM
    # ------------------------------------------------------------

    python -m tools.slam.run \
        --dataset replica \
        --sequence "${SEQUENCE}" \
        --output "${OUTPUT}" \
        --orb-config "${ORB_CONFIG}" \
        --feature-registry "${ENCODER_CONFIG}" \
        --gaussian-config "${GAUSSIAN_CONFIG}" \
        --feature-model "${FEATURE_MODEL}"

    echo
    echo "Finished: ${SCENE}"
    echo

done


echo
echo "================================================================================"
echo "All Replica scenes finished."
echo "================================================================================"