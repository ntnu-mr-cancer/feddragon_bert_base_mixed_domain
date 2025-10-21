#!/usr/bin/env bash

SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

./build.sh

VOLUME_SUFFIX=$(dd if=/dev/urandom bs=32 count=1 | md5sum | cut -c 1-10)
# Maximum is currently 30g, configurable in your algorithm image settings on grand challenge
MEM_LIMIT="15g"


# set up temporary volumes for the docker run
docker volume create dragon-output-$VOLUME_SUFFIX

fold=0
for task_name in "Task101_Example_sl_bin_clf" "Task102_Example_sl_mc_clf" "Task103_Example_mednli" "Task104_Example_ml_bin_clf" "Task105_Example_ml_mc_clf" "Task106_Example_sl_reg" "Task107_Example_ml_reg" "Task108_Example_sl_ner" "Task109_Example_ml_ner"
do
    jobname="$task_name-fold$fold"

    echo "=========================================="
    echo "Running test for $jobname..."

    # Do not change any of the parameters to docker run, these are fixed
    docker run --rm \
        --gpus=all \
        --memory="${MEM_LIMIT}" \
        --memory-swap="${MEM_LIMIT}" \
        --network="none" \
        --cap-drop="ALL" \
        --security-opt="no-new-privileges" \
        --shm-size="128m" \
        --pids-limit="256" \
        -v $SCRIPTPATH/test-input/$jobname:/input:ro \
        -v dragon-output-$VOLUME_SUFFIX:/output \
        skarrea/dragon_submission

    # Display the output file
    docker run --rm \
        -v dragon-output-$VOLUME_SUFFIX:/output/ \
        python:3.10-slim cat /output/nlp-predictions-dataset.json

    # Collect the output file
    docker run --rm \
        -v "dragon-output-$VOLUME_SUFFIX":/output \
        alpine /bin/sh -c "mkdir -p /output/$jobname; mv /output/nlp-predictions-dataset.json /output/$jobname/;"

done

echo "=========================================="
echo "Test result: "

# Evaluate the output
docker run --rm \
    -v dragon-output-$VOLUME_SUFFIX:/input:ro \
    -v $SCRIPTPATH/test-ground-truth:/opt/app/ground-truth \
    --entrypoint python \
    skarrea/dragon_submission -m dragon_eval --folds 0

docker volume rm dragon-output-$VOLUME_SUFFIX

echo "Please check that all performances are above random guessing! For tasks 101-107, the performance should be above 0.7, for tasks 108-109 above 0.2."
echo "=========================================="
echo "Test completed."
echo "=========================================="
