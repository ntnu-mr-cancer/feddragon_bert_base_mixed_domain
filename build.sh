#!/usr/bin/env bash
SCRIPTPATH="$( cd "$(dirname "$0")" ; pwd -P )"

docker build -t skarrea/dragon_submission:latest "$SCRIPTPATH"
