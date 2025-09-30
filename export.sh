#!/usr/bin/env bash

./build.sh

docker save skarrea/dragon_submission:latest | gzip -c > dragon_submission.tar.gz
