#!/bin/bash
REPO=reidkr
IMAGE=hair-cam
VERSION=$(grep API_VERSION Dockerfile | head -n 1 | awk '{print $3}')
docker build -t ${REPO}/${IMAGE}:latest .
docker tag ${REPO}/${IMAGE}:latest ${REPO}/${IMAGE}:${VERSION}
