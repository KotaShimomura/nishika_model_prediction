#!/bin/bash

TAG="kotashimomura/kaggledockerimage"
cd "$(dirname "${0}")/.." || exit

DOCKER_BUILDKIT=1 docker build --progress=plain -t ${TAG} -f docker/Dockerfile .