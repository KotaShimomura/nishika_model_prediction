#!/bin/bash

TAG="kotashimomura/kaggledockerimage"
PROJECT_DIR="$(cd "$(dirname "${0}")/.." || exit; pwd)"

docker run -it --rm \
  -v "${PROJECT_DIR}:/workspace" \
  -w "/workspace" \
  "${TAG}"
