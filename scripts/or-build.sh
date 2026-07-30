#!/bin/bash
set -eux
set -o pipefail

OR_SPEC=$(base64 -w 0 pkg/or-docker/bart.json)
TARGET=$(jq -r '"OpenRecon_\(.general.vendor)_\(.general.id)_V\(.general.version)"' < pkg/or-docker/bart.json)

if [ -z "${DEPS:-}" ]; then
       if [ -d "/home/ibi/data/ci-software" ]; then
               DEPS=/home/ibi/data/ci-software
       else
               DEPS="."
       fi
fi

podman build										\
  --format docker									\
  --label "com.siemens-healthineers.magneticresonance.openrecon.metadata:1.1.0=$OR_SPEC"\
  --build-arg=DEBUG_HOST="${DEBUG_HOST:-}"						\
  --build-arg=OR_GPU_SUPPORT="${GPU:-1}"						\
  --build-context=deps="${DEPS}"							\
  -t bart-on-fire -f pkg/or-docker/Dockerfile .

mkdir $TARGET
podman save -o $TARGET/$TARGET.tar bart-on-fire
pandoc -o $TARGET/$TARGET.pdf pkg/or-docker/application_guide.md
cd $TARGET
zip -Z deflate -r ../$TARGET.zip .
cd ..
rm -rf $TARGET
