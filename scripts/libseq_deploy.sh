#!/bin/bash
set -eu

SCRIPTDIR=$(dirname $(readlink -f "$0"))
cd $SCRIPTDIR/..

: "${VM_BART_PATH:=.}"
: "${VM_BIN_PATH:=.}"

error() { echo $1; exit -1; }
git diff --quiet || error "git status not clean!"

LIBSEQ_NAME=bart_seq_$(git rev-parse --short=10 HEAD)

set -x
make allclean
BARTDLL=1 make bart.dll
cp bart.dll $VM_BIN_PATH/$LIBSEQ_NAME.dll

make allclean
BARTSO=1 make libbart.so
cp libbart.so $VM_BART_PATH/lib/lib$LIBSEQ_NAME.so

