#!/bin/bash
set -eu

SCRIPTDIR=$(dirname $(readlink -f "$0"))
cd $SCRIPTDIR/..

: "${VM_BART_PATH:=.}"
: "${VM_BIN_PATH:=.}"

error() { echo $1; exit -1; }
git diff --quiet || error "git status not clean!"

LIBSEQ_NAME=bart_seq_$(git rev-parse --short=10 HEAD)
MINGWDLLTOOL=x86_64-w64-mingw32-dlltool

set -x
OMP=0 BARTDLL=1 make "$@" bart.dll
OMP=0 make "$@" lib/libbart.a

$MINGWDLLTOOL -l lib/$LIBSEQ_NAME.lib --dllname $LIBSEQ_NAME.dll -d bart.def
cp lib/$LIBSEQ_NAME.lib $VM_BART_PATH/lib/$LIBSEQ_NAME.lib || true
cp lib/libbart.a $VM_BART_PATH/lib/lib$LIBSEQ_NAME.a
cp bart.dll $VM_BIN_PATH/$LIBSEQ_NAME.dll
