#!/bin/bash
# Copyright 2026. TU Graz. Institute of Biomedical Imaging.
# Author: Daniel Mackner
#

set -eu

helpstr=$(cat <<- EOF
Compute Pulseq phase correction for FOV-shifted acquisition with equidistant sampling,
symmetric echo and constant readout gradient.

<adc>		ADC of FOV-shifted sequence (--no-spoiling)
<corr>		correction term

-s shift	shift in seconds (Siemens: ~5us)
-h	help
EOF
)


usage="Usage: $0 [-h] [-s shift] <adc> <corr>"

SHIFT=5E-6

while getopts "hs:" opt; do
        case $opt in
        s)
		SHIFT=$OPTARG
        ;;
	h)
		echo "$usage"
		echo
		echo "$helpstr"
		exit 0
	;;
	\?)
		echo "$usage" >&2
		exit 1
	;;
        esac
done

shift $((OPTIND - 1))


if [ $# -ne 2 ] ; then

        echo "$usage" >&2
        exit 1
fi

if [ ! -e "$BART_TOOLBOX_PATH"/bart ] ; then
	echo "\$BART_TOOLBOX_PATH is not set correctly!" >&2
	exit 1
fi
export PATH="$BART_TOOLBOX_PATH:$PATH"

adc=$(readlink -f "$1")
corr=$(readlink -f "$2")

WORKDIR=`mktemp -d 2>/dev/null || mktemp -d -t 'mytmpdir'`
trap 'rm -rf "$WORKDIR"' EXIT
cd $WORKDIR


bart slice 0 3 $adc time
bart slice 0 4 $adc phase

RO=$(bart show -d1 $adc)

echo "shift of $SHIFT for $RO samples"


# shift in units of dwell
bart circshift 1 1 time - | bart saxpy -- -1 - time - |\
bart extract 1 1 $RO - - | bart avg 65535 - - |\
bart invert - - | bart scale $SHIFT - shift


bart transpose 0 1 phase - | bart interpolate -C -D 1 - shift - |\
bart transpose 0 1 - - | bart fmac -C - phase - |\
bart slice 1 $((RO/2)) - $corr

