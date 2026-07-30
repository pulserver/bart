#!/bin/bash

get_file() {
	[ "-" = "$1" ] && echo - || echo $(readlink -f "$1")
}

## debugging
dump() {
	cat $1 > ksp
	bart phantom -B $2
}


## static cartesian nlinv reco
cart_reco () {
	KSP=$(get_file $1)
	DST=$(get_file $2)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart copy $KSP ksp;
	bart nlinv -d4 $gpu_flag -i 11 ksp img;
	bart -r img -l$(bart bitmask 10 13) copy img $DST;
}

normalize() {
	KSP=$(get_file $1)
	DST=$(get_file $2)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	f() {
		bart -r in.fifo cabs in.fifo -		|\
		bart tee -n scale.fifo data.fifo	&

		bart -l $(bart bitmask 10 13) -e 1:1	\
			copy scale.fifo -		|\
		bart mip 3 - -				|\
		bart invert - -				|\
		bart scale 4096 - -			|\
		bart show -R -f"%.6f" - > scale.txt


		bart -r data.fifo scale			\
			$(cat scale.txt) data.fifo $DST
	}

	f &
	bart -r $KSP copy $KSP in.fifo
	wait
}

connect_ismrmrd() {
	meta=$(mktemp -u).fifo
	mkfifo $meta

	## send results
	bart ismrmrd --meta $meta --stream      \
	 --write-ismrm-cfl-img img.fifo - &

	## read input data (note: this must happen in fg to capture stdin!)
	bart ismrmrd --meta $meta --stream - ksp.fifo
	wait
}

connect_bart_streams() {

	normalize img.fifo -			|\
	bart -r - --stream-bin-out copy - - &

	## read input data (note: this must happen in fg to capture stdin!)
	cat - > ksp.fifo
	wait
}

main() {

	set -eux
	set -o pipefail

	export BART_DEBUG_STREAM=1
	export TMPDIR=/dev/shm/
	export TMP_TEMPLATE=tmp.or-reco.XXXXXXXXXX

	export PATH="${BART_TOOLBOX_PATH}:${PATH}"

	gpu_flag=""
	if [ 1 -eq "${OR_GPU_SUPPORT:-0}" ]; then
	       gpu_flag=-g
	fi

	mkfifo ksp.fifo
	mkfifo img.fifo

	# OR_BART_STREAMS=1 signifies script is called
	# as the reconstruction of the BART BOOST Sequence.
	# This implies:
	# a) Data in BART Streams format
	# b) Radial acquisition.
	# Conversely, if OR_BART_STREAMS=0, this script is called
	# from the "normal" OpenRecon workflow. This means:
	# a) Data in ISMRMRD format
	# Furthermore, we assume:
	# b) Cartesian acquisition,
	# as OpenRecon currently does not support non-Cartesian.

	if [ 1 -eq "${OR_BART_STREAMS:-0}" ]; then

		$BART_TOOLBOX_PATH/scripts/rtreco.sh -t5 -f -p8 ksp.fifo img.fifo &
		connect_bart_streams
	else
		cart_reco ksp.fifo img.fifo &
		connect_ismrmrd
	fi
}

(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -eq 0 ]; then
	main "$@"
fi
