#!/bin/bash
# Copyright 2024-2026. Institute of Biomedical Imaging. TU Graz.
# All rights reserved. Use of this source code is governed by
# a BSD-style license which can be found in the LICENSE file.
#
# Authors:
# 2024-2026 Moritz Blumenthal <blumenthal@tugraz.at>
# 2024-2026 Philip Schaten <philip.schaten@tugraz.at>



main() {
	set -euo pipefail

	MAIN_PID=$$
	export MAIN_PID

	LOGFILE=/dev/stderr

	title=$(cat <<- EOF
		Real-Time Reconstruction
	EOF
	)

	helpstr=$(cat <<- EOF
	-h help
	-l logfile
	-L timestamp logfile
	-t #turns / tiny golden angle parameter
	-f median filter
	-R ROVIR
	-G Real-time geometric decomposition coil compression
	-S Static Coil Compression matrix estimated from first frame.
	-T Output timestamp on stderr
	-s spokes per frame, activates RAGA reconstruction.
	EOF
	)

	usage="Usage: rtreco.sh [-h,l,f,n,T,A,d] [(-R|-G|-S|-N)] [-p <# virt. chan.>] [-l <logfile>] [-L <timestamp logfile>] [-s <spokes per frame>] [-t <TURNS=5 / Tiny GA=1>] <kspace> <output> [<coils>]"

	TURNS=
	ROVIR=false
	GEOM=false
	FILTER=false
	NLMEANS=false
	STATIC_COILS=false
	OVERGRIDDING=1.5
	DELAY=2
	CHANNELS=8
	RAGA=false
	TINY=1
	SPOKES_PER_FRAME=0
	TOPTS="-o2 -r -D -O"
	SLW=false
	CC_NONE=false
	PHASE_POLE=
	FAST=" "
	GD_CORR=true
	TIMELOG=
	GRAD_DELAYS=""

	: "${NLMEANS_OPTS:=-p3 -d3 -H0.00005 3}"
	: "${NLINV_OPTS:=--cgiter=10 -S --real-time --sens-os=1.25 -i6}"
	: "${ESTDELAY_OPTS:=-R -p20 -r2}"

	export TMPDIR=/dev/shm/
	export TMP_TEMPLATE=tmp.rtreco.XXXXXXXXXX

	export OMP_NUM_THREADS=1

	while getopts "hl:t:fnRTp:SGL:s:ANPd" opt; do
	        case $opt in
		h)
			echo "$usage"
			echo
			echo "$helpstr"
			exit 0
		;;
		l)
			LOGFILE=$(readlink -f "$OPTARG")
		;;
		L)
			TIMELOG=$(readlink -f "$OPTARG")
		;;
		t)
			TURNS=$OPTARG
		;;
		f)
			FILTER=true
		;;
		n)
			NLMEANS=true
		;;
		T)
			TIME=-t
		;;
		R)
			ROVIR=true
		;;
		S)
			STATIC_COILS=true
		;;
		p)
			CHANNELS="$OPTARG"
		;;
		P)
			PHASE_POLE=" --phase-pole=6"
			FAST=""
		;;
		G)
			GEOM=true
		;;
		s)
			SPOKES_PER_FRAME="$OPTARG"
		;;
		A)
			SLW=true
		;;
		N)
			CC_NONE=true
		;;
		d)
			GD_CORR=false
		;;
	        \?)
			echo "$usage" >&2
			exit 1
	        ;;
	        esac
	done

	shift $((OPTIND - 1))

	if [ $# -lt 2 -o $# -gt 3 ] ; then

		echo "$usage" >&2
		exit 1
	fi

	echo "$title"	>>$LOGFILE
	echo		>>$LOGFILE

	if [ 0 -lt $SPOKES_PER_FRAME ]; then

		: "${TURNS:=1}"

		RAGA=true
		TINY=$TURNS
		TURNS=1
	else
		: "${TURNS:=5}"
	fi

	if $SLW; then
		OVERGRIDDING=1
	fi

	if [ ! -e "$BART_TOOLBOX_PATH"/bart ] ; then
		echo "\$BART_TOOLBOX_PATH is not set correctly!" >&2
		exit 1
	fi

	GPU=

	if [ -n "$(bart version -V | grep CUDA=1)" ]; then
		GPU="-g"
	fi

	export GPU
	export GD_CORR
	export ESTDELAY_OPTS
	export ROVIR
	export TURNS
	export DELAY
	export CHANNELS
	export STATIC_COILS
	export GEOM
	export TOPTS
	export RAGA
	export TINY
	export SLW
	export CC_NONE
	export PHASE_POLE
	export FAST
	export BART_DEBUG_STREAM=1
	export PATH="$BART_TOOLBOX_PATH:$PATH"

	KSP=$1
	REC=$2
	COILS=
	if [ $# -eq 3 ]; then
		COILS=$(readlink -f "$3")
	fi

	if [ "-" = "$1" ]; then
		stdin_tmpdir=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
		trap 'rm -rf "$stdin_tmpdir"' EXIT
		KSP=$stdin_tmpdir/stdin.fifo
		mkfifo $KSP
	fi

	rtreco()
	{
		KSP=$(get_file $1)
		REC=$(get_file $2)

		if [ $# -eq 3 ]; then
			COILS=$(get_file $3)
			COILS_TMP=coils.fifo
		else
			COILS=
			COILS_TMP=
		fi

		WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
		trap 'rm -rf "$WORKDIR"' EXIT
		cd "$WORKDIR" || exit

		echo "WORKING_DIR:    $WORKDIR" >&2
		echo "k-Space:        $KSP" 	>&2
		echo "Reconstruction: $REC" 	>&2

		reshaped_ksp=ksp.fifo
		if $RAGA; then
			reshaped_ksp=ksp0.fifo
			rebin_raga ksp0.fifo ksp.fifo &
		fi

		reshape_radial_ksp $KSP - |\
		BART_STREAM_LOG="${TIMELOG:+${TIMELOG}_incoming}" bart tee -n --out0 meta.fifo $reshaped_ksp &

		dims=$(get_dims meta.fifo)

		READ=$(get_dim 1 $dims)
		SPOKE_FF=$(get_dim 2 $dims)
		N_COILS=$(get_dim 3 $dims)
		FULL_FRAMES=$(get_dim 10 $dims)
		SLICES=$(get_dim 13 $dims)
		RDIMS=$((READ/2))
		GDIMS=$(echo "scale=0;($RDIMS*$OVERGRIDDING+0.5)/1" | bc -l)

		: ${IMG_SIZE:=$GDIMS:$GDIMS:1}

		if [ $FULL_FRAMES -lt $TURNS ]; then
			FILTER=false
		fi

		# Using delayed data for gradient delay corr. & coil compression
		# requires sufficient number of frames
		if [ $FULL_FRAMES -lt $((3 * TURNS)) ]; then
			echo "Setting DELAY=0 due to small ($FULL_FRAMES < $((3 * TURNS))) framecount" >&2
			DELAY=0
			export DELAY
		fi

		if $RAGA && [ $SLICES -ne 1 ]; then
			error "RAGA currently does not support multi slice."
		fi

		if [ $CHANNELS -ge $N_COILS ]; then
			if $ROVIR; then
				error "ROVIR specified, but no of virt. channels >= real channels: $CHANNELS >= $COILS"
			fi
			STATIC_COILS=false;
			GEOM=false;
			CC_NONE=true;
		fi

		export SPOKE_FF
		export FULL_FRAMES


		trajectory ksp_gd.fifo trj.fifo $GRAD_DELAYS &

		if $ROVIR ; then
			coilcompression_rovir		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
		elif $STATIC_COILS ; then
			coilcompression_svd_first	ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
		elif $GEOM; then
			coilcompression_geom		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
		elif $CC_NONE; then
			coilcompression_none		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
		else
			coilcompression_svd		ksp_cc.fifo trj_cc.fifo ksp_reco.fifo &
		fi

		bart tee -i trj.fifo trj_cc.fifo | bart -r - scale -- $OVERGRIDDING - trj_reco.fifo &
		bart tee -i ksp.fifo -n ksp_gd.fifo ksp_cc.fifo &

		# Postprocessing
		if $FILTER; then
			OUT=reco.fifo
			filter 5 reco.fifo reco_fil.fifo &
			if $NLMEANS; then

				OMP_NUM_THREADS=16 BART_STREAM_LOG="${TIMELOG:+${TIMELOG}_filter}" bart -r reco_fil.fifo \
					nlmeans $NLMEANS_OPTS reco_fil.fifo $REC &
			else
				BART_STREAM_LOG="${TIMELOG:+${TIMELOG}_filter}" bart -r reco_fil.fifo copy reco_fil.fifo $REC &

			fi
		else
			OUT=$REC
		fi

		# Reconstruction
		if $SLW; then

			if [ $FULL_FRAMES -lt $TURNS ]; then
				window_size=$FULL_FRAMES
			else
				window_size=$TURNS
			fi

			sliding_window $window_size trj_reco.fifo -		|\
				bart tee -n trj_sw1.fifo trj_sw2.fifo		&

			sliding_window $window_size ksp_reco.fifo  -		|\
				rl_filter_ksp - trj_sw1.fifo tmp.fifo		&

			OMP_NUM_THREADS=4 BART_STREAM_LOG="${TIMELOG:+${TIMELOG}}" bart -r tmp.fifo		\
				nufft -x$RDIMS:$RDIMS:1 -a trj_sw2.fifo tmp.fifo tmp2.fifo	&

			bart -r tmp2.fifo rss 8 tmp2.fifo $OUT &
		else

			bart tee -i ksp_reco.fifo --out0 meta2.fifo			|\
			bart -r - -l $(bart bitmask 10 13) copy - ksp_reco2.fifo	&

			dims=$(get_dims meta2.fifo)
			FRAMES=$(get_dim 10 $dims)

			## Move slice dim (13) to time dim 2 (11), as slice dim
			## invokes special NLINV semantics (SMS-NLINV)
			flags=$(bart bitmask 10 11 13)
			bart reshape -s $flags $flags $FRAMES $SLICES 1 ksp_reco2.fifo -	|\
			bart -r - -l $(bart bitmask 10 11) copy - kspr.fifo &

			bart reshape -s $flags $flags $FRAMES $SLICES 1 trj_reco.fifo -	|\
			bart -r - -l $(bart bitmask 10 11) copy - trj_reco2.fifo &

			## RT-NLINV reconstruction
			BART_STREAM_LOG="${TIMELOG:+${TIMELOG}}" bart nlinv				\
				$NLINV_OPTS						\
				$FAST $PHASE_POLE $GPU -x$IMG_SIZE 			\
				-t trj_reco2.fifo kspr.fifo imgr.fifo $COILS_TMP		&

			## Move slices back to slice dim and remove oversampling:
			bart reshape -s $flags $flags $FRAMES 1 $SLICES imgr.fifo -	|\
			bart -r - -l $(bart bitmask 10 13) copy - -			|\
			bart -r - resize -c 0 $RDIMS 1 $RDIMS - $OUT			&

			## Postprocess coils, if applicable
			if [ ! -z $COILS_TMP ]; then
				bart reshape -s $flags $flags $FRAMES 1 $SLICES $COILS_TMP -	|\
				bart -r - -l $(bart bitmask 10 13) copy - -			|\
				bart resize -c 0 $RDIMS 1 $RDIMS - $COILS			&
			fi
		fi

		# before removing tmpdir:
		wait
	}
	rtreco $KSP $REC $COILS 2>>$LOGFILE &

	if [ "-" = "$1" ]; then
		# 'catch' stdin and block until it's closed:
		cat > $KSP
	fi

	wait
}



# Helper functions:
get_file() {
	[ "-" = "$1" ] && echo - || echo $(readlink -f "$1")
}

get_dims() {
	bart show -m $(get_file $1) | tail -n1 | cut -f2-
}

get_dim() {
	dim=$1
	shift
	echo "$@" | cut -f$((dim + 1)) -d' '
}

get_output_flags() {
	if [ 1 -eq $1 ]; then
		bart bitmask 10
	else
		bart bitmask 10 13
	fi
}

error() {
	red="\e[0;31m"
	black="\e[0;30m"

	echo -e "$red" >&2
	echo $1 >&2
	echo -e "$black" >&2

	pgid="$(ps -o pgid= $$ | grep -o '[0-9]*')"
	kill -- -$pgid

	exit
}

# delay <shift-dim> <shift> <end> <infile> <outfile>
## Delay <infile> by <shift> frames along <shift-dim>,
## by repeating the initial <shift> frames, optionally
## cutting of the last <end> frames to keep dims equal
delay () (
	dim=$1
	start=$2
	end=$3
	src=$(get_file $4)
	dst=$(get_file $5)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	f () {

		if [ $start -eq 0 ]; then
			bart show -m meta.fifo >/dev/null &
			bart show -m in2.fifo >/dev/null &
			bart -r in1.fifo copy in1.fifo $dst
			return
		fi

		flags=$(bart bitmask $dim 13)

		dims=$(get_dims meta.fifo)
		frames=$(get_dim $dim $dims)
		slices=$(get_dim 13 $dims)

		if [ $slices -eq 1 ]; then
			out_flags=$(bart bitmask 10 $dim)
		else
			out_flags=$(bart bitmask 10 13 $dim)
		fi

		if [ $frames -lt $start -o $frames -lt $end ]; then
			error "Error(rtreco.sh) : delay start=$start, end=$end called on data with\
				frames=$frames"
		fi

		end=$((frames-end))

		bart -l$flags -e$end:$slices	copy in1.fifo end.fifo		&
		bart -l$flags -e$start:$slices	copy in2.fifo -			|\
		bart				join -s $dim - end.fifo -	|\
		bart -l $out_flags -r - copy - $dst
	}

	f &
	bart tee -i $src --out0 meta.fifo -n in1.fifo in2.fifo
	wait
)

## Reshape incoming k-space data, ensure dims[0] = 1.
reshape_radial_ksp() (
	src=$(get_file $1)
	dst=$(get_file $2)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	f () {

		dims=$(get_dims meta.fifo)

		samples=$(get_dim 0 $dims)
		spokes=$(get_dim 1 $dims)
		frames=$(get_dim 10 $dims)
		slices=$(get_dim 13 $dims)

		if [ 1 -eq $samples ]; then

			bart -r ksp_in.fifo -l $(bart bitmask 10 13) copy ksp_in.fifo out.fifo &
		else
			bart -r ksp_in.fifo -l $(bart bitmask 2 10 13) copy ksp_in.fifo - |\
			bart reshape -s $(bart bitmask 2 10 13)				\
			        $(bart bitmask 0 1 2 10 13)				\
			        1 $samples $spokes $frames $slices - out.fifo &
		fi

		# Cut off frames at the end when using turn-based reconstruction with more than $TURNS frames
		remainder=$((frames - (TURNS * (frames / TURNS))))
		if [ $TURNS -gt 1 -a $frames -gt $TURNS -a $remainder -gt 0 ]; then
			bart -l $(bart bitmask 10 13) -e $((frames - remainder)):$slices copy out.fifo out2.fifo &
		else
			bart -l $(bart bitmask 10 13) -r out.fifo			 copy out.fifo out2.fifo &
		fi

		bart -l $(get_output_flags $slices) -r out2.fifo copy out2.fifo $dst
	}

	f &
	bart tee -i $src -n --out0 meta.fifo ksp_in.fifo
	wait
)

# Arguments:
# $1 = Input File (stream) to be reordered.
# $2 = Rebinned and reordered output file.
#
# Environment parameters:
# SPOKES_PER_FRAME = Spokes per frame after raga rebinning.
# TINY =
rebin_raga() (

	SRC=$(get_file $1)
	DST=$(get_file $2)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	f() {
		dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)

		# Full frame
		spokes=$(echo $dims | cut -f3 -d' ')
		frames=$(echo $dims | cut -f11 -d' ')
		spokes2=$SPOKES_PER_FRAME

		all_spokes=$((spokes * frames))
		frames2=$((all_spokes / spokes2))
		all_spokes2=$((frames2 * spokes2))

		# Calculate raga indices
		bart raga -s $TINY $spokes ind
		bart index 0 $all_spokes - |\
		bart reshape $(bart bitmask 0 1 2 10) 1 1 $spokes $frames - range

		bart bin -o ind range order
		bart reshape 1028 $all_spokes 1 order order_flat

		# Remove extra input
		pad=$(($all_spokes-$all_spokes2))

		# Rebin
		bart reshape -s1028 1028 $all_spokes 1 ksp.fifo - |\
		bart bin --stream -o order_flat - - |\
		bart -l4 -e $all_spokes2 copy - - |\
		bart reshape -s1028 1028 $spokes2 $frames2 - - |\
		bart -r - -l1024 copy - $DST
	}

	f &
	bart tee -i $SRC -n --out0 meta.fifo ksp.fifo
	wait
)

# simulate_raga <in> <out>
# Environment parameters:
# DELAY = Delay in milliseconds between successive spokes, to simulate acquisition time.
# TINY: tiny factor
simulate_raga() (

	SRC=$(get_file $1)
	DST=$(get_file $2)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

        SPOKES=$(bart show -d 2 $SRC)
        FRAMES=$(bart show -d 10 $SRC)
        ALL_SPOKES=$((SPOKES * FRAMES))
	: "${TINY:=1}"
	: "${DELAY:=0}"

        # Calculate raga order
        bart raga -s$TINY $SPOKES ind
        bart index 0 $ALL_SPOKES - |\
        bart reshape $(bart bitmask 0 1 2 10) 1 1 $SPOKES $FRAMES - range

        bart bin -o ind range - |\
        bart reshape 1028 $ALL_SPOKES 1 - order_flat
        bart bin -l3 ind range - |\
        bart reshape 1028 $ALL_SPOKES 1 - inv_order_flat

        bart reshape $(bart bitmask 2 10) $ALL_SPOKES 1 $SRC - |\
	bart bin -o order_flat - - |\
	bart bin --nlout -o --stream inv_order_flat - - |\
	bart reshape -s $(bart bitmask 2 10) $(bart bitmask 2 10) $SPOKES $FRAMES - - |\
	bart -r - copy --delay $DELAY - $DST

)



# stack_shift <max_shift> <stack_dim> <in> <out>
# Create <max_shift> shifted (dim 10) copies of the input,
# joined along <stack_dim>
stack_shift() (

	max_shift=$1
	stack_dim=$2
	src=$(get_file $3)
	dst=$(get_file $4)

	delays=$((max_shift - 1))

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	flags=$(bart bitmask 10 13)

	f() {

		dims=$(get_dims meta.fifo)
		slices=$(get_dim 13 $dims)

		for i in $(seq $delays); do
			delay 10 $i $i input_$i.fifo delay_$i.fifo		&
		done

		bart -l$flags -r tmp.fifo \
			join $stack_dim \
			tmp.fifo $(seq -s' ' -f "delay_%g.fifo" $delays) -	|\
		bart copy --stream $(get_output_flags $slices) - $dst
	}

	f &
	bart copy --stream $flags $src - |\
	bart tee -n --out0 meta.fifo tmp.fifo $(seq -s' ' -f "input_%g.fifo" $delays)
	wait
)

## temporal median filter along dim 10
filter () (

	win=$1
	src=$(get_file $2)
	dst=$(get_file $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	flags=$(bart bitmask 10 13)

	stack_shift $win 11 $src -				|\
	bart -r - -l$flags	filter -m11 -l$win - -		|\
	bart copy --stream $flags - $dst
)


sliding_window() (

	win=$1
	src=$(get_file $2)
	dst=$(get_file $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	f() {

		dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)

		spokes=$(echo $dims | cut -f3 -d' ')
		frames=$(echo $dims | cut -f11 -d' ')

		stack_shift $win 9 in.fifo -						|\
		bart -r - reshape $(bart bitmask 2 9) $((spokes * win)) 1 - $dst
	}
	f &
	bart tee -n --out0 meta.fifo -i $src in.fifo
	wait
)

rl_filter_ksp () (

	KSP=$(get_file $1)
	TRJ=$(get_file $2)
	OUT=$(get_file $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart -r $TRJ rss 1 $TRJ filter.fifo							&
	bart -r $KSP fmac filter.fifo $KSP $OUT
)


trajectory () (

	# generate trajectory and correct gradient delays with ring
	# gradient delays are taken from previous turns shifted by DELAY

	KSP=$(readlink -f $1)
	DST=$(readlink -f $2)

	GRAD_DELAYS=""
	if [ $# -eq 3 ]; then
		GRAD_DELAYS=$(readlink -f $3)
	fi

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	if $RAGA; then
		ksp_flags=$(bart bitmask 10)
	else
		ksp_flags=$(bart bitmask 10 13)
	fi

	bart tee -i $KSP --out0 meta.fifo			|\
	bart copy --stream $ksp_flags - ksp_tmp.fifo	&

	dims=$(get_dims meta.fifo)
	READ=$(($(get_dim 1 $dims) / 2))
	PHS1=$(get_dim 2 $dims)
	TOT=$(get_dim 10 $dims)
	SLC=$(get_dim 13 $dims)
	batch_size=$(get_batch_size $TOT $TURNS)
	batch_spokes=$((PHS1*batch_size))
	batch_frames=$(get_batch_frames $TOT $TURNS)

	DST1=$DST
	trj_gd=
	if $GD_CORR; then
		DST1=trj.fifo
		trj_gd=trj_gd.fifo
	fi

	if $RAGA; then
		TOPTS="$TOPTS -x$READ -y$SPOKE_FF"

		bart traj $TOPTS - 					|\
		bart repmat 10 $FULL_FRAMES - trj_full
		rebin_raga trj_full trj_rebinned.fifo			&
		bart tee -n -i trj_rebinned.fifo $trj_gd $DST1	 	&

		if $GD_CORR; then
			bart -r ksp_tmp.fifo copy ksp_tmp.fifo ksp_gd.fifo &
		fi
	else
		TOPTS="$TOPTS -x$READ -y$PHS1 -t$TURNS"

		bart traj $TOPTS trj_tmp
		if [ $TOT -le $TURNS ]; then
			bart resize 10 $TOT trj_tmp trj_tmp
		fi

		if $SLW && ! $GD_CORR; then
			bart repmat 11 $((TOT / TURNS)) trj_tmp -		|\
			bart reshape $(bart bitmask 10 11) $TOT 1 - $DST1 	&
		else
			bart copy trj_tmp $DST1					&
		fi

		if $GD_CORR; then
			bart reshape $(bart bitmask 2 10)			\
				$batch_spokes 1 trj_tmp $trj_gd			&

			bart 	reshape -s $(bart bitmask 11 13)\
				$(bart bitmask 2 10 11 13) 	\
				$batch_spokes 1 $batch_frames $SLC 	\
				ksp_tmp.fifo ksp_gd.fifo 			&
		fi
	fi

	if $GD_CORR; then
		echo "estdelay options: $ESTDELAY_OPTS"				>&2

		bart -t4 -r ksp_gd.fifo estdelay $ESTDELAY_OPTS			\
			$trj_gd ksp_gd.fifo - 					|\
		bart tee -n $GRAD_DELAYS predelay.fifo 				&

		delay 11 $DELAY $DELAY predelay.fifo delays.fifo		&

		bart -t4 -r delays.fifo -l$(bart bitmask 11 13)			\
			trajcor -V delays.fifo $DST1 trj_cor.fifo		&

		if $RAGA; then
			bart -r trj_cor.fifo copy trj_cor.fifo $DST		&
		else
			bart reshape -s $(bart bitmask 10 13) $(bart bitmask 2 10 11 13) 	\
				$PHS1 $TOT 1 $SLC trj_cor.fifo -		|\
			bart -r - -l $(get_output_flags $SLC) copy - $DST
		fi
	else
		bart show -m ksp_tmp.fifo > /dev/null &
	fi

	# prevent early deletion of static files:
	wait
)

get_batch_size() {

	frames=$1
	turns=$2

	batch_size=$turns
	if [ $frames -lt $batch_size ]; then
		batch_size=$frames;
	fi
	echo $batch_size
}

get_batch_frames() {

	frames=$1
	turns=$2

	batch_frames=$((frames / turns))
	if [ $batch_frames -lt 1 ]; then
		batch_frames=1;
	fi
	echo $batch_frames
}

batch_turns() (

	src=$(get_file $1)
	dst=$(get_file $2)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit


	f () {

		dims=$(get_dims meta.fifo)

		spokes=$(get_dim 2 $dims)
		frames=$(get_dim 10 $dims)
		slices=$(get_dim 13 $dims)

		batch_size=$(get_batch_size $frames $TURNS)
		batch_spokes=$((spokes*batch_size))
		batch_frames=$(get_batch_frames $frames $TURNS)

		if [ 1 -eq $slices ]; then
			flags=$(bart bitmask 10)
			bart reshape -s $flags $(bart bitmask 2 10) \
				$batch_spokes $batch_frames in.fifo $dst
		else
			flags=$(bart bitmask 10 13)
			bart reshape -s $flags $(bart bitmask 2 10 13) \
				$batch_spokes $batch_frames $slices in.fifo $dst
		fi
	}
	f &
	bart tee -i $src --out0 meta.fifo -n in.fifo
)

coilcompression_svd () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo |\
		bart -r - copy - ksp_tmp.fifo &

	dims=$(get_dims meta.fifo)
	PHS=$(get_dim 2 $dims)
	TOT=$(get_dim 10 $dims)
	SLC=$(get_dim 13 $dims)

	STREAM_SLC="13"
	if [ 1 -eq $SLC ]; then
		STREAM_SLC=""
		SLC=""
	fi

	flags=$(bart bitmask 10 $STREAM_SLC)


	mkfifo $TRJ || true
	cat $TRJ > /dev/null &

	bart	  tee -i ksp_tmp.fifo tmp.fifo								| \
	batch_turns - -											| \
	bart -r - cc -M - predelay.fifo									&

	delay 10 $DELAY $DELAY predelay.fifo cc.fifo							&

	bart -r cc.fifo	repmat 9 $(get_batch_size $TOT $TURNS) cc.fifo -				| \
	bart		reshape -s $flags  $(bart bitmask 9 10 $STREAM_SLC) 1 $TOT $SLC - - 			| \
	bart -r -	ccapply -p$CHANNELS -- tmp.fifo - $DST
)

coilcompression_svd_first () (

	# SVD based coil compression
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -n -i $KSP --out0 meta.fifo ksp_tmp.fifo &
	dims=$(get_dims meta.fifo)
	PHS=$(get_dim 2 $dims)
	TOT=$(get_dim 10 $dims)
	SLC=$(get_dim 13 $dims)

	if [ $SLC -ne 1 ]; then
		flags=$(bart bitmask 10 13)
	else
		flags=$(bart bitmask 10)
	fi

	mkfifo $TRJ || true;
	cat $TRJ > /dev/null &

	bart copy --stream $flags ksp_tmp.fifo -				|\
	bart tee tmp.fifo							| \
	batch_turns - -								| \
	bart -r - cc -M -- - - | bart tee -n cc.fifo				&

	bart -l $(bart bitmask 10) -e 1 copy cc.fifo cc2

	bart -r tmp.fifo ccapply -p$CHANNELS -- tmp.fifo cc2 $DST
)


coilcompression_rovir () (

	# ROVir based coil compression

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	bart tee -i $KSP --out0 meta.fifo	| \
	bart copy --stream 1024 - ksp_tmp.fifo	&

	dims=$(bart show -m meta.fifo | tail -n1 | cut -f2-)
	READ=$(($(echo $dims | cut -f2 -d' ')/2))
	PHS=$(echo $dims | cut -f3 -d' ')
	TOT=$(echo $dims | cut -f11 -d' ')

	bart ones 2 20 20 o
	bart resize -c 0 40 1 40 o pos

	bart ones 2 25 25 o
	bart resize -c 0 40 1 40 o t
	bart ones 2 40 40 o
	bart saxpy -- -1 t o neg

	topts=(-o2 -r -D -l -x"$READ" -y"$PHS" -t"$TURNS" -O)

	bart traj "${topts[@]}" -- - | bart reshape -- $(bart bitmask 2 10) $((TURNS*PHS)) 1 - trj
	bart scale 2 trj trjos
	DIMS=40:40:1
	bart nufftbase $DIMS trjos pat


	bart		tee -i ksp_tmp.fifo tmp.fifo								| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - ksp_rovir.fifo	&

	bart -r $TRJ copy $TRJ -										| \
	bart -t4 -r -	scale -- 2 - - 										| \
	bart 		reshape -s1024 -- $(bart bitmask 2 10) $((PHS*TURNS)) $((TOT/TURNS)) - -		| \
	bart 		tee trj_rovir1.fifo trj_rovir2.fifo							| \
	bart -r - 	nufft $GPU -p pat -i -x$DIMS -- - ksp_rovir.fifo -			 		| \
	bart 		tee cim1.fifo > cim2.fifo								&

	bart -r cim1.fifo	fmac -- cim1.fifo pos ipos.fifo							&
	bart -r cim2.fifo	fmac -- cim2.fifo neg ineg.fifo							&

	bart -t4 -r trj_rovir1.fifo	nufft -p pat -- trj_rovir1.fifo ipos.fifo pos.fifo			&
	bart -t4 -r trj_rovir2.fifo	nufft -p pat -- trj_rovir2.fifo ineg.fifo neg.fifo			&

	bart -t4 -r pos.fifo		rovir -- pos.fifo neg.fifo predelay.fifo				&

	delay 10 $DELAY $DELAY predelay.fifo cc.fifo 								&

	bart -r cc.fifo			repmat -- 9 $TURNS cc.fifo -						| \
	bart				reshape -s1024 -- $(bart bitmask 9 10) 1 $TOT - - 			| \
	bart -r -			ccapply -p$CHANNELS -- tmp.fifo - $DST
)

coilcompression_geom () (

	# SVD Coil Compression with alignment along time dim.
	# TRJ is void but provided for easy replacement with rovir

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo $TRJ || true
	cat $TRJ > /dev/null &

	f() {
		dims=$(get_dims meta.fifo)
		PHS=$(get_dim 2 $dims)
		TOT=$(get_dim 10 $dims)
		SLC=$(get_dim 13 $dims)

		if [ $SLC -ne 1 ]; then
			error "Dynamic/Geometric (along t) coil compression \
				currently only works with one slice!"
		fi

		batch_turns in.fifo -						| \
		bart -r - cc -M - -						|\
		delay 10 $DELAY $DELAY - -					|\
		bart -r -	repmat 9 $(get_batch_size $TOT $TURNS)  - -	|\
		bart		reshape -s$(bart bitmask 10) \
				$(bart bitmask 9 10) 1 $TOT -  cc.fifo		&


		bart ccapply -A10 -p$CHANNELS in2.fifo cc.fifo $DST
	}
	f &

	bart tee -n -i $KSP --out0 meta.fifo in.fifo in2.fifo
	wait
)


coilcompression_none () (

	KSP=$(readlink -f $1)
	TRJ=$(readlink -f $2)
	DST=$(readlink -f $3)

	WORKDIR=$(mktemp --tmpdir -d $TMP_TEMPLATE 2>/dev/null)
	trap 'rm -rf "$WORKDIR"' EXIT
	cd "$WORKDIR" || exit

	mkfifo $TRJ || true;
	cat $TRJ > /dev/null &

	bart -r $KSP copy $KSP $DST;
)


(return 0 2>/dev/null) && sourced=1 || sourced=0

if [ $sourced -eq 0 ]; then
	main "$@"
fi
