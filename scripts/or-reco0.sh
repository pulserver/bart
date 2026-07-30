#!/bin/bash

: "${MARS:=root@192.168.2.2}"
: "${CUSTOM_SCRIPT:=/opt/medcom/MriCustomer/CustomerSeq/bart/eisbaer/reco.sh}"
: "${SCRIPT:=$BART_TOOLBOX_PATH/scripts/or-reco.sh}"

open_debug_shell() {

	echo "or-reco0.sh: Open debug shell" >&2
	rm -f tmp.fifo;
	mkfifo tmp.fifo;
	ncat localhost 22 < tmp.fifo |\
		ncat -w 1 $DEBUG_HOST 2221 > tmp.fifo
}

fwd_stdin() {

	rm -f tmp_in_$1_$2.fifo
	mkfifo tmp_in_$1_$2.fifo
	# order matters: tee output needs to go to ncat
	tee -p tmp_in_$1_$2.fifo | ncat -w 1 --send-only $1 $2 &
	cat tmp_in_$1_$2.fifo
}

# In OpenRecon mode, manually copy over scripts from mars
# In BartStreams mode, this happens via bind mount.
if [ 1 -ne "${OR_BART_STREAMS:-0}" -a "-" != $CUSTOM_SCRIPT ]; then
	mkdir -p /scanner
	scp -oStrictHostKeyChecking=no $MARS:$CUSTOM_SCRIPT /scanner/ ||\
		echo "no custom script found, using builtin." >&2
fi

if [ -f "/scanner/reco.sh" ]; then
	SCRIPT="/scanner/reco.sh"
fi



origin="${NCAT_REMOTE_ADDR:-stdin}${NCAT_REMOTE_PORT:+:}${NCAT_REMOTE_PORT:+}"

echo "or-reco0.sh: Con from $origin" >&2

if [ -n "${DEBUG_HOST:-}" ]; then
	open_debug_shell

	fwd_stdin $DEBUG_HOST 1234 |\
	$SCRIPT |\
	fwd_stdin $DEBUG_HOST 1235
else
	$SCRIPT
fi

wait

echo "or-reco0.sh: COMPLETE $origin" >&2
