#!/bin/bash

: "${HOST:=0.0.0.0}"
: "${PORT:=9002}"
: "${SCRIPT:=$BART_TOOLBOX_PATH/scripts/or-reco0.sh}"

if [ -n "${DEBUG_HOST:-}" ]; then
	# openssh-server relies on systemd creating this directory
	mkdir /run/sshd
	chmod 0755 /run/sshd
	# Spawn ssh server for debugging
	/sbin/sshd -oPermitRootLogin=yes -oPermitEmptyPasswords=yes
fi

ncat -l $HOST $PORT --exec $SCRIPT 1>&2 &
sleep 1

if [ -n "$PID_FILE" ]; then
	touch $PID_FILE
fi

wait
