#!/usr/bin/env bash

name=$1
logdir=$2
shift
shift
flags=$@

if [ ! -d "${logdir}" ]; then
	echo 'error!' && exit
fi
if [ ! -d "${logdir}/${name}" ]; then
    mkdir "${logdir}/${name}" || exit
fi

evallogdir="${logdir}/${name}/$(date +%Y%m%d_%H%M%S)"

if [ ! -d "${evallogdir}" ]; then
    mkdir "${evallogdir}" || exit
fi

stdbuf -o 0 python ${name}.py --log ${logdir} --eval_log ${evallogdir} ${flags} | tee ${evallogdir}/stdout
