#!/usr/bin/env bash

name=$1
shift
flags=$@

if [ ! -d "logs" ]; then
	mkdir logs
fi
if [ ! -d "logs/${name}" ]; then
    mkdir "logs/${name}"
fi

logdir="logs/${name}/$(date +%Y%m%d_%H%M%S)"

if [ ! -d "${logdir}" ]; then
    mkdir "${logdir}"
fi

stdbuf -o 0 python ${name}.py --log ${logdir} ${flags} | tee ${logdir}/stdout
