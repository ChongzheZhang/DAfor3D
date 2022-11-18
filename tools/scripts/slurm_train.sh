#!/usr/bin/env bash

set -x

PARTITION=$1
JOB_NAME=$2
GPUS=$3
PY_ARGS=${@:4}

GPUS_PER_NODE=${GPUS_PER_NODE:-2}
CPUS_PER_TASK=${CPUS_PER_TASK:-5}
SRUN_ARGS=${SRUN_ARGS:-""}

while true
do
    PORT=$(( ((RANDOM<<15)|RANDOM) % 49152 + 10000 ))
    status="$(nc -z 127.0.0.1 $PORT < /dev/null &>/dev/null; echo $?)"
    if [ "${status}" != "0" ]; then
        break;
    fi
done
echo $PORT
module load cuda/11.1
pyenv activate open
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --output=${JOB_NAME}%j.%N.out
    --gpus=geforce_rtx_2080_ti:2 \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --time=168:00:00 \
    --mem=24G \
    --kill-on-bad-exit=1 \
    ${SRUN_ARGS} \
    python -u train.py --launcher slurm --tcp_port $PORT ${PY_ARGS}
