#!/usr/bin/env bash

# print out the ID
echo $SLURM_JOB_ID

SWEEP_ID=$1
SWEEP_PATH=$2
LOGDIR_NAME=$3
NUM_AGENTS=$4
AGENTS_PER_DEVICE=$5
DEVICES=4

HOSTNAME=$(cat /etc/hostname)

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )
cd $SCRIPT_DIR/..

LOG_DIR=$(pwd)/log/$LOGDIR_NAME
mkdir -p $LOG_DIR

>&2 echo "Starting $NUM_AGENTS agents"
>&2 echo "with $AGENTS_PER_DEVICE agents per GPU..."

for ((i=1; i <= NUM_AGENTS; i++))
do
    LOG_FILE="$LOG_DIR/$i.log"
    AGENT_COMMAND="wandb agent $SWEEP_PATH"
    if [ -z $NIX_STORE ]; then
        AGENT_COMMAND="nix develop .#job --command $AGENT_COMMAND"
    fi
    # name
    NAME="agent-$i"

    echo "$NAME"
    echo "$LOG_FILE"

    run_agent() {
        DEVICE=$(((i + 1) / AGENTS_PER_DEVICE - 1))
        DEVICE=$(( DEVICE % DEVICES))
        export CUDA_VISIBLE_DEVICES=$DEVICE
        export XLA_PYTHON_CLIENT_MEM_FRACTION=0.45
        echo "Running agent $NAME"
        echo "  log file: $LOG_FILE"
        echo "  command: $AGENT_COMMAND"
        echo "  host $HOSTNAME"
        echo "  visible devices $CUDA_VISIBLE_DEVICES"
        echo ""
        exec $AGENT_COMMAND
    }
    run_agent &> $LOG_FILE &
done
echo "--initialized--"
sleep infinity