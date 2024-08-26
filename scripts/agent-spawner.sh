#!/usr/bin/env bash

SESSION_NAME=$1
SWEEP_ID=$2
SWEEP_PATH=$3
NUM_AGENTS=$4
AGENTS_PER_DEVICE=$5

if [ -z "$SESSION_NAME" ]; then
    echo "No session name"
    exit 1
fi
if [ -z "$SWEEP_ID" ]; then
    echo "No sweep id"
    exit 1
fi
if [ -z "$NUM_AGENTS" ]; then
    echo "No num agents"
    exit 1
fi

SCRIPT_DIR=$( cd -- "$( dirname -- "${BASH_SOURCE[0]}" )" &> /dev/null && pwd )

AGENT_RUN=$SCRIPT_DIR/agent-run.sh
# Make the agents think they are not in nix
# since are srunning the agent-run
export NIX_STORE=

SLURM_COMMAND="srun -N 1 --gres=gpu:4 --mem=1TB -c64 --time 6:00:00 $AGENT_RUN $SWEEP_ID $SWEEP_PATH $SWEEP_ID $NUM_AGENTS $AGENTS_PER_DEVICE"
echo "Running: $SLURM_COMMAND"
exec 3< <($SLURM_COMMAND)

read <&3 JOB_ID
echo "Job id: $JOB_ID"
echo "Sweep: $SWEEP_PATH"

sleep 1

while :
do
    read <&3 NAME
    if [ $NAME = "--initialized--" ]; then
        break
    fi
    read <&3 LOG
    echo "Tailing log file: $LOG"
    tmux new-window -t $SESSION_NAME -n $NAME tail -f $LOG
done

echo "Done starting agents for sweep: $SWEEP_ID"

# Close all of the windows in the session when
# the spawner dies, and cancel the job
inter() {
    scancel $JOB_ID
    tmux kill-session -t $SESSION_NAME
}

trap 'inter' SIGINT
sleep infinity