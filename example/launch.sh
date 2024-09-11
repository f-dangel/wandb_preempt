#!/bin/bash
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --qos=m5
#SBATCH --open-mode=append
#SBATCH --time=00:04:00
#SBATCH --array=0-9
#SBATCH --signal=B:SIGUSR1@120  # Send signal SIGUSR1 120 seconds before the job hits the time limit

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) begins on $(hostname), submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"
echo ""

# wait for a specific time to avoid simultaneous API requests from multiple agents
if [ "$SLURM_ARRAY_TASK_COUNT" != "" ]; then
    sleep $((5 * ( SLURM_ARRAY_TASK_ID - SLURM_ARRAY_TASK_MIN) ))
fi

# NOTE that we need to use srun here, otherwise the Python process won't receive the SIGUSR1 signal
srun wandb agent --count=1 f-dangel-team/example-preemptable-sweep/4m89qo6r &
child="$!"

# Set up a handler to pass the SIGUSR1 to the python session launched by the agent
function term_handler()
{
    echo "$(date) ** Job $SLURM_JOB_NAME ($SLURM_JOB_ID) received SIGUSR1 **"
    # The Checkpointer will have written the PID of the Python process to a file
    # so we can send it the SIGUSR1 signal
    PID=$(cat "${SLURM_JOB_ID}.pid")
    echo "$(date) ** Sending kill signal to python process $PID **"
    # Send the signal multiple times because it may not be caught if the Python
    # process happens to be in the middle of writing a checkpoint. The while loop
    # exits when `kill` errors, which happens when the python process has exited.
    while kill -SIGUSR1 "$PID" 2>/dev/null
    do
        echo "$(date) Sent SIGUSR1 signal to python"
        sleep 10
    done
}

# Call this term_handler function when the job recieves the SIGUSR1 or SIGTERM signal
# SIGUSR1 is sent by SLURM 120s before the time limit, thanks to the SBATCH --signal=...
# setting in the header.
# SIGTERM is sent shortly* before the job is killed (*with interval between the signal
# and being properly killed depending on SLURM cluster's `GraceTime` value)
trap term_handler SIGUSR1
trap term_handler SIGTERM  # NOTE we trap SIGTERM but send SIGUSR1 to the Python process

# The srun command is running in the background, and we need to wait for it to finish.
# The wait command here is in the foreground and so it will be interrupted by the trap
# handler when we receive a SIGUSR1 or SIGTERM signal.
wait "$child"
# Clean up the pid file
rm "${SLURM_JOB_ID}.pid"
echo "$(date) Reached EOF"
