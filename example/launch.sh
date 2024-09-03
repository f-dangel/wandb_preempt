#!/bin/bash
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:04:00
#SBATCH --qos=m5
#SBATCH --array=0-19
#SBATCH --signal=B:SIGUSR1@120  # Send signal SIGUSR1 120 seconds before the job hits the time limit
#SBATCH --open-mode=append

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) begins on $(hostname), submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"
echo ""

# wait for a random amount of time to make runs start agents at different times
MINWAIT=0
MAXWAIT=15
sleep $((5 * SLURM_ARRAY_TASK_ID))

# NOTE that we need to use srun here, otherwise the Python process won't receive the SIGUSR1 signal
srun --unbuffered wandb agent --count=1 f-dangel-team/quickstart/i75puhon &
child="$!"

# Set up a handler to pass the SIGUSR1 to the python session launched by the agent
function term_handler()
{
    echo "$(date) ** Job $SLURM_JOB_NAME ($SLURM_JOB_ID) received SIGUSR1 at $(date) **"
    # The CheckpointHandler will have written the PID of the Python process to a file
    # so we can send it the SIGUSR1 signal
    PID=$(cat "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pid")
    echo "$(date) ** Sending kill signal to python process $PID **"
    # Send the signal multiple times because it may not be caught if the Python
    # process happens to be in the middle of writing a checkpoint. The while loop
    # exits when `kill` errors, which happens when the python process has exited.
    while kill -SIGUSR1 "$PID" 2>/dev/null
    do
        echo "$(date) Sent kill signal"
        sleep 10
    done
}

# Call this term_handler function when the job recieves the SIGUSR1 signal
trap term_handler SIGUSR1

# The srun command is running in the background, and we need to wait for it to finish.
# The wait command here is in the foreground and so it will be interrupted by the trap
# handler when we receive a SIGUSR1 signal.
wait "$child"
# Clean up the pid file
rm "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pid"
echo "$(date) Reached EOF"
