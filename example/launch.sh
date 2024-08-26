#!/bin/bash
#SBATCH --partition=a40
#SBATCH --nodes=1
#SBATCH --tasks-per-node=1
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-gpu=4
#SBATCH --mem-per-gpu=16G
#SBATCH --time=00:06:00
#SBATCH --qos=m5
#SBATCH --array=0-2
#SBATCH --signal=B:SIGUSR1@60  # Send signal SIGUSR1 60 seconds before the job hits the time limit
#SBATCH --open-mode=append

echo "Job $SLURM_JOB_NAME ($SLURM_JOB_ID) begins on $(hostname), submitted from $SLURM_SUBMIT_HOST ($SLURM_CLUSTER_NAME)"
echo ""

# wait for a random amount of time to make runs start agents at different times
MINWAIT=0
MAXWAIT=15
sleep $((MINWAIT+RANDOM % (MAXWAIT-MINWAIT)))

# NOTE that we need to use srun here, otherwise the Python process won't receive the SIGUSR1 signal
srun --unbuffered wandb agent --count=1 f-dangel-team/quickstart/evv70pcr &
child="$!"

# Set up a handler to pass the SIGUSR1 to the python session launched by the agent
function term_handler()
{
    echo "$(date) ** Job $SLURM_JOB_NAME ($SLURM_JOB_ID) received SIGUSR1 at $(date) **"
    PID=$(cat "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pid")
    echo "$(date) ** Sending kill signal to python process $PID **"
    kill -SIGUSR1 "$PID"
    echo "$(date) Sent kill signal"
    wait "$child"
    echo "$(date) Finished waiting for child inside trap handler"
}

# Call this term_handler function when the job recieves the SIGUSR1 signal
trap term_handler SIGUSR1

# This will listen to signals, and then call the term handler
wait "$child"
# Clean up the pid file
rm "${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}.pid"
echo "$(date) Reached EOF"
