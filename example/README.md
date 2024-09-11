# Walkthrough

## Overview

This section explains how to create and launch a preempt-able [wandb sweep](https://docs.wandb.ai/guides/sweeps) on a SLURM cluster. You have to set up three files:

1. A training script (e.g. `train.py`) that the sweep will execute multiple times using different hyper-parameters.

2. A `wandb` sweep configuration script (e.g. `sweep.yaml`) that defines the hyper-parameter search space.

3. A SLURM launch script (e.g. `launch.sh`) which we will use to submit runs on the cluster.

The repository's [`example`](https://github.com/f-dangel/wandb_preempt/tree/main/example) directory contains examples for each of these files, and here we will demonstrate how to make them work together. We will operate inside the example directory, so let's navigate to it:
```bash
# If you haven't already, you'll need to clone the repository first
git clone git@github.com:f-dangel/wandb_preempt.git && cd wandb_preempt

# And pip install the package from the repository
pip install -e .[example]

# Then navigate to the example directory within the repo
cd example
```

## Training Script

First up, we need to write a training script that we will sweep over. The sweep will call this script with different hyper-parameters to find the hyper-parameters that work best.

For demonstration purposes, we will train a small CNN on MNIST using SGD, and our goal is to find a good learning rate through random search using a `wandb` sweep. To keep things simple and cheap, we fix a batch size and use a (very) small number of epochs. Finally, we also use a learning rate scheduler and mixed-precision training with a gradient scaler. These are overkill for MNIST, of course, but important when training large models so we include them here to show how they are checkpointed too. In summary, we will call the training script using the following pattern:
```bash
python train.py --lr=X
```
with `X` being some floating point number that the sweep will search over.

The training script also contains code to checkpoint the training loop at the end of an epoch, so we can resume training after our pre-emptable job is interrupted. Roughly speaking, we need to create a `Checkpointer` that is responsible for saving and storing checkpoints, listening to signals from the Slurm process, and for requeuing the job on the cluster if it runs out of time before it is finished. Every time we call the checkpointer's `.step` method, it will save a checkpoint and check whether a signal has been sent by Slurm that indicates our job is about to be killed and so we must preemptively halt it and requeue the job.
For more details, please expand the code snippet below.

/// details | Details of the training script `example/train.py` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/train.py))
```py hl_lines="37-38 40-41"
--8<-- "example/train.py"
```
///

## Sweep Configuration

Our next goal will be to define and create a sweep.

For that, we need to write a `.yaml` file which specifies how the training script is called, and what the search space looks like. To learn more, take a look at the [Weights & Biases documentation](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).

The following configuration file defines a random search over the learning rate, using a log-uniform density for the search space.
By default, the example config will create a new project called `example-preemptable-sweep` owned by your default wandb entity (controlled by your [Default team setting](https://wandb.ai/settings)).

/// details | Details of the sweep configuration `example/sweep.yaml` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/sweep.yaml))
```yaml hl_lines="1"
--8<-- "example/sweep.yaml"
```
///

Let's create a sweep using this configuration:
```bash
wandb sweep sweep.yaml
```
/// details | Output
```sh hl_lines="4"
wandb: Creating sweep from: sweep.yaml
wandb: Creating sweep with ID: qmzevsi8
wandb: View sweep at: https://wandb.ai/f-dangel-team/example-preemptable-sweep/sweeps/qmzevsi8
wandb: Run sweep agent with: wandb agent f-dangel-team/example-preemptable-sweep/qmzevsi8
```
///

Each sweep has its own ID (you can have more than one sweep in the same project), so to launch jobs in this sweep that we've just recreated, we'll need to launch them using the correct sweep ID.
To do this, note the command in the last line of the output—copy this to use later.

Navigate to the `wandb` web interface and you should be able to see the sweep now exists:

![empty sweep](./assets/01_empty_sweep.png)

### Optional Step: Local Run

To make sure the configuration file works, I will execute a single run locally on my machine as a sanity check. This step is optional and obviously not recommended if your machine's hardware is not beefy enough (note that I use the command from above, but add the `--count=1` flag to carry out only a single run):

```bash
wandb agent --count=1 f-dangel-team/example-preemptable-sweep/aaq70gt8
```

/// details | Training script output
```
wandb: Starting wandb agent 🕵️
2024-09-09 22:16:59,894 - wandb.wandb_agent - INFO - Running runs: []
2024-09-09 22:17:00,448 - wandb.wandb_agent - INFO - Agent received command: run
2024-09-09 22:17:00,449 - wandb.wandb_agent - INFO - Agent starting run with config:
        epochs: 20
        lr: 0.005417051459795853
2024-09-09 22:17:00,452 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python train.py --epochs=20 --lr=0.005417051459795853
2024-09-09 22:17:05,473 - wandb.wandb_agent - INFO - Running runs: ['lmp9ebp1']
wandb: Currently logged in as: f-dangel (f-dangel-team). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.17.9
wandb: Run data is saved locally in ~/wandb_preempt/example/wandb/run-20240909_221710-lmp9ebp1
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run denim-sweep-1
wandb: ⭐️ View project at https://wandb.ai/f-dangel-team/example-preemptable-sweep
wandb: 🧹 View sweep at https://wandb.ai/f-dangel-team/example-preemptable-sweep/sweeps/qmzevsi8
wandb: 🚀 View run at https://wandb.ai/f-dangel-team/example-preemptable-sweep/runs/lmp9ebp1
Using SGD with learning rate 0.005417051459795853.
[0.0 s | 2024-09-09 22:17:14.343841] Creating checkpoint directory: ~/wandb_preempt/example/checkpoints.
[0.0 s | 2024-09-09 22:17:14.343944] SLURM job ID: None, array ID: None, task ID: None
[0.0 s | 2024-09-09 22:17:14.344300] No checkpoint found. Starting from scratch.
Epoch 0, Step 0, Loss 2.31327e+00
Epoch 0, Step 50, Loss 2.31236e+00
Epoch 0, Step 100, Loss 2.29828e+00
Epoch 0, Step 150, Loss 2.28944e+00
Epoch 0, Step 200, Loss 2.28544e+00
[2.8 s | 2024-09-09 22:17:17.181593] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000000.pt.
Epoch 1, Step 0, Loss 2.27385e+00
Epoch 1, Step 50, Loss 2.27472e+00
Epoch 1, Step 100, Loss 2.25237e+00
Epoch 1, Step 150, Loss 2.22541e+00
Epoch 1, Step 200, Loss 2.21894e+00
[5.6 s | 2024-09-09 22:17:19.918572] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000001.pt.
[5.6 s | 2024-09-09 22:17:19.919702] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000000.pt.
Epoch 2, Step 0, Loss 2.15103e+00
Epoch 2, Step 50, Loss 2.11397e+00
Epoch 2, Step 100, Loss 2.02360e+00
Epoch 2, Step 150, Loss 1.89612e+00
Epoch 2, Step 200, Loss 1.77511e+00
[8.3 s | 2024-09-09 22:17:22.654042] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000002.pt.
[8.3 s | 2024-09-09 22:17:22.655190] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000001.pt.
Epoch 3, Step 0, Loss 1.64844e+00
Epoch 3, Step 50, Loss 1.41444e+00
Epoch 3, Step 100, Loss 1.24010e+00
Epoch 3, Step 150, Loss 1.09945e+00
Epoch 3, Step 200, Loss 9.38731e-01
[11.0 s | 2024-09-09 22:17:25.391665] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000003.pt.
[11.0 s | 2024-09-09 22:17:25.392811] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000002.pt.
Epoch 4, Step 0, Loss 8.39498e-01
Epoch 4, Step 50, Loss 7.73968e-01
Epoch 4, Step 100, Loss 7.30065e-01
Epoch 4, Step 150, Loss 5.72235e-01
Epoch 4, Step 200, Loss 6.52813e-01
[13.8 s | 2024-09-09 22:17:28.133137] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000004.pt.
[13.8 s | 2024-09-09 22:17:28.134257] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000003.pt.
Epoch 5, Step 0, Loss 5.88400e-01
Epoch 5, Step 50, Loss 5.26665e-01
Epoch 5, Step 100, Loss 5.13180e-01
Epoch 5, Step 150, Loss 5.11152e-01
Epoch 5, Step 200, Loss 5.30249e-01
[16.5 s | 2024-09-09 22:17:30.880319] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000005.pt.
[16.5 s | 2024-09-09 22:17:30.881465] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000004.pt.
Epoch 6, Step 0, Loss 4.68262e-01
Epoch 6, Step 50, Loss 4.94186e-01
Epoch 6, Step 100, Loss 5.16372e-01
Epoch 6, Step 150, Loss 4.25439e-01
Epoch 6, Step 200, Loss 5.06348e-01
[19.3 s | 2024-09-09 22:17:33.594644] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000006.pt.
[19.3 s | 2024-09-09 22:17:33.595731] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000005.pt.
Epoch 7, Step 0, Loss 4.81532e-01
Epoch 7, Step 50, Loss 4.62612e-01
Epoch 7, Step 100, Loss 4.84398e-01
Epoch 7, Step 150, Loss 4.34298e-01
Epoch 7, Step 200, Loss 4.44681e-01
[21.9 s | 2024-09-09 22:17:36.292046] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000007.pt.
[21.9 s | 2024-09-09 22:17:36.293158] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000006.pt.
Epoch 8, Step 0, Loss 4.52199e-01
Epoch 8, Step 50, Loss 4.06567e-01
Epoch 8, Step 100, Loss 3.89836e-01
Epoch 8, Step 150, Loss 4.20403e-01
Epoch 8, Step 200, Loss 4.92639e-01
[24.7 s | 2024-09-09 22:17:39.035341] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000008.pt.
[24.7 s | 2024-09-09 22:17:39.036695] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000007.pt.
Epoch 9, Step 0, Loss 4.22285e-01
Epoch 9, Step 50, Loss 4.19266e-01
Epoch 9, Step 100, Loss 3.77887e-01
Epoch 9, Step 150, Loss 3.27919e-01
Epoch 9, Step 200, Loss 3.68881e-01
[27.4 s | 2024-09-09 22:17:41.770130] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000009.pt.
[27.4 s | 2024-09-09 22:17:41.771950] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000008.pt.
Epoch 10, Step 0, Loss 3.94362e-01
Epoch 10, Step 50, Loss 4.10359e-01
Epoch 10, Step 100, Loss 3.79892e-01
Epoch 10, Step 150, Loss 4.42654e-01
Epoch 10, Step 200, Loss 4.29457e-01
[30.1 s | 2024-09-09 22:17:44.474406] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000010.pt.
[30.1 s | 2024-09-09 22:17:44.475494] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000009.pt.
Epoch 11, Step 0, Loss 3.87773e-01
Epoch 11, Step 50, Loss 4.67186e-01
Epoch 11, Step 100, Loss 3.57660e-01
Epoch 11, Step 150, Loss 3.54450e-01
Epoch 11, Step 200, Loss 3.43293e-01
[32.8 s | 2024-09-09 22:17:47.191162] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000011.pt.
[32.8 s | 2024-09-09 22:17:47.192310] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000010.pt.
Epoch 12, Step 0, Loss 4.69982e-01
Epoch 12, Step 50, Loss 3.77749e-01
Epoch 12, Step 100, Loss 4.01224e-01
Epoch 12, Step 150, Loss 3.48171e-01
Epoch 12, Step 200, Loss 3.61441e-01
[35.6 s | 2024-09-09 22:17:49.924267] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000012.pt.
[35.6 s | 2024-09-09 22:17:49.925361] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000011.pt.
Epoch 13, Step 0, Loss 3.35668e-01
Epoch 13, Step 50, Loss 3.34111e-01
Epoch 13, Step 100, Loss 3.22253e-01
Epoch 13, Step 150, Loss 3.37382e-01
Epoch 13, Step 200, Loss 3.90798e-01
[38.3 s | 2024-09-09 22:17:52.667451] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000013.pt.
[38.3 s | 2024-09-09 22:17:52.668601] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000012.pt.
Epoch 14, Step 0, Loss 3.83771e-01
Epoch 14, Step 50, Loss 3.24402e-01
Epoch 14, Step 100, Loss 3.12806e-01
Epoch 14, Step 150, Loss 3.40762e-01
Epoch 14, Step 200, Loss 4.72378e-01
[41.0 s | 2024-09-09 22:17:55.393290] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000014.pt.
[41.1 s | 2024-09-09 22:17:55.394609] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000013.pt.
Epoch 15, Step 0, Loss 2.78277e-01
Epoch 15, Step 50, Loss 4.86174e-01
Epoch 15, Step 100, Loss 3.18030e-01
Epoch 15, Step 150, Loss 3.39720e-01
Epoch 15, Step 200, Loss 3.45562e-01
[43.8 s | 2024-09-09 22:17:58.183117] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000015.pt.
[43.8 s | 2024-09-09 22:17:58.184218] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000014.pt.
Epoch 16, Step 0, Loss 4.09605e-01
Epoch 16, Step 50, Loss 4.02795e-01
Epoch 16, Step 100, Loss 3.30050e-01
Epoch 16, Step 150, Loss 3.29713e-01
Epoch 16, Step 200, Loss 3.47872e-01
[46.6 s | 2024-09-09 22:18:00.935960] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000016.pt.
[46.6 s | 2024-09-09 22:18:00.937244] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000015.pt.
Epoch 17, Step 0, Loss 3.61874e-01
Epoch 17, Step 50, Loss 4.21045e-01
Epoch 17, Step 100, Loss 3.68629e-01
Epoch 17, Step 150, Loss 3.50573e-01
Epoch 17, Step 200, Loss 4.53324e-01
[49.3 s | 2024-09-09 22:18:03.630036] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000017.pt.
[49.3 s | 2024-09-09 22:18:03.631169] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000016.pt.
Epoch 18, Step 0, Loss 2.75897e-01
Epoch 18, Step 50, Loss 3.52856e-01
Epoch 18, Step 100, Loss 3.95981e-01
Epoch 18, Step 150, Loss 3.01999e-01
Epoch 18, Step 200, Loss 3.86351e-01
[52.0 s | 2024-09-09 22:18:06.381923] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000018.pt.
[52.0 s | 2024-09-09 22:18:06.383072] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000017.pt.
Epoch 19, Step 0, Loss 3.14332e-01
Epoch 19, Step 50, Loss 2.94699e-01
Epoch 19, Step 100, Loss 2.69020e-01
Epoch 19, Step 150, Loss 3.27822e-01
Epoch 19, Step 200, Loss 3.05174e-01
[54.8 s | 2024-09-09 22:18:09.101715] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000019.pt.
[54.8 s | 2024-09-09 22:18:09.103077] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000018.pt.
wandb:
wandb:
wandb: Run history:
wandb:       loss ████▇▇▆▄▃▃▂▂▂▁▂▁▂▁▁▁▁▂▁▁▂▁▁▁▁▁▂▁▁▁▁▁▁▁▁▁
wandb: loss_scale ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         lr ████████▇▇▇▇▇▇▆▆▆▆▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁
wandb:    resumes ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:       loss 0.30517
wandb: loss_scale 1.0
wandb:         lr 3e-05
wandb:    resumes 0
wandb:
wandb: 🚀 View run denim-sweep-1 at: https://wandb.ai/f-dangel-team/example-preemptable-sweep/runs/lmp9ebp1
wandb: ⭐️ View project at: https://wandb.ai/f-dangel-team/example-preemptable-sweep
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240909_221710-lmp9ebp1/logs
wandb: WARNING The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.
[62.1 s | 2024-09-09 22:18:16.395233] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/lmp9ebp1_00000019.pt.
2024-09-09 22:18:21,429 - wandb.wandb_agent - INFO - Cleaning up finished run: lmp9ebp1
wandb: Terminating and syncing runs. Press ctrl-c to kill.
```
///

On the Weights & Biases web API, we can see the successfully finished run:

![local run](./assets/02_local_run.png)

## SLURM Launcher

The last step is to launch multiple jobs on a SLURM cluster.

For that, we use the following launch script. If you are running this yourself, you will need to modify the wandb agent command to be the one we copied before, when the wandb sweep was created. Note that we still include the `--count=1` argument, to ensure each Slurm job in the array completes a single task from the sweep.
We will explain the script in more detail below; for now our focus is to launch jobs.

/// details | Details of the SLURM script `example/launch.sh` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/launch.sh))
```sh hl_lines="10-12 23"
--8<-- "example/launch.sh"
```
///

Log into your SLURM cluster, then navigate to the `example` directory and submit jobs to SLURM:

```sh
sbatch launch.sh
```

Use `watch squeue --me` to monitor the job queue. You will observe that the jobs will launch and run for a short amount of time before receiving the pre-emption signal from SLURM. After that, they will requeue themselves and pick up from the latest checkpoint, until training is completely finished.

On the Weights & Biases website, you will see the runs transitioning between the states 'Running', 'Preempted', and 'Finished'. Here is an example view:



## Details
The launch script divides into three parts:

1. **SLURM configuration:** The first block specifies the SLURM resources and task array we are about to submit (lines starting with `#SBATCH`). The important configurations are
    - `--time` (how long to request the job will run for?)
    - `--array` (how many jobs will be submitted?), and
    - `--signal` specifications (how much time before the limit will we start pre-empting?)

    *These values are optimized for demonstration purposes. You definitely want to tweak them for your use case.*

    Note that the `--time` request does not need to be the total amount of time the model takes to train, since the `checkpointer` will automatically requeue the job if/when time limit is about to be reached and the training script is still running.

    The `--signal` argument is used to tell the python script that the Slurm job is about to end (using the signal SIGUSR1). The time in the `--signal` argument should be set to (at least) the amount of time (in seconds) between calls to `checkpointer.step()` in your training script.
    In the example script, we use an epoch-based training routine, and so the time specified in the signal argument needs to be the amount of time taken to complete one training epoch and save the model, or longer.

    The other configuration flags are resource-specific, and some like the `--partition=a40` will depend on your cluster. In our script, we request NVIDIA A40 GPUs because they support mixed-precision training with `bfloat16` that is used by many modern training pipelines.

2. **Printing details and wandb agent launch** This part executes the `wandb agent` on our sweep and puts it into the background so our launch script can start listening to signals from SLURM.

3. **Installing a signal handler**
