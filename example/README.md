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
python train.py --lr_max=X
```
with `X` being some floating point number that the sweep will search over.

The training script also contains code to checkpoint the training loop at the end of an epoch, so we can resume training after our pre-emptable job is interrupted. Roughly speaking, we need to create a `Checkpointer` that is responsible for saving and storing checkpoints, listening to signals from the Slurm process, and for requeuing the job on the cluster if it runs out of time before it is finished. Every time we call the checkpointer's `.step` method, it will save a checkpoint and check whether a signal has been sent by Slurm that indicates our job is about to be killed and so we must preemptively halt it and requeue the job.
For more details, please expand the code snippet below.

/// details | Details of the training script `example/train.py` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/train.py))
```py
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
wandb: Creating sweep with ID: 4m89qo6r
wandb: View sweep at: https://wandb.ai/f-dangel-team/example-preemptable-sweep/sweeps/4m89qo6r
wandb: Run sweep agent with: wandb agent f-dangel-team/example-preemptable-sweep/4m89qo6r
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
2024-09-11 15:50:13,390 - wandb.wandb_agent - INFO - Running runs: []
2024-09-11 15:50:13,596 - wandb.wandb_agent - INFO - Agent received command: run
2024-09-11 15:50:13,596 - wandb.wandb_agent - INFO - Agent starting run with config:
        epochs: 20
        lr_max: 0.002046505897436452
2024-09-11 15:50:13,597 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python train.py --epochs=20 --lr_max=0.002046505897436452
wandb: Currently logged in as: f-dangel (f-dangel-team). Use `wandb login --relogin` to force relogin
2024-09-11 15:50:18,609 - wandb.wandb_agent - INFO - Running runs: ['2zoz0rl8']
wandb: Tracking run with wandb version 0.17.9
wandb: Run data is saved locally in ~/wandb_preempt/example/wandb/run-20240911_155016-2zoz0rl8
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run unique-sweep-1
wandb: ⭐️ View project at https://wandb.ai/f-dangel-team/example-preemptable-sweep
wandb: 🧹 View sweep at https://wandb.ai/f-dangel-team/example-preemptable-sweep/sweeps/4m89qo6r
wandb: 🚀 View run at https://wandb.ai/f-dangel-team/example-preemptable-sweep/runs/2zoz0rl8
Using SGD with learning rate 0.002046505897436452.
[0.0 s | 2024-09-11 15:50:19.844072] Creating checkpoint directory: ~/wandb_preempt/example/checkpoints.
[0.0 s | 2024-09-11 15:50:19.844166] SLURM job ID: None, array ID: None, task ID: None
[0.0 s | 2024-09-11 15:50:19.844522] No checkpoint found. Starting from scratch.
Epoch 0, Step 0, Loss 2.31327e+00
Epoch 0, Step 50, Loss 2.31634e+00
Epoch 0, Step 100, Loss 2.30524e+00
Epoch 0, Step 150, Loss 2.29960e+00
Epoch 0, Step 200, Loss 2.30427e+00
[3.0 s | 2024-09-11 15:50:22.801220] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000000.pt.
Epoch 1, Step 0, Loss 2.29656e+00
Epoch 1, Step 50, Loss 2.30689e+00
Epoch 1, Step 100, Loss 2.29662e+00
Epoch 1, Step 150, Loss 2.28267e+00
Epoch 1, Step 200, Loss 2.30082e+00
[5.7 s | 2024-09-11 15:50:25.576881] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000001.pt.
[5.7 s | 2024-09-11 15:50:25.578131] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000000.pt.
Epoch 2, Step 0, Loss 2.28091e+00
Epoch 2, Step 50, Loss 2.27995e+00
Epoch 2, Step 100, Loss 2.27886e+00
Epoch 2, Step 150, Loss 2.27613e+00
Epoch 2, Step 200, Loss 2.27744e+00
[8.5 s | 2024-09-11 15:50:28.328779] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000002.pt.
[8.5 s | 2024-09-11 15:50:28.329942] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000001.pt.
Epoch 3, Step 0, Loss 2.27019e+00
Epoch 3, Step 50, Loss 2.27712e+00
Epoch 3, Step 100, Loss 2.26028e+00
Epoch 3, Step 150, Loss 2.25132e+00
Epoch 3, Step 200, Loss 2.25152e+00
[11.2 s | 2024-09-11 15:50:31.057194] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000003.pt.
[11.2 s | 2024-09-11 15:50:31.058385] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000002.pt.
Epoch 4, Step 0, Loss 2.23886e+00
Epoch 4, Step 50, Loss 2.24897e+00
Epoch 4, Step 100, Loss 2.23878e+00
Epoch 4, Step 150, Loss 2.21464e+00
Epoch 4, Step 200, Loss 2.21080e+00
[14.0 s | 2024-09-11 15:50:33.822246] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000004.pt.
[14.0 s | 2024-09-11 15:50:33.823408] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000003.pt.
Epoch 5, Step 0, Loss 2.19485e+00
Epoch 5, Step 50, Loss 2.19484e+00
Epoch 5, Step 100, Loss 2.16891e+00
Epoch 5, Step 150, Loss 2.16754e+00
Epoch 5, Step 200, Loss 2.13477e+00
[16.7 s | 2024-09-11 15:50:36.518798] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000005.pt.
[16.7 s | 2024-09-11 15:50:36.519988] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000004.pt.
Epoch 6, Step 0, Loss 2.12859e+00
Epoch 6, Step 50, Loss 2.10682e+00
Epoch 6, Step 100, Loss 2.09931e+00
Epoch 6, Step 150, Loss 2.08149e+00
Epoch 6, Step 200, Loss 2.04833e+00
[19.4 s | 2024-09-11 15:50:39.239762] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000006.pt.
[19.4 s | 2024-09-11 15:50:39.240944] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000005.pt.
Epoch 7, Step 0, Loss 2.02058e+00
Epoch 7, Step 50, Loss 1.97293e+00
Epoch 7, Step 100, Loss 1.94745e+00
Epoch 7, Step 150, Loss 1.90756e+00
Epoch 7, Step 200, Loss 1.89235e+00
[22.1 s | 2024-09-11 15:50:41.981751] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000007.pt.
[22.1 s | 2024-09-11 15:50:41.983563] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000006.pt.
Epoch 8, Step 0, Loss 1.82919e+00
Epoch 8, Step 50, Loss 1.80327e+00
Epoch 8, Step 100, Loss 1.74424e+00
Epoch 8, Step 150, Loss 1.68607e+00
Epoch 8, Step 200, Loss 1.67496e+00
[25.0 s | 2024-09-11 15:50:44.813117] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000008.pt.
[25.0 s | 2024-09-11 15:50:44.814475] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000007.pt.
Epoch 9, Step 0, Loss 1.62010e+00
Epoch 9, Step 50, Loss 1.56824e+00
Epoch 9, Step 100, Loss 1.50516e+00
Epoch 9, Step 150, Loss 1.48588e+00
Epoch 9, Step 200, Loss 1.44233e+00
[27.8 s | 2024-09-11 15:50:47.612270] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000009.pt.
[27.8 s | 2024-09-11 15:50:47.613580] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000008.pt.
Epoch 10, Step 0, Loss 1.37147e+00
Epoch 10, Step 50, Loss 1.34652e+00
Epoch 10, Step 100, Loss 1.31548e+00
Epoch 10, Step 150, Loss 1.31214e+00
Epoch 10, Step 200, Loss 1.31763e+00
[30.5 s | 2024-09-11 15:50:50.342514] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000010.pt.
[30.5 s | 2024-09-11 15:50:50.343644] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000009.pt.
Epoch 11, Step 0, Loss 1.18105e+00
Epoch 11, Step 50, Loss 1.18585e+00
Epoch 11, Step 100, Loss 1.10869e+00
Epoch 11, Step 150, Loss 1.08874e+00
Epoch 11, Step 200, Loss 1.06454e+00
[33.4 s | 2024-09-11 15:50:53.222490] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000011.pt.
[33.4 s | 2024-09-11 15:50:53.223652] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000010.pt.
Epoch 12, Step 0, Loss 1.13357e+00
Epoch 12, Step 50, Loss 9.96835e-01
Epoch 12, Step 100, Loss 1.06371e+00
Epoch 12, Step 150, Loss 9.63902e-01
Epoch 12, Step 200, Loss 9.63633e-01
[36.1 s | 2024-09-11 15:50:55.968639] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000012.pt.
[36.1 s | 2024-09-11 15:50:55.970050] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000011.pt.
Epoch 13, Step 0, Loss 9.34712e-01
Epoch 13, Step 50, Loss 8.95310e-01
Epoch 13, Step 100, Loss 9.12703e-01
Epoch 13, Step 150, Loss 9.39363e-01
Epoch 13, Step 200, Loss 9.21194e-01
[38.9 s | 2024-09-11 15:50:58.722370] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000013.pt.
[38.9 s | 2024-09-11 15:50:58.723519] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000012.pt.
Epoch 14, Step 0, Loss 8.49049e-01
Epoch 14, Step 50, Loss 9.19110e-01
Epoch 14, Step 100, Loss 8.95127e-01
Epoch 14, Step 150, Loss 8.37601e-01
Epoch 14, Step 200, Loss 9.13763e-01
[41.6 s | 2024-09-11 15:51:01.492518] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000014.pt.
[41.6 s | 2024-09-11 15:51:01.493670] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000013.pt.
Epoch 15, Step 0, Loss 8.37812e-01
Epoch 15, Step 50, Loss 9.73370e-01
Epoch 15, Step 100, Loss 7.91447e-01
Epoch 15, Step 150, Loss 8.27363e-01
Epoch 15, Step 200, Loss 8.46579e-01
[44.4 s | 2024-09-11 15:51:04.212638] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000015.pt.
[44.4 s | 2024-09-11 15:51:04.213888] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000014.pt.
Epoch 16, Step 0, Loss 8.59434e-01
Epoch 16, Step 50, Loss 9.20763e-01
Epoch 16, Step 100, Loss 7.62155e-01
Epoch 16, Step 150, Loss 7.71248e-01
Epoch 16, Step 200, Loss 8.11831e-01
[47.1 s | 2024-09-11 15:51:06.972560] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000016.pt.
[47.1 s | 2024-09-11 15:51:06.973763] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000015.pt.
Epoch 17, Step 0, Loss 8.56807e-01
Epoch 17, Step 50, Loss 8.06021e-01
Epoch 17, Step 100, Loss 8.36283e-01
Epoch 17, Step 150, Loss 7.88259e-01
Epoch 17, Step 200, Loss 8.26321e-01
[49.9 s | 2024-09-11 15:51:09.694183] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000017.pt.
[49.9 s | 2024-09-11 15:51:09.695488] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000016.pt.
Epoch 18, Step 0, Loss 7.45168e-01
Epoch 18, Step 50, Loss 7.74083e-01
Epoch 18, Step 100, Loss 8.39497e-01
Epoch 18, Step 150, Loss 7.77645e-01
Epoch 18, Step 200, Loss 8.34373e-01
[52.6 s | 2024-09-11 15:51:12.442971] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000018.pt.
[52.6 s | 2024-09-11 15:51:12.444134] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000017.pt.
Epoch 19, Step 0, Loss 7.31586e-01
Epoch 19, Step 50, Loss 7.85134e-01
Epoch 19, Step 100, Loss 7.31892e-01
Epoch 19, Step 150, Loss 7.79394e-01
Epoch 19, Step 200, Loss 7.37044e-01
[55.3 s | 2024-09-11 15:51:15.190555] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000019.pt.
[55.3 s | 2024-09-11 15:51:15.191727] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000018.pt.
wandb:
wandb:
wandb: Run history:
wandb:      epoch ▁▁▁▁▂▂▂▂▂▂▃▃▃▃▄▄▄▄▄▄▅▅▅▅▅▅▆▆▆▆▇▇▇▇▇▇████
wandb:       loss ██████████▇▇▇▇▇▆▆▅▅▄▄▄▃▃▃▂▂▂▂▁▂▁▂▁▁▁▁▁▁▁
wandb: loss_scale ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         lr ████████▇▇▇▇▇▇▆▆▆▆▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁
wandb:    resumes ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:      epoch 19
wandb:       loss 0.73704
wandb: loss_scale 1.0
wandb:         lr 1e-05
wandb:    resumes 0
wandb:
wandb: 🚀 View run unique-sweep-1 at: https://wandb.ai/f-dangel-team/example-preemptable-sweep/runs/2zoz0rl8
wandb: ⭐️ View project at: https://wandb.ai/f-dangel-team/example-preemptable-sweep
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240911_155016-2zoz0rl8/logs
wandb: WARNING The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.
[60.5 s | 2024-09-11 15:51:20.373116] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-11/2zoz0rl8_00000019.pt.
2024-09-11 15:51:24,319 - wandb.wandb_agent - INFO - Cleaning up finished run: 2zoz0rl8
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

![slurm run_preempted](./assets/03_slurm_preempted.png)

After full completion, the 'Workspace' tab on Weights & Biases looks as follows (of course, your curves will look slightly different due to sweep's and compute environment's stochastic nature):

![slurm_run_finished](./assets/04_slurm_finished.png)

The `resume` panel shows that different runs pre-empted a different number of times.
In total, we now have 1 (local) + 10 (SLURM) = 11 (total) finished runs.

## Conclusion

And that is pretty much it. Feel free to stop reading at this point.

## SLURM Launcher Details
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

2. **Printing details and wandb agent launch:** This part executes the `wandb agent` on our sweep and puts it into the background so our launch script can start listening to signals from SLURM.

3. **Installing a trap handler:** This function will process the signal sent by SLURM and pass it on to the python process, which will then initiate pre-emption and requeueing.
