# Walkthrough

## Overview

This section explains how to create and launch a preempt-able `wandb` sweep on a SLURM cluster. You have to set up three files:

1. A training script (e.g. `train.py`) that the sweep will execute multiple times using different hyper-parameters.

2. A `wandb` sweep configuration script (e.g. `sweep.yaml`) that defines the hyper-parameter search space.

3. A SLURM launch script (e.g. `launch.sh`) which we will use to submit runs on the cluster.

The repository's [`example`](https://github.com/f-dangel/wandb_preempt/tree/main/example) directory contains examples for each of these files, and here we will demonstrate how to make them work together. We will operate inside the example directory, so let's navigate to it:
```bash
cd example
```

## Training Script

First up, we need to write a training script that we will sweep over, i.e. call with different hyper-parameters.

For demonstration purposes, we will train a small CNN on MNIST using SGD, and our goal is to find a good learning rate through random search using a `wandb` sweep. To keep things simple and cheap, we fix a batch size and use a (very) small number of epochs. Finally, we also use a learning rate scheduler and mixed-precision training with a gradient scaler. These are overkill for MNIST, of course. But we want to show how these need to be checkpointed. In summary, we will call the training script using the following pattern:
```bash
python train.py --lr=X
```
with `X` some floating point number.

The training script also contains code to checkpoint the training loop at the end of an epoch whenever the Python process receives a `SIGUSR1` signal from the OS. Roughly speaking, we need to create a `Checkpointer` that is responsible for saving and storing checkpoints, listening to signals, and for initiating the job requeue on the cluster. Every time we call the checkpointer's `.step` function, it will save a checkpoint and check whether a signal has been sent that indicates we must pre-empt and requeue. For more details, please expand the code snippet.

/// details | Details of the training script `example/train.py` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/train.py))
```py hl_lines="37-38 40-41"
--8<-- "example/train.py"
```
///

## Sweep Configuration

Our next goal will be to define and create a sweep.

For that, we need to write a `.yaml` file which specifies how the training script is called, and how the search space looks like. To learn more, have a look at the [Weights & Biases documentation](https://docs.wandb.ai/guides/sweeps/define-sweep-configuration).

The following configuration file defines a random search over the learning rate, using a log-uniform search space. Note that you need to modify the `entity` and `project` entries to match with a `wandb` project that you have access to.

/// details | Details of the sweep configuration `example/sweep.yaml` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/sweep.yaml))
```yaml hl_lines="1 2"
--8<-- "example/sweep.yaml"
```
///

Let's create the sweep:
```bash
wandb sweep sweep.yaml
```
/// details | Output
```sh hl_lines="4"
wandb: Creating sweep from: sweep.yaml
wandb: Creating sweep with ID: aaq70gt8
wandb: View sweep at: https://wandb.ai/f-dangel-team/quickstart/sweeps/aaq70gt8
wandb: Run sweep agent with: wandb agent f-dangel-team/quickstart/aaq70gt8
```
///

The important part of that output is the command in the last line. Copy it for later.

Navigate to the `wandb` web interface and you should be able to see the sweep:

![empty sweep](./assets/01_empty_sweep.png)

### Optional Step: Local Run

To make sure the configuration file works, I will execute a single run locally on my machine as a sanity check. This step is optional and obviously not recommended if your machine's hardware is not beefy enough (note that I specified the `--count=1` flag to carry out only a single run):

```bash
wandb agent --count=1 f-dangel-team/quickstart/aaq70gt8
```

/// details | Training script output
```
wandb: Starting wandb agent 🕵️
2024-09-09 15:05:49,720 - wandb.wandb_agent - INFO - Running runs: []
2024-09-09 15:05:49,962 - wandb.wandb_agent - INFO - Agent received command: run
2024-09-09 15:05:49,963 - wandb.wandb_agent - INFO - Agent starting run with config:
        lr: 0.0014109383616700757
        max_epochs: 20
2024-09-09 15:05:49,966 - wandb.wandb_agent - INFO - About to run command: /usr/bin/env python train.py --lr=0.0014109383616700757 --max_epochs=20
Environment variables containing 'WANDB'
WANDB_ENTITY: f-dangel-team
WANDB_PROJECT: quickstart
WANDB_SWEEP_ID: aaq70gt8
WANDB_DIR: ~/wandb_preempt/example
WANDB_RUN_ID: mxgkpzzn
WANDB_SWEEP_PARAM_PATH: ~/wandb_preempt/example/wandb/sweep-aaq70gt8/config-mxgkpzzn.yaml
Agent's run has ID mxgkpzzn and state running. Using resume='allow'.
wandb: Currently logged in as: f-dangel (f-dangel-team). Use `wandb login --relogin` to force relogin
wandb: Tracking run with wandb version 0.17.9
wandb: Run data is saved locally in ~/wandb_preempt/example/wandb/run-20240909_150552-mxgkpzzn
wandb: Run `wandb offline` to turn off syncing.
wandb: Syncing run colorful-sweep-3
wandb: ⭐️ View project at https://wandb.ai/f-dangel-team/quickstart
wandb: 🧹 View sweep at https://wandb.ai/f-dangel-team/quickstart/sweeps/aaq70gt8
wandb: 🚀 View run at https://wandb.ai/f-dangel-team/quickstart/runs/mxgkpzzn
2024-09-09 15:05:54,989 - wandb.wandb_agent - INFO - Running runs: ['mxgkpzzn']
Using SGD with learning rate 0.0014109383616700757.
[0.0 s | 2024-09-09 15:05:55.010675] Creating checkpoint directory: ~/wandb_preempt/example/checkpoints.
[0.0 s | 2024-09-09 15:05:55.010766] SLURM job ID: None, array ID: None, task ID: None
[0.0 s | 2024-09-09 15:05:55.010984] No checkpoint found. Starting from scratch.
Epoch 0, Step 0, Loss 2.31327e+00
Epoch 0, Step 50, Loss 2.31711e+00
Epoch 0, Step 100, Loss 2.30655e+00
Epoch 0, Step 150, Loss 2.30137e+00
Epoch 0, Step 200, Loss 2.30741e+00
[2.8 s | 2024-09-09 15:05:57.818613] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000000.pt.
Epoch 1, Step 0, Loss 2.29989e+00
Epoch 1, Step 50, Loss 2.31124e+00
Epoch 1, Step 100, Loss 2.30144e+00
Epoch 1, Step 150, Loss 2.28741e+00
Epoch 1, Step 200, Loss 2.30759e+00
[5.5 s | 2024-09-09 15:06:00.536954] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000001.pt.
[5.5 s | 2024-09-09 15:06:00.538081] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000000.pt.
Epoch 2, Step 0, Loss 2.28908e+00
Epoch 2, Step 50, Loss 2.28920e+00
Epoch 2, Step 100, Loss 2.29056e+00
Epoch 2, Step 150, Loss 2.28977e+00
Epoch 2, Step 200, Loss 2.29214e+00
[8.2 s | 2024-09-09 15:06:03.259775] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000002.pt.
[8.3 s | 2024-09-09 15:06:03.260907] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000001.pt.
Epoch 3, Step 0, Loss 2.28515e+00
Epoch 3, Step 50, Loss 2.29505e+00
Epoch 3, Step 100, Loss 2.28214e+00
Epoch 3, Step 150, Loss 2.27602e+00
Epoch 3, Step 200, Loss 2.28020e+00
[11.0 s | 2024-09-09 15:06:05.982865] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000003.pt.
[11.0 s | 2024-09-09 15:06:05.983939] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000002.pt.
Epoch 4, Step 0, Loss 2.27190e+00
Epoch 4, Step 50, Loss 2.28208e+00
Epoch 4, Step 100, Loss 2.27995e+00
Epoch 4, Step 150, Loss 2.26419e+00
Epoch 4, Step 200, Loss 2.26638e+00
[13.7 s | 2024-09-09 15:06:08.706617] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000004.pt.
[13.7 s | 2024-09-09 15:06:08.707936] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000003.pt.
Epoch 5, Step 0, Loss 2.25696e+00
Epoch 5, Step 50, Loss 2.26390e+00
Epoch 5, Step 100, Loss 2.25003e+00
Epoch 5, Step 150, Loss 2.25402e+00
Epoch 5, Step 200, Loss 2.24690e+00
[16.4 s | 2024-09-09 15:06:11.447655] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000005.pt.
[16.4 s | 2024-09-09 15:06:11.448819] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000004.pt.
Epoch 6, Step 0, Loss 2.24057e+00
Epoch 6, Step 50, Loss 2.23889e+00
Epoch 6, Step 100, Loss 2.24205e+00
Epoch 6, Step 150, Loss 2.23790e+00
Epoch 6, Step 200, Loss 2.23253e+00
[19.2 s | 2024-09-09 15:06:14.191520] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000006.pt.
[19.2 s | 2024-09-09 15:06:14.192684] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000005.pt.
Epoch 7, Step 0, Loss 2.22399e+00
Epoch 7, Step 50, Loss 2.20625e+00
Epoch 7, Step 100, Loss 2.20944e+00
Epoch 7, Step 150, Loss 2.20490e+00
Epoch 7, Step 200, Loss 2.20541e+00
[21.9 s | 2024-09-09 15:06:16.919781] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000007.pt.
[21.9 s | 2024-09-09 15:06:16.920934] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000006.pt.
Epoch 8, Step 0, Loss 2.19063e+00
Epoch 8, Step 50, Loss 2.18892e+00
Epoch 8, Step 100, Loss 2.17470e+00
Epoch 8, Step 150, Loss 2.16582e+00
Epoch 8, Step 200, Loss 2.16229e+00
[24.6 s | 2024-09-09 15:06:19.642306] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000008.pt.
[24.6 s | 2024-09-09 15:06:19.643417] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000007.pt.
Epoch 9, Step 0, Loss 2.15864e+00
Epoch 9, Step 50, Loss 2.14666e+00
Epoch 9, Step 100, Loss 2.13153e+00
Epoch 9, Step 150, Loss 2.13505e+00
Epoch 9, Step 200, Loss 2.12925e+00
[27.3 s | 2024-09-09 15:06:22.357240] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000009.pt.
[27.3 s | 2024-09-09 15:06:22.358962] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000008.pt.
Epoch 10, Step 0, Loss 2.10835e+00
Epoch 10, Step 50, Loss 2.10394e+00
Epoch 10, Step 100, Loss 2.09672e+00
Epoch 10, Step 150, Loss 2.08576e+00
Epoch 10, Step 200, Loss 2.10477e+00
[30.1 s | 2024-09-09 15:06:25.068843] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000010.pt.
[30.1 s | 2024-09-09 15:06:25.069888] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000009.pt.
Epoch 11, Step 0, Loss 2.05669e+00
Epoch 11, Step 50, Loss 2.05382e+00
Epoch 11, Step 100, Loss 2.03040e+00
Epoch 11, Step 150, Loss 2.03818e+00
Epoch 11, Step 200, Loss 2.01292e+00
[32.8 s | 2024-09-09 15:06:27.767231] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000011.pt.
[32.8 s | 2024-09-09 15:06:27.768345] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000010.pt.
Epoch 12, Step 0, Loss 2.02450e+00
Epoch 12, Step 50, Loss 1.98715e+00
Epoch 12, Step 100, Loss 2.00718e+00
Epoch 12, Step 150, Loss 1.99295e+00
Epoch 12, Step 200, Loss 1.97624e+00
[35.6 s | 2024-09-09 15:06:30.564774] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000012.pt.
[35.6 s | 2024-09-09 15:06:30.565946] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000011.pt.
Epoch 13, Step 0, Loss 1.95026e+00
Epoch 13, Step 50, Loss 1.94186e+00
Epoch 13, Step 100, Loss 1.94720e+00
Epoch 13, Step 150, Loss 1.95091e+00
Epoch 13, Step 200, Loss 1.96935e+00
[38.3 s | 2024-09-09 15:06:33.267992] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000013.pt.
[38.3 s | 2024-09-09 15:06:33.269097] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000012.pt.
Epoch 14, Step 0, Loss 1.88487e+00
Epoch 14, Step 50, Loss 1.92402e+00
Epoch 14, Step 100, Loss 1.93435e+00
Epoch 14, Step 150, Loss 1.87301e+00
Epoch 14, Step 200, Loss 1.91263e+00
[40.9 s | 2024-09-09 15:06:35.931019] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000014.pt.
[40.9 s | 2024-09-09 15:06:35.932336] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000013.pt.
Epoch 15, Step 0, Loss 1.88196e+00
Epoch 15, Step 50, Loss 1.90995e+00
Epoch 15, Step 100, Loss 1.86447e+00
Epoch 15, Step 150, Loss 1.88345e+00
Epoch 15, Step 200, Loss 1.88344e+00
[43.6 s | 2024-09-09 15:06:38.633177] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000015.pt.
[43.6 s | 2024-09-09 15:06:38.634305] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000014.pt.
Epoch 16, Step 0, Loss 1.85623e+00
Epoch 16, Step 50, Loss 1.88199e+00
Epoch 16, Step 100, Loss 1.84634e+00
Epoch 16, Step 150, Loss 1.79202e+00
Epoch 16, Step 200, Loss 1.84115e+00
[46.4 s | 2024-09-09 15:06:41.416646] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000016.pt.
[46.4 s | 2024-09-09 15:06:41.417763] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000015.pt.
Epoch 17, Step 0, Loss 1.86561e+00
Epoch 17, Step 50, Loss 1.80409e+00
Epoch 17, Step 100, Loss 1.81886e+00
Epoch 17, Step 150, Loss 1.82025e+00
Epoch 17, Step 200, Loss 1.81553e+00
[49.2 s | 2024-09-09 15:06:44.194540] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000017.pt.
[49.2 s | 2024-09-09 15:06:44.195673] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000016.pt.
Epoch 18, Step 0, Loss 1.80697e+00
Epoch 18, Step 50, Loss 1.79723e+00
Epoch 18, Step 100, Loss 1.83584e+00
Epoch 18, Step 150, Loss 1.81160e+00
Epoch 18, Step 200, Loss 1.83512e+00
[52.5 s | 2024-09-09 15:06:47.510459] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000018.pt.
[52.5 s | 2024-09-09 15:06:47.511873] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000017.pt.
Epoch 19, Step 0, Loss 1.79649e+00
Epoch 19, Step 50, Loss 1.82962e+00
Epoch 19, Step 100, Loss 1.80719e+00
Epoch 19, Step 150, Loss 1.82502e+00
Epoch 19, Step 200, Loss 1.79152e+00
[55.3 s | 2024-09-09 15:06:50.269146] Saving checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000019.pt.
[55.3 s | 2024-09-09 15:06:50.270317] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000018.pt.
wandb:
wandb:
wandb: Run history:
wandb:       loss ████████▇█▇▇▇▇▇▇▆▆▆▆▅▅▅▄▄▄▃▃▃▂▃▂▂▁▁▁▁▁▂▁
wandb: loss_scale ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:         lr ████████▇▇▇▇▇▇▆▆▆▆▅▅▄▄▄▄▃▃▃▃▂▂▂▂▂▂▁▁▁▁▁▁
wandb:    resumes ▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁▁
wandb:
wandb: Run summary:
wandb:       loss 1.79152
wandb: loss_scale 1.0
wandb:         lr 1e-05
wandb:    resumes 0
wandb:
wandb: 🚀 View run colorful-sweep-3 at: https://wandb.ai/f-dangel-team/quickstart/runs/mxgkpzzn
wandb: ⭐️ View project at: https://wandb.ai/f-dangel-team/quickstart
wandb: Synced 6 W&B file(s), 0 media file(s), 0 artifact file(s) and 0 other file(s)
wandb: Find logs at: ./wandb/run-20240909_150552-mxgkpzzn/logs
wandb: WARNING The new W&B backend becomes opt-out in version 0.18.0; try it out with `wandb.require("core")`! See https://wandb.me/wandb-core for more information.
[61.0 s | 2024-09-09 15:06:56.003246] Removing checkpoint ~/wandb_preempt/example/checkpoints/2024-09-09/mxgkpzzn_00000019.pt.
2024-09-09 15:07:01,045 - wandb.wandb_agent - INFO - Cleaning up finished run: mxgkpzzn
wandb: Terminating and syncing runs. Press ctrl-c to kill.
```
///

On the Weights & Biases web API, we can see the successfully finished run:

![local run](./assets/02_local_run.png)

## SLURM Launcher

The last step is to launch multiple jobs on a SLURM cluster.

For that, we use the following launch script and insert the wandb agent's command into it. We will explain the script in more detail below; for now our focus is to demonstrate how everything works.

/// details | Details of the SLURM script `example/launch.sh` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/launch.sh))
```sh hl_lines="10-12 23"
--8<-- "example/launch.sh"
```
///


## Details
The launch script divides into three parts:

1. **SLURM configuration:** The first block specifies the SLURM resources and task array we are about to submit (lines starting with `#SBATCH`). The important configurations are
    - `--time` (how long will a job run?)
    - `--array` (how many jobs will be submitted?), and
    - `--signal` specifications (how much time before the limit will we start pre-empting?)

    **These values are optimized for demonstration purposes. You definitely want to tweak them for your use case.**

    The other configuration flags are resource-specific, and some like the `--partition=a40` will depend on your cluster. In our script, we will request NVIDIA A40 GPUs because they support mixed-precision training in `bfloat16` and is used by many modern training pipelines.

2. **Printing details and wandb agent launch** This part executes the `wandb agent` on our sweep and puts it into the background so our launch script can start listening to signals from SLURM.

3. **Installing a signal handler**
