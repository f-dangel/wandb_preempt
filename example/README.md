# Walkthrough

We will operate inside the [`example`](https://github.com/f-dangel/wandb_preempt/tree/main/example) directory:

```bash
cd example
```

/// details | Details of the training script `example/train.py` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/train.py))
```py
--8<-- "example/train.py"
```
///

/// details | Details of the sweep configuration `example/sweep.yaml` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/sweep.yaml))
```yaml
--8<-- "example/sweep.yaml"
```
///

Let's create a sweep:
```bash
wandb sweep sweep.yaml
```
Output:

/// details | Details of the SLURM script `example/launch.sh` ([source](https://github.com/f-dangel/wandb_preempt/blob/main/example/launch.sh))
```sh
--8<-- "example/launch.sh"
```
///

The launch script divides into three parts:

1. **SLURM configuration:** The first block specifies the SLURM resources and task array we are about to submit (lines starting with `#SBATCH`). The important configurations are
    - `--time` (how long will a job run?)
    - `--array` (how many jobs will be submitted?), and
    - `--signal` specifications (how much time before the limit will we start pre-empting?)

    **These values are optimized demonstration purposes. You definitely want to tweak them for your use case.**

    The other configuration flags are resource-specific, and some like the `--partition=a40` will depend on your cluster. In our script, we will request NVIDIA A40 GPUs because they support mixed-precision training in `bfloat16` and is used by many modern training pipelines.

2. **Printing details and wandb agent launch** This part executes the `wandb agent` on our sweep and puts it into the background so our launch script can start listening to signals from SLURM.

3. **Installing a signal handler**
