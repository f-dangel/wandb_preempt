# wandb_preempt

[![Documentation Status](https://readthedocs.org/projects/wandb-preempt/badge/?version=latest)](https://wandb-preempt.readthedocs.io/en/latest/?badge=latest)

This repository contains a tutorial on how to combine [wandb](https://wandb.ai/) sweeps
with [Slurm](https://slurm.schedmd.com/)'s pre-emption, i.e. how to automatically
re-queue and resume runs from a Weights & Biases sweep on a Slurm cluster.

## Getting started

### Installation

```bash
pip install git+https://github.com/f-dangel/wandb_preempt.git@main
```

### Basic Example

Please see the [docs](https://wandb-preempt.readthedocs.io/en/latest/walkthrough/).
