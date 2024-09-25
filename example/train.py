#!/usr/bin/env python
"""Train a simple CNN on MNIST using checkpoints, integrated with Weights & Biases.

The changes required to integrate checkpointing with wandb are tagged with 'NOTE'.
"""

from argparse import ArgumentParser
from os import environ

import wandb
from torch import autocast, bfloat16, cuda, device, manual_seed
from torch.cuda.amp import GradScaler
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from wandb_preempt.checkpointer import Checkpointer

LOGGING_INTERVAL = 50  # Num batches between logging to stdout and wandb
VERBOSE = True  # Enable verbose output


def get_parser():
    r"""Create argument parser."""
    parser = ArgumentParser("Train a simple CNN on MNIST using SGD.")
    parser.add_argument(
        "--lr_max", type=float, default=0.01, help="Learning rate. Default: %(default)s"
    )
    parser.add_argument(
        "--epochs", type=int, default=20, help="Number of epochs. Default: %(default)s"
    )
    parser.add_argument(
        "--batch_size", type=int, default=256, help="Batch size. Default: %(default)s"
    )
    parser.add_argument(
        "--checkpoint_dir", type=str, default="checkpoints", help="Checkpoint save dir."
    )
    return parser


def main(args):
    r"""Train model."""
    manual_seed(0)  # make deterministic
    DEV = device("cuda" if cuda.is_available() else "cpu")

    # Set up the data, neural net, loss function, and optimizer
    train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
    train_loader = DataLoader(
        dataset=train_dataset, batch_size=args.batch_size, shuffle=True
    )
    model = Sequential(
        Conv2d(1, 3, kernel_size=5, stride=2),
        ReLU(),
        Flatten(),
        Linear(432, 50),
        ReLU(),
        Linear(50, 10),
    ).to(DEV)
    loss_func = CrossEntropyLoss().to(DEV)
    print(f"Using SGD with learning rate {args.lr_max}.")
    optimizer = SGD(model.parameters(), lr=args.lr_max)
    lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.epochs)
    scaler = GradScaler()

    # NOTE: Set up a check-pointer which will load and save checkpoints.
    # Pass the run ID to obtain unique file names for the checkpoints.
    run_id = environ["WANDB_RUN_ID"]
    checkpointer = Checkpointer(
        run_id,
        model,
        optimizer,
        lr_scheduler=lr_scheduler,
        scaler=scaler,
        savedir=args.checkpoint_dir,
        verbose=VERBOSE,
    )

    # NOTE: If existing, load model, optimizer, and learning rate scheduler state from
    # latest checkpoint, set random number generator states. If there was no checkpoint
    # to load, it does nothing and returns `None` for the step count.
    checkpoint_index, extra_info = checkpointer.load_latest_checkpoint()
    # Select the remaining epochs to train
    start_epoch = 0 if checkpoint_index is None else checkpoint_index + 1

    # NOTE forking must be enabled by the wandb team for your project.
    wandb_resume_step = extra_info.get("wandb_step", None)
    resume_from = (
        None if wandb_resume_step is None else f"{run_id}?_step={wandb_resume_step}"
    )
    resume = "allow" if resume_from is None else None
    print("resume_from:", resume_from)
    print("resume:", resume)
    wandb.init(resume=resume, resume_from=resume_from)

    # NOTE: Allow runs to resume by passing 'allow' to wandb
    # wandb.init(resume="allow")
    # print("Wandb step before manually setting it:", wandb.run.step)
    # NOTE: Currently getting an error from setattr here
    # wandb.run.step = extra_info.get("wandb_step", 0)
    # print("Wandb step after manually setting it:", wandb.run.step)

    # training
    for epoch in range(start_epoch, args.epochs):
        model.train()
        for step, (inputs, target) in enumerate(train_loader):
            optimizer.zero_grad()

            with autocast(device_type="cuda", dtype=bfloat16):
                output = model(inputs.to(DEV))
                loss = loss_func(output, target.to(DEV))

            if step % LOGGING_INTERVAL == 0:
                print(f"Epoch {epoch}, Step {step}, Loss {loss.item():.5e}")
                wandb.log(
                    {
                        "loss": loss.item(),
                        "lr": optimizer.param_groups[0]["lr"],
                        "loss_scale": scaler.get_scale(),
                        "epoch": epoch,
                        "resumes": checkpointer.num_resumes,
                    }
                )

            scaler.scale(loss).backward()
            scaler.step(optimizer)  # update neural network parameters
            scaler.update()  # update the gradient scaler

        lr_scheduler.step()  # update learning rate

        # NOTE Put validation code here
        # eval(model, ...)

        # NOTE Call checkpointer.step() at the end of the epoch to save a
        # checkpoint. If SLURM sent us a signal that our time for this job is
        # running out, it will now also take care of pre-empting the wandb job
        # and requeuing the SLURM job, killing the current python training script
        # to resume with the requeued job.
        checkpointer.step(extra_info={"wandb_step": wandb.run.step})

    wandb.finish()
    # NOTE Remove all created checkpoints once we are done training. If you want to
    # keep the trained model, remove this line.
    checkpointer.remove_checkpoints()


if __name__ == "__main__":
    # Run as a script
    parser = get_parser()
    args = parser.parse_args()
    main(args)
