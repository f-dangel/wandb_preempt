"""Train a simple CNN on MNIST using checkpoints, integrated with Weights & Biases.

The changes required to integrate checkpointing with wandb are tagged with 'NOTE'.
"""

from argparse import ArgumentParser

import wandb
from torch import autocast, bfloat16, cuda, device, manual_seed
from torch.cuda.amp import GradScaler
from torch.nn import Conv2d, CrossEntropyLoss, Flatten, Linear, ReLU, Sequential
from torch.optim import SGD
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from torchvision.transforms import ToTensor

from wandb_preempt.checkpointer import CheckpointHandler, get_resume_value

parser = ArgumentParser("Train a simple CNN on MNIST using SGD.")
parser.add_argument("--lr", type=float, default=0.01, help="SGD's learning rate.")
parser.add_argument(
    "--max_epochs",
    type=int,
    default=10,
    help="Number of epochs to train for.",
)
args = parser.parse_args()

LOGGING_INTERVAL = 50  # print and log loss at this frequency
BATCH_SIZE = 256
VERBOSE = True

manual_seed(0)  # make deterministic
DEV = device("cuda" if cuda.is_available() else "cpu")

# NOTE: Define the directory where checkpoints are stored
SAVEDIR = "checkpoints"

# NOTE: Figure out the `resume` value and pass it to wandb
run = wandb.init(resume=get_resume_value(verbose=VERBOSE))

# Set up the data, neural net, loss function, and optimizer
train_dataset = MNIST("./data", train=True, download=True, transform=ToTensor())
train_loader = DataLoader(dataset=train_dataset, batch_size=BATCH_SIZE, shuffle=True)
model = Sequential(
    Conv2d(1, 3, kernel_size=5, stride=2),
    ReLU(),
    Flatten(),
    Linear(432, 50),
    ReLU(),
    Linear(50, 10),
).to(DEV)
loss_func = CrossEntropyLoss().to(DEV)
print(f"Using SGD with learning rate {args.lr}.")
optimizer = SGD(model.parameters(), lr=args.lr)
lr_scheduler = CosineAnnealingLR(optimizer, T_max=args.max_epochs)
scaler = GradScaler()

# NOTE: Set up a check-pointer which will load and save checkpoints.
# Pass the run ID to obtain unique file names for the checkpoints.
checkpoint_handler = CheckpointHandler(
    run.id,
    model,
    optimizer,
    lr_scheduler=lr_scheduler,
    scaler=scaler,
    savedir=SAVEDIR,
    verbose=VERBOSE,
)

# NOTE: If existing, load model, optimizer, and learning rate scheduler state from
# latest checkpoint, set random number generator states, and recover the epoch to start
# training from. Does nothing if there was no checkpoint.
start_epoch = checkpoint_handler.load_latest_checkpoint()

# training
for epoch in range(start_epoch, args.max_epochs):
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
                    "resumes": checkpoint_handler.num_resumes,
                }
            )

        scaler.scale(loss).backward()
        scaler.step(optimizer)  # update neural network parameters
        scaler.update()  # update the gradient scaler

    lr_scheduler.step()  # update learning rate

    # NOTE Put validation code here
    # eval(model, ...)

    # NOTE Call checkpoint_handler.step() at the end of the epoch to save a checkpoint.
    # If SLURM sent us a signal that our time for this job is running out, it will now
    # also take care of pre-empting the wandb job and requeuing the SLURM job, killing
    # the current python training script to resume with the requeued job.
    checkpoint_handler.step()

wandb.finish()
# NOTE Remove all created checkpoints once we are done training. If you want to
# keep the trained model, remove this line.
checkpoint_handler.remove_checkpoints()
