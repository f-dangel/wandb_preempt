"""Class for handling checkpointing."""

from datetime import datetime
from glob import glob
from os import environ, getenv, getpid, makedirs, path, remove, rename
from signal import SIGTERM, SIGUSR1, signal
from subprocess import run
from sys import exit
from time import sleep, time
from types import FrameType, TracebackType
from typing import Dict, List, Optional, Set, Type, Union

import wandb
from torch import cuda, device, get_rng_state, load, save, set_rng_state
from torch.cuda.amp import GradScaler
from torch.optim.lr_scheduler import LRScheduler
from wandb import Api


def get_resume_value(verbose: bool = False) -> str:
    """Return the `resume` value a the agent's run.

    Args:
        verbose: Whether to print information to the command line. Default: `False`.

    Returns:
        The run's resume value. Either `'must'` or `'allow'`.

    Raises:
        RuntimeError: If the environment variables `WANDB_ENTITY`, `WANDB_PROJECT`, or
            `WANDB_RUN_ID`, which are usually set by a wandb agent, are not set.
    """
    if verbose:
        print("Environment variables containing 'WANDB'")
        for key, value in environ.items():
            if "WANDB" in key:
                print(f"{key}: {value}")

    for var in {"WANDB_ENTITY", "WANDB_PROJECT", "WANDB_RUN_ID"}:
        if var not in environ:
            raise RuntimeError(f"Environment variable {var!r} was not set.")

    entity = environ["WANDB_ENTITY"]
    project = environ["WANDB_PROJECT"]
    run_id = environ["WANDB_RUN_ID"]

    run = Api().run(f"{entity}/{project}/{run_id}")
    resume = "must" if run.state == "preempted" else "allow"
    if verbose:
        print(
            f"Agent's run has ID {run.id} and state {run.state}."
            + f" Using resume={resume!r}."
        )

    return resume


class CheckpointHandler:
    """Class for storing, loading, and removing checkpoints.

    Can be marked as pre-empted by sending a `SIGUSR1` signal to a Python session.

    How to use this class:

    - Create an instance in your training loop, `handler = CheckpointHandler(...)`.
    - Wrap each epoch in a `CheckpointAtEnd(handler, ...)` context manager.
    """

    def __init__(
        self,
        run_id: str,
        model,
        optimizer,
        lr_scheduler: Optional[LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        metadata: Optional[Dict] = None,
        savedir: str = "checkpoints",
        verbose: bool = False,
    ) -> None:
        """Set up a checkpoint handler.

        Args:
            run_id: A unique identifier for this run.
            model: The model that is trained and checkpointed.
            optimizer: The optimizer that is used for training and checkpointed.
            lr_scheduler: The learning rate scheduler that is used for training. If
                `None`, no learning rate scheduler is assumed. Default: `None`.
            scaler: The gradient scaler that is used when training in mixed precision.
                If `None`, no gradient scaler is assumed. Default: `None`.
            metadata: Additional metadata to store in the checkpoint. Default: `None`.
            savedir: Directory to store checkpoints in. Default: `'checkpoints'`.
            verbose: Whether to print messages about saving and loading checkpoints.
                Default: `False`

        Raises:
            RuntimeError: If the environment variables `SLURM_ARRAY_JOB_ID` and
                `SLURM_ARRAY_TASK_ID` are not set. This indicates we are not running
                a SLURM task array job.
        """
        self.time_created = time()
        self.run_id = run_id
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.metadata = {} if metadata is None else metadata
        self.verbose = verbose
        self.marked_preempted = False
        self.num_resumes = 0

        # Set up signal handler listening for SIGUSR1, when we receive this signal,
        # we mark the job as about to be pre-empted.
        # Similarly, try to gracefully end if we receive the SIGTERM signal.
        signal(SIGUSR1, self.mark_preempted)
        signal(SIGTERM, self.mark_preempted)

        self.savedir = path.abspath(savedir)
        self.maybe_print(f"Creating checkpoint directory: {self.savedir}.")
        makedirs(self.savedir, exist_ok=True)
        self.savedir_job = path.join(self.savedir, environ["SLURM_JOB_ID"])

        # write Python PID to a file so it can be read by the signal handler from the
        # sbatch script, because it has to send a kill signal with SIGUSR1 to that PID.
        array_id = getenv("SLURM_ARRAY_JOB_ID")
        task_id = getenv("SLURM_ARRAY_TASK_ID")
        self.maybe_print(f"Array ID: {array_id}, Task ID: {task_id}")

        if array_id is None or task_id is None:
            raise RuntimeError("One of SLURM_ARRAY_JOB/TASK_ID are not set.")

        filename = f"{array_id}_{task_id}.pid"
        pid = str(getpid())
        self.maybe_print(f"Writing PID {pid} to file {filename}.")
        with open(filename, "w") as f:
            f.write(pid)

    def mark_preempted(self, sig: int, frame: Optional[FrameType]):
        """Mark the checkpointer as pre-empted.

        This information can be used by the `CheckpointAtEnd` context manager to stop
        training after the current epoch.

        Args:
            sig: The signal number.
            frame: The current stack frame.
        """
        self.maybe_print(
            f"Got signal {sig}. Marking as pre-empted. This will be the last epoch."
        )
        self.marked_preempted = True

    def checkpoint_path(self, epoch: int) -> str:
        """Get the path to a checkpoint file for a given epoch.

        Args:
            epoch: The epoch number.

        Returns:
            The path to the checkpoint file for this epoch.
        """
        return path.join(self.savedir_job, f"{self.run_id}_epoch_{epoch:08g}.pt")

    def save_checkpoint(self, epoch: int) -> None:
        """Save a checkpoint for a given epoch.

        Stores optimizer, model, lr scheduler, gradient scaler, and random number
        generator states.

        Args:
            epoch: The epoch number.
        """
        savepath = self.checkpoint_path(epoch)

        # get random number generator states for all devices
        devices = [device("cpu")]
        if cuda.is_available():
            devices.extend([device(f"cuda:{i}") for i in range(cuda.device_count())])

        rng_states = {
            str(dev): cuda.get_rng_state(dev) if "cuda" in str(dev) else get_rng_state()
            for dev in devices
        }
        data = {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "rng_states": rng_states,
            "epoch": epoch,
            "resumes": self.num_resumes,
            "metadata": self.metadata,
        }
        if self.lr_scheduler is not None:
            data["lr_scheduler"] = self.lr_scheduler.state_dict()
        if self.scaler is not None:
            data["scaler"] = self.scaler.state_dict()

        self.maybe_print(f"Saving checkpoint {savepath}.")
        if not path.exists(path.dirname(savepath)):
            # We protect this inside an if statement because sometimes the server is
            # configured so the checkpoint directory is automatically created without
            # us having the permissions to create it ourselves.
            makedirs(path.dirname(savepath), exist_ok=True)

        # Save to a temporary file first, then move the temporary file to the target
        # destination. This ensures we don't confuse a partially written file with
        # a valid checkpoint if we are interrupted halfway through saving. (Moving is
        # atomic, so it either happens or doesn't.)
        tmp_savepath = savepath + ".tmp"
        save(data, tmp_savepath)
        rename(tmp_savepath, savepath)

    def load_latest_checkpoint(self) -> int:
        """Load the latest checkpoint and set random number generator states.

        Updates the model, optimizer, lr scheduler, and gradient scaler states
        passed at initialization.

        Returns:
            The epoch number at which training should resume.
        """
        loadpath = self.latest_checkpoint()
        if loadpath is None:
            self.maybe_print("No checkpoint found. Starting from scratch.")
            return 0

        self.maybe_print(f"Loading checkpoint {loadpath}.")

        data = load(loadpath)
        self.maybe_print("Loading model.")
        self.model.load_state_dict(data["model"])
        self.maybe_print("Loading optimizer.")
        self.optimizer.load_state_dict(data["optimizer"])
        if self.lr_scheduler is not None:
            self.maybe_print("Loading lr scheduler.")
            self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        if self.scaler is not None:
            self.maybe_print("Loading gradient scaler.")
            self.scaler.load_state_dict(data["scaler"])

        self.num_resumes = data["resumes"] + 1

        # restore random number generator states for all devices
        self.maybe_print("Setting RNG states.")
        for dev, rng_state in data["rng_states"].items():
            if "cuda" in dev:
                cuda.set_rng_state(rng_state, dev)
            else:
                set_rng_state(rng_state)

        return data["epoch"] + 1

    def remove_checkpoints(self, keep_latest: bool = False):
        """Remove checkpoints.

        Args:
            keep_latest: Whether to keep the latest checkpoint. Default: `False`.

        Raises:
            RuntimeError: If a non-`.pt` file is found in the checkpoint directory.
        """
        checkpoints = self.old_checkpoints() if keep_latest else self.all_checkpoints()
        for checkpoint in checkpoints:
            if not checkpoint.endswith(".pt"):
                raise RuntimeError(f"Was asked to delete a non-.pt-file: {checkpoint}.")
            self.maybe_print(f"Removing checkpoint {checkpoint}.")
            remove(checkpoint)

    def all_checkpoints(self) -> List[str]:
        """Return all existing checkpoints for a run.

        Returns:
            A list of paths to all existing checkpoints.
        """
        return glob(path.join(self.savedir, "*", f"{self.run_id}_epoch_*.pt"))

    @staticmethod
    def checkpointed_run_ids(savedir: str = "checkpoints") -> Set[str]:
        """Return the run IDs of checkpointed runs.

        Args:
            savedir: The directory to search for resumable runs.

        Returns:
            A set of run IDs that have at least one checkpoint.
        """
        run_ids = set()
        for checkpoint in glob(path.join(savedir, "*_epoch_*.pt")):
            run_id = path.basename(checkpoint).split("_epoch_")[0]
            run_ids.add(run_id)
        return run_ids

    def latest_checkpoint(self) -> Union[None, str]:
        """Return the path to the latest checkpoint.

        Returns:
            The path to the latest checkpoint, or `None` if no checkpoints exist.
        """
        checkpoints_sorted = sorted(
            self.all_checkpoints(),
            key=lambda path: int(path.split("/")[-1].replace(".pt", "").split("_")[-1]),
        )
        return checkpoints_sorted[-1] if checkpoints_sorted else None

    def old_checkpoints(self) -> List[str]:
        """Return all but the latest checkpoint.

        Returns:
            A list of paths to all but the latest checkpoint.
        """
        existing = self.all_checkpoints()

        if existing:
            latest = self.latest_checkpoint()
            existing.remove(latest)

        return existing

    def maybe_print(self, msg: str, verbose: Optional[bool] = None) -> None:
        """Print a message with time stamp if verbose mode is enabled.

        Args:
            msg: The message to print.
            verbose: Whether to print the message. If `None`, the instance's `verbose`
                attribute is used. Default: `None`.
        """
        verbose = self.verbose if verbose is None else verbose
        if verbose:
            elapsed = time() - self.time_created
            print(f"[{elapsed:.1f} s | {datetime.now()}] {msg}")

    def requeue_slurm_job(self):
        """Requeue the Slurm job.

        Raises:
            RuntimeError: If the job is not a Slurm job.
        """
        array_id = getenv("SLURM_ARRAY_JOB_ID")
        task_id = getenv("SLURM_ARRAY_TASK_ID")

        if array_id is None:
            raise RuntimeError("Not a SLURM job. Variable SLURM_ARRAY_JOB_ID not set.")

        slurm_id = array_id if task_id is None else f"{array_id}_{task_id}"
        cmd = ["scontrol", "requeue", slurm_id]
        self.maybe_print(f"Requeuing SLURM job with `{' '.join(cmd)}`.")
        run(cmd, check=True)

    def preempt_wandb_run(self):
        """If using Weights & Biases, mark the run as pre-empted."""
        self.maybe_print("Marking wandb run as preempted.")
        wandb.mark_preempting()
        self.maybe_print("Terminating wandb with non-zero exit code.")
        wandb.finish(exit_code=1)
        self.maybe_print("Sleeping for 15 s to give wandb enough time.")
        sleep(15)


class CheckpointAtEnd:
    """Context manager for checkpointing at the end.

    Can abort training early if the checkpointer was marked as pre-empted via a
    `SIGUSR1` signal sent to the Python session.
    """

    def __init__(
        self, checkpoint_handler: CheckpointHandler, epoch: int, verbose: bool = False
    ) -> None:
        """Initialize the context manager.

        Args:
            checkpoint_handler: The `CheckpointHandler` instance to use for saving
                checkpoints.
            epoch: The current epoch number.
            verbose: Whether to print messages about saving and loading checkpoints.
                Default: `False`.
        """
        self.checkpoint_handler = checkpoint_handler
        self.epoch = epoch
        self.verbose = verbose

    def __enter__(self) -> None:
        """Enter a block."""
        pass

    def __exit__(
        self,
        exc_type: Optional[Type[BaseException]],
        exc_value: Optional[BaseException],
        traceback: Optional[TracebackType],
    ):
        """Exit a block.

        If everything went normal, save a checkpoint and remove older checkpoints.
        If other errors occured, remove all checkpoints.

        If the run was marked as pre-empted, try requeuing the slurm job.

        Args:
            exc_type: The type of the exception that was raised, if any.
            exc_value: The exception that was raised, if any.
            traceback: The traceback of the exception that was raised, if any.
        """
        # save a checkpoint if everything went normal
        normal_exit = exc_type is None
        if normal_exit:
            self.checkpoint_handler.save_checkpoint(self.epoch)

        # remove all checkpoints if other errors occured
        self.checkpoint_handler.remove_checkpoints(keep_latest=normal_exit)

        # requeue the job if the run was marked as pre-empted and exit
        if self.checkpoint_handler.marked_preempted:
            self.maybe_print("Run was marked as pre-empted via signal.")
            self.checkpoint_handler.preempt_wandb_run()
            self.checkpoint_handler.requeue_slurm_job()
            self.maybe_print("Exiting with error code 1.")
            exit(1)

    def maybe_print(self, msg: str) -> None:
        """Print a message if verbose mode is enabled.

        Args:
            msg: The message to print.
        """
        self.checkpoint_handler.maybe_print(msg, verbose=self.verbose)
