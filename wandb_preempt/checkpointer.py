"""Class for handling checkpointing."""

from datetime import date, datetime
from glob import glob
from os import environ, getenv, getpid, makedirs, path, remove, rename
from signal import SIGTERM, SIGUSR1, signal
from subprocess import run
from sys import exit
from time import sleep, time
from types import FrameType
from typing import Dict, List, Optional, Set, Tuple, Union

import wandb
from torch import cuda, device, get_rng_state, load, save, set_rng_state
from torch.cuda.amp import GradScaler
from torch.nn import Module
from torch.optim import Optimizer
from torch.optim.lr_scheduler import LRScheduler


class Checkpointer:
    """Class for storing, loading, and removing checkpoints.

    Can be marked as pre-empted by sending a `SIGUSR1` signal to a Python session.

    How to use this class:

    - Create an instance of this class `checkpointer = Checkpointer(...)`.
    - At the end of each epoch, call `checkpointer.step()` to save a checkpoint.
      If the job received the `SIGUSR1` or `SIGTERM` signal, the checkpointer will
      requeue the Slurm job at the end of its checkpointing step.
    """

    def __init__(
        self,
        run_id: str,
        model: Module,
        optimizer: Union[Optimizer, None],
        lr_scheduler: Optional[LRScheduler] = None,
        scaler: Optional[GradScaler] = None,
        savedir: str = "checkpoints",
        verbose: bool = False,
    ) -> None:
        """Set up a checkpointer.

        Args:
            run_id: A unique identifier for this run.
            model: The model that is trained and checkpointed.
            optimizer: The optimizer that is used for training and should be
                checkpointed. Use `None` to explicitly ignore the optimizer. This can
                be useful if your optimizer does not implement `.state_dict` and
                `.load_state_dict`.
            lr_scheduler: The learning rate scheduler that is used for training. If
                `None`, no learning rate scheduler is assumed. Default: `None`.
            scaler: The gradient scaler that is used when training in mixed precision.
                If `None`, no gradient scaler is assumed. Default: `None`.
            savedir: Directory to store checkpoints in. Default: `'checkpoints'`.
            verbose: Whether to print messages about saving and loading checkpoints.
                Default: `False`
        """
        self.time_created = time()
        self.run_id = run_id
        self.model = model
        self.optimizer = optimizer
        self.lr_scheduler = lr_scheduler
        self.scaler = scaler
        self.verbose = verbose
        self.marked_preempted = False
        self.step_count = 0
        self.num_resumes = 0

        # Set up signal handler listening for SIGUSR1, when we receive this signal,
        # we mark the job as about to be pre-empted.
        # Similarly, try to gracefully end if we receive the SIGTERM signal.
        signal(SIGUSR1, self.mark_preempted)
        signal(SIGTERM, self.mark_preempted)

        self.savedir = path.abspath(savedir)
        self.maybe_print(f"Creating checkpoint directory: {self.savedir}.")
        makedirs(self.savedir, exist_ok=True)

        # Detect whether we are running inside a SLURM session
        job_id = getenv("SLURM_JOB_ID")
        array_id = getenv("SLURM_ARRAY_JOB_ID")
        task_id = getenv("SLURM_ARRAY_TASK_ID")
        self.maybe_print(
            f"SLURM job ID: {job_id}, array ID: {array_id}, task ID: {task_id}"
        )
        self.uses_slurm = any(var is not None for var in {job_id, array_id, task_id})

        # We will create sub-folders in the directory supplied by the user where
        # checkpoints are stored. If we are on SLURM, we will use the `SLURM_JOB_ID`
        # variable as name, otherwise we will use the formatted day.
        self.savedir_job = path.join(
            self.savedir,
            f"{environ['SLURM_JOB_ID'] if self.uses_slurm else date.today()}",
        )

        # write Python PID to a file so it can be read by the signal handler from the
        # sbatch script, because it has to send a kill signal with SIGUSR1 to that PID.
        if self.uses_slurm:
            filename = f"{job_id}.pid"
            pid = str(getpid())
            self.maybe_print(f"Writing PID {pid} to file {filename}.")
            with open(filename, "w") as f:
                f.write(pid)

    def mark_preempted(self, sig: int, frame: Optional[FrameType]):
        """Mark the checkpointer as pre-empted.

        This information can be used by :meth:`step` to stop training after the current
        epoch.

        Args:
            sig: The signal number.
            frame: The current stack frame.
        """
        self.maybe_print(
            f"Received signal {sig}. Marking as pre-empted and will halt and requeue"
            " the job at next call of checkpointer.step()."
        )
        self.marked_preempted = True

    def checkpoint_path(self, counter: int) -> str:
        """Get the path to a checkpoint file for a given checkpointing step.

        Args:
            counter: The checkpointing step number.

        Returns:
            The path to the checkpoint file.
        """
        return path.join(self.savedir_job, f"{self.run_id}_{counter:08g}.pt")

    def save_checkpoint(self, extra_info: Dict) -> None:
        """Save a checkpoint.

        Stores optimizer, model, lr scheduler, gradient scaler, and random number
        generator states.

        Args:
            extra_info: Additional information to store in the checkpoint.
        """
        savepath = self.checkpoint_path(self.step_count)

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
            "rng_states": rng_states,
            "checkpoint_step": self.step_count,
            "resumes": self.num_resumes,
            "extra_info": extra_info,
        }
        if self.optimizer is not None:
            data["optimizer"] = self.optimizer.state_dict()
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
        tmp_savepath = f"{savepath}.tmp"
        save(data, tmp_savepath)
        rename(tmp_savepath, savepath)

    def load_latest_checkpoint(
        self, weights_only: bool = True, **kwargs
    ) -> Tuple[Union[int, None], Dict]:
        """Load the latest checkpoint and set random number generator states.

        Updates the model, optimizer, lr scheduler, and gradient scaler states
        passed at initialization.

        Args:
            weights_only: Whether to only unpickle objects that are safe to unpickle.
                If `True`, the only types that will be loaded are tensors, primitive
                types, dictionaries and types added via
                `torch.serialization.add_safe_globals()`.
                See `torch.load` for more information.
                Default: `True`.
            **kwargs: Additional keyword arguments to pass to the `torch.load` function.

        Returns:
            loaded_step: The index of the checkpoint that was loaded, or `None` if no
                checkpoint was found.
            extra_info: Extra information that was passed by the user to the `step`
                function when the checkpoint was saved, or an empty dictionary if there
                is no extra information.
        """
        loadpath = self.latest_checkpoint()
        if loadpath is None:
            self.maybe_print("No checkpoint found. Starting from scratch.")
            return None, {}

        self.maybe_print(f"Loading checkpoint {loadpath}.")

        data = load(loadpath, weights_only=weights_only, **kwargs)
        self.maybe_print("Loading model.")
        self.model.load_state_dict(data["model"])
        if self.optimizer is not None:
            self.maybe_print("Loading optimizer.")
            self.optimizer.load_state_dict(data["optimizer"])
        if self.lr_scheduler is not None:
            self.maybe_print("Loading lr scheduler.")
            self.lr_scheduler.load_state_dict(data["lr_scheduler"])
        if self.scaler is not None:
            self.maybe_print("Loading gradient scaler.")
            self.scaler.load_state_dict(data["scaler"])

        self.step_count = data["checkpoint_step"] + 1
        self.num_resumes = data["resumes"] + 1

        # restore random number generator states for all devices
        self.maybe_print("Setting RNG states.")
        for dev, rng_state in data["rng_states"].items():
            if "cuda" in dev:
                cuda.set_rng_state(rng_state, dev)
            else:
                set_rng_state(rng_state)

        # N.B. We return the checkpoint step index of the saved file that was loaded,
        # but the checkpointer.step_count is one larger than that because we increment
        # it after saving - it tracks the index of the next checkpoint to be saved.
        return data["checkpoint_step"], data["extra_info"]

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
        return glob(path.join(self.savedir, "*", f"{self.run_id}_*.pt"))

    @staticmethod
    def checkpointed_run_ids(savedir: str = "checkpoints") -> Set[str]:
        """Return the run IDs of checkpointed runs.

        Args:
            savedir: The directory to search for resumable runs.

        Returns:
            A set of run IDs that have at least one checkpoint.
        """
        run_ids = set()
        for checkpoint in glob(path.join(savedir, "*.pt")):
            run_id = path.basename(checkpoint).split("_")[0]
            run_ids.add(run_id)
        return run_ids

    def latest_checkpoint(self) -> Union[None, str]:
        """Return the path to the latest checkpoint.

        Returns:
            The path to the latest checkpoint, or `None` if no checkpoints exist.
        """
        checkpoints_sorted = sorted(
            self.all_checkpoints(),
            key=lambda x: int(path.basename(x).replace(".pt", "").split("_")[-1]),
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

    def maybe_requeue_slurm_job(self):
        """Requeue the SLURM job if we are running in a SLURM session."""
        if not self.uses_slurm:
            return

        job_id = getenv("SLURM_JOB_ID")
        array_id = getenv("SLURM_ARRAY_JOB_ID")
        task_id = getenv("SLURM_ARRAY_TASK_ID")

        uses_array = array_id is not None and task_id is not None
        requeue_id = f"{array_id}_{task_id}" if uses_array else job_id

        cmd = ["scontrol", "requeue", requeue_id]
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

    def step(self, extra_info: Optional[Dict] = None):
        """Perform a checkpointing step.

        Save the checkpoint. If we were pre-empted we requeue the job
        and exit the training script after saving.

        Args:
            extra_info: Additional information to save in the checkpoint. This
                dictionary is returned when loading the latest checkpoint with
                `checkpointer.load_latest_checkpoint()`.
                By default, an empty dictionary is saved.
        """
        self.save_checkpoint({} if extra_info is None else extra_info)
        # Remove stale checkpoints
        self.remove_checkpoints(keep_latest=True)

        # requeue the job if the run was marked as pre-empted and exit
        if self.marked_preempted:
            self.maybe_print("Run was marked as pre-empted via signal.")
            self.preempt_wandb_run()
            self.maybe_requeue_slurm_job()
            self.maybe_print("Exiting with error code 1.")
            exit(1)
        # Increase the number of steps taken
        self.step_count += 1
