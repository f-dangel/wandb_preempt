"""Check that the training script is working."""

from os import environ, getenv, path
from test.utils import run_verbose

HERE_DIR = path.dirname(path.abspath(__file__))
TRAINING_SCRIPT = path.abspath(path.join(HERE_DIR, "..", "..", "example", "train.py"))


def test_training_script():
    """Execute the training script."""
    # Use wandb in offline mode. We do not want to upload the logs this test generates
    ORIGINAL_WANDB_MODE = getenv("WANDB_MODE")
    environ["WANDB_MODE"] = "offline"

    # Run the training script
    run_verbose(["python", TRAINING_SCRIPT, "--epochs=3"])

    # Restore the original value of WANDB_MODE
    if ORIGINAL_WANDB_MODE is None:
        environ.pop("WANDB_MODE")
    else:
        environ["WANDB_MODE"] = ORIGINAL_WANDB_MODE
