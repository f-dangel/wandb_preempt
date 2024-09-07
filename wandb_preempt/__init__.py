"""wandb_preempt library."""

from wandb_preempt.checkpointer import Checkpointer, get_resume_value

__all__ = [
    "Checkpointer",
    "get_resume_value",
]


# TODO Remove this function once we have a unit test that uses the checkpointer code
def hello(name):
    """Say hello to a name.

    Args:
        name (str): Name to say hello to.
    """
    print(f"Hello, {name}")
