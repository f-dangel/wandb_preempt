"""Tests for wandb_preempt/__init__.py."""

import time

import pytest

import wandb_preempt

NAMES = ["world", "github"]
IDS = NAMES


@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello(name: str):
    """Test hello function.

    Args:
        name: Name to greet.
    """
    wandb_preempt.hello(name)


@pytest.mark.expensive
@pytest.mark.parametrize("name", NAMES, ids=IDS)
def test_hello_expensive(name: str):
    """Expensive test of hello. Will only be run on master/main and development.

    Args:
        name: Name to greet.
    """
    time.sleep(1)
    wandb_preempt.hello(name)
