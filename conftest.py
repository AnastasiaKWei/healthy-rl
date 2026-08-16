"""Make every test import the ``src`` tree of the checkout it lives in.

The venv's editable install points at one fixed checkout, so tests run from a
git worktree would otherwise silently exercise another tree's code (a trap
recorded in docs/infrastructure.md). Putting this checkout's ``src`` first on
``sys.path`` before collection makes ``import healthy_rl`` unambiguous.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent / "src"))
