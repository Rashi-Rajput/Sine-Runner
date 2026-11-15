#!/usr/bin/env python3
"""Compatibility launcher for the modular :mod:`game` package."""

from game import *  # Re-export the historical public API.
from game.main import main


if __name__ == "__main__":
    raise SystemExit(main())
