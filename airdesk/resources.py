"""Locate bundled resources in both the source tree and a PyInstaller build."""

from __future__ import annotations

import sys
from pathlib import Path


def base_dir() -> Path:
    """Directory that contains the `airdesk` package data and `assets/`."""
    frozen = getattr(sys, "_MEIPASS", None)
    if frozen:
        return Path(frozen)
    return Path(__file__).resolve().parent.parent


def resource_path(*parts: str) -> Path:
    """Path of a file shipped inside the airdesk package (airdesk/<parts>)."""
    return base_dir() / "airdesk" / Path(*parts)


def asset_path(*parts: str) -> Path:
    """Path of a file under the top-level assets/ directory."""
    return base_dir() / "assets" / Path(*parts)


def model_path() -> Path:
    return asset_path("models", "hand_landmarker.task")
