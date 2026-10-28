"""Shared fixtures: planner files in the layout groundtag's exporter writes."""

from __future__ import annotations

import itertools
import json
from collections.abc import Callable
from pathlib import Path
from typing import Any

import numpy as np
import pytest

Parts = tuple[dict[str, np.ndarray], dict[str, Any]]


@pytest.fixture
def planner_parts() -> Callable[..., Parts]:
    """Build the arrays and the meta record of a planner, in chasetag's layout.

    The network reads 17 features and has the layer sizes given, ending in a planar command.
    """

    def make(sizes: tuple[int, ...] = (17, 8, 2), activation: str = "tanh", seed: int = 0) -> Parts:
        rng = np.random.default_rng(seed)
        arrays: dict[str, np.ndarray] = {}
        for i, (fan_in, fan_out) in enumerate(itertools.pairwise(sizes)):
            # chasetag stores each matrix as (out, in).
            arrays[f"net/w{i}"] = rng.normal(0.0, 0.5, size=(fan_out, fan_in))
            arrays[f"net/b{i}"] = rng.normal(0.0, 0.1, size=fan_out)
        arrays["net/obs_mean"] = rng.normal(0.0, 0.5, size=17)
        arrays["net/obs_var"] = rng.uniform(0.5, 2.0, size=17)
        arrays["net/log_std"] = np.full(2, -0.7)
        meta: dict[str, Any] = {
            "name": "test-planner",
            "stage": "hierarchical",
            "activation": activation,
            "n_layers": len(sizes) - 1,
            "decision_interval": 10,
            "max_speed_mps": 1.5,
            "extra": {},
        }
        return arrays, meta

    return make


@pytest.fixture
def save_planner() -> Callable[..., Path]:
    """Write arrays and a meta record to an .npz file. ``meta`` may be a dict, raw bytes or None."""

    def save(path: Path, arrays: dict[str, np.ndarray], meta: Any) -> Path:
        out = dict(arrays)
        if isinstance(meta, bytes):
            out["meta"] = np.frombuffer(meta, dtype=np.uint8)
        elif meta is not None:
            out["meta"] = np.frombuffer(json.dumps(meta).encode(), dtype=np.uint8)
        np.savez(path, **out)
        return path

    return save


@pytest.fixture
def planner_file(tmp_path: Path, planner_parts: Callable[..., Parts], save_planner: Any) -> Path:
    """A valid planner file, named ``planner.npz``."""
    arrays, meta = planner_parts()
    return save_planner(tmp_path / "planner.npz", arrays, meta)
