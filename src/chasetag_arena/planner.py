"""The planner seat: a trained policy from a groundtag agent file, run for one body.

A planner file is chasetag's ``FrozenAgent`` format, as groundtag's exporter writes it. The
actor is in ``net/w{i}`` and ``net/b{i}``, where ``w`` is stored as (out, in) and applied as
``x @ w.T + b``. The observation normaliser is in ``net/obs_mean`` and ``net/obs_var``, and a
``meta`` JSON record names the activation and the layer count. The planner reads the 17
strategy features of spec v1 and outputs a raw planar command. ``to_command`` maps that output
to a command, exactly as it maps a bot's output.

The seat plays the mean of the policy, not a sample from it, so a round depends only on its
state. A walker policy (``stage`` "walker", the full-body stage) is refused. So is a file whose
decision interval or speed limit differs from the arena's rules. The walker arrays, if present,
are ignored, because this arena moves the body in the plane.
"""

from __future__ import annotations

import json
import math
import zipfile
from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Final

import jax.numpy as jnp
import numpy as np
from jax import Array

from chasetag_arena.agents.bots import Bot, init_stateless
from chasetag_arena.core.spec.features import SPEC_V1
from chasetag_arena.core.spec.rules import OFFICIAL

INPUT_DIM: Final[int] = len(SPEC_V1.names)  # 17 features, spec v1
OUTPUT_DIM: Final[int] = 2  # one raw planar command
# The normaliser's variance floor and clip, as chasetag applies them before the network.
VAR_FLOOR: Final[float] = 1e-4
OBS_CLIP: Final[float] = 10.0
ACTIVATIONS: Final[dict[str, Callable[[np.ndarray], np.ndarray]]] = {
    "elu": lambda x: np.where(x > 0, x, np.expm1(np.minimum(x, 0.0))),
    "tanh": np.tanh,
    "relu": lambda x: np.maximum(x, 0.0),
}


class PlannerError(ValueError):
    """A planner file that cannot be read, or that the arena cannot run."""


@dataclass(frozen=True)
class Planner:
    """A planner ready for inference. ``name`` is shown on the page; ``file`` is the file's name."""

    name: str
    file: str
    weights: tuple[np.ndarray, ...]  # each (out, in), as chasetag stores them
    biases: tuple[np.ndarray, ...]
    activation: str
    obs_mean: np.ndarray
    obs_var: np.ndarray

    def mean_action(self, features: np.ndarray) -> np.ndarray:
        """The raw planar command for a (17,) feature vector, as a (2,) array."""
        scale = np.sqrt(np.maximum(self.obs_var, VAR_FLOOR))
        x = np.clip((features - self.obs_mean) / scale, -OBS_CLIP, OBS_CLIP)
        act = ACTIVATIONS[self.activation]
        last = len(self.weights) - 1
        for i, (w, b) in enumerate(zip(self.weights, self.biases, strict=True)):
            x = x @ w.T + b
            if i < last:
                x = act(x)
        return x

    def as_bot(self) -> Bot:
        """The planner in the form a seat's bot takes. It keeps no state of its own."""

        def act(obs: Array, key: Array, state: Array) -> tuple[Array, Array]:
            del key
            raw = self.mean_action(np.asarray(obs, dtype=np.float64))
            return jnp.asarray(raw, dtype=obs.dtype), state

        return Bot(init_stateless, act)


def load_planner(path: str | Path) -> Planner:
    """Read a planner file, or raise ``PlannerError`` saying why the arena cannot run it."""
    file = Path(path)
    try:
        with np.load(file, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        raise PlannerError(f"cannot read planner file {file}: {exc}") from exc
    if "meta" not in arrays:
        raise PlannerError(f"{file} is not a planner file: it has no meta record")
    meta = _read_meta(file, arrays.pop("meta"))

    if meta.get("stage") == "walker":
        raise PlannerError(
            f"{file} is a walker policy for the full body; the arena needs a planner"
        )
    interval = meta.get("decision_interval")
    if interval != OFFICIAL.decision_substeps:
        raise PlannerError(
            f"{file} decides every {interval} control steps; "
            f"the arena decides every {OFFICIAL.decision_substeps}"
        )
    speed = meta.get("max_speed_mps")
    if not _is_number(speed) or not math.isclose(speed, OFFICIAL.max_speed_mps):
        raise PlannerError(
            f"{file} has a speed limit of {speed} m/s; the arena's is {OFFICIAL.max_speed_mps} m/s"
        )
    activation = meta.get("activation")
    if activation not in ACTIVATIONS:
        raise PlannerError(
            f"{file} uses activation {activation!r}; the arena supports {sorted(ACTIVATIONS)}"
        )
    n_layers = meta.get("n_layers")
    if not isinstance(n_layers, int) or isinstance(n_layers, bool) or n_layers < 1:
        raise PlannerError(f"{file} has no valid n_layers")

    weights: list[np.ndarray] = []
    biases: list[np.ndarray] = []
    width = INPUT_DIM
    for i in range(n_layers):
        w = _array(file, arrays, f"net/w{i}")
        b = _array(file, arrays, f"net/b{i}")
        if w.ndim != 2 or w.shape[1] != width:
            raise PlannerError(f"{file}: net/w{i} has shape {w.shape}, expected (out, {width})")
        if b.shape != (w.shape[0],):
            raise PlannerError(f"{file}: net/b{i} has shape {b.shape}, expected ({w.shape[0]},)")
        weights.append(w)
        biases.append(b)
        width = w.shape[0]
    if width != OUTPUT_DIM:
        raise PlannerError(f"{file} outputs {width} values; a planner outputs {OUTPUT_DIM}")

    obs_mean = _array(file, arrays, "net/obs_mean")
    obs_var = _array(file, arrays, "net/obs_var")
    if obs_mean.shape != (INPUT_DIM,) or obs_var.shape != (INPUT_DIM,):
        raise PlannerError(f"{file}: the normaliser must hold {INPUT_DIM} values, as spec v1 does")

    name = meta.get("name")
    if not isinstance(name, str) or name == "":
        name = file.stem
    return Planner(
        name=name,
        file=file.name,
        weights=tuple(weights),
        biases=tuple(biases),
        activation=activation,
        obs_mean=obs_mean,
        obs_var=obs_var,
    )


def _read_meta(file: Path, raw: np.ndarray) -> dict[str, Any]:
    try:
        meta = json.loads(raw.tobytes().decode("utf-8"))
    except ValueError as exc:
        raise PlannerError(f"{file}: the meta record is not readable JSON") from exc
    if not isinstance(meta, dict):
        raise PlannerError(f"{file}: the meta record must be a JSON object")
    return meta


def _array(file: Path, arrays: dict[str, np.ndarray], key: str) -> np.ndarray:
    if key not in arrays:
        raise PlannerError(f"{file} is missing {key}")
    try:
        value = np.asarray(arrays[key], dtype=np.float64)
    except (TypeError, ValueError) as exc:
        raise PlannerError(f"{file}: {key} is not an array of numbers") from exc
    if not np.all(np.isfinite(value)):
        raise PlannerError(f"{file}: {key} has values that are not finite")
    return value


def _is_number(value: object) -> bool:
    return isinstance(value, int | float) and not isinstance(value, bool)
