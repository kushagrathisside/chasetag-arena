"""The planner seat: the file layout, the forward pass, and the files the arena refuses."""

from __future__ import annotations

import math
import re
from collections.abc import Callable
from pathlib import Path
from typing import Any

import jax
import jax.numpy as jnp
import numpy as np
import pytest

from chasetag_arena.planner import PlannerError, load_planner

Parts = tuple[dict[str, np.ndarray], dict[str, Any]]
Mutate = Callable[[dict[str, np.ndarray], dict[str, Any]], None]

# Written out separately from the module, so the test checks the arithmetic and not a copy of it.
REFERENCE_ACTIVATIONS: dict[str, Callable[[float], float]] = {
    "tanh": math.tanh,
    "relu": lambda x: max(x, 0.0),
    "elu": lambda x: x if x > 0 else math.expm1(x),
}


def reference_action(arrays: dict[str, np.ndarray], meta: dict[str, Any], obs: Any) -> np.ndarray:
    """The planner's output for one observation, computed one number at a time."""
    h = []
    for j in range(17):
        var = max(float(arrays["net/obs_var"][j]), 1e-4)
        z = (float(obs[j]) - float(arrays["net/obs_mean"][j])) / math.sqrt(var)
        h.append(min(max(z, -10.0), 10.0))
    last = meta["n_layers"] - 1
    for i in range(meta["n_layers"]):
        w, b = arrays[f"net/w{i}"], arrays[f"net/b{i}"]
        out = [
            float(b[k]) + sum(float(w[k, j]) * h[j] for j in range(len(h))) for k in range(len(b))
        ]
        if i < last:
            out = [REFERENCE_ACTIVATIONS[meta["activation"]](x) for x in out]
        h = out
    return np.array(h)


@pytest.mark.parametrize(
    ("activation", "sizes"),
    [("tanh", (17, 8, 2)), ("relu", (17, 8, 2)), ("elu", (17, 16, 8, 2))],
)
def test_the_forward_pass_matches_a_written_out_reference(
    planner_parts: Callable[..., Parts],
    save_planner: Any,
    tmp_path: Path,
    activation: str,
    sizes: tuple[int, ...],
) -> None:
    arrays, meta = planner_parts(sizes=sizes, activation=activation)
    planner = load_planner(save_planner(tmp_path / "p.npz", arrays, meta))
    rng = np.random.default_rng(1)
    for _ in range(20):
        obs = rng.normal(0.0, 3.0, size=17)  # wide enough that some features hit the clip
        np.testing.assert_allclose(
            planner.mean_action(obs), reference_action(arrays, meta, obs), atol=1e-9
        )


def test_the_normaliser_floors_the_variance_and_clips_the_input(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    # One linear layer that sums the 17 normalised features, so the output shows the scaling.
    arrays, meta = planner_parts(sizes=(17, 2))
    arrays["net/w0"] = np.ones((2, 17))
    arrays["net/b0"] = np.zeros(2)
    arrays["net/obs_mean"] = np.zeros(17)
    arrays["net/obs_var"] = np.full(17, 1e-6)  # below the floor, so the divisor is 0.01
    planner = load_planner(save_planner(tmp_path / "p.npz", arrays, meta))

    # 1.0 / 0.01 = 100, clipped to 10 for each of 17 features.
    np.testing.assert_allclose(planner.mean_action(np.ones(17)), [170.0, 170.0])
    # 0.02 / 0.01 = 2, which is inside the clip.
    np.testing.assert_allclose(planner.mean_action(np.full(17, 0.02)), [34.0, 34.0])


def test_walker_arrays_are_present_and_ignored(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    arrays, meta = planner_parts()
    arrays["walker/w0"] = np.zeros((3, 5))
    arrays["walker/b0"] = np.zeros(3)
    meta["walker_activation"] = "elu"
    meta["walker_layers"] = 1
    planner = load_planner(save_planner(tmp_path / "p.npz", arrays, meta))
    assert planner.mean_action(np.zeros(17)).shape == (2,)


def test_the_name_comes_from_the_meta_and_falls_back_to_the_file_name(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    arrays, meta = planner_parts()
    named = load_planner(save_planner(tmp_path / "walk.npz", arrays, meta))
    assert (named.name, named.file) == ("test-planner", "walk.npz")

    meta["name"] = ""
    unnamed = load_planner(save_planner(tmp_path / "hopper.npz", arrays, meta))
    assert (unnamed.name, unnamed.file) == ("hopper", "hopper.npz")


def test_the_bot_form_returns_the_raw_action_in_the_observation_dtype(
    planner_file: Path,
) -> None:
    planner = load_planner(planner_file)
    bot = planner.as_bot()
    obs = jnp.asarray(np.linspace(-1.0, 1.0, 17), dtype=jnp.float32)
    raw, state = bot.act(obs, jax.random.PRNGKey(0), bot.init_state(jax.random.PRNGKey(1)))
    np.testing.assert_allclose(
        np.asarray(raw), planner.mean_action(np.asarray(obs, dtype=np.float64)), atol=1e-6
    )
    assert raw.dtype == obs.dtype
    assert state.shape == (2,)


def _drop(key: str) -> Mutate:
    def mutate(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        del meta
        del arrays[key]

    return mutate


def _set(key: str, value: np.ndarray) -> Mutate:
    def mutate(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        del meta
        arrays[key] = value

    return mutate


def _meta(**changes: Any) -> Mutate:
    def mutate(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
        del arrays
        meta.update(changes)

    return mutate


def _wide_output(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    del meta
    arrays["net/w1"] = np.zeros((3, 8))
    arrays["net/b1"] = np.zeros(3)


def _non_finite(arrays: dict[str, np.ndarray], meta: dict[str, Any]) -> None:
    del meta
    arrays["net/w0"][0, 0] = np.nan


REFUSALS = [
    pytest.param(_meta(decision_interval=5), "decides every 5 control steps", id="interval"),
    pytest.param(_meta(max_speed_mps=2.0), "speed limit of 2.0", id="speed"),
    pytest.param(_meta(activation="gelu"), "activation 'gelu'", id="activation"),
    pytest.param(_meta(n_layers=0), "n_layers", id="no-layers"),
    pytest.param(_drop("net/w1"), "missing net/w1", id="missing-layer"),
    pytest.param(_drop("net/b0"), "missing net/b0", id="missing-bias"),
    pytest.param(_set("net/w0", np.zeros((8, 21))), "net/w0 has shape", id="input-width"),
    pytest.param(_set("net/b0", np.zeros(7)), "net/b0 has shape", id="bias-length"),
    pytest.param(_wide_output, "outputs 3 values", id="output-width"),
    pytest.param(_set("net/obs_mean", np.zeros(16)), "normaliser", id="normaliser-length"),
    pytest.param(_non_finite, "not finite", id="non-finite"),
    pytest.param(_meta(stage="walker"), "full-body model", id="walker-stage"),
]


@pytest.mark.parametrize(("mutate", "message"), REFUSALS)
def test_a_file_the_arena_cannot_run_is_refused(
    planner_parts: Callable[..., Parts],
    save_planner: Any,
    tmp_path: Path,
    mutate: Mutate,
    message: str,
) -> None:
    arrays, meta = planner_parts()
    mutate(arrays, meta)
    path = save_planner(tmp_path / "p.npz", arrays, meta)
    with pytest.raises(PlannerError, match=re.escape(message)):
        load_planner(path)


def test_a_file_that_is_not_an_archive_is_refused(tmp_path: Path) -> None:
    path = tmp_path / "planner.npz"
    path.write_text("not a planner", encoding="utf-8")
    with pytest.raises(PlannerError, match="cannot read planner file"):
        load_planner(path)


def test_a_missing_file_is_refused(tmp_path: Path) -> None:
    with pytest.raises(PlannerError, match="cannot read planner file"):
        load_planner(tmp_path / "none.npz")


def test_a_file_without_a_meta_record_is_refused(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    arrays, _ = planner_parts()
    path = save_planner(tmp_path / "p.npz", arrays, None)
    with pytest.raises(PlannerError, match="has no meta record"):
        load_planner(path)


@pytest.mark.parametrize(
    ("meta_bytes", "message"),
    [(b"{not json", "not readable JSON"), (b"[1, 2]", "must be a JSON object")],
)
def test_a_meta_record_that_is_not_an_object_is_refused(
    planner_parts: Callable[..., Parts],
    save_planner: Any,
    tmp_path: Path,
    meta_bytes: bytes,
    message: str,
) -> None:
    arrays, _ = planner_parts()
    path = save_planner(tmp_path / "p.npz", arrays, meta_bytes)
    with pytest.raises(PlannerError, match=message):
        load_planner(path)


# --- feature spec v2: the 17 features plus the last two decision commands -----------------------


def test_a_file_without_a_feature_spec_reads_spec_v1(planner_file: Path) -> None:
    planner = load_planner(planner_file)
    assert planner.feature_spec == 1


def test_a_spec_v2_file_reads_twenty_one_features(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    arrays, meta = planner_parts(sizes=(21, 8, 2), feature_spec=2)
    planner = load_planner(save_planner(tmp_path / "v2.npz", arrays, meta))
    assert planner.feature_spec == 2
    action = planner.mean_action(np.linspace(-1.0, 1.0, 21))
    assert action.shape == (2,)
    assert np.all(np.isfinite(action))


@pytest.mark.parametrize(
    ("sizes", "feature_spec", "message"),
    [
        pytest.param((17, 8, 2), 3, "feature spec 3", id="unknown-spec"),
        pytest.param((17, 8, 2), 2, "net/w0 has shape", id="v2-with-v1-width"),
        pytest.param((21, 8, 2), 1, "net/w0 has shape", id="v1-with-v2-width"),
        pytest.param((17, 8, 2), True, "feature spec True", id="bool-spec"),
    ],
)
def test_a_feature_spec_that_does_not_fit_the_network_is_refused(
    planner_parts: Callable[..., Parts],
    save_planner: Any,
    tmp_path: Path,
    sizes: tuple[int, ...],
    feature_spec: Any,
    message: str,
) -> None:
    arrays, meta = planner_parts(sizes=sizes, feature_spec=feature_spec)
    path = save_planner(tmp_path / "p.npz", arrays, meta)
    with pytest.raises(PlannerError, match=re.escape(message)):
        load_planner(path)


def test_the_normaliser_must_match_the_feature_spec(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    arrays, meta = planner_parts(sizes=(21, 8, 2), feature_spec=2)
    arrays["net/obs_mean"] = np.zeros(17)
    path = save_planner(tmp_path / "p.npz", arrays, meta)
    with pytest.raises(PlannerError, match=r"normaliser must hold 21 values, as spec v2 does"):
        load_planner(path)


def test_a_spec_v2_planner_plays_a_seat_with_the_v2_features(
    planner_parts: Callable[..., Parts], save_planner: Any, tmp_path: Path
) -> None:
    from chasetag_arena.arcade.game_loop import GameLoop

    arrays, meta = planner_parts(sizes=(21, 8, 2), feature_spec=2)
    planner = load_planner(save_planner(tmp_path / "v2.npz", arrays, meta))
    loop = GameLoop(seed=3, selectors=("planner", "greedy"), planner=planner)
    for _ in range(25):
        loop.decide()
    assert loop.decisions >= 5
    assert np.all(np.isfinite(np.asarray(loop.last_actions)))
