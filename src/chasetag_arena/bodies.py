"""The body registry: the named bodies a game can be played with.

A body is a physics model (a family from ``core/families``) with a set of parameters. Every
body has a short name, such as ``dot`` or ``stepping``, and a label, ``<family>/<name>``, that
goes into frames and logs. Both seats play with the same body in a game, so the two sides
always have the same physics.

To add a body, write a function that returns its ``BodyParams`` and register it:

    register(BodySpec("heavy", "K", "Slow to start and to stop.", _heavy))

The tests in ``tests/test_bodies.py`` and ``tests/test_body_fairness.py`` then cover it.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass, replace
from typing import Final

import jax.numpy as jnp

from chasetag_arena.core.families import kinematic, spring, stepping
from chasetag_arena.core.families.base import BodyParams

DEFAULT_BODY: Final[str] = "default"


@dataclass(frozen=True)
class BodySpec:
    """One named body. ``build`` makes its parameters, and must not depend on any state."""

    name: str
    family: str  # "K", "SP" or "LS"
    summary: str  # one sentence, shown in the arcade and in the docs
    build: Callable[[], BodyParams]

    @property
    def label(self) -> str:
        return f"{self.family}/{self.name}"


def _k(**changes: float | int) -> BodyParams:
    """The default kinematic body with some fields changed. Values are floats, but ``delay``."""
    base = kinematic.default_params()
    values = {
        key: (jnp.asarray(value, dtype=jnp.int32) if key == "delay" else jnp.asarray(value))
        for key, value in changes.items()
    }
    return BodyParams(model=base.model, theta=replace(base.theta, **values))


def _default() -> BodyParams:
    return kinematic.default_params()


def _dot() -> BodyParams:
    """A point that follows commands almost at once, and never falls."""
    return _k(
        v_fwd=1.5,
        v_back=1.5,
        v_lat=1.5,
        a_max=8.0,
        a_dec=10.0,
        tau=0.05,
        delay=0,
        sigma_v=0.01,
        k_psi=8.0,
        omega_max=6.0,
        beta0=-30.0,
        beta_v=0.0,
        beta_a=0.0,
        beta_w=0.0,
        beta_c=0.0,
        beta_s=0.0,
    )


def _limited() -> BodyParams:
    """The default limits, with no command delay and a quicker response."""
    return _k(delay=0, tau=0.2)


def _sprinter() -> BodyParams:
    """Full speed forward but slow to turn, to stop and to move sideways."""
    return _k(
        v_fwd=1.5,
        v_back=0.6,
        v_lat=0.6,
        a_max=3.0,
        a_dec=1.5,
        tau=0.4,
        delay=8,
        k_psi=2.0,
        omega_max=1.5,
    )


def _dodger() -> BodyParams:
    """A little slower on the straight, but quick to turn, to stop and to sidestep.

    A hard change of direction raises the chance of a fall more than it does for the default.
    """
    return _k(
        v_fwd=1.2,
        v_back=1.0,
        v_lat=1.2,
        a_max=4.0,
        a_dec=5.0,
        tau=0.15,
        delay=5,
        k_psi=6.0,
        omega_max=5.0,
        beta_c=0.8,
    )


def _spring() -> BodyParams:
    return spring.default_params()


def _stepping() -> BodyParams:
    """The research priors, with a quicker stride.

    The research prior steps every 0.5 s, and then almost any turn loses balance: with the
    scripted bots, nearly every game ended in a fall. A 0.35 s step keeps the same reach and
    leaves a real risk (about 1 game in 15 with two greedy bots) without making the body
    unplayable. ``tests/test_body_fairness.py`` guards this.
    """
    base = stepping.default_params()
    return BodyParams(model=base.model, theta=replace(base.theta, T=jnp.asarray(0.35)))


_REGISTRY: dict[str, BodySpec] = {}


def register(spec: BodySpec) -> BodySpec:
    """Add a body. A name can be used once."""
    if spec.name in _REGISTRY:
        raise ValueError(f"body {spec.name!r} is already registered")
    if not spec.name.replace("-", "").isalnum() or spec.name != spec.name.lower():
        raise ValueError(f"body name must be lower-case letters, digits and '-': {spec.name!r}")
    _REGISTRY[spec.name] = spec
    return spec


register(
    BodySpec(
        "default",
        "K",
        "A person-like walker: 1.5 m/s, a tenth of a second of delay, and a risk of falling.",
        _default,
    )
)
register(BodySpec("dot", "K", "A point that follows commands at once and never falls.", _dot))
register(BodySpec("limited", "K", "Like the default, but with no command delay.", _limited))
register(
    BodySpec(
        "sprinter",
        "K",
        "Full speed forward, but slow to turn, to stop and to sidestep.",
        _sprinter,
    )
)
register(
    BodySpec(
        "dodger",
        "K",
        "A bit slower, but quick to turn and sidestep. Sharp changes can make it fall.",
        _dodger,
    )
)
register(
    BodySpec(
        "spring",
        "SP",
        "A wheeled base with a pendulum body on a spring. A hard push tilts it, and it can tip.",
        _spring,
    )
)
register(
    BodySpec(
        "stepping",
        "LS",
        "An inverted pendulum that takes steps. Too much speed or a sharp turn loses balance.",
        _stepping,
    )
)


def body_names() -> tuple[str, ...]:
    """Every registered body name, in registration order."""
    return tuple(_REGISTRY)


def get_body(name: str) -> BodySpec:
    """The body called ``name``. Raises ``ValueError`` for an unknown name."""
    try:
        return _REGISTRY[name]
    except KeyError:
        raise ValueError(f"unknown body {name!r}; choose one of {body_names()}") from None


def body_by_label(label: str) -> BodySpec:
    """The body with a label such as ``K/default``."""
    for spec in _REGISTRY.values():
        if spec.label == label:
            return spec
    raise ValueError(f"unknown body label {label!r}")


__all__ = [
    "DEFAULT_BODY",
    "BodySpec",
    "body_by_label",
    "body_names",
    "get_body",
    "register",
]
