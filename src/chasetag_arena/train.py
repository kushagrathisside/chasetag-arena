"""Headless PPO training for a planar planner on body family K.

The policy learns against every kind of opponent the game has. Each training game draws one
opponent: self-play, or one of the scripted bots (greedy, intercept, wander, idle, fallback). The
policy plays one seat, drawn at random, and the bot plays the other. In self-play both seats are
the policy. Only the policy's own decisions train it.

A run trains on one or more round lengths (20, 60 or 180 s). Each length keeps its own group of
games, because the rules are a static argument to the simulator. One update uses every group's
decisions, so a run over several lengths trains one planner for all of them.

The reward is the official score (design section 2.2), paid densely. Write ``frac`` for the
elapsed share of the round. A game that does not end in a fall scores the runner ``frac`` and the
chaser ``1 - frac``, so each decision pays the runner ``+dfrac`` and the chaser ``-dfrac``, and the
chaser gets ``+1`` at the start. A fall scores both seats 0, so the falling decision takes back
what each seat holds. Summed without discount, a game's rewards equal its official scores. The
dense form matters for long rounds: a 180 s round is 1800 decisions.

Only the actor is exported. The critic also sees the other seat's features, the opponent type and
the round length, since it is used only in training.

The planner file follows ``planner.py`` exactly, so training and the game compute the same mean
action. Run it from a checkout, with no display needed:

    uv run --no-sync python -m chasetag_arena.train --run-dir runs/k20 --limits 20 --iters 500
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import pickle
import time
from dataclasses import dataclass
from functools import partial
from pathlib import Path
from typing import Any, Final, NamedTuple

import jax
import jax.numpy as jnp
import numpy as np
from jax import Array, lax

from chasetag_arena.agents.bots import BOTS
from chasetag_arena.arcade.game_loop import rules_for_limit
from chasetag_arena.core import game
from chasetag_arena.core.families.base import BodyParams
from chasetag_arena.core.families.kinematic import default_params
from chasetag_arena.core.spec.features import SPEC_V1
from chasetag_arena.core.spec.rules import OFFICIAL, Rules
from chasetag_arena.core.state import FALL, RUNNING, TAG, TIMEOUT, GameState
from chasetag_arena.planner import OBS_CLIP, OUTPUT_DIM, VAR_FLOOR, load_planner

INPUT_DIM: Final[int] = SPEC_V1.dim  # 17 features, spec v1
BOT_NAMES: Final[tuple[str, ...]] = ("greedy", "intercept", "wander", "idle", "fallback")
N_BOTS: Final[int] = len(BOT_NAMES)
# Opponent codes: 0 is self-play, code k is the bot BOT_NAMES[k - 1], and code N_BOTS + 1 + s
# is the league snapshot in slot s.
N_OPPONENTS: Final[int] = N_BOTS + 1  # self-play and the bots, before any league slot
OPPONENT_LABELS: Final[tuple[str, ...]] = ("self", *BOT_NAMES)
# The bots a planner is ranked by. Idle never moves, so beating it says little.
SCORED_BOTS: Final[tuple[str, ...]] = tuple(b for b in BOT_NAMES if b != "idle")
LONGEST_LIMIT_S: Final[float] = 180.0  # scales the critic's round-length input
# The critic's context: the opponent kind (self, each bot, or any snapshot), then the length.
CTX_DIM: Final[int] = N_OPPONENTS + 2
CRITIC_DIM: Final[int] = 2 * INPUT_DIM + CTX_DIM
WARM_START_COUNT: Final[float] = 1e6  # weight of a warm start's normaliser statistics
Layers = list[tuple[Array, Array]]


class Agent(NamedTuple):
    """The learned parameters: the actor's mean, the critic, and the log std of the raw command."""

    actor: Layers
    critic: Layers
    log_std: Array  # (2,)


class Normalizer(NamedTuple):
    """Running mean and variance of the 17 features. The planner file stores these."""

    mean: Array  # (17,)
    var: Array  # (17,)
    count: Array  # ()


class Episode(NamedTuple):
    """Who each game is against. Redrawn whenever a game restarts."""

    opp: Array  # (N,) int32, opponent code (0 is self-play)
    learner: Array  # (N,) int32, the seat the policy plays against a bot
    bots: Array  # (N, K, 2) carried state of each bot, one row per bot


class Group(NamedTuple):
    """The games of one round length, carried from one round of training to the next."""

    states: GameState
    episodes: Episode


class League(NamedTuple):
    """Frozen past actors the policy can be drawn to play against (fictitious self-play).

    The slots are filled in turn as training goes on, and the oldest is replaced once all are
    full. Each snapshot keeps the normaliser it was saved with.
    """

    actors: Layers  # every leaf has a leading slot axis
    mean: Array  # (S, 17)
    var: Array  # (S, 17)
    filled: Array  # (S,) bool


class Traj(NamedTuple):
    """One rollout, shaped (T, N, ...): T decisions over N games."""

    obs: Array  # (T, N, 2, 17)
    ctx: Array  # (T, N, CTX_DIM) what the critic knows about the game
    raw: Array  # (T, N, 2, 2) sampled raw commands, one per seat
    logp: Array  # (T, N, 2)
    value: Array  # (T, N, 2)
    reward: Array  # (T, N, 2) dense official score
    done: Array  # (T, N) bool, this decision ended the game
    ended_by: Array  # (T, N) the game's end cause, read where done is set
    learn: Array  # (T, N, 2) bool, the policy chose this seat's command
    scores: Array  # (T, N, 2) official scores, set on the deciding step only


class Batch(NamedTuple):
    """A flattened rollout, ready for minibatches. ``weight`` is 1 where the policy learns."""

    obs: Array  # (M, 17)
    opp_obs: Array  # (M, 17) the other seat's features, for the critic
    ctx: Array  # (M, CTX_DIM)
    raw: Array  # (M, 2)
    logp: Array  # (M,)
    value: Array  # (M,)
    adv: Array  # (M,)
    ret: Array  # (M,)
    weight: Array  # (M,)


class Adam(NamedTuple):
    m: Agent
    v: Agent
    t: Array


@dataclass(frozen=True)
class Config:
    """Training settings. Frozen, so it can be a static argument to ``jit``."""

    limits: tuple[int, ...] = (20,)  # round lengths, in game seconds
    envs: int = 512  # games played in parallel, per round length
    iters: int = 500  # rollout-and-update rounds
    rollout: int = 256  # decisions per game in each round
    hidden: tuple[int, ...] = (64, 64)
    critic_hidden: tuple[int, ...] = (128, 128)
    lr: float = 3e-4
    anneal_lr: bool = True
    gamma: float = 0.999
    lam: float = 0.95
    clip: float = 0.2
    vf_coef: float = 0.5
    ent_coef: float = 0.001
    epochs: int = 4
    minibatches: int = 8
    max_grad_norm: float = 0.5
    seed: int = 0
    init_log_std: float = -0.5  # starting log std of the raw command noise
    # Rounds at the start that train only the critic. A warm-started actor keeps its cloned
    # behaviour while the critic, which starts from scratch, catches up.
    critic_warmup: int = 0
    # The league, and how each new game's opponent is drawn (prioritised fictitious self-play).
    league_slots: int = 8  # 0 turns the league off
    snapshot_every: int = 50  # rounds between snapshots of the actor
    self_play_share: float = 0.25
    snapshot_share: float = 0.25  # goes to the bots until the first snapshot exists
    pfsp_power: float = 1.0  # weight of an opponent is (1 - planner's score) ** power
    idle_weight: float = 0.1  # idle's weight among the bots, before normalising
    eval_every: int = 25
    eval_games: int = 512
    checkpoint_every: int = 25


def init_mlp(key: Array, sizes: tuple[int, ...], out_gain: float) -> Layers:
    """Orthogonal layers stored as (out, in), as the planner file stores them."""
    hidden_init = jax.nn.initializers.orthogonal(scale=jnp.sqrt(2.0))
    out_init = jax.nn.initializers.orthogonal(scale=out_gain)
    keys = jax.random.split(key, len(sizes) - 1)
    layers: Layers = []
    last = len(sizes) - 2
    for i, (k, n_in, n_out) in enumerate(zip(keys, sizes[:-1], sizes[1:], strict=True)):
        init = out_init if i == last else hidden_init
        layers.append((init(k, (n_out, n_in)), jnp.zeros(n_out)))
    return layers


def init_agent(
    key: Array, hidden: tuple[int, ...], critic_hidden: tuple[int, ...], log_std: float = -0.5
) -> Agent:
    k_actor, k_critic = jax.random.split(key)
    actor = init_mlp(k_actor, (INPUT_DIM, *hidden, OUTPUT_DIM), out_gain=0.01)
    critic = init_mlp(k_critic, (CRITIC_DIM, *critic_hidden, 1), out_gain=1.0)
    return Agent(actor, critic, jnp.full((OUTPUT_DIM,), log_std))


def actor_from_planner(path: Path, hidden: tuple[int, ...]) -> tuple[Layers, Normalizer]:
    """A planner file's actor and normaliser, to start training from (a warm start).

    The normaliser's count is set high, so the running statistics move only slowly away from
    the ones the actor was fitted with.
    """
    planner = load_planner(path)
    sizes = tuple(w.shape[0] for w in planner.weights[:-1])
    if sizes != hidden:
        raise ValueError(f"{path} has hidden sizes {sizes}; this run uses {hidden}")
    if planner.activation != "tanh":
        raise ValueError(f"{path} uses {planner.activation}; the trainer's actor uses tanh")
    layers = [
        (jnp.asarray(w, jnp.float32), jnp.asarray(b, jnp.float32))
        for w, b in zip(planner.weights, planner.biases, strict=True)
    ]
    norm = Normalizer(
        mean=jnp.asarray(planner.obs_mean, jnp.float32),
        var=jnp.asarray(planner.obs_var, jnp.float32),
        count=jnp.float32(WARM_START_COUNT),
    )
    return layers, norm


def init_normalizer() -> Normalizer:
    return Normalizer(mean=jnp.zeros(INPUT_DIM), var=jnp.ones(INPUT_DIM), count=jnp.float32(1e-4))


def mlp(layers: Layers, x: Array) -> Array:
    """tanh on every layer but the last, as ``planner.Planner.mean_action`` applies it."""
    last = len(layers) - 1
    for i, (w, b) in enumerate(layers):
        x = x @ w.T + b
        if i < last:
            x = jnp.tanh(x)
    return x


def normalize(norm: Normalizer, obs: Array) -> Array:
    scale = jnp.sqrt(jnp.maximum(norm.var, VAR_FLOOR))
    return jnp.clip((obs - norm.mean) / scale, -OBS_CLIP, OBS_CLIP)


def actor(agent: Agent, norm: Normalizer, obs: Array) -> Array:
    """The mean raw command (..., 2): the only part a planner file holds."""
    return mlp(agent.actor, normalize(norm, obs))


def critic(agent: Agent, norm: Normalizer, obs: Array, opp_obs: Array, ctx: Array) -> Array:
    """The value (...) of a seat, from both seats' features and the game's context."""
    x = jnp.concatenate([normalize(norm, obs), normalize(norm, opp_obs), ctx], axis=-1)
    return mlp(agent.critic, x)[..., 0]


def gaussian_logp(mu: Array, log_std: Array, raw: Array) -> Array:
    var = jnp.exp(2.0 * log_std)
    terms = -0.5 * (raw - mu) ** 2 / var - log_std - 0.5 * jnp.log(2.0 * jnp.pi)
    return jnp.sum(terms, axis=-1)


def seat_observations(states: GameState, rules: Rules = OFFICIAL) -> Array:
    """The spec v1 features of both seats, shape (games, 2, 17).

    The time feature is relative to ``rules.episode_s``, so pass the round's own rules, as the
    game loop does.
    """

    def one(state: GameState) -> Array:
        return jnp.stack([SPEC_V1.encode(state, 0, rules), SPEC_V1.encode(state, 1, rules)])

    return jax.vmap(one)(states)


def batched_reset(keys: Array, params: BodyParams, rules: Rules = OFFICIAL) -> GameState:
    return jax.vmap(lambda k: game.reset(k, params, rules))(keys)


def batched_step(
    states: GameState, raw: Array, params: BodyParams, rules: Rules
) -> tuple[GameState, Array]:
    return jax.vmap(lambda s, a: game.step(s, a, params, rules))(states, raw)


def dense_rewards(before: GameState, after: GameState, rules: Rules) -> Array:
    """One decision's share of each seat's official score, shape (2,).

    Summed over a game without discount, these equal ``game.official_scores`` at its end.
    """
    dtype = after.bodies.pos.dtype
    per_step = rules.control_dt / rules.episode_s
    frac0 = before.step.astype(dtype) * per_step
    frac1 = after.step.astype(dtype) * per_step
    running = before.ended_by == RUNNING
    gained = jnp.where(running, frac1 - frac0, 0.0)
    is_chaser = jnp.arange(2) == before.chaser
    start = jnp.where(running & (before.step == 0), 1.0, 0.0)
    reward = jnp.where(is_chaser, start - gained, gained)
    fell = running & (after.ended_by == FALL)
    taken = jnp.where(is_chaser, 1.0 - frac1, frac1)
    return jnp.where(fell, reward - taken, reward).astype(dtype)


def select_games(mask: Array, fresh: Any, old: Any) -> Any:
    """Per game, take ``fresh`` where ``mask`` is set and ``old`` elsewhere."""

    def pick(new_leaf: Array, old_leaf: Array) -> Array:
        return jnp.where(mask.reshape(mask.shape + (1,) * (old_leaf.ndim - 1)), new_leaf, old_leaf)

    return jax.tree_util.tree_map(pick, fresh, old)


def init_bots(key: Array, n: int) -> Array:
    """The starting state of every bot in every game, shape (N, K, 2)."""
    keys = jax.random.split(key, n * N_BOTS).reshape(n, N_BOTS, 2)
    cols = [jax.vmap(BOTS[name].init_state)(keys[:, k]) for k, name in enumerate(BOT_NAMES)]
    return jnp.stack(cols, axis=1)


def league_slots(cfg: Config) -> int:
    """The league's slot count. Always at least one, so its arrays keep a shape."""
    return max(cfg.league_slots, 1)


def init_league(actor_layers: Layers, slots: int) -> League:
    """An empty league with room for ``slots`` snapshots of an actor shaped like this one."""
    return League(
        actors=jax.tree_util.tree_map(lambda x: jnp.zeros((slots, *x.shape)), actor_layers),
        mean=jnp.zeros((slots, INPUT_DIM)),
        var=jnp.ones((slots, INPUT_DIM)),
        filled=jnp.zeros((slots,), dtype=bool),
    )


def add_snapshot(league: League, slot: int, agent: Agent, norm: Normalizer) -> League:
    """The league with the current actor frozen into ``slot``."""
    return League(
        actors=jax.tree_util.tree_map(lambda s, x: s.at[slot].set(x), league.actors, agent.actor),
        mean=league.mean.at[slot].set(norm.mean),
        var=league.var.at[slot].set(norm.var),
        filled=league.filled.at[slot].set(True),
    )


def new_episodes(key: Array, n: int, probs: Array | None = None) -> Episode:
    """Draw an opponent and a learner seat for each of ``n`` games.

    ``probs`` weights the opponent codes. Without it, self-play and the bots are equally likely.
    """
    k_opp, k_seat, k_bots = jax.random.split(key, 3)
    if probs is None:
        opp = jax.random.randint(k_opp, (n,), 0, N_OPPONENTS, dtype=jnp.int32)
    else:
        opp = jax.random.choice(k_opp, probs.shape[0], (n,), p=probs).astype(jnp.int32)
    learner = jax.random.randint(k_seat, (n,), 0, 2, dtype=jnp.int32)
    return Episode(opp=opp, learner=learner, bots=init_bots(k_bots, n))


def context(episodes: Episode, rules: Rules) -> Array:
    """What the critic knows beyond the features: the opponent kind and the round length."""
    n = episodes.opp.shape[0]
    kind = jnp.where(episodes.opp > N_BOTS, N_OPPONENTS, episodes.opp)
    onehot = jax.nn.one_hot(kind, N_OPPONENTS + 1)
    length = jnp.full((n, 1), rules.episode_s / LONGEST_LIMIT_S)
    return jnp.concatenate([onehot, length], axis=-1)


def opponent_obs(episodes: Episode, obs: Array) -> Array:
    """The features of the seat the policy does not play against a bot or snapshot, (N, 17)."""
    opp_seat = 1 - episodes.learner
    return jnp.take_along_axis(obs, opp_seat[:, None, None], axis=1)[:, 0]


def snapshot_actions(league: League, opp_obs: Array) -> Array:
    """Every snapshot's mean raw command for each game's opponent seat, (S, N, 2)."""

    def one(layers: Layers, mean: Array, var: Array) -> Array:
        return mlp(layers, normalize(Normalizer(mean, var, jnp.float32(1.0)), opp_obs))

    return jax.vmap(one)(league.actors, league.mean, league.var)


def bot_actions(episodes: Episode, obs: Array, key: Array) -> tuple[Array, Array]:
    """The raw command of each game's bot, and every bot's next state.

    Every bot is run on each game, then each game keeps the one it was drawn to play. The bots
    are cheap, so this costs less than branching per game.
    """
    n = obs.shape[0]
    opp_obs = opponent_obs(episodes, obs)
    keys = jax.random.split(key, n * N_BOTS).reshape(n, N_BOTS, 2)
    raws, states = [], []
    for k, name in enumerate(BOT_NAMES):
        raw_k, state_k = jax.vmap(BOTS[name].act)(opp_obs, keys[:, k], episodes.bots[:, k])
        raws.append(raw_k)
        states.append(state_k)
    all_raw = jnp.stack(raws, axis=1)  # (N, K, 2)
    index = jnp.clip(episodes.opp - 1, 0, N_BOTS - 1)
    chosen = jnp.take_along_axis(all_raw, index[:, None, None], axis=1)[:, 0]
    return chosen, jnp.stack(states, axis=1)


def joint_actions(
    episodes: Episode, obs: Array, own_raw: Array, key: Array, league: League
) -> tuple[Array, Array, Array]:
    """Both seats' raw commands: the policy's where it plays, the opponent's elsewhere.

    The opponent is a bot or a league snapshot. Returns the commands (N, 2, 2), the mask of
    seats the policy controls (N, 2), and the bots' next states.
    """
    n = obs.shape[0]
    bot_raw, bots = bot_actions(episodes, obs, key)
    slots = league.filled.shape[0]
    slot = jnp.clip(episodes.opp - N_BOTS - 1, 0, slots - 1)
    snap_raw = snapshot_actions(league, opponent_obs(episodes, obs))[slot, jnp.arange(n)]
    opp_raw = jnp.where((episodes.opp > N_BOTS)[:, None], snap_raw, bot_raw)
    seats = jnp.arange(2)
    on_opp = seats[None, :] != episodes.learner[:, None]
    learn = (episodes.opp == 0)[:, None] | ~on_opp
    raw = jnp.where(
        learn[..., None],
        own_raw,
        jnp.where(on_opp[..., None], opp_raw[:, None, :], 0.0),
    )
    return raw, learn, bots


def seat_values(agent: Agent, norm: Normalizer, obs: Array, ctx: Array) -> Array:
    """The critic's value of both seats, (N, 2), from features (N, 2, 17) and context (N, C)."""
    ctx2 = jnp.broadcast_to(ctx[:, None, :], (*obs.shape[:2], ctx.shape[-1]))
    return critic(agent, norm, obs, obs[:, ::-1], ctx2)


def advantages(traj: Traj, last_value: Array, cfg: Config) -> tuple[Array, Array]:
    """Generalised advantage estimates and their returns, per decision and seat.

    A game that ends is terminal. A game still running at the end of the rollout is bootstrapped
    from ``last_value``.
    """
    next_value = jnp.concatenate([traj.value[1:], last_value[None]], axis=0)
    not_done = (1.0 - traj.done.astype(traj.reward.dtype))[..., None]
    delta = traj.reward + cfg.gamma * next_value * not_done - traj.value

    def back(carry: Array, x: tuple[Array, Array]) -> tuple[Array, Array]:
        d, nd = x
        adv = d + cfg.gamma * cfg.lam * nd * carry
        return adv, adv

    _, adv = lax.scan(back, jnp.zeros_like(last_value), (delta, not_done), reverse=True)
    return adv, adv + traj.value


def collect(
    agent: Agent,
    norm: Normalizer,
    group: Group,
    key: Array,
    params: BodyParams,
    rules: Rules,
    steps: int,
    league: League,
    probs: Array,
) -> tuple[Group, Traj]:
    """Play ``steps`` decisions in every game. A finished game restarts with a new opponent."""
    n = group.states.ended_by.shape[0]

    def decide(carry: tuple[GameState, Episode, Array], _: None) -> tuple[Any, Traj]:
        states, episodes, key = carry
        key, k_act, k_bot, k_reset, k_new = jax.random.split(key, 5)
        obs = seat_observations(states, rules)
        ctx = context(episodes, rules)
        mu = actor(agent, norm, obs)
        value = seat_values(agent, norm, obs, ctx)
        sampled = mu + jnp.exp(agent.log_std) * jax.random.normal(k_act, mu.shape)
        logp = gaussian_logp(mu, agent.log_std, sampled)
        raw, learn, bots = joint_actions(episodes, obs, sampled, k_bot, league)
        nxt, scores = batched_step(states, raw, params, rules)
        reward = jax.vmap(lambda b, a: dense_rewards(b, a, rules))(states, nxt)
        done = nxt.ended_by != RUNNING
        fresh_states = batched_reset(jax.random.split(k_reset, n), params, rules)
        fresh_episodes = new_episodes(k_new, n, probs)
        states = select_games(done, fresh_states, nxt)
        episodes = select_games(done, fresh_episodes, episodes._replace(bots=bots))
        record = Traj(obs, ctx, sampled, logp, value, reward, done, nxt.ended_by, learn, scores)
        return (states, episodes, key), record

    init = (group.states, group.episodes, key)
    (states, episodes, _), traj = lax.scan(decide, init, None, length=steps)
    return Group(states, episodes), traj


def flatten(traj: Traj, adv: Array, ret: Array) -> Batch:
    ctx2 = jnp.broadcast_to(traj.ctx[:, :, None, :], (*traj.obs.shape[:3], CTX_DIM))
    return Batch(
        obs=traj.obs.reshape(-1, INPUT_DIM),
        opp_obs=traj.obs[:, :, ::-1].reshape(-1, INPUT_DIM),
        ctx=ctx2.reshape(-1, CTX_DIM),
        raw=traj.raw.reshape(-1, OUTPUT_DIM),
        logp=traj.logp.reshape(-1),
        value=traj.value.reshape(-1),
        adv=adv.reshape(-1),
        ret=ret.reshape(-1),
        weight=traj.learn.reshape(-1).astype(jnp.float32),
    )


def ppo_loss(agent: Agent, norm: Normalizer, mb: Batch, cfg: Config, policy_scale: Array) -> Array:
    """The PPO loss, averaged over the decisions the policy made (weight 1).

    ``policy_scale`` is 0 while the critic warms up, so only the value term trains.
    """
    mu = actor(agent, norm, mb.obs)
    value = critic(agent, norm, mb.obs, mb.opp_obs, mb.ctx)
    logp = gaussian_logp(mu, agent.log_std, mb.raw)
    ratio = jnp.exp(logp - mb.logp)
    clipped = jnp.clip(ratio, 1.0 - cfg.clip, 1.0 + cfg.clip)
    denom = jnp.maximum(jnp.sum(mb.weight), 1.0)
    policy = -jnp.sum(mb.weight * jnp.minimum(ratio * mb.adv, clipped * mb.adv)) / denom
    value_clipped = mb.value + jnp.clip(value - mb.value, -cfg.clip, cfg.clip)
    value_terms = jnp.maximum((value - mb.ret) ** 2, (value_clipped - mb.ret) ** 2)
    value_loss = 0.5 * jnp.sum(mb.weight * value_terms) / denom
    entropy = jnp.sum(agent.log_std + 0.5 * jnp.log(2.0 * jnp.pi * jnp.e))
    return policy_scale * (policy - cfg.ent_coef * entropy) + cfg.vf_coef * value_loss


def clip_global_norm(grads: Agent, max_norm: float) -> Agent:
    leaves = jax.tree_util.tree_leaves(grads)
    norm = jnp.sqrt(sum(jnp.sum(x * x) for x in leaves))
    scale = jnp.minimum(1.0, max_norm / (norm + 1e-8))
    return jax.tree_util.tree_map(lambda g: g * scale, grads)


def adam_step(agent: Agent, grads: Agent, opt: Adam, lr: Array, cfg: Config) -> tuple[Agent, Adam]:
    grads = clip_global_norm(grads, cfg.max_grad_norm)
    t = opt.t + 1.0
    b1, b2 = 0.9, 0.999
    m = jax.tree_util.tree_map(lambda a, g: b1 * a + (1 - b1) * g, opt.m, grads)
    v = jax.tree_util.tree_map(lambda a, g: b2 * a + (1 - b2) * g * g, opt.v, grads)
    step = lr * jnp.sqrt(1.0 - b2**t) / (1.0 - b1**t)
    agent = jax.tree_util.tree_map(lambda p, a, b: p - step * a / (jnp.sqrt(b) + 1e-8), agent, m, v)
    return agent, Adam(m, v, t)


def ppo_update(
    agent: Agent,
    opt: Adam,
    norm: Normalizer,
    batch: Batch,
    key: Array,
    lr: Array,
    policy_scale: Array,
    cfg: Config,
) -> tuple[Agent, Adam, Array]:
    """Several epochs of shuffled minibatch updates on one rollout."""
    size = batch.adv.shape[0] // cfg.minibatches
    grad_fn = jax.value_and_grad(ppo_loss)

    def minibatch(carry: tuple[Agent, Adam], ids: Array) -> tuple[tuple[Agent, Adam], Array]:
        agent, opt = carry
        mb = jax.tree_util.tree_map(lambda x: x[ids], batch)
        loss, grads = grad_fn(agent, norm, mb, cfg, policy_scale)
        agent, opt = adam_step(agent, grads, opt, lr, cfg)
        return (agent, opt), loss

    def epoch(carry: tuple[Agent, Adam], k: Array) -> tuple[tuple[Agent, Adam], Array]:
        perm = jax.random.permutation(k, batch.adv.shape[0])[: size * cfg.minibatches]
        carry, losses = lax.scan(minibatch, carry, perm.reshape(cfg.minibatches, size))
        return carry, losses.mean()

    (agent, opt), losses = lax.scan(epoch, (agent, opt), jax.random.split(key, cfg.epochs))
    return agent, opt, losses.mean()


def update_normalizer(norm: Normalizer, obs: Array) -> Normalizer:
    """Merge a batch's mean and variance into the running ones (Chan et al.)."""
    count = obs.shape[0]
    delta = obs.mean(axis=0) - norm.mean
    total = norm.count + count
    mean = norm.mean + delta * count / total
    m2 = norm.var * norm.count + obs.var(axis=0) * count + delta**2 * norm.count * count / total
    return Normalizer(mean=mean, var=m2 / total, count=total)


def episode_stats(traj: Traj) -> dict[str, Array]:
    def count(code: int) -> Array:
        return jnp.sum(traj.done & (traj.ended_by == code))

    learned = traj.done[..., None] & traj.learn
    return {
        "games": jnp.sum(traj.done),
        "tag": count(TAG),
        "fall": count(FALL),
        "timeout": count(TIMEOUT),
        "learner_score_sum": jnp.sum(jnp.where(learned, traj.scores, 0.0)),
        "learner_seats": jnp.sum(learned),
    }


def make_iteration(params: BodyParams, cfg: Config):
    """One round: collect every length's games, update the policy, then the normaliser.

    The normaliser used for the update is the one the rollout used, so the stored log
    probabilities stay valid. The new statistics take effect in the next round.
    """
    all_rules = tuple(rules_for_limit(limit) for limit in cfg.limits)

    @jax.jit
    def iteration(agent, opt, norm, groups, key, lr, policy_scale, league, probs):
        key, k_upd, *k_rolls = jax.random.split(key, 2 + len(all_rules))
        new_groups, batches, stats = [], [], []
        for rules, group, k_roll in zip(all_rules, groups, k_rolls, strict=True):
            group, traj = collect(
                agent, norm, group, k_roll, params, rules, cfg.rollout, league, probs
            )
            last_obs = seat_observations(group.states, rules)
            last_value = seat_values(agent, norm, last_obs, context(group.episodes, rules))
            adv, ret = advantages(traj, last_value, cfg)
            new_groups.append(group)
            batches.append(flatten(traj, adv, ret))
            stats.append(episode_stats(traj))
        batch = jax.tree_util.tree_map(lambda *xs: jnp.concatenate(xs), *batches)
        # Normalise the advantages over the decisions the policy made, not the bots'.
        total = jnp.maximum(batch.weight.sum(), 1.0)
        mean = jnp.sum(batch.weight * batch.adv) / total
        std = jnp.sqrt(jnp.sum(batch.weight * (batch.adv - mean) ** 2) / total)
        batch = batch._replace(adv=(batch.adv - mean) / (std + 1e-8))
        agent, opt, loss = ppo_update(agent, opt, norm, batch, k_upd, lr, policy_scale, cfg)
        new_norm = update_normalizer(norm, batch.obs)
        return agent, opt, new_norm, tuple(new_groups), key, loss, stats

    return iteration


def make_evaluator(params: BodyParams):
    """Play full games with the mean action, against itself, one bot, or one league snapshot.

    The planner always plays seat 0. Each game's roles still come from the reset, so the planner
    is chaser in about half the games and runner in the rest, and both are reported.
    """

    @partial(jax.jit, static_argnames=("n", "opp_code", "limit_s"))
    def run(
        agent: Agent,
        norm: Normalizer,
        league: League,
        key: Array,
        n: int,
        opp_code: int,
        limit_s: int,
    ):
        rules = rules_for_limit(limit_s)
        k_states, k_bots, k_steps = jax.random.split(key, 3)
        states = batched_reset(jax.random.split(k_states, n), params, rules)
        episodes = Episode(
            opp=jnp.full((n,), opp_code, dtype=jnp.int32),
            learner=jnp.zeros((n,), dtype=jnp.int32),
            bots=init_bots(k_bots, n),
        )

        def decide(carry: tuple[GameState, Episode, Array], _: None) -> tuple[Any, None]:
            states, episodes, key = carry
            key, k_bot = jax.random.split(key)
            obs = seat_observations(states, rules)
            raw, _, bots = joint_actions(episodes, obs, actor(agent, norm, obs), k_bot, league)
            states, _ = batched_step(states, raw, params, rules)
            return (states, episodes._replace(bots=bots), key), None

        steps = rules.decisions_per_episode
        (states, _, _), _ = lax.scan(decide, (states, episodes, k_steps), None, length=steps)
        score = jax.vmap(lambda s: game.official_scores(s, rules))(states)[:, 0]
        chaser = states.chaser == 0

        def mean_where(mask: Array) -> Array:
            return jnp.sum(jnp.where(mask, score, 0.0)) / jnp.maximum(jnp.sum(mask), 1)

        return {
            "score": jnp.mean(score),
            "score_chaser": mean_where(chaser),
            "score_runner": mean_where(~chaser),
            "tag": jnp.mean(states.ended_by == TAG),
            "fall": jnp.mean(states.ended_by == FALL),
            "timeout": jnp.mean(states.ended_by == TIMEOUT),
        }

    return run


def evaluate(
    evaluator, agent: Agent, norm: Normalizer, cfg: Config, league: League | None = None
) -> dict[str, Any]:
    """The evaluation matrix: every round length against self-play and every bot.

    Filled league snapshots are played at the shortest round length only, to keep evaluation
    cheap; their scores weight how often they are drawn. ``opponent_scores`` holds the score
    against each opponent, averaged over the lengths it was played at.
    """
    if league is None:
        league = init_league(agent.actor, 1)
    key = jax.random.PRNGKey(cfg.seed + 1000)
    matrix: dict[str, dict[str, dict[str, float]]] = {}
    filled = np.asarray(league.filled)
    for i, limit in enumerate(cfg.limits):
        row = {}
        codes = list(enumerate(OPPONENT_LABELS))
        if i == 0:
            codes += [(N_BOTS + 1 + s, f"snap{s}") for s in np.nonzero(filled)[0]]
        for code, label in codes:
            res = evaluator(agent, norm, league, key, cfg.eval_games, code, limit)
            row[label] = {k: round(float(v), 4) for k, v in res.items()}
        matrix[str(limit)] = row
    ranked = [matrix[str(limit)][b]["score"] for limit in cfg.limits for b in SCORED_BOTS]
    labels = {label for row in matrix.values() for label in row}
    opponent_scores = {
        label: float(np.mean([row[label]["score"] for row in matrix.values() if label in row]))
        for label in labels
    }
    return {"matrix": matrix, "mean_score": float(np.mean(ranked)), "opponents": opponent_scores}


def opponent_probs(cfg: Config, scores: dict[str, float] | None, filled: np.ndarray) -> np.ndarray:
    """How likely each opponent code is to be drawn for a new game.

    Self-play gets a fixed share. The bots and the league snapshots share the rest, each
    weighted by how badly the planner does against it (prioritised fictitious self-play, as in
    AlphaStar). Before the first evaluation every bot weighs the same. Idle keeps a small fixed
    weight, since it never moves.
    """
    slots = len(filled)
    probs = np.zeros(N_OPPONENTS + slots)
    probs[0] = cfg.self_play_share

    def hard(label: str) -> float:
        if not scores or label not in scores:
            return 1.0
        return max(1.0 - scores[label], 0.05) ** cfg.pfsp_power

    snap_share = cfg.snapshot_share if cfg.league_slots > 0 and filled.any() else 0.0
    bot_w = np.array([cfg.idle_weight if b == "idle" else hard(b) for b in BOT_NAMES])
    probs[1 : 1 + N_BOTS] = (1.0 - probs[0] - snap_share) * bot_w / bot_w.sum()
    if snap_share > 0:
        snap_w = np.array([hard(f"snap{s}") if filled[s] else 0.0 for s in range(slots)])
        probs[1 + N_BOTS :] = snap_share * snap_w / snap_w.sum()
    return probs / probs.sum()


def save_planner(
    path: Path, agent: Agent, norm: Normalizer, name: str, cfg: Config, iteration: int
) -> None:
    """Write the planner file that ``planner.load_planner`` reads."""
    arrays: dict[str, np.ndarray] = {}
    for i, (w, b) in enumerate(agent.actor):
        arrays[f"net/w{i}"] = np.asarray(w, dtype=np.float64)
        arrays[f"net/b{i}"] = np.asarray(b, dtype=np.float64)
    arrays["net/obs_mean"] = np.asarray(norm.mean, dtype=np.float64)
    arrays["net/obs_var"] = np.asarray(norm.var, dtype=np.float64)
    meta: dict[str, Any] = {
        "name": name,
        "stage": "planar",
        "family": "K",
        "spec": SPEC_V1.version,
        "activation": "tanh",
        "n_layers": len(agent.actor),
        "decision_interval": OFFICIAL.decision_substeps,
        "max_speed_mps": OFFICIAL.max_speed_mps,
        "hidden": list(cfg.hidden),
        "round_limits_s": list(cfg.limits),
        "opponents": list(BOT_NAMES),
        "seed": cfg.seed,
        "iteration": iteration,
        "trainer": "chasetag_arena.train",
    }
    arrays["meta"] = np.frombuffer(json.dumps(meta).encode("utf-8"), dtype=np.uint8)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp.npz")
    np.savez(tmp, **arrays)
    tmp.replace(path)


def save_checkpoint(path: Path, state: dict[str, Any]) -> None:
    """Everything needed to resume a run, written atomically."""
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("wb") as f:
        pickle.dump(jax.device_get(state), f)
    tmp.replace(path)


def load_checkpoint(path: Path) -> dict[str, Any]:
    """Read a checkpoint this trainer wrote. Only open checkpoints you made yourself."""
    with path.open("rb") as f:
        state = pickle.load(f)
    for key in ("agent", "opt", "norm", "groups", "key", "league"):
        state[key] = jax.tree_util.tree_map(jnp.asarray, state[key])
    return state


def _append_jsonl(path: Path, record: dict[str, Any]) -> None:
    with path.open("a") as f:
        f.write(json.dumps(record) + "\n")


def train(
    cfg: Config,
    run_dir: Path,
    name: str,
    resume: bool = False,
    init: Path | None = None,
    log=print,
):
    """Train against every opponent type, and keep the planner with the best ranked score.

    The run folder holds ``planner.npz`` (the best planner so far), ``checkpoint.pkl`` (to
    resume), ``metrics.jsonl`` (one line per round, and one per evaluation) and ``config.json``.
    The ranked score is the planner's mean official score against each bot but idle, over the
    run's round lengths. ``init`` is a planner file whose actor and normaliser start the run,
    such as one from ``chasetag_arena.bc``. A resumed run ignores it.
    """
    run_dir.mkdir(parents=True, exist_ok=True)
    params = default_params()
    ckpt_path = run_dir / "checkpoint.pkl"
    metrics = run_dir / "metrics.jsonl"
    if resume and ckpt_path.exists():
        state = load_checkpoint(ckpt_path)
        if state["config"] != dataclasses.asdict(cfg):
            log("  note: the config differs from the checkpoint's; resuming with the new one")
        agent, opt, norm = state["agent"], state["opt"], state["norm"]
        groups, key = tuple(Group(*g) for g in state["groups"]), state["key"]
        start_it, best = state["iteration"] + 1, state["best"]
        league, scores = League(*state["league"]), state["scores"]
        snapshots_taken = state["snapshots_taken"]
        log(f"resumed from {ckpt_path} at round {start_it}")
        evaluator = make_evaluator(params)
    else:
        key = jax.random.PRNGKey(cfg.seed)
        key, k_init, *k_groups = jax.random.split(key, 2 + 2 * len(cfg.limits))
        agent = init_agent(k_init, cfg.hidden, cfg.critic_hidden, cfg.init_log_std)
        norm = init_normalizer()
        if init is not None:
            layers, norm = actor_from_planner(init, cfg.hidden)
            agent = agent._replace(actor=layers)
            log(f"actor and normaliser start from {init}")
        opt = Adam(
            m=jax.tree_util.tree_map(jnp.zeros_like, agent),
            v=jax.tree_util.tree_map(jnp.zeros_like, agent),
            t=jnp.zeros((), jnp.float32),
        )
        groups = tuple(
            Group(
                batched_reset(jax.random.split(k_s, cfg.envs), params, rules_for_limit(limit)),
                new_episodes(k_e, cfg.envs),
            )
            for limit, k_s, k_e in zip(cfg.limits, k_groups[0::2], k_groups[1::2], strict=True)
        )
        start_it, best = 1, -np.inf
        league, scores, snapshots_taken = init_league(agent.actor, league_slots(cfg)), None, 0
        (run_dir / "config.json").write_text(json.dumps(dataclasses.asdict(cfg), indent=2))
        evaluator = make_evaluator(params)
        if init is not None:
            # Score the warm start itself, so training only keeps a planner that beats it.
            result = evaluate(evaluator, agent, norm, cfg, league)
            _append_jsonl(metrics, {"iter": 0, "eval": result})
            _log_eval(result, best, log)
            best, scores = result["mean_score"], result["opponents"]
            save_planner(run_dir / "planner.npz", agent, norm, name, cfg, 0)
            log(f"  saved {run_dir / 'planner.npz'} (the warm start)")
    iteration = make_iteration(params, cfg)
    start = time.monotonic()
    for it in range(start_it, cfg.iters + 1):
        frac_left = 1.0 - (it - 1) / cfg.iters if cfg.anneal_lr else 1.0
        lr = jnp.float32(cfg.lr * frac_left)
        policy_scale = jnp.float32(0.0 if it <= cfg.critic_warmup else 1.0)
        probs = jnp.asarray(opponent_probs(cfg, scores, np.asarray(league.filled)), jnp.float32)
        agent, opt, norm, groups, key, loss, stats = iteration(
            agent, opt, norm, groups, key, lr, policy_scale, league, probs
        )
        record: dict[str, Any] = {"iter": it, "seconds": round(time.monotonic() - start, 1)}
        record["loss"] = round(float(loss), 5)
        record["lr"] = float(lr)
        parts = []
        for limit, s in zip(cfg.limits, stats, strict=True):
            games = max(float(s["games"]), 1.0)
            row = {k: round(float(s[k]) / games, 4) for k in ("tag", "fall", "timeout")}
            row["games"] = int(s["games"])
            row["learner_score"] = round(
                float(s["learner_score_sum"]) / max(float(s["learner_seats"]), 1.0), 4
            )
            record[str(limit)] = row
            parts.append(
                f"{limit}s: games {row['games']:5d} score {row['learner_score']:.3f} "
                f"tag {row['tag']:.2f} fall {row['fall']:.2f} timeout {row['timeout']:.2f}"
            )
        _append_jsonl(metrics, record)
        log(
            f"iter {it:5d} {record['seconds']:8.1f}s loss {record['loss']:+.4f}  "
            + " | ".join(parts)
        )
        if cfg.league_slots > 0 and it % cfg.snapshot_every == 0:
            slot = snapshot_slot(np.asarray(league.filled), scores, snapshots_taken)
            league = add_snapshot(league, slot, agent, norm)
            snapshots_taken += 1
            log(f"  snapshot {snapshots_taken} of the actor joins the league in slot {slot}")
        if it % cfg.eval_every == 0 or it == cfg.iters:
            result = evaluate(evaluator, agent, norm, cfg, league)
            scores = result["opponents"]
            probs_next = opponent_probs(cfg, scores, np.asarray(league.filled))
            result["opponent_probs"] = [round(float(p), 4) for p in probs_next]
            _append_jsonl(metrics, {"iter": it, "eval": result})
            _log_eval(result, best, log)
            if result["mean_score"] > best:
                best = result["mean_score"]
                save_planner(run_dir / "planner.npz", agent, norm, name, cfg, it)
                log(f"  saved {run_dir / 'planner.npz'}")
        if it % cfg.checkpoint_every == 0 or it == cfg.iters:
            save_checkpoint(
                ckpt_path,
                {
                    "iteration": it,
                    "best": best,
                    "config": dataclasses.asdict(cfg),
                    "agent": agent,
                    "opt": opt,
                    "norm": norm,
                    "groups": [tuple(g) for g in groups],
                    "key": key,
                    "league": tuple(league),
                    "scores": scores,
                    "snapshots_taken": snapshots_taken,
                },
            )
    return agent, norm


def snapshot_slot(filled: np.ndarray, scores: dict[str, float] | None, taken: int) -> int:
    """Where the next snapshot goes: an empty slot, or else the weakest snapshot's.

    The weakest snapshot is the one the planner scores best against, so it is relegated, as the
    bottom team of a league would be. Without scores to rank them, the slots are reused in turn.
    """
    empty = np.nonzero(~filled)[0]
    if len(empty):
        return int(empty[0])
    if scores:
        ranked = [scores.get(f"snap{s}", -np.inf) for s in range(len(filled))]
        return int(np.argmax(ranked))
    return taken % len(filled)


def _log_eval(result: dict[str, Any], best: float, log) -> None:
    for limit, row in result["matrix"].items():
        for label, r in row.items():
            log(
                f"  eval {limit:>3}s vs {label:<9} score {r['score']:.3f} "
                f"(chaser {r['score_chaser']:.3f}, runner {r['score_runner']:.3f})  "
                f"tag {r['tag']:.2f} fall {r['fall']:.2f} timeout {r['timeout']:.2f}"
            )
    log(f"  ranked score: {result['mean_score']:.4f} (best {best:.4f})")


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    parser.add_argument("--run-dir", type=Path, required=True, help="folder for the run's files")
    parser.add_argument("--name", help="display name in the planner file (default: the folder)")
    parser.add_argument("--limits", default="20", help="round lengths in seconds, e.g. 20,60,180")
    parser.add_argument("--iters", type=int, default=Config.iters)
    parser.add_argument("--envs", type=int, default=Config.envs, help="games per round length")
    parser.add_argument("--rollout", type=int, default=Config.rollout)
    parser.add_argument("--hidden", default="64,64", help="actor hidden sizes, comma separated")
    parser.add_argument("--seed", type=int, default=Config.seed)
    parser.add_argument("--eval-every", type=int, default=Config.eval_every)
    parser.add_argument("--eval-games", type=int, default=Config.eval_games)
    parser.add_argument("--checkpoint-every", type=int, default=Config.checkpoint_every)
    parser.add_argument("--resume", action="store_true", help="continue from checkpoint.pkl")
    parser.add_argument("--init", type=Path, help="planner file to start the actor from")
    parser.add_argument(
        "--critic-warmup",
        type=int,
        default=Config.critic_warmup,
        help="rounds that train only the critic (use with --init)",
    )
    parser.add_argument("--init-log-std", type=float, default=Config.init_log_std)
    args = parser.parse_args(argv)
    cfg = Config(
        limits=tuple(int(x) for x in args.limits.split(",") if x),
        envs=args.envs,
        iters=args.iters,
        rollout=args.rollout,
        hidden=tuple(int(x) for x in args.hidden.split(",") if x),
        seed=args.seed,
        eval_every=args.eval_every,
        eval_games=args.eval_games,
        checkpoint_every=args.checkpoint_every,
        critic_warmup=args.critic_warmup,
        init_log_std=args.init_log_std,
    )
    name = args.name or args.run_dir.name
    print(
        f"training {name}: body K, limits {cfg.limits}, {cfg.envs} games per length, "
        f"{cfg.iters} rounds, seed {cfg.seed}",
        flush=True,
    )
    train(
        cfg,
        args.run_dir,
        name,
        resume=args.resume,
        init=args.init,
        log=lambda s: print(s, flush=True),
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
