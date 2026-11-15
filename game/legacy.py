from __future__ import annotations

import functools
import hashlib
import math
from dataclasses import dataclass, field
from typing import Iterable

import numpy as np

from .surfer import Inputs
from .terrain import (DT, INVULN_TIME, JUMP_DISTANCE, JUMP_GRAVITY,
                      JUMP_TIME, JUMP_VELOCITY, LANE_CHANGE_DISTANCE,
                      LANE_WIDTH, LEVELS, PHI_WEYL, PI, SLIDE_TIME, TAU,
                      clamp01, fract, sine_hash)

@dataclass(frozen=True)
class TrackHarmonic:
    amplitude: float
    frequency: float
    phase: float


class SineTrack:
    def __init__(self, length: float, level: int, samples: int = 8193):
        self.length = length
        scale = 1.0 + 0.16 * level
        self.harmonics = (
            TrackHarmonic(0.20 * scale, TAU / (length * 0.43), 0.4 * level),
            TrackHarmonic(0.09 * scale, TAU / (length * 0.19), 1.1 + level),
            TrackHarmonic(0.04 * scale, TAU / (length * 0.071), 2.3),
        )
        self.s_grid = np.linspace(0.0, length, samples, dtype=np.float64)
        theta = np.zeros(samples, dtype=np.float64)
        for h in self.harmonics:
            theta += h.amplitude * np.sin(h.frequency * self.s_grid + h.phase)
        tx, ty = np.cos(theta), np.sin(theta)
        ds = length / (samples - 1)
        self.x_grid = np.zeros(samples, dtype=np.float64)
        self.y_grid = np.zeros(samples, dtype=np.float64)
        self.x_grid[1:] = np.cumsum((tx[1:] + tx[:-1]) * (0.5 * ds))
        self.y_grid[1:] = np.cumsum((ty[1:] + ty[:-1]) * (0.5 * ds))

    def theta(self, s: float | np.ndarray) -> float | np.ndarray:
        value = np.zeros_like(s, dtype=np.float64) if isinstance(s, np.ndarray) else 0.0
        for h in self.harmonics:
            value = value + h.amplitude * np.sin(h.frequency * s + h.phase)
        return value

    def curvature(self, s: float | np.ndarray) -> float | np.ndarray:
        value = np.zeros_like(s, dtype=np.float64) if isinstance(s, np.ndarray) else 0.0
        for h in self.harmonics:
            value = value + h.amplitude * h.frequency * np.cos(h.frequency * s + h.phase)
        return value

    def frame(self, s: float) -> tuple[tuple[float, float], tuple[float, float]]:
        a = float(self.theta(s))
        return (math.cos(a), math.sin(a)), (-math.sin(a), math.cos(a))

    def position(self, s: float) -> tuple[float, float]:
        if s < 0.0:
            s = 0.0
        if s > self.length:
            s = self.length
        return float(np.interp(s, self.s_grid, self.x_grid)), float(np.interp(s, self.s_grid, self.y_grid))


@functools.lru_cache(maxsize=12)
def cached_track(length: float, level: int, samples: int = 8193) -> SineTrack:
    """Reuse immutable per-level track data instead of rebuilding its grids."""
    return SineTrack(length, level, samples)


def level_length(index: int) -> float:
    cfg = LEVELS[index]
    # Integral of the quarter-sine speed ramp, evaluated numerically to keep one source of truth.
    total = 0.0
    steps = int(cfg["duration"] * 60.0)
    for i in range(steps):
        p = i / steps
        total += (cfg["v0"] + (cfg["v1"] - cfg["v0"]) * math.sin(0.5 * PI * p)) * DT
    return total


# ============================== INTEGRATORS ==================================
def oscillator_accel(x: float, v: float, w0: float, zeta: float) -> float:
    return -2.0 * zeta * w0 * v - w0 * w0 * x


def oscillator_step(kind: str, x: float, v: float, dt: float, w0: float = 4.0, zeta: float = 0.08) -> tuple[float, float]:
    if kind == "euler":
        return x + dt * v, v + dt * oscillator_accel(x, v, w0, zeta)
    if kind == "symplectic":
        v = v + dt * oscillator_accel(x, v, w0, zeta)
        return x + dt * v, v
    if kind == "verlet":
        a0 = oscillator_accel(x, v, w0, zeta)
        xn = x + dt * v + 0.5 * dt * dt * a0
        vm = v + 0.5 * dt * a0
        a1 = oscillator_accel(xn, vm, w0, zeta)
        return xn, vm + 0.5 * dt * a1
    # First-order system RK4.
    k1x, k1v = v, oscillator_accel(x, v, w0, zeta)
    k2x, k2v = v + 0.5 * dt * k1v, oscillator_accel(x + 0.5 * dt * k1x, v + 0.5 * dt * k1v, w0, zeta)
    k3x, k3v = v + 0.5 * dt * k2v, oscillator_accel(x + 0.5 * dt * k2x, v + 0.5 * dt * k2v, w0, zeta)
    k4x, k4v = v + dt * k3v, oscillator_accel(x + dt * k3x, v + dt * k3v, w0, zeta)
    return x + dt * (k1x + 2.0 * k2x + 2.0 * k3x + k4x) / 6.0, v + dt * (k1v + 2.0 * k2v + 2.0 * k3v + k4v) / 6.0


@dataclass
class Spring:
    x: float = 0.0
    v: float = 0.0
    target: float = 0.0

    def update(self, kind: str, dt: float, w0: float = 4.0, zeta: float = 0.08) -> None:
        q, self.v = oscillator_step(kind, self.x - self.target, self.v, dt, w0, zeta)
        self.x = q + self.target


# ============================== LEVEL / DP ===================================
@dataclass(frozen=True)
class Thing:
    z: int
    lane: int
    kind: str
    value: int = 1
    phase: float = 0.0


@dataclass
class LevelData:
    index: int
    seed: int
    length: float
    slice_m: float
    things: list[Thing]
    optimal_coins: int = 0
    optimal_path: list[tuple[int, str]] = field(default_factory=list)


def _chunk_things(level: int, seed: int, start: int, count: int, retry: int) -> list[Thing]:
    out: list[Thing] = []
    difficulty = LEVELS[level]["difficulty"]
    for j in range(count):
        z = start + j
        key = seed + retry * 997 + z * 17
        density = 0.12 + 0.025 * difficulty
        if sine_hash(key, 1.0) < density:
            lane = int(math.floor(sine_hash(key, 2.0) * 3.0)) - 1
            r = sine_hash(key, 3.0)
            kind = "low" if r < 0.32 else ("high" if r < 0.59 else ("block" if r < 0.82 else "moving"))
            out.append(Thing(z, lane, kind, phase=TAU * sine_hash(key, 4.0)))
        # Weyl-distributed lane coins, with occasional sine arc heights encoded by value.
        if sine_hash(key, 5.0) < 0.48:
            wl = int(math.floor(fract(z * PHI_WEYL + sine_hash(seed, 6.0)) * 3.0)) - 1
            out.append(Thing(z, wl, "coin", 1 + int(math.floor(2.0 * math.sin(PI * fract(z * PHI_WEYL)) ** 2)),
                             TAU * fract(z * PHI_WEYL)))
        if j > 2 and sine_hash(key, 7.0) < 0.006 + 0.002 * difficulty:
            out.append(Thing(z, int(math.floor(sine_hash(key, 8.0) * 3.0)) - 1, "magnet"))
        if j > 2 and sine_hash(key, 9.0) < 0.005:
            out.append(Thing(z, int(math.floor(sine_hash(key, 10.0) * 3.0)) - 1, "shield"))
    return out


def obstacle_blocks(kind: str, vertical: str) -> bool:
    if kind == "low":
        return vertical != "jump"
    if kind == "high":
        return vertical != "slide"
    if kind in ("block", "moving", "wall"):
        return True
    return False


def moving_lane_position(thing: Thing, at_time: float) -> float:
    if thing.kind == "moving":
        position = float(thing.lane) + 0.32 * math.sin(thing.phase + 1.35 * at_time)
        if position < -1.0:
            position = -1.0
        if position > 1.0:
            position = 1.0
        return position
    return float(thing.lane)


def solve_dp(things: Iterable[Thing], slices: int) -> tuple[bool, int, list[tuple[int, str]]]:
    by_z: list[list[Thing]] = [[] for _ in range(slices + 1)]
    for thing in things:
        if 0 <= thing.z <= slices:
            by_z[thing.z].append(thing)
    # state=(lane, action, remaining); value=(coins,path). Jump lasts 3 slices, slide 2.
    states: dict[tuple[int, str, int], tuple[int, list[tuple[int, str]]]] = {(0, "run", 0): (0, [])}
    for z in range(slices + 1):
        nxt: dict[tuple[int, str, int], tuple[int, list[tuple[int, str]]]] = {}
        for (lane, action, remain), (coins, path) in states.items():
            options: list[tuple[int, str, int]] = []
            if remain > 0:
                options.append((lane, action, remain - 1))
            else:
                for dl in (-1, 0, 1):
                    nl = lane + dl
                    if -1 <= nl <= 1:
                        options.append((nl, "run", 0))
                options.append((lane, "jump", 2))
                options.append((lane, "slide", 1))
            for nl, na, nr in options:
                blocked = False
                gain = 0
                for thing in by_z[z]:
                    effective_lane = thing.lane
                    if effective_lane == nl and obstacle_blocks(thing.kind, na):
                        blocked = True
                    if effective_lane == nl and thing.kind == "coin":
                        gain += thing.value
                if not blocked:
                    key = (nl, na, nr)
                    score = coins + gain
                    old = nxt.get(key)
                    if old is None or score > old[0]:
                        nxt[key] = (score, path + [(nl, na)])
        states = nxt
        if not states:
            return False, 0, []
    best_coins = -1
    best_path: list[tuple[int, str]] = []
    for coins, path in states.values():
        if coins > best_coins:
            best_coins, best_path = coins, path
    return True, best_coins, best_path


def generate_level(index: int, seed: int | None = None) -> LevelData:
    if seed is None:
        seed = int(LEVELS[index]["seed"])
    length = level_length(index)
    slices = int(length / 4.0)
    things: list[Thing] = []
    chunk = 20
    for start in range(3, slices, chunk):
        count = chunk
        if start + count > slices:
            count = slices - start
        retry = 0
        while True:
            candidate = things + _chunk_things(index, seed, start, count, retry)
            ok, _, _ = solve_dp(candidate, start + count)
            if ok:
                things = candidate
                break
            retry += 1
    ok, optimum, path = solve_dp(things, slices)
    if not ok:
        raise RuntimeError("deterministic generator produced an invalid level")
    return LevelData(index, seed, length, 4.0, things, optimum, path)


# ============================== GAME STATE ===================================
@dataclass
class GameState:
    level: LevelData
    time: float = 0.0
    distance: float = 0.0
    lane: int = 0
    lane_from: float = 0.0
    lane_phase: float = 1.0
    height: float = 0.0
    vertical_velocity: float = 0.0
    landing_compression: float = 0.0
    action: str = "run"
    action_time: float = 0.0
    action_duration: float = JUMP_TIME
    hearts: int = 3
    coins: int = 0
    combo: int = 0
    score: int = 0
    crashes: int = 0
    invulnerable: float = 0.0
    magnet: float = 0.0
    shield: bool = False
    finished: bool = False
    failed: bool = False
    integrator: str = "verlet"
    camera: Spring = field(default_factory=Spring)
    seen: set[int] = field(default_factory=set)
    events: list[str] = field(default_factory=list)

    @property
    def lane_x(self) -> float:
        p = clamp01(self.lane_phase)
        target = self.lane * LANE_WIDTH
        # A raised-cosine trajectory has zero lateral velocity at both ends.
        blend = 0.5 - 0.5 * math.cos(PI * p)
        return self.lane_from + (target - self.lane_from) * blend

    @property
    def lane_velocity(self) -> float:
        if self.lane_phase >= 1.0:
            return 0.0
        target = self.lane * LANE_WIDTH
        phase_rate = speed_at(self) / LANE_CHANGE_DISTANCE
        return (target - self.lane_from) * 0.5 * PI * math.sin(PI * clamp01(self.lane_phase)) * phase_rate


def speed_at(state: GameState) -> float:
    cfg = LEVELS[state.level.index]
    p = clamp01(state.time / float(cfg["duration"]))
    return float(cfg["v0"]) + (float(cfg["v1"]) - float(cfg["v0"])) * math.sin(0.5 * PI * p)


def final_score(state: GameState) -> int:
    distance_points = int(state.distance)
    if not state.finished:
        return distance_points + state.score
    cfg = LEVELS[state.level.index]
    par = float(cfg["duration"]) + 8.0
    time_bonus = int((par - state.time) * 100.0) if state.time < par else 0
    return distance_points + state.score + time_bonus + state.hearts * 500


def stars_for(finished: bool, coins: int, optimum: int, crashes: int) -> int:
    if not finished:
        return 0
    stars = 1
    if optimum == 0 or coins * 100 >= optimum * 70:
        stars = 2
    if (optimum == 0 or coins * 100 >= optimum * 95) and crashes == 0:
        stars = 3
    return stars


def update_game(state: GameState, inputs: Inputs, dt: float = DT) -> None:
    if state.finished or state.failed:
        return
    state.events.clear()
    was_jumping = state.action == "jump"
    if inputs.left and state.lane > -1 and state.action != "slide":
        state.lane_from = state.lane_x
        state.lane -= 1
        state.lane_phase = 0.0
    if inputs.right and state.lane < 1 and state.action != "slide":
        state.lane_from = state.lane_x
        state.lane += 1
        state.lane_phase = 0.0
    if inputs.jump and state.action == "run":
        state.action, state.action_time = "jump", 0.0
        state.action_duration = JUMP_TIME
        state.vertical_velocity = JUMP_VELOCITY
        state.events.append("jump")
    elif inputs.slide and state.action == "run":
        state.action, state.action_time = "slide", 0.0
        state.events.append("slide")
    state.lane_phase += speed_at(state) * dt / LANE_CHANGE_DISTANCE
    if state.lane_phase > 1.0:
        state.lane_phase = 1.0
    state.action_time += dt
    state.landing_compression -= 5.5 * dt
    if state.landing_compression < 0.0:
        state.landing_compression = 0.0
    if state.action == "jump":
        # Constant-acceleration flight integrated analytically over each fixed step.
        state.height += state.vertical_velocity * dt + 0.5 * JUMP_GRAVITY * dt * dt
        state.vertical_velocity += JUMP_GRAVITY * dt
        if state.height <= 0.0 and state.vertical_velocity < 0.0:
            state.action, state.action_time, state.height = "run", 0.0, 0.0
            state.vertical_velocity = 0.0
            state.landing_compression = 1.0
            state.events.append("land")
    elif state.action == "slide" and state.action_time >= SLIDE_TIME:
        state.action, state.action_time = "run", 0.0
    old_distance = state.distance
    state.time += dt
    state.distance += speed_at(state) * dt
    state.invulnerable -= dt
    state.magnet -= dt
    curvature = float(cached_track(state.level.length, state.level.index, 65).curvature(state.distance))
    state.camera.target = 11.0 * curvature + 0.018 * state.lane_velocity
    state.camera.update(state.integrator, dt, 7.0, 0.9)
    # Continuous world-space contacts: visuals and collisions share the same metre positions.
    for idx, thing in enumerate(state.level.things):
        if idx in state.seen:
            continue
        world_z = thing.z * state.level.slice_m
        if state.distance > world_z + 1.25:
            state.seen.add(idx)
            continue
        crossed = old_distance <= world_z + 0.42 and state.distance >= world_z - 0.42
        if not crossed:
            continue
        lane_position = moving_lane_position(thing, state.time)
        lateral = state.lane_x - lane_position * LANE_WIDTH
        close = -0.58 < lateral < 0.58
        if thing.kind == "coin" and (close or state.magnet > 0.0):
            state.seen.add(idx)
            state.coins += thing.value
            state.combo += 1
            multiplier = 1 + state.combo // 10
            state.score += 10 * thing.value * multiplier
            state.events.append("coin")
        elif thing.kind == "magnet" and close:
            state.seen.add(idx)
            state.magnet = 8.0
            state.events.append("magnet")
        elif thing.kind == "shield" and close:
            state.seen.add(idx)
            state.shield = True
            state.events.append("shield")
        low_hit = thing.kind == "low" and state.action != "jump" and not was_jumping
        high_hit = thing.kind == "high" and state.action != "slide"
        solid_hit = thing.kind in ("block", "moving", "wall")
        if close and (low_hit or high_hit or solid_hit) and state.invulnerable <= 0.0:
            state.seen.add(idx)
            if state.shield:
                state.shield = False
                state.events.append("shield_hit")
            else:
                state.hearts -= 1
                state.crashes += 1
                state.combo = 0
                state.invulnerable = INVULN_TIME
                state.events.append("crash")
                if state.hearts <= 0:
                    state.failed = True
    if state.distance >= state.level.length:
        state.distance = state.level.length
        state.finished = True
        state.events.append("finish")


def logic_hash(state: GameState) -> str:
    payload = (state.level.index, state.level.seed, round(state.time, 9), round(state.distance, 9), state.lane,
               round(state.height, 9), state.action, state.hearts, state.coins, state.combo, state.score,
               state.crashes, state.finished, state.failed, tuple(sorted(state.seen)))
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


def scripted_input(frame: int, state: GameState, script: dict[str, list[int]] | None = None) -> Inputs:
    if script:
        return Inputs(frame in script.get("left", []), frame in script.get("right", []),
                      frame in script.get("jump", []), frame in script.get("slide", []))
    # Deterministic DP ghost bot chooses the precomputed state at each distance slice.
    z = int(state.distance / state.level.slice_m)
    fraction = state.distance / state.level.slice_m - z
    path_index = z if fraction < 0.18 else z + 1
    if path_index < len(state.level.optimal_path):
        lane, action = state.level.optimal_path[path_index]
        return Inputs(left=lane < state.lane, right=lane > state.lane,
                      jump=action == "jump" and state.action == "run",
                      slide=action == "slide" and state.action == "run")
    return Inputs()



