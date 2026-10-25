#!/usr/bin/env python3
"""WAVE SURFER -- deterministic sine-surface riding and procedural graphics."""

# ============================== IMPORTS ======================================
from __future__ import annotations

import argparse
import ast
import functools
import hashlib
import json
import math
import os
import struct
import sys
import time
import wave
from dataclasses import dataclass, field
from pathlib import Path
from typing import Callable, Iterable

import numpy as np

PI = math.pi
TAU = 2.0 * PI
DT = 1.0 / 60.0
LANE_WIDTH = 1.35
LANE_CHANGE_DISTANCE = 2.8
JUMP_TIME = 0.82
JUMP_DISTANCE = 12.8
SLIDE_TIME = 0.72
JUMP_GRAVITY = -17.0
JUMP_VELOCITY = -0.5 * JUMP_GRAVITY * JUMP_TIME
INVULN_TIME = 1.5
PHI_WEYL = 0.6180339887498949
LEVELS = (
    {"name": "WAVE SURFER", "seed": 42, "duration": 75.0, "v0": 13.0, "v1": 20.0, "difficulty": 1},
)
BANNED = frozenset(("abs", "sqrt", "pow", "exp", "log", "tan", "atan", "atan2", "asin", "acos", "smoothstep", "min", "max", "random", "noise"))


# ============================== SINE RULE ====================================
def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def fract(x: float) -> float:
    return x - math.floor(x)


def sine_hash(n: float, k: float) -> float:
    """The only pseudo-random source used by gameplay."""
    return fract(43758.5453 * math.sin(12.9898 * n + 78.233 * k))


def sine_lint_source(source: str) -> list[str]:
    """AST lint Python and token-lint embedded GLSL, excluding report code."""
    begin_tag = "# --- " + "REPORT (lint-exempt) ---"
    end_tag = "# --- END " + "REPORT (lint-exempt) ---"
    begin = source.find(begin_tag)
    end = source.find(end_tag)
    if begin >= 0 and end > begin:
        source = source[:begin] + "\npass\n" + source[end + len(end_tag):]
    errors: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            names: list[str] = []
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                cursor: ast.AST = node.func
                while isinstance(cursor, ast.Attribute):
                    names.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    names.append(cursor.id)
            for name in names:
                if name in BANNED:
                    errors.append(f"line {node.lineno}: banned call {name}()")
                    break
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("SHADER" in t for t in targets):
                glsl = node.value.value
                word = ""
                words: list[str] = []
                for char in glsl:
                    if char.isalnum() or char == "_":
                        word += char
                    elif word:
                        words.append(word)
                        word = ""
                if word:
                    words.append(word)
                for banned in BANNED:
                    if banned in words:
                        errors.append(f"GLSL: banned token {banned}")
    return errors


def lint_game(path: str | Path = __file__) -> list[str]:
    return sine_lint_source(Path(path).read_text(encoding="utf8"))


# ============================== TRACK MATH ===================================
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
class Inputs:
    left: bool = False
    right: bool = False
    jump: bool = False
    slide: bool = False


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


# ============================== WAVE SURFER =================================
SURF_GRAVITY = -18.0
SURF_JUMP_SPEED = 8.4
BOARD_CLEARANCE = 0.34


@dataclass(frozen=True)
class SurfThing:
    distance: float
    kind: str
    phase: float = 0.0


@dataclass
class SurfLevel:
    index: int
    seed: int
    length: float
    things: list[SurfThing]
    optimal_coins: int


def wave_geometry(s: float, level: int, length: float) -> tuple[float, float, float]:
    """Composite chirped sine height, analytic slope, and analytic curvature."""
    p = clamp01(s / length)
    amplitude0 = 1.0
    amplitude_gain = 1.45
    amplitude = amplitude0 + amplitude_gain * p
    amplitude_rate = amplitude_gain / length
    k0 = 0.10
    k1 = 0.21
    chirp = (k1 - k0) / length
    phase = k0 * s + 0.5 * chirp * s * s
    primary = math.sin(phase)
    secondary = math.sin(2.3 * phase + 1.1)
    detail = math.sin(4.7 * phase - 0.6)
    shape = primary + 0.42 * secondary + 0.18 * detail
    shape_rate = math.cos(phase) + 0.966 * math.cos(2.3 * phase + 1.1) + 0.846 * math.cos(4.7 * phase - 0.6)
    shape_curve = -primary - 2.2218 * secondary - 3.9762 * detail
    phase_rate = k0 + chirp * s
    height = amplitude * shape
    slope = amplitude_rate * shape + amplitude * shape_rate * phase_rate
    curvature = 2.0 * amplitude_rate * shape_rate * phase_rate
    curvature += amplitude * (shape_curve * phase_rate * phase_rate + shape_rate * chirp)
    return height, slope, curvature


def generate_surf_level(index: int, seed: int | None = None) -> SurfLevel:
    if index != 0:
        raise ValueError("WAVE SURFER has one definitive track")
    if seed is None:
        seed = int(LEVELS[index]["seed"])
    length = level_length(index)
    things: list[SurfThing] = []
    coins = 0
    spacing = 7.5
    count = int(length / spacing)
    for i in range(5, count):
        distance = i * spacing
        roll = sine_hash(seed + i * 13, 21.0)
        phase = TAU * sine_hash(seed + i * 17, 22.0)
        coin_cut = 0.56
        boost_cut = 0.65
        spring_cut = 0.75
        breaker_cut = 0.84
        if roll < coin_cut:
            things.append(SurfThing(distance, "coin", phase))
            coins += 1
        elif roll < boost_cut:
            things.append(SurfThing(distance, "boost", phase))
        elif roll < spring_cut:
            things.append(SurfThing(distance, "spring", phase))
        elif roll < breaker_cut:
            things.append(SurfThing(distance, "breaker", phase))
    return SurfLevel(index, seed, length, things, coins)


@dataclass
class SurferState:
    level: SurfLevel
    time: float = 0.0
    distance: float = 0.0
    speed: float = 10.0
    y: float = 0.0
    vertical_velocity: float = 0.0
    grounded: bool = True
    pitch: float = 0.0
    camera_y: float = 0.0
    camera_velocity: float = 0.0
    fuel: float = 100.0
    boosting: bool = False
    steep_time: float = 0.0
    hearts: int = 3
    coins: int = 0
    combo: int = 0
    score: int = 0
    crashes: int = 0
    invulnerable: float = 0.0
    finished: bool = False
    failed: bool = False
    action: str = "surf"
    events: list[str] = field(default_factory=list)
    seen: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        ground, slope, _ = wave_geometry(self.distance, self.level.index, self.level.length)
        self.y = ground + BOARD_CLEARANCE
        self.camera_y = ground
        self.pitch = slope / (1.0 + 0.32 * slope * slope)


def surfer_wipeout(state: SurferState) -> None:
    if state.invulnerable > 0.0:
        return
    state.hearts -= 1
    state.crashes += 1
    state.combo = 0
    state.invulnerable = 1.35
    state.speed *= 0.58
    state.vertical_velocity = 4.2
    state.grounded = False
    state.action = "wipeout"
    state.events.append("crash")
    if state.hearts <= 0:
        state.failed = True


def surfer_stumble(state: SurferState) -> None:
    """Recoverable terrain mistake: never removes a heart."""
    state.speed *= 0.76
    if state.speed < 6.0:
        state.speed = 6.0
    state.events.append("slide")


def update_surfer(state: SurferState, inputs: Inputs, dt: float = DT) -> None:
    if state.finished or state.failed:
        return
    state.events.clear()
    state.time += dt
    state.invulnerable -= dt
    progress = clamp01(state.distance / state.level.length)
    cruise = 10.5 + 8.0 * math.sin(0.5 * PI * progress)
    state.boosting = (inputs.right or inputs.slide) and state.fuel > 0.0
    if state.boosting:
        state.speed += 10.0 * dt
        state.fuel -= 24.0 * dt
        if state.speed > cruise + 7.0:
            state.speed = cruise + 7.0
    else:
        state.speed += (cruise - state.speed) * 1.8 * dt
        state.fuel += 3.5 * dt
    if inputs.left:
        state.speed -= 8.0 * dt
    if state.speed < 6.0:
        state.speed = 6.0
    if state.fuel < 0.0:
        state.fuel = 0.0
    if state.fuel > 100.0:
        state.fuel = 100.0

    old_distance = state.distance
    old_ground, old_slope, _ = wave_geometry(old_distance, state.level.index, state.level.length)
    if inputs.jump and state.grounded:
        state.grounded = False
        state.vertical_velocity = old_slope * state.speed + SURF_JUMP_SPEED
        state.action = "air"
        state.events.append("jump")

    state.distance += state.speed * dt
    ground, slope, curvature = wave_geometry(state.distance, state.level.index, state.level.length)
    if state.grounded:
        state.y = ground + BOARD_CLEARANCE
        state.vertical_velocity = slope * state.speed
        state.action = "boost" if state.boosting else "surf"
        if slope * slope > 1.08 and not state.boosting:
            state.steep_time += dt
            if state.steep_time > 0.24:
                surfer_stumble(state)
                state.steep_time = -0.45
        else:
            state.steep_time -= 2.0 * dt
            if state.steep_time < 0.0:
                state.steep_time = 0.0
    else:
        state.y += state.vertical_velocity * dt + 0.5 * SURF_GRAVITY * dt * dt
        state.vertical_velocity += SURF_GRAVITY * dt
        if state.y <= ground + BOARD_CLEARANCE and state.vertical_velocity < slope * state.speed:
            relative = state.vertical_velocity - slope * state.speed
            impact2 = relative * relative / (1.0 + slope * slope)
            state.y = ground + BOARD_CLEARANCE
            if impact2 > 100.0 and state.action != "wipeout":
                surfer_stumble(state)
                state.grounded = True
                state.vertical_velocity = slope * state.speed
                state.action = "surf"
            else:
                state.grounded = True
                state.vertical_velocity = slope * state.speed
                state.action = "surf"
                state.events.append("land")

    state.pitch = slope / (1.0 + 0.32 * slope * slope)
    for idx, thing in enumerate(state.level.things):
        if idx in state.seen or thing.distance < old_distance - 1.0:
            continue
        if thing.distance > state.distance + 0.65:
            break
        if old_distance <= thing.distance + 0.45 and state.distance >= thing.distance - 0.45:
            object_ground, _, _ = wave_geometry(thing.distance, state.level.index, state.level.length)
            altitude = state.y - object_ground - BOARD_CLEARANCE
            if thing.kind == "coin" and altitude < 2.4:
                state.seen.add(idx)
                state.coins += 1
                state.combo += 1
                state.score += 25 * (1 + state.combo // 8)
                state.events.append("coin")
            elif thing.kind == "boost":
                state.seen.add(idx)
                state.fuel += 32.0
                if state.fuel > 100.0:
                    state.fuel = 100.0
                state.events.append("magnet")
            elif thing.kind == "spring" and state.grounded:
                state.seen.add(idx)
                state.grounded = False
                state.vertical_velocity = slope * state.speed + 13.5
                state.action = "air"
                state.events.append("spring")
            elif thing.kind == "breaker" and altitude < 0.25:
                state.seen.add(idx)
                surfer_wipeout(state)

    camera_target = ground + 0.28 * (state.y - ground - BOARD_CLEARANCE)
    camera_accel = 32.0 * (camera_target - state.camera_y) - 10.0 * state.camera_velocity
    state.camera_velocity += camera_accel * dt
    state.camera_y += state.camera_velocity * dt
    if state.distance >= state.level.length:
        state.distance = state.level.length
        state.finished = True
        state.events.append("finish")


def surf_logic_hash(state: SurferState) -> str:
    payload = (state.level.index, state.level.seed, round(state.time, 9), round(state.distance, 9),
               round(state.y, 9), round(state.vertical_velocity, 9), round(state.speed, 9),
               state.grounded, state.hearts, state.coins, state.score, state.crashes,
               state.finished, state.failed, tuple(sorted(state.seen)))
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


def logic_hash(state: GameState) -> str:
    payload = (state.level.index, state.level.seed, round(state.time, 9), round(state.distance, 9), state.lane,
               round(state.height, 9), state.action, state.hearts, state.coins, state.combo, state.score,
               state.crashes, state.finished, state.failed, tuple(sorted(state.seen)))
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()


# ============================== PERSISTENCE ==================================
class ScoreStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf8"))
        except (OSError, ValueError):
            self.data = {}

    def record(self, level: int, score: int, run_time: float, stars: int) -> bool:
        key = str(level + 1)
        old = self.data.get(key, {})
        is_new = score > int(old.get("score", -1))
        best_time = float(old.get("time", run_time))
        if run_time < best_time:
            best_time = run_time
        self.data[key] = {"score": score if is_new else int(old.get("score", score)),
                          "time": best_time, "stars": stars if stars > int(old.get("stars", 0)) else int(old.get("stars", 0))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf8")
        return is_new


# ============================== WAVE PDE =====================================
class WaveField:
    """Leapfrog 2-D wave solver. CFL limit uses documented 0.70710678 constant."""
    def __init__(self, size: int = 128, ratio: float = 0.45, damping: float = 0.001):
        if ratio > 0.7071067811865476:
            raise ValueError("CFL condition violated")
        self.size, self.ratio, self.damping = size, ratio, damping
        self.u = np.zeros((size, size), dtype=np.float64)
        self.old = np.zeros_like(self.u)
        self.version = 0

    def step(self) -> None:
        u = self.u
        lap = np.zeros_like(u)
        lap[1:-1, 1:-1] = u[2:, 1:-1] + u[:-2, 1:-1] + u[1:-1, 2:] + u[1:-1, :-2] - 4.0 * u[1:-1, 1:-1]
        new = 2.0 * u - self.old + self.ratio * self.ratio * lap - self.damping * (u - self.old)
        new[0, :] = new[-1, :] = 0.0
        new[:, 0] = new[:, -1] = 0.0
        self.old, self.u = u, new
        self.version += 1

    def inject(self, x: int, y: int, radius: int = 9, amplitude: float = 0.8) -> None:
        for j in range(-radius, radius + 1):
            for i in range(-radius, radius + 1):
                q = (i * i + j * j) / float(radius * radius)
                if q <= 1.0:
                    xx, yy = x + i, y + j
                    if 0 < xx < self.size - 1 and 0 < yy < self.size - 1:
                        self.u[yy, xx] += amplitude * math.sin(PI * (1.0 - q)) ** 2
        self.version += 1

    def energy(self) -> float:
        velocity = self.u - self.old
        gx = self.u[:, 1:] - self.u[:, :-1]
        gy = self.u[1:, :] - self.u[:-1, :]
        return float(0.5 * (np.sum(velocity * velocity) + self.ratio * self.ratio * (np.sum(gx * gx) + np.sum(gy * gy))))


# ============================== SHADERS ======================================
VERTEX_SHADER = """#version 330
in vec2 in_pos;
out vec2 uv;
void main(){ uv=in_pos*0.5+0.5; gl_Position=vec4(in_pos,0.0,1.0); }
"""

FRAGMENT_SHADER = """#version 330
uniform vec2 resolution;
uniform float time;
uniform float progress;
uniform float travel;
uniform float lane_x;
uniform float lane_velocity;
uniform float jump_y;
uniform float jump_velocity;
uniform float slide_factor;
uniform float landing;
uniform float hurt;
uniform float shield;
uniform float bank;
uniform int object_count;
uniform int character_visible;
uniform vec4 objects[24];
uniform sampler2D wave_tex;
in vec2 uv;
out vec4 fragColor;
const float PI=3.141592653589793;
float sat(float x){if(x<0.0)return 0.0;if(x>1.0)return 1.0;return x;}
float hash(float n,float k){return fract(43758.5453*sin(12.9898*n+78.233*k));}
float pill(vec2 p,vec2 scale,float c){
 if(p.x < -scale.x*0.5 || p.x > scale.x*0.5 || p.y < -scale.y*0.5 || p.y > scale.y*0.5) return 0.0;
 return step(c,cos(PI*p.x/scale.x)*cos(PI*p.y/scale.y));
}
vec3 palette(float x){float glow=0.5+0.5*sin(x);return vec3(0.54,0.39,0.27)+vec3(0.26,0.20,0.14)*glow;}
float lineMask(float x,float width){return step(cos(PI*width),cos(PI*x));}
float band(float x,float width){if(x>-width && x<width)return 1.0;return 0.0;}
vec2 rot(vec2 p,float a){return vec2(cos(a)*p.x-sin(a)*p.y,sin(a)*p.x+cos(a)*p.y);}
float limb(vec2 p,vec2 center,vec2 size,float angle){return pill(rot(p-center,-angle),size,0.12);}
void main(){
 vec2 p=(uv-0.5)*vec2(resolution.x/resolution.y,1.0);
 p=rot(p,0.08*bank);
 float horizon=0.54;
 vec3 skyA=vec3(0.012,0.018,0.035);
 vec3 skyB=vec3(0.08,0.025,0.065);
 if(theme==1){skyA=vec3(0.014,0.010,0.035);skyB=vec3(0.075,0.025,0.11);}
 if(theme==2){skyA=vec3(0.008,0.027,0.038);skyB=vec3(0.025,0.095,0.10);}
 vec3 col=mix(skyA,skyB,sat(uv.y));
 float sunR=(p.x*p.x+(p.y-0.21)*(p.y-0.21));
 float sun=step(sunR,0.036)*(0.78+0.22*sin(105.0*p.y+2.0*sin(8.0*p.y)));
 col+=sun*vec3(1.0,0.48,0.24)*0.86;
 float g1=0.5+0.5*sin(52.0*p.x+7.0*sin(p.y*5.0));
 float g2=0.5+0.5*sin(55.0*p.x-6.0*p.y);
 col+=vec3(0.025,0.018,0.012)*g1*g2;
 for(int ri=0;ri<3;ri++){
  float fr=float(ri);
  float ridge=0.055+fr*0.052+0.045*sin((2.4+fr*0.35)*p.x+fr*1.7+time*0.025);
  if(p.y<ridge && p.y>-0.18) col=mix(col,vec3(0.075+fr*0.025,0.057+fr*0.018,0.043+fr*0.012),0.72);
  col+=band(p.y-ridge,0.0018)*vec3(0.36,0.27,0.19)*(0.45-0.1*fr);
 }
 if(theme==1){
  float q=0.0;
  q+=cos(18.0*(p.x*1.0+p.y*0.0)+time*0.2);
  q+=cos(18.0*(p.x*0.309+p.y*0.951)+time*0.2);
  q+=cos(18.0*(p.x*-0.809+p.y*0.588)+time*0.2);
  q+=cos(18.0*(p.x*-0.809+p.y*-0.588)+time*0.2);
  q+=cos(18.0*(p.x*0.309+p.y*-0.951)+time*0.2);
  col+=vec3(0.39,0.29,0.20)*step(3.6,q)*0.26;
 }
 if(theme==0){
  for(int bi=0;bi<9;bi++){
   float fb=float(bi);
   float bx=-0.86+fb*0.21;
   float bh=0.08+0.17*hash(fb,3.0);
   if(p.x>bx-0.085 && p.x<bx+0.085 && p.y>-0.04 && p.y<bh){
    float windows=step(0.72,(0.5+0.5*sin(95.0*(p.x-bx)))*(0.5+0.5*sin(82.0*p.y+fb)));
    col=mix(vec3(0.035,0.028,0.021),vec3(0.62,0.44,0.27),windows*0.62);
   }
  }
 }
 if(theme==2){
  float aur=sin(9.0*p.x+time)+sin(13.0*p.x-time*0.7)+sin(7.0*p.x+time*0.3);
  float ribbon=step(cos(PI*0.055),cos(PI*(p.y-0.18-0.035*aur)));
  col+=ribbon*vec3(0.42,0.31,0.22)*0.16;
 }
 float v=sat((horizon-uv.y)/horizon);
 float depth=sin(PI*0.5*v);
 float halfroad=0.07+0.64*depth;
 float curve=0.08*sin(progress*6.0+v*2.0)+0.025*sin(progress*15.0+v*7.0)-0.035*bank*v;
 float road=0.0;
 if(p.x-curve > -halfroad && p.x-curve < halfroad && uv.y < horizon) road=1.0;
 float ripple=texture(wave_tex,vec2(sat(0.5+(p.x-curve)/(2.0*halfroad)),v)).r;
 float ch=cos((6.0+float(theme))*PI*(p.x-curve))*cos(4.0*PI*v)-cos(4.0*PI*(p.x-curve))*cos((6.0+float(theme))*PI*v);
 float roadLight=0.075+0.018*sin(12.0*v+ripple*5.0);
 if(theme==2) roadLight=0.075+0.045*sin(ch*5.0+ripple*6.0);
 vec3 roadBase=vec3(0.025,0.032,0.050);
 if(theme==1) roadBase=vec3(0.035,0.024,0.055);
 if(theme==2) roadBase=vec3(0.018,0.047,0.052);
 col=mix(col,roadBase+palette(v*2.0+time*0.04)*roadLight,road);
 float laneCoord=(p.x-curve)/halfroad;
 float dividers=band(laneCoord-0.333,0.010+0.008*depth)+band(laneCoord+0.333,0.010+0.008*depth);
 float dash=step(0.10,sin(76.0*v-travel*1.8));
 col+=road*dividers*dash*vec3(0.92,0.66,0.36)*depth;
 float innerRoad=0.0;
 if(p.x-curve > -halfroad+0.006 && p.x-curve < halfroad-0.006 && uv.y < horizon) innerRoad=1.0;
 col+=(road-innerRoad)*vec3(0.82,0.68,0.52)*(0.35+0.45*depth);
 // Track-anchored roadside scenery: every element advances through the same perspective as obstacles.
 for(int si=0;si<14;si++){
  float fs=float(si);
  float phase=fract(fs/14.0+travel/92.0);
  float sd=sin(PI*0.5*phase);
  float side=1.0;
  if(mod(fs,2.0)<1.0) side=-1.0;
  vec2 sp=vec2(side*(0.11+0.73*sd)+0.08*sin(progress*6.0+phase*2.0),0.035-0.47*sd);
  vec2 sr=p-sp;
  float trunk=pill(sr-vec2(0.0,0.025*sd),vec2(0.012+0.018*sd,0.07+0.17*sd),0.08);
  float crown=step(sr.x*sr.x+(sr.y-0.09*sd)*(sr.y-0.09*sd),(0.016+0.052*sd)*(0.016+0.052*sd));
  float monolith=pill(sr-vec2(0.0,0.035*sd),vec2(0.025+0.05*sd,0.08+0.19*sd),0.08);
  float lamp=trunk+step(sr.x*sr.x+(sr.y-0.12*sd)*(sr.y-0.12*sd),(0.008+0.018*sd)*(0.008+0.018*sd));
  float scenery=trunk+crown;
  if(theme==0) scenery=lamp;
  if(theme==1) scenery=monolith+band(sr.x+sr.y*0.55,0.004+0.007*sd)*step(sr.y,-0.02);
  col=mix(col,vec3(0.19,0.14,0.10)+vec3(0.22,0.17,0.12)*hash(fs,float(theme)+2.0),sat(scenery));
 }
 if(progress>0.965){
  float gateScale=sin(PI*0.5*sat((progress-0.965)/0.035));
  float posts=pill(p-vec2(-0.23,0.05),vec2(0.035,0.48),0.08)+pill(p-vec2(0.23,0.05),vec2(0.035,0.48),0.08);
  float beam=pill(p-vec2(0.0,0.27),vec2(0.5,0.035),0.08);
  col=mix(col,vec3(0.92,0.79,0.63),sat(posts+beam)*gateScale);
 }
 // CPU-generated deterministic objects; y is quarter-sine projected depth.
 for(int oi=0;oi<24;oi++){
  if(oi<object_count){
   vec4 ob=objects[oi];
   float od=sat(ob.y);
   vec2 op=vec2(ob.x*(0.035+0.25*od)+0.08*sin(progress*6.0+ob.y*2.0),0.035-0.335*od);
   if(ob.z<0.5){op.x+=0.018*od*sin(2.0*ob.w);op.y+=0.055*od*sin(ob.w)*sin(ob.w);}
   vec2 rel=p-op;
   float coinOuter=step(rel.x*rel.x+rel.y*rel.y,(0.010+0.018*od)*(0.010+0.018*od));
   float coinInner=step(rel.x*rel.x+rel.y*rel.y,(0.005+0.009*od)*(0.005+0.009*od));
   float block=pill(rel-vec2(0.0,0.022*od),vec2(0.025+0.055*od,0.035+0.13*od),0.08);
   float low=pill(rel+vec2(0.0,0.045*od),vec2(0.05+0.06*od,0.018+0.035*od),0.08);
   float archX=0.020+0.034*od;
   float archTop=pill(rel-vec2(0.0,0.038*od),vec2(0.06+0.07*od,0.015+0.022*od),0.08);
   float archLeft=pill(rel-vec2(-archX,0.0),vec2(0.012+0.014*od,0.06+0.085*od),0.08);
   float archRight=pill(rel-vec2(archX,0.0),vec2(0.012+0.014*od,0.06+0.085*od),0.08);
   float high=sat(archTop+archLeft+archRight);
   float pendulum=coinOuter;
   if(rel.x>-0.003-0.004*od && rel.x<0.003+0.004*od && rel.y>0.0 && rel.y<0.07+0.10*od) pendulum=1.0;
   if(ob.z<0.5) col=mix(col,vec3(1.00,0.66,0.16),coinOuter-coinInner);
   else if(ob.z<1.5) col=mix(col,vec3(1.00,0.31,0.20),low);
   else if(ob.z<2.5) col=mix(col,vec3(0.18,0.88,0.96),high);
   else if(ob.z<3.5) col=mix(col,vec3(0.78,0.23,0.62),block);
   else if(ob.z<4.5) col=mix(col,vec3(0.96,0.39,0.26),pendulum);
   else col+=coinOuter*vec3(0.30,0.92,1.00)*(0.65+0.25*sin(time*5.0));
  }
 }
 // Implicit pill character and sine skeleton.
 vec2 playerPos=vec2(lane_x*0.105,-0.300+jump_y*0.105);
 vec2 shadowP=p-vec2(playerPos.x,-0.305);
 float shadow=step(shadowP.x*shadowP.x*95.0+shadowP.y*shadowP.y*1700.0,0.52)*(0.32-0.10*sat(jump_y));
 col=mix(col,vec3(0.003,0.005,0.009),shadow*float(character_visible));
 vec2 cp=p-playerPos;
 float gait=time*11.0;
 float airborne=step(0.02,jump_y);
 float bob=0.010*sin(2.0*gait)*(1.0-airborne);
 cp.y-=bob;
 cp=rot(cp,-0.018*lane_velocity);
 float crouch=slide_factor;
 cp.y+=0.050*crouch-0.018*landing;
 cp.x-=0.030*crouch;
 float legA=(1.0-airborne)*0.55*sin(gait)+airborne*(0.28+0.06*jump_velocity);
 float armA=(1.0-airborne)*-0.48*sin(gait)+airborne*(-0.42+0.035*jump_velocity);
 float body=pill(cp-vec2(0.0,0.082-0.025*crouch),vec2(0.048+0.025*crouch,0.108-0.035*crouch),0.05);
 float head=pill(cp-vec2(0.0+0.025*crouch,0.176-0.065*crouch),vec2(0.052,0.056),0.12);
 float hood=0.0;
 float legs=limb(cp,vec2(-0.019,0.0),vec2(0.019,0.112-0.035*crouch),legA)+limb(cp,vec2(0.019,0.0),vec2(0.019,0.112-0.035*crouch),-legA);
 float arms=limb(cp,vec2(-0.040,0.086),vec2(0.016,0.092),armA)+limb(cp,vec2(0.040,0.086),vec2(0.016,0.092),-armA);
 float scarf=0.0;
 for(int j=0;j<6;j++){float fj=float(j);vec2 sc=cp-vec2(0.030+fj*0.020,0.142+0.008*fj*sin(fj*0.9-time*8.0));scarf+=pill(sc,vec2(0.024,0.010),0.15);}
 float blink=step(0.0,sin(time*34.0));
 float character=sat(body+head+hood+legs+arms)*(1.0-hurt+hurt*(0.35+0.65*blink));
 character*=float(character_visible);
 vec3 runnerColor=vec3(0.96,0.86,0.68)+vec3(0.08,0.12,0.20)*(0.5+0.5*sin(time*2.0));
 col=mix(col,runnerColor,character);
 col=mix(col,vec3(0.12,0.86,0.98),sat(scarf)*float(character_visible));
 float ring=lineMask((cp.x*cp.x+cp.y*cp.y)*22.0+sin(time*5.0)*0.1,0.025)*shield;
 col+=ring*vec3(0.95,0.78,0.55);
 // Sine-only scanline, bloom-like colour bleed, chromatic pulse and vignette.
 col*=0.97+0.03*sin(PI*resolution.y*uv.y);
 float vig=sin(PI*uv.x)*sin(PI*uv.y);
 col*=0.5+0.5*vig*vig;
 fragColor=vec4(col,1.0);
}
"""

SURF_FRAGMENT_SHADER = """#version 330
uniform vec2 resolution;
uniform float time;
uniform float travel;
uniform float level_length;
uniform float surfer_y;
uniform float camera_y;
uniform float surfer_pitch;
uniform float surfer_vy;
uniform float speed;
uniform float fuel;
uniform float boost;
uniform float hurt;
uniform float grounded;
uniform int object_count;
uniform vec4 objects[24];
uniform sampler2D wave_tex;
in vec2 uv;
out vec4 fragColor;
const float PI=3.141592653589793;
float sat(float x){if(x<0.0)return 0.0;if(x>1.0)return 1.0;return x;}
float band(float x,float w){if(x>-w && x<w)return 1.0;return 0.0;}
vec2 rot(vec2 p,float a){return vec2(cos(a)*p.x-sin(a)*p.y,sin(a)*p.x+cos(a)*p.y);}
float pill(vec2 p,vec2 scale,float c){
 if(p.x < -scale.x*0.5 || p.x > scale.x*0.5 || p.y < -scale.y*0.5 || p.y > scale.y*0.5)return 0.0;
 return step(c,cos(PI*p.x/scale.x)*cos(PI*p.y/scale.y));
}
float waveHeight(float s){
 float p=s/level_length;
 p=sat(p);
 float a0=1.0;
 float gain=1.45;
 float k0=0.10;
 float k1=0.21;
 float amp=a0+gain*p;
 float chirp=(k1-k0)/level_length;
 float ph=k0*s+0.5*chirp*s*s;
 float shape=sin(ph)+0.42*sin(2.3*ph+1.1)+0.18*sin(4.7*ph-0.6);
 return amp*shape;
}
void main(){
 float aspect=resolution.x/resolution.y;
 float wx=travel+(uv.x-0.30)*42.0;
 float gy=0.39+(waveHeight(wx)-camera_y)*0.075;
 vec3 sky=vec3(0.94,0.90,0.82)+vec3(0.025,0.018,0.006)*sin(PI*uv.y);
 vec3 col=sky;
 vec2 sunp=uv-vec2(0.72,0.76);
 float sun=step(sunp.x*sunp.x+sunp.y*sunp.y,0.0065);
 vec3 sunColor=vec3(0.84,0.76,0.64);
 col=mix(col,sunColor,sun*0.72);
 // Parallax ridges are sums of incommensurate sine modes.
 for(int mi=0;mi<4;mi++){
  float fm=float(mi);
  float ridge=0.43+fm*0.055+0.055*sin(5.2*uv.x+travel*(0.006+fm*0.002)+fm*1.7);
  ridge+=0.022*sin(11.3*uv.x-travel*0.004+fm);
  vec3 ridgeColor=vec3(0.74+fm*0.025,0.67+fm*0.035,0.57+fm*0.045);
  if(uv.y<ridge)col=mix(col,ridgeColor,0.34);
 }
 // Fine airborne wave diagrams echo the terrain equation.
 float cloudWave=0.72+0.022*sin(32.0*uv.x+time*0.08)+0.009*sin(73.6*uv.x+1.1);
 col=mix(col,vec3(0.60,0.50,0.40),band(uv.y-cloudWave,0.0012)*0.35);
 float field=texture(wave_tex,vec2(uv.x,sat((gy-uv.y)*2.0))).r;
 vec3 earth=vec3(0.105,0.092,0.078)+vec3(0.012,0.009,0.006)*sin(20.0*uv.y+field*3.0);
 if(uv.y<gy)col=earth;
 float crest=band(uv.y-gy,0.007);
 col=mix(col,vec3(0.985,0.965,0.915),crest);
 float contour=band(uv.y-(gy-0.035),0.0015);
 col=mix(col,vec3(0.57,0.47,0.36),contour*step(uv.y,gy));
 float under=band(uv.y-(gy-0.115-0.018*sin(wx*0.33)),0.0014);
 col=mix(col,vec3(0.76,0.61,0.43),under*step(uv.y,gy));
 // Pickups and hazards are anchored to the same analytic surface.
 for(int oi=0;oi<24;oi++){
  if(oi<object_count){
   vec4 ob=objects[oi];
   vec2 d=(uv-ob.xy)*vec2(aspect,1.0);
   float r2=d.x*d.x+d.y*d.y;
   float ring=step(r2,0.00072)-step(r2,0.00030);
   float halo=step(r2,0.00115)*0.18;
   float coinWave=band(d.y-0.005*sin(125.0*d.x),0.0017)*step(r2,0.00028);
   float spring=band(d.x-0.014*sin(190.0*d.y),0.0048)*step(-0.055,d.y)*step(d.y,0.055);
   float breaker=step(r2,0.00130+0.00042*sin(120.0*d.x)*sin(120.0*d.y));
   float breakerInner=step(r2,0.00083+0.00025*sin(120.0*d.x)*sin(120.0*d.y));
   float core=step(r2,0.00042);
   if(ob.z<0.5){col=mix(col,vec3(0.96,0.66,0.18),halo);col=mix(col,vec3(0.92,0.55,0.10),ring);col=mix(col,vec3(0.24,0.16,0.08),coinWave);}
   else if(ob.z<1.5){col=mix(col,vec3(0.10,0.48,0.78),step(r2,0.00105));col=mix(col,vec3(0.96,0.94,0.78),core);}
   else if(ob.z<2.5){col=mix(col,vec3(0.12,0.82,0.57),spring);col=mix(col,vec3(0.08,0.30,0.22),band(d.y+0.057,0.005)*step(r2,0.0022));}
   else {col=mix(col,vec3(0.98,0.90,0.78),breaker);col=mix(col,vec3(0.72,0.10,0.08),breakerInner);}
  }
 }
 // Surfer: all parts are implicit cosine pills rotated into the tangent frame.
 vec2 surferPos=vec2(0.30,0.39+(surfer_y-camera_y)*0.075);
 vec2 q=(uv-surferPos)*vec2(aspect,1.0);
 float tumble=hurt*(0.45+0.35*sin(time*9.0));
 q=rot(q,-surfer_pitch-tumble);
 float board=pill(q-vec2(0.0,-0.022),vec2(0.165,0.019),0.12);
 float boardEdge=pill(q-vec2(0.0,-0.018),vec2(0.174,0.008),0.10);
 float crouch=0.5+0.5*sin(sat(speed/25.0)*PI*0.5)+0.06*sin(surfer_vy)+0.08*(1.0-grounded);
 float body=pill(rot(q-vec2(-0.004,0.061),-0.20),vec2(0.042,0.086),0.08);
 vec2 headq=q-vec2(-0.020,0.119);
 float head=step(headq.x*headq.x+headq.y*headq.y,0.00042);
 float rearLeg=pill(rot(q-vec2(-0.027,0.019),-0.58+0.08*crouch),vec2(0.020,0.074),0.10);
 float frontLeg=pill(rot(q-vec2(0.026,0.020),0.58-0.06*crouch),vec2(0.020,0.076),0.10);
 float rearArm=pill(rot(q-vec2(-0.032,0.070),-0.72),vec2(0.016,0.063),0.10);
 float frontArm=pill(rot(q-vec2(0.025,0.071),0.82),vec2(0.016,0.066),0.10);
 float scarf=pill(rot(q-vec2(-0.060,0.091+0.008*sin(time*8.0)),-1.28),vec2(0.014,0.085),0.10);
 float limbs=sat(head+rearLeg+frontLeg+rearArm+frontArm);
 col=mix(col,vec3(0.055,0.050,0.046),limbs);
 col=mix(col,vec3(0.12,0.30,0.33),body);
 col=mix(col,vec3(0.86,0.55,0.20),sat(board+boardEdge));
 col=mix(col,vec3(0.72,0.16,0.10),scarf);
 if(boost>0.5){
  for(int ti=0;ti<6;ti++){
   float ft=float(ti);
   vec2 td=q-vec2(-0.09-ft*0.025,-0.018+0.008*sin(time*12.0+ft));
   float trail=step(td.x*td.x+td.y*td.y,0.00010);
   col=mix(col,vec3(0.75,0.48,0.22),trail*(1.0-ft/7.0)*(0.45+0.55*fuel/100.0));
  }
 }
 float vig=sin(PI*uv.x)*sin(PI*uv.y);
 col*=0.82+0.18*vig;
 fragColor=vec4(col,1.0);
}
"""


# ============================== RENDERING ====================================
class PerspectiveLookup:
    """Monotone sampled quarter-sine used by CPU-projected world objects."""
    def __init__(self, samples: int = 2049):
        self.input = np.linspace(0.0, 1.0, samples, dtype=np.float64)
        self.depth = np.sin(0.5 * PI * self.input)

    def project(self, fraction: float) -> float:
        return float(np.interp(clamp01(fraction), self.input, self.depth))


class Renderer:
    def __init__(self, width: int, height: int, standalone: bool):
        import moderngl
        self.width, self.height = width, height
        self.perspective = PerspectiveLookup()
        if standalone:
            try:
                self.ctx = moderngl.create_standalone_context(require=330, backend="egl")
            except Exception:
                self.ctx = moderngl.create_standalone_context(require=330)
        else:
            self.ctx = moderngl.create_context(require=330)
        self.program = self.ctx.program(vertex_shader=VERTEX_SHADER, fragment_shader=SURF_FRAGMENT_SHADER)
        vertices = np.array((-1, -1, 1, -1, -1, 1, 1, 1), dtype="f4")
        self.vbo = self.ctx.buffer(vertices.tobytes())
        self.vao = self.ctx.simple_vertex_array(self.program, self.vbo, "in_pos")
        self.color = self.ctx.texture((width, height), 4)
        self.wave = self.ctx.texture((128, 128), 1, dtype="f4")
        self.wave.filter = (moderngl.LINEAR, moderngl.LINEAR)
        self.fbo = self.ctx.framebuffer((self.color,))
        self.program["wave_tex"].value = 0
        self.wave_version = -1
        self.present_program = None
        self.overlay = None
        if not standalone:
            self.present_program = self.ctx.program(
                vertex_shader=VERTEX_SHADER,
                fragment_shader="""#version 330
uniform sampler2D scene_tex;
uniform sampler2D overlay_tex;
in vec2 uv;
out vec4 fragColor;
void main(){
 vec3 scene=texture(scene_tex,uv).rgb;
 vec4 over=texture(overlay_tex,uv);
 fragColor=vec4(mix(scene,over.rgb,over.a),1.0);
}
""")
            self.present_vao = self.ctx.simple_vertex_array(self.present_program, self.vbo, "in_pos")
            self.overlay = self.ctx.texture((1280, 720), 4)
            self.overlay.filter = (moderngl.LINEAR, moderngl.LINEAR)
            self.present_program["scene_tex"].value = 0
            self.present_program["overlay_tex"].value = 1
        print(f"GL renderer: {self.ctx.info.get('GL_RENDERER', 'unknown')}")

    def render(self, state: GameState, field: WaveField, alpha: float = 1.0, visual_time: float | None = None,
               show_objects: bool = True, readback: bool = True) -> bytes | None:
        render_distance = state.distance - speed_at(state) * DT * (1.0 - clamp01(alpha))
        if render_distance < 0.0:
            render_distance = 0.0
        if field.version != self.wave_version:
            self.wave.write(np.asarray(field.u, dtype="f4").tobytes())
            self.wave_version = field.version
        self.wave.use(0)
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.program["resolution"].value = (float(self.width), float(self.height))
        self.program["time"].value = state.time if visual_time is None else visual_time
        self.program["progress"].value = render_distance / state.level.length
        self.program["travel"].value = render_distance
        self.program["lane_x"].value = state.lane_x
        self.program["lane_velocity"].value = state.lane_velocity
        self.program["jump_y"].value = state.height
        self.program["jump_velocity"].value = state.vertical_velocity
        self.program["slide_factor"].value = math.sin(PI * clamp01(state.action_time / SLIDE_TIME)) if state.action == "slide" else 0.0
        self.program["landing"].value = state.landing_compression
        self.program["hurt"].value = 1.0 if state.invulnerable > 0.0 else 0.0
        self.program["shield"].value = 1.0 if state.shield else 0.0
        self.program["bank"].value = state.camera.x
        packed = np.zeros((24, 4), dtype="f4")
        count = 0
        kind_id = {"coin": 0.0, "low": 1.0, "high": 2.0, "block": 3.0, "moving": 4.0, "magnet": 5.0, "shield": 6.0}
        if show_objects:
            for idx, thing in enumerate(state.level.things):
                if count >= 24:
                    break
                dz = thing.z * state.level.slice_m - render_distance
                if idx not in state.seen and 0.0 <= dz <= 100.0:
                    lane = moving_lane_position(thing, state.time)
                    if thing.kind == "coin" and state.magnet > 0.0:
                        pull = clamp01((22.0 - dz) / 22.0)
                        target_lane = state.lane_x / LANE_WIDTH
                        lane = lane + (target_lane - lane) * math.sin(0.5 * PI * pull)
                    packed[count] = (lane, self.perspective.project(1.0 - dz / 100.0), kind_id.get(thing.kind, 3.0), thing.phase)
                    count += 1
        self.program["object_count"].value = count
        self.program["character_visible"].value = 1 if show_objects else 0
        self.program["objects"].write(packed.tobytes())
        self.vao.render(mode=5)
        if readback:
            return self.fbo.read(components=3, alignment=1)
        return None

    def render(self, state: SurferState, field: WaveField, alpha: float = 1.0, visual_time: float | None = None,
               show_objects: bool = True, readback: bool = True) -> bytes | None:
        render_distance = state.distance - state.speed * DT * (1.0 - clamp01(alpha))
        if render_distance < 0.0:
            render_distance = 0.0
        render_y = state.y - state.vertical_velocity * DT * (1.0 - clamp01(alpha))
        if field.version != self.wave_version:
            self.wave.write(np.asarray(field.u, dtype="f4").tobytes())
            self.wave_version = field.version
        self.wave.use(0)
        self.fbo.use()
        self.ctx.viewport = (0, 0, self.width, self.height)
        self.program["resolution"].value = (float(self.width), float(self.height))
        self.program["time"].value = state.time if visual_time is None else visual_time
        self.program["travel"].value = render_distance
        self.program["level_length"].value = state.level.length
        self.program["surfer_y"].value = render_y
        self.program["camera_y"].value = state.camera_y
        self.program["surfer_pitch"].value = state.pitch
        self.program["surfer_vy"].value = state.vertical_velocity
        self.program["speed"].value = state.speed
        self.program["fuel"].value = state.fuel
        self.program["boost"].value = 1.0 if state.boosting else 0.0
        self.program["hurt"].value = 1.0 if state.invulnerable > 0.0 else 0.0
        self.program["grounded"].value = 1.0 if state.grounded else 0.0
        packed = np.zeros((24, 4), dtype="f4")
        kind_id = {"coin": 0.0, "boost": 1.0, "spring": 2.0, "breaker": 3.0}
        count = 0
        if show_objects:
            for idx, thing in enumerate(state.level.things):
                dx = thing.distance - render_distance
                if dx > 32.0:
                    break
                if idx in state.seen or dx < -2.0:
                    continue
                ground, _, _ = wave_geometry(thing.distance, state.level.index, state.level.length)
                screen_x = 0.30 + dx / 42.0
                screen_y = 0.39 + (ground - state.camera_y) * 0.075
                if thing.kind in ("coin", "boost"):
                    screen_y += 0.075 + 0.025 * math.sin(thing.phase)
                packed[count] = (screen_x, screen_y, kind_id[thing.kind], thing.phase)
                count += 1
                if count >= 24:
                    break
        self.program["object_count"].value = count
        self.program["objects"].write(packed.tobytes())
        self.vao.render(mode=5)
        if readback:
            return self.fbo.read(components=3, alignment=1)
        return None

    def present(self, overlay_surface: object, window_size: tuple[int, int]) -> None:
        """Composite entirely on the GPU, avoiding a framebuffer readback."""
        import pygame
        if self.present_program is None or self.overlay is None:
            raise RuntimeError("present() requires a window-backed renderer")
        self.overlay.write(pygame.image.tobytes(overlay_surface, "RGBA", True))
        self.ctx.screen.use()
        self.ctx.viewport = (0, 0, window_size[0], window_size[1])
        self.ctx.disable(self.ctx.BLEND)
        self.color.use(0)
        self.overlay.use(1)
        self.present_vao.render(mode=5)


# ============================== AUDIO ========================================
def half_sine_envelope(t: np.ndarray, duration: float) -> np.ndarray:
    phase = np.asarray(t, dtype=np.float64) / duration
    phase = np.where(phase < 0.0, 0.0, np.where(phase > 1.0, 1.0, phase))
    return np.sin(PI * phase) ** 2


def synth_audio(seconds: float, level: int = 0, sample_rate: int = 44100) -> np.ndarray:
    """The melodic Wave Surfer loop, built entirely from sine partials."""
    count = int(seconds * sample_rate)
    t = np.arange(count, dtype=np.float64) / sample_rate
    root = 110.0
    bpm = 84.0
    beat_time = 60.0 / bpm
    melody_ratios = (1.0, 9.0/8.0, 5.0/4.0, 3.0/2.0, 5.0/3.0, 3.0/2.0, 5.0/4.0, 9.0/8.0,
                     1.0, 5.0/4.0, 3.0/2.0, 2.0, 5.0/3.0, 3.0/2.0, 9.0/8.0, 1.0)
    note_time = 0.5 * beat_time
    step = np.floor(t / note_time)
    local = t - step * note_time
    melody = np.zeros(count, dtype=np.float64)
    counter = np.zeros(count, dtype=np.float64)
    for note_index, ratio in enumerate(melody_ratios):
        mask = np.where(np.mod(step, 16.0) == float(note_index), 1.0, 0.0)
        frequency = root * 2.0 * ratio
        envelope = np.sin(PI * local / note_time) ** 2
        vibrato = 0.018 * np.sin(TAU * 4.2 * local)
        tone = np.sin(TAU * frequency * local + vibrato)
        tone += 0.12 * np.sin(TAU * frequency * 2.0 * local)
        melody += mask * envelope * tone
        counter_ratio = melody_ratios[(note_index + 5) % 16]
        counter += mask * envelope * np.sin(TAU * root * counter_ratio * local + 0.35 * math.sin(note_index))

    bass_step = np.floor(t / (2.0 * beat_time))
    bass_local = t - bass_step * 2.0 * beat_time
    bass = np.zeros(count, dtype=np.float64)
    pad = np.zeros(count, dtype=np.float64)
    bass_ratios = (1.0, 4.0/3.0, 3.0/2.0, 5.0/4.0)
    for note_index, ratio in enumerate(bass_ratios):
        mask = np.where(np.mod(bass_step, 4.0) == float(note_index), 1.0, 0.0)
        envelope = np.sin(PI * bass_local / (2.0 * beat_time)) ** 2
        bass += mask * envelope * (np.sin(TAU * root * 0.5 * ratio * bass_local) +
                                    0.12 * np.sin(TAU * root * ratio * bass_local))
        third = 5.0/4.0
        pad += mask * envelope * (np.sin(TAU * root * ratio * t) +
                                  0.50 * np.sin(TAU * root * ratio * third * t) +
                                  0.36 * np.sin(TAU * root * ratio * 3.0/2.0 * t))

    kick_step = np.floor(t / beat_time)
    kick_local = t - kick_step * beat_time
    kick_env = np.where(kick_local < 0.22, np.sin(PI * kick_local / 0.22) ** 2, 0.0)
    kick = kick_env * np.sin(TAU * (48.0 * kick_local + 8.0 * np.sin(PI * kick_local / 0.22)))
    pulse_step = np.floor(t / (0.5 * beat_time))
    pulse_local = t - pulse_step * 0.5 * beat_time
    pulse_env = np.where(pulse_local < 0.11, np.sin(PI * pulse_local / 0.11) ** 2, 0.0)
    pulse = pulse_env * (np.sin(TAU * 900.0 * pulse_local) +
                         0.22 * np.sin(TAU * 1350.0 * pulse_local))
    audio = 0.105 * pad + 0.17 * bass + 0.19 * melody + 0.055 * counter + 0.050 * kick + 0.010 * pulse
    loop_fade = 0.08
    start_fade = np.where(t < loop_fade, np.sin(0.5 * PI * t / loop_fade) ** 2, 1.0)
    remaining = seconds - t
    end_fade = np.where(remaining < loop_fade, np.sin(0.5 * PI * remaining / loop_fade) ** 2, 1.0)
    audio *= start_fade * end_fade
    return np.asarray(audio, dtype=np.float32)


def synth_sfx(name: str, combo: int = 0, sample_rate: int = 44100) -> np.ndarray:
    durations = {"jump": 0.24, "coin": 0.16, "slide": 0.20, "crash": 0.55, "shield": 0.38,
                 "shield_hit": 0.30, "magnet": 0.48, "spring": 0.52, "finish": 1.2}
    duration = durations.get(name, 0.22)
    t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    bases = {"jump": 220.0, "coin": 660.0 + 22.0 * combo, "slide": 100.0, "crash": 55.0,
             "shield": 330.0, "shield_hit": 180.0, "magnet": 146.7, "spring": 293.3, "finish": 440.0}
    base = bases.get(name, 220.0)
    sweep = 1.0 + 0.24 * np.sin(0.5 * PI * t / duration)
    carrier = np.sin(TAU * base * t * sweep)
    if name == "coin":
        carrier += 0.38 * np.sin(TAU * base * 2.0 * t)
    elif name == "spring":
        carrier += 0.46 * np.sin(TAU * base * 5.0/4.0 * t)
        carrier += 0.32 * np.sin(TAU * base * 3.0/2.0 * t)
    elif name in ("crash", "slide"):
        carrier = np.sin(TAU * base * t * (1.0 - 0.36 * np.sin(0.5 * PI * t / duration)))
        carrier += 0.28 * np.sin(TAU * (base * 1.47) * t)
    elif name in ("shield", "shield_hit", "magnet"):
        carrier += 0.30 * np.sin(TAU * base * 3.0 / 2.0 * t + 0.65 * np.sin(TAU * 5.0 * t))
    elif name == "finish":
        carrier += 0.5 * np.sin(TAU * base * 5.0 / 4.0 * t) + 0.35 * np.sin(TAU * base * 3.0 / 2.0 * t)
    return np.asarray(0.22 * carrier * half_sine_envelope(t, duration), dtype=np.float32)


def doppler_frequency(source_frequency: float, source_velocity: float, sound_speed: float = 343.0) -> float:
    denominator = sound_speed - source_velocity
    if denominator < 1.0:
        denominator = 1.0
    return source_frequency * sound_speed / denominator


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = 44100) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(audio * 30000.0, dtype=np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


class AudioEngine:
    def __init__(self, out_dir: Path, level: int):
        self.available = False
        self.audio = synth_audio(16.0, level)
        self.position = 0
        self.effects: list[tuple[np.ndarray, int]] = []
        self.stream = None
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=44100, channels=1, dtype="float32", blocksize=1024,
                                          callback=self._callback)
            self.stream.start()
            self.available = True
        except Exception:
            write_wav(out_dir / f"level_{level + 1}_fallback.wav", self.audio)

    def _callback(self, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        indices = (np.arange(frames) + self.position) % len(self.audio)
        block = self.audio[indices].copy()
        self.position = int((self.position + frames) % len(self.audio))
        remaining: list[tuple[np.ndarray, int]] = []
        for effect, position in self.effects:
            count = len(effect) - position
            if count > frames:
                count = frames
            if count > 0:
                block[:count] += effect[position:position + count]
            position += count
            if position < len(effect):
                remaining.append((effect, position))
        self.effects = remaining
        outdata[:, 0] = np.where(block < -0.95, -0.95, np.where(block > 0.95, 0.95, block))

    def trigger(self, name: str, combo: int = 0) -> None:
        self.effects.append((synth_sfx(name, combo), 0))

    def set_level(self, level: int) -> None:
        self.audio = synth_audio(16.0, level)
        self.position = 0
        self.effects.clear()

    def stop(self) -> None:
        if self.available:
            try:
                if self.stream is not None:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass


# ============================== SINE FONT / UI ===============================
# Segment strokes are illuminated with a sin^2 pulse; no font asset is loaded.
SEGMENTS = {
    "0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
    "A": "abcefg", "B": "fegcd", "C": "afed", "D": "begcd", "E": "afged", "F": "afge",
    "G": "afedcg", "H": "febcg", "I": "ed", "J": "bcd", "K": "feg", "L": "fed",
    "M": "abcef", "N": "ceg", "O": "abcdef", "P": "abfge", "Q": "abcfg", "R": "eg",
    "S": "afgcd", "T": "fged", "U": "febcd", "V": "ecd", "W": "bcdef", "X": "fgbc",
    "Y": "fgbcd", "Z": "abged", "-": "g", " ": "", ":": "h", ".": "i", "%": "hi",
}
SEG_LINES = {"a": ((1, 0), (4, 0)), "b": ((5, 1), (5, 3)), "c": ((5, 4), (5, 6)),
             "d": ((1, 7), (4, 7)), "e": ((0, 4), (0, 6)), "f": ((0, 1), (0, 3)),
             "g": ((1, 3.5), (4, 3.5)), "h": ((2.5, 2), (2.5, 2)), "i": ((2.5, 7), (2.5, 7))}

PIXEL_FONT = {
    "A": ("01110","10001","10001","11111","10001","10001","10001"),
    "B": ("11110","10001","10001","11110","10001","10001","11110"),
    "C": ("01111","10000","10000","10000","10000","10000","01111"),
    "D": ("11110","10001","10001","10001","10001","10001","11110"),
    "E": ("11111","10000","10000","11110","10000","10000","11111"),
    "F": ("11111","10000","10000","11110","10000","10000","10000"),
    "G": ("01111","10000","10000","10111","10001","10001","01111"),
    "H": ("10001","10001","10001","11111","10001","10001","10001"),
    "I": ("11111","00100","00100","00100","00100","00100","11111"),
    "J": ("00111","00010","00010","00010","10010","10010","01100"),
    "K": ("10001","10010","10100","11000","10100","10010","10001"),
    "L": ("10000","10000","10000","10000","10000","10000","11111"),
    "M": ("10001","11011","10101","10101","10001","10001","10001"),
    "N": ("10001","11001","10101","10011","10001","10001","10001"),
    "O": ("01110","10001","10001","10001","10001","10001","01110"),
    "P": ("11110","10001","10001","11110","10000","10000","10000"),
    "Q": ("01110","10001","10001","10001","10101","10010","01101"),
    "R": ("11110","10001","10001","11110","10100","10010","10001"),
    "S": ("01111","10000","10000","01110","00001","00001","11110"),
    "T": ("11111","00100","00100","00100","00100","00100","00100"),
    "U": ("10001","10001","10001","10001","10001","10001","01110"),
    "V": ("10001","10001","10001","10001","10001","01010","00100"),
    "W": ("10001","10001","10001","10101","10101","11011","10001"),
    "X": ("10001","10001","01010","00100","01010","10001","10001"),
    "Y": ("10001","10001","01010","00100","00100","00100","00100"),
    "Z": ("11111","00001","00010","00100","01000","10000","11111"),
    "0": ("01110","10001","10011","10101","11001","10001","01110"),
    "1": ("00100","01100","00100","00100","00100","00100","01110"),
    "2": ("01110","10001","00001","00010","00100","01000","11111"),
    "3": ("11110","00001","00001","01110","00001","00001","11110"),
    "4": ("00010","00110","01010","10010","11111","00010","00010"),
    "5": ("11111","10000","10000","11110","00001","00001","11110"),
    "6": ("01110","10000","10000","11110","10001","10001","01110"),
    "7": ("11111","00001","00010","00100","01000","01000","01000"),
    "8": ("01110","10001","10001","01110","10001","10001","01110"),
    "9": ("01110","10001","10001","01111","00001","00001","01110"),
    "-": ("00000","00000","00000","11111","00000","00000","00000"),
    ":": ("00000","00100","00100","00000","00100","00100","00000"),
    ".": ("00000","00000","00000","00000","00000","00100","00100"),
    "+": ("00000","00100","00100","11111","00100","00100","00000"),
    "=": ("00000","00000","11111","00000","11111","00000","00000"),
    "/": ("00001","00010","00010","00100","01000","01000","10000"),
    "(": ("00010","00100","01000","01000","01000","00100","00010"),
    ")": ("01000","00100","00010","00010","00010","00100","01000"),
    ">": ("10000","01000","00100","00010","00100","01000","10000"),
    "<": ("00001","00010","00100","01000","00100","00010","00001"),
    "?": ("01110","10001","00001","00010","00100","00000","00100"),
    " ": ("00000","00000","00000","00000","00000","00000","00000"),
}


def draw_sine_text(surface: object, text: str, x: int, y: int, scale: int, color: tuple[int, int, int], phase: float = 0.0) -> None:
    import pygame
    glow = math.sin(phase * 0.7) ** 2
    for pos, char in enumerate(text.upper()):
        rows = PIXEL_FONT.get(char, PIXEL_FONT[" "])
        for row, bits in enumerate(rows):
            for column, bit in enumerate(bits):
                if bit == "1":
                    shimmer = math.sin(phase + row * 0.31 + column * 0.23) ** 2
                    ink = tuple(int(c * (0.82 + 0.12 * glow + 0.06 * shimmer)) for c in color)
                    rect = (x + pos * 7 * scale + column * scale, y + row * scale, scale, scale)
                    pygame.draw.rect(surface, (18, 13, 9), (rect[0] + scale // 2, rect[1] + scale // 2, scale, scale), border_radius=scale // 3)
                    pygame.draw.rect(surface, ink, rect, border_radius=scale // 3)


def sine_text_width(text: str, scale: int) -> int:
    return len(text) * 7 * scale


def draw_centered_text(surface: object, text: str, y: int, scale: int, color: tuple[int, int, int], phase: float = 0.0) -> None:
    width = surface.get_width()
    draw_sine_text(surface, text, int((width - sine_text_width(text, scale)) * 0.5), y, scale, color, phase)


def draw_panel(surface: object, rect: tuple[int, int, int, int], accent: tuple[int, int, int], phase: float = 0.0,
               selected: bool = False, alpha: int | None = None) -> None:
    import pygame
    pulse = math.sin(phase) ** 2
    fill_alpha = (218 if selected else 202) if alpha is None else alpha
    border = tuple(int(c * (0.70 + 0.30 * pulse)) for c in accent)
    pygame.draw.rect(surface, (17, 13, 10, fill_alpha), rect, border_radius=20)
    pygame.draw.rect(surface, border, rect, 3 if selected else 1, border_radius=20)
    x, y, w, h = rect
    pygame.draw.line(surface, (*border, 90), (x + 24, y + 12), (x + w - 24, y + 12), 2)


def draw_heart_icon(surface: object, x: int, y: int, filled: bool, phase: float) -> None:
    import pygame
    pulse = 1.0 + 0.08 * math.sin(phase)
    radius = int(8 * pulse)
    color = (230, 195, 151) if filled else (55, 45, 36)
    pygame.draw.circle(surface, color, (x - radius // 2, y), radius)
    pygame.draw.circle(surface, color, (x + radius // 2, y), radius)
    pygame.draw.polygon(surface, color, ((x - radius - 3, y + 2), (x + radius + 3, y + 2), (x, y + radius + 12)))


def draw_star_icon(surface: object, x: int, y: int, filled: bool, phase: float) -> None:
    import pygame
    points: list[tuple[int, int]] = []
    for i in range(10):
        radius = 18 if i % 2 == 0 else 8
        radius = int(radius * (1.0 + 0.06 * math.sin(phase + i)))
        angle = -0.5 * PI + i * PI / 5.0
        points.append((x + int(radius * math.cos(angle)), y + int(radius * math.sin(angle))))
    pygame.draw.polygon(surface, (231, 192, 137) if filled else (55, 45, 36), points)


# ============================== HEADLESS / APP ================================
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


def scripted_surf_input(frame: int, state: SurferState, script: dict[str, list[int]] | None = None) -> Inputs:
    if script:
        return Inputs(frame in script.get("left", []), frame in script.get("right", []),
                      frame in script.get("jump", []), frame in script.get("slide", []))
    _, slope, _ = wave_geometry(state.distance + 4.5, state.level.index, state.level.length)
    hazard_ahead = False
    for idx, thing in enumerate(state.level.things):
        if idx not in state.seen and thing.kind == "breaker" and 1.5 < thing.distance - state.distance < 5.2:
            hazard_ahead = True
            break
    jump = state.grounded and (hazard_ahead or slope * slope > 1.0)
    boost = state.grounded and not jump and state.fuel > 35.0 and slope * slope > 0.48
    return Inputs(right=boost, jump=jump)


def run_headless(args: argparse.Namespace) -> str:
    from PIL import Image
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    level = generate_surf_level(0, args.seed)
    state = SurferState(level)
    field = WaveField()
    script = None
    if args.script:
        script = json.loads(Path(args.script).read_text(encoding="utf8"))
    try:
        renderer = Renderer(args.width, args.height, True)
    except Exception as exc:
        renderer = None
        print(f"GL unavailable ({exc}); using deterministic Pillow diagnostic frames")
    start = time.perf_counter()
    for frame in range(args.frames):
        update_surfer(state, scripted_surf_input(frame, state, script))
        if "coin" in state.events or "crash" in state.events:
            field.inject(64, 74 if "coin" in state.events else 52)
        field.step()
        if frame % args.capture_every == 0 or frame == args.frames - 1:
            if renderer:
                raw = renderer.render(state, field)
                image = Image.frombytes("RGB", (args.width, args.height), raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                image = Image.new("RGB", (args.width, args.height), (8, 4, 24))
                px = image.load()
                for yy in range(args.height):
                    for xx in range(args.width):
                        q = math.sin(PI * xx / args.width) * math.sin(PI * yy / args.height)
                        px[xx, yy] = (int(20 + 80 * q * q), int(10 + 30 * q * q), int(45 + 120 * q * q))
            image.save(out / f"wave_frame{frame:06d}.png")
    elapsed = time.perf_counter() - start
    digest = surf_logic_hash(state)
    if state.finished:
        result_stars = stars_for(True, state.coins, state.level.optimal_coins, state.crashes)
        is_new = ScoreStore(out / "scores.json").record(state.level.index, final_score(state), state.time, result_stars)
        print(f"results score={final_score(state)} stars={result_stars} new_high_score={is_new}")
    elif state.failed:
        print("results failed retry=true")
    print(f"logic-sha256 {digest}")
    print(f"frames {args.frames}; mean frame time {1000.0 * elapsed / args.frames:.3f} ms")
    return digest


def run_interactive(args: argparse.Namespace) -> None:
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((args.window_width, args.window_height),
                                     pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=1)
    pygame.display.set_caption("WAVE SURFER")
    renderer = Renderer(args.width, args.height, False)
    level_index = 0
    state = SurferState(generate_surf_level(level_index, args.seed))
    field = WaveField()
    store = ScoreStore(Path(args.out) / "scores.json")
    audio = AudioEngine(Path(args.out), level_index)
    clock = pygame.time.Clock()
    accumulator = 0.0
    previous = time.perf_counter()
    paused = False
    help_open = False
    recorded = False
    mode = "menu"
    new = False
    pending_input = Inputs()
    running = True
    while running:
        now = time.perf_counter()
        accumulator += now - previous
        previous = now
        one_shot = Inputs()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and mode == "menu":
                    state = SurferState(generate_surf_level(level_index, args.seed)); field = WaveField(); recorded = False; mode = "run"
                    audio.set_level(level_index)
                    audio.trigger("finish")
                    accumulator = 0.0
                    previous = time.perf_counter()
                elif event.key == pygame.K_p: paused = not paused
                elif event.key == pygame.K_h: help_open = not help_open
                elif event.key == pygame.K_r: state = SurferState(generate_surf_level(level_index, args.seed)); field = WaveField(); recorded = False; mode = "run"
                elif event.key in (pygame.K_LEFT, pygame.K_a): one_shot.left = True
                elif event.key in (pygame.K_RIGHT, pygame.K_d): one_shot.right = True
                elif event.key in (pygame.K_UP, pygame.K_w, pygame.K_SPACE): one_shot.jump = True
                elif event.key in (pygame.K_DOWN, pygame.K_s): one_shot.slide = True
        pending_input.left = pending_input.left or one_shot.left
        pending_input.right = pending_input.right or one_shot.right
        pending_input.jump = pending_input.jump or one_shot.jump
        pending_input.slide = pending_input.slide or one_shot.slide
        continuous_input = Inputs()
        if mode == "run":
            held = pygame.key.get_pressed()
            pending_input.left = pending_input.left or held[pygame.K_LEFT] or held[pygame.K_a]
            pending_input.right = pending_input.right or held[pygame.K_RIGHT] or held[pygame.K_d]
            pending_input.slide = pending_input.slide or held[pygame.K_DOWN] or held[pygame.K_s]
            continuous_input = Inputs(left=held[pygame.K_LEFT] or held[pygame.K_a],
                                      right=held[pygame.K_RIGHT] or held[pygame.K_d],
                                      slide=held[pygame.K_DOWN] or held[pygame.K_s])
        if mode == "run" and not paused and not help_open:
            first = True
            while accumulator >= DT:
                update_surfer(state, pending_input if first else continuous_input)
                if first:
                    pending_input = Inputs()
                first = False
                for sound_event in state.events:
                    audio.trigger(sound_event, state.combo)
                if "coin" in state.events or "crash" in state.events:
                    field.inject(64, 72)
                field.step()
                accumulator -= DT
        else:
            accumulator = 0.0
        ui_time = now % 1000.0
        renderer.render(state, field, accumulator / DT, ui_time if mode != "run" else None, mode == "run", readback=False)
        overlay = pygame.Surface((1280, 720), pygame.SRCALPHA)
        best = store.data.get(str(level_index + 1), {}).get("score", 0)
        if mode == "menu":
            overlay.fill((238, 229, 214, 72))
            draw_centered_text(overlay, "W A V E", 122, 7, (43, 37, 31), ui_time)
            draw_centered_text(overlay, "S U R F E R", 212, 5, (43, 37, 31), ui_time)
            pygame.draw.rect(overlay, (45, 39, 33), (420, 438, 440, 82), border_radius=38)
            draw_centered_text(overlay, "RIDE >", 461, 4, (239, 229, 211), ui_time)
            pygame.draw.rect(overlay, (76, 65, 54), (603, 558, 74, 64), 2, border_radius=12)
            draw_sine_text(overlay, "H", 628, 577, 4, (76, 65, 54), ui_time)
            draw_centered_text(overlay, "H FOR HELP", 648, 2, (101, 82, 64), ui_time)
        else:
            pygame.draw.rect(overlay, (222, 198, 166), (28, 24, 58, 58), 2, border_radius=12)
            pygame.draw.rect(overlay, (231, 210, 181), (46, 40, 6, 26), border_radius=3)
            pygame.draw.rect(overlay, (231, 210, 181), (61, 40, 6, 26), border_radius=3)
            for heart_index in range(3):
                draw_heart_icon(overlay, 122 + heart_index * 38, 47, heart_index < state.hearts, state.time * 4.0 + heart_index)
            score_text = f"{final_score(state):05d}"
            draw_sine_text(overlay, score_text, 1228 - sine_text_width(score_text, 4), 24, 4, (237, 217, 190), state.time)
            draw_sine_text(overlay, f"BEST {best}", 1228 - sine_text_width(f"BEST {best}", 2), 66, 2, (139, 117, 95), state.time)
            pygame.draw.circle(overlay, (212, 158, 96), (1136, 105), 14)
            pygame.draw.circle(overlay, (239, 199, 143), (1136, 105), 8, 3)
            draw_sine_text(overlay, str(state.coins), 1163, 94, 3, (237, 217, 190), state.time)
            draw_centered_text(overlay, str(LEVELS[level_index]["name"]), 31, 2, (169, 143, 117), state.time)
            if state.combo > 1:
                draw_centered_text(overlay, f"COMBO {state.combo}", 66, 2, (224, 184, 132), state.time)
            pygame.draw.rect(overlay, (52, 41, 32, 210), (140, 682, 1000, 4), border_radius=2)
            progress_width = int(1000 * state.distance / state.level.length)
            pygame.draw.rect(overlay, (224, 192, 151), (140, 682, progress_width, 4), border_radius=2)
            pygame.draw.circle(overlay, (239, 218, 190), (140 + progress_width, 684), 6)
            pygame.draw.rect(overlay, (54, 47, 40, 180), (1010, 144, 218, 18), border_radius=9)
            fuel_width = int(210.0 * state.fuel / 100.0)
            pygame.draw.rect(overlay, (198, 139, 70), (1014, 148, fuel_width, 10), border_radius=5)
            draw_sine_text(overlay, f"BOOST {int(state.fuel)}", 1010, 174, 2, (89, 72, 56), state.time)
            draw_sine_text(overlay, "UP JUMP   RIGHT BOOST   LEFT BRAKE", 28, 620, 2, (110, 88, 67), state.time)
            if not audio.available:
                draw_sine_text(overlay, "AUDIO OFF  WAV SAVED", 28, 650, 2, (169, 133, 99), state.time)
        if help_open:
            overlay.fill((25, 21, 18, 224))
            draw_panel(overlay, (90, 54, 1100, 610), (218, 190, 153), state.time, True)
            lines = ("HOW TO SURF", "UP OR SPACE: JUMP", "RIGHT OR DOWN: HOLD TO BOOST", "LEFT: BRAKE",
                     "GOLD WAVE RINGS: COINS", "BLUE ORBS: BOOST FUEL", "GREEN SPRINGS: HIGH JUMP", "RED BREAKERS: OBSTACLES",
                     "ONLY OBSTACLE COLLISIONS COST HEARTS", "H: CLOSE HELP")
            for i, line in enumerate(lines):
                color = (239, 215, 184) if i == 0 else (178, 150, 120)
                draw_sine_text(overlay, line, 132, 92 + i * 52, 3 if i == 0 else 2, color, state.time + i)
        if mode == "run" and (state.finished or state.failed):
            overlay.fill((10, 8, 6, 184))
            title = "LEVEL COMPLETE" if state.finished else "GAME OVER"
            draw_panel(overlay, (330, 138, 620, 440), (225, 198, 162), state.time, True)
            draw_centered_text(overlay, title, 184, 5, (239, 218, 190), state.time)
            if state.finished:
                stars = stars_for(True, state.coins, state.level.optimal_coins, state.crashes)
                if not recorded:
                    new = store.record(level_index, final_score(state), state.time, stars)
                    recorded = True
                for star_index in range(3):
                    draw_star_icon(overlay, 550 + star_index * 90, 302, star_index < stars, state.time + star_index)
                draw_centered_text(overlay, f"SCORE {final_score(state)}", 358, 3, (235, 206, 167), state.time)
                draw_centered_text(overlay, f"TIME {int(state.time)}  COINS {state.coins}", 410, 2, (164, 137, 109), state.time)
                if new: draw_centered_text(overlay, "NEW HIGH SCORE", 460, 2, (224, 183, 131), state.time)
            pygame.draw.rect(overlay, (226, 203, 173), (440, 505, 400, 52), border_radius=10)
            draw_centered_text(overlay, "R RETRY     ESC QUIT", 521, 2, (39, 29, 22), state.time)
        renderer.present(overlay, screen.get_size())
        pygame.display.flip()
        pygame.display.set_caption(f"WAVE SURFER | hearts {state.hearts} coins {state.coins} score {final_score(state)}")
        # Presentation is paced by the display; the fixed-step accumulator keeps
        # gameplay at exactly 60 Hz even on high-refresh or non-vsync systems.
        clock.tick()
    audio.stop()
    pygame.quit()


# --- REPORT (lint-exempt) ---
def run_report(out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ("euler", "symplectic", "verlet", "rk4")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    orders: dict[str, float] = {}
    stability: dict[str, float] = {}
    for kind in kinds:
        x, v = 1.0, 0.0
        energies = []
        for i in range(100000):
            x, v = oscillator_step(kind, x, v, 0.001, 4.0, 0.0)
            if i % 100 == 0:
                energies.append(0.5 * (v * v + 16.0 * x * x))
        axes[0].plot(np.arange(len(energies)) * 0.1, energies, label=kind)
        dts = np.array((0.04, 0.02, 0.01, 0.005), dtype=float)
        errors = []
        exact_x = math.cos(4.0)
        exact_v = -4.0 * math.sin(4.0)
        for dt in dts:
            xx, vv, t = 1.0, 0.0, 0.0
            while t + dt <= 1.0 + 1e-12:
                xx, vv = oscillator_step(kind, xx, vv, dt, 4.0, 0.0)
                t += dt
            errors.append(abs(xx - exact_x) + abs(vv - exact_v))
        order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
        orders[kind] = order
        axes[1].loglog(dts, errors, "o-", label=f"{kind} p={order:.2f}")
        stable_dt = 0.0
        for candidate in np.linspace(0.005, 1.0, 200):
            sx, sv, bounded = 1.0, 0.0, True
            for _ in range(2000):
                sx, sv = oscillator_step(kind, sx, sv, float(candidate), 4.0, 0.0)
                if abs(sx) > 100.0 or abs(sv) > 400.0:
                    bounded = False
                    break
            if bounded:
                stable_dt = float(candidate)
        stability[kind] = stable_dt
    axes[0].set_title("Undamped energy, 100,000 steps")
    axes[0].legend(); axes[1].set_title("Convergence order"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "integrators.png", dpi=150); plt.close(fig)

    field = WaveField(damping=0.0)
    field.inject(64, 64)
    snapshots = []
    energy = []
    for i in range(360):
        field.step(); energy.append(field.energy())
        if i in (0, 60, 120, 240): snapshots.append(field.u.copy())
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, data in zip(axes, snapshots): ax.imshow(data, cmap="magma"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "wave_pde.png", dpi=150); plt.close(fig)

    values = np.array([sine_hash(i, 7.0) for i in range(20000)])
    serial = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    builtin = np.random.default_rng(7).random(20000)
    weyl = np.array([fract(i * PHI_WEYL) for i in range(20000)])
    target = (np.arange(20000) + 0.5) / 20000.0
    discrepancies = {"sine_hash": float(np.max(np.abs(np.sort(values) - target))),
                     "builtin": float(np.max(np.abs(np.sort(builtin) - target))),
                     "weyl": float(np.max(np.abs(np.sort(weyl) - target)))}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(values, bins=50); axes[0].set_title("Sine-hash uniformity")
    axes[1].scatter(values[:-1:20], values[1::20], s=2); axes[1].set_title(f"serial r={serial:.4f}")
    fig.tight_layout(); fig.savefig(out / "hash_quality.png", dpi=150); plt.close(fig)

    x = np.linspace(0.0, 6.0 * PI, 1200)
    moire = np.sin(20.0 * x) * np.sin(22.0 * x)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(x, moire); axes[0].set_title("Moire beat |22-20|=2")
    signal = np.sin(TAU * 440.0 * np.arange(4096) / 44100.0 + 2.0 * np.sin(TAU * 110.0 * np.arange(4096) / 44100.0))
    spectrum = abs(np.fft.rfft(signal * np.sin(PI * np.arange(4096) / 4095.0) ** 2))
    axes[1].plot(np.fft.rfftfreq(4096, 1 / 44100.0), spectrum); axes[1].set_xlim(0, 1500); axes[1].set_title("FM sidebands + Hann window")
    fig.tight_layout(); fig.savefig(out / "moire_alias_audio.png", dpi=150); plt.close(fig)
    (out / "report.json").write_text(json.dumps({"orders": orders, "stability_dt": stability,
        "hash_serial_correlation": serial, "uniform_discrepancy": discrepancies,
        "pde_energy_drift": (max(energy)-min(energy))/max(energy)}, indent=2))
    print(f"report written to {out}")
# --- END REPORT (lint-exempt) ---


# ============================== CLI ==========================================
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WAVE SURFER")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--script")
    p.add_argument("--out", default="out")
    p.add_argument("--width", type=int, default=int(os.environ.get("SINE_WIDTH", "640")))
    p.add_argument("--height", type=int, default=int(os.environ.get("SINE_HEIGHT", "360")))
    p.add_argument("--window-width", type=int, default=1280)
    p.add_argument("--window-height", type=int, default=720)
    p.add_argument("--capture-every", type=int, default=120)
    return p


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    errors = lint_game()
    if errors:
        print("sine-rule lint failed:\n" + "\n".join(errors), file=sys.stderr)
        return 2
    if args.report:
        run_report(args.out)
    elif args.headless:
        run_headless(args)
    else:
        run_interactive(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
