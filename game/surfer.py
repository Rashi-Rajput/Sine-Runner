from __future__ import annotations

import hashlib
import math
from dataclasses import dataclass, field

from .collision import (CollisionRecord, apply_breaker_damage,
                        process_contacts)
from .terrain import (BOARD_CLEARANCE, DT, LEVELS, PI, SURF_JUMP_SPEED,
                      clamp01, surf_geometry)

@dataclass
class Inputs:
    left: bool = False
    right: bool = False
    jump: bool = False
    slide: bool = False


def final_score(state: object) -> int:
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
    sliding: bool = False
    landing_compression: float = 0.0
    air_time: float = 0.0
    events: list[str] = field(default_factory=list)
    collisions: list[CollisionRecord] = field(default_factory=list)
    last_collision: CollisionRecord | None = None
    seen: set[int] = field(default_factory=set)

    def __post_init__(self) -> None:
        ground, slope, _ = surf_geometry(self.level, self.distance)
        self.y = ground + BOARD_CLEARANCE
        self.camera_y = ground
        self.pitch = slope / (1.0 + 0.32 * slope * slope)


def surfer_wipeout(state: SurferState) -> bool:
    """Apply obstacle damage and report whether a heart was actually removed."""
    return apply_breaker_damage(state)


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
    state.collisions.clear()
    state.time += dt
    state.invulnerable -= dt
    state.landing_compression -= 4.5 * dt
    if state.landing_compression < 0.0:
        state.landing_compression = 0.0
    progress = clamp01(state.distance / state.level.length)
    cruise = 10.5 + 8.0 * math.sin(0.5 * PI * progress)
    state.sliding = inputs.slide and state.grounded
    state.boosting = inputs.right and state.fuel > 0.0
    if state.boosting:
        state.speed += state.level.domain.boost_strength * dt
        state.fuel -= 24.0 * dt
        if state.speed > cruise + 7.0:
            state.speed = cruise + 7.0
    else:
        state.speed += (cruise - state.speed) * state.level.domain.friction * dt
        state.fuel += 3.5 * dt
    if state.sliding:
        state.speed += 1.2 * dt
    if inputs.left:
        state.speed -= 8.0 * dt
    if state.speed < 6.0:
        state.speed = 6.0
    if state.fuel < 0.0:
        state.fuel = 0.0
    if state.fuel > 100.0:
        state.fuel = 100.0

    old_distance = state.distance
    old_y = state.y
    old_ground, old_slope, _ = surf_geometry(state.level, old_distance)
    if inputs.jump and state.grounded:
        state.grounded = False
        state.vertical_velocity = old_slope * state.speed + SURF_JUMP_SPEED
        state.action = "air"
        state.air_time = 0.0
        state.events.append("jump")

    state.distance += state.speed * dt
    ground, slope, curvature = surf_geometry(state.level, state.distance)
    if state.grounded:
        state.y = ground + BOARD_CLEARANCE
        state.vertical_velocity = slope * state.speed
        state.action = "slide" if state.sliding else ("boost" if state.boosting else "surf")
        state.air_time = 0.0
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
        state.air_time += dt
        gravity = state.level.domain.gravity
        state.y += state.vertical_velocity * dt + 0.5 * gravity * dt * dt
        state.vertical_velocity += gravity * dt
        if state.y <= ground + BOARD_CLEARANCE and state.vertical_velocity < slope * state.speed:
            relative = state.vertical_velocity - slope * state.speed
            impact2 = relative * relative / (1.0 + slope * slope)
            state.y = ground + BOARD_CLEARANCE
            compression = impact2 / 100.0
            if compression > 1.0:
                compression = 1.0
            state.landing_compression = compression
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
    process_contacts(state, old_distance, old_y, slope)

    camera_target = ground + 0.28 * (state.y - ground - BOARD_CLEARANCE)
    camera_accel = 32.0 * (camera_target - state.camera_y) - 10.0 * state.camera_velocity
    state.camera_velocity += camera_accel * dt
    state.camera_y += state.camera_velocity * dt
    if state.distance >= state.level.length:
        state.distance = state.level.length
        state.finished = True
        state.events.append("finish")


def surf_logic_hash(state: SurferState) -> str:
    domain = state.level.domain
    payload = (state.level.index, state.level.seed, domain.stage, round(domain.difficulty, 9),
               round(domain.friction, 9), round(domain.boost_strength, 9), round(domain.gravity, 9),
               round(state.time, 9), round(state.distance, 9),
               round(state.y, 9), round(state.vertical_velocity, 9), round(state.speed, 9),
               round(state.fuel, 9), state.grounded, state.boosting, state.sliding, state.action,
               state.hearts, state.coins, state.score, state.crashes,
               state.finished, state.failed, tuple(sorted(state.seen)))
    return hashlib.sha256(repr(payload).encode("ascii")).hexdigest()
def scripted_surf_input(frame: int, state: SurferState, script: dict[str, list[int]] | None = None) -> Inputs:
    if script:
        return Inputs(frame in script.get("left", []), frame in script.get("right", []),
                      frame in script.get("jump", []), frame in script.get("slide", []))
    _, slope, _ = surf_geometry(state.level, state.distance + 4.5)
    hazard_ahead = False
    for idx, thing in enumerate(state.level.things):
        if idx not in state.seen and thing.kind == "breaker" and 1.5 < thing.distance - state.distance < 5.2:
            hazard_ahead = True
            break
    jump = state.grounded and (hazard_ahead or slope * slope > 1.0)
    boost = state.grounded and not jump and state.fuel > 35.0 and slope * slope > 0.48
    return Inputs(right=boost, jump=jump)


