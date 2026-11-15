from __future__ import annotations

from dataclasses import dataclass

from .terrain import BOARD_CLEARANCE, clamp01, surf_geometry

@dataclass(frozen=True)
class CollisionRecord:
    """One swept gameplay contact, retained for tests, UI, and RL rewards."""
    time: float
    object_index: int
    kind: str
    object_distance: float
    altitude: float
    heart_delta: int
    outcome: str


def apply_breaker_damage(state: object) -> bool:
    """Remove one heart only for a non-invulnerable breaker contact."""
    if state.invulnerable > 0.0:
        return False
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
    return True


def process_contacts(state: object, old_distance: float, old_y: float, slope: float) -> None:
    """Resolve swept pickup and obstacle contacts for one physics step."""
    for idx, thing in enumerate(state.level.things):
        if idx in state.seen or thing.distance < old_distance - 1.0:
            continue
        if thing.distance > state.distance + 0.65:
            break
        if old_distance > thing.distance + 0.45 or state.distance < thing.distance - 0.45:
            continue
        object_ground, _, _ = surf_geometry(state.level, thing.distance)
        travel = state.distance - old_distance
        crossing = 1.0
        if travel > 0.0:
            crossing = clamp01((thing.distance - old_distance) / travel)
        contact_y = old_y + (state.y - old_y) * crossing
        altitude = contact_y - object_ground - BOARD_CLEARANCE
        if thing.kind == "coin" and altitude < 2.4:
            state.seen.add(idx)
            state.coins += 1
            state.combo += 1
            state.score += 25 * (1 + state.combo // 8)
            state.events.append("coin")
            state.collisions.append(CollisionRecord(state.time, idx, thing.kind, thing.distance,
                                                    altitude, 0, "collected"))
        elif thing.kind == "boost":
            state.seen.add(idx)
            state.fuel += 32.0
            if state.fuel > 100.0:
                state.fuel = 100.0
            state.events.append("magnet")
            state.collisions.append(CollisionRecord(state.time, idx, thing.kind, thing.distance,
                                                    altitude, 0, "fuel"))
        elif thing.kind == "spring" and state.grounded:
            state.seen.add(idx)
            state.grounded = False
            state.vertical_velocity = slope * state.speed + 13.5
            state.action = "air"
            state.air_time = 0.0
            state.events.append("spring")
            state.collisions.append(CollisionRecord(state.time, idx, thing.kind, thing.distance,
                                                    altitude, 0, "launched"))
        elif thing.kind == "breaker" and altitude < 0.42:
            state.seen.add(idx)
            damaged = apply_breaker_damage(state)
            record = CollisionRecord(state.time, idx, thing.kind, thing.distance, altitude,
                                     -1 if damaged else 0,
                                     "damaged" if damaged else "invulnerable")
            state.collisions.append(record)
            state.last_collision = record

