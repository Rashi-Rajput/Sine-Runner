import numpy as np
from game.terrain import surf_geometry, BOARD_CLEARANCE

SIZE = 40


def observe(state, previous_action):
    ground, slope, curve = surf_geometry(state.level, state.distance)
    values = [state.distance / state.level.length, state.speed / 30,
              state.vertical_velocity / 30, state.fuel / 100, state.hearts / 3,
              float(state.grounded), state.air_time / 5, slope / 3, curve / 3,
              (state.y - ground - BOARD_CLEARANCE) / 10,
              max(0, state.invulnerable) / 1.35, state.steep_time,
              state.combo / 100]
    for offset in (2, 4, 8, 12, 20, 28):
        _, slope, curve = surf_geometry(state.level, min(state.level.length, state.distance + offset))
        values.extend((slope / 3, curve / 3))
    for kind in ('coin', 'boost', 'spring', 'breaker'):
        nearby = [obj.distance - state.distance for i, obj in enumerate(state.level.things)
                  if i not in state.seen and obj.kind == kind and 0 <= obj.distance - state.distance <= 28]
        values.extend((float(bool(nearby)), min(nearby) / 28 if nearby else 1.0))
    values.extend(float(previous_action == i) for i in range(7))
    return np.clip(np.asarray(values, dtype=np.float32), -1, 1)
