import math
from pathlib import Path

import numpy as np
import pytest

import game


def test_sine_linter_and_injected_banned_call():
    assert game.lint_game() == []
    source = "def bad(x):\n    return " + "abs(x)\n"
    errors = game.sine_lint_source(source)
    assert errors and "banned call" in errors[0]
    assert game.sine_lint_source("import numpy as np\ndef bad():\n    return np.random.default_rng()\n")
    assert game.sine_lint_source("def bad(a, b):\n    return min(a, b)\n")
    glsl = 'BAD_SHADER = "void main(){ float x=' + 'pow(2.0,3.0); }"\n'
    assert any("GLSL" in error for error in game.sine_lint_source(glsl))


def test_sine_hash_determinism_range_and_distribution():
    a = [game.sine_hash(i, 3.0) for i in range(2000)]
    b = [game.sine_hash(i, 3.0) for i in range(2000)]
    assert a == b
    assert all(0.0 <= x < 1.0 for x in a)
    assert 0.47 < sum(a) / len(a) < 0.53


def test_lane_easing_endpoints_and_jump_ground_return():
    level = game.generate_level(0)
    state = game.GameState(level, lane=1, lane_from=0.0, lane_phase=0.0)
    assert state.lane_x == pytest.approx(0.0)
    state.lane_phase = 1.0
    assert state.lane_x == pytest.approx(game.LANE_WIDTH)
    state.action = "jump"
    for _ in range(60):
        game.update_game(state, game.Inputs())
    assert state.action == "run"
    assert state.height == 0.0


def test_quarter_sine_perspective_lookup_is_monotone():
    lookup = game.PerspectiveLookup(257)
    values = [lookup.project(i / 100.0) for i in range(101)]
    assert values[0] == pytest.approx(0.0)
    assert values[-1] == pytest.approx(1.0)
    assert all(a <= b for a, b in zip(values, values[1:]))


def test_moving_obstacle_sways_inside_its_lane():
    thing = game.Thing(4, 0, "moving", phase=0.3)
    positions = [game.moving_lane_position(thing, i * 0.1) for i in range(80)]
    assert all(-0.321 <= position <= 0.321 for position in positions)
    assert max(positions) - min(positions) > 0.6


def test_lane_change_consumes_fixed_world_distance():
    state = game.GameState(game.generate_level(0), lane=1, lane_from=0.0, lane_phase=0.0)
    start = state.distance
    while state.lane_phase < 1.0:
        game.update_game(state, game.Inputs())
    travelled = state.distance - start
    assert travelled == pytest.approx(game.LANE_CHANGE_DISTANCE, abs=0.3)


@pytest.mark.parametrize("kind,action,hit", [
    ("low", "run", True), ("low", "jump", False),
    ("high", "run", True), ("high", "slide", False),
    ("block", "run", True), ("moving", "run", True),
])
def test_collision_types(kind, action, hit):
    phase = -0.0135 if kind == "moving" else 0.0
    level = game.LevelData(0, 1, 100.0, 4.0, [game.Thing(1, 0, kind, phase=phase)])
    state = game.GameState(level, distance=3.95, action=action)
    game.update_game(state, game.Inputs(), 0.01)
    assert (state.hearts == 2) is hit


def test_shield_absorbs_crash_and_magnet_collects():
    things = [game.Thing(1, 0, "block"), game.Thing(1, 1, "coin", 2)]
    level = game.LevelData(0, 1, 100.0, 4.0, things)
    state = game.GameState(level, distance=3.95, shield=True, magnet=2.0)
    game.update_game(state, game.Inputs(), 0.01)
    assert state.hearts == 3 and not state.shield and state.coins == 2


def test_spawner_reproducibility_for_the_single_track():
    a, b = game.generate_level(0), game.generate_level(0)
    assert a.things == b.things
    assert a.optimal_coins == b.optimal_coins
    assert len(game.LEVELS) == 1


def test_every_shipped_seed_survivable():
    for i in range(len(game.LEVELS)):
        level = game.generate_level(i)
        ok, optimum, path = game.solve_dp(level.things, int(level.length / level.slice_m))
        assert ok and optimum == level.optimal_coins and path


def test_dp_rejects_unwinnable_wall():
    wall = [game.Thing(1, lane, "wall") for lane in (-1, 0, 1)]
    assert not game.solve_dp(wall, 4)[0]


def test_dp_optimum_beats_1000_deterministic_policy_rollouts():
    level = game.generate_level(0)
    slices = int(level.length / level.slice_m)
    by_z = [[] for _ in range(slices + 1)]
    for item in level.things:
        if item.z <= slices:
            by_z[item.z].append(item)
    best = 0
    for policy in range(1000):
        lane, coins = 0, 0
        for z in range(slices + 1):
            choice = int(math.floor(game.sine_hash(policy, z + 1) * 3.0)) - 1
            if choice < lane: lane -= 1
            elif choice > lane: lane += 1
            blocked = any(item.lane == lane and game.obstacle_blocks(item.kind, "run") for item in by_z[z])
            if blocked:
                break
            coins += sum(item.value for item in by_z[z] if item.kind == "coin" and item.lane == lane)
        if coins > best: best = coins
    assert level.optimal_coins >= best


def test_track_unit_frame_and_exact_curvature():
    track = game.SineTrack(game.level_length(0), 0)
    for s in np.linspace(1.0, track.length - 1.0, 30):
        tangent, normal = track.frame(float(s))
        assert tangent[0] ** 2 + tangent[1] ** 2 == pytest.approx(1.0, abs=1e-12)
        assert tangent[0] * normal[0] + tangent[1] * normal[1] == pytest.approx(0.0, abs=1e-12)
        ds = 1e-4
        numerical = (track.theta(s + ds) - track.theta(s - ds)) / (2.0 * ds)
        assert track.curvature(s) == pytest.approx(numerical, rel=1e-7, abs=1e-9)


def _integrator_error(kind, dt):
    x, v, t = 1.0, 0.0, 0.0
    while t + dt <= 1.0 + 1e-12:
        x, v = game.oscillator_step(kind, x, v, dt, 4.0, 0.0)
        t += dt
    return abs(x - math.cos(4.0)) + abs(v + 4.0 * math.sin(4.0))


def test_integrator_orders_and_energy_behavior():
    expected = {"euler": 1.0, "symplectic": 1.0, "verlet": 2.0, "rk4": 4.0}
    dts = np.array((0.04, 0.02, 0.01, 0.005))
    for kind, target in expected.items():
        errors = np.array([_integrator_error(kind, float(dt)) for dt in dts])
        order = np.polyfit(np.log(dts), np.log(errors), 1)[0]
        assert order == pytest.approx(target, abs=0.24)
    results = {}
    for kind in ("euler", "verlet"):
        x, v = 1.0, 0.0
        energies = []
        for _ in range(10000):
            x, v = game.oscillator_step(kind, x, v, 0.005, 4.0, 0.0)
            energies.append(0.5 * (v * v + 16.0 * x * x))
        results[kind] = energies
    assert results["euler"][-1] > results["euler"][0] * 2.0
    assert max(results["verlet"]) - min(results["verlet"]) < 0.02


def test_wave_energy_standing_mode_and_cfl():
    with pytest.raises(ValueError):
        game.WaveField(32, ratio=0.8)
    size, ratio = 48, 0.35
    field = game.WaveField(size, ratio, damping=0.0)
    grid = np.arange(size)
    mode = np.sin(math.pi * 2 * grid / (size - 1))[:, None] * np.sin(math.pi * 3 * grid / (size - 1))[None, :]
    eigen = 4.0 * math.sin(math.pi * 2 / (2 * (size - 1))) ** 2 + 4.0 * math.sin(math.pi * 3 / (2 * (size - 1))) ** 2
    omega = math.acos(1.0 - 0.5 * ratio * ratio * eigen)
    field.u = mode.copy()
    field.old = mode * math.cos(omega)
    energies = []
    for step in range(160):
        field.step()
        energies.append(field.energy())
        projection = float(np.sum(field.u * mode) / np.sum(mode * mode))
        assert projection == pytest.approx(math.cos((step + 1) * omega), abs=2e-9)
    mean_e = sum(energies) / len(energies)
    assert (max(energies) - min(energies)) / mean_e < 0.18


def test_scoring_stars_and_persistence(tmp_path):
    assert game.stars_for(False, 100, 100, 0) == 0
    assert game.stars_for(True, 69, 100, 0) == 1
    assert game.stars_for(True, 70, 100, 1) == 2
    assert game.stars_for(True, 95, 100, 0) == 3
    path = tmp_path / "scores.json"
    store = game.ScoreStore(path)
    assert store.record(0, 1000, 60.0, 2)
    assert not store.record(0, 900, 58.0, 1)
    loaded = game.ScoreStore(path)
    assert loaded.data["1"] == {"score": 1000, "time": 58.0, "stars": 2}


def test_audio_frequency_and_click_free_envelope():
    audio = game.synth_audio(2.0, 0)
    spectrum = np.abs(np.fft.rfft(audio * np.sin(math.pi * np.arange(len(audio)) / (len(audio) - 1)) ** 2))
    freqs = np.fft.rfftfreq(len(audio), 1 / 44100.0)
    dominant = freqs[int(np.argmax(spectrum))]
    assert 50.0 < dominant < 500.0
    env = game.half_sine_envelope(np.linspace(0.0, 1.0, 1001), 1.0)
    assert env[0] == pytest.approx(0.0, abs=1e-15)
    assert env[-1] == pytest.approx(0.0, abs=1e-15)
    assert np.max(np.abs(np.diff(env))) < 0.004


def test_logic_hash_determinism():
    hashes = []
    for _ in range(2):
        state = game.GameState(game.generate_level(0, 42))
        for frame in range(1000):
            game.update_game(state, game.scripted_input(frame, state))
        hashes.append(game.logic_hash(state))
    assert hashes[0] == hashes[1]


def test_wave_surfer_analytic_derivatives_match_finite_differences():
    level = game.generate_surf_level(0)
    ds = 1e-3
    for s in np.linspace(10.0, level.length - 10.0, 40):
        height, slope, curvature = game.wave_geometry(float(s), level.index, level.length)
        left = game.wave_geometry(float(s - ds), level.index, level.length)[0]
        right = game.wave_geometry(float(s + ds), level.index, level.length)[0]
        numerical_slope = (right - left) / (2.0 * ds)
        numerical_curve = (right - 2.0 * height + left) / (ds * ds)
        assert slope == pytest.approx(numerical_slope, rel=2e-6, abs=8e-8)
        assert curvature == pytest.approx(numerical_curve, rel=2e-4, abs=2e-7)


def test_wave_surfer_ballistic_jump_and_determinism():
    hashes = []
    for _ in range(2):
        state = game.SurferState(game.generate_surf_level(0, 42))
        start_y = state.y
        game.update_surfer(state, game.Inputs(jump=True))
        assert not state.grounded and state.y > start_y
        for frame in range(1, 600):
            game.update_surfer(state, game.scripted_surf_input(frame, state))
        hashes.append(game.surf_logic_hash(state))
    assert hashes[0] == hashes[1]


def test_surfer_hearts_only_change_on_breaker_collision():
    empty = game.SurfLevel(0, 1, 200.0, [], 0)
    state = game.SurferState(empty)
    state.y += 5.0
    state.grounded = False
    state.vertical_velocity = -18.0
    for _ in range(120):
        game.update_surfer(state, game.Inputs())
    assert state.hearts == 3

    obstacle = game.SurfLevel(0, 1, 200.0, [game.SurfThing(0.15, "breaker")], 0)
    state = game.SurferState(obstacle)
    game.update_surfer(state, game.Inputs(), 0.02)
    assert state.hearts == 2
    assert state.crashes == 1


def test_spring_is_high_jump_and_only_one_track_is_shipped():
    spring_level = game.SurfLevel(0, 1, 200.0, [game.SurfThing(0.15, "spring")], 0)
    state = game.SurferState(spring_level)
    game.update_surfer(state, game.Inputs(), 0.02)
    assert not state.grounded
    assert state.vertical_velocity > game.SURF_JUMP_SPEED * 1.4
    assert "spring" in state.events

    assert len(game.LEVELS) == 1
    with pytest.raises(ValueError):
        game.generate_surf_level(1)
