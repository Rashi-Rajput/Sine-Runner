"""Small checks for the environment contracts most likely to break training."""
import numpy as np
import pytest
from gymnasium.utils.env_checker import check_env
from game.surfer import SurferState, update_surfer, surf_logic_hash
from game.terrain import generate_surf_level, SurfThing
from rl.env import WaveSurferEnv, to_inputs
from rl.evaluate_rl import evaluate


def test_contract_and_physics():
    check_env(WaveSurferEnv(), skip_render_check=True)
    env = WaveSurferEnv()
    first, info = env.reset(seed=42)
    np.testing.assert_array_equal(first, env.reset(seed=42)[0])
    direct = SurferState(generate_surf_level(0, info['seed']))
    for action in range(7):
        obs, reward, _, _, _ = env.step(action)
        for frame in range(3):
            update_surfer(direct, to_inputs(action, frame == 0))
        assert surf_logic_hash(env.state) == surf_logic_hash(direct)
        assert env.observation_space.contains(obs) and np.isfinite(reward)


def test_episode_endings_and_pickup_reward():
    env = WaveSurferEnv()
    env.reset(seed=42)
    env.state.hearts = 1
    env.state.level.things = [SurfThing(0.15, 'breaker')]
    assert env.step(0)[2]
    with pytest.raises(RuntimeError):
        env.step(0)
    env.reset(seed=42)
    env.state.distance = env.state.level.length - 0.01
    assert env.step(0)[2]
    env.reset(seed=42)
    env.state.level.things = [SurfThing(0.15, 'boost')]
    assert env.step(0)[1] > 0.49  # Pickup counted once across repeated frames.
    short = WaveSurferEnv(max_seconds=0.01)
    short.reset(seed=42)
    assert short.step(0)[2:4] == (False, True)


def test_fairness_retry(monkeypatch):
    import rl.env as module
    calls = []

    def generate(index, seed, **kwargs):
        calls.append(seed)
        if len(calls) == 1:
            raise ValueError('unfair generated terrain: test rejection')
        return generate_surf_level(index, seed, **kwargs)

    monkeypatch.setattr(module, 'generate_surf_level', generate)
    _, info = WaveSurferEnv().reset(seed=42)
    assert info['rejected_seeds'] == (calls[0],)
    assert info['seed'] == calls[-1]
    with pytest.raises(ValueError, match='unfair generated terrain'):
        WaveSurferEnv().reset(options={'level_seed': 677})


def test_evaluation_reproducibility():
    kwargs = dict(model=None, seeds=[3_000_000, 3_000_001], policy='random', max_seconds=0.1)
    assert evaluate(**kwargs) == evaluate(**kwargs)
