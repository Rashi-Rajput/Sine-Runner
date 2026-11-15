import gymnasium as gym
import numpy as np
from game.surfer import Inputs, SurferState, update_surfer, final_score
from game.terrain import DT, DIFFICULTY_STAGES, generate_surf_level
from .observations import SIZE, observe

NAMES = ('coast', 'jump', 'slide', 'boost', 'brake', 'jump + boost', 'slide + brake')


def to_inputs(action, first=True):
    return Inputs(left=action in (4, 6), right=action in (3, 5),
                  jump=first and action in (1, 5), slide=action in (2, 6))


class WaveSurferEnv(gym.Env):
    metadata = {'render_modes': ['rgb_array'], 'render_fps': 20}

    def __init__(self, stage='full', difficulty=1.0, max_seconds=240.0,
                 action_repeat=3, render_mode=None):
        if stage not in DIFFICULTY_STAGES or not 0 <= difficulty <= 1:
            raise ValueError('Invalid terrain configuration')
        if max_seconds <= 0 or action_repeat < 1 or int(action_repeat) != action_repeat:
            raise ValueError('Invalid episode timing')
        if render_mode not in (None, 'rgb_array'):
            raise ValueError('Unsupported render mode')
        self.stage, self.difficulty = stage, difficulty
        self.max_seconds, self.action_repeat = max_seconds, action_repeat
        self.render_mode = render_mode
        self.action_space = gym.spaces.Discrete(7)
        self.observation_space = gym.spaces.Box(-1, 1, (SIZE,), np.float32)
        self.state = None
        self.done = True
        self.renderer = None
        self.last_seed = None

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        options = options or {}
        self.done = True
        self.rejected_seeds = []
        if 'level_seed' in options:
            level_seed = int(options['level_seed'])
            # Explicit evaluation/replay seeds must never silently change.
            level = generate_surf_level(0, level_seed, stage=self.stage, difficulty=self.difficulty)
        else:
            for _ in range(100):
                level_seed = int(self.np_random.integers(0, 1_000_000))
                if seed is None and level_seed == self.last_seed:
                    continue
                try:
                    level = generate_surf_level(0, level_seed, stage=self.stage, difficulty=self.difficulty)
                except ValueError as exc:
                    if not str(exc).startswith('unfair generated terrain:'):
                        raise
                    self.rejected_seeds.append(level_seed)
                    continue
                break
            else:
                raise RuntimeError('Could not generate a fair training course after 100 attempts')
        self.last_seed = level_seed
        self.state = SurferState(level)
        self.previous_action, self.total_reward, self.steps = 0, 0.0, 0
        self.done = False
        return observe(self.state, 0), self._info()

    def _info(self):
        s = self.state
        return dict(seed=s.level.seed, rejected_seeds=tuple(self.rejected_seeds),
                    success=bool(s.finished and not s.failed),
                    distance=s.distance, score=final_score(s), coins=s.coins,
                    crashes=s.crashes, hearts=s.hearts, seconds=s.time,
                    episode_return=self.total_reward, episode_length=self.steps)

    def step(self, action):
        if self.done:
            raise RuntimeError('Call reset() before stepping a completed episode')
        if not self.action_space.contains(action):
            raise ValueError('Action must be an integer from 0 through 6')
        reward = 0.0
        s = self.state
        for frame in range(self.action_repeat):
            before = s.distance, s.coins, s.hearts
            update_surfer(s, to_inputs(int(action), first=frame == 0))
            # Progress, pickups, damage, elapsed time, completion and failure.
            distance, coins, hearts = before
            reward += (0.02 * (s.distance - distance) + 0.5 * (s.coins - coins)
                       + 0.5 * sum(c.outcome == 'fuel' for c in s.collisions)
                       - 5 * (hearts - s.hearts) - 0.05 * DT
                       + 20 * (s.finished and not s.failed) - 20 * s.failed)
            terminated = bool(s.finished or s.failed or s.hearts <= 0)
            truncated = bool(not terminated and s.time + 1e-9 >= self.max_seconds)
            if terminated or truncated:
                break
        self.previous_action = int(action)
        self.steps += 1
        self.total_reward += reward
        self.done = terminated or truncated
        return observe(s, self.previous_action), float(reward), terminated, truncated, self._info()

    def render(self):
        if self.render_mode is None:
            return None
        if self.state is None:
            raise RuntimeError('Reset before rendering')
        if self.renderer is None:
            from game.renderer import Renderer, WaveField
            self.renderer = Renderer(960, 540, standalone=True)
            self.field = WaveField(128)
        pixels = self.renderer.render(self.state, self.field)
        return np.frombuffer(pixels, np.uint8).reshape(540, 960, 3)[::-1].copy()

    def close(self):
        if self.renderer is not None:
            self.renderer.ctx.release()
            self.renderer = None
