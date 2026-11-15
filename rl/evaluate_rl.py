import argparse
import json
from pathlib import Path
from stable_baselines3 import PPO
import numpy as np
from .env import WaveSurferEnv
from game.surfer import scripted_surf_input

VALIDATION_SEEDS = tuple(range(1_000_000, 1_000_050))
TEST_SEEDS = tuple(range(3_000_000, 3_000_100))


def evaluate(model, seeds=TEST_SEEDS, policy='ppo', **env_kwargs):
    if policy not in ('ppo', 'random', 'scripted'):
        raise ValueError('Unknown evaluation policy')
    episodes = []
    skipped = []
    env = WaveSurferEnv(**env_kwargs)
    try:
        for seed in seeds:
            try:
                obs, _ = env.reset(options={'level_seed': int(seed)})
            except ValueError as exc:
                if not str(exc).startswith('unfair generated terrain:'):
                    raise
                skipped.append(int(seed))
                continue
            rng = np.random.default_rng(int(seed))
            while True:
                if policy == 'random':
                    action = rng.integers(7)
                elif policy == 'scripted':
                    controls = scripted_surf_input(0, env.state)
                    action = 5 if controls.jump and controls.right else 1 if controls.jump else 3 if controls.right else 0
                else:
                    action, _ = model.predict(obs, deterministic=True)
                obs, _, terminated, truncated, info = env.step(int(action))
                if terminated or truncated:
                    episodes.append(dict(info, truncated=truncated))
                    break
    finally:
        env.close()
    if not episodes:
        raise ValueError('No valid evaluation courses')
    summary = {key: {'mean': float(np.mean([e[key] for e in episodes])),
                     'std': float(np.std([e[key] for e in episodes]))}
               for key in ('episode_return', 'distance', 'score', 'coins', 'crashes', 'hearts', 'episode_length', 'seconds')}
    summary['completion_rate'] = float(np.mean([e['success'] for e in episodes]))
    summary['episodes'] = len(episodes)
    return {'summary': summary, 'episodes': episodes, 'skipped_seeds': skipped}


def configuration(model_path):
    path = Path(model_path).parent / 'config.json'
    if not path.exists():
        raise FileNotFoundError(f'Model configuration missing: {path}')
    return json.loads(path.read_text())['environment']


def main():
    p = argparse.ArgumentParser()
    p.add_argument('--model', required=True)
    p.add_argument('--json', type=Path)
    p.add_argument('--compare', action='store_true', help='Also evaluate random and scripted policies')
    args = p.parse_args()
    model = PPO.load(args.model, device='cpu')
    config = configuration(args.model)
    policies = ('ppo', 'random', 'scripted') if args.compare else ('ppo',)
    result = {policy: evaluate(model, policy=policy, **config) for policy in policies}
    print(json.dumps({key: value['summary'] for key, value in result.items()}, indent=2))
    if args.json:
        args.json.parent.mkdir(parents=True, exist_ok=True)
        args.json.write_text(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
