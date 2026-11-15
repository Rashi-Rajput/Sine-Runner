import argparse
import json
from pathlib import Path
from importlib.metadata import version
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.env_util import make_vec_env
from .env import WaveSurferEnv
from .callbacks import TrainingCallback
from .evaluate_rl import configuration


def main():
    p = argparse.ArgumentParser(description='Train PPO on Wave Surfer')
    p.add_argument('--timesteps', type=int, default=2_000_000)
    p.add_argument('--envs', type=int, default=8)
    p.add_argument('--seed', type=int, default=42)
    p.add_argument('--n-steps', type=int, default=1024)
    p.add_argument('--eval-every', type=int, default=50_000)
    p.add_argument('--output', type=Path, default=Path('rl/checkpoints'))
    p.add_argument('--resume', type=Path, help='Continue a checkpoint into a fresh output folder')
    p.add_argument('--gamma', type=float, default=None, help='Discount factor; new runs default to 0.995')
    args = p.parse_args()
    if min(args.timesteps, args.envs, args.n_steps, args.eval_every) <= 0 or args.envs * args.n_steps < 2:
        p.error('Positive counts and at least two rollout samples are required')
    if args.gamma is not None and not 0 < args.gamma < 1:
        p.error('--gamma must be between zero and one')
    if args.resume and args.gamma is not None:
        p.error('Use a fresh run to compare gamma; resume preserves the saved configuration')
    args.output.mkdir(parents=True, exist_ok=True)
    if (args.output / 'config.json').exists():
        p.error('Output already contains a run; choose a new --output directory')
    config = configuration(args.resume) if args.resume else dict(stage='full', difficulty=1.0, max_seconds=240)
    settings = dict(vars(args), output=str(args.output), resume=str(args.resume) if args.resume else None,
                    environment=config, versions={name: version(name) for name in ('torch', 'gymnasium', 'stable-baselines3')})
    (args.output / 'config.json').write_text(json.dumps(settings, indent=2))
    torch.set_num_threads(1)
    env = make_vec_env(WaveSurferEnv, n_envs=args.envs, seed=args.seed, env_kwargs=config)
    try:
        model = PPO.load(args.resume, env=env, device='cpu') if args.resume else PPO('MlpPolicy', env, seed=args.seed, device='cpu', n_steps=args.n_steps,
                    batch_size=min(64, args.n_steps * args.envs), ent_coef=0.01,
                    gamma=args.gamma if args.gamma is not None else 0.995,
                    tensorboard_log=str(args.output / 'runs'), verbose=1)
        model.set_random_seed(args.seed)
        settings['ppo'] = dict(gamma=model.gamma, n_steps=model.n_steps, batch_size=model.batch_size,
                               ent_coef=model.ent_coef, gae_lambda=model.gae_lambda)
        (args.output / 'config.json').write_text(json.dumps(settings, indent=2))
        model.tensorboard_log = str(args.output / 'runs')
        model.learn(args.timesteps, reset_num_timesteps=not bool(args.resume),
                    callback=TrainingCallback(args.output, args.eval_every, config))
        model.save(args.output / 'final_model')
    finally:
        env.close()


if __name__ == '__main__':
    main()
