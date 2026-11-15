import argparse
import pygame
from stable_baselines3 import PPO
from .env import WaveSurferEnv, NAMES
from .evaluate_rl import configuration


def main():
    p = argparse.ArgumentParser(description='Watch a policy; run TensorBoard alongside for training graphs')
    p.add_argument('--model', required=True)
    p.add_argument('--seed', type=int, default=2_000_000)
    p.add_argument('--episodes', type=int, default=1)
    p.add_argument('--follow', action='store_true', help='Reload checkpoint between episodes')
    args = p.parse_args()
    env = WaveSurferEnv(render_mode='rgb_array', **configuration(args.model))
    pygame.init()
    screen = pygame.display.set_mode((960, 540))
    font, clock = pygame.font.Font(None, 24), pygame.time.Clock()
    try:
        model = PPO.load(args.model, device='cpu')
        for episode in range(args.episodes):
            if args.follow and episode:
                try:
                    model = PPO.load(args.model, device='cpu')
                except (OSError, ValueError):
                    pass  # A writer may currently be replacing the checkpoint.
            obs, _ = env.reset(options={'level_seed': args.seed + episode})
            while True:
                if any(e.type == pygame.QUIT or (e.type == pygame.KEYDOWN and e.key == pygame.K_ESCAPE) for e in pygame.event.get()):
                    return
                action, _ = model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(int(action))
                screen.blit(pygame.surfarray.make_surface(env.render().transpose(1, 0, 2)), (0, 0))
                label = f"Seed {info['seed']} | {NAMES[int(action)]} | reward {reward:.2f} | return {info['episode_return']:.1f} | score {info['score']} | hearts {info['hearts']}"
                screen.blit(font.render(label, True, 'white', 'black'), (10, 10))
                pygame.display.flip()
                clock.tick(20)
                if terminated or truncated:
                    print(info)
                    break
    finally:
        env.close()
        pygame.quit()


if __name__ == '__main__':
    main()
