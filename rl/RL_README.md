# PPO agent

The agent uses Stable-Baselines3 PPO with an MLP and a Gymnasium wrapper around the existing game. Run commands from the repository root.

## Install and train

```powershell
python -m pip install -r rl/requirements-rl.txt
python -m rl.train_rl --timesteps 5000000 --envs 8 --seed 42 --gamma 0.995 --output rl/checkpoints-new
```

Use a fresh output folder for each run. The existing measured run is in `rl/checkpoints-g995-42`. Eight environments collect synchronous vectorized rollouts, not eight parallel processes. Training is headless. Configuration records seeds, environment settings, PPO settings and library versions.

## Watch and inspect the measured model

```powershell
python -m rl.play_agent --model rl/checkpoints-g995-42/final_model.zip --seed 42 --episodes 10
tensorboard --logdir rl/checkpoints-g995-42/runs
python -m rl.evaluate_rl --model rl/checkpoints-g995-42/final_model.zip --compare --json rl/results/g995-final.json
python -m rl.plot_results
python -m pytest -q tests rl/tests
```

The published 91% result belongs to `final_model.zip`, not `best_model.zip`. Best checkpoints are selected by validation completion rate, then mean reward. An initial validation saves a checkpoint before learning; later evaluations occur every 50,000 steps. `best_score.json` records the selected step and score, and `evaluations.jsonl` records validation history. Test results do not select checkpoints.

Playback can follow a running experiment using `--follow`; it reloads saved weights between episodes. Wait until its checkpoint exists. Playback does not train. TensorBoard records episode reward, completion, distance, score, coins, crashes, hearts and PPO diagnostics. Loss alone does not measure playing skill.

## Continue training

```powershell
python -m rl.train_rl --resume rl/checkpoints-g995-42/final_model.zip --timesteps 1000000 --seed 43 --output rl/checkpoints-continued
```

This adds one million steps and retains the checkpoint's optimizer and PPO settings. RNG sequences and current episodes restart; continuation is not exact interrupted-run replay. New runs default to gamma 0.995; use `--gamma 0.99` for a separate experiment. Gamma cannot be overridden during resume. No curriculum or larger network is implemented.

## Environment and evaluation

Seven actions: coast, jump, slide, boost, brake, jump+boost, slide+brake. Each action advances three fixed 60 Hz physics frames; jump is pulsed on the first frame. Death and completion terminate an episode; 240 seconds truncates it.

`observations.py` supplies 40 clipped float32 values: player telemetry, local terrain geometry up to 28 units ahead, nearby object presence/distances and the previous action. This is structured state input, not pixel learning. Hidden randomized physics and coarse observations make the task partially observable.

Training samples course seeds below 1,000,000 and retries unfair terrain up to 100 times. Validation uses 1,000,000-1,000,049. The saved benchmark uses 3,000,000-3,000,099. Evaluation reports and excludes rejected courses identically for all policies. Explicit viewer seeds are not silently substituted.

Rewards in `env.py`: +0.02 per distance unit, +0.5 per coin/fuel pickup, +20 finish, -5 per heart lost, -20 failure and -0.05 per second. Fuel regeneration does not earn pickup rewards.

## Code and evidence

- `env.py` and `observations.py`: environment, controls, rewards and state features.
- `train_rl.py` and `callbacks.py`: PPO training, logs, validation and checkpoints.
- `evaluate_rl.py`: PPO/random/scripted evaluation on identical courses.
- `play_agent.py`: rendered policy playback.
- `plot_results.py`: chart from saved evaluation JSON.
- `tests/test_rl.py`: environment contract, physics parity, episode endings, pickup rewards, terrain rejection and evaluation repeatability.

[Measured results](results/README.md): PPO completed 91/100 courses, scripted 71/100, random 26/100. This is one trained policy, not a mean across independent training seeds. The gamma and training budget both changed from the earlier experiment, so the improvement cannot be attributed to gamma alone. No multi-run standard deviation has been measured.

Weights and TensorBoard logs are ignored by Git; the JSON results and chart can be committed. The measured checkpoint must be supplied separately for exact reproduction. A fresh run may produce different results.
