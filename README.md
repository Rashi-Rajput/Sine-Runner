# Wave Surfer

Wave Surfer is a small deterministic riding game written in Python. The player surfs over a moving mathematical landscape, jumps over red breakers, collects gold rings, grabs fuel, and uses green springs for high jumps.

The important design choice is that the game is generated from equations instead of imported art. The ground, rider, UI, sound, and decorative ripples are all made from sine functions and simple geometry.

## Run it

```bash
python -m pip install -r requirements.txt
python game.py
```

Without `--seed`, every launch creates a new terrain waveform and a new object layout. The chosen seed is printed so the exact course can be replayed later:

```bash
python game.py --seed 12345
```

Restarting with `R` keeps the current seed and course, so a retry is fair. Relaunch without `--seed` for another terrain.

Useful checks:

```bash
pytest -q
python game.py --headless --seed 42 --frames 1000 --out out/
python game.py --benchmark --seed 42 --frames 1000
python game.py --report --out out/report/
```

## Reinforcement-learning player

The PPO agent lives entirely in `rl/` and wraps the unchanged game engine. See [the RL guide](rl/RL_README.md) for training, evaluation, live playback, and TensorBoard graphs.

```bash
python -m pip install -r rl/requirements-rl.txt
python -m rl.train_rl --timesteps 5000000 --envs 8 --seed 42 --gamma 0.995 --output rl/checkpoints-new
python -m rl.evaluate_rl --model rl/checkpoints-g995-42/final_model.zip
python -m rl.play_agent --model rl/checkpoints-g995-42/final_model.zip --seed 42
tensorboard --logdir rl/checkpoints-g995-42/runs
```

For a reproducible comparison against random actions and the scripted controller:

```bash
python -m rl.evaluate_rl --model rl/checkpoints-g995-42/final_model.zip --compare --json rl/results/g995-final.json
python -m pytest -q tests rl/tests
```

The RL guide includes the benchmark protocol, measured results, limitations, and checkpoint resume command. The policy uses structured game telemetry rather than images. On [100 unseen courses](rl/results/README.md), the final PPO model completed 91%, versus 26% for random actions and 71% for the scripted controller. This is a single-run result; repeat it with additional training seeds before claiming a stable average.

Headless mode runs the same fixed-step game loop without opening a window. It writes `wave_frameXXXXXX.png` captures and prints a SHA-256 logic hash. If an OpenGL context cannot be created, it announces that fact and writes a simple Pillow diagnostic frame instead.

## Controls

- `W`, `Up`, or `Space`: jump
- `Right` or `D`: hold boost
- `Left` or `A`: brake
- `Down` or `S`: slide
- `P`: pause
- `H`: show or hide help
- `R`: restart
- `Esc`: quit

The game has one course style and one coherent visual and musical theme, but its terrain waveform and object layout are regenerated for each unseeded launch.

## How the game is organised

The game is a Python package split by responsibility:

1. `game/terrain.py` owns sine math, seeded terrain, domain randomization, and fairness validation.
2. `game/surfer.py` owns inputs, rider state, fixed-step movement, scoring, and logic hashes.
3. `game/collision.py` owns swept contacts, collision records, pickups, springs, and breaker damage.
4. `game/renderer.py` owns the float32 wave field, shaders, and ModernGL renderer.
5. `game/audio.py` owns procedural music, sound effects, and playback.
6. `game/main.py` owns menus, UI, headless mode, benchmarks, reports, and the CLI.
7. `tests/` contains game tests; `rl/tests/` checks the RL environment and evaluation.

The root `game.py` is only a compatibility launcher, so existing `python game.py` commands still work. The older lane-runner API is isolated in `game/legacy.py`; the executable, renderer, human game, and RL player use only `SurferState` and `update_surfer`.

## Terrain difficulty, fairness, and domain randomization

Every generated course carries a `TerrainDomain`. It contains the seeded amplitude, frequency, fine-detail amount, object spacing, friction, boost strength, and gravity. Supplying the same seed, stage, and difficulty reproduces the same values exactly.

Five difficulty stages are available:

1. `smooth`: broad low waves with almost no hazards
2. `moderate`: taller and faster terrain
3. `compound`: layered harmonics and finer surface detail
4. `obstacles`: wider hazard variety
5. `full`: all terrain and object variation

`validate_surf_level()` samples the analytic terrain before play. It reports peak squared slope, peak squared curvature, closest breaker spacing, and reasons for rejection. Breakers are only placed on safe sections with enough reaction distance; an unsafe planned breaker is replaced by a coin. This keeps random maps varied without making them arbitrarily unfair.

## 1. The terrain: a layered sine wave

The ground height is based on several sine waves added together:

```text
F(phi) = sin(phi)
       + a2 sin(r2 phi + p2)
       + a3 sin(r3 phi + p3)

y(s) = A(s) F(phi(s))
```

The run seed deterministically chooses `a2`, `a3`, `r2`, `r3`, the phase offsets, the starting amplitude, and the frequency range. This makes each new run a genuinely different layered wave rather than merely moving the obstacles.

`s` is distance along the course. `A(s)` grows with progress, so the hills become taller. The main phase is a chirp:

```text
phi(s) = p0 + k0 s + 1/2 chirp s²
```

Because the phase contains `s²`, its derivative increases as the rider travels. The hills gradually become closer together instead of repeating with one fixed wavelength.

### Why use several waves?

A single sine wave looks artificial. Adding harmonics gives the surface broad hills, smaller shoulders, and fine detail while keeping the result smooth and analytically differentiable.

### Analytic slope and curvature

`wave_geometry()` returns height, slope, and curvature. The derivatives are calculated from the chain rule:

```text
y'  = A'F + A F' phi'
y'' = 2A'F'phi' + A(F''phi'² + F'phi'')
```

This matters because the same mathematics controls several parts of the game:

- The board follows the terrain slope.
- The rider pitch follows the tangent of the wave.
- Jump take-off inherits the slope velocity.
- Curvature can be used to identify difficult sections.
- The shader can reproduce the same shape visually.

The geometry is evaluated directly rather than sampled from an image or a noisy heightmap.

## 2. Fixed-step physics

The simulation advances at exactly 60 Hz:

```python
DT = 1.0 / 60.0
```

Rendering and simulation are separate. The interactive loop accumulates real time and performs as many `DT` updates as necessary. This prevents physics from changing when the monitor refresh rate changes.

The surfer uses ordinary equations of motion while airborne:

```text
y(t + dt)  = y(t) + vy dt + 1/2 g dt²
vy(t + dt) = vy(t) + g dt
```

When grounded, the board is placed on the current wave height and its vertical velocity is tied to:

```text
ground_velocity = slope * forward_speed
```

This is why the rider follows a hill naturally instead of hovering over a flat invisible track.

### Collision severity without square root

When landing, the code compares the rider's vertical velocity with the ground's vertical velocity. The normal impact is measured as:

```text
impact² = (vy - slope * speed)² / (1 + slope²)
```

Only the square of the impact is needed for a threshold test, so no square root is necessary. Large landings cause a stumble; only an actual breaker contact removes a heart.

## 3. Springs, boost, and collisions

The gameplay objects have different jobs:

- **Gold rings** increase coins and combo score.
- **Blue fuel orbs** refill the boost meter.
- **Green springs** apply a strong upward impulse for a high jump.
- **Red breakers** are the only objects that can cause a wipeout and remove a heart.

Objects are checked using the distance crossed during the current fixed step. This is important: checking only the current position could skip a thin obstacle when the rider moves quickly. The code checks whether the interval from the old distance to the new distance overlaps the object's position.

Each contact creates a `CollisionRecord` containing the object type, object index, contact altitude, outcome, and exact heart change. Coins, fuel, and springs always have a heart delta of zero. Only a damaging breaker record can have `heart_delta = -1`. This makes collision bugs visible to tests, the UI, and the RL reward system.

## 4. Deterministic placement and the sine hash

The game terrain generator uses a sine hash rather than Python's `random` module. The RL libraries use their own seeded random number generators:

```python
def sine_hash(n, k):
    return fract(43758.5453 * math.sin(12.9898 * n + 78.233 * k))
```

The sine produces a repeatable but irregular-looking value. `fract()` keeps the fractional part in the range `[0, 1)`.

The same seed and object index therefore always produce the same level. This is useful for:

- Replays.
- Regression tests.
- Comparing two physics changes.
- Reproducing a bug exactly.

## 5. Lane easing in the retained lane model

The older lane-runner code uses a raised-cosine interpolation:

```python
blend = 0.5 - 0.5 * math.cos(math.pi * p)
position = lane_from + (target - lane_from) * blend
```

At `p = 0`, `blend = 0`, so the move starts at the current lane.

At `p = 0.5`, `blend = 0.5`, so the rider is halfway across.

At `p = 1`, `blend = 1`, so the rider reaches the target lane.

The derivative is zero at both ends. That means the lane change starts and finishes gently instead of snapping to a new velocity.

The commonly suggested expression `sin(pi * p / 2)` is also smooth, but it is an ease-out curve: it moves quickly at the beginning and slows near the end. The raised-cosine version is symmetric and gives smooth acceleration and deceleration.

## 6. Weyl coin distribution in the retained lane model

The older generator uses:

```python
PHI_WEYL = 0.6180339887498949
lane = int(math.floor(
    fract(z * PHI_WEYL + sine_hash(seed, 6.0)) * 3.0
)) - 1
```

`PHI_WEYL` is the golden-ratio conjugate, `1 / phi`.

The sequence `fract(n * PHI_WEYL)` is equidistributed: it visits the interval `[0, 1)` without the obvious clumps that a small repeating pattern would create. That makes it a low-discrepancy sequence, which is useful when spreading coins across three lanes.

This is not random in the usual sense. It is deliberately structured, repeatable, and relatively even.

## 7. The wave-field simulation and the CFL constant

`WaveField` is a small two-dimensional leapfrog wave solver used only for decorative ripples. It uses float32 arrays and rotates preallocated current, previous, next, Laplacian, and scratch buffers instead of creating full grids every frame. It then applies a discrete Laplacian:

```text
u(next) = 2u(current) - u(previous)
          + ratio² * laplacian(u)
          - damping * (u(current) - u(previous))
```

For this 2D scheme the stability limit is:

```text
1 / sqrt(2) = 0.7071067811865476
```

The code uses the literal constant because `sqrt()` is forbidden by the project's sine-only rule:

```python
if ratio > 0.7071067811865476:
    raise ValueError("CFL condition violated")
```

If the ratio is too large, numerical energy grows without bound and the ripple explodes. The test `test_wave_energy_standing_mode_and_cfl` deliberately passes `0.8` and checks that the constructor rejects it.

The ripple field is visual only. It does not control the rider or collision logic.

## 8. Procedural rendering

The renderer uses a fullscreen OpenGL triangle and a fragment shader. For each pixel, the shader calculates colors and shapes from equations.

It draws:

- Layered sine ridges in the background.
- A warm sky and sun.
- The dark wave surface.
- Coins, fuel, springs, and breakers.
- The rider and board using cosine-shaped implicit forms.
- Decorative motion and lighting using sine modulation.

This approach avoids large texture assets and keeps the scene resolution-independent. The CPU sends state and object data; the GPU performs the per-pixel work.

The human interactive path does not read the framebuffer back to the CPU every frame. Game headless mode reads pixels for captures; the RL viewer reads rendered pixels each frame for its Pygame window.

The rider is also procedural. Sine-driven poses react to terrain pitch, vertical velocity, airtime, boost, sliding, landing compression, and collision tumble. No sprite sheet is needed, and the animation remains aligned with the same authoritative physics state used by collision detection.

## 9. Procedural audio

The music is additive synthesis: several sine oscillators are summed together.

The melody and bass use just-intonation ratios such as:

```python
ratios = (1.0, 5.0 / 4.0, 3.0 / 2.0, 2.0)
```

These mean:

- `1.0`: root or unison
- `5/4`: major third
- `3/2`: perfect fifth
- `2.0`: octave

These simple ratios produce very clean harmonic relationships. They also create slowly changing beats when combined with other partials, which fits the mathematical visual style.

Sound effects use the same idea with short envelopes. A sine-squared envelope fades a sound in and out:

```text
envelope(t) = sin²(pi * t / duration)
```

That prevents clicks at the beginning and end of a sound. Jumps, coins, springs, crashes, and the finish cue each use different partials and frequency sweeps.

## 10. Camera and UI animation

The camera follows the terrain and player height using a damped spring model. This avoids instantly locking the camera to every small movement.

The UI is drawn using Pygame primitives and a procedural segment/pixel font. Text brightness, panel motion, and icon pulses use sine functions, so the interface follows the same visual language as the terrain.

## 11. Determinism and testing

`surf_logic_hash()` hashes the authoritative state, including:

- Time and distance.
- Speed, height, and vertical velocity.
- Hearts, coins, score, and crashes.
- Finished and failed flags.
- Objects already collected or hit.

Rendering is not part of this hash. Different GPU drivers may produce slightly different pixels, but the gameplay result should remain identical for the same input sequence.

The test suite covers:

- Deterministic level generation.
- Analytic terrain derivatives.
- Jump and spring behavior.
- Collision and heart rules.
- Wave-equation stability.
- Audio envelopes.
- Logic-hash determinism.
- The sine-only linter.
- Terrain fairness and deterministic domain randomization.
- Collision records and heart deltas.
- Float32 buffer reuse and a core frame-time budget.

The performance check reports mean, 95th-percentile, and 99th-percentile simulation cost. It excludes disk capture and display synchronization so it can catch physics or wave-solver regressions reliably on development machines.

## What could have been used instead?

The current implementation is intentionally small and equation-driven. Other reasonable choices would be:

### Terrain

- A hand-authored spline for stronger level-design control.
- A heightmap for artist-painted terrain.
- Perlin or simplex noise for natural landscapes.
- A 3D displaced mesh for true perspective and lighting.

Sine waves were chosen because they are smooth, deterministic, cheap to evaluate, and satisfy the project's mathematical constraint.

### Physics

- Box2D or another rigid-body engine.
- Verlet integration for a physically flexible board.
- RK4 for more complicated continuous dynamics.
- A constraint solver for board-to-wave contact.

A full physics engine would add realism, but it would also add complexity and make exact deterministic behavior harder to guarantee.

### Rendering

- Sprite-based 2D rendering.
- A tilemap.
- Instanced 3D meshes.
- Compute shaders for the wave field.
- A game engine such as Godot or Unity.

The procedural shader keeps the project self-contained and makes the sine mathematics visible in the final image.

### Audio

- Pre-recorded music and sound assets.
- MIDI playback.
- FMOD or Wwise.
- A larger real-time synthesizer.

Recorded audio would usually sound more expressive. Procedural audio was chosen because it is compact, reproducible, and follows the same mathematical design.

### Architecture

- An Entity-Component-System architecture for a larger game.
- Separate modules for physics, rendering, audio, and input.
- A replay or command-log system for input recording.
- A GPU ping-pong texture for the ripple solver.

The project now uses separate terrain, simulation, collision, renderer, audio, application, RL, and test modules. A larger version could go further with an Entity-Component-System, but that would add unnecessary indirection to the current game.

## Sine-only rule

The runtime linter rejects calls and shader tokens associated with `sqrt`, `pow`, `random`, `noise`, logarithms, inverse trigonometric functions, and similar operations. The report section is explicitly exempt because it performs numerical analysis and plotting.

The restriction is not only a limitation. It encourages analytic derivatives, deterministic generation, implicit shapes, additive synthesis, and simple numerical methods throughout the game.
