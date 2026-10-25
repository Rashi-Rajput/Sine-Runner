# Wave Surfer

Wave Surfer is a small deterministic riding game written in Python. The player surfs over a moving mathematical landscape, jumps over red breakers, collects gold rings, grabs fuel, and uses green springs for high jumps.

The important design choice is that the game is generated from equations instead of imported art. The ground, rider, UI, sound, and decorative ripples are all made from sine functions and simple geometry.

## Run it

```bash
python -m pip install -r requirements.txt
python game.py
```

Useful checks:

```bash
pytest -q
python game.py --headless --seed 42 --frames 1000 --out out/
python game.py --report --out out/report/
```

Headless mode runs the same fixed-step game loop without opening a window. It writes `wave_frameXXXXXX.png` captures and prints a SHA-256 logic hash. If an OpenGL context cannot be created, it announces that fact and writes a simple Pillow diagnostic frame instead.

## Controls

- `W`, `Up`, or `Space`: jump
- `Right` or `D`: hold boost
- `Left` or `A`: brake
- `P`: pause
- `H`: show or hide help
- `R`: restart
- `Esc`: quit

The game has one course. It is deliberately one coherent visual and musical experience rather than a selector containing several loosely related themes.

## How the game is organised

`game.py` is split into a few practical systems:

1. **Math and deterministic helpers** generate repeatable values from sine.
2. **Wave Surfer simulation** advances the rider, terrain, pickups, and collisions.
3. **WaveField** creates a decorative 2D ripple simulation.
4. **ModernGL renderer** draws the scene procedurally in a fragment shader.
5. **Audio synthesis** creates the music and sound effects as arrays of samples.
6. **Pygame loop** handles input, timing, menus, and presentation.
7. **Headless mode and tests** verify that the simulation remains reproducible.

The file also contains an older lane-runner model (`GameState`, `LevelData`, and `update_game`) retained for compatibility tests. The interactive game uses the newer `SurferState` and `update_surfer` path.

## 1. The terrain: a layered sine wave

The ground height is based on several sine waves added together:

```text
F(phi) = sin(phi)
       + 0.42 sin(2.3 phi + 1.1)
       + 0.18 sin(4.7 phi - 0.6)

y(s) = A(s) F(phi(s))
```

`s` is distance along the course. `A(s)` grows with progress, so the hills become taller. The phase is a chirp:

```text
phi(s) = k0 s + 1/2 chirp s²
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

## 4. Deterministic placement and the sine hash

The project does not use Python's `random` module. It uses:

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

`WaveField` is a small two-dimensional leapfrog wave solver used only for decorative ripples. It stores the current grid and the previous grid, then applies a discrete Laplacian:

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

The interactive path does not read the framebuffer back to the CPU every frame. Headless mode reads pixels only when it needs to save a capture.

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

For this small single-player project, the current single-file structure is convenient, but splitting the active surfer code into modules would improve maintainability as the game grows.

## Sine-only rule

The runtime linter rejects calls and shader tokens associated with `sqrt`, `pow`, `random`, `noise`, logarithms, inverse trigonometric functions, and similar operations. The report section is explicitly exempt because it performs numerical analysis and plotting.

The restriction is not only a limitation. It encourages analytic derivatives, deterministic generation, implicit shapes, additive synthesis, and simple numerical methods throughout the game.
