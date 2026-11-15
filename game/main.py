from __future__ import annotations

import argparse
import json
import math
import os
import sys
import time
from pathlib import Path
import numpy as np

from .audio import AudioEngine, synth_audio
from .legacy import oscillator_step
from .renderer import Renderer, WaveField
from .surfer import (Inputs, SurferState, final_score, scripted_surf_input,
                     stars_for, surf_logic_hash, update_surfer)
from .terrain import (DT, LEVELS, PI, clamp01, generate_surf_level, lint_game,
                      sine_hash, surf_geometry)

class ScoreStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)
        try:
            self.data = json.loads(self.path.read_text(encoding="utf8"))
        except (OSError, ValueError):
            self.data = {}

    def record(self, level: int, score: int, run_time: float, stars: int) -> bool:
        key = str(level + 1)
        old = self.data.get(key, {})
        is_new = score > int(old.get("score", -1))
        best_time = float(old.get("time", run_time))
        if run_time < best_time:
            best_time = run_time
        self.data[key] = {"score": score if is_new else int(old.get("score", score)),
                          "time": best_time, "stars": stars if stars > int(old.get("stars", 0)) else int(old.get("stars", 0))}
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.path.write_text(json.dumps(self.data, indent=2), encoding="utf8")
        return is_new



# ============================== SINE FONT / UI ===============================
# Segment strokes are illuminated with a sin^2 pulse; no font asset is loaded.
SEGMENTS = {
    "0": "abcedf", "1": "bc", "2": "abged", "3": "abgcd", "4": "fgbc",
    "5": "afgcd", "6": "afgecd", "7": "abc", "8": "abcdefg", "9": "abfgcd",
    "A": "abcefg", "B": "fegcd", "C": "afed", "D": "begcd", "E": "afged", "F": "afge",
    "G": "afedcg", "H": "febcg", "I": "ed", "J": "bcd", "K": "feg", "L": "fed",
    "M": "abcef", "N": "ceg", "O": "abcdef", "P": "abfge", "Q": "abcfg", "R": "eg",
    "S": "afgcd", "T": "fged", "U": "febcd", "V": "ecd", "W": "bcdef", "X": "fgbc",
    "Y": "fgbcd", "Z": "abged", "-": "g", " ": "", ":": "h", ".": "i", "%": "hi",
}
SEG_LINES = {"a": ((1, 0), (4, 0)), "b": ((5, 1), (5, 3)), "c": ((5, 4), (5, 6)),
             "d": ((1, 7), (4, 7)), "e": ((0, 4), (0, 6)), "f": ((0, 1), (0, 3)),
             "g": ((1, 3.5), (4, 3.5)), "h": ((2.5, 2), (2.5, 2)), "i": ((2.5, 7), (2.5, 7))}

PIXEL_FONT = {
    "A": ("01110","10001","10001","11111","10001","10001","10001"),
    "B": ("11110","10001","10001","11110","10001","10001","11110"),
    "C": ("01111","10000","10000","10000","10000","10000","01111"),
    "D": ("11110","10001","10001","10001","10001","10001","11110"),
    "E": ("11111","10000","10000","11110","10000","10000","11111"),
    "F": ("11111","10000","10000","11110","10000","10000","10000"),
    "G": ("01111","10000","10000","10111","10001","10001","01111"),
    "H": ("10001","10001","10001","11111","10001","10001","10001"),
    "I": ("11111","00100","00100","00100","00100","00100","11111"),
    "J": ("00111","00010","00010","00010","10010","10010","01100"),
    "K": ("10001","10010","10100","11000","10100","10010","10001"),
    "L": ("10000","10000","10000","10000","10000","10000","11111"),
    "M": ("10001","11011","10101","10101","10001","10001","10001"),
    "N": ("10001","11001","10101","10011","10001","10001","10001"),
    "O": ("01110","10001","10001","10001","10001","10001","01110"),
    "P": ("11110","10001","10001","11110","10000","10000","10000"),
    "Q": ("01110","10001","10001","10001","10101","10010","01101"),
    "R": ("11110","10001","10001","11110","10100","10010","10001"),
    "S": ("01111","10000","10000","01110","00001","00001","11110"),
    "T": ("11111","00100","00100","00100","00100","00100","00100"),
    "U": ("10001","10001","10001","10001","10001","10001","01110"),
    "V": ("10001","10001","10001","10001","10001","01010","00100"),
    "W": ("10001","10001","10001","10101","10101","11011","10001"),
    "X": ("10001","10001","01010","00100","01010","10001","10001"),
    "Y": ("10001","10001","01010","00100","00100","00100","00100"),
    "Z": ("11111","00001","00010","00100","01000","10000","11111"),
    "0": ("01110","10001","10011","10101","11001","10001","01110"),
    "1": ("00100","01100","00100","00100","00100","00100","01110"),
    "2": ("01110","10001","00001","00010","00100","01000","11111"),
    "3": ("11110","00001","00001","01110","00001","00001","11110"),
    "4": ("00010","00110","01010","10010","11111","00010","00010"),
    "5": ("11111","10000","10000","11110","00001","00001","11110"),
    "6": ("01110","10000","10000","11110","10001","10001","01110"),
    "7": ("11111","00001","00010","00100","01000","01000","01000"),
    "8": ("01110","10001","10001","01110","10001","10001","01110"),
    "9": ("01110","10001","10001","01111","00001","00001","01110"),
    "-": ("00000","00000","00000","11111","00000","00000","00000"),
    ":": ("00000","00100","00100","00000","00100","00100","00000"),
    ".": ("00000","00000","00000","00000","00000","00100","00100"),
    "+": ("00000","00100","00100","11111","00100","00100","00000"),
    "=": ("00000","00000","11111","00000","11111","00000","00000"),
    "/": ("00001","00010","00010","00100","01000","01000","10000"),
    "(": ("00010","00100","01000","01000","01000","00100","00010"),
    ")": ("01000","00100","00010","00010","00010","00100","01000"),
    ">": ("10000","01000","00100","00010","00100","01000","10000"),
    "<": ("00001","00010","00100","01000","00100","00010","00001"),
    "?": ("01110","10001","00001","00010","00100","00000","00100"),
    " ": ("00000","00000","00000","00000","00000","00000","00000"),
}


def draw_sine_text(surface: object, text: str, x: int, y: int, scale: int, color: tuple[int, int, int], phase: float = 0.0) -> None:
    import pygame
    glow = math.sin(phase * 0.7) ** 2
    for pos, char in enumerate(text.upper()):
        rows = PIXEL_FONT.get(char, PIXEL_FONT[" "])
        for row, bits in enumerate(rows):
            for column, bit in enumerate(bits):
                if bit == "1":
                    shimmer = math.sin(phase + row * 0.31 + column * 0.23) ** 2
                    ink = tuple(int(c * (0.82 + 0.12 * glow + 0.06 * shimmer)) for c in color)
                    rect = (x + pos * 7 * scale + column * scale, y + row * scale, scale, scale)
                    pygame.draw.rect(surface, (18, 13, 9), (rect[0] + scale // 2, rect[1] + scale // 2, scale, scale), border_radius=scale // 3)
                    pygame.draw.rect(surface, ink, rect, border_radius=scale // 3)


def sine_text_width(text: str, scale: int) -> int:
    return len(text) * 7 * scale


def draw_centered_text(surface: object, text: str, y: int, scale: int, color: tuple[int, int, int], phase: float = 0.0) -> None:
    width = surface.get_width()
    draw_sine_text(surface, text, int((width - sine_text_width(text, scale)) * 0.5), y, scale, color, phase)


def draw_panel(surface: object, rect: tuple[int, int, int, int], accent: tuple[int, int, int], phase: float = 0.0,
               selected: bool = False, alpha: int | None = None) -> None:
    import pygame
    pulse = math.sin(phase) ** 2
    fill_alpha = (218 if selected else 202) if alpha is None else alpha
    border = tuple(int(c * (0.70 + 0.30 * pulse)) for c in accent)
    pygame.draw.rect(surface, (17, 13, 10, fill_alpha), rect, border_radius=20)
    pygame.draw.rect(surface, border, rect, 3 if selected else 1, border_radius=20)
    x, y, w, h = rect
    pygame.draw.line(surface, (*border, 90), (x + 24, y + 12), (x + w - 24, y + 12), 2)


def draw_heart_icon(surface: object, x: int, y: int, filled: bool, phase: float) -> None:
    import pygame
    pulse = 1.0 + 0.08 * math.sin(phase)
    radius = int(8 * pulse)
    color = (230, 195, 151) if filled else (55, 45, 36)
    pygame.draw.circle(surface, color, (x - radius // 2, y), radius)
    pygame.draw.circle(surface, color, (x + radius // 2, y), radius)
    pygame.draw.polygon(surface, color, ((x - radius - 3, y + 2), (x + radius + 3, y + 2), (x, y + radius + 12)))


def draw_star_icon(surface: object, x: int, y: int, filled: bool, phase: float) -> None:
    import pygame
    points: list[tuple[int, int]] = []
    for i in range(10):
        radius = 18 if i % 2 == 0 else 8
        radius = int(radius * (1.0 + 0.06 * math.sin(phase + i)))
        angle = -0.5 * PI + i * PI / 5.0
        points.append((x + int(radius * math.cos(angle)), y + int(radius * math.sin(angle))))
    pygame.draw.polygon(surface, (231, 192, 137) if filled else (55, 45, 36), points)



def run_headless(args: argparse.Namespace) -> str:
    from PIL import Image
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    level = generate_surf_level(0, args.seed)
    state = SurferState(level)
    field = WaveField()
    script = None
    if args.script:
        script = json.loads(Path(args.script).read_text(encoding="utf8"))
    try:
        renderer = Renderer(args.width, args.height, True)
    except Exception as exc:
        renderer = None
        print(f"GL unavailable ({exc}); using deterministic Pillow diagnostic frames")
    start = time.perf_counter()
    for frame in range(args.frames):
        update_surfer(state, scripted_surf_input(frame, state, script))
        if "coin" in state.events or "crash" in state.events:
            field.inject(64, 74 if "coin" in state.events else 52)
        field.step()
        if frame % args.capture_every == 0 or frame == args.frames - 1:
            if renderer:
                raw = renderer.render(state, field)
                image = Image.frombytes("RGB", (args.width, args.height), raw).transpose(Image.Transpose.FLIP_TOP_BOTTOM)
            else:
                image = Image.new("RGB", (args.width, args.height), (8, 4, 24))
                px = image.load()
                for yy in range(args.height):
                    for xx in range(args.width):
                        q = math.sin(PI * xx / args.width) * math.sin(PI * yy / args.height)
                        px[xx, yy] = (int(20 + 80 * q * q), int(10 + 30 * q * q), int(45 + 120 * q * q))
            image.save(out / f"wave_frame{frame:06d}.png")
    elapsed = time.perf_counter() - start
    digest = surf_logic_hash(state)
    if state.finished:
        result_stars = stars_for(True, state.coins, state.level.optimal_coins, state.crashes)
        is_new = ScoreStore(out / "scores.json").record(state.level.index, final_score(state), state.time, result_stars)
        print(f"results score={final_score(state)} stars={result_stars} new_high_score={is_new}")
    elif state.failed:
        print("results failed retry=true")
    print(f"logic-sha256 {digest}")
    print(f"frames {args.frames}; mean frame time {1000.0 * elapsed / args.frames:.3f} ms")
    return digest


def benchmark_core(frames: int = 1000, seed: int = 42) -> dict[str, float]:
    """Benchmark deterministic simulation without display or file I/O."""
    if frames < 1:
        raise ValueError("frames must be positive")
    state = SurferState(generate_surf_level(0, seed))
    field = WaveField()
    timings = np.empty(frames, dtype=np.float64)
    for frame in range(frames):
        start = time.perf_counter_ns()
        update_surfer(state, scripted_surf_input(frame, state))
        if "coin" in state.events or "crash" in state.events:
            field.inject(64, 72)
        field.step()
        timings[frame] = (time.perf_counter_ns() - start) / 1_000_000.0
    return {
        "frames": float(frames),
        "mean_ms": float(np.mean(timings)),
        "p95_ms": float(np.percentile(timings, 95.0)),
        "p99_ms": float(np.percentile(timings, 99.0)),
    }


def run_interactive(args: argparse.Namespace) -> None:
    import pygame
    pygame.init()
    screen = pygame.display.set_mode((args.window_width, args.window_height),
                                     pygame.OPENGL | pygame.DOUBLEBUF | pygame.RESIZABLE, vsync=1)
    pygame.display.set_caption("WAVE SURFER")
    renderer = Renderer(args.width, args.height, False)
    level_index = 0
    state = SurferState(generate_surf_level(level_index, args.seed))
    field = WaveField()
    store = ScoreStore(Path(args.out) / "scores.json")
    audio = AudioEngine(Path(args.out), level_index)
    clock = pygame.time.Clock()
    accumulator = 0.0
    previous = time.perf_counter()
    paused = False
    help_open = False
    recorded = False
    mode = "menu"
    new = False
    pending_input = Inputs()
    running = True
    while running:
        now = time.perf_counter()
        accumulator += now - previous
        previous = now
        one_shot = Inputs()
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                running = False
            elif event.type == pygame.KEYDOWN:
                if event.key == pygame.K_ESCAPE: running = False
                elif event.key in (pygame.K_RETURN, pygame.K_SPACE) and mode == "menu":
                    state = SurferState(generate_surf_level(level_index, args.seed)); field = WaveField(); recorded = False; mode = "run"
                    audio.set_level(level_index)
                    audio.trigger("finish")
                    accumulator = 0.0
                    previous = time.perf_counter()
                elif event.key == pygame.K_p: paused = not paused
                elif event.key == pygame.K_h: help_open = not help_open
                elif event.key == pygame.K_r: state = SurferState(generate_surf_level(level_index, args.seed)); field = WaveField(); recorded = False; mode = "run"
                elif event.key in (pygame.K_LEFT, pygame.K_a): one_shot.left = True
                elif event.key in (pygame.K_RIGHT, pygame.K_d): one_shot.right = True
                elif event.key in (pygame.K_UP, pygame.K_w, pygame.K_SPACE): one_shot.jump = True
                elif event.key in (pygame.K_DOWN, pygame.K_s): one_shot.slide = True
        pending_input.left = pending_input.left or one_shot.left
        pending_input.right = pending_input.right or one_shot.right
        pending_input.jump = pending_input.jump or one_shot.jump
        pending_input.slide = pending_input.slide or one_shot.slide
        continuous_input = Inputs()
        if mode == "run":
            held = pygame.key.get_pressed()
            pending_input.left = pending_input.left or held[pygame.K_LEFT] or held[pygame.K_a]
            pending_input.right = pending_input.right or held[pygame.K_RIGHT] or held[pygame.K_d]
            pending_input.slide = pending_input.slide or held[pygame.K_DOWN] or held[pygame.K_s]
            continuous_input = Inputs(left=held[pygame.K_LEFT] or held[pygame.K_a],
                                      right=held[pygame.K_RIGHT] or held[pygame.K_d],
                                      slide=held[pygame.K_DOWN] or held[pygame.K_s])
        if mode == "run" and not paused and not help_open:
            first = True
            while accumulator >= DT:
                update_surfer(state, pending_input if first else continuous_input)
                if first:
                    pending_input = Inputs()
                first = False
                for sound_event in state.events:
                    audio.trigger(sound_event, state.combo)
                if "coin" in state.events or "crash" in state.events:
                    field.inject(64, 72)
                field.step()
                accumulator -= DT
        else:
            accumulator = 0.0
        ui_time = now % 1000.0
        renderer.render(state, field, accumulator / DT, ui_time if mode != "run" else None, mode == "run", readback=False)
        overlay = pygame.Surface((1280, 720), pygame.SRCALPHA)
        best = store.data.get(str(level_index + 1), {}).get("score", 0)
        if mode == "menu":
            overlay.fill((238, 229, 214, 72))
            draw_centered_text(overlay, "W A V E", 122, 7, (43, 37, 31), ui_time)
            draw_centered_text(overlay, "S U R F E R", 212, 5, (43, 37, 31), ui_time)
            pygame.draw.rect(overlay, (45, 39, 33), (420, 438, 440, 82), border_radius=38)
            draw_centered_text(overlay, "RIDE >", 461, 4, (239, 229, 211), ui_time)
            pygame.draw.rect(overlay, (76, 65, 54), (603, 558, 74, 64), 2, border_radius=12)
            draw_sine_text(overlay, "H", 628, 577, 4, (76, 65, 54), ui_time)
            draw_centered_text(overlay, "H FOR HELP", 648, 2, (101, 82, 64), ui_time)
        else:
            pygame.draw.rect(overlay, (222, 198, 166), (28, 24, 58, 58), 2, border_radius=12)
            pygame.draw.rect(overlay, (231, 210, 181), (46, 40, 6, 26), border_radius=3)
            pygame.draw.rect(overlay, (231, 210, 181), (61, 40, 6, 26), border_radius=3)
            for heart_index in range(3):
                draw_heart_icon(overlay, 122 + heart_index * 38, 47, heart_index < state.hearts, state.time * 4.0 + heart_index)
            score_text = f"{final_score(state):05d}"
            draw_sine_text(overlay, score_text, 1228 - sine_text_width(score_text, 4), 24, 4, (237, 217, 190), state.time)
            draw_sine_text(overlay, f"BEST {best}", 1228 - sine_text_width(f"BEST {best}", 2), 66, 2, (139, 117, 95), state.time)
            pygame.draw.circle(overlay, (212, 158, 96), (1136, 105), 14)
            pygame.draw.circle(overlay, (239, 199, 143), (1136, 105), 8, 3)
            draw_sine_text(overlay, str(state.coins), 1163, 94, 3, (237, 217, 190), state.time)
            draw_centered_text(overlay, str(LEVELS[level_index]["name"]), 31, 2, (169, 143, 117), state.time)
            if state.combo > 1:
                draw_centered_text(overlay, f"COMBO {state.combo}", 66, 2, (224, 184, 132), state.time)
            pygame.draw.rect(overlay, (52, 41, 32, 210), (140, 682, 1000, 4), border_radius=2)
            progress_width = int(1000 * state.distance / state.level.length)
            pygame.draw.rect(overlay, (224, 192, 151), (140, 682, progress_width, 4), border_radius=2)
            pygame.draw.circle(overlay, (239, 218, 190), (140 + progress_width, 684), 6)
            pygame.draw.rect(overlay, (54, 47, 40, 180), (1010, 144, 218, 18), border_radius=9)
            fuel_width = int(210.0 * state.fuel / 100.0)
            pygame.draw.rect(overlay, (198, 139, 70), (1014, 148, fuel_width, 10), border_radius=5)
            draw_sine_text(overlay, f"BOOST {int(state.fuel)}", 1010, 174, 2, (89, 72, 56), state.time)
            draw_sine_text(overlay, "UP JUMP  RIGHT BOOST  DOWN SLIDE  LEFT BRAKE", 28, 620, 2, (110, 88, 67), state.time)
            if not audio.available:
                draw_sine_text(overlay, "AUDIO OFF  WAV SAVED", 28, 650, 2, (169, 133, 99), state.time)
        if help_open:
            overlay.fill((25, 21, 18, 224))
            draw_panel(overlay, (90, 54, 1100, 610), (218, 190, 153), state.time, True)
            lines = ("HOW TO SURF", "UP OR SPACE: JUMP", "RIGHT: HOLD TO BOOST", "DOWN: SLIDE", "LEFT: BRAKE",
                     "GOLD WAVE RINGS: COINS", "BLUE ORBS: BOOST FUEL", "GREEN SPRINGS: HIGH JUMP", "RED BREAKERS: OBSTACLES",
                     "ONLY OBSTACLE COLLISIONS COST HEARTS", "H: CLOSE HELP")
            for i, line in enumerate(lines):
                color = (239, 215, 184) if i == 0 else (178, 150, 120)
                draw_sine_text(overlay, line, 132, 76 + i * 48, 3 if i == 0 else 2, color, state.time + i)
        if mode == "run" and (state.finished or state.failed):
            overlay.fill((10, 8, 6, 184))
            title = "LEVEL COMPLETE" if state.finished else "GAME OVER"
            draw_panel(overlay, (330, 138, 620, 440), (225, 198, 162), state.time, True)
            draw_centered_text(overlay, title, 184, 5, (239, 218, 190), state.time)
            if state.finished:
                stars = stars_for(True, state.coins, state.level.optimal_coins, state.crashes)
                if not recorded:
                    new = store.record(level_index, final_score(state), state.time, stars)
                    recorded = True
                for star_index in range(3):
                    draw_star_icon(overlay, 550 + star_index * 90, 302, star_index < stars, state.time + star_index)
                draw_centered_text(overlay, f"SCORE {final_score(state)}", 358, 3, (235, 206, 167), state.time)
                draw_centered_text(overlay, f"TIME {int(state.time)}  COINS {state.coins}", 410, 2, (164, 137, 109), state.time)
                if new: draw_centered_text(overlay, "NEW HIGH SCORE", 460, 2, (224, 183, 131), state.time)
            pygame.draw.rect(overlay, (226, 203, 173), (440, 505, 400, 52), border_radius=10)
            draw_centered_text(overlay, "R RETRY     ESC QUIT", 521, 2, (39, 29, 22), state.time)
        renderer.present(overlay, screen.get_size())
        pygame.display.flip()
        pygame.display.set_caption(f"WAVE SURFER | hearts {state.hearts} coins {state.coins} score {final_score(state)}")
        # Presentation is paced by the display; the fixed-step accumulator keeps
        # gameplay at exactly 60 Hz even on high-refresh or non-vsync systems.
        clock.tick()
    audio.stop()
    pygame.quit()


# --- REPORT (lint-exempt) ---
def run_report(out_dir: str | Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    kinds = ("euler", "symplectic", "verlet", "rk4")
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    orders: dict[str, float] = {}
    stability: dict[str, float] = {}
    for kind in kinds:
        x, v = 1.0, 0.0
        energies = []
        for i in range(100000):
            x, v = oscillator_step(kind, x, v, 0.001, 4.0, 0.0)
            if i % 100 == 0:
                energies.append(0.5 * (v * v + 16.0 * x * x))
        axes[0].plot(np.arange(len(energies)) * 0.1, energies, label=kind)
        dts = np.array((0.04, 0.02, 0.01, 0.005), dtype=float)
        errors = []
        exact_x = math.cos(4.0)
        exact_v = -4.0 * math.sin(4.0)
        for dt in dts:
            xx, vv, t = 1.0, 0.0, 0.0
            while t + dt <= 1.0 + 1e-12:
                xx, vv = oscillator_step(kind, xx, vv, dt, 4.0, 0.0)
                t += dt
            errors.append(abs(xx - exact_x) + abs(vv - exact_v))
        order = float(np.polyfit(np.log(dts), np.log(errors), 1)[0])
        orders[kind] = order
        axes[1].loglog(dts, errors, "o-", label=f"{kind} p={order:.2f}")
        stable_dt = 0.0
        for candidate in np.linspace(0.005, 1.0, 200):
            sx, sv, bounded = 1.0, 0.0, True
            for _ in range(2000):
                sx, sv = oscillator_step(kind, sx, sv, float(candidate), 4.0, 0.0)
                if abs(sx) > 100.0 or abs(sv) > 400.0:
                    bounded = False
                    break
            if bounded:
                stable_dt = float(candidate)
        stability[kind] = stable_dt
    axes[0].set_title("Undamped energy, 100,000 steps")
    axes[0].legend(); axes[1].set_title("Convergence order"); axes[1].legend()
    fig.tight_layout(); fig.savefig(out / "integrators.png", dpi=150); plt.close(fig)

    field = WaveField(damping=0.0)
    field.inject(64, 64)
    snapshots = []
    energy = []
    for i in range(360):
        field.step(); energy.append(field.energy())
        if i in (0, 60, 120, 240): snapshots.append(field.u.copy())
    fig, axes = plt.subplots(1, 4, figsize=(12, 3))
    for ax, data in zip(axes, snapshots): ax.imshow(data, cmap="magma"); ax.axis("off")
    fig.tight_layout(); fig.savefig(out / "wave_pde.png", dpi=150); plt.close(fig)

    values = np.array([sine_hash(i, 7.0) for i in range(20000)])
    serial = float(np.corrcoef(values[:-1], values[1:])[0, 1])
    builtin = np.random.default_rng(7).random(20000)
    weyl = np.array([fract(i * PHI_WEYL) for i in range(20000)])
    target = (np.arange(20000) + 0.5) / 20000.0
    discrepancies = {"sine_hash": float(np.max(np.abs(np.sort(values) - target))),
                     "builtin": float(np.max(np.abs(np.sort(builtin) - target))),
                     "weyl": float(np.max(np.abs(np.sort(weyl) - target)))}
    fig, axes = plt.subplots(1, 2, figsize=(9, 4))
    axes[0].hist(values, bins=50); axes[0].set_title("Sine-hash uniformity")
    axes[1].scatter(values[:-1:20], values[1::20], s=2); axes[1].set_title(f"serial r={serial:.4f}")
    fig.tight_layout(); fig.savefig(out / "hash_quality.png", dpi=150); plt.close(fig)

    x = np.linspace(0.0, 6.0 * PI, 1200)
    moire = np.sin(20.0 * x) * np.sin(22.0 * x)
    fig, axes = plt.subplots(1, 2, figsize=(10, 3))
    axes[0].plot(x, moire); axes[0].set_title("Moire beat |22-20|=2")
    signal = np.sin(TAU * 440.0 * np.arange(4096) / 44100.0 + 2.0 * np.sin(TAU * 110.0 * np.arange(4096) / 44100.0))
    spectrum = abs(np.fft.rfft(signal * np.sin(PI * np.arange(4096) / 4095.0) ** 2))
    axes[1].plot(np.fft.rfftfreq(4096, 1 / 44100.0), spectrum); axes[1].set_xlim(0, 1500); axes[1].set_title("FM sidebands + Hann window")
    fig.tight_layout(); fig.savefig(out / "moire_alias_audio.png", dpi=150); plt.close(fig)
    (out / "report.json").write_text(json.dumps({"orders": orders, "stability_dt": stability,
        "hash_serial_correlation": serial, "uniform_discrepancy": discrepancies,
        "pde_energy_drift": (max(energy)-min(energy))/max(energy)}, indent=2))
    print(f"report written to {out}")
# --- END REPORT (lint-exempt) ---


# ============================== CLI ==========================================
def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="WAVE SURFER")
    p.add_argument("--headless", action="store_true")
    p.add_argument("--report", action="store_true")
    p.add_argument("--benchmark", action="store_true",
                   help="measure 60 Hz gameplay and wave-solver frame cost")
    p.add_argument("--seed", type=int, default=None)
    p.add_argument("--frames", type=int, default=1000)
    p.add_argument("--script")
    p.add_argument("--out", default="out")
    p.add_argument("--width", type=int, default=int(os.environ.get("SINE_WIDTH", "640")))
    p.add_argument("--height", type=int, default=int(os.environ.get("SINE_HEIGHT", "360")))
    p.add_argument("--window-width", type=int, default=1280)
    p.add_argument("--window-height", type=int, default=720)
    p.add_argument("--capture-every", type=int, default=120)
    return p


def resolve_run_seed(seed: int | None) -> int:
    """Use external entropy for a new run while keeping explicit seeds replayable."""
    if seed is not None:
        return seed
    return int(time.time_ns() % 2147483647)


def main(argv: list[str] | None = None) -> int:
    args = parser().parse_args(argv)
    errors = lint_game()
    if errors:
        print("sine-rule lint failed:\n" + "\n".join(errors), file=sys.stderr)
        return 2
    if args.report:
        run_report(args.out)
    else:
        args.seed = resolve_run_seed(args.seed)
        print(f"run seed {args.seed}")
        if args.benchmark:
            print(json.dumps(benchmark_core(args.frames, args.seed), indent=2))
        elif args.headless:
            run_headless(args)
        else:
            run_interactive(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

