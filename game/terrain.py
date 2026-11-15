from __future__ import annotations

import ast
import functools
import math
from dataclasses import dataclass
from pathlib import Path

PI = math.pi
TAU = 2.0 * PI
DT = 1.0 / 60.0
LANE_WIDTH = 1.35
LANE_CHANGE_DISTANCE = 2.8
JUMP_TIME = 0.82
JUMP_DISTANCE = 12.8
SLIDE_TIME = 0.72
JUMP_GRAVITY = -17.0
JUMP_VELOCITY = -0.5 * JUMP_GRAVITY * JUMP_TIME
INVULN_TIME = 1.5
PHI_WEYL = 0.6180339887498949
LEVELS = (
    {"name": "WAVE SURFER", "seed": 42, "duration": 75.0, "v0": 13.0, "v1": 20.0, "difficulty": 1},
)
BANNED = frozenset(("abs", "sqrt", "pow", "exp", "log", "tan", "atan", "atan2", "asin", "acos", "smoothstep", "min", "max", "random", "noise"))



def clamp01(x: float) -> float:
    if x < 0.0:
        return 0.0
    if x > 1.0:
        return 1.0
    return x


def fract(x: float) -> float:
    return x - math.floor(x)


def sine_hash(n: float, k: float) -> float:
    """The only pseudo-random source used by gameplay."""
    return fract(43758.5453 * math.sin(12.9898 * n + 78.233 * k))


def sine_lint_source(source: str) -> list[str]:
    """AST lint Python and token-lint embedded GLSL, excluding report code."""
    begin_tag = "# --- " + "REPORT (lint-exempt) ---"
    end_tag = "# --- END " + "REPORT (lint-exempt) ---"
    begin = source.find(begin_tag)
    end = source.find(end_tag)
    if begin >= 0 and end > begin:
        source = source[:begin] + "\npass\n" + source[end + len(end_tag):]
    errors: list[str] = []
    tree = ast.parse(source)
    for node in ast.walk(tree):
        if isinstance(node, ast.Call):
            names: list[str] = []
            if isinstance(node.func, ast.Name):
                names.append(node.func.id)
            elif isinstance(node.func, ast.Attribute):
                cursor: ast.AST = node.func
                while isinstance(cursor, ast.Attribute):
                    names.append(cursor.attr)
                    cursor = cursor.value
                if isinstance(cursor, ast.Name):
                    names.append(cursor.id)
            for name in names:
                if name in BANNED:
                    errors.append(f"line {node.lineno}: banned call {name}()")
                    break
    for node in ast.walk(tree):
        if isinstance(node, ast.Assign) and isinstance(node.value, ast.Constant) and isinstance(node.value.value, str):
            targets = [t.id for t in node.targets if isinstance(t, ast.Name)]
            if any("SHADER" in t for t in targets):
                glsl = node.value.value
                word = ""
                words: list[str] = []
                for char in glsl:
                    if char.isalnum() or char == "_":
                        word += char
                    elif word:
                        words.append(word)
                        word = ""
                if word:
                    words.append(word)
                for banned in BANNED:
                    if banned in words:
                        errors.append(f"GLSL: banned token {banned}")
    return errors


def lint_game(path: str | Path | None = None) -> list[str]:
    if path is not None:
        return sine_lint_source(Path(path).read_text(encoding="utf8"))
    errors: list[str] = []
    for source_path in sorted(Path(__file__).parent.glob("*.py")):
        if source_path.name == "__init__.py":
            continue
        for error in sine_lint_source(source_path.read_text(encoding="utf8")):
            errors.append(f"{source_path.name}: {error}")
    return errors



SURF_GRAVITY = -18.0
SURF_JUMP_SPEED = 8.4
BOARD_CLEARANCE = 0.34
TERRAIN_VERSION = 2
DIFFICULTY_STAGES = ("smooth", "moderate", "compound", "obstacles", "full")


def level_length(index: int) -> float:
    cfg = LEVELS[index]
    total = 0.0
    steps = int(cfg["duration"] * 60.0)
    for i in range(steps):
        p = i / steps
        total += (cfg["v0"] + (cfg["v1"] - cfg["v0"]) * math.sin(0.5 * PI * p)) * DT
    return total


@dataclass(frozen=True)
class TerrainDomain:
    """Seeded gameplay parameters shared by humans and learning agents."""
    stage: str
    difficulty: float
    amplitude_scale: float
    frequency_scale: float
    detail_scale: float
    object_spacing: float
    friction: float
    boost_strength: float
    gravity: float


@dataclass(frozen=True)
class FairnessReport:
    valid: bool
    max_slope_squared: float
    max_curvature_squared: float
    closest_breakers: float
    reasons: tuple[str, ...]


@dataclass(frozen=True)
class SurfThing:
    distance: float
    kind: str
    phase: float = 0.0


@dataclass
class SurfLevel:
    index: int
    seed: int
    length: float
    things: list[SurfThing]
    optimal_coins: int
    domain: TerrainDomain | None = None
    fairness: FairnessReport | None = None

    def __post_init__(self) -> None:
        if self.domain is None:
            self.domain = terrain_domain(self.seed)


def curriculum_stage(episode: int) -> str:
    """A stable five-step curriculum; full difficulty begins at episode 60."""
    if episode < 8:
        return "smooth"
    if episode < 18:
        return "moderate"
    if episode < 32:
        return "compound"
    if episode < 48:
        return "obstacles"
    return "full"


@functools.lru_cache(maxsize=512)
def terrain_domain(seed: int, stage: str = "full", difficulty: float = 1.0) -> TerrainDomain:
    if stage not in DIFFICULTY_STAGES:
        raise ValueError(f"unknown difficulty stage: {stage}")
    difficulty = clamp01(difficulty)
    stage_index = DIFFICULTY_STAGES.index(stage)
    stage_mix = stage_index / float(len(DIFFICULTY_STAGES) - 1)
    amplitude = (0.62 + 0.38 * stage_mix) * (0.88 + 0.24 * sine_hash(seed, 51.0))
    frequency = (0.70 + 0.30 * stage_mix) * (0.90 + 0.20 * sine_hash(seed, 52.0))
    detail = (0.08 + 0.92 * stage_mix) * (0.82 + 0.30 * sine_hash(seed, 53.0))
    spacing = (10.5 - 3.0 * stage_mix) * (0.92 + 0.16 * sine_hash(seed, 54.0))
    friction = (1.52 + 0.48 * difficulty) * (0.90 + 0.20 * sine_hash(seed, 55.0))
    boost = (8.4 + 2.8 * difficulty) * (0.90 + 0.20 * sine_hash(seed, 56.0))
    gravity = -(16.2 + 2.8 * difficulty) * (0.94 + 0.12 * sine_hash(seed, 57.0))
    return TerrainDomain(stage, difficulty, amplitude, frequency, detail, spacing,
                         friction, boost, gravity)


@functools.lru_cache(maxsize=512)
def terrain_parameters(seed: int, stage: str = "full", difficulty: float = 1.0) -> tuple[float, ...]:
    """Create one reproducible layered-wave terrain from a run seed."""
    domain = terrain_domain(seed, stage, difficulty)
    return (
        domain.amplitude_scale * (0.82 + 0.36 * sine_hash(seed, 31.0)),
        domain.amplitude_scale * (1.15 + 0.60 * sine_hash(seed, 32.0)),
        domain.frequency_scale * (0.085 + 0.030 * sine_hash(seed, 33.0)),
        domain.frequency_scale * (0.185 + 0.050 * sine_hash(seed, 34.0)),
        TAU * (sine_hash(seed, 35.0) - 0.5),
        domain.detail_scale * (0.32 + 0.18 * sine_hash(seed, 36.0)),
        2.05 + 0.50 * sine_hash(seed, 37.0),
        TAU * (sine_hash(seed, 38.0) - 0.5),
        domain.detail_scale * (0.12 + 0.12 * sine_hash(seed, 39.0)),
        4.15 + 1.10 * sine_hash(seed, 40.0),
        TAU * (sine_hash(seed, 41.0) - 0.5),
    )


def wave_geometry(s: float, level: int, length: float, seed: int | None = None,
                  stage: str = "full", difficulty: float = 1.0) -> tuple[float, float, float]:
    """Seeded composite sine height with analytic slope and curvature."""
    if seed is None:
        seed = int(LEVELS[level]["seed"])
    a0, gain, k0, k1, phase0, a2, r2, phase2, a3, r3, phase3 = terrain_parameters(seed, stage, difficulty)
    p = clamp01(s / length)
    amplitude = a0 + gain * p
    amplitude_rate = gain / length
    chirp = (k1 - k0) / length
    phase = phase0 + k0 * s + 0.5 * chirp * s * s
    primary = math.sin(phase)
    secondary = math.sin(r2 * phase + phase2)
    detail = math.sin(r3 * phase + phase3)
    shape = primary + a2 * secondary + a3 * detail
    shape_rate = math.cos(phase) + a2 * r2 * math.cos(r2 * phase + phase2)
    shape_rate += a3 * r3 * math.cos(r3 * phase + phase3)
    shape_curve = -primary - a2 * r2 * r2 * secondary - a3 * r3 * r3 * detail
    phase_rate = k0 + chirp * s
    height = amplitude * shape
    slope = amplitude_rate * shape + amplitude * shape_rate * phase_rate
    curvature = 2.0 * amplitude_rate * shape_rate * phase_rate
    curvature += amplitude * (shape_curve * phase_rate * phase_rate + shape_rate * chirp)
    return height, slope, curvature


def surf_geometry(level: SurfLevel, distance: float) -> tuple[float, float, float]:
    return wave_geometry(distance, level.index, level.length, level.seed,
                         level.domain.stage, level.domain.difficulty)


def validate_surf_level(level: SurfLevel, samples: int = 384) -> FairnessReport:
    """Reject numerically harsh terrain and impossible hazard arrangements."""
    slope2_peak = 0.0
    curve2_peak = 0.0
    reasons: list[str] = []
    for sample in range(samples + 1):
        distance = level.length * sample / samples
        _, slope, curvature = surf_geometry(level, distance)
        slope2 = slope * slope
        curve2 = curvature * curvature
        if slope2 > slope2_peak:
            slope2_peak = slope2
        if curve2 > curve2_peak:
            curve2_peak = curve2
    if slope2_peak > 4.5:
        reasons.append("terrain slope exceeds the safe riding envelope")
    if curve2_peak > 1.6:
        reasons.append("terrain curvature changes too sharply")

    previous_breaker = -level.length
    closest_breakers = level.length
    for thing in level.things:
        if thing.kind != "breaker":
            continue
        gap = thing.distance - previous_breaker
        if previous_breaker >= 0.0 and gap < closest_breakers:
            closest_breakers = gap
        previous_breaker = thing.distance
        _, slope, curvature = surf_geometry(level, thing.distance)
        if slope * slope > 1.7 or curvature * curvature > 0.08:
            reasons.append("breaker occupies an unsafe landing zone")
            break
    if closest_breakers < 8.0:
        reasons.append("breakers are too close for a fair reaction window")
    return FairnessReport(not reasons, slope2_peak, curve2_peak, closest_breakers, tuple(reasons))


def generate_surf_level(index: int, seed: int | None = None, *, stage: str = "full",
                        difficulty: float = 1.0, validate: bool = True) -> SurfLevel:
    if index != 0:
        raise ValueError("WAVE SURFER has one definitive track")
    if seed is None:
        seed = int(LEVELS[index]["seed"])
    length = level_length(index)
    domain = terrain_domain(seed, stage, difficulty)
    things: list[SurfThing] = []
    coins = 0
    spacing = domain.object_spacing
    count = int(length / spacing)
    previous_breaker = -length
    for i in range(5, count):
        distance = i * spacing
        roll = sine_hash(seed + i * 13, 21.0)
        phase = TAU * sine_hash(seed + i * 17, 22.0)
        stage_index = DIFFICULTY_STAGES.index(stage)
        hazard_mix = stage_index / float(len(DIFFICULTY_STAGES) - 1)
        coin_cut = 0.62 - 0.08 * hazard_mix
        boost_cut = coin_cut + 0.10
        spring_cut = boost_cut + 0.10
        breaker_cut = spring_cut + 0.02 + 0.10 * hazard_mix
        if roll < coin_cut:
            things.append(SurfThing(distance, "coin", phase))
            coins += 1
        elif roll < boost_cut:
            things.append(SurfThing(distance, "boost", phase))
        elif roll < spring_cut:
            things.append(SurfThing(distance, "spring", phase))
        elif roll < breaker_cut:
            _, slope, curvature = wave_geometry(distance, index, length, seed, stage, difficulty)
            fair_zone = slope * slope <= 1.7 and curvature * curvature <= 0.08
            reaction_gap = distance - previous_breaker >= 8.0
            if fair_zone and reaction_gap:
                things.append(SurfThing(distance, "breaker", phase))
                previous_breaker = distance
            else:
                things.append(SurfThing(distance, "coin", phase))
                coins += 1
    level = SurfLevel(index, seed, length, things, coins, domain)
    report = validate_surf_level(level)
    level.fairness = report
    if validate and not report.valid:
        raise ValueError("unfair generated terrain: " + "; ".join(report.reasons))
    return level
