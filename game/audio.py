from __future__ import annotations

import math
import struct
import wave
from pathlib import Path
import numpy as np

from .terrain import PI, TAU, clamp01

def half_sine_envelope(t: np.ndarray, duration: float) -> np.ndarray:
    phase = np.asarray(t, dtype=np.float64) / duration
    phase = np.where(phase < 0.0, 0.0, np.where(phase > 1.0, 1.0, phase))
    return np.sin(PI * phase) ** 2


def synth_audio(seconds: float, level: int = 0, sample_rate: int = 44100) -> np.ndarray:
    """The melodic Wave Surfer loop, built entirely from sine partials."""
    count = int(seconds * sample_rate)
    t = np.arange(count, dtype=np.float64) / sample_rate
    root = 110.0
    bpm = 84.0
    beat_time = 60.0 / bpm
    melody_ratios = (1.0, 9.0/8.0, 5.0/4.0, 3.0/2.0, 5.0/3.0, 3.0/2.0, 5.0/4.0, 9.0/8.0,
                     1.0, 5.0/4.0, 3.0/2.0, 2.0, 5.0/3.0, 3.0/2.0, 9.0/8.0, 1.0)
    note_time = 0.5 * beat_time
    step = np.floor(t / note_time)
    local = t - step * note_time
    melody = np.zeros(count, dtype=np.float64)
    counter = np.zeros(count, dtype=np.float64)
    for note_index, ratio in enumerate(melody_ratios):
        mask = np.where(np.mod(step, 16.0) == float(note_index), 1.0, 0.0)
        frequency = root * 2.0 * ratio
        envelope = np.sin(PI * local / note_time) ** 2
        vibrato = 0.018 * np.sin(TAU * 4.2 * local)
        tone = np.sin(TAU * frequency * local + vibrato)
        tone += 0.12 * np.sin(TAU * frequency * 2.0 * local)
        melody += mask * envelope * tone
        counter_ratio = melody_ratios[(note_index + 5) % 16]
        counter += mask * envelope * np.sin(TAU * root * counter_ratio * local + 0.35 * math.sin(note_index))

    bass_step = np.floor(t / (2.0 * beat_time))
    bass_local = t - bass_step * 2.0 * beat_time
    bass = np.zeros(count, dtype=np.float64)
    pad = np.zeros(count, dtype=np.float64)
    bass_ratios = (1.0, 4.0/3.0, 3.0/2.0, 5.0/4.0)
    for note_index, ratio in enumerate(bass_ratios):
        mask = np.where(np.mod(bass_step, 4.0) == float(note_index), 1.0, 0.0)
        envelope = np.sin(PI * bass_local / (2.0 * beat_time)) ** 2
        bass += mask * envelope * (np.sin(TAU * root * 0.5 * ratio * bass_local) +
                                    0.12 * np.sin(TAU * root * ratio * bass_local))
        third = 5.0/4.0
        pad += mask * envelope * (np.sin(TAU * root * ratio * t) +
                                  0.50 * np.sin(TAU * root * ratio * third * t) +
                                  0.36 * np.sin(TAU * root * ratio * 3.0/2.0 * t))

    kick_step = np.floor(t / beat_time)
    kick_local = t - kick_step * beat_time
    kick_env = np.where(kick_local < 0.22, np.sin(PI * kick_local / 0.22) ** 2, 0.0)
    kick = kick_env * np.sin(TAU * (48.0 * kick_local + 8.0 * np.sin(PI * kick_local / 0.22)))
    pulse_step = np.floor(t / (0.5 * beat_time))
    pulse_local = t - pulse_step * 0.5 * beat_time
    pulse_env = np.where(pulse_local < 0.11, np.sin(PI * pulse_local / 0.11) ** 2, 0.0)
    pulse = pulse_env * (np.sin(TAU * 900.0 * pulse_local) +
                         0.22 * np.sin(TAU * 1350.0 * pulse_local))
    audio = 0.105 * pad + 0.17 * bass + 0.19 * melody + 0.055 * counter + 0.050 * kick + 0.010 * pulse
    loop_fade = 0.08
    start_fade = np.where(t < loop_fade, np.sin(0.5 * PI * t / loop_fade) ** 2, 1.0)
    remaining = seconds - t
    end_fade = np.where(remaining < loop_fade, np.sin(0.5 * PI * remaining / loop_fade) ** 2, 1.0)
    audio *= start_fade * end_fade
    return np.asarray(audio, dtype=np.float32)


def synth_sfx(name: str, combo: int = 0, sample_rate: int = 44100) -> np.ndarray:
    durations = {"jump": 0.24, "coin": 0.16, "slide": 0.20, "crash": 0.55, "shield": 0.38,
                 "shield_hit": 0.30, "magnet": 0.48, "spring": 0.52, "finish": 1.2}
    duration = durations.get(name, 0.22)
    t = np.arange(int(duration * sample_rate), dtype=np.float64) / sample_rate
    bases = {"jump": 220.0, "coin": 660.0 + 22.0 * combo, "slide": 100.0, "crash": 55.0,
             "shield": 330.0, "shield_hit": 180.0, "magnet": 146.7, "spring": 293.3, "finish": 440.0}
    base = bases.get(name, 220.0)
    sweep = 1.0 + 0.24 * np.sin(0.5 * PI * t / duration)
    carrier = np.sin(TAU * base * t * sweep)
    if name == "coin":
        carrier += 0.38 * np.sin(TAU * base * 2.0 * t)
    elif name == "spring":
        carrier += 0.46 * np.sin(TAU * base * 5.0/4.0 * t)
        carrier += 0.32 * np.sin(TAU * base * 3.0/2.0 * t)
    elif name in ("crash", "slide"):
        carrier = np.sin(TAU * base * t * (1.0 - 0.36 * np.sin(0.5 * PI * t / duration)))
        carrier += 0.28 * np.sin(TAU * (base * 1.47) * t)
    elif name in ("shield", "shield_hit", "magnet"):
        carrier += 0.30 * np.sin(TAU * base * 3.0 / 2.0 * t + 0.65 * np.sin(TAU * 5.0 * t))
    elif name == "finish":
        carrier += 0.5 * np.sin(TAU * base * 5.0 / 4.0 * t) + 0.35 * np.sin(TAU * base * 3.0 / 2.0 * t)
    return np.asarray(0.22 * carrier * half_sine_envelope(t, duration), dtype=np.float32)


def doppler_frequency(source_frequency: float, source_velocity: float, sound_speed: float = 343.0) -> float:
    denominator = sound_speed - source_velocity
    if denominator < 1.0:
        denominator = 1.0
    return source_frequency * sound_speed / denominator


def write_wav(path: str | Path, audio: np.ndarray, sample_rate: int = 44100) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    pcm = np.asarray(audio * 30000.0, dtype=np.int16)
    with wave.open(str(path), "wb") as out:
        out.setnchannels(1)
        out.setsampwidth(2)
        out.setframerate(sample_rate)
        out.writeframes(pcm.tobytes())


class AudioEngine:
    def __init__(self, out_dir: Path, level: int):
        self.available = False
        self.audio = synth_audio(16.0, level)
        self.position = 0
        self.effects: list[tuple[np.ndarray, int]] = []
        self.stream = None
        try:
            import sounddevice as sd
            self.stream = sd.OutputStream(samplerate=44100, channels=1, dtype="float32", blocksize=1024,
                                          callback=self._callback)
            self.stream.start()
            self.available = True
        except Exception:
            write_wav(out_dir / f"level_{level + 1}_fallback.wav", self.audio)

    def _callback(self, outdata: np.ndarray, frames: int, timing: object, status: object) -> None:
        indices = (np.arange(frames) + self.position) % len(self.audio)
        block = self.audio[indices].copy()
        self.position = int((self.position + frames) % len(self.audio))
        remaining: list[tuple[np.ndarray, int]] = []
        for effect, position in self.effects:
            count = len(effect) - position
            if count > frames:
                count = frames
            if count > 0:
                block[:count] += effect[position:position + count]
            position += count
            if position < len(effect):
                remaining.append((effect, position))
        self.effects = remaining
        outdata[:, 0] = np.where(block < -0.95, -0.95, np.where(block > 0.95, 0.95, block))

    def trigger(self, name: str, combo: int = 0) -> None:
        self.effects.append((synth_sfx(name, combo), 0))

    def set_level(self, level: int) -> None:
        self.audio = synth_audio(16.0, level)
        self.position = 0
        self.effects.clear()

    def stop(self) -> None:
        if self.available:
            try:
                if self.stream is not None:
                    self.stream.stop()
                    self.stream.close()
            except Exception:
                pass



