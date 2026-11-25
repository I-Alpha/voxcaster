import time
import zlib

import numpy as np


class RecordingDucker:
    def __init__(self):
        self.calls = []

    def acquire(self, pid):
        self.calls.append("acquire")

    def duck(self, pid):
        self.calls.append("duck")

    def release(self):
        self.calls.append("release")

    def shutdown(self):
        pass


SAMPLE_RATE = 24000


class FakeKokoro:
    """``pipeline_factory`` standing in for Kokoro: one result per newline-separated
    segment, taking ``delay`` seconds (a number, or a function of the running piece
    number) and carrying ``audio_sec`` seconds of non-zero speech between
    ``lead_sec`` and ``tail_sec`` of silence. ``fail_at`` makes that piece raise.

    ``pieces`` records every rendered piece as (segment, voice, int16 samples,
    time.monotonic() of delivery); the float32 samples handed to the engine are
    exactly those int16 values over 32767."""

    def __init__(self, delay=0.0, audio_sec=0.1, lead_sec=0.0, tail_sec=0.0, fail_at=None,
                 sec_per_char=None, ratio=None):
        self.delay = delay
        self.audio_sec = audio_sec
        self.sec_per_char = sec_per_char
        self.ratio = ratio
        self.lead_sec = lead_sec
        self.tail_sec = tail_sec
        self.fail_at = fail_at
        self.pieces = []
        self.built = []

    def __call__(self, lang):
        self.built.append(lang)
        return FakeKokoroPipeline(self)

    def audio_seconds(self, segment):
        return len(segment) * self.sec_per_char if self.sec_per_char else self.audio_sec

    def delay_for(self, segment, n):
        if self.ratio is not None:
            return self.ratio * self.audio_seconds(segment)
        return self.delay(n) if callable(self.delay) else self.delay

    def speech(self, segment):
        rng = np.random.default_rng(zlib.crc32(segment.encode()))
        n = int(self.audio_seconds(segment) * SAMPLE_RATE)
        return (rng.integers(1, 20000, n) * rng.choice([-1, 1], n)).astype(np.int16)

    def samples(self, segment):
        def silence(sec):
            return np.zeros(int(sec * SAMPLE_RATE), dtype=np.int16)

        return np.concatenate([silence(self.lead_sec), self.speech(segment), silence(self.tail_sec)])

    def voices(self):
        return [voice for _, voice, _, _ in self.pieces]

    def received(self, count=None):
        """The int16 stream a player must be fed for the first ``count`` rendered pieces
        when no join is trimmed."""
        chosen = self.pieces[:count]
        return np.concatenate([p[2] for p in chosen]) if chosen else np.zeros(0, np.int16)


class FakeKokoroPipeline:
    model = None

    def __init__(self, factory):
        self.factory = factory

    def __call__(self, text, voice, speed):
        f = self.factory
        for segment in [s for s in text.split("\n") if s.strip()]:
            n = len(f.pieces)
            time.sleep(f.delay_for(segment, n))
            if f.fail_at == n:
                raise RuntimeError("fake synthesis failure")
            ints = f.samples(segment)
            f.pieces.append((segment, voice, ints, time.monotonic()))
            yield type("Result", (), {"audio": ints.astype(np.float32) / 32767.0,
                                      "sample_rate": SAMPLE_RATE, "graphemes": segment})()
