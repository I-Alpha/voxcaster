"""Rendering a line while it plays.

A :class:`LineSynth` renders one line on its own thread into a bounded buffer of
16-bit PCM. A :class:`FifoFeeder` copies that buffer into a named pipe as a WAV
stream of unknown length, which the player reads like a file. Playback starts
once :func:`ready_to_start` says the rest of the line will arrive in time, and
silence at the joins between chunks of one paragraph is trimmed to a single
sentence pause by :class:`JoinTrimmer`.
"""
from __future__ import annotations

import errno
import logging
import os
import select
import struct
import threading
import time
from collections import deque
from collections.abc import Callable
from dataclasses import dataclass

import numpy as np

from .engine import KokoroEngine, Piece, SynthesisError

log = logging.getLogger("voxcaster.streaming")

# Characters per chunk of a line that is streamed, and the length under which a
# chunk is joined to a neighbour because Kokoro reads isolated fragments badly.
STREAM_CHUNK_CHARS = 100
MIN_CHUNK_CHARS = 25
# Seconds of audio a line may buffer ahead of the player before its renderer waits.
BUFFER_CAP_SEC = 60.0
# The most silence kept at a join inside a paragraph, split evenly between the two
# pieces either side of it.
SENTENCE_PAUSE_SEC = 0.3
# Samples quieter than this are silence (about -54 dBFS).
SILENCE_AMPLITUDE = 0.002
# Safety factor on the measured compute-per-audio ratio when deciding when to start.
RATIO_MARGIN = 1.25
# Below this much rendered audio the previous line's ratio is a floor on this line's.
MIN_RATIO_AUDIO_SEC = 3.0
POLL_SEC = 0.05

PCM_BYTES_PER_SAMPLE = 2
PCM_FULL_SCALE = 32767.0
WAV_UNKNOWN_SIZE = 0xFFFFFFFF


def to_pcm16(samples: np.ndarray) -> bytes:
    scaled = np.rint(samples.astype(np.float64) * PCM_FULL_SCALE)
    return np.clip(scaled, -32768, 32767).astype("<i2").tobytes()


def wav_stream_header(sample_rate: int) -> bytes:
    """Header of a mono 16-bit WAV whose length is not known yet."""
    byte_rate = sample_rate * PCM_BYTES_PER_SAMPLE
    return (b"RIFF" + struct.pack("<I", WAV_UNKNOWN_SIZE) + b"WAVEfmt "
            + struct.pack("<IHHIIHH", 16, 1, 1, sample_rate, byte_rate, PCM_BYTES_PER_SAMPLE, 16)
            + b"data" + struct.pack("<I", WAV_UNKNOWN_SIZE))


class JoinTrimmer:
    """Shortens the silence between the pieces of one paragraph to one pause.

    Each piece keeps at most half of ``SENTENCE_PAUSE_SEC`` of silence at either end
    of an inner join, so a piece can be emitted before the next one exists. The
    silence cut from a piece's tail is held back until the next piece shows whether
    the join was an authored paragraph break, which restores it unchanged. Only
    samples quieter than ``SILENCE_AMPLITUDE`` are ever removed.
    """

    def __init__(self) -> None:
        self._held = np.zeros(0, dtype=np.float32)
        self._first = True

    def feed(self, samples: np.ndarray, sample_rate: int, paragraph_break: bool) -> np.ndarray:
        keep = int(SENTENCE_PAUSE_SEC / 2 * sample_rate)
        parts = []
        if not self._first:
            if paragraph_break:
                parts.append(self._held)
            else:
                samples = samples[max(self._leading_silence(samples) - keep, 0):]
        self._first = False
        cut = max(self._trailing_silence(samples) - keep, 0)
        self._held = samples[len(samples) - cut:]
        parts.append(samples[:len(samples) - cut])
        return np.concatenate(parts)

    def flush(self) -> np.ndarray:
        held, self._held = self._held, np.zeros(0, dtype=np.float32)
        return held

    @staticmethod
    def _leading_silence(samples: np.ndarray) -> int:
        loud = np.flatnonzero(np.abs(samples) > SILENCE_AMPLITUDE)
        return len(samples) if loud.size == 0 else int(loud[0])

    @staticmethod
    def _trailing_silence(samples: np.ndarray) -> int:
        loud = np.flatnonzero(np.abs(samples) > SILENCE_AMPLITUDE)
        return 0 if loud.size == 0 else len(samples) - 1 - int(loud[-1])


@dataclass(frozen=True)
class Progress:
    pieces: int
    audio_sec: float
    compute_sec: float
    chars_done: int
    remaining_chars: tuple[int, ...]
    finished: bool
    buffer_full: bool

    @property
    def ratio(self) -> float:
        """Seconds of compute spent per second of audio rendered."""
        return self.compute_sec / self.audio_sec if self.audio_sec > 0 else 0.0


def ready_to_start(progress: Progress, previous_ratio: float | None) -> bool:
    """Whether playback can begin without the player running out of audio.

    Audio arrives a whole chunk at a time. Playing from now, chunk ``j`` of the
    ``remaining_chars`` lands after ``ratio * margin`` seconds of compute per second
    of audio of every chunk up to and including it, and by then the player has
    consumed that many seconds from the audio rendered so far plus the chunks
    before ``j``. So for every remaining chunk the buffered audio plus the audio of
    the chunks before it must cover that compute. Chunk audio is estimated from
    its characters. A full buffer starts playback regardless, since the renderer
    cannot add audio until the player drains some.
    """
    if progress.pieces == 0:
        return False
    if progress.finished or progress.buffer_full:
        return True
    ratio = progress.ratio
    if progress.audio_sec < MIN_RATIO_AUDIO_SEC and previous_ratio is not None:
        ratio = max(ratio, previous_ratio)
    ratio *= RATIO_MARGIN
    seconds_per_char = progress.audio_sec / max(progress.chars_done, 1)
    before = 0.0
    for chars in progress.remaining_chars:
        through = before + chars * seconds_per_char
        if progress.audio_sec + before < ratio * through:
            return False
        before = through
    return True


class LineSynth:
    """Renders one line on a background thread into a bounded buffer of PCM.

    The thread starts at construction. It holds the engine's synthesis lock for as
    long as it renders and releases it before ``on_done`` runs. ``cancel`` makes it
    stop after the piece being rendered. A failure part-way is kept in ``error``;
    the audio rendered before it is still delivered.
    """

    def __init__(self, engine: KokoroEngine, text: str, voice: str | None,
                 on_done: Callable[["LineSynth"], None]) -> None:
        self.engine = engine
        self.chunks = engine.plan_chunks(text, STREAM_CHUNK_CHARS, MIN_CHUNK_CHARS)
        self.voice = voice
        self.error: SynthesisError | None = None
        self._on_done = on_done
        self._cv = threading.Condition()
        self._buffer: deque[bytes] = deque()
        self._buffered_sec = 0.0
        self._cancelled = False
        self._done = False
        self._sample_rate: int | None = None
        self._pieces = 0
        self._audio_sec = 0.0
        self._compute_sec = 0.0
        self._chars_done = 0
        self._chunk_done = [0] * len(self.chunks)
        self._last_chunk = -1
        self._thread = threading.Thread(target=self._run, name="voxcaster-synth", daemon=True)
        self._thread.start()

    @property
    def sample_rate(self) -> int | None:
        return self._sample_rate

    @property
    def cancelled(self) -> bool:
        return self._cancelled

    def progress(self) -> Progress:
        with self._cv:
            return Progress(self._pieces, self._audio_sec, self._compute_sec,
                            self._chars_done, self._remaining_chars(), self._done,
                            self._buffered_sec >= BUFFER_CAP_SEC)

    def _remaining_chars(self) -> tuple[int, ...]:
        """Characters still to render in each unfinished chunk, the one in progress first."""
        left = []
        for index in range(max(self._last_chunk, 0), len(self.chunks)):
            rest = len(self.chunks[index].text)
            if index == self._last_chunk:
                rest -= self._chunk_done[index]
            if rest > 0:
                left.append(rest)
        return tuple(left)

    def wait_progress(self, timeout: float) -> None:
        with self._cv:
            self._cv.wait(timeout)

    def cancel(self) -> None:
        with self._cv:
            self._cancelled = True
            self._cv.notify_all()

    def take(self, timeout: float) -> tuple[bytes, bool]:
        """Next buffered PCM and whether the line has ended; waits up to ``timeout`` for audio."""
        with self._cv:
            if not self._buffer and not self._done and not self._cancelled:
                self._cv.wait(timeout)
            if self._cancelled:
                return b"", True
            if self._buffer:
                data = self._buffer.popleft()
                self._buffered_sec -= self._seconds(len(data))
                self._cv.notify_all()
                return data, False
            return b"", self._done

    def unspoken_text(self) -> str:
        """The text from the chunk the failure hit onwards, paragraphs kept apart."""
        start = self.error.chunk_index if self.error else len(self.chunks)
        text = ""
        for chunk in self.chunks[start:]:
            if text:
                text += "\n" if chunk.paragraph_start else " "
            text += chunk.text
        return text

    def _seconds(self, nbytes: int) -> float:
        return nbytes / PCM_BYTES_PER_SAMPLE / self._sample_rate

    def _run(self) -> None:
        stream = self.engine.stream(self.chunks, self.voice)
        trimmer = JoinTrimmer()
        try:
            while not self._cancelled:
                piece = next(stream, None)
                if piece is None:
                    self._append(trimmer.flush())
                    break
                self._accept(piece, trimmer)
        except SynthesisError as e:
            self.error = e
            self._append(trimmer.flush())
        except Exception as e:
            self.error = SynthesisError(str(e))
            log.exception("line render failed")
        finally:
            stream.close()
            with self._cv:
                self._done = True
                self._cv.notify_all()
            self._on_done(self)

    def _accept(self, piece: Piece, trimmer: JoinTrimmer) -> None:
        if self._sample_rate is None:
            self._sample_rate = piece.sample_rate
        elif piece.sample_rate != self._sample_rate:
            raise SynthesisError(f"sample rate changed from {self._sample_rate} to "
                                 f"{piece.sample_rate}", piece.chunk_index)
        self._append(trimmer.feed(piece.samples, piece.sample_rate, piece.paragraph_break))
        with self._cv:
            self._pieces += 1
            self._compute_sec += piece.compute_seconds
            self._chars_done += piece.chars
            self._chunk_done[piece.chunk_index] += piece.chars
            self._last_chunk = max(self._last_chunk, piece.chunk_index)
            self._cv.notify_all()

    def _append(self, samples: np.ndarray) -> None:
        if not len(samples):
            return
        data = to_pcm16(samples)
        with self._cv:
            while self._buffered_sec >= BUFFER_CAP_SEC and not self._cancelled:
                self._cv.wait()
            if self._cancelled:
                return
            self._buffer.append(data)
            self._buffered_sec += self._seconds(len(data))
            self._audio_sec += self._seconds(len(data))
            self._cv.notify_all()


class FifoFeeder(threading.Thread):
    """Copies a line's PCM into the named pipe ``path`` behind a WAV stream header.

    Opens the pipe once the player has it open for reading, ends the stream by
    closing it, and stops early when ``stop`` is called or the player is gone.
    ``starved`` is True while the player is waiting on the renderer for audio.
    """

    def __init__(self, path: str, synth: LineSynth, player_alive: Callable[[], bool]) -> None:
        super().__init__(name="voxcaster-feed", daemon=True)
        self._path = path
        self._synth = synth
        self._player_alive = player_alive
        self._stopped = threading.Event()
        self.starved = False

    def stop(self) -> None:
        self._stopped.set()

    def run(self) -> None:
        fd = self._open()
        if fd is None:
            return
        try:
            self._write(fd, wav_stream_header(self._synth.sample_rate))
            while not self._stopped.is_set():
                data, ended = self._synth.take(POLL_SEC)
                self.starved = not data and not ended
                if data:
                    self._write(fd, data)
                elif ended:
                    return
        except OSError as e:
            log.info("player stopped reading: %s", e)
        finally:
            os.close(fd)

    def _open(self) -> int | None:
        while not self._stopped.is_set() and self._player_alive():
            try:
                return os.open(self._path, os.O_WRONLY | os.O_NONBLOCK)
            except OSError as e:
                if e.errno != errno.ENXIO:
                    raise
                time.sleep(POLL_SEC / 5)
        return None

    def _write(self, fd: int, data: bytes) -> None:
        view = memoryview(data)
        while view and not self._stopped.is_set():
            if not select.select([], [fd], [], POLL_SEC)[1]:
                continue
            try:
                view = view[os.write(fd, view):]
            except BlockingIOError:
                continue
