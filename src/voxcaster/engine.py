"""Kokoro TTS engine with a CPU <-> GPU device policy.

Device policies:

``cpu``
    The model lives on the CPU permanently. CUDA is hidden from the process
    before torch can be imported, so the daemon never touches the GPU.

``gpu``
    The model moves to the GPU on first use and stays there (no offload).

``auto`` (default)
    Dynamic offload: the model is loaded on the CPU, hops onto the GPU for
    synthesis bursts, and is moved back to the CPU after
    ``keepalive_seconds`` without a request — releasing its VRAM (~3 GB)
    for other workloads such as model training. If a GPU move fails
    (e.g. another process holds all VRAM), synthesis continues on the CPU
    for that request instead of erroring.

One Kokoro pipeline (a private model plus the language's G2P, about 0.5 GB of
RAM) is kept per voice language. At most ``max_resident_pipelines`` stay
loaded, least recently used evicted first, and a pipeline unused for
``idle_unload_seconds`` is unloaded; both return the freed heap to the OS.

All device moves, evictions and synthesis are serialised by an internal lock. A
synthesis stream holds it from its first piece until the stream ends or is
closed, so the idle-offload timer can never move a model, and a second request
can never use a pipeline, while a line is still being rendered.
"""
from __future__ import annotations

import ctypes
import gc
import logging
import os
import re
import threading
import time
from collections import OrderedDict
from collections.abc import Callable, Iterator
from dataclasses import dataclass

import numpy as np

log = logging.getLogger("voxcaster.engine")

# Kokoro's native output sample rate; individual results may override it.
DEFAULT_SAMPLE_RATE = 24000

# Characters per chunk when rendering a whole line to a file.
RENDER_CHUNK_CHARS = 320


class SynthesisError(RuntimeError):
    """TTS synthesis failed (model unavailable, OOM, bad input, ...).

    Callers should treat this as 'use the fallback engine', not as fatal.
    ``chunk_index`` is the chunk of the plan being rendered when it failed.
    """

    def __init__(self, message: str, chunk_index: int = 0) -> None:
        super().__init__(message)
        self.chunk_index = chunk_index


@dataclass(frozen=True)
class Chunk:
    """One run of text handed to the pipeline. ``paragraph_start`` marks the first
    chunk of every paragraph after the first."""
    text: str
    paragraph_start: bool


@dataclass(frozen=True)
class Piece:
    """What the pipeline yielded for one result. ``paragraph_break`` is True when
    the silence before it is an authored paragraph break rather than a join
    between chunks of one paragraph."""
    samples: np.ndarray
    sample_rate: int
    chunk_index: int
    paragraph_break: bool
    chars: int
    compute_seconds: float


KOKORO_LANG_CODES = "abefhijpz"


def load_kokoro_pipeline(lang: str):
    """Construct a Kokoro pipeline for ``lang`` on the CPU."""
    from kokoro import KPipeline
    return KPipeline(lang_code=lang, device="cpu")


def release_freed_memory() -> None:
    """Hand freed heap back to the OS (collect garbage, drop the CUDA cache, ``malloc_trim``)."""
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except Exception as e:
        log.warning("torch.cuda.empty_cache() failed: %s", e)
    try:
        ctypes.CDLL("libc.so.6").malloc_trim(0)
    except (OSError, AttributeError):
        log.warning("malloc_trim unavailable; freed heap stays with the process")


class KokoroEngine:
    def __init__(
        self,
        *,
        voice: str,
        speed: float = 1.0,
        lang_code: str = "a",
        policy: str = "auto",
        keepalive_seconds: int = 180,
        max_resident_pipelines: int = 2,
        idle_unload_seconds: float = 600.0,
        pipeline_factory: Callable[[str], object] | None = None,
        clock: Callable[[], float] = time.monotonic,
    ) -> None:
        if max_resident_pipelines < 1:
            raise ValueError(f"max_resident_pipelines must be >= 1, got {max_resident_pipelines}")
        if idle_unload_seconds <= 0:
            raise ValueError(f"idle_unload_seconds must be > 0, got {idle_unload_seconds}")
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.keepalive_seconds = keepalive_seconds
        self.max_resident_pipelines = max_resident_pipelines
        self.idle_unload_seconds = idle_unload_seconds
        self._pipeline_factory = pipeline_factory or load_kokoro_pipeline
        self._clock = clock

        # Lock protecting ALL device moves and synthesis calls. A plain
        # threading.Lock is correct here: synthesis runs in a thread executor,
        # and the idle-offload path also runs in an executor, never on the
        # asyncio event loop itself.
        self._lock = threading.Lock()
        self._pipelines: OrderedDict[str, object] = OrderedDict()  # least recently used first
        self._pipeline_last_used: dict[str, float] = {}
        # lang_codes whose model currently sits on the GPU (holding VRAM).
        self._gpu_langs: set[str] = set()
        self._last_used = 0.0  # self._clock() of the last completed request

        if policy == "cpu":
            # Hide CUDA before torch is ever imported.
            os.environ["CUDA_VISIBLE_DEVICES"] = ""
        self.gpu_available = self._cuda_available() if policy != "cpu" else False
        self.always_gpu = policy == "gpu" and self.gpu_available
        self.dynamic_offload = policy == "auto" and self.gpu_available
        if policy == "gpu" and not self.gpu_available:
            log.warning("device policy 'gpu' requested but CUDA is unavailable — using CPU")
        log.info(
            "device policy=%s cuda=%s dynamic_offload=%s keepalive=%ds",
            policy, self.gpu_available, self.dynamic_offload, keepalive_seconds,
        )

    # ── device management ─────────────────────────────────────────────────

    @staticmethod
    def _cuda_available() -> bool:
        try:
            import torch
            return torch.cuda.is_available()
        except Exception:
            return False

    @staticmethod
    def _lang_for_voice(voice: str | None) -> str:
        """Kokoro lang_code implied by a voice id.

        Voice ids are ``<lang><gender>_<name>`` (af_heart, bm_george, ef_dora).
        Anything unrecognised falls back to 'a'.
        """
        return voice[0] if voice and voice[0] in KOKORO_LANG_CODES else "a"

    @property
    def on_gpu(self) -> bool:
        """Whether ANY cached model currently holds VRAM (lock-free read).

        'Any', not 'the active one': the idle-offload timer uses this to decide
        whether there is VRAM to reclaim, and a warm-but-inactive pipeline on
        the GPU is exactly as much VRAM as an active one.
        """
        return bool(self._gpu_langs)

    def is_ready(self) -> bool:
        """Whether at least one Kokoro pipeline has been loaded (lock-free read).

        Pipelines load lazily on first use, so this is False until at least one
        line has been spoken."""
        return bool(self._pipelines)

    def resident_langs(self) -> list[str]:
        """Languages with a loaded pipeline, least recently used first."""
        return list(self._pipelines)

    def idle_unload_due(self) -> bool:
        """Whether some pipeline has been unused for ``idle_unload_seconds`` (lock-free read)."""
        now = self._clock()
        return any(now - used >= self.idle_unload_seconds
                   for used in list(self._pipeline_last_used.values()))

    def unload_idle(self) -> list[str]:
        """Unload every pipeline unused for ``idle_unload_seconds``; returns their languages.

        Takes the synthesis lock, so a pipeline is never unloaded mid-render."""
        with self._lock:
            now = self._clock()
            idle = [lang for lang in self._pipelines
                    if now - self._pipeline_last_used[lang] >= self.idle_unload_seconds]
            for lang in idle:
                self._evict_locked(lang)
            if idle:
                release_freed_memory()
                log.info("unloaded idle pipelines %s (unused >= %gs); resident now %s",
                         idle, self.idle_unload_seconds, self.resident_langs())
            return idle

    def idle_seconds(self) -> float:
        """Seconds since the last completed synthesis request."""
        return self._clock() - self._last_used

    def _evict_locked(self, lang: str) -> None:
        """Drop ``lang``'s pipeline and its GPU residency record. Lock must be held."""
        del self._pipelines[lang]
        del self._pipeline_last_used[lang]
        self._gpu_langs.discard(lang)

    def _pipeline_locked(self, lang: str):
        """Resident pipeline for ``lang``, loading it if absent. Lock must be held.

        Marks ``lang`` most recently used. A load first evicts least recently
        used pipelines until fewer than ``max_resident_pipelines`` remain, so
        the cap holds during the load too.
        """
        pipe = self._pipelines.get(lang)
        if pipe is None:
            evicted = []
            while len(self._pipelines) >= self.max_resident_pipelines:
                oldest = next(iter(self._pipelines))
                self._evict_locked(oldest)
                evicted.append(oldest)
            if evicted:
                release_freed_memory()
                log.info("evicted least recently used pipelines %s (cap %d)",
                         evicted, self.max_resident_pipelines)
            try:
                pipe = self._pipeline_factory(lang)
            except Exception as e:
                raise SynthesisError(
                    f"failed to load Kokoro pipeline (lang_code={lang!r}): {e}") from e
            self._pipelines[lang] = pipe
            log.info("Kokoro pipeline constructed for lang_code=%r on CPU; resident now %s",
                     lang, self.resident_langs())
        self._pipelines.move_to_end(lang)
        self._pipeline_last_used[lang] = self._clock()
        return pipe

    def _to_gpu_locked(self, lang: str) -> bool:
        """Move ``lang``'s model to the GPU. False on OOM/unavailable.
        Lock must be held."""
        if lang in self._gpu_langs:
            return True
        pipe = self._pipelines.get(lang)
        if not self.gpu_available or pipe is None or pipe.model is None:
            return False
        try:
            pipe.model.to("cuda")
            self._gpu_langs.add(lang)
            log.info("model moved to GPU (lang_code=%r; on GPU: %s)",
                     lang, sorted(self._gpu_langs))
            return True
        except RuntimeError as e:
            # OOM or similar — stay on CPU for this request.
            log.warning("model.to('cuda') failed for lang_code=%r (OOM?) — "
                        "staying on CPU: %s", lang, e)
            return False

    def _to_cpu_locked(self) -> None:
        """Move EVERY cached model to the CPU and release VRAM. Lock must be held.

        All of them: offloading only the active pipeline would leave a warm
        inactive one pinning VRAM that the offload exists to reclaim.
        """
        moved = []
        for lang in sorted(self._gpu_langs):
            pipe = self._pipelines.get(lang)
            if pipe is None or pipe.model is None:
                continue
            try:
                pipe.model.to("cpu")
                moved.append(lang)
            except Exception as e:
                log.warning("model.to('cpu') failed for lang_code=%r: %s", lang, e)
        for lang in moved:
            self._gpu_langs.discard(lang)
        if moved:
            try:
                import torch
                torch.cuda.empty_cache()
            except Exception as e:
                log.warning("torch.cuda.empty_cache() failed: %s", e)
            log.info("models offloaded to CPU; VRAM released (lang_codes=%s)", moved)

    def maybe_offload(self) -> bool:
        """Offload the model to the CPU if it has been idle for at least
        ``keepalive_seconds``. Thread-safe; returns True if a move happened.

        Blocks while a synthesis holds the lock, then re-checks idleness —
        so a request that finished while we waited resets the timer."""
        if not self.dynamic_offload:
            return False
        with self._lock:
            if not self._gpu_langs:
                return False
            idle = self._clock() - self._last_used
            if idle < self.keepalive_seconds:
                return False
            log.info("idle %.0fs >= %ds keepalive — offloading model(s) to CPU",
                     idle, self.keepalive_seconds)
            self._to_cpu_locked()
            return not self._gpu_langs

    # ── synthesis ─────────────────────────────────────────────────────────

    # Chunking keeps breaks on sentence boundaries. Kokoro batches long text
    # itself (KPipeline.en_tokenize, 510 phonemes), so nothing is lost without it.
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?:;])\s+(?=[A-Z0-9"\'(])')

    @classmethod
    def plan_chunks(cls, text: str, budget: int, min_chars: int = 0) -> list[Chunk]:
        """Split ``text`` into the chunks the pipeline renders one at a time.

        Each newline-separated paragraph is packed into chunks of whole sentences
        of at most ``budget`` characters; a sentence longer than the budget stays
        whole. With ``min_chars``, a chunk shorter than that is joined to the next
        chunk of its paragraph, or to the previous one when it is the last, so a
        chunk can exceed the budget by up to ``min_chars``. A paragraph shorter
        than ``min_chars`` is never joined to its neighbours.
        """
        chunks: list[Chunk] = []
        for para in text.split('\n'):
            para = para.strip()
            if not para:
                continue
            pieces = cls._pack_sentences(para, budget)
            if min_chars:
                pieces = cls._join_short(pieces, min_chars)
            for n, piece in enumerate(pieces):
                chunks.append(Chunk(piece, paragraph_start=n == 0 and bool(chunks)))
        return chunks

    @classmethod
    def _pack_sentences(cls, para: str, budget: int) -> list[str]:
        if len(para) <= budget:
            return [para]
        pieces: list[str] = []
        cur = ''
        for sent in cls._SENTENCE_SPLIT.split(para):
            if not sent:
                continue
            if cur and len(cur) + 1 + len(sent) > budget:
                pieces.append(cur)
                cur = sent
            else:
                cur = f'{cur} {sent}' if cur else sent
        if cur:
            pieces.append(cur)
        return pieces

    @staticmethod
    def _join_short(pieces: list[str], min_chars: int) -> list[str]:
        joined: list[str] = []
        carry = ''
        for n, piece in enumerate(pieces):
            if carry:
                piece, carry = f'{carry} {piece}', ''
            if len(piece) >= min_chars:
                joined.append(piece)
            elif n < len(pieces) - 1:
                carry = piece
            elif joined:
                joined[-1] += f' {piece}'
            else:
                joined.append(piece)
        return joined

    @staticmethod
    def _result_samples(result) -> np.ndarray | None:
        # Explicit None-checks: truth-testing a tensor raises
        # "Boolean value of Tensor is ambiguous".
        audio = None
        if getattr(result, "audio", None) is not None:
            audio = result.audio
        elif getattr(result, "wav", None) is not None:
            audio = result.wav
        if audio is None:
            return None
        arr = audio.squeeze()
        return arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr)

    def stream(self, chunks: list[Chunk], voice: str | None = None) -> Iterator[Piece]:
        """Render ``chunks`` and yield each piece as the pipeline produces it.

        ``voice`` overrides the engine's default voice for this request only
        (falls back to ``self.voice`` when None).

        The synthesis lock is taken when the first piece is requested and held
        until the generator finishes or is closed, so close it promptly when
        abandoning a stream; it is released after at most the piece being
        rendered. Raises :class:`SynthesisError` on any failure, including a plan
        that yields no audio. Blocking: iterate from a worker thread.
        """
        voice = voice or self.voice
        # CROSS-CALLER CONTENTION GUARD. One daemon serves every caller on the
        # box, and each request may carry its own voice. When callers disagree,
        # consecutive lines of what the listener experiences as one report come
        # out in different voices and different accents. That is not visible in
        # any per-clip log line — each clip reports success with a plausible
        # duration — so make the disagreement itself loud.
        if voice != self.voice:
            log.warning("[voice-contention] request voice=%r != configured default=%r "
                        "— consecutive lines will not match; if this alternates, a "
                        "shared listener hears one report in several speakers",
                        voice, self.voice)
        lang = self._lang_for_voice(voice)
        with self._lock:
            if lang != self.lang_code:
                log.info("active lang_code %s->%s (voice=%s) — %s",
                         self.lang_code, lang, voice,
                         "cache hit" if lang in self._pipelines else "first use, constructing")
                self.lang_code = lang
            pipeline = self._pipeline_locked(lang)
            if (self.dynamic_offload or self.always_gpu) and lang not in self._gpu_langs:
                self._to_gpu_locked(lang)  # failure is logged; CPU synthesis continues

            index = 0
            produced = 0
            try:
                for index, chunk in enumerate(chunks):
                    results = iter(pipeline(chunk.text, voice=voice, speed=self.speed))
                    first_in_chunk = True
                    while True:
                        began = time.perf_counter()
                        result = next(results, None)
                        compute = time.perf_counter() - began
                        if result is None:
                            break
                        samples = self._result_samples(result)
                        if samples is None:
                            continue
                        graphemes = getattr(result, "graphemes", None)
                        yield Piece(
                            samples=samples,
                            sample_rate=getattr(result, "sample_rate", None) or DEFAULT_SAMPLE_RATE,
                            chunk_index=index,
                            paragraph_break=first_in_chunk and chunk.paragraph_start,
                            chars=len(graphemes) if graphemes else (len(chunk.text) if first_in_chunk else 0),
                            compute_seconds=compute,
                        )
                        produced += 1
                        first_in_chunk = False
                if not produced:
                    raise SynthesisError("no audio chunks produced")
            except SynthesisError:
                raise
            except Exception as e:
                raise SynthesisError(str(e), chunk_index=index) from e
            finally:
                # Stamp even on failure so the offload timer measures real
                # activity, not just successes.
                self._last_used = self._clock()
                if lang in self._pipeline_last_used:
                    self._pipeline_last_used[lang] = self._last_used

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` to a mono float waveform.

        Returns ``(samples, sample_rate)``. Raises :class:`SynthesisError`
        on any failure. Blocking: call from a thread executor.
        """
        pieces = list(self.stream(self.plan_chunks(text, RENDER_CHUNK_CHARS), voice))
        return np.concatenate([p.samples for p in pieces]), pieces[-1].sample_rate
