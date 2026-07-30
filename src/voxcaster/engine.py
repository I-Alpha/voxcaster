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

All device moves and synthesis calls are serialised by an internal lock,
so the idle-offload timer can never move the model mid-synthesis.
"""
from __future__ import annotations

import logging
import os
import re
import threading
import time

import numpy as np

log = logging.getLogger("voxcaster.engine")

# Kokoro's native output sample rate; individual results may override it.
DEFAULT_SAMPLE_RATE = 24000


class SynthesisError(RuntimeError):
    """TTS synthesis failed (model unavailable, OOM, bad input, ...).

    Callers should treat this as 'use the fallback engine', not as fatal.
    """


class KokoroEngine:
    def __init__(
        self,
        *,
        voice: str,
        speed: float = 1.0,
        lang_code: str = "a",
        policy: str = "auto",
        keepalive_seconds: int = 180,
    ) -> None:
        self.voice = voice
        self.speed = speed
        self.lang_code = lang_code
        self.keepalive_seconds = keepalive_seconds

        # Lock protecting ALL device moves and synthesis calls. A plain
        # threading.Lock is correct here: synthesis runs in a thread executor,
        # and the idle-offload path also runs in an executor, never on the
        # asyncio event loop itself.
        self._lock = threading.Lock()
        self._pipeline = None
        self._on_gpu = False
        self._last_used = 0.0  # time.monotonic() of the last completed request

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

    @property
    def on_gpu(self) -> bool:
        """Whether the model currently resides on the GPU (lock-free read)."""
        return self._on_gpu

    def is_ready(self) -> bool:
        """Whether the Kokoro pipeline has been loaded (lock-free read).

        The pipeline loads lazily on the first synthesis, so this is False
        until at least one line has been spoken."""
        return self._pipeline is not None

    def idle_seconds(self) -> float:
        """Seconds since the last completed synthesis request."""
        return time.monotonic() - self._last_used

    def _load_locked(self) -> None:
        """Load the Kokoro pipeline onto the CPU. Idempotent. Lock must be held."""
        if self._pipeline is not None:
            return
        try:
            from kokoro import KPipeline
            # Always load onto CPU — moved to GPU per-request by policy.
            self._pipeline = KPipeline(lang_code=self.lang_code, device="cpu")
        except Exception as e:
            raise SynthesisError(f"failed to load Kokoro pipeline: {e}") from e
        self._on_gpu = False
        log.info("Kokoro pipeline loaded on CPU (voice=%s)", self.voice)

    def _to_gpu_locked(self) -> bool:
        """Move the model to the GPU. Returns False on OOM/unavailable.
        Lock must be held."""
        if self._on_gpu:
            return True
        if not self.gpu_available or self._pipeline is None or self._pipeline.model is None:
            return False
        try:
            self._pipeline.model.to("cuda")
            self._on_gpu = True
            log.info("model moved to GPU")
            return True
        except RuntimeError as e:
            # OOM or similar — stay on CPU for this request.
            log.warning("model.to('cuda') failed (OOM?) — staying on CPU: %s", e)
            return False

    def _to_cpu_locked(self) -> None:
        """Move the model to the CPU and release VRAM. Lock must be held."""
        if not self._on_gpu or self._pipeline is None or self._pipeline.model is None:
            return
        try:
            import torch
            self._pipeline.model.to("cpu")
            torch.cuda.empty_cache()
            self._on_gpu = False
            log.info("model offloaded to CPU; VRAM released")
        except Exception as e:
            log.warning("model.to('cpu') failed: %s", e)

    def maybe_offload(self) -> bool:
        """Offload the model to the CPU if it has been idle for at least
        ``keepalive_seconds``. Thread-safe; returns True if a move happened.

        Blocks while a synthesis holds the lock, then re-checks idleness —
        so a request that finished while we waited resets the timer."""
        if not self.dynamic_offload:
            return False
        with self._lock:
            if not self._on_gpu:
                return False
            idle = time.monotonic() - self._last_used
            if idle < self.keepalive_seconds:
                return False
            log.info("idle %.0fs >= %ds keepalive — offloading model to CPU",
                     idle, self.keepalive_seconds)
            self._to_cpu_locked()
            return not self._on_gpu

    # ── synthesis ─────────────────────────────────────────────────────────

    # 2026-07-30: Kokoro's KPipeline splits input with split_pattern=r'\n+',
    # i.e. on NEWLINES ONLY. A long single-paragraph line therefore reaches the
    # model as ONE chunk and is silently truncated at the model's token limit —
    # the caller gets a clean exit, the daemon logs "spoke", and the listener
    # hears the first ~30 seconds of a two-minute report with no error anywhere.
    # Observed live: multi-paragraph status narrations cut off mid-sentence.
    #
    # The playback path was already written to refuse a fixed ceiling (see
    # daemon._speak's max_playback_seconds comment); this closes the same hole
    # on the SYNTHESIS side, which is where the loss actually happened.
    #
    # Fix by inserting newlines the pipeline will split on: break into
    # sentences, then greedily pack them into chunks under a conservative
    # character budget. Packing rather than one-sentence-per-chunk keeps
    # prosody natural — sentence-at-a-time synthesis sounds clipped.
    _SENTENCE_SPLIT = re.compile(r'(?<=[.!?:;])\s+')
    _CHUNK_BUDGET_CHARS = 320

    @classmethod
    def _chunk_for_pipeline(cls, text: str) -> str:
        """Insert newlines so KPipeline chunks long text instead of truncating.

        Existing newlines are preserved as hard breaks (the caller may have
        meant them). Each resulting paragraph is packed into <=
        _CHUNK_BUDGET_CHARS pieces on sentence boundaries. A single sentence
        longer than the budget is left intact rather than cut mid-clause — a
        very long clause is a caller problem, and truncating it here would
        reintroduce exactly the silent loss this exists to prevent.
        """
        out: list[str] = []
        for para in text.split('\n'):
            para = para.strip()
            if not para:
                continue
            if len(para) <= cls._CHUNK_BUDGET_CHARS:
                out.append(para)
                continue
            cur = ''
            for sent in cls._SENTENCE_SPLIT.split(para):
                if not sent:
                    continue
                if cur and len(cur) + 1 + len(sent) > cls._CHUNK_BUDGET_CHARS:
                    out.append(cur)
                    cur = sent
                else:
                    cur = f'{cur} {sent}' if cur else sent
            if cur:
                out.append(cur)
        return '\n'.join(out)

    def synthesize(self, text: str, voice: str | None = None) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` to a mono float waveform.

        ``voice`` overrides the engine's default voice for this request only
        (falls back to ``self.voice`` when None).

        Returns ``(samples, sample_rate)``. Raises :class:`SynthesisError`
        on any failure. Blocking — call from a thread executor.
        """
        voice = voice or self.voice
        with self._lock:
            self._load_locked()
            if (self.dynamic_offload or self.always_gpu) and not self._on_gpu:
                self._to_gpu_locked()  # failure is logged; CPU synthesis continues
            # Voice lang_code switching: Kokoro's KPipeline is created with
            # a fixed lang_code ('a'=American, 'b'=British). When the requested
            # voice has a different first-letter lang_code than the current
            # pipeline, we must recreate the pipeline with the new lang_code.
            # Without this, switching from bf_emma (British) to am_michael
            # (American) speaks with the wrong accent/phoneme model.
            new_lang = voice[0] if voice and voice[0] in ('a', 'b') else 'a'
            if new_lang != self.lang_code:
                log.info("voice lang_code changed %s->%s (voice=%s)",
                         self.lang_code, new_lang, voice)
                from kokoro import KPipeline
                self._pipeline = KPipeline(lang_code=new_lang, device="cpu")
                self.lang_code = new_lang
                self._on_gpu = False

            try:
                chunks: list[np.ndarray] = []
                sample_rate = DEFAULT_SAMPLE_RATE
                chunked = self._chunk_for_pipeline(text)
                for result in self._pipeline(chunked, voice=voice, speed=self.speed):
                    # Explicit None-checks: truth-testing a tensor raises
                    # "Boolean value of Tensor is ambiguous".
                    audio = None
                    if getattr(result, "audio", None) is not None:
                        audio = result.audio
                    elif getattr(result, "wav", None) is not None:
                        audio = result.wav
                    if audio is not None:
                        arr = audio.squeeze()
                        chunks.append(arr.cpu().numpy() if hasattr(arr, "cpu") else np.asarray(arr))
                    if getattr(result, "sample_rate", None):
                        sample_rate = result.sample_rate
                if not chunks:
                    raise RuntimeError("no audio chunks produced")
                wav = np.concatenate(chunks)
            except Exception as e:
                raise SynthesisError(str(e)) from e
            finally:
                # Stamp even on failure so the offload timer measures real
                # activity, not just successes.
                self._last_used = time.monotonic()
        return wav, sample_rate
