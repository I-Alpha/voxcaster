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
import threading
import time

import numpy as np

log = logging.getLogger("speakd.engine")

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

    def synthesize(self, text: str) -> tuple[np.ndarray, int]:
        """Synthesize ``text`` to a mono float waveform.

        Returns ``(samples, sample_rate)``. Raises :class:`SynthesisError`
        on any failure. Blocking — call from a thread executor.
        """
        with self._lock:
            self._load_locked()
            if (self.dynamic_offload or self.always_gpu) and not self._on_gpu:
                self._to_gpu_locked()  # failure is logged; CPU synthesis continues
            try:
                chunks: list[np.ndarray] = []
                sample_rate = DEFAULT_SAMPLE_RATE
                for result in self._pipeline(text, voice=self.voice, speed=self.speed):
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
