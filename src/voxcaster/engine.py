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
        # lang_code -> KPipeline. A KPipeline is bound to one lang_code at
        # construction, and a per-request voice can be from either family, so a
        # single slot means every cross-family voice swap is a full rebuild.
        # ⊨ MEASURED 2026-07-30: rebuild costs 2807ms (lang a) / 2320ms (lang b),
        #   and daemon.log recorded 167 `voice lang_code changed` events, i.e.
        #   ~428s of pure reconstruction, plus a forced CPU->GPU move each time
        #   because the rebuild reset the device flag. Worse than the cost: a
        #   report ends up delivered in alternating voices AND accents
        #   (American af_heart vs British bf_emma), which is what "the voice
        #   sounds robotic" actually was. Per-clip logs all reported success
        #   with a plausible duration, so it was invisible.
        # Bounded by construction: _lang_for_voice only ever returns 'a' or 'b',
        # so the cache holds at most 2 Kokoro-82M models (~660MB fp32 total).
        self._pipelines: dict[str, object] = {}
        # lang_codes whose model currently sits on the GPU (holding VRAM).
        self._gpu_langs: set[str] = set()
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

    @staticmethod
    def _lang_for_voice(voice: str | None) -> str:
        """Kokoro lang_code implied by a voice id ('a'=American, 'b'=British).

        Voice ids are ``<lang><gender>_<name>`` (af_heart, bm_george). Anything
        unrecognised falls back to 'a', which is also what the pipeline default
        would have been.
        """
        return voice[0] if voice and voice[0] in ("a", "b") else "a"

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

    def idle_seconds(self) -> float:
        """Seconds since the last completed synthesis request."""
        return time.monotonic() - self._last_used

    def _pipeline_locked(self, lang: str):
        """Cached KPipeline for ``lang``, constructing it once. Lock must be held.

        This is the whole point of the cache: a cross-family voice swap must be
        a dict lookup, not a 2.3-2.8s reconstruction that also drops the model
        back to the CPU.
        """
        pipe = self._pipelines.get(lang)
        if pipe is not None:
            return pipe
        try:
            from kokoro import KPipeline
            # Always construct on CPU — moved to GPU per-request by policy.
            pipe = KPipeline(lang_code=lang, device="cpu")
        except Exception as e:
            raise SynthesisError(
                f"failed to load Kokoro pipeline (lang_code={lang!r}): {e}") from e
        self._pipelines[lang] = pipe
        log.info("Kokoro pipeline constructed for lang_code=%r on CPU; cache now %s",
                 lang, sorted(self._pipelines))
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
            idle = time.monotonic() - self._last_used
            if idle < self.keepalive_seconds:
                return False
            log.info("idle %.0fs >= %ds keepalive — offloading model(s) to CPU",
                     idle, self.keepalive_seconds)
            self._to_cpu_locked()
            return not self._gpu_langs

    # ── synthesis ─────────────────────────────────────────────────────────

    # PROSODY / DEFENCE-IN-DEPTH ONLY. This is NOT a truncation fix.
    #
    # It was introduced on the theory that KPipeline's split_pattern=r'\n+'
    # (newlines only) made a long single-paragraph line arrive as ONE chunk and
    # get truncated at the model's token limit. That theory is FALSIFIED.
    #
    # ⊨ MEASURED 2026-07-30, controlled two-arm, 2531-char newline-free input,
    #   identical except for this function:
    #     without chunking: 139.2s of audio @ 24000Hz (18.2 chars/sec)
    #     with    chunking: 147.4s of audio @ 24000Hz (17.2 chars/sec)
    #   ratio 1.06x. Both durations are full-length for the input, so nothing
    #   was ever being lost here. Corroborated by daemon.log, which shows
    #   pre-fix clips of 171.8s / 126.7s / 91.2s all completing with rc=0.
    #
    # ■ MECHANISM (kokoro/pipeline.py:195-220): split_pattern only builds the
    #   OUTER graphemes. Each grapheme is then sub-batched internally by
    #   KPipeline.en_tokenize at a 510-phoneme boundary, and every batch is
    #   yielded as its own Result — which synthesize() below concatenates. The
    #   only real truncation in Kokoro (pipeline.py:275-278, "Unexpected
    #   len(ps) ... Truncating to 510") is reachable from
    #   generate_from_tokens(), never from __call__().
    #
    # The actual silent loss was elsewhere and is fixed elsewhere: the wire
    # protocol cut every payload at its first newline (see protocol.py's
    # "Newlines inside the payload" — measured 61% loss on a 3-paragraph line),
    # and tools/voice/speak_last_response.py truncated spans at 320 chars.
    #
    # Kept, not reverted, for two reasons that do not depend on the false
    # theory: it makes chunk boundaries deterministic and sentence-aligned
    # rather than phoneme-count-aligned (audibly better breaks), and it holds
    # the line if a future kokoro drops the internal 510-phoneme batching.
    # Its cost is the +6% duration above, from inter-chunk silence.
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
            # Voice lang_code switching: a KPipeline is bound to one lang_code
            # ('a'=American, 'b'=British) at construction, so a voice from the
            # other family needs that family's pipeline — otherwise the line is
            # spoken with the wrong accent/phoneme model. Previously this
            # RECONSTRUCTED the pipeline every swap; now it is a cache lookup.
            if lang != self.lang_code:
                log.info("active lang_code %s->%s (voice=%s) — %s",
                         self.lang_code, lang, voice,
                         "cache hit" if lang in self._pipelines else "first use, constructing")
                self.lang_code = lang
            pipeline = self._pipeline_locked(lang)
            if (self.dynamic_offload or self.always_gpu) and lang not in self._gpu_langs:
                self._to_gpu_locked(lang)  # failure is logged; CPU synthesis continues

            try:
                chunks: list[np.ndarray] = []
                sample_rate = DEFAULT_SAMPLE_RATE
                chunked = self._chunk_for_pipeline(text)
                for result in pipeline(chunked, voice=voice, speed=self.speed):
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
