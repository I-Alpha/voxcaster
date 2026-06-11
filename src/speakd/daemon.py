"""The speakd daemon: a fire-and-forget TTS narration queue behind a Unix socket.

Design
------
- One asyncio Unix-socket server accepts newline-terminated requests
  (see :mod:`speakd.protocol`) and feeds a FIFO queue.
- A single worker drains the queue, synthesizing each line with Kokoro in a
  thread executor (the event loop never blocks on the model) and playing it
  through an external audio player.
- An interrupt request drains everything queued, kills in-flight playback,
  and jumps its own text to the front.
- An idle timer offloads the model from GPU to CPU after a configurable
  keepalive, releasing VRAM for other workloads (see :mod:`speakd.engine`).
- A flock(2) singleton guarantees at most one daemon per socket path, so
  clients may race to auto-spawn it without harm.
- If synthesis or playback fails, the line is spoken through a configurable
  fallback engine (espeak by default) — narration degrades, never disappears.

Run it with ``speakd`` (installed console script) or ``python -m speakd.daemon``.
"""
from __future__ import annotations

import argparse
import asyncio
import fcntl
import logging
import os
import signal
import socket as socket_module
import subprocess
import sys
import tempfile
import threading
import time

from . import __version__, protocol
from .config import Config, load_config
from .engine import KokoroEngine, SynthesisError

log = logging.getLogger("speakd.daemon")

# Seconds a connected client gets to deliver its single request line.
READLINE_TIMEOUT_SEC = 5.0


class VoiceDaemon:
    def __init__(self, cfg: Config) -> None:
        self.cfg = cfg
        self.engine = KokoroEngine(
            voice=cfg.voice,
            speed=cfg.speed,
            lang_code=cfg.lang_code,
            policy=cfg.device,
            keepalive_seconds=cfg.keepalive_seconds,
        )
        self.volume = protocol.clamp_volume(cfg.volume)

        self._queue: asyncio.Queue | None = None
        self._stop: asyncio.Event | None = None

        # The live audio-player process (or None). Written by _speak_sync in a
        # thread executor; read and killed by the interrupt handler on the
        # event loop — hence the lock.
        self._current_player: subprocess.Popen | None = None
        self._player_lock = threading.Lock()

        # Bumped on every interrupt. A line that started before the bump
        # skips its playback: an urgent line must not wait behind a stale
        # one that happened to be mid-synthesis when the interrupt arrived.
        # (int reads/writes are atomic under the GIL — no extra lock needed.)
        self._interrupt_epoch = 0

    # ── synthesis + playback (blocking; runs in a thread executor) ────────

    def _speak_sync(self, text: str) -> None:
        epoch = self._interrupt_epoch
        try:
            wav, sample_rate = self.engine.synthesize(text)
        except SynthesisError as e:
            log.warning("synthesis failed — using fallback: %s", e)
            self._speak_fallback(text)
            return
        if self._interrupt_epoch != epoch:
            log.info("dropped superseded line (interrupted mid-synthesis): %s", text[:60])
            return
        self._play(wav, sample_rate, text)

    def _play(self, wav, sample_rate: int, text: str) -> None:
        import soundfile as sf
        try:
            with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
                sf.write(f.name, wav, sample_rate)
                tmp_path = f.name
            argv = self._render_player_argv(tmp_path)
            proc = subprocess.Popen(
                argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
            )
            # Track the player so an interrupt can kill it mid-line.
            with self._player_lock:
                self._current_player = proc
            try:
                proc.wait(timeout=self.cfg.max_playback_seconds)
            except subprocess.TimeoutExpired:
                proc.kill()
                proc.wait()
            with self._player_lock:
                if self._current_player is proc:
                    self._current_player = None
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            if proc.returncode == 0:
                log.info("spoke (%dHz, %.1fs): %s",
                         sample_rate, len(wav) / sample_rate, text[:60])
            else:
                log.info("playback cut short (player rc=%s): %s",
                         proc.returncode, text[:60])
        except Exception as e:
            log.warning("audio playback failed — using fallback: %s", e)
            self._speak_fallback(text)

    def _render_player_argv(self, wav_path: str) -> list[str]:
        argv = [a.format(file=wav_path, volume=self.volume) for a in self.cfg.player]
        if not any("{file}" in a for a in self.cfg.player):
            argv.append(wav_path)  # tolerate templates without a {file} slot
        return argv

    def _speak_fallback(self, text: str) -> None:
        """Speak through the fallback engine (non-blocking fire-and-forget)."""
        if not self.cfg.fallback:
            return  # fallback disabled by config
        argv = [a.format(text=text) for a in self.cfg.fallback]
        if not any("{text}" in a for a in self.cfg.fallback):
            argv.append(text)
        try:
            subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        except (OSError, FileNotFoundError) as e:
            log.error("fallback engine %r failed: %s", argv[0], e)

    def _kill_current_player(self) -> bool:
        """Terminate in-flight playback, if any. Thread-safe."""
        with self._player_lock:
            proc = self._current_player
        if proc is not None and proc.poll() is None:
            proc.terminate()
            return True
        return False

    # ── async plumbing ─────────────────────────────────────────────────────

    async def _worker(self, loop: asyncio.AbstractEventLoop) -> None:
        """Drain the queue one line at a time; ack each client when spoken."""
        while not self._stop.is_set():
            try:
                text, writer = await asyncio.wait_for(self._queue.get(), timeout=1.0)
            except asyncio.TimeoutError:
                continue
            try:
                await loop.run_in_executor(None, self._speak_sync, text)
            except Exception:
                # _speak_sync handles its own fallbacks; this guards the worker
                # task against anything unexpected so the queue never stalls.
                log.exception("unexpected error while speaking %r", text[:60])
            try:
                writer.write(protocol.ACK)
                await writer.drain()
                writer.close()
            except Exception:
                pass  # fire-and-forget client already hung up
            self._queue.task_done()

    async def _interrupt(self, text: str, writer: asyncio.StreamWriter) -> None:
        """Drain pending lines, cut off playback, speak ``text`` next."""
        self._interrupt_epoch += 1  # marks any in-flight synthesis as superseded
        drained = 0
        while True:
            try:
                _old_text, old_writer = self._queue.get_nowait()
                self._queue.task_done()
                drained += 1
                try:
                    old_writer.close()
                except Exception:
                    pass
            except asyncio.QueueEmpty:
                break
        killed = self._kill_current_player()
        log.info("[interrupt] drained %d queued, stopped current=%s: %s",
                 drained, killed, text[:60])
        await self._queue.put((text, writer))

    async def _idle_offload(self, loop: asyncio.AbstractEventLoop) -> None:
        """Move the model off the GPU after ``keepalive_seconds`` of quiet."""
        if not self.engine.dynamic_offload:
            return
        poll = min(30.0, max(1.0, self.engine.keepalive_seconds / 4))
        while not self._stop.is_set():
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=poll)
                break  # stop requested
            except asyncio.TimeoutError:
                pass
            if not self.engine.on_gpu:
                continue
            if self.engine.idle_seconds() >= self.engine.keepalive_seconds:
                # Run in an executor: maybe_offload waits on the model lock,
                # which a synthesis may hold — never block the event loop.
                await loop.run_in_executor(None, self.engine.maybe_offload)

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=READLINE_TIMEOUT_SEC)
            text = data.decode(errors="ignore").strip()
            if not text:
                writer.close()
                return
            if text.startswith(protocol.VOLUME_MARKER):
                try:
                    self.volume = protocol.clamp_volume(
                        int(text[len(protocol.VOLUME_MARKER):])
                    )
                    log.info("[volume] set to %d", self.volume)
                except ValueError:
                    pass
                writer.close()
                return
            if text.startswith(protocol.INTERRUPT_MARKER):
                payload = text[len(protocol.INTERRUPT_MARKER):]
                if payload:
                    await self._interrupt(payload, writer)
                else:
                    writer.close()
            else:
                await self._queue.put((text, writer))
        except Exception:
            writer.close()

    # ── socket lifecycle ───────────────────────────────────────────────────

    def _remove_stale_socket(self) -> None:
        """Unlink the socket file if nothing is listening on it."""
        path = self.cfg.socket_path
        if not os.path.exists(path):
            return
        try:
            s = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_STREAM)
            s.settimeout(0.3)
            s.connect(path)
            s.close()
            # Live listener — the flock singleton should have caught this,
            # but be safe and leave the socket alone.
        except OSError:
            log.info("removing stale socket %s", path)
            try:
                os.unlink(path)
            except OSError:
                pass

    async def run(self) -> None:
        self._queue = asyncio.Queue()
        self._stop = asyncio.Event()
        loop = asyncio.get_running_loop()

        self._remove_stale_socket()
        server = await asyncio.start_unix_server(self._handle, path=self.cfg.socket_path)
        os.chmod(self.cfg.socket_path, self.cfg.socket_mode)
        log.info("speakd %s ready  voice=%s  socket=%s",
                 __version__, self.cfg.voice, self.cfg.socket_path)

        def _shutdown(sig: int) -> None:
            log.info("shutdown (signal %d)", sig)
            self._stop.set()
            server.close()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown, sig)

        worker_task = asyncio.create_task(self._worker(loop))
        offload_task = asyncio.create_task(self._idle_offload(loop))

        async with server:
            await self._stop.wait()

        for task in (worker_task, offload_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if os.path.exists(self.cfg.socket_path):
            os.unlink(self.cfg.socket_path)
        log.info("speakd stopped")


# ── singleton lock ──────────────────────────────────────────────────────────

def acquire_singleton_lock(lock_path: str):
    """Take an exclusive flock on ``lock_path``; exit 0 if another daemon holds it.

    Returns the open file object — the caller must keep it alive for the
    daemon's lifetime. Opened O_RDWR|O_CREAT (no truncate) so a racing second
    process can still read the holder's pid while the lock is held; the winner
    then truncates and writes its own pid.
    """
    raw_fd = os.open(lock_path, os.O_RDWR | os.O_CREAT, 0o644)
    lock_file = os.fdopen(raw_fd, "r+")
    try:
        fcntl.flock(lock_file, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        try:
            pid = lock_file.read().strip()
            log.info("daemon already running (pid %s), exiting", pid)
        except Exception:
            log.info("daemon already running, exiting")
        lock_file.close()
        raise SystemExit(0)
    lock_file.seek(0)
    lock_file.truncate(0)
    lock_file.write(str(os.getpid()))
    lock_file.flush()
    return lock_file


# ── entry point ─────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(
        prog="speakd",
        description="Fire-and-forget TTS narration daemon (Kokoro over a Unix socket).",
    )
    parser.add_argument("--config", metavar="PATH", help="TOML config file")
    parser.add_argument("--socket", metavar="PATH", help="Unix socket path override")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"),
                        help="device policy override")
    parser.add_argument("--voice", help="Kokoro voice id override (e.g. af_heart, bf_emma)")
    parser.add_argument("--volume", type=int, metavar="N",
                        help="initial playback volume (0-130)")
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective configuration and exit")
    parser.add_argument("--version", action="version", version=f"speakd {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    cfg = load_config(args.config)
    if args.socket:
        cfg.socket_path = args.socket
    if args.device:
        cfg.device = args.device
    if args.voice:
        cfg.voice = args.voice
    if args.volume is not None:
        cfg.volume = args.volume

    if args.print_config:
        print(cfg.describe())
        return

    lock_file = acquire_singleton_lock(cfg.lock_path)
    try:
        asyncio.run(VoiceDaemon(cfg).run())
    finally:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            os.unlink(cfg.lock_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
