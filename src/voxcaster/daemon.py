"""The voxcaster daemon: a fire-and-forget TTS narration queue behind a Unix socket.

Design
------
- One asyncio Unix-socket server accepts newline-terminated requests
  (see :mod:`voxcaster.protocol`) and feeds a FIFO queue.
- A single worker drains the queue, synthesizing each line with Kokoro in a
  thread executor (the event loop never blocks on the model) and playing it
  through an external audio player.
- An interrupt request drains everything queued, kills in-flight playback,
  and jumps its own text to the front.
- An idle timer offloads the model from GPU to CPU after a configurable
  keepalive, releasing VRAM for other workloads (see :mod:`voxcaster.engine`).
- A flock(2) singleton guarantees at most one daemon per socket path, so
  clients may race to auto-spawn it without harm.
- If synthesis or playback fails, the line is spoken through a configurable
  fallback engine (espeak by default) — narration degrades, never disappears.

Run it with ``speakd`` (installed console script) or ``python -m voxcaster.daemon``.
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

log = logging.getLogger("voxcaster.daemon")

# Seconds a connected client gets to deliver its single request line.
READLINE_TIMEOUT_SEC = 5.0


def _sd_notify(msg: str) -> None:
    """Send one sd_notify(3) datagram to systemd, if running under it.

    Stdlib-only on purpose (no python-systemd dependency): NOTIFY_SOCKET is
    just an AF_UNIX datagram socket; a leading ``@`` means the abstract
    namespace (swap it for a NUL byte). Silently a no-op when not under
    systemd or on any socket error — notification must never be able to
    take the daemon down.
    """
    addr = os.environ.get("NOTIFY_SOCKET")
    if not addr:
        return
    if addr.startswith("@"):
        addr = "\0" + addr[1:]
    try:
        s = socket_module.socket(socket_module.AF_UNIX, socket_module.SOCK_DGRAM)
        try:
            s.connect(addr)
            s.sendall(msg.encode("utf-8"))
        finally:
            s.close()
    except OSError:
        pass


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

    def _speak_sync(self, text: str, voice: str | None = None) -> None:
        epoch = self._interrupt_epoch
        try:
            wav, sample_rate = self.engine.synthesize(text, voice=voice)
        except SynthesisError as e:
            log.warning("synthesis failed — using fallback: %s", e)
            self._speak_fallback(text)
            return
        if self._interrupt_epoch != epoch:
            log.info("dropped superseded line (interrupted mid-synthesis): %s", text[:60])
            return
        self._play(wav, sample_rate, text)

    def _render_sync(self, text: str, voice: str | None,
                     render_path: str) -> None:
        """Synthesize ``text`` and write a mono wav to ``render_path``.

        The file-output sibling of :meth:`_speak_sync`: identical synthesis
        path (same voice override, same interrupt-supersede check), but the
        waveform is written to disk with ``soundfile`` instead of being
        handed to a player. Used by the RENDER wire message so callers can
        request an audio file (voice notes, OpenClaw TTS integration) from
        the same warm daemon -- no per-call model load.

        Unlike playback there is no audible fallback: if synthesis or the
        write fails there is nothing useful to say out loud, so the failure
        is logged and the request is simply not acked (the client's
        blocking wait times out and reports the error).
        """
        epoch = self._interrupt_epoch
        try:
            wav, sample_rate = self.engine.synthesize(text, voice=voice)
        except SynthesisError as e:
            log.warning("render synthesis failed (no fallback): %s", e)
            return
        if self._interrupt_epoch != epoch:
            log.info("dropped superseded render (interrupted mid-synthesis): %s",
                     text[:60])
            return
        try:
            import soundfile as sf
            parent = os.path.dirname(os.path.abspath(render_path))
            if parent:
                os.makedirs(parent, exist_ok=True)
            sf.write(render_path, wav, sample_rate)
        except Exception as e:
            log.error("render write failed for %s: %s", render_path, e)
            return
        log.info("rendered (%dHz, %.1fs): %s -> %s",
                 sample_rate, len(wav) / sample_rate, text[:60], render_path)

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
            # cfg.max_playback_seconds is a HANG-PROTECTION FLOOR, not a cap on
            # legitimate content: a fixed ceiling here would silently kill
            # mid-playback the moment a long ("book"-length) line's audio
            # outran it. The real timeout is however long this clip actually
            # runs, plus headroom for player startup/seek overhead; the config
            # value only sets a minimum so short clips still get hang
            # protection even when duration is ~0.
            duration_seconds = len(wav) / sample_rate if sample_rate else 0.0
            timeout = max(self.cfg.max_playback_seconds, duration_seconds + 30.0)
            try:
                proc.wait(timeout=timeout)
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
                text, voice, render_path, writer = await asyncio.wait_for(
                    self._queue.get(), timeout=1.0
                )
            except asyncio.TimeoutError:
                continue
            try:
                if render_path is not None:
                    await loop.run_in_executor(
                        None, self._render_sync, text, voice, render_path
                    )
                else:
                    await loop.run_in_executor(None, self._speak_sync, text, voice)
            except Exception:
                # _speak_sync/_render_sync handle their own fallbacks; this
                # guards the worker task against anything unexpected so the
                # queue never stalls.
                log.exception("unexpected error while speaking %r", text[:60])
            try:
                writer.write(protocol.ACK)
                await writer.drain()
                writer.close()
            except Exception:
                pass  # fire-and-forget client already hung up
            self._queue.task_done()

    async def _interrupt(self, text: str, writer: asyncio.StreamWriter,
                         voice: str | None = None,
                         render_path: str | None = None) -> None:
        """Drain pending lines, cut off playback, speak/render ``text`` next."""
        self._interrupt_epoch += 1  # marks any in-flight synthesis as superseded
        drained = 0
        while True:
            try:
                (_old_text, _old_voice, _old_render_path,
                 old_writer) = self._queue.get_nowait()
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
        await self._queue.put((text, voice, render_path, writer))

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

    async def _watchdog_heartbeat(self) -> None:
        """Ping systemd's watchdog from INSIDE the event loop.

        ``Type=notify`` + ``WatchdogSec=`` in the unit make systemd expect a
        ``WATCHDOG=1`` datagram at least every WatchdogSec. Sending it from a
        loop task means the heartbeat stops the moment the event loop dies —
        an unhandled crash or a wedged loop — even though the process itself
        is still alive. That is exactly the failure mode where systemd shows
        ``Active: running`` while connect() gets ECONNREFUSED: without a
        watchdog systemd trusts the live PID forever; with one it aborts and
        restarts the unit as soon as the pings stop.

        No-op outside systemd (no WATCHDOG_USEC) and when the watchdog was
        armed for a different PID (WATCHDOG_PID mismatch, e.g. a manually
        forked child).
        """
        usec = os.environ.get("WATCHDOG_USEC")
        wd_pid = os.environ.get("WATCHDOG_PID")
        if not usec or (wd_pid and wd_pid != str(os.getpid())):
            return
        try:
            # Ping at half the budget — the interval systemd itself documents.
            interval = max(1.0, int(usec) / 2_000_000)
        except ValueError:
            return
        while not self._stop.is_set():
            _sd_notify("WATCHDOG=1")
            try:
                await asyncio.wait_for(self._stop.wait(), timeout=interval)
                break  # stop requested
            except asyncio.TimeoutError:
                pass

    async def _handle(self, reader: asyncio.StreamReader, writer: asyncio.StreamWriter) -> None:
        try:
            data = await asyncio.wait_for(reader.readline(), timeout=READLINE_TIMEOUT_SEC)
            text = data.decode(errors="ignore").strip()
            if not text:
                writer.close()
                return
            if text.startswith(protocol.PING_MARKER):
                reply = protocol.PING_READY if self.engine.is_ready() else protocol.PING_STARTING
                try:
                    writer.write(reply)
                    await writer.drain()
                except Exception:
                    pass
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

            # Strip the interrupt prefix (outermost), then the render prefix,
            # then the voice prefix -- the same nesting order encode_speak /
            # encode_render build, so one parser serves both message kinds.
            interrupt = text.startswith(protocol.INTERRUPT_MARKER)
            if interrupt:
                text = text[len(protocol.INTERRUPT_MARKER):]
            render_path: str | None = None
            if text.startswith(protocol.RENDER_MARKER):
                rest = text[len(protocol.RENDER_MARKER):]
                render_path, _, text = rest.partition("\x03")
                render_path = render_path or None
            voice: str | None = None
            if text.startswith(protocol.VOICE_MARKER):
                rest = text[len(protocol.VOICE_MARKER):]
                voice, _, text = rest.partition("\x03")
                voice = voice or None

            # Restore literal newlines the client escaped to survive the
            # line-oriented framing. Must run AFTER marker stripping (markers
            # are matched against the raw wire text) and BEFORE markdown
            # preprocessing (which is line-structure aware).
            text = protocol.unescape_newlines(text)

            if self.cfg.markdown_preprocess:
                from .markdown import preprocess_for_speech
                text = preprocess_for_speech(text).strip()
            if not text:
                writer.close()
                return

            if interrupt or self.cfg.always_interrupt:
                await self._interrupt(text, writer, voice=voice,
                                      render_path=render_path)
            else:
                await self._queue.put((text, voice, render_path, writer))
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
        # `limit` bounds asyncio's internal StreamReader buffer, not just an
        # advisory: the stdlib default (64 KiB) makes readline() raise
        # LimitOverrunError on anything longer, which the broad except in
        # _handle() swallows silently — a long paste (a "book") would just
        # vanish with no ack and no log line. cfg.max_message_bytes must cover
        # the longest single utterance a caller will ever send in one line.
        server = await asyncio.start_unix_server(
            self._handle, path=self.cfg.socket_path, limit=self.cfg.max_message_bytes
        )
        os.chmod(self.cfg.socket_path, self.cfg.socket_mode)
        log.info("voxcaster %s ready  voice=%s  socket=%s",
                 __version__, self.cfg.voice, self.cfg.socket_path)
        # Under Type=notify, systemd holds the unit in "activating" until
        # this arrives — sent only now, with the socket actually listening,
        # so "active (running)" is never claimed for a daemon that can't
        # yet accept a connection.
        _sd_notify("READY=1")

        def _shutdown(sig: int) -> None:
            log.info("shutdown (signal %d)", sig)
            self._stop.set()
            server.close()

        for sig in (signal.SIGTERM, signal.SIGINT):
            loop.add_signal_handler(sig, _shutdown, sig)

        worker_task = asyncio.create_task(self._worker(loop))
        offload_task = asyncio.create_task(self._idle_offload(loop))
        watchdog_task = asyncio.create_task(self._watchdog_heartbeat())

        # asyncio swallows task exceptions by default (surfacing them only
        # as a GC-time "exception was never retrieved" warning) — a worker
        # killed by anything its per-line guard doesn't catch would leave
        # the listener happily queueing lines that never get spoken, with
        # the loop (and therefore the watchdog heartbeat) still healthy.
        # Convert that silent death into a clean daemon exit instead;
        # systemd's Restart= policy brings back a working instance.
        def _core_task_died(task: asyncio.Task) -> None:
            if task.cancelled() or self._stop.is_set():
                return
            exc = task.exception()
            if exc is not None:
                log.error("core task died: %r — stopping daemon so the "
                          "supervisor can restart it", exc)
                self._stop.set()
                server.close()

        worker_task.add_done_callback(_core_task_died)

        async with server:
            await self._stop.wait()

        _sd_notify("STOPPING=1")
        for task in (worker_task, offload_task, watchdog_task):
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass

        if os.path.exists(self.cfg.socket_path):
            os.unlink(self.cfg.socket_path)
        log.info("voxcaster stopped")


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
        prog="voxcaster",
        description="Fire-and-forget TTS narration daemon (Kokoro over a Unix socket).",
    )
    parser.add_argument("--config", metavar="PATH", help="TOML config file")
    parser.add_argument("--socket", metavar="PATH", help="Unix socket path override")
    parser.add_argument("--device", choices=("auto", "cpu", "gpu"),
                        help="device policy override")
    parser.add_argument("--voice", help="Kokoro voice id override (e.g. af_heart, bf_emma)")
    parser.add_argument("--volume", type=int, metavar="N",
                        help="initial playback volume (0-130)")
    parser.add_argument("--no-markdown", dest="markdown_preprocess",
                        action="store_false", default=None,
                        help="disable markdown stripping / TTS_SUMMARY extraction")
    parser.add_argument("--always-interrupt", dest="always_interrupt",
                        action="store_true", default=None,
                        help="treat every request as an interrupt (speak latest only)")
    parser.add_argument("--print-config", action="store_true",
                        help="print the effective configuration and exit")
    parser.add_argument("--version", action="version", version=f"voxcaster {__version__}")
    args = parser.parse_args(argv)

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s %(levelname)-7s %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    # voxcaster is a persistent background service, not a terminal job: it
    # must survive its launching shell/tty hanging up (e.g. an ad-hoc
    # `daemon &` spawn whose parent session tears down). SIGHUP's default
    # disposition is immediate termination with ZERO Python-visible trace —
    # no exception, no log line. This was root-caused as the chronic
    # "spawns then dies, falls back silently" bug: fallback.log showed 12+
    # unexplained deaths over weeks with nothing in daemon.log to explain
    # them; reproducing `kill -HUP <daemon pid>` against the unpatched
    # daemon killed it instantly and silently, matching the pattern exactly.
    # Installed here (before any slow imports/torch init) so the whole
    # process lifetime is covered, not just the post-bind asyncio loop.
    def _log_hangup(signum, _frame) -> None:
        log.info("ignoring signal %d (SIGHUP) — daemon is a persistent "
                 "service and does not exit on terminal hangup", signum)

    signal.signal(signal.SIGHUP, _log_hangup)

    cfg = load_config(args.config)
    if args.socket:
        cfg.socket_path = args.socket
    if args.device:
        cfg.device = args.device
    if args.voice:
        cfg.voice = args.voice
    if args.volume is not None:
        cfg.volume = args.volume
    if args.markdown_preprocess is not None:
        cfg.markdown_preprocess = args.markdown_preprocess
    if args.always_interrupt is not None:
        cfg.always_interrupt = args.always_interrupt

    if args.print_config:
        print(cfg.describe())
        return

    lock_file = acquire_singleton_lock(cfg.lock_path)
    try:
        asyncio.run(VoiceDaemon(cfg).run())
    except (KeyboardInterrupt, SystemExit):
        raise
    except BaseException:
        # Belt-and-suspenders: guarantee any crash (not just silent-signal
        # deaths) leaves a full traceback in the daemon log — then take the
        # process down HARD. A plain `raise` here can leave a half-dead
        # zombie: non-daemon executor threads (torch/CUDA synthesis work
        # scheduled via run_in_executor) outlive the dead event loop, so the
        # interpreter never exits — the socket refuses connections while
        # `ps`/systemd still see a live process holding the flock (observed
        # live 2026-07-16: a spawned daemon crashed mid-synthesis on a pip
        # PEP 668 error and lingered exactly like this). os._exit() skips
        # the thread joins; the process dies for real and systemd's
        # Restart= policy brings a fresh one back.
        log.exception("daemon crashed — hard exit so no half-dead process lingers")
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            os.unlink(cfg.lock_path)
        except OSError:
            pass
        logging.shutdown()
        os._exit(70)  # EX_SOFTWARE; unreachable by the finally below
    finally:
        try:
            fcntl.flock(lock_file, fcntl.LOCK_UN)
            lock_file.close()
            os.unlink(cfg.lock_path)
        except OSError:
            pass


if __name__ == "__main__":
    main()
