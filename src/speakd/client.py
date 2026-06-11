"""speakd client: send text to the daemon, with auto-spawn and clean fallback.

This module is intentionally dependency-light (stdlib only) so that
``import speakd`` and the ``speak`` CLI stay instant even on machines where
the TTS stack is heavy. The daemon's dependencies are only imported inside
the daemon process.

Python API
----------
    from speakd import speak, set_volume, ensure_daemon

    speak("checkpoint saved")                      # fire-and-forget
    speak("eval finished", blocking=True)          # wait until spoken
    speak("loss is NaN — stopping", interrupt=True)  # jump the queue
    set_volume(85)                                 # live, 0-130

Every call is safe when the daemon is down: the client auto-spawns it once,
and if that fails it degrades to the configured fallback engine (espeak by
default) and logs the event — narration never silently disappears.

CLI
---
    speak "build finished"
    speak --interrupt "disk is full"
    speak --blocking "done"
    speak --volume 85
    long_running_job | speak        # reads stdin when no text args are given
"""
from __future__ import annotations

import argparse
import datetime
import os
import shlex
import socket
import subprocess
import sys
import time

from . import protocol
from .config import Config, load_config

# Process-wide default config, loaded lazily on first use.
_default_config: Config | None = None


def _get_config(config: Config | None = None) -> Config:
    global _default_config
    if config is not None:
        return config
    if _default_config is None:
        _default_config = load_config()
    return _default_config


# ── low-level helpers ───────────────────────────────────────────────────────

def _socket_alive(cfg: Config) -> bool:
    """True if the daemon accepts connections (~1 ms; safe in hot loops)."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(cfg.connect_timeout)
        s.connect(cfg.socket_path)
        s.close()
        return True
    except OSError:
        return False


def _send(payload: bytes, cfg: Config, wait_ack: bool = False) -> bool:
    """Deliver one wire-protocol line. Returns False on any socket error."""
    try:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(cfg.connect_timeout)
        s.connect(cfg.socket_path)
        s.sendall(payload)
        if wait_ack:
            # Speech can take a while — switch to the generous ack timeout.
            s.settimeout(cfg.ack_timeout)
            s.recv(len(protocol.ACK) + 62)
        s.close()
        return True
    except OSError:
        return False


def _log_fallback(cfg: Config, reason: str) -> None:
    """Record a fallback event (file + stderr) so degraded audio is diagnosable."""
    try:
        os.makedirs(os.path.dirname(cfg.fallback_log), exist_ok=True)
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        with open(cfg.fallback_log, "a") as f:
            f.write(f"{timestamp}  FALLBACK  reason={reason}\n")
    except OSError:
        pass
    print(f"[speakd] WARNING: fallback engine used — {reason}  (see {cfg.fallback_log})",
          file=sys.stderr, flush=True)


def _fallback_speak(text: str, interrupt: bool, cfg: Config) -> None:
    """Last resort: speak through the configured fallback engine."""
    if not cfg.fallback:
        return  # fallback disabled by config
    argv = [a.format(text=text) for a in cfg.fallback]
    if not any("{text}" in a for a in cfg.fallback):
        argv.append(text)
    try:
        if interrupt:
            # Best-effort: cut off any in-flight fallback speech first.
            subprocess.run(["pkill", "-x", os.path.basename(argv[0])],
                           stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    except (OSError, FileNotFoundError):
        pass  # fallback engine not installed either — nothing left to try


# ── public API ──────────────────────────────────────────────────────────────

def ensure_daemon(config: Config | None = None) -> bool:
    """Idempotent: make sure a daemon is listening on the configured socket.

    Fast path returns immediately when the socket answers. Otherwise a
    detached daemon is spawned (``python -m speakd.daemon`` with this
    interpreter, overridable via ``$SPEAKD_DAEMON_CMD``) and we wait up to
    ``client.spawn_wait`` seconds for it to come up. The daemon's flock
    singleton makes concurrent spawn attempts harmless.
    """
    cfg = _get_config(config)
    if _socket_alive(cfg):
        return True

    custom = os.environ.get("SPEAKD_DAEMON_CMD", "")
    cmd = shlex.split(custom) if custom else [sys.executable, "-m", "speakd.daemon"]
    env = dict(os.environ, SPEAKD_SOCKET=cfg.socket_path)
    try:
        os.makedirs(os.path.dirname(cfg.log_file), exist_ok=True)
        with open(cfg.log_file, "a") as log_fh:
            subprocess.Popen(
                cmd,
                stdout=log_fh,
                stderr=log_fh,
                env=env,
                close_fds=True,
                start_new_session=True,
            )
    except OSError:
        return False

    deadline = time.monotonic() + cfg.spawn_wait
    while time.monotonic() < deadline:
        if _socket_alive(cfg):
            return True
        time.sleep(0.2)
    return False


def set_volume(level: int, config: Config | None = None) -> bool:
    """Set the daemon's live playback volume (0-130; 100 = nominal).

    Applies from the next spoken line — no restart needed. Returns True if
    the daemon received it.
    """
    cfg = _get_config(config)
    if _send(protocol.encode_volume(level), cfg):
        return True
    print(f"[speakd] daemon not running — start it, or export SPEAKD_VOLUME={level}",
          file=sys.stderr)
    return False


def speak(
    text: str,
    blocking: bool = False,
    interrupt: bool = False,
    config: Config | None = None,
) -> bool:
    """Send text to the voice daemon.

    Args:
        text:      The text to speak. Empty/whitespace-only text is a no-op.
        blocking:  Wait until the daemon has finished speaking the line.
        interrupt: Drain the pending queue and cut off in-flight playback
                   before speaking this line.
        config:    Optional explicit :class:`speakd.config.Config`.

    Returns:
        True if the line was delivered to the daemon; False if the fallback
        engine had to be used (or nothing could speak at all).
    """
    text = text.strip()
    if not text:
        return True

    cfg = _get_config(config)
    wire = protocol.encode_speak(text, interrupt=interrupt)

    # Fast path — daemon already up.
    if _send(wire, cfg, wait_ack=blocking):
        return True

    # Recovery — bring the daemon up, retry once.
    if ensure_daemon(cfg) and _send(wire, cfg, wait_ack=blocking):
        return True

    # Last resort — fallback engine.
    _log_fallback(cfg, "daemon down after spawn attempt")
    _fallback_speak(text, interrupt, cfg)
    return False


# ── CLI ─────────────────────────────────────────────────────────────────────

def main(argv: list[str] | None = None) -> int:
    from . import __version__

    parser = argparse.ArgumentParser(
        prog="speak",
        description="Send text to the speakd narration daemon.",
        epilog="With no TEXT arguments, text is read from stdin (pipe-friendly).",
    )
    parser.add_argument("text", nargs="*", help="text to speak")
    parser.add_argument("-i", "--interrupt", action="store_true",
                        help="cut off current speech and drain the queue first")
    parser.add_argument("-b", "--blocking", action="store_true",
                        help="wait until the line has been spoken")
    parser.add_argument("--volume", type=int, metavar="N",
                        help="set live playback volume (0-130) before speaking")
    parser.add_argument("--socket", metavar="PATH", help="Unix socket path override")
    parser.add_argument("--config", metavar="PATH", help="TOML config file")
    parser.add_argument("--version", action="version", version=f"speakd {__version__}")
    args = parser.parse_args(argv)

    cfg = load_config(args.config)
    if args.socket:
        cfg.socket_path = args.socket

    if args.volume is not None:
        set_volume(args.volume, config=cfg)

    text = " ".join(args.text)
    if not text and not sys.stdin.isatty():
        text = sys.stdin.read().strip()
    if not text:
        if args.volume is not None:
            return 0  # volume-only invocation
        parser.print_usage(sys.stderr)
        return 2

    delivered = speak(text, blocking=args.blocking, interrupt=args.interrupt, config=cfg)
    return 0 if delivered else 1


if __name__ == "__main__":
    sys.exit(main())
