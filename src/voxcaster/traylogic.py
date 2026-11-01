"""Tray control logic: config edits, volume maths and the actions behind each
menu entry. Stdlib only and free of any GUI import, so it runs under any
Python and is testable without a display."""
from __future__ import annotations

import json
import math
import os
import re
import socket
import subprocess
from typing import Callable

from . import protocol
from .config import _runtime_path, tomllib

DEFAULT_CONFIG = os.path.join(
    os.environ.get("XDG_CONFIG_HOME", os.path.expanduser("~/.config")),
    "voxcaster", "config.toml")
DEFAULT_AGENTS = os.path.expanduser("~/.voxcaster/config.json")
SERVICE = "voxcaster.service"
TEST_LINE = "Voxcaster test voice. If you can hear this, the volume is fine."

# (label, dB relative to mpv volume 100). The player's volume scale is cubic.
PRESETS = (("Quiet", -6.0), ("Medium", 0.0), ("Normal", 3.9), ("Loud", 6.8))
DEPTHS = (("Gentle  -6 dB", 0.5), ("Normal  -10.5 dB", 0.3), ("Deep  -12 dB", 0.25))
HOLDS_MS = (0, 500, 1500, 3000)


def speaker_label(voice: str | None, agent_voices: list[tuple[str, str]]) -> str:
    """The agent that owns ``voice``; the voice id itself when several agents
    share it, ``default`` when the line carries no voice."""
    if voice is None:
        return "default"
    owners = [name for name, v in agent_voices if v == voice]
    return owners[0] if len(owners) == 1 else voice


def line_label(line: dict, agent_voices: list[tuple[str, str]]) -> str:
    ellipsis = "..." if len(line["text"]) >= protocol.PREVIEW_CHARS else ""
    text = " ".join(line["text"].split())
    return f"{speaker_label(line['voice'], agent_voices)}: {text}{ellipsis}"


def volume_to_db(volume: int) -> float:
    return 60.0 * math.log10(volume / 100) if volume > 0 else float("-inf")


def db_to_volume(db: float) -> int:
    return protocol.clamp_volume(round(100 * 10 ** (db / 60)))


def _render(value) -> str:
    if isinstance(value, bool):
        return "true" if value else "false"
    if isinstance(value, (int, float)):
        return repr(value)
    return json.dumps(value)


_SECTION = re.compile(r"^\s*\[([^\[\]]+)\]\s*(#.*)?$")
_ASSIGNMENT = re.compile(r"^\s*[\w\"'.-]+\s*=")


def set_toml_value(text: str, section: str, key: str, value) -> str:
    """Return ``text`` with ``[section] key`` set to ``value``.

    An existing assignment is edited in place, keeping its inline comment; a
    missing key is added after the section's last assignment; a missing section
    is appended. Nothing else changes, so comments and layout survive. The
    result is parsed before it is returned."""
    rendered = _render(value)
    lines = text.splitlines(keepends=True)
    current = None
    header = last_assignment = None
    edited = False
    key_line = re.compile(r"^(\s*" + re.escape(key) + r"\s*=\s*)(.*?)(\s+#.*)?$")
    for i, line in enumerate(lines):
        body = line.rstrip("\r\n")
        eol = line[len(body):]
        m = _SECTION.match(body)
        if m:
            current = m.group(1).strip()
            if current == section:
                header = i
            continue
        if current != section:
            continue
        if _ASSIGNMENT.match(body):
            last_assignment = i
            k = key_line.match(body)
            if k and not edited:
                lines[i] = f"{k.group(1)}{rendered}{k.group(3) or ''}{eol}"
                edited = True
    if not edited:
        new = f"{key} = {rendered}\n"
        if header is None:
            if lines and not lines[-1].endswith("\n"):
                lines[-1] += "\n"
            lines += ([] if not lines else ["\n"]) + [f"[{section}]\n", new]
        else:
            at = (last_assignment if last_assignment is not None else header) + 1
            if not lines[at - 1].endswith("\n"):
                lines[at - 1] += "\n"
            lines.insert(at, new)
    result = "".join(lines)
    try:
        parsed = tomllib.loads(result)
    except tomllib.TOMLDecodeError as e:
        raise ValueError(f"edit would break the config: {e}") from e
    if parsed.get(section, {}).get(key) != value:
        raise ValueError(f"edit did not take: [{section}] {key}")
    return result


def write_config_value(path: str, section: str, key: str, value) -> None:
    try:
        with open(path) as fh:
            text = fh.read()
    except FileNotFoundError:
        text = ""
    updated = set_toml_value(text, section, key, value)
    tmp = path + ".tmp"
    with open(tmp, "w") as fh:
        fh.write(updated)
    try:
        os.chmod(tmp, os.stat(path).st_mode & 0o7777)
    except FileNotFoundError:
        pass
    os.replace(tmp, path)


class SocketWire:
    def __init__(self, socket_path: str, timeout: float = 1.0) -> None:
        self._path = socket_path
        self._timeout = timeout

    def _connect(self) -> socket.socket:
        s = socket.socket(socket.AF_UNIX, socket.SOCK_STREAM)
        s.settimeout(self._timeout)
        s.connect(self._path)
        return s

    def send(self, payload: bytes) -> None:
        with self._connect() as s:
            s.sendall(payload)

    def request(self, payload: bytes) -> bytes:
        with self._connect() as s:
            s.sendall(payload)
            reply = b""
            while not reply.endswith(b"\n"):
                chunk = s.recv(4096)
                if not chunk:
                    break
                reply += chunk
            return reply


def _spawn(argv: list[str]) -> None:
    subprocess.Popen(argv, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)


class TrayController:
    def __init__(self, wire, *, config_path: str = DEFAULT_CONFIG,
                 state_path: str | None = None, agents_path: str = DEFAULT_AGENTS,
                 run: Callable[[list[str]], None] = _spawn) -> None:
        self._wire = wire
        self.config_path = config_path
        self._state_path = state_path or _runtime_path("voxcaster-tray", ".json")
        self._agents_path = agents_path
        self._run = run

    def status(self) -> dict | None:
        try:
            return json.loads(self._wire.request(protocol.encode_status()))
        except (OSError, ValueError):
            return None

    def set_volume(self, volume: int, persist: bool = True) -> int:
        volume = protocol.clamp_volume(volume)
        self._wire.send(protocol.encode_volume(volume))
        if persist:
            write_config_value(self.config_path, "audio", "volume", volume)
        return volume

    def configured_volume(self) -> int:
        try:
            with open(self.config_path, "rb") as fh:
                return int(tomllib.load(fh).get("audio", {}).get("volume", 100))
        except (OSError, ValueError):
            return 100

    def toggle_mute(self) -> int | None:
        """Mute by sending volume 0 (the daemon then skips speech and ducking);
        unmute by sending back the volume remembered at mute time. Returns the
        volume now in force, or None if the daemon did not answer."""
        s = self.status()
        if s is None:
            return None
        if s["volume"] > 0:
            self._remember(s["volume"])
            self._wire.send(protocol.encode_volume(0))
            return 0
        restored = self._recall() or self.configured_volume()
        return self.set_volume(restored, persist=False)

    def _remember(self, volume: int) -> None:
        with open(self._state_path, "w") as fh:
            json.dump({"unmuted_volume": volume}, fh)

    def _recall(self) -> int | None:
        try:
            with open(self._state_path) as fh:
                return int(json.load(fh)["unmuted_volume"])
        except (OSError, ValueError, KeyError):
            return None

    def set_ducking(self, *, enabled: bool | None = None, factor: float | None = None,
                    hold_ms: int | None = None) -> str:
        for key, value in (("enabled", enabled), ("factor", factor), ("hold_ms", hold_ms)):
            if value is not None:
                write_config_value(self.config_path, "ducking", key, value)
        return self._wire.request(protocol.encode_reload()).decode().strip()

    def test_voice(self) -> None:
        self._wire.send(protocol.encode_speak(TEST_LINE))

    def stop_speech(self) -> None:
        self._wire.request(protocol.encode_stop())

    def skip(self) -> None:
        self._wire.request(protocol.encode_skip())

    def toggle_pause(self) -> bool | None:
        """Returns whether the daemon is now paused, or None if it did not answer."""
        s = self.status()
        if s is None:
            return None
        paused = not s.get("paused", False)
        self._wire.request(protocol.encode_pause() if paused else protocol.encode_resume())
        return paused

    def clear_queue(self) -> None:
        self._wire.request(protocol.encode_clear())

    def drop_line(self, line_id: int) -> bool:
        return self._wire.request(protocol.encode_drop(line_id)) == protocol.ACK

    def restart_daemon(self) -> None:
        self._run(["systemctl", "--user", "restart", SERVICE])

    def open_config(self) -> None:
        self._run(["xdg-open", self.config_path])

    def agent_voices(self) -> list[tuple[str, str]]:
        """The per-agent voice map, read-only; it is policy-controlled."""
        try:
            with open(self._agents_path) as fh:
                return sorted(json.load(fh).get("agents", {}).items())
        except (OSError, ValueError, AttributeError):
            return []
