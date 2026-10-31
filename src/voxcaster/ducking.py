"""Dim every other playback stream while voxcaster speaks, then put it back.

Talks to the audio server through ``pactl`` only (PulseAudio protocol, so it
works on PipeWire too). Voxcaster's own player is told apart by process id:
a sink-input does not carry the pid itself, so it is joined through its
client's ``application.process.id``.

Originals are written to a state file before any volume changes, so a daemon
that dies while ducked can put the streams back on its next start
(:meth:`Ducker.recover`).
"""
from __future__ import annotations

import functools
import json
import logging
import os
import subprocess
import threading
import time
from dataclasses import asdict, dataclass
from typing import Callable, Protocol

log = logging.getLogger("voxcaster.ducking")

FADE_STEP_SEC = 0.03
POLL_INTERVAL_SEC = 0.5
OWN_STREAM_WAIT_SEC = 1.0
OWN_STREAM_POLL_SEC = 0.05
PACTL_TIMEOUT_SEC = 2.0

Volume = tuple[int, ...]


class PactlError(RuntimeError):
    pass


@dataclass(frozen=True)
class Stream:
    index: int
    volume: Volume  # raw per-channel values, in channel-map order
    muted: bool
    pid: int | None
    app: str
    restore_id: str  # the server's stream-restore key; streams sharing it inherit each other's volume


class Pactl(Protocol):
    def sink_inputs(self) -> list[Stream]: ...

    def set_volume(self, index: int, volume: Volume) -> bool:
        """False when the stream no longer exists."""


@dataclass
class Ducked:
    index: int
    app: str
    restore_id: str
    original: Volume
    ducked: Volume

    @classmethod
    def from_json(cls, raw: dict) -> "Ducked":
        return cls(int(raw["index"]), str(raw["app"]), str(raw["restore_id"]),
                   tuple(raw["original"]), tuple(raw["ducked"]))


def parse_streams(sink_inputs_json: str, clients_json: str) -> list[Stream]:
    pids = {}
    for client in json.loads(clients_json):
        raw_pid = client.get("properties", {}).get("application.process.id")
        if raw_pid is not None:
            pids[str(client["index"])] = int(raw_pid)
    streams = []
    for item in json.loads(sink_inputs_json):
        channels = [c for c in item.get("channel_map", "").split(",") if c]
        volume = tuple(int(item["volume"][c]["value"]) for c in channels
                       if c in item.get("volume", {}))
        props = item.get("properties", {})
        streams.append(Stream(
            index=int(item["index"]),
            volume=volume,
            muted=bool(item.get("mute", False)),
            pid=pids.get(str(item.get("client"))),
            app=props.get("application.name", ""),
            restore_id=props.get("module-stream-restore.id", ""),
        ))
    return streams


class PactlCli:
    def _run(self, *args: str) -> str:
        try:
            done = subprocess.run(("pactl", *args), capture_output=True, text=True,
                                  timeout=PACTL_TIMEOUT_SEC)
        except (OSError, subprocess.TimeoutExpired) as e:
            raise PactlError(f"pactl {' '.join(args)}: {e}") from e
        if done.returncode != 0:
            raise PactlError(f"pactl {' '.join(args)}: rc={done.returncode} "
                             f"{done.stderr.strip()}")
        return done.stdout

    def sink_inputs(self) -> list[Stream]:
        try:
            return parse_streams(
                self._run("--format=json", "list", "sink-inputs"),
                self._run("--format=json", "list", "clients"),
            )
        except (ValueError, KeyError, TypeError) as e:
            raise PactlError(f"unparseable pactl output: {e}") from e

    def set_volume(self, index: int, volume: Volume) -> bool:
        try:
            self._run("set-sink-input-volume", str(index), *map(str, volume))
        except PactlError:
            return False
        return True


def _never_raises(method):
    @functools.wraps(method)
    def wrapper(self, *args, **kwargs):
        try:
            return method(self, *args, **kwargs)
        except Exception as e:
            log.warning("ducking %s failed: %s", method.__name__, e)
    return wrapper


def _is_ours(current: Volume, entry: Ducked) -> bool:
    """True while ``current`` still lies between the ducked and original levels,
    i.e. nobody but us has moved this stream's volume."""
    return len(current) == len(entry.original) and all(
        min(d, o) <= c <= max(d, o)
        for c, d, o in zip(current, entry.ducked, entry.original)
    )


class Ducker:
    def __init__(self, pactl: Pactl, state_path: str, *, factor: float,
                 fade_down_sec: float, fade_up_sec: float, enabled: bool = True,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 poll_interval: float | None = POLL_INTERVAL_SEC) -> None:
        self._pactl = pactl
        self._state_path = state_path
        self._factor = factor
        self._fade_down = fade_down_sec
        self._fade_up = fade_up_sec
        self._enabled = enabled
        self._sleep = sleep
        self._clock = clock
        self._poll_interval = poll_interval

        self._lock = threading.RLock()
        self._holders = 0
        self._own_pids: set[int] = set()
        self._lifted: set[int] = set()
        self._ducked: dict[int, Ducked] = {}
        self._active = False
        self._poller_stop: threading.Event | None = None

    # ── lifecycle ─────────────────────────────────────────────────────────

    @_never_raises
    def acquire(self, own_pid: int) -> None:
        """Duck everything except ``own_pid``'s streams. Calls nest."""
        if not self._enabled:
            return
        with self._lock:
            self._holders += 1
            self._own_pids.add(own_pid)
            self._active = True
            self._start_poller()
            self._duck_new_streams()
            self._await_own_stream(own_pid)

    @_never_raises
    def release(self, keep_ducked: bool = False) -> None:
        """Drop one holder. The last one restores, unless ``keep_ducked`` says
        another line is about to play (then :meth:`settle` restores later)."""
        if not self._enabled:
            return
        with self._lock:
            self._holders = max(0, self._holders - 1)
            if self._holders == 0 and not keep_ducked:
                self._restore_all()

    @_never_raises
    def settle(self) -> None:
        with self._lock:
            if self._active and self._holders == 0:
                self._restore_all()

    @_never_raises
    def shutdown(self) -> None:
        with self._lock:
            self._holders = 0
            if self._active:
                self._restore_all()

    @_never_raises
    def rescan(self) -> None:
        """Duck streams that appeared since the last look."""
        with self._lock:
            if self._active:
                self._duck_new_streams()

    @_never_raises
    def recover(self) -> None:
        """Undo a duck left behind by a previous process."""
        entries = self._load_state()
        if not entries:
            return
        streams = self._pactl.sink_inputs()
        claimed: set[int] = set()
        for entry in entries:
            match = self._recovery_match(entry, streams, claimed)
            if match is None:
                continue
            claimed.add(match.index)
            self._pactl.set_volume(match.index, entry.original)
            log.info("recovered stale duck on stream %d (%s)", match.index, match.app)
        self._clear_state()

    # ── ducking ───────────────────────────────────────────────────────────

    def _duck_new_streams(self) -> None:
        streams = self._pactl.sink_inputs()
        live = {s.index: s for s in streams}
        for index in [i for i, e in self._ducked.items()
                      if i not in live or live[i].app != e.app]:
            del self._ducked[index]

        moves: dict[int, tuple[Volume, Volume]] = {}
        for s in streams:
            if (s.index in self._ducked or s.pid in self._own_pids
                    or s.muted or not s.volume):
                continue
            entry = self._entry_for(s)
            self._ducked[s.index] = entry
            if s.volume != entry.ducked:
                moves[s.index] = (s.volume, entry.ducked)
        if self._ducked:
            self._save_state()
        self._fade(moves, self._fade_down)
        self._lift_inherited_own_streams(streams)

    def _await_own_stream(self, own_pid: int) -> None:
        deadline = self._clock() + OWN_STREAM_WAIT_SEC
        while self._clock() < deadline:
            streams = self._pactl.sink_inputs()
            if any(s.pid == own_pid for s in streams):
                self._lift_inherited_own_streams(streams)
                return
            self._sleep(OWN_STREAM_POLL_SEC)

    def _lift_inherited_own_streams(self, streams: list[Stream]) -> None:
        """The server gives a new stream the volume last written under its
        stream-restore key, so the player can be born at a sibling's ducked
        level. Put it back where the sibling started."""
        for s in streams:
            if s.pid not in self._own_pids or s.index in self._lifted:
                continue
            for sibling in self._ducked.values():
                if (s.restore_id and s.restore_id == sibling.restore_id
                        and s.volume != sibling.original
                        and _is_ours(s.volume, sibling)):
                    self._pactl.set_volume(s.index, sibling.original)
                    self._lifted.add(s.index)
                    break

    def _entry_for(self, s: Stream) -> Ducked:
        """A stream created mid-speech may start at a volume the server copied
        from an already-ducked sibling; its true original is the sibling's."""
        for sibling in self._ducked.values():
            if (s.restore_id and s.restore_id == sibling.restore_id
                    and s.volume == sibling.ducked):
                return Ducked(s.index, s.app, s.restore_id,
                              sibling.original, sibling.ducked)
        target = tuple(round(v * self._factor) for v in s.volume)
        return Ducked(s.index, s.app, s.restore_id, s.volume, target)

    def _restore_all(self) -> None:
        self._active = False
        if self._poller_stop is not None:
            self._poller_stop.set()
            self._poller_stop = None
        self._own_pids.clear()
        self._lifted.clear()
        entries, self._ducked = self._ducked, {}
        try:
            try:
                live = {s.index: s for s in self._pactl.sink_inputs()}
            except PactlError as e:
                log.warning("cannot list streams to restore, restoring blind: %s", e)
                live = {i: Stream(i, entry.ducked, False, None, entry.app,
                                  entry.restore_id)
                        for i, entry in entries.items()}
            moves: dict[int, tuple[Volume, Volume]] = {}
            for index, entry in entries.items():
                s = live.get(index)
                if s is None or s.app != entry.app:
                    continue
                if not _is_ours(s.volume, entry):
                    log.info("stream %d volume changed during speech; leaving it", index)
                    continue
                moves[index] = (s.volume, entry.original)
            self._fade(moves, self._fade_up)
        finally:
            self._clear_state()

    def _fade(self, moves: dict[int, tuple[Volume, Volume]], seconds: float) -> None:
        if not moves:
            return
        began = self._clock()
        while True:
            elapsed = self._clock() - began
            if seconds > 0:
                self._sleep(min(FADE_STEP_SEC, max(0.0, seconds - elapsed)))
                elapsed = self._clock() - began
            done = seconds <= 0 or elapsed >= seconds - 1e-9
            for index, (start, end) in moves.items():
                if done:
                    volume = end
                else:
                    volume = tuple(round(a + (b - a) * elapsed / seconds)
                                   for a, b in zip(start, end))
                try:
                    self._pactl.set_volume(index, volume)
                except PactlError as e:
                    log.warning("set volume on stream %d failed: %s", index, e)
            if done:
                return

    # ── new-stream poller ─────────────────────────────────────────────────

    def _start_poller(self) -> None:
        if self._poll_interval is None or self._poller_stop is not None:
            return
        stop = threading.Event()
        self._poller_stop = stop
        threading.Thread(target=self._poll, args=(stop,), daemon=True,
                         name="voxcaster-duck-poll").start()

    def _poll(self, stop: threading.Event) -> None:
        while not stop.wait(self._poll_interval):
            with self._lock:
                if stop.is_set():
                    return
                self.rescan()

    # ── state file ────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({"entries": [asdict(e) for e in self._ducked.values()]}, fh)
        os.replace(tmp, self._state_path)

    def _load_state(self) -> list[Ducked]:
        try:
            with open(self._state_path) as fh:
                return [Ducked.from_json(e) for e in json.load(fh)["entries"]]
        except FileNotFoundError:
            return []
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("ignoring unreadable duck state %s: %s", self._state_path, e)
            self._clear_state()
            return []

    def _clear_state(self) -> None:
        try:
            os.unlink(self._state_path)
        except FileNotFoundError:
            pass

    @staticmethod
    def _recovery_match(entry: Ducked, streams: list[Stream],
                        claimed: set[int]) -> Stream | None:
        candidates = [
            s for s in streams
            if s.index not in claimed and _is_ours(s.volume, entry)
            and ((s.index == entry.index and s.app == entry.app)
                 or (entry.restore_id and s.restore_id == entry.restore_id))
        ]
        candidates.sort(key=lambda s: s.index != entry.index)
        return candidates[0] if candidates else None
