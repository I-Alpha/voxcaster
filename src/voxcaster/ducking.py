"""Dim every other playback stream while voxcaster speaks, then put it back.

Talks to the audio server through ``pactl`` only (PulseAudio protocol, so it
works on PipeWire too). Voxcaster's own player is told apart by process id:
a sink-input does not carry the pid itself, so it is joined through its
client's ``application.process.id``.

Originals are written to a state file before any volume changes, so a daemon
that dies while ducked can put the streams back on its next start
(:meth:`Ducker.recover`).

``factor`` is an amplitude multiplier. The server's raw volume scale is cubic
(amplitude = (raw / 65536) ** 3), so the raw value is scaled by its cube root.
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
NOMINAL_VOLUME = 65536  # 100% on the server's raw scale
PACTL_TIMEOUT_SEC = 2.0
RESIDUE_TTL_SEC = 1800.0
IDLE_POLL_FACTOR = 4

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

    def set_mute(self, index: int, muted: bool) -> bool: ...


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

    def set_mute(self, index: int, muted: bool) -> bool:
        try:
            self._run("set-sink-input-mute", str(index), "1" if muted else "0")
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


def _key(restore_id: str, app: str) -> str:
    """Streams sharing a key inherit each other's last volume from the server."""
    return restore_id or f"app:{app}"


def _is_ours(current: Volume, entry: Ducked) -> bool:
    """True while ``current`` still lies between the ducked and original levels,
    i.e. nobody but us has moved this stream's volume."""
    return len(current) == len(entry.original) and all(
        min(d, o) <= c <= max(d, o)
        for c, d, o in zip(current, entry.ducked, entry.original)
    )


class Ducker:
    def __init__(self, pactl: Pactl, state_path: str, *, factor: float,
                 fade_down_sec: float, fade_up_sec: float, hold_sec: float = 0.0,
                 enabled: bool = True,
                 sleep: Callable[[float], None] = time.sleep,
                 clock: Callable[[], float] = time.monotonic,
                 poll_interval: float | None = POLL_INTERVAL_SEC) -> None:
        self._pactl = pactl
        self._state_path = state_path
        self._volume_ratio = factor ** (1 / 3)
        self._fade_down = fade_down_sec
        self._fade_up = fade_up_sec
        self._hold = hold_sec
        self._enabled = enabled
        self._sleep = sleep
        self._clock = clock
        self._poll_interval = poll_interval

        self._lock = threading.RLock()
        self._holders = 0
        self._own_pids: set[int] = set()
        self._ducked: dict[int, Ducked] = {}
        self._active = False
        self._release_at: float | None = None
        # Ducked levels left under a restore key by a stream that died ducked:
        # the server hands them to the next stream with that key.
        self._residue: dict[str, tuple[Ducked, float]] = {}
        self._poller_running = False
        self._wake = threading.Condition(self._lock)

    # ── lifecycle ─────────────────────────────────────────────────────────

    def acquire(self, own_pid: int) -> None:
        """Pin the player's own stream to nominal volume and, when enabled,
        duck everything else. Calls nest."""
        self._duck_others(own_pid)
        self._pin_player(own_pid)

    @_never_raises
    def _duck_others(self, own_pid: int) -> None:
        if not self._enabled:
            return
        with self._lock:
            self._holders += 1
            self._release_at = None
            self._own_pids.add(own_pid)
            self._active = True
            self._start_poller()
            self._duck_new_streams()

    @_never_raises
    def release(self) -> None:
        """Drop one holder. When the last goes, the duck is kept for the hold
        time so a line that follows closely does not bounce the other audio;
        :meth:`tick` restores once the hold has passed."""
        if not self._enabled:
            return
        with self._lock:
            self._holders = max(0, self._holders - 1)
            if self._holders > 0:
                return
            if self._hold <= 0:
                self._restore_all()
                return
            self._release_at = self._clock() + self._hold
            self._wake.notify_all()

    @_never_raises
    def shutdown(self) -> None:
        with self._lock:
            self._holders = 0
            if self._active:
                self._restore_all()

    @_never_raises
    def reconfigure(self, *, factor: float, fade_down_sec: float, fade_up_sec: float,
                    hold_sec: float, enabled: bool) -> None:
        """Apply new settings. Turning ducking off puts a live duck back at once;
        a new depth applies from the next duck."""
        with self._lock:
            if self._active and not enabled:
                self._holders = 0
                self._restore_all()
            self._volume_ratio = factor ** (1 / 3)
            self._fade_down = fade_down_sec
            self._fade_up = fade_up_sec
            self._hold = hold_sec
            self._enabled = enabled

    @_never_raises
    def tick(self) -> None:
        """Restore once the hold after the last line has passed; otherwise duck
        streams that appeared since the last look."""
        with self._lock:
            if not self._active:
                self._repair_residue()
                return
            if self._release_at is not None and self._clock() >= self._release_at:
                self._restore_all()
            else:
                self._duck_new_streams()

    @_never_raises
    def recover(self) -> None:
        """Undo a duck left behind by a previous process."""
        entries, residue = self._load_state()
        with self._lock:
            for entry in residue:
                self._remember_residue(entry)
            if entries:
                streams = self._pactl.sink_inputs()
                claimed: set[int] = set()
                for entry in entries:
                    match = self._recovery_match(entry, streams, claimed)
                    if match is None:
                        self._remember_residue(entry)
                        continue
                    claimed.add(match.index)
                    self._pactl.set_volume(match.index, entry.original)
                    log.info("recovered stale duck on stream %d (%s)",
                             match.index, match.app)
            self._save_state()
            if self._residue:
                self._start_poller()

    # ── ducking ───────────────────────────────────────────────────────────

    def _duck_new_streams(self) -> None:
        streams = self._pactl.sink_inputs()
        live = {s.index: s for s in streams}
        for index in [i for i, e in self._ducked.items()
                      if i not in live or live[i].app != e.app]:
            self._remember_residue(self._ducked.pop(index))

        moves: dict[int, tuple[Volume, Volume]] = {}
        for s in streams:
            if (s.index in self._ducked or s.pid in self._own_pids
                    or s.muted or not s.volume):
                continue
            entry = self._entry_for(s)
            self._ducked[s.index] = entry
            if s.volume != entry.ducked:
                moves[s.index] = (s.volume, entry.ducked)
        self._save_state()
        self._fade(moves, self._fade_down)

    @_never_raises
    def _pin_player(self, own_pid: int) -> None:
        """The server gives a new stream the volume and mute last stored under
        its restore key (the media role for mpv), which other streams'
        volume changes, ducking included, overwrite. The player must not
        depend on that residue, so it is set explicitly."""
        deadline = self._clock() + OWN_STREAM_WAIT_SEC
        while True:
            own = [s for s in self._pactl.sink_inputs() if s.pid == own_pid]
            if own or self._clock() >= deadline:
                break
            self._sleep(OWN_STREAM_POLL_SEC)
        for s in own:
            if s.muted:
                self._pactl.set_mute(s.index, False)
            if s.volume and s.volume != (NOMINAL_VOLUME,) * len(s.volume):
                self._pactl.set_volume(s.index, (NOMINAL_VOLUME,) * len(s.volume))

    def _entry_for(self, s: Stream) -> Ducked:
        """A stream created mid-speech may start at a volume the server copied
        from an already-ducked sibling, or from one that died ducked; the
        original is then the one recorded for that sibling, not what it reads."""
        key = _key(s.restore_id, s.app)
        for sibling in self._ducked.values():
            if _key(sibling.restore_id, sibling.app) == key and s.volume == sibling.ducked:
                return Ducked(s.index, s.app, s.restore_id,
                              sibling.original, sibling.ducked)
        dead = self._residue.get(key)
        if dead is not None and s.volume == dead[0].ducked:
            del self._residue[key]
            return Ducked(s.index, s.app, s.restore_id,
                          dead[0].original, dead[0].ducked)
        target = tuple(round(v * self._volume_ratio) for v in s.volume)
        return Ducked(s.index, s.app, s.restore_id, s.volume, target)

    def _remember_residue(self, entry: Ducked) -> None:
        self._residue[_key(entry.restore_id, entry.app)] = (
            entry, self._clock() + RESIDUE_TTL_SEC)

    def _repair_residue(self) -> None:
        """Lift streams born at a dead stream's ducked level while nothing is
        being ducked."""
        now = self._clock()
        before = set(self._residue)
        for key in [k for k, (_, expires) in self._residue.items() if expires <= now]:
            del self._residue[key]
        if self._residue:
            for s in self._pactl.sink_inputs():
                dead = self._residue.get(_key(s.restore_id, s.app))
                if dead is not None and s.volume == dead[0].ducked:
                    self._pactl.set_volume(s.index, dead[0].original)
                    del self._residue[_key(s.restore_id, s.app)]
                    log.info("lifted stream %d (%s) born at a dead stream's ducked level",
                             s.index, s.app)
        if set(self._residue) != before:
            self._save_state()

    def _restore_all(self) -> None:
        self._active = False
        self._release_at = None
        self._wake.notify_all()
        self._own_pids.clear()
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
                    self._remember_residue(entry)
                    continue
                if not _is_ours(s.volume, entry):
                    log.info("stream %d volume changed during speech; leaving it", index)
                    continue
                moves[index] = (s.volume, entry.original)
            self._fade(moves, self._fade_up)
            self._verify_restored({i: end for i, (_, end) in moves.items()})
        finally:
            self._save_state()
            if self._residue:
                self._start_poller()

    def _verify_restored(self, targets: dict[int, Volume]) -> None:
        for attempt in (1, 2):
            live = {s.index: s for s in self._pactl.sink_inputs()}
            wrong = {i: v for i, v in targets.items()
                     if i in live and live[i].volume != v}
            if not wrong:
                return
            if attempt == 2:
                log.warning("restore did not take on streams %s (wanted %s)",
                            sorted(wrong), wrong)
                return
            for index, volume in wrong.items():
                self._pactl.set_volume(index, volume)

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
        if self._poll_interval is None or self._poller_running:
            return
        self._poller_running = True
        threading.Thread(target=self._poll, daemon=True,
                         name="voxcaster-duck-poll").start()

    def _poll(self) -> None:
        with self._lock:
            while self._active or self._residue:
                self._wake.wait(self._next_wake_delay())
                self.tick()
            self._poller_running = False

    def _next_wake_delay(self) -> float:
        if not self._active:
            return self._poll_interval * IDLE_POLL_FACTOR
        if self._release_at is None:
            return self._poll_interval
        return min(self._poll_interval, max(0.0, self._release_at - self._clock()))

    # ── state file ────────────────────────────────────────────────────────

    def _save_state(self) -> None:
        if not self._ducked and not self._residue:
            self._clear_state()
            return
        tmp = self._state_path + ".tmp"
        with open(tmp, "w") as fh:
            json.dump({
                "entries": [asdict(e) for e in self._ducked.values()],
                "residue": [asdict(e) for e, _ in self._residue.values()],
            }, fh)
        os.replace(tmp, self._state_path)

    def _load_state(self) -> tuple[list[Ducked], list[Ducked]]:
        try:
            with open(self._state_path) as fh:
                raw = json.load(fh)
            return ([Ducked.from_json(e) for e in raw["entries"]],
                    [Ducked.from_json(e) for e in raw.get("residue", [])])
        except FileNotFoundError:
            return [], []
        except (OSError, ValueError, KeyError, TypeError) as e:
            log.warning("ignoring unreadable duck state %s: %s", self._state_path, e)
            self._clear_state()
            return [], []

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
