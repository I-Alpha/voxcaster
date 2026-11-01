"""One running player process: live volume and pause over mpv's JSON IPC, and
the duck that is held only while the voice is audible."""
from __future__ import annotations

import json
import logging
import os
import socket
import subprocess
import threading
import time
from typing import Callable, NamedTuple, Protocol

log = logging.getLogger("voxcaster.playback")

IPC_WAIT_SEC = 1.0
IPC_POLL_SEC = 0.02
IPC_TIMEOUT_SEC = 0.5
WAIT_SLICE_SEC = 0.25


class Mix(NamedTuple):
    """What the listener has asked for; the daemon owns it."""
    volume: int
    paused: bool

    @property
    def audible(self) -> bool:
        return self.volume > 0 and not self.paused


class Duck(Protocol):
    def acquire(self, own_pid: int) -> None: ...

    def duck(self, own_pid: int) -> None: ...

    def release(self) -> None: ...


class MpvControl:
    def __init__(self, path: str) -> None:
        self.path = path

    def set_property(self, name: str, value) -> bool:
        request = json.dumps({"command": ["set_property", name, value], "request_id": 1})
        try:
            with socket.socket(socket.AF_UNIX) as s:
                s.settimeout(IPC_TIMEOUT_SEC)
                s.connect(self.path)
                s.sendall((request + "\n").encode())
                for raw in s.makefile():
                    reply = json.loads(raw)
                    if "error" in reply:
                        return reply["error"] == "success"
        except (OSError, ValueError) as e:
            log.debug("player control %s=%r failed: %s", name, value, e)
        return False

    def wait_until_listening(self, proc: subprocess.Popen) -> bool:
        deadline = time.monotonic() + IPC_WAIT_SEC
        while not os.path.exists(self.path):
            if proc.poll() is not None or time.monotonic() >= deadline:
                return False
            time.sleep(IPC_POLL_SEC)
        return True


class Playback:
    """``mix`` is read under the playback's lock at each send, so the last
    request wins however the sends interleave. Without a control channel the
    player cannot be adjusted and counts as audible throughout."""

    def __init__(self, proc: subprocess.Popen, control: MpvControl | None,
                 duck: Duck, mix: Callable[[], Mix]) -> None:
        self._proc = proc
        self._control = control
        self._duck = duck
        self._mix = mix
        self._lock = threading.Lock()
        self._duck_lock = threading.Lock()
        self._pinned = False
        self._ducking = False
        self._finished = False
        self._in_sync = False

    @property
    def alive(self) -> bool:
        return self._proc.poll() is None

    @property
    def returncode(self) -> int | None:
        return self._proc.returncode

    def start(self) -> None:
        if self._control is not None and not self._control.wait_until_listening(self._proc):
            log.warning("player control socket never appeared: %s", self._control.path)
        self.apply()

    def apply(self) -> None:
        with self._lock:
            if self._finished:
                return
            if self._control is not None:
                mix = self._mix()
                volume_sent = self._control.set_property("volume", mix.volume)
                pause_sent = self._control.set_property("pause", mix.paused)
                self._in_sync = volume_sent and pause_sent
        self._settle_duck()

    def _settle_duck(self) -> None:
        """Duck while audible, release otherwise. The slow duck calls run outside
        the property lock; the mix is read again here so the last settle wins."""
        with self._duck_lock:
            if self._finished:
                return
            audible = self._mix().audible if self._control is not None else True
            if audible and not self._ducking:
                (self._duck.duck if self._pinned else self._duck.acquire)(self._proc.pid)
                self._pinned = self._ducking = True
            elif not audible and self._ducking:
                self._duck.release()
                self._ducking = False

    def wait(self, budget_sec: float) -> None:
        """Wait for the player to end, re-sending the mix while the last send
        failed. Time the player is paused does not count against ``budget_sec``;
        a player still running when the budget is spent is killed."""
        while True:
            try:
                self._proc.wait(timeout=WAIT_SLICE_SEC)
                return
            except subprocess.TimeoutExpired:
                if self._control is not None and not self._in_sync:
                    self.apply()
                if self._control is None or not self._mix().paused:
                    budget_sec -= WAIT_SLICE_SEC
                if budget_sec <= 0:
                    self._proc.kill()
                    self._proc.wait()
                    return

    def terminate(self) -> bool:
        if not self.alive:
            return False
        self._proc.terminate()
        return True

    def finish(self) -> None:
        with self._lock:
            self._finished = True
        with self._duck_lock:
            if self._ducking:
                self._duck.release()
                self._ducking = False
