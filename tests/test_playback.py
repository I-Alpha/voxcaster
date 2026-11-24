import subprocess
import sys
import threading
import time

from fakes import RecordingDucker

from voxcaster import playback
from voxcaster.playback import Mix, Playback


class StubControl:
    def set_property(self, name, value):
        return True


def run_wait(control, monkeypatch, starved=lambda: False):
    monkeypatch.setattr(playback, "WAIT_SLICE_SEC", 0.05)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    paused = {"now": True}
    threading.Timer(0.5, lambda: paused.update(now=False)).start()
    player = Playback(proc, control, RecordingDucker(), lambda: Mix(100, paused["now"]))
    began = time.monotonic()
    player.wait(lambda: 0.2, starved)
    return time.monotonic() - began, proc.returncode


def test_paused_time_does_not_count_against_the_hang_budget(monkeypatch):
    elapsed, returncode = run_wait(StubControl(), monkeypatch)
    assert elapsed >= 0.5 and returncode == -9


def test_a_player_that_cannot_be_paused_is_still_bounded_while_the_daemon_is_paused(monkeypatch):
    elapsed, returncode = run_wait(None, monkeypatch)
    assert elapsed < 0.5 and returncode == -9


def test_time_spent_waiting_for_the_renderer_does_not_count_against_the_hang_budget(monkeypatch):
    began = time.monotonic()
    elapsed, returncode = run_wait(None, monkeypatch, starved=lambda: time.monotonic() - began < 0.5)
    assert elapsed >= 0.5 and returncode == -9


def test_a_budget_that_grows_while_the_player_runs_is_honoured(monkeypatch):
    monkeypatch.setattr(playback, "WAIT_SLICE_SEC", 0.05)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(0.8)"])
    player = Playback(proc, None, RecordingDucker(), lambda: Mix(100, False))
    began = time.monotonic()
    player.wait(lambda: 0.3 if time.monotonic() - began < 0.2 else 30)
    assert proc.returncode == 0
