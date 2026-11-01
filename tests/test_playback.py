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


def run_wait(control, monkeypatch):
    monkeypatch.setattr(playback, "WAIT_SLICE_SEC", 0.05)
    proc = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(30)"])
    paused = {"now": True}
    threading.Timer(0.5, lambda: paused.update(now=False)).start()
    player = Playback(proc, control, RecordingDucker(), lambda: Mix(100, paused["now"]))
    began = time.monotonic()
    player.wait(0.2)
    return time.monotonic() - began, proc.returncode


def test_paused_time_does_not_count_against_the_hang_budget(monkeypatch):
    elapsed, returncode = run_wait(StubControl(), monkeypatch)
    assert elapsed >= 0.5 and returncode == -9


def test_a_player_that_cannot_be_paused_is_still_bounded_while_the_daemon_is_paused(monkeypatch):
    elapsed, returncode = run_wait(None, monkeypatch)
    assert elapsed < 0.5 and returncode == -9
