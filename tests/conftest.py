import pytest


@pytest.fixture(autouse=True)
def player_out(tmp_path, monkeypatch):
    """Where fake_mpv writes its logs and received audio: outside the daemon's own
    stream directory, so a test can demand that directory ends empty."""
    monkeypatch.setenv("FAKE_PLAYER_OUT", str(tmp_path / "player"))
