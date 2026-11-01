import json
import os

import pytest

from voxcaster import protocol
from voxcaster.traylogic import (
    ICON_DIR,
    PRESETS,
    WAVE_FRAMES,
    TrayController,
    db_to_volume,
    icon_name,
    line_label,
    set_toml_value,
    volume_to_db,
)

CONFIG = """# user config
[client]
ack_timeout = 3600.0   # long on purpose

[ducking]
enabled = true
# depth comment
factor = 0.3
hold_ms = 1500

[audio]
# calibration notes
volume = 116  # live value
"""


class FakeWire:
    def __init__(self, status=None):
        self.sent = []
        self.status_reply = status

    def send(self, payload):
        self.sent.append(payload)

    def request(self, payload):
        self.sent.append(payload)
        if payload == protocol.encode_status() and self.status_reply is not None:
            return (json.dumps(self.status_reply) + "\n").encode()
        return protocol.ACK


def controller(tmp_path, wire=None, text=CONFIG):
    path = tmp_path / "config.toml"
    path.write_text(text)
    calls = []
    c = TrayController(
        wire or FakeWire(),
        config_path=str(path),
        state_path=str(tmp_path / "tray.json"),
        run=lambda argv: calls.append(argv),
    )
    return c, path, calls


def test_toml_value_is_replaced_in_place_keeping_comments_and_layout():
    out = set_toml_value(CONFIG, "audio", "volume", 90)
    assert out == CONFIG.replace("volume = 116  # live value", "volume = 90  # live value")


def test_toml_value_in_one_section_does_not_touch_the_same_key_elsewhere():
    text = "[a]\nx = 1\n[b]\nx = 2\n"
    assert set_toml_value(text, "b", "x", 9) == "[a]\nx = 1\n[b]\nx = 9\n"


@pytest.mark.parametrize("value,rendered", [(True, "true"), (False, "false"),
                                            (0.5, "0.5"), (1500, "1500")])
def test_toml_scalars_are_rendered_as_toml(value, rendered):
    out = set_toml_value(CONFIG, "ducking", "enabled", value)
    assert f"enabled = {rendered}" in out


def test_missing_key_is_added_to_its_section_after_the_last_key():
    out = set_toml_value(CONFIG, "client", "connect_timeout", 0.5)
    assert out.index("connect_timeout = 0.5") > out.index("ack_timeout")
    assert out.index("connect_timeout = 0.5") < out.index("[ducking]")


def test_missing_section_is_appended():
    out = set_toml_value("[a]\nx = 1\n", "ducking", "enabled", False)
    assert out.endswith("[ducking]\nenabled = false\n")
    assert out.startswith("[a]\nx = 1\n")


def test_edit_that_would_break_the_file_is_refused():
    with pytest.raises(ValueError):
        set_toml_value("[a]\nx = 1\n", "a", "bad key", 1)


def test_dB_and_volume_follow_the_players_cubic_scale():
    assert volume_to_db(100) == pytest.approx(0.0)
    assert volume_to_db(130) == pytest.approx(6.84, abs=0.01)
    assert db_to_volume(0.0) == 100
    assert db_to_volume(-6.0) == 79
    assert db_to_volume(99.0) == protocol.VOLUME_MAX


def test_presets_cover_quiet_to_loud_in_increasing_volume():
    volumes = [db_to_volume(db) for _, db in PRESETS]
    assert volumes == sorted(volumes) and len(set(volumes)) == len(volumes)


def test_set_volume_sends_the_wire_message_and_persists_to_the_config(tmp_path):
    wire = FakeWire()
    c, path, _ = controller(tmp_path, wire)
    c.set_volume(88)
    assert protocol.encode_volume(88) in wire.sent
    assert "volume = 88  # live value" in path.read_text()


def test_volume_is_clamped_to_the_wire_range(tmp_path):
    wire = FakeWire()
    c, path, _ = controller(tmp_path, wire)
    c.set_volume(999)
    assert protocol.encode_volume(130) in wire.sent
    assert "volume = 130" in path.read_text()


def test_mute_sends_zero_without_touching_the_saved_volume_and_unmute_restores(tmp_path):
    wire = FakeWire({"volume": 116})
    c, path, _ = controller(tmp_path, wire)
    c.toggle_mute()
    assert wire.sent[-1] == protocol.encode_volume(0)
    assert "volume = 116  # live value" in path.read_text()
    wire.status_reply = {"volume": 0}
    c.toggle_mute()
    assert wire.sent[-1] == protocol.encode_volume(116)


def test_unmute_with_no_memory_falls_back_to_the_configured_volume(tmp_path):
    wire = FakeWire({"volume": 0})
    c, _, _ = controller(tmp_path, wire)
    c.toggle_mute()
    assert wire.sent[-1] == protocol.encode_volume(116)


def test_ducking_depth_writes_the_factor_and_asks_the_daemon_to_reload(tmp_path):
    wire = FakeWire()
    c, path, _ = controller(tmp_path, wire)
    c.set_ducking(factor=0.25)
    assert "factor = 0.25" in path.read_text()
    assert wire.sent[-1] == protocol.encode_reload()


def test_ducking_toggle_and_hold_are_written_and_reloaded(tmp_path):
    wire = FakeWire()
    c, path, _ = controller(tmp_path, wire)
    c.set_ducking(enabled=False, hold_ms=3000)
    text = path.read_text()
    assert "enabled = false" in text and "hold_ms = 3000" in text
    assert wire.sent.count(protocol.encode_reload()) == 1


def test_stop_and_test_voice_use_the_wire_messages(tmp_path):
    wire = FakeWire()
    c, _, _ = controller(tmp_path, wire)
    c.stop_speech()
    c.test_voice()
    assert wire.sent[0] == protocol.encode_stop()
    assert wire.sent[1].startswith(b"Voxcaster") or b"test" in wire.sent[1]


def test_restart_and_open_config_run_the_expected_commands(tmp_path):
    c, path, calls = controller(tmp_path)
    c.restart_daemon()
    c.open_config()
    assert calls[0] == ["systemctl", "--user", "restart", "voxcaster.service"]
    assert calls[1] == ["xdg-open", str(path)]


def test_status_is_none_when_the_daemon_does_not_answer(tmp_path):
    class Down(FakeWire):
        def request(self, payload):
            raise OSError("down")

    c, _, _ = controller(tmp_path, Down())
    assert c.status() is None


def test_agent_voice_map_is_read_only(tmp_path):
    agents = tmp_path / "agents.json"
    agents.write_text(json.dumps({"agents": {"builder": "am_michael"}}))
    c, _, _ = controller(tmp_path)
    c._agents_path = str(agents)
    assert c.agent_voices() == [("builder", "am_michael")]
    assert json.loads(agents.read_text()) == {"agents": {"builder": "am_michael"}}


def test_icon_name_per_state_and_wave_frames_cycle_while_speaking():
    idle = {"volume": 100, "speaking": False}
    speaking = {"volume": 100, "speaking": True}
    assert icon_name(None) == "voxcaster-down-symbolic"
    assert icon_name({"volume": 0, "speaking": True}) == "voxcaster-muted-symbolic"
    assert icon_name(idle, frame=2) == "voxcaster-idle-symbolic"
    frames = [icon_name(speaking, i) for i in range(len(WAVE_FRAMES) + 1)]
    assert frames[0] == frames[len(WAVE_FRAMES)] != frames[1]
    every = set(frames) | {icon_name(idle), icon_name(None), icon_name({"volume": 0, "speaking": False})}
    for name in every:
        assert os.path.exists(os.path.join(ICON_DIR, name + ".svg"))


def test_skip_and_clear_send_their_wire_messages(tmp_path):
    wire = FakeWire()
    c, _, _ = controller(tmp_path, wire)
    c.skip()
    c.clear_queue()
    assert wire.sent == [protocol.encode_skip(), protocol.encode_clear()]


def test_pause_toggle_follows_the_daemons_state(tmp_path):
    wire = FakeWire({"paused": False, "volume": 100})
    c, _, _ = controller(tmp_path, wire)
    assert c.toggle_pause() is True
    assert wire.sent[-1] == protocol.encode_pause()
    wire.status_reply = {"paused": True, "volume": 100}
    assert c.toggle_pause() is False
    assert wire.sent[-1] == protocol.encode_resume()


def test_pause_toggle_with_the_daemon_down_sends_nothing(tmp_path):
    class Down(FakeWire):
        def request(self, payload):
            raise ConnectionRefusedError

    wire = Down()
    c, _, _ = controller(tmp_path, wire)
    assert c.toggle_pause() is None
    assert wire.sent == []


def test_drop_line_reports_whether_the_daemon_removed_it(tmp_path):
    class Refusing(FakeWire):
        def request(self, payload):
            self.sent.append(payload)
            return b"ERR no queued line 9\n"

    c, _, _ = controller(tmp_path, FakeWire())
    assert c.drop_line(4) is True
    r, _, _ = controller(tmp_path, Refusing())
    assert r.drop_line(9) is False


def test_a_queued_line_is_labelled_with_its_agent_unless_the_voice_is_shared():
    voices = [("builder", "am_michael"), ("scribe", "bf_emma"), ("clerk", "bf_emma")]
    assert line_label({"voice": "am_michael", "text": "hello"}, voices) == "builder: hello"
    assert line_label({"voice": "bf_emma", "text": "hello"}, voices) == "bf_emma: hello"
    assert line_label({"voice": None, "text": "hello"}, voices) == "default: hello"
    assert line_label({"voice": None, "text": "two\nlines"}, voices) == "default: two lines"


def test_a_preview_at_the_daemons_cut_length_shows_an_ellipsis():
    cut = "x" * protocol.PREVIEW_CHARS
    assert line_label({"voice": None, "text": cut}, []).endswith("...")
    assert not line_label({"voice": None, "text": cut[:-1]}, []).endswith("...")
    spaced = ("word " * 20)[:protocol.PREVIEW_CHARS]
    assert line_label({"voice": None, "text": spaced}, []).endswith("...")


def test_a_paused_line_does_not_animate_the_speaking_icon():
    paused = {"volume": 100, "speaking": True, "paused": True}
    assert icon_name(paused, frame=1) == "voxcaster-idle-symbolic"
