import json
from dataclasses import replace

import pytest

from voxcaster.config import load_config
from voxcaster.ducking import Ducker, PactlError, Stream, parse_streams

FULL = 65536


def stream(index, volume=(FULL, FULL), *, muted=False, pid=100, app="chrome",
           restore_id=""):
    return Stream(index=index, volume=tuple(volume), muted=muted, pid=pid,
                  app=app, restore_id=restore_id)


class FakePactl:
    def __init__(self, *streams):
        self.streams = {s.index: s for s in streams}
        self.sets = []
        self.fail = False

    def sink_inputs(self):
        if self.fail:
            raise PactlError("pactl unavailable")
        return list(self.streams.values())

    def set_volume(self, index, volume):
        if index not in self.streams:
            return False
        self.sets.append((index, tuple(volume)))
        self.streams[index] = replace(self.streams[index], volume=tuple(volume))
        return True

    def volume(self, index):
        return self.streams[index].volume


class FakeClock:
    def __init__(self):
        self.now = 0.0
        self.sleeps = []

    def sleep(self, seconds):
        self.sleeps.append(seconds)
        self.now += seconds

    def __call__(self):
        return self.now


def make_ducker(pactl, tmp_path, *, factor=0.3, down=0.0, up=0.0, enabled=True,
                clock=None):
    clock = clock or FakeClock()
    return Ducker(
        pactl,
        state_path=str(tmp_path / "duck.json"),
        factor=factor,
        fade_down_sec=down,
        fade_up_sec=up,
        enabled=enabled,
        sleep=clock.sleep,
        clock=clock,
        poll_interval=None,
    )


def test_other_streams_are_scaled_per_channel_and_own_stream_is_left(tmp_path):
    pactl = FakePactl(stream(1, (FULL, 32768)), stream(2, pid=777))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    assert pactl.volume(1) == (19661, 9830)
    assert pactl.volume(2) == (FULL, FULL)


def test_release_restores_exact_original_channels(tmp_path):
    pactl = FakePactl(stream(1, (51118, 40001)))
    d = make_ducker(pactl, tmp_path, down=0.15, up=0.4)
    d.acquire(own_pid=777)
    assert pactl.volume(1) != (51118, 40001)
    d.release()
    assert pactl.volume(1) == (51118, 40001)


def test_muted_stream_is_never_touched(tmp_path):
    pactl = FakePactl(stream(1, muted=True))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    d.release()
    assert pactl.sets == []


def test_fades_are_monotonic_and_land_exactly(tmp_path):
    pactl = FakePactl(stream(1, (FULL, FULL)), stream(2, pid=777))
    clock = FakeClock()
    d = make_ducker(pactl, tmp_path, down=0.15, up=0.4, clock=clock)
    d.acquire(own_pid=777)
    down = [v[0] for _, v in pactl.sets]
    assert len(down) > 2
    assert down == sorted(down, reverse=True)
    assert down[-1] == 19661
    assert clock.now == pytest.approx(0.15)

    pactl.sets.clear()
    clock.now = 0.0
    d.release()
    up = [v[0] for _, v in pactl.sets]
    assert len(up) > 2
    assert up == sorted(up)
    assert up[-1] == FULL
    assert clock.now == pytest.approx(0.4)


def test_nested_holders_duck_once_and_restore_on_last_release(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    sets_after_first = len(pactl.sets)
    d.acquire(own_pid=778)
    assert len(pactl.sets) == sets_after_first
    d.release()
    assert pactl.volume(1) == (19661, 19661)
    d.release()
    assert pactl.volume(1) == (FULL, FULL)


def test_back_to_back_lines_never_record_ducked_volume_as_original(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    d.release(keep_ducked=True)
    assert pactl.volume(1) == (19661, 19661)
    d.acquire(own_pid=778)
    assert len(pactl.sets) == 1
    d.release()
    assert pactl.volume(1) == (FULL, FULL)


def test_settle_restores_a_held_duck(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    d.release(keep_ducked=True)
    d.settle()
    assert pactl.volume(1) == (FULL, FULL)


def test_second_line_after_full_restore_ducks_from_the_real_original(tmp_path):
    pactl = FakePactl(stream(1, (50000, 50000)))
    d = make_ducker(pactl, tmp_path)
    for pid in (777, 778):
        d.acquire(own_pid=pid)
        assert pactl.volume(1) == (15000, 15000)
        d.release()
        assert pactl.volume(1) == (50000, 50000)


def test_stream_appearing_mid_speech_is_ducked_and_restored(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[2] = stream(2, (40000, 40000), app="firefox")
    d.rescan()
    assert pactl.volume(2) == (12000, 12000)
    d.release()
    assert pactl.volume(2) == (40000, 40000)


def test_stream_that_inherits_the_ducked_volume_restores_to_the_adopted_original(tmp_path):
    role = "sink-input-by-media-role:music"
    pactl = FakePactl(stream(1, (FULL, FULL), restore_id=role))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[2] = stream(2, pactl.volume(1), app="mpv", restore_id=role)
    d.rescan()
    assert pactl.volume(2) == (19661, 19661)
    d.release()
    assert pactl.volume(2) == (FULL, FULL)


def test_player_born_at_a_ducked_siblings_volume_is_lifted_to_the_siblings_original(tmp_path):
    role = "sink-input-by-media-role:music"
    pactl = FakePactl(stream(1, (50000, 50000), restore_id=role))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[9] = stream(9, (15000, 15000), pid=777, app="mpv", restore_id=role)
    d.rescan()
    assert pactl.volume(9) == (50000, 50000)
    assert pactl.volume(1) == (15000, 15000)


def test_player_is_lifted_once_even_with_several_ducked_siblings(tmp_path):
    role = "sink-input-by-media-role:music"
    pactl = FakePactl(stream(1, (50000, 50000), restore_id=role),
                      stream(2, (33333, 33333), restore_id=role))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[9] = stream(9, (10000, 10000), pid=777, app="mpv", restore_id=role)
    d.rescan()
    lifted = pactl.volume(9)
    assert lifted == (33333, 33333)
    d.rescan()
    assert pactl.volume(9) == lifted


def test_player_with_its_own_restore_key_is_never_adjusted(tmp_path):
    pactl = FakePactl(stream(1, restore_id="sink-input-by-media-role:music"))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[9] = stream(9, (30000, 30000), pid=777, app="mpv",
                              restore_id="sink-input-by-media-role:event")
    d.rescan()
    assert pactl.volume(9) == (30000, 30000)


def test_stream_vanishing_mid_speech_is_forgotten(tmp_path):
    pactl = FakePactl(stream(1), stream(2, app="firefox"))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    del pactl.streams[1]
    d.release()
    assert pactl.volume(2) == (FULL, FULL)
    assert all(index != 1 for index, _ in pactl.sets[-2:])


def test_reused_index_belonging_to_another_app_is_not_restored(tmp_path):
    pactl = FakePactl(stream(1, app="chrome"))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[1] = stream(1, (30000, 30000), app="vlc")
    d.release()
    assert pactl.volume(1) == (30000, 30000)


def test_volume_the_user_changed_mid_speech_is_left_alone(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[1] = replace(pactl.streams[1], volume=(5000, 5000))
    sets_before = len(pactl.sets)
    d.release()
    assert len(pactl.sets) == sets_before
    assert pactl.volume(1) == (5000, 5000)


def test_state_file_exists_only_while_ducked(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    state = tmp_path / "duck.json"
    d.acquire(own_pid=777)
    saved = json.loads(state.read_text())
    assert saved["entries"][0]["original"] == [FULL, FULL]
    d.release()
    assert not state.exists()


def test_crash_while_ducked_is_repaired_by_the_next_start(tmp_path):
    pactl = FakePactl(stream(1, (51118, 40001)))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    assert pactl.volume(1) != (51118, 40001)
    fresh = make_ducker(pactl, tmp_path)
    fresh.recover()
    assert pactl.volume(1) == (51118, 40001)
    assert not (tmp_path / "duck.json").exists()


def test_recovery_finds_a_new_stream_that_inherited_the_ducked_volume(tmp_path):
    role = "sink-input-by-media-role:music"
    pactl = FakePactl(stream(1, restore_id=role))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    ducked = pactl.volume(1)
    pactl.streams = {9: stream(9, ducked, app="chrome", restore_id=role)}
    make_ducker(pactl, tmp_path).recover()
    assert pactl.volume(9) == (FULL, FULL)


def test_recovery_leaves_a_stream_the_user_has_since_changed(tmp_path):
    pactl = FakePactl(stream(1))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    pactl.streams[1] = replace(pactl.streams[1], volume=(5000, 5000))
    pactl.sets.clear()
    make_ducker(pactl, tmp_path).recover()
    assert pactl.sets == []


def test_recovery_runs_even_when_ducking_is_disabled(tmp_path):
    pactl = FakePactl(stream(1))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    make_ducker(pactl, tmp_path, enabled=False).recover()
    assert pactl.volume(1) == (FULL, FULL)


def test_disabled_ducker_never_touches_the_server(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path, enabled=False)
    d.acquire(own_pid=777)
    d.release()
    assert pactl.sets == []


def test_server_failure_never_raises_into_playback(tmp_path):
    pactl = FakePactl(stream(1))
    pactl.fail = True
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    d.release()


def test_parse_streams_joins_the_owning_client_pid_and_orders_channels():
    sink_inputs = json.dumps([{
        "index": 9556, "client": "9555", "mute": False,
        "channel_map": "front-left,front-right",
        "volume": {"front-right": {"value": 111}, "front-left": {"value": 222}},
        "properties": {"application.name": "voxduck_synth",
                       "module-stream-restore.id": "sink-input-by-media-role:music"},
    }, {
        "index": 9557, "client": "7", "mute": True,
        "channel_map": "mono",
        "volume": {"mono": {"value": 5}},
        "properties": {},
    }])
    clients = json.dumps([
        {"index": 9555, "properties": {"application.process.id": "3739267"}},
        {"index": 7, "properties": {}},
    ])
    a, b = parse_streams(sink_inputs, clients)
    assert (a.index, a.volume, a.pid, a.muted) == (9556, (222, 111), 3739267, False)
    assert a.restore_id == "sink-input-by-media-role:music"
    assert (b.pid, b.muted, b.volume, b.app) == (None, True, (5,), "")


def test_ducking_config_defaults_and_toml(tmp_path):
    assert load_config(str(_write(tmp_path, ""))).duck_enabled is True
    cfg = load_config(str(_write(tmp_path, (
        "[ducking]\nenabled = false\nfactor = 0.5\n"
        "fade_down_ms = 100\nfade_up_ms = 250\n"))))
    assert (cfg.duck_enabled, cfg.duck_factor) == (False, 0.5)
    assert (cfg.duck_fade_down_ms, cfg.duck_fade_up_ms) == (100, 250)
    default = load_config(str(_write(tmp_path, "")))
    assert (default.duck_factor, default.duck_fade_down_ms,
            default.duck_fade_up_ms) == (0.3, 150, 400)


def test_ducking_factor_outside_unit_interval_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_config(str(_write(tmp_path, "[ducking]\nfactor = 1.5\n")))


def _write(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path
