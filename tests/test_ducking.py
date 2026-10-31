import json
import time
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
        self.ignore_sets = {}

    def sink_inputs(self):
        if self.fail:
            raise PactlError("pactl unavailable")
        return list(self.streams.values())

    def set_volume(self, index, volume):
        if index not in self.streams:
            return False
        self.sets.append((index, tuple(volume)))
        if self.ignore_sets.get(index, 0) > 0:
            self.ignore_sets[index] -= 1
            return True
        self.streams[index] = replace(self.streams[index], volume=tuple(volume))
        return True

    def set_mute(self, index, muted):
        if index not in self.streams:
            return False
        self.streams[index] = replace(self.streams[index], muted=muted)
        return True

    def muted(self, index):
        return self.streams[index].muted

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
                clock=None, hold=0.0):
    clock = clock or FakeClock()
    return Ducker(
        pactl,
        state_path=str(tmp_path / "duck.json"),
        factor=factor,
        fade_down_sec=down,
        fade_up_sec=up,
        hold_sec=hold,
        enabled=enabled,
        sleep=clock.sleep,
        clock=clock,
        poll_interval=None,
    )


def test_other_streams_are_scaled_per_channel_and_own_stream_is_left(tmp_path):
    pactl = FakePactl(stream(1, (FULL, 32768)), stream(2, pid=777))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    assert pactl.volume(1) == (43872, 21936)
    assert pactl.volume(2) == (FULL, FULL)


@pytest.mark.parametrize("factor", [0.5, 0.3, 0.25])
def test_factor_is_the_amplitude_ratio_on_the_servers_cubic_scale(tmp_path, factor):
    pactl = FakePactl(stream(1, (FULL, FULL)))
    make_ducker(pactl, tmp_path, factor=factor).acquire(own_pid=777)
    amplitude = (pactl.volume(1)[0] / FULL) ** 3
    assert amplitude == pytest.approx(factor, abs=1e-3)


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
    assert down[-1] == 43872
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
    assert pactl.volume(1) == (43872, 43872)
    d.release()
    assert pactl.volume(1) == (FULL, FULL)


def test_gap_shorter_than_the_hold_keeps_the_duck_and_never_ducks_twice(tmp_path):
    pactl = FakePactl(stream(1))
    clock = FakeClock()
    d = make_ducker(pactl, tmp_path, clock=clock, hold=1.5)
    d.acquire(own_pid=777)
    d.release()
    clock.now += 1.0
    d.tick()
    assert pactl.volume(1) == (43872, 43872)
    d.acquire(own_pid=778)
    assert len(pactl.sets) == 1
    d.release()
    clock.now += 2.0
    d.tick()
    assert pactl.volume(1) == (FULL, FULL)


def test_gap_longer_than_the_hold_restores_and_the_next_line_ducks_again(tmp_path):
    pactl = FakePactl(stream(1, (50000, 50000)))
    clock = FakeClock()
    d = make_ducker(pactl, tmp_path, clock=clock, hold=1.5)
    d.acquire(own_pid=777)
    d.release()
    clock.now += 1.6
    d.tick()
    assert pactl.volume(1) == (50000, 50000)
    assert not (tmp_path / "duck.json").exists()
    d.acquire(own_pid=778)
    assert pactl.volume(1) == (33472, 33472)


def test_duck_is_never_released_while_a_line_is_playing(tmp_path):
    pactl = FakePactl(stream(1))
    clock = FakeClock()
    d = make_ducker(pactl, tmp_path, clock=clock, hold=1.5)
    d.acquire(own_pid=777)
    clock.now += 60.0
    d.tick()
    assert pactl.volume(1) == (43872, 43872)


def test_zero_hold_restores_as_soon_as_the_line_ends(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path, hold=0.0)
    d.acquire(own_pid=777)
    d.release()
    assert pactl.volume(1) == (FULL, FULL)


def test_shutdown_restores_a_duck_still_in_its_hold(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path, hold=1.5)
    d.acquire(own_pid=777)
    d.release()
    d.shutdown()
    assert pactl.volume(1) == (FULL, FULL)


def test_background_poller_restores_after_the_hold_without_any_call(tmp_path):
    pactl = FakePactl(stream(1))
    d = Ducker(pactl, state_path=str(tmp_path / "duck.json"), factor=0.3,
               fade_down_sec=0.0, fade_up_sec=0.0, hold_sec=0.05,
               poll_interval=0.02)
    d.acquire(own_pid=777)
    d.release()
    deadline = time.monotonic() + 3.0
    while pactl.volume(1) != (FULL, FULL) and time.monotonic() < deadline:
        time.sleep(0.01)
    assert pactl.volume(1) == (FULL, FULL)


def test_second_line_after_full_restore_ducks_from_the_real_original(tmp_path):
    pactl = FakePactl(stream(1, (50000, 50000)))
    d = make_ducker(pactl, tmp_path)
    for pid in (777, 778):
        d.acquire(own_pid=pid)
        assert pactl.volume(1) == (33472, 33472)
        d.release()
        assert pactl.volume(1) == (50000, 50000)


def test_stream_appearing_mid_speech_is_ducked_and_restored(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[2] = stream(2, (40000, 40000), app="firefox")
    d.tick()
    assert pactl.volume(2) == (26777, 26777)
    d.release()
    assert pactl.volume(2) == (40000, 40000)


def test_stream_that_inherits_the_ducked_volume_restores_to_the_adopted_original(tmp_path):
    role = "sink-input-by-media-role:music"
    pactl = FakePactl(stream(1, (FULL, FULL), restore_id=role))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.streams[2] = stream(2, pactl.volume(1), app="mpv", restore_id=role)
    d.tick()
    assert pactl.volume(2) == (43872, 43872)
    d.release()
    assert pactl.volume(2) == (FULL, FULL)


def test_player_born_at_residue_volume_is_pinned_to_nominal(tmp_path):
    pactl = FakePactl(stream(1), stream(9, (33333,), pid=777, app="mpv"))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    assert pactl.volume(9) == (FULL,)
    assert pactl.volume(1) == (43872, 43872)


def test_player_born_muted_is_unmuted(tmp_path):
    pactl = FakePactl(stream(9, (FULL,), muted=True, pid=777, app="mpv"))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    assert pactl.muted(9) is False


def test_player_already_nominal_is_not_rewritten(tmp_path):
    pactl = FakePactl(stream(9, (FULL, FULL), pid=777, app="mpv"))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    assert pactl.sets == []


def test_player_is_pinned_even_when_ducking_is_disabled(tmp_path):
    pactl = FakePactl(stream(1), stream(9, (900,), pid=777, app="mpv"))
    make_ducker(pactl, tmp_path, enabled=False).acquire(own_pid=777)
    assert pactl.volume(9) == (FULL,)
    assert pactl.volume(1) == (FULL, FULL)


def test_acquire_waits_for_the_player_stream_to_appear(tmp_path):
    pactl = FakePactl()
    clock = FakeClock()
    d = make_ducker(pactl, tmp_path, clock=clock)
    original_sleep = clock.sleep

    def sleep_then_spawn(seconds):
        original_sleep(seconds)
        if clock.now >= 0.2:
            pactl.streams[9] = stream(9, (3000,), pid=777, app="mpv")

    d._sleep = sleep_then_spawn
    d.acquire(own_pid=777)
    assert pactl.volume(9) == (FULL,)


def test_other_apps_streams_are_not_pinned(tmp_path):
    pactl = FakePactl(stream(1, (33333, 33333), pid=100))
    make_ducker(pactl, tmp_path, enabled=False).acquire(own_pid=777)
    assert pactl.sets == []


CHROME = "sink-input-by-application-name:Google Chrome"


def chrome(index, volume=(FULL, FULL)):
    return stream(index, volume, app="Google Chrome", restore_id=CHROME)


def test_stream_reborn_at_a_dead_streams_ducked_volume_keeps_the_true_original(tmp_path):
    pactl = FakePactl(chrome(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    del pactl.streams[1]
    pactl.streams[2] = chrome(2, (43872, 43872))
    d.tick()
    assert pactl.volume(2) == (43872, 43872)
    d.release()
    assert pactl.volume(2) == (FULL, FULL)


def test_repeated_stream_deaths_never_compound_the_duck(tmp_path):
    pactl = FakePactl(chrome(1))
    d = make_ducker(pactl, tmp_path)
    index = 1
    for cycle in range(3):
        d.acquire(own_pid=777)
        assert pactl.volume(index) == (43872, 43872)
        del pactl.streams[index]
        index += 1
        pactl.streams[index] = chrome(index, (43872, 43872))
        d.tick()
        d.release()
        assert pactl.volume(index) == (FULL, FULL)
        pactl.streams[index] = chrome(index)


def test_stream_born_ducked_after_the_line_is_repaired_while_idle(tmp_path):
    pactl = FakePactl(chrome(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    del pactl.streams[1]
    d.release()
    pactl.streams[2] = chrome(2, (43872, 43872))
    d.tick()
    assert pactl.volume(2) == (FULL, FULL)
    d.tick()
    assert len(pactl.sets) == 2
    assert not (tmp_path / "duck.json").exists()


def test_stream_at_a_different_volume_than_the_dead_ones_ducked_level_is_left_alone(tmp_path):
    pactl = FakePactl(chrome(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    del pactl.streams[1]
    d.release()
    pactl.streams[2] = chrome(2, (50000, 50000))
    d.tick()
    assert pactl.volume(2) == (50000, 50000)


def test_residue_is_forgotten_after_its_time_is_up(tmp_path):
    pactl = FakePactl(chrome(1))
    clock = FakeClock()
    d = make_ducker(pactl, tmp_path, clock=clock)
    d.acquire(own_pid=777)
    del pactl.streams[1]
    d.release()
    clock.now += 3600.0
    pactl.streams[2] = chrome(2, (43872, 43872))
    d.tick()
    assert pactl.volume(2) == (43872, 43872)
    assert not (tmp_path / "duck.json").exists()


def test_residue_survives_a_daemon_restart(tmp_path):
    pactl = FakePactl(chrome(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    del pactl.streams[1]
    d.release()
    pactl.streams[2] = chrome(2, (43872, 43872))
    fresh = make_ducker(pactl, tmp_path)
    fresh.recover()
    fresh.tick()
    assert pactl.volume(2) == (FULL, FULL)


def test_recovery_restores_the_recorded_original_exactly_once(tmp_path):
    pactl = FakePactl(stream(1, (51118, 40001)))
    make_ducker(pactl, tmp_path).acquire(own_pid=777)
    make_ducker(pactl, tmp_path).recover()
    assert pactl.volume(1) == (51118, 40001)
    sets = len(pactl.sets)
    make_ducker(pactl, tmp_path).recover()
    assert len(pactl.sets) == sets


def test_restore_that_did_not_take_is_retried_once(tmp_path):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.ignore_sets[1] = 1
    d.release()
    assert pactl.volume(1) == (FULL, FULL)


def test_restore_that_never_takes_is_logged_as_a_warning(tmp_path, caplog):
    pactl = FakePactl(stream(1))
    d = make_ducker(pactl, tmp_path)
    d.acquire(own_pid=777)
    pactl.ignore_sets[1] = 100
    with caplog.at_level("WARNING", logger="voxcaster.ducking"):
        d.release()
    assert any("did not take" in r.message for r in caplog.records)


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
        "fade_down_ms = 100\nfade_up_ms = 250\nhold_ms = 300\n"))))
    assert (cfg.duck_enabled, cfg.duck_factor) == (False, 0.5)
    assert (cfg.duck_fade_down_ms, cfg.duck_fade_up_ms, cfg.duck_hold_ms) == (100, 250, 300)
    default = load_config(str(_write(tmp_path, "")))
    assert (default.duck_factor, default.duck_fade_down_ms,
            default.duck_fade_up_ms, default.duck_hold_ms) == (0.3, 150, 400, 1500)


def test_ducking_factor_outside_unit_interval_is_refused(tmp_path):
    with pytest.raises(ValueError):
        load_config(str(_write(tmp_path, "[ducking]\nfactor = 1.5\n")))


def _write(tmp_path, body):
    path = tmp_path / "config.toml"
    path.write_text(body)
    return path
