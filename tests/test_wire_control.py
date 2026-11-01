import asyncio
import glob
import inspect
import json
import os
import sys
import time

import numpy as np
from fakes import RecordingDucker

from voxcaster import playback, protocol
from voxcaster.config import Config
from voxcaster.daemon import VoiceDaemon


FAKE_PLAYER = os.path.join(os.path.dirname(__file__), "fake_mpv.py")
PLAYER_TEMPLATE = [sys.executable, FAKE_PLAYER, "--volume={volume}",
                   "--input-ipc-server={ipc}", "{file}"]


def make_cfg(tmp_path, config_text=""):
    config_file = tmp_path / "config.toml"
    config_file.write_text(config_text)
    return Config(
        socket_path=str(tmp_path / "v.sock"),
        duck_state_path=str(tmp_path / "duck.json"),
        duck_enabled=False,
        device="cpu",
        volume=116,
        source_file=str(config_file),
        player=PLAYER_TEMPLATE,
    )


async def request(cfg, payload, timeout=5.0):
    reader, writer = await asyncio.open_unix_connection(cfg.socket_path)
    writer.write(payload)
    await writer.drain()
    reply = await asyncio.wait_for(reader.readline(), timeout)
    writer.close()
    return reply


async def status(cfg):
    return json.loads(await request(cfg, protocol.encode_status()))


def run_daemon(cfg, scenario):
    async def main():
        daemon = VoiceDaemon(cfg)
        task = asyncio.create_task(daemon.run())
        for _ in range(200):
            if os.path.exists(cfg.socket_path) and daemon._stop is not None:
                break
            await asyncio.sleep(0.02)
        try:
            return await scenario(daemon)
        finally:
            daemon._stop.set()
            await task

    return asyncio.run(main())


def test_status_reports_volume_queue_and_ducking_settings(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        return await status(cfg)

    s = run_daemon(cfg, scenario)
    assert s["volume"] == 116
    assert s["queued"] == 0 and s["busy"] is False
    assert s["ducking"] == {"enabled": False, "factor": 0.3, "fade_down_ms": 150,
                            "fade_up_ms": 400, "hold_ms": 1500}


def test_live_volume_message_shows_in_status(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await request_no_reply(cfg, protocol.encode_volume(42))
        await asyncio.sleep(0.1)
        return await status(cfg)

    assert run_daemon(cfg, scenario)["volume"] == 42


async def request_no_reply(cfg, payload):
    reader, writer = await asyncio.open_unix_connection(cfg.socket_path)
    writer.write(payload)
    await writer.drain()
    writer.close()


def test_reload_applies_the_ducking_section_without_a_restart(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        (tmp_path / "config.toml").write_text(
            "[ducking]\nenabled = true\nfactor = 0.5\nfade_down_ms = 90\n"
            "fade_up_ms = 200\nhold_ms = 700\n")
        reply = await request(cfg, protocol.encode_reload())
        return reply, await status(cfg)

    reply, s = run_daemon(cfg, scenario)
    assert reply == protocol.ACK
    assert s["ducking"] == {"enabled": True, "factor": 0.5, "fade_down_ms": 90,
                            "fade_up_ms": 200, "hold_ms": 700}


def test_reload_of_a_broken_config_is_refused_and_keeps_the_old_settings(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        (tmp_path / "config.toml").write_text("[ducking]\nfactor = 4.0\n")
        reply = await request(cfg, protocol.encode_reload())
        return reply, await status(cfg)

    reply, s = run_daemon(cfg, scenario)
    assert reply.startswith(b"ERR")
    assert s["ducking"]["factor"] == 0.3


def test_muted_daemon_does_not_synthesize(tmp_path):
    cfg = make_cfg(tmp_path)
    calls = []

    async def scenario(daemon):
        daemon.engine.synthesize = lambda *a, **k: calls.append(a)
        await request_no_reply(cfg, protocol.encode_volume(0))
        await asyncio.sleep(0.1)
        reader, writer = await asyncio.open_unix_connection(cfg.socket_path)
        writer.write(protocol.encode_speak("hello"))
        await writer.drain()
        ack = await asyncio.wait_for(reader.readline(), 5)
        writer.close()
        return ack

    assert run_daemon(cfg, scenario) == protocol.ACK
    assert calls == []


async def speak(cfg, *texts, voice=None):
    for text in texts:
        await request_no_reply(cfg, protocol.encode_speak(text, voice=voice))


async def until(predicate, timeout=4.0):
    deadline = asyncio.get_running_loop().time() + timeout
    while not (await predicate() if inspect.iscoroutinefunction(predicate) else predicate()):
        assert asyncio.get_running_loop().time() < deadline, "condition never held"
        await asyncio.sleep(0.02)


def player_records(cfg):
    records = []
    for path in glob.glob(os.path.join(os.path.dirname(cfg.socket_path), "*.log")):
        records += [json.loads(line) for line in open(path)]
    return records


def commands(cfg):
    return [r["command"] for r in player_records(cfg) if "command" in r]


def spawns(cfg):
    return [r for r in player_records(cfg) if "spawn" in r]


def volume_commands(cfg):
    return [c[2] for c in commands(cfg) if c[:2] == ["set_property", "volume"]]


def run_playing_daemon(cfg, scenario):
    def silent_synth(text, voice=None):
        return np.zeros(2400, dtype="float32"), 24000

    async def wrapped(daemon):
        daemon.engine.synthesize = silent_synth
        daemon._ducker = RecordingDucker()
        return await scenario(daemon)

    return run_daemon(cfg, wrapped)


async def wait_speaking(cfg):
    async def speaking():
        return (await status(cfg))["speaking"]

    await until(speaking)


def test_stop_drains_the_queue_and_kills_the_player(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one", "two", "three")
        await wait_speaking(cfg)
        before = await status(cfg)
        reply = await request(cfg, protocol.encode_stop())
        await until(lambda: not daemon._current_player)
        return before, reply, await status(cfg)

    before, reply, after = run_playing_daemon(cfg, scenario)
    assert before["queued"] == 2
    assert reply == protocol.ACK
    assert after["queued"] == 0 and after["speaking"] is False


def test_volume_message_reaches_the_playing_line(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one")
        await wait_speaking(cfg)
        await until(lambda: volume_commands(cfg) == [116])
        await request_no_reply(cfg, protocol.encode_volume(42))
        await until(lambda: volume_commands(cfg) == [116, 42])
        return await status(cfg)

    assert run_playing_daemon(cfg, scenario)["speaking"] is True


def test_mute_silences_the_playing_line_and_unmute_restores_it_without_a_respawn(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one")
        await wait_speaking(cfg)
        await until(lambda: volume_commands(cfg) == [116])
        await request_no_reply(cfg, protocol.encode_volume(0))
        await until(lambda: volume_commands(cfg)[-1:] == [0])
        await request_no_reply(cfg, protocol.encode_volume(80))
        await until(lambda: volume_commands(cfg)[-1:] == [80])
        return await status(cfg)

    assert run_playing_daemon(cfg, scenario)["speaking"] is True
    assert len(spawns(cfg)) == 1


def test_volume_sent_before_the_player_listens_is_applied_once_it_does(tmp_path, monkeypatch):
    monkeypatch.setattr(playback, "IPC_WAIT_SEC", 0.2)
    monkeypatch.setenv("FAKE_PLAYER_BIND_DELAY", "0.8")
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one")
        await wait_speaking(cfg)
        await request_no_reply(cfg, protocol.encode_volume(33))
        await until(lambda: volume_commands(cfg)[-1:] == [33], timeout=6)

    run_playing_daemon(cfg, scenario)


def test_skip_ends_only_the_current_line(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one", "two")
        await until(lambda: len(spawns(cfg)) == 1)
        reply = await request(cfg, protocol.encode_skip())
        await until(lambda: len(spawns(cfg)) == 2)
        return reply, await status(cfg)

    reply, s = run_playing_daemon(cfg, scenario)
    assert reply == protocol.ACK
    assert s["current"]["text"] == "two" and s["queued"] == 0


def test_pause_freezes_the_line_and_holds_the_queue_until_resume(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PLAYER_PLAY_SEC", "0.5")
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one", "two")
        await wait_speaking(cfg)
        assert await request(cfg, protocol.encode_pause()) == protocol.ACK
        await until(lambda: ["set_property", "pause", True] in commands(cfg))
        await asyncio.sleep(1.0)
        paused = await status(cfg)
        spawned_while_paused = len(spawns(cfg))
        assert await request(cfg, protocol.encode_resume()) == protocol.ACK
        await until(lambda: len(spawns(cfg)) == 2)
        return paused, spawned_while_paused, await status(cfg)

    paused, spawned_while_paused, resumed = run_playing_daemon(cfg, scenario)
    assert paused["paused"] is True and paused["current"]["text"] == "one"
    assert spawned_while_paused == 1
    assert resumed["paused"] is False
    assert ["set_property", "pause", False] in commands(cfg)


def test_clear_drops_the_queued_lines_and_keeps_the_current_one(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one", "two", "three")
        await wait_speaking(cfg)
        reply = await request(cfg, protocol.encode_clear())
        return reply, await status(cfg)

    reply, s = run_playing_daemon(cfg, scenario)
    assert reply == protocol.ACK
    assert s["queued"] == 0 and s["queue"] == []
    assert s["speaking"] is True and s["current"]["text"] == "one"


def test_drop_removes_exactly_the_named_queued_line(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one", "two", "three", "four")
        await wait_speaking(cfg)
        listed = (await status(cfg))["queue"]
        victim = next(q["id"] for q in listed if q["text"] == "three")
        reply = await request(cfg, protocol.encode_drop(victim))
        again = await request(cfg, protocol.encode_drop(victim))
        return listed, reply, again, await status(cfg)

    listed, reply, again, s = run_playing_daemon(cfg, scenario)
    assert [q["text"] for q in listed] == ["two", "three", "four"]
    assert reply == protocol.ACK
    assert again.startswith(b"ERR")
    assert [q["text"] for q in s["queue"]] == ["two", "four"]
    assert s["queued"] == 2


def test_status_lists_the_current_line_and_the_queue_with_unique_ids_and_short_previews(tmp_path):
    cfg = make_cfg(tmp_path)
    long_text = "word " * 40

    async def scenario(daemon):
        await speak(cfg, long_text, "next", voice="am_michael")
        await speak(cfg, "last")
        await wait_speaking(cfg)
        await asyncio.sleep(0.15)
        return await status(cfg)

    s = run_playing_daemon(cfg, scenario)
    assert s["current"]["text"] == long_text.strip()[:60]
    assert s["current"]["elapsed_sec"] > 0
    assert [q["text"] for q in s["queue"]] == ["next", "last"]
    assert s["queue"][0]["voice"] == "am_michael"
    ids = [s["current"]["id"]] + [q["id"] for q in s["queue"]]
    assert len(set(ids)) == 3 and ids == sorted(ids)


def test_the_duck_is_held_only_while_the_voice_is_audible(tmp_path, monkeypatch):
    monkeypatch.setenv("FAKE_PLAYER_PLAY_SEC", "3")
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        ducker = daemon._ducker
        await speak(cfg, "one")
        await wait_speaking(cfg)
        await until(lambda: ducker.calls == ["acquire"])
        for message, expected in ((protocol.encode_pause(), "release"),
                                  (protocol.encode_resume(), "duck"),
                                  (protocol.encode_volume(0), "release"),
                                  (protocol.encode_volume(50), "duck")):
            before = len(ducker.calls)
            await request(cfg, message)
            await until(lambda: len(ducker.calls) > before)
            assert ducker.calls[-1] == expected
        await request(cfg, protocol.encode_stop())
        await until(lambda: ducker.calls[-1] == "release" and len(ducker.calls) == 6)

    run_playing_daemon(cfg, scenario)


def test_stop_clears_a_pause_so_the_next_line_is_spoken(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one")
        await until(lambda: len(spawns(cfg)) == 1)
        await request(cfg, protocol.encode_pause())
        await request(cfg, protocol.encode_stop())
        await speak(cfg, "two")
        await until(lambda: len(spawns(cfg)) == 2)
        return await status(cfg)

    assert run_playing_daemon(cfg, scenario)["paused"] is False


def test_the_daemon_shuts_down_while_a_line_is_paused(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        await speak(cfg, "one", "two")
        await wait_speaking(cfg)
        await request(cfg, protocol.encode_pause())

    began = time.monotonic()
    run_playing_daemon(cfg, scenario)
    assert time.monotonic() - began < 5
