import asyncio
import json
import os
import subprocess
import sys
import time

from voxcaster import protocol
from voxcaster.config import Config
from voxcaster.daemon import VoiceDaemon


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


def test_stop_drains_the_queue_and_kills_the_player(tmp_path):
    cfg = make_cfg(tmp_path)

    async def scenario(daemon):
        def slow_speak(text, voice=None):
            time.sleep(0.6)

        daemon._speak_sync = slow_speak
        player = subprocess.Popen([sys.executable, "-c", "import time; time.sleep(60)"])
        daemon._current_player = player
        for word in ("one", "two", "three"):
            await request_no_reply(cfg, protocol.encode_speak(word))
        await asyncio.sleep(0.2)
        before = await status(cfg)
        reply = await request(cfg, protocol.encode_stop())
        after = await status(cfg)
        player.wait(timeout=5)
        return before, reply, after, player.returncode

    before, reply, after, rc = run_daemon(cfg, scenario)
    assert before["queued"] == 2
    assert reply == protocol.ACK
    assert after["queued"] == 0
    assert rc is not None
