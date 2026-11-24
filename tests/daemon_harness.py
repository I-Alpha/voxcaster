import asyncio
import glob
import inspect
import json
import os
import sys

from fakes import FakeKokoro, RecordingDucker

from voxcaster import protocol
from voxcaster.config import Config
from voxcaster.daemon import VoiceDaemon
from voxcaster.engine import KokoroEngine

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
        fallback=[],
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


async def request_no_reply(cfg, payload):
    reader, writer = await asyncio.open_unix_connection(cfg.socket_path)
    writer.write(payload)
    await writer.drain()
    writer.close()


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
    for path in glob.glob(os.path.join(os.path.dirname(cfg.socket_path), "player.*.log")):
        records += [json.loads(line) for line in open(path)]
    return records


def commands(cfg):
    return [r["command"] for r in player_records(cfg) if "command" in r]


def spawns(cfg):
    return [r for r in player_records(cfg) if "spawn" in r]


def volume_commands(cfg):
    return [c[2] for c in commands(cfg) if c[:2] == ["set_property", "volume"]]


def run_playing_daemon(cfg, scenario, kokoro=None):
    kokoro = kokoro or FakeKokoro()

    async def wrapped(daemon):
        daemon.engine = KokoroEngine(voice=cfg.voice, policy="cpu", pipeline_factory=kokoro)
        daemon._ducker = RecordingDucker()
        return await scenario(daemon)

    return run_daemon(cfg, wrapped)


async def wait_speaking(cfg):
    async def speaking():
        return (await status(cfg))["speaking"]

    await until(speaking)
