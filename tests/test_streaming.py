import asyncio
import glob
import json
import os
import sys
import threading

import numpy as np
import pytest
from daemon_harness import (FAKE_PLAYER, make_cfg, request, request_no_reply,
                            run_playing_daemon, speak, status, until)
from fakes import SAMPLE_RATE, FakeKokoro

from voxcaster import protocol, streaming
from voxcaster.streaming import Progress, ready_to_start


@pytest.fixture(autouse=True)
def realtime_player(monkeypatch):
    monkeypatch.setenv("FAKE_PLAYER_REALTIME", "1")


class Played:
    def __init__(self, records, pcm):
        self.records = records
        self.pcm = pcm

    def t(self, key):
        return next((r["t"] for r in self.records if key in r), None)

    def count(self, key):
        return sum(1 for r in self.records if key in r)


def played(cfg):
    out = []
    for log in glob.glob(os.path.join(os.path.dirname(cfg.socket_path), "player.*.log")):
        records = [json.loads(line) for line in open(log)]
        pcm = np.frombuffer(open(log[:-len(".log")] + ".pcm", "rb").read(), dtype="<i2")
        out.append(Played(records, pcm))
    return sorted(out, key=lambda p: p.t("spawn"))


def lines(n, tag="Paragraph"):
    return "\n".join(f"{tag} {i} of the line." for i in range(n))


def sentences(n):
    return " ".join(f"Sentence {i} " + "word " * 27 + "ends." for i in range(n))


def stream_dir_files(cfg):
    return os.listdir(cfg.stream_dir) if os.path.isdir(cfg.stream_dir) else []


def synth_threads():
    return [t for t in threading.enumerate() if t.name.startswith("voxcaster-")]


def speak_and_wait(cfg, text, kokoro):
    async def scenario(daemon):
        await request(cfg, protocol.encode_speak(text), timeout=30)

    run_playing_daemon(cfg, scenario, kokoro)
    return played(cfg)


def test_first_audio_reaches_the_player_before_the_last_piece_is_rendered(tmp_path):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=0.15, audio_sec=0.3)
    (player,) = speak_and_wait(cfg, lines(6), kokoro)
    assert len(kokoro.pieces) == 6
    assert player.t("first_byte") < kokoro.pieces[-1][3]


def test_the_player_receives_every_piece_in_order_sample_for_sample(tmp_path):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=0.05, audio_sec=0.2)
    (player,) = speak_and_wait(cfg, lines(6), kokoro)
    assert np.array_equal(player.pcm, kokoro.received())


def test_a_player_without_ipc_receives_the_same_stream(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.player = [sys.executable, FAKE_PLAYER, "{file}"]
    kokoro = FakeKokoro(delay=0.05, audio_sec=0.2)
    (player,) = speak_and_wait(cfg, lines(4), kokoro)
    assert np.array_equal(player.pcm, kokoro.received())


def test_joins_inside_a_paragraph_are_trimmed_to_one_pause_and_paragraph_breaks_are_kept(tmp_path):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(audio_sec=0.2, lead_sec=0.4, tail_sec=0.4)
    (player,) = speak_and_wait(cfg, sentences(6) + "\nA new paragraph.", kokoro)
    assert len(kokoro.pieces) == 4
    speech = np.flatnonzero(player.pcm)
    gaps = np.diff(speech)[np.diff(speech) > 1] - 1
    assert len(gaps) == 3
    assert np.array_equal(player.pcm[speech],
                          np.concatenate([kokoro.speech(p[0]) for p in kokoro.pieces]))
    inside = gaps[:2] / SAMPLE_RATE
    assert ((0.1 <= inside) & (inside <= 0.4)).all()
    assert gaps[2] == int(0.8 * SAMPLE_RATE)


def test_slow_synthesis_delays_the_start_instead_of_running_dry(tmp_path):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=0.3, audio_sec=0.2)
    (player,) = speak_and_wait(cfg, lines(6), kokoro)
    assert kokoro.pieces[2][3] <= player.t("first_byte") < kokoro.pieces[-1][3]
    assert player.count("underrun") == 0
    assert np.array_equal(player.pcm, kokoro.received())


def test_a_stall_after_the_start_waits_at_a_piece_boundary_and_loses_nothing(tmp_path):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=lambda n: 1.0 if n == 3 else 0.02, audio_sec=0.2)
    (player,) = speak_and_wait(cfg, lines(6), kokoro)
    assert player.count("underrun") >= 1
    assert np.array_equal(player.pcm, kokoro.received())


@pytest.mark.parametrize("cut", ["skip", "stop", "interrupt"])
def test_ending_a_line_mid_stream_stops_synthesis_and_leaves_nothing_behind(tmp_path, cut):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=0.2, audio_sec=0.5)
    message = {"skip": protocol.encode_skip(), "stop": protocol.encode_stop(),
               "interrupt": protocol.encode_speak("urgent", interrupt=True)}[cut]

    async def scenario(daemon):
        await speak(cfg, lines(20))
        await until(lambda: played(cfg) and played(cfg)[0].t("first_byte"))
        rendered = len(kokoro.pieces)
        assert rendered < 20
        await request_no_reply(cfg, message)
        await asyncio.sleep(1.0)
        assert len(kokoro.pieces) <= rendered + 2 + (1 if cut == "interrupt" else 0)
        await until(lambda: not stream_dir_files(cfg) and not synth_threads())
        unloader = threading.Thread(target=daemon.engine.unload_idle)
        unloader.start()
        unloader.join(timeout=3)
        assert not unloader.is_alive()

    run_playing_daemon(cfg, scenario, kokoro)


def test_the_next_line_is_rendered_while_the_current_one_plays_and_in_its_own_voice(tmp_path):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=lambda n: 0.05 if n < 6 else 0.12, audio_sec=0.2)

    async def scenario(daemon):
        await request_no_reply(cfg, protocol.encode_speak(lines(6, "First")))
        await request_no_reply(cfg, protocol.encode_speak(lines(6, "Second"), voice="bf_emma"))
        await until(lambda: len(played(cfg)) == 2 and played(cfg)[1].t("exit"), timeout=10)

    run_playing_daemon(cfg, scenario, kokoro)
    first, second = played(cfg)
    assert kokoro.pieces[-1][3] < first.t("exit")
    assert second.t("first_byte") - first.t("exit") < 0.5
    assert kokoro.voices() == [cfg.voice] * 6 + ["bf_emma"] * 6
    assert np.array_equal(second.pcm, kokoro.received()[len(first.pcm):])


@pytest.mark.parametrize("how", ["drop", "clear"])
def test_dropping_or_clearing_a_line_being_prepared_discards_it(tmp_path, how):
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=lambda n: 0.02 if n < 4 else 0.3, audio_sec=0.5)

    async def scenario(daemon):
        await speak(cfg, lines(4, "First"), lines(20, "Second"))
        await until(lambda: len(kokoro.pieces) > 4)
        queued = (await status(cfg))["queue"]
        assert [q["text"] for q in queued][:1] == [lines(20, "Second")[:60]]
        message = (protocol.encode_drop(queued[0]["id"]) if how == "drop"
                   else protocol.encode_clear())
        assert await request(cfg, message) == protocol.ACK
        rendered = len(kokoro.pieces)
        await asyncio.sleep(1.0)
        assert len(kokoro.pieces) <= rendered + 1
        await speak(cfg, "Third.")
        await until(lambda: len(played(cfg)) == 2, timeout=10)

    run_playing_daemon(cfg, scenario, kokoro)
    assert [p[0] for p in kokoro.pieces if p[0].startswith("Second")] != []
    assert np.array_equal(played(cfg)[0].pcm, kokoro.received(4))


def test_a_pause_longer_than_the_buffer_stalls_synthesis_and_loses_nothing(tmp_path, monkeypatch):
    monkeypatch.setattr(streaming, "BUFFER_CAP_SEC", 1.0)
    cfg = make_cfg(tmp_path)
    kokoro = FakeKokoro(delay=0.01, audio_sec=0.5)

    async def scenario(daemon):
        await speak(cfg, lines(16))
        await until(lambda: played(cfg) and played(cfg)[0].t("first_byte"))
        await request(cfg, protocol.encode_pause())
        await asyncio.sleep(1.5)
        stalled = len(kokoro.pieces)
        assert stalled < 16
        await asyncio.sleep(0.5)
        assert len(kokoro.pieces) == stalled
        await request(cfg, protocol.encode_resume())
        await until(lambda: played(cfg)[0].t("exit"), timeout=15)

    run_playing_daemon(cfg, scenario, kokoro)
    assert np.array_equal(played(cfg)[0].pcm, kokoro.received())


def recorder(tmp_path):
    return [sys.executable, "-c", "import sys; open(sys.argv[1], 'a').write(sys.argv[2] + '|')",
            str(tmp_path / "spoken.txt"), "{text}"]


def spoken(tmp_path):
    path = tmp_path / "spoken.txt"
    return path.read_text() if path.exists() else ""


def test_a_failure_after_some_audio_speaks_only_the_rest_through_the_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.fallback = recorder(tmp_path)
    kokoro = FakeKokoro(delay=0.02, audio_sec=0.2, fail_at=3)

    async def scenario(daemon):
        await request(cfg, protocol.encode_speak(lines(5)), timeout=10)
        await until(lambda: spoken(tmp_path), timeout=3)

    run_playing_daemon(cfg, scenario, kokoro)
    players = played(cfg)
    assert len(players) == 1
    assert np.array_equal(players[0].pcm, kokoro.received(3))
    assert spoken(tmp_path) == "Paragraph 3 of the line.\nParagraph 4 of the line.|"


def test_a_failure_before_any_audio_speaks_the_whole_line_through_the_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.fallback = recorder(tmp_path)
    kokoro = FakeKokoro(fail_at=0)

    async def scenario(daemon):
        await request(cfg, protocol.encode_speak(lines(3)), timeout=10)
        await until(lambda: spoken(tmp_path), timeout=3)

    run_playing_daemon(cfg, scenario, kokoro)
    assert played(cfg) == []
    assert spoken(tmp_path) == lines(3) + "|"


def test_a_player_that_cannot_start_leaves_no_file_behind_and_the_line_is_spoken_by_the_fallback(tmp_path):
    cfg = make_cfg(tmp_path)
    cfg.fallback = recorder(tmp_path)
    cfg.player = [str(tmp_path / "no-such-player"), "{file}"]

    async def scenario(daemon):
        await request(cfg, protocol.encode_speak(lines(3)), timeout=10)
        await until(lambda: spoken(tmp_path), timeout=3)
        await until(lambda: not synth_threads())

    run_playing_daemon(cfg, scenario, FakeKokoro(audio_sec=0.2))
    assert stream_dir_files(cfg) == []


def test_starting_the_daemon_empties_its_own_directory_and_nothing_else(tmp_path):
    cfg = make_cfg(tmp_path)
    os.makedirs(cfg.stream_dir)
    os.mkfifo(os.path.join(cfg.stream_dir, "stale-stream.wav"))
    open(os.path.join(cfg.stream_dir, "stale.sock"), "w").close()
    neighbour = tmp_path / f"{os.path.basename(cfg.stream_dir)}-other"
    neighbour.write_text("not ours")
    foreign = tmp_path / "voxcaster-stream-1.wav"
    foreign.write_text("not ours")

    async def scenario(daemon):
        return None

    run_playing_daemon(cfg, scenario)
    assert stream_dir_files(cfg) == []
    assert neighbour.exists() and foreign.exists()


def test_a_slow_previous_line_holds_back_the_start_of_a_line_that_has_rendered_little():
    early = Progress(pieces=1, audio_sec=1.0, compute_sec=0.1, chars_done=10, chars_total=100,
                     finished=False)
    assert ready_to_start(early, previous_ratio=None)
    assert not ready_to_start(early, previous_ratio=3.0)
