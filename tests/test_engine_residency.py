import threading

import numpy as np
import pytest

from fakes import FakeKokoro

from voxcaster import engine as engine_module
from voxcaster.engine import KokoroEngine

IDLE_UNLOAD = 600.0
AMERICAN, BRITISH, SPANISH = "af_heart", "bf_emma", "ef_dora"


class Clock:
    def __init__(self):
        self.now = 1000.0

    def __call__(self):
        return self.now


class FakePipeline:
    model = None

    def __init__(self, factory):
        self.factory = factory

    def __call__(self, text, voice, speed):
        if self.factory.on_call:
            self.factory.on_call()
        yield type("Result", (), {"audio": np.zeros(240, dtype=np.float32), "sample_rate": 24000})()


class Factory:
    def __init__(self):
        self.built = []
        self.on_call = None

    def __call__(self, lang):
        self.built.append(lang)
        return FakePipeline(self)


@pytest.fixture
def clock():
    return Clock()


@pytest.fixture
def factory():
    return Factory()


@pytest.fixture
def engine(clock, factory):
    return KokoroEngine(
        voice=AMERICAN, policy="cpu", max_resident_pipelines=2,
        idle_unload_seconds=IDLE_UNLOAD, pipeline_factory=factory, clock=clock,
    )


def test_a_voice_selects_the_pipeline_of_its_own_language(engine, factory):
    engine.synthesize("hi", voice=SPANISH)
    assert factory.built == ["e"]


def test_only_the_two_most_recently_used_pipelines_stay_resident(engine):
    for voice in (AMERICAN, BRITISH, AMERICAN, SPANISH):
        engine.synthesize("hi", voice=voice)
    assert sorted(engine.resident_langs()) == ["a", "e"]


def test_every_pipeline_is_unloaded_after_the_idle_period_and_reloads_on_use(engine, clock, factory):
    engine.synthesize("hi", voice=AMERICAN)
    engine.synthesize("hi", voice=BRITISH)
    clock.now += IDLE_UNLOAD
    assert sorted(engine.unload_idle()) == ["a", "b"]
    assert engine.resident_langs() == []
    assert not engine.is_ready()
    engine.synthesize("hi", voice=AMERICAN)
    assert engine.resident_langs() == ["a"]
    assert factory.built == ["a", "b", "a"]


def test_only_the_pipelines_idle_past_the_period_are_unloaded(engine, clock):
    engine.synthesize("hi", voice=AMERICAN)
    clock.now += IDLE_UNLOAD / 2
    engine.synthesize("hi", voice=BRITISH)
    clock.now += IDLE_UNLOAD / 2 + 1
    assert engine.unload_idle() == ["a"]
    assert engine.resident_langs() == ["b"]


def test_a_pipeline_is_never_unloaded_while_it_renders(engine, clock, factory):
    engine.synthesize("hi", voice=AMERICAN)
    clock.now += 10 * IDLE_UNLOAD
    sweeper = threading.Thread(target=engine.unload_idle)
    blocked_mid_render = []

    def sweep_mid_render():
        sweeper.start()
        sweeper.join(timeout=0.3)
        blocked_mid_render.append(sweeper.is_alive())

    factory.on_call = sweep_mid_render
    engine.synthesize("hi", voice=AMERICAN)
    sweeper.join()
    assert blocked_mid_render == [True]
    assert engine.resident_langs() == ["a"]


def test_evicting_a_pipeline_returns_its_memory_to_the_os(engine, monkeypatch):
    released = []
    monkeypatch.setattr(engine_module, "release_freed_memory", lambda: released.append(1))
    for voice in (AMERICAN, BRITISH, SPANISH):
        engine.synthesize("hi", voice=voice)
    assert released


def test_synthesize_returns_every_piece_untrimmed_and_in_order(clock):
    kokoro = FakeKokoro(audio_sec=0.1, lead_sec=0.2, tail_sec=0.2)
    engine = KokoroEngine(voice=AMERICAN, policy="cpu", pipeline_factory=kokoro, clock=clock)
    wav, rate = engine.synthesize("one\ntwo\nthree")
    assert rate == 24000
    assert np.array_equal(np.rint(wav * 32767).astype(np.int16), kokoro.received())


def test_an_open_stream_keeps_its_pipeline_in_place_until_it_is_closed(engine, clock):
    engine.synthesize("hi")
    clock.now += 10 * IDLE_UNLOAD
    stream = engine.stream(engine.plan_chunks("one\ntwo"))
    next(stream)
    sweeper = threading.Thread(target=engine.unload_idle)
    sweeper.start()
    sweeper.join(timeout=0.3)
    assert sweeper.is_alive()
    stream.close()
    sweeper.join(timeout=3)
    assert not sweeper.is_alive()
