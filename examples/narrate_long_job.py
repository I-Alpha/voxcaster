#!/usr/bin/env python3
"""Narrate a long-running job without ever blocking it.

Each `speak()` call costs ~1 ms (one Unix-socket write); the daemon queues
and serialises the audio. If the daemon is down it is spawned once, and if
that fails the line falls back to espeak — the job itself never crashes
because of narration.

Run:  python examples/narrate_long_job.py
"""
import time

from speakd import speak

STAGES = [
    ("loading dataset", 2),
    ("training epoch one", 3),
    ("training epoch two", 3),
    ("running evaluation", 2),
]


def main() -> None:
    speak("pipeline starting")
    for stage, seconds in STAGES:
        speak(stage)               # fire-and-forget: returns immediately
        time.sleep(seconds)        # ... the real work happens here ...
    speak("pipeline finished, all stages green", blocking=True)  # wait for the last line


if __name__ == "__main__":
    main()
