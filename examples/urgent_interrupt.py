#!/usr/bin/env python3
"""Demonstrate the interrupt protocol.

Queue several chatty status lines, then fire an urgent alert: the daemon
drains everything still queued, cuts off the line being spoken mid-word,
and speaks the alert immediately.

Run:  python examples/urgent_interrupt.py
"""
import time

from speakd import speak


def main() -> None:
    # Flood the queue with routine narration.
    for i in range(1, 6):
        speak(f"processing batch {i} of 5, all metrics nominal")

    time.sleep(3)  # let the first line or two start playing

    # Something went wrong — jump the queue.
    speak("alert: disk usage at ninety five percent, pausing writes",
          interrupt=True, blocking=True)


if __name__ == "__main__":
    main()
