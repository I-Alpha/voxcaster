"""Wire protocol shared by the speakd daemon and client.

The protocol is deliberately tiny: one newline-terminated UTF-8 line per
connection. Control messages are flagged with ASCII control characters
(SOH / STX) that can never appear in normal speech text, so no escaping
is needed and any language with a Unix-socket API can be a client.

    Speak:     ``<text>\\n``
    Interrupt: ``\\x01INTERRUPT\\x01<text>\\n``
               (drain the queue, cut off in-flight playback, speak now)
    Volume:    ``\\x02VOLUME\\x02<int>\\n``
               (set live playback volume, 0-130, mpv scale)

The daemon replies ``b"OK\\n"`` once the line has been spoken. Fire-and-forget
clients simply close the connection without reading the ack.
"""
from __future__ import annotations

INTERRUPT_MARKER = "\x01INTERRUPT\x01"  # SOH-flanked - never in UTF-8 speech text
VOLUME_MARKER = "\x02VOLUME\x02"        # STX-flanked - likewise
ACK = b"OK\n"

VOLUME_MIN = 0
VOLUME_MAX = 130  # mpv volume scale: 100 = nominal, >100 = software gain


def clamp_volume(level: int) -> int:
    """Clamp a volume level to the valid wire range."""
    return max(VOLUME_MIN, min(VOLUME_MAX, int(level)))


def encode_speak(text: str, interrupt: bool = False) -> bytes:
    """Encode a speak (or interrupt-and-speak) request."""
    return ((INTERRUPT_MARKER + text if interrupt else text) + "\n").encode()


def encode_volume(level: int) -> bytes:
    """Encode a live volume-change request."""
    return (VOLUME_MARKER + str(clamp_volume(level)) + "\n").encode()
