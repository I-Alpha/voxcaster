"""Wire protocol shared by the voxcaster daemon and client.

The protocol is deliberately tiny: one newline-terminated UTF-8 line per
connection. Control messages are flagged with ASCII control characters
(SOH / STX / ETX / EOT) that can never appear in normal speech text, so no
escaping is needed and any language with a Unix-socket API can be a client.

    Speak:     ``<text>\\n``
    Interrupt: ``\\x01INTERRUPT\\x01<text>\\n``
               (drain the queue, cut off in-flight playback, speak now)
    Volume:    ``\\x02VOLUME\\x02<int>\\n``
               (set live playback volume, 0-130, mpv scale)
    Voice:     ``\\x03VOICE\\x03<voice_id>\\x03<text>\\n``
               (speak ``text`` with a per-request voice override; the voice
               prefix may combine with the interrupt prefix, in either order)
    Ping:      ``\\x04PING\\x04\\n``
               (health check — daemon replies ``READY``/``STARTING``, no speech)

The daemon replies ``b"OK\\n"`` once a speak line has been spoken. Fire-and-forget
clients simply close the connection without reading the ack.
"""
from __future__ import annotations

INTERRUPT_MARKER = "\x01INTERRUPT\x01"  # SOH-flanked - never in UTF-8 speech text
VOLUME_MARKER = "\x02VOLUME\x02"        # STX-flanked - likewise
VOICE_MARKER = "\x03VOICE\x03"          # ETX-flanked per-request voice override
PING_MARKER = "\x04PING\x04"            # EOT-flanked health check
ACK = b"OK\n"
PING_READY = b"READY\n"                 # daemon reply: model loaded
PING_STARTING = b"STARTING\n"           # daemon reply: model not loaded yet

VOLUME_MIN = 0
VOLUME_MAX = 130  # mpv volume scale: 100 = nominal, >100 = software gain


def clamp_volume(level: int) -> int:
    """Clamp a volume level to the valid wire range."""
    return max(VOLUME_MIN, min(VOLUME_MAX, int(level)))


def encode_speak(text: str, interrupt: bool = False, voice: str | None = None) -> bytes:
    """Encode a speak request (optionally interrupt-and-speak, optional voice).

    The voice prefix is applied to the text payload, so it sits inside the
    interrupt prefix when both are present: the daemon strips interrupt first,
    then voice, then is left with the bare text.
    """
    payload = text
    if voice:
        payload = VOICE_MARKER + voice + "\x03" + payload
    if interrupt:
        payload = INTERRUPT_MARKER + payload
    return (payload + "\n").encode()


def encode_volume(level: int) -> bytes:
    """Encode a live volume-change request."""
    return (VOLUME_MARKER + str(clamp_volume(level)) + "\n").encode()


def encode_ping() -> bytes:
    """Encode a health-check request."""
    return (PING_MARKER + "\n").encode()
