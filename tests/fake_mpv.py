"""Stand-in for mpv that needs no audio hardware.

Serves mpv's JSON IPC on the socket named by ``--input-ipc-server=`` (when
given), reads the media file named by the last non-option argument to its end,
and writes ``<out>.<pid>.log`` and ``<out>.<pid>.pcm``, where ``<out>`` is
``$FAKE_PLAYER_OUT`` or the IPC socket path. The log holds a ``spawn`` record,
every received IPC command, ``first_byte``, ``eof``, ``underrun`` and ``exit``
records stamped with ``time.monotonic()``; the pcm file holds the audio after
the 44 byte WAV header.

While paused it stops reading, so a writer feeding a FIFO is held back as it
would be by a real player. With FAKE_PLAYER_REALTIME=1 it plays the audio it has
read in real time, logs an ``underrun`` whenever it runs out of audio, once it has had some, before the
end of the stream, and exits when everything read has been played. Otherwise it
exits after FAKE_PLAYER_PLAY_SEC seconds (default 10) of unpaused time.
FAKE_PLAYER_BIND_DELAY delays creating the IPC socket.
"""
import json
import os
import socket
import struct
import sys
import threading
import time

PLAY_SEC = float(os.environ.get("FAKE_PLAYER_PLAY_SEC", "10"))
BIND_DELAY_SEC = float(os.environ.get("FAKE_PLAYER_BIND_DELAY", "0"))
REALTIME = os.environ.get("FAKE_PLAYER_REALTIME") == "1"
TICK_SEC = 0.02
READ_BYTES = 4096
WAV_HEADER_BYTES = 44


class Player:
    def __init__(self, tag):
        self.log = open(tag + ".log", "a", buffering=1)
        self.pcm = open(tag + ".pcm", "wb", buffering=0)
        self.running = threading.Event()
        self.running.set()
        self.lock = threading.Lock()
        self.fed_bytes = 0
        self.bytes_per_sec = 48000.0
        self.eof = False

    def record(self, **fields):
        fields["t"] = time.monotonic()
        with self.lock:
            self.log.write(json.dumps(fields) + "\n")

    def read_media(self, media):
        with open(media, "rb") as f:
            header = b""
            while len(header) < WAV_HEADER_BYTES:
                part = f.read(WAV_HEADER_BYTES - len(header))
                if not part:
                    break
                header += part
            if len(header) == WAV_HEADER_BYTES:
                rate, = struct.unpack_from("<I", header, 24)
                channels, = struct.unpack_from("<H", header, 22)
                bits, = struct.unpack_from("<H", header, 34)
                self.bytes_per_sec = float(rate * channels * bits // 8)
            while True:
                self.running.wait()
                chunk = f.read(READ_BYTES)
                if not chunk:
                    break
                if self.fed_bytes == 0:
                    self.record(first_byte=True)
                self.pcm.write(chunk)
                self.fed_bytes += len(chunk)
        self.eof = True
        self.record(eof=True)


def main(argv):
    ipc = next((a.split("=", 1)[1] for a in argv if a.startswith("--input-ipc-server=")), None)
    media = [a for a in argv[1:] if not a.startswith("--")][-1]
    out = os.environ.get("FAKE_PLAYER_OUT") or ipc
    player = Player(f"{out}.{os.getpid()}")
    player.record(spawn=argv[1:])
    threading.Thread(target=player.read_media, args=(media,), daemon=True).start()
    time.sleep(BIND_DELAY_SEC)
    server = None
    if ipc is not None:
        server = socket.socket(socket.AF_UNIX)
        server.bind(ipc)
        server.listen(4)
        server.settimeout(TICK_SEC)
    paused = False
    remaining = PLAY_SEC
    played = 0.0
    starved = False
    last = time.monotonic()
    while True:
        now = time.monotonic()
        step, last = now - last, now
        if not paused:
            remaining -= step
            if REALTIME:
                available = player.fed_bytes / player.bytes_per_sec - played
                played += min(step, max(available, 0.0))
                if available <= 0 and not player.eof and player.fed_bytes:
                    if not starved:
                        player.record(underrun=True)
                    starved = True
                else:
                    starved = False
        if REALTIME:
            if player.eof and played >= player.fed_bytes / player.bytes_per_sec:
                break
        elif remaining <= 0:
            break
        if server is None:
            time.sleep(TICK_SEC)
            continue
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        conn.settimeout(1.0)
        with conn:
            request = json.loads(conn.makefile().readline())
            command = request["command"]
            player.record(command=command)
            if command[:2] == ["set_property", "pause"]:
                paused = bool(command[2])
                if paused:
                    player.running.clear()
                else:
                    player.running.set()
            if command[0] == "quit":
                break
            conn.sendall(b'{"event":"property-change"}\n')
            conn.sendall((json.dumps({"error": "success", "request_id": request.get("request_id", 0)}) + "\n").encode())
    player.record(exit=True)
    if server is not None:
        server.close()
        os.unlink(ipc)


if __name__ == "__main__":
    main(sys.argv)
