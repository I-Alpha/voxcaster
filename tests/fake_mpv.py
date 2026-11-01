"""Stand-in for mpv that needs no audio hardware.

Serves mpv's JSON IPC on the socket named by ``--input-ipc-server=`` and appends
every received command, preceded by one ``spawn`` record, to ``<socket>.log``.
It "plays" for FAKE_PLAYER_PLAY_SEC seconds of unpaused time, then exits 0.
FAKE_PLAYER_BIND_DELAY delays creating the socket.
"""
import json
import os
import socket
import sys
import time

PLAY_SEC = float(os.environ.get("FAKE_PLAYER_PLAY_SEC", "10"))
BIND_DELAY_SEC = float(os.environ.get("FAKE_PLAYER_BIND_DELAY", "0"))
TICK_SEC = 0.02


def main(argv):
    path = next(a.split("=", 1)[1] for a in argv if a.startswith("--input-ipc-server="))
    log = open(path + ".log", "a", buffering=1)
    log.write(json.dumps({"spawn": argv[1:]}) + "\n")
    time.sleep(BIND_DELAY_SEC)
    server = socket.socket(socket.AF_UNIX)
    server.bind(path)
    server.listen(4)
    server.settimeout(TICK_SEC)
    paused = False
    remaining = PLAY_SEC
    last = time.monotonic()
    while remaining > 0:
        now = time.monotonic()
        if not paused:
            remaining -= now - last
        last = now
        try:
            conn, _ = server.accept()
        except socket.timeout:
            continue
        conn.settimeout(1.0)
        with conn:
            request = json.loads(conn.makefile().readline())
            log.write(json.dumps({"command": request["command"]}) + "\n")
            command = request["command"]
            if command[:2] == ["set_property", "pause"]:
                paused = bool(command[2])
            if command[0] == "quit":
                break
            conn.sendall(b'{"event":"property-change"}\n')
            conn.sendall((json.dumps({"error": "success", "request_id": request.get("request_id", 0)}) + "\n").encode())
    server.close()
    os.unlink(path)


if __name__ == "__main__":
    main(sys.argv)
