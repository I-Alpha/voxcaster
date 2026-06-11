# speakd

**Fire-and-forget local TTS narration for long-running work.**

`speakd` is a small Unix daemon that turns text lines into speech with
[Kokoro](https://github.com/hexgrad/kokoro) (a fast, high-quality local TTS
model). Any process — a training run, a build, a cron job, a shell one-liner —
sends a line to a Unix socket and moves on; the daemon queues, synthesizes,
and plays it. If anything in the audio stack fails, the line degrades to
espeak instead of disappearing.

It was built to narrate machine-learning training runs on a single-GPU
workstation, which shaped its defining feature: **the TTS model dynamically
offloads itself from the GPU** when narration goes quiet, so it never holds
VRAM hostage from the workload it is narrating.

```
$ pip install .
$ speak "training started"          # daemon auto-spawns on first use
$ speak --interrupt "loss is NaN"   # cuts off whatever is playing, speaks NOW
$ make 2>&1 | tail -1 | speak       # pipe-friendly
```

## Why a daemon?

Calling a TTS library inline is the obvious approach and the wrong one for
narration: it blocks the caller for seconds per line, loads a model per
process, and overlapping lines talk over each other. `speakd` inverts this:

- **~1 ms per call.** The client writes one line to a Unix socket and returns.
  Narration can sit inside hot loops and signal handlers.
- **One model, one queue.** A single daemon owns the model and serialises
  playback. Ten processes can narrate concurrently without crosstalk.
- **Failure-proof by design.** Daemon down? The client spawns it. Spawn fails?
  espeak fallback. No audio at all? The caller still never raises.

## Architecture

```
 any process, any language                 speakd daemon (one per socket, flock-enforced)
┌──────────────────────┐            ┌───────────────────────────────────────────────┐
│  speak "epoch done"  │──┐         │   asyncio Unix-socket server                  │
└──────────────────────┘  │         │        │                                      │
┌──────────────────────┐  │  UTF-8  │        ├── volume msg ──▶ live volume         │
│  Python: speak(...)  │──┼─ line ─▶│        ├── interrupt ───▶ drain queue +       │
└──────────────────────┘  │  over   │        │                  kill playback       │
┌──────────────────────┐  │  socket │        ▼                                      │
│  CI job, cron, hook  │──┘         │   FIFO queue ──▶ worker (thread executor)     │
└──────────────────────┘            │                     │                         │
          ▲                         │                     ▼                         │
          │ "OK\n" ack              │   Kokoro TTS ──▶ wav ──▶ mpv ──▶ 🔊           │
          │ (blocking mode only)    │   CPU ⇄ GPU                                   │
          └─────────────────────────│   (offloads after idle keepalive)             │
                                    │                                               │
                                    │   any failure ──▶ espeak fallback             │
                                    └───────────────────────────────────────────────┘
```

## Features

- **Fire-and-forget socket design** — newline-terminated UTF-8 over a Unix
  domain socket; trivially scriptable from any language. Optional `OK` ack
  for blocking callers.
- **Dynamic GPU offload with keepalive** — the model loads on CPU, hops onto
  the GPU for narration bursts, and releases its VRAM (~3 GB) after a
  configurable idle period. If the GPU is full (another job grabbed it), that
  request simply synthesizes on CPU instead of failing.
- **Interrupt protocol** — an urgent line drains the pending queue, kills
  in-flight playback mid-word, and speaks immediately.
- **Live volume control** — one socket message, applies from the next line;
  no restart.
- **Singleton via `flock(2)`** — clients can race to auto-spawn the daemon;
  exactly one wins, the rest exit cleanly. Stale sockets are detected and
  removed on startup.
- **Graceful fallback** — Kokoro import error, synthesis failure, playback
  failure, or daemon unreachable: the line is spoken by espeak and the event
  is logged. Narration degrades; it never silently vanishes.
- **One TOML file, env-var overrides, zero-config defaults** — works out of
  the box on CPU with no config file at all.

## Requirements

- Linux or macOS (Unix sockets + `flock`), Python ≥ 3.10
- [mpv](https://mpv.io/) for playback (`apt install mpv`) — or any player,
  via config
- [espeak](https://espeak.sourceforge.net/) for the fallback voice
  (`apt install espeak`) — optional but recommended
- A CUDA-capable GPU is **optional**; everything works on CPU

## Install

```bash
git clone <this-repo> && cd speakd
pip install .
```

This installs the `kokoro` TTS package (which pulls in PyTorch) and two
console commands: `speakd` (the daemon) and `speak` (the client).

## Quickstart

```bash
# 1. Just speak — the daemon auto-spawns on first use:
speak "hello from speakd"

# 2. Or run the daemon in the foreground to watch it work:
speakd --device cpu --voice af_heart

# 3. Script it:
speak --blocking "waits until this has been spoken"
speak --interrupt "queue drained, this plays immediately"
speak --volume 60 "quieter from now on"
echo "pipes work too" | speak
```

From Python:

```python
from speakd import speak, set_volume

speak("checkpoint saved")                        # ~1 ms, non-blocking
speak("eval finished", blocking=True)            # wait until spoken
speak("loss is NaN — stopping", interrupt=True)  # jump the queue
set_volume(85)
```

See [`examples/`](examples/) for runnable demos of narration, interrupts,
and volume control.

## Configuration

Defaults work with no config at all. To customise, copy
[`config.example.toml`](config.example.toml) to `~/.config/speakd/config.toml`
(or point `$SPEAKD_CONFIG` at any path). Environment variables override the
file; CLI flags override both.

| TOML key | Env override | Default | Meaning |
|---|---|---|---|
| `tts.voice` | `SPEAKD_VOICE` | `af_heart` | Kokoro voice id (`af_*`, `am_*`, `bf_*`, `bm_*`, ...) |
| `tts.speed` | `SPEAKD_SPEED` | `1.0` | Speech-rate multiplier |
| `tts.lang_code` | `SPEAKD_LANG` | `a` | Kokoro language code (`a` US English, `b` UK English) |
| `device.policy` | `SPEAKD_DEVICE` | `auto` | `auto` (dynamic offload) / `cpu` / `gpu` |
| `device.keepalive_seconds` | `SPEAKD_KEEPALIVE` | `180` | Idle seconds before GPU→CPU offload |
| `daemon.socket_path` | `SPEAKD_SOCKET` | `$XDG_RUNTIME_DIR/speakd.sock` | Unix socket path |
| `daemon.socket_mode` | — | `"600"` | Octal permissions on the socket file |
| `daemon.log_file` | `SPEAKD_LOG_FILE` | `~/.local/state/speakd/daemon.log` | Log target for auto-spawned daemons |
| `audio.volume` | `SPEAKD_VOLUME` | `100` | Playback volume `0–130` (mpv scale) |
| `audio.max_playback_seconds` | — | `120` | Kill a single line's playback after this |
| `audio.player` | — | mpv template | Player argv; `{file}` and `{volume}` are substituted |
| `fallback.command` | — | espeak template | Fallback argv; `{text}` is substituted; `[]` disables |
| `client.connect_timeout` | — | `0.5` | Socket connect/send timeout (s) |
| `client.ack_timeout` | — | `300.0` | `--blocking` wait for the spoken-ack (s) |
| `client.spawn_wait` | — | `4.0` | Wait for an auto-spawned daemon (s) |

`speakd --print-config` shows the fully-resolved effective configuration.

## Wire protocol

One newline-terminated UTF-8 line per connection — easy to speak from any
language without a client library:

| Message | Bytes | Effect |
|---|---|---|
| Speak | `<text>\n` | Queue the line; daemon replies `OK\n` when spoken |
| Interrupt | `\x01INTERRUPT\x01<text>\n` | Drain queue, kill playback, speak now |
| Volume | `\x02VOLUME\x02<int>\n` | Set live volume (0–130) |

```bash
# speak from raw shell, no client needed:
printf 'hello from netcat\n' | nc -U "$XDG_RUNTIME_DIR/speakd.sock"
```

The control markers are ASCII SOH/STX characters that cannot occur in normal
text, so no escaping is ever needed.

## GPU offload in detail

The `auto` policy exists for machines where the GPU has a day job:

1. The model loads on **CPU** at first request.
2. Each synthesis tries to move it to the **GPU** first (a few hundred ms,
   then synthesis is much faster). If CUDA is busy or OOM, that line
   synthesizes on CPU — no error, just slower.
3. After `keepalive_seconds` (default 180 s) without a request, an idle timer
   moves the model back to **CPU** and calls `torch.cuda.empty_cache()`,
   releasing the VRAM.

The effect: during an active narration burst the voice is snappy and
GPU-accelerated; ten minutes into a silent stretch, your training job has its
VRAM back. All device moves are serialised with synthesis under one lock, so
the model can never be moved mid-utterance.

## Troubleshooting

| Symptom | Likely cause / fix |
|---|---|
| `speak` says *fallback engine used* | Daemon failed to start — check `~/.local/state/speakd/daemon.log`. Most common: `kokoro` not installed in the Python that spawned it (set `SPEAKD_DAEMON_CMD="/path/to/python -m speakd.daemon"`). |
| No audio, no errors | Is `mpv` installed and does it play a wav from your terminal? Swap `audio.player` if you use a different player. |
| First line is slow | Cold start: model weights load on first request (a few seconds). Subsequent lines are fast. |
| Robotic voice instead of Kokoro | That *is* the espeak fallback working as designed — see the first row. |
| Two daemons after a crash | They cannot coexist: the flock singleton makes the second exit immediately, and stale sockets are cleaned on startup. Delete `<socket>.lock` only if a machine crash left it owned by a dead PID holder (flock releases on process death, so this is near-impossible). |
| `daemon already running (pid N)` | Working as intended — the running daemon serves all clients. |
| GPU memory not released | The model offloads after `device.keepalive_seconds` of *no requests*; lower it, or run with `--device cpu`. |

## License

[MIT](LICENSE) © 2026 ibrahim Alfa
