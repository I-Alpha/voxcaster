# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.4.0] - 2026-09-09

### Added
- **Render to file (RENDER wire message)**: `speak --output out.wav "text"`
  (or the `\x05RENDER\x05` protocol message) synthesizes through the same
  warm daemon and writes a mono wav to the requested path instead of playing
  it. No per-call model load, no speaker required — the intended consumer is
  voice-note / TTS integration (e.g. OpenClaw's `tts-local-cli` provider)
  where the destination is a file, not a speaker. Honours the existing
  `\x03VOICE\x03` voice and `\x01INTERRUPT\x01` prefixes with the same
  nesting order as Speak, so one parser serves both message kinds. No espeak
  fallback for renders (a file cannot be spoken): failures are logged and the
  request is simply not acked.
- New client API: `voxcaster.render(text, output_path, voice=None)` and the
  `render`/`--output` CLI surface.

### Fixed
- **Multi-paragraph payloads were cut at the first newline** (a4fefe7,
  ed21fe7). The wire protocol is line-oriented and the client did not escape
  newlines, so everything after the first newline was dropped after the ACK
  had been sent and the client had exited 0. Measured: a 168-byte
  three-paragraph payload arrived as 65 bytes. Newlines now travel as RS
  (`\x1e`) and the daemon restores them before synthesis. Byte-exact arrival
  was checked on a 200KB payload; `max_message_bytes` (16 MiB, already the
  StreamReader limit) is now an explicit config field.
- **Long clips were cut off mid-sentence** (ed21fe7). `max_playback_seconds`
  killed the player at a fixed 120s. It is now a hang-protection floor: the
  effective timeout is `max(floor, clip_duration + 30s)`.
- **Sentence chunking** (2c57292, 4126229). Paragraphs longer than the chunk
  budget are packed into sentence-aligned chunks of at most 320 characters,
  and a single over-long sentence is left intact. The original commit
  described this as a fix for Kokoro truncating long text; a controlled
  measurement (2531 chars, 139.2s without chunking, 147.4s with) showed
  Kokoro was never truncating, so chunking is retained only for
  sentence-aligned boundaries, at a measured cost of about 6% extra duration
  from inter-chunk silence. Chunking no longer splits mid-sentence on
  abbreviations and numbers ("i.e. ", "vs. ", "4.016; "): a boundary now
  requires the next character to look like a sentence start. This is a
  heuristic, not a tokenizer: a sentence opening with a lowercase proper noun
  (e.g. "iOS 18") is not split.
- **Voice and accent changed between consecutive lines** (a4fefe7). The
  Kokoro pipeline was rebuilt on every cross-family voice swap, which also
  reset device state. Pipelines are now cached per `lang_code` (bounded at 2)
  and idle offload moves all of them; 8 alternating requests build 2
  pipelines instead of 8. A `[voice-contention]` warning is logged when a
  request's voice differs from the configured default.
- README config/env table still used the pre-rename `SPEAKD_*` variable
  names; corrected to `VOXCASTER_*` (matches `config.py`'s env overrides).

## [0.3.0] - 2026-07-16

### Added
- **systemd watchdog integration (`Type=notify`)**: the daemon now speaks
  sd_notify(3) natively — stdlib-only, no `python-systemd` dependency.
  `READY=1` is sent only once the Unix socket is actually listening (so
  systemd never reports "active" for a daemon that can't accept a
  connection yet), and a `WATCHDOG=1` heartbeat is pinged from *inside* the
  asyncio event loop at half the `WatchdogSec` budget. If the loop ever
  dies or wedges while the process stays alive — previously invisible to
  `Restart=` supervision, which only tracks PID liveness — the heartbeat
  stops and systemd force-restarts the unit. Guards the `WATCHDOG_PID`
  mismatch case (pings from a PID systemd didn't arm are silently ignored
  upstream). No-op outside systemd. Verified live: `SIGSTOP` on the daemon
  → watchdog timeout at 30s → automatic restart → socket answering again.
- `packaging/systemd/voxcaster.service.example` updated to match:
  `Type=notify`, `WatchdogSec=30`, `Restart=always`, and a note on serving
  the package-default socket plus a compat symlink for tooling that
  hardcodes a legacy socket path.

### Fixed
- **Half-dead zombie daemons**: an unhandled crash of the event loop could
  leave the process alive indefinitely — non-daemon executor threads
  (torch/CUDA synthesis work) outlive the dead loop, so the interpreter
  never exits: the socket refuses connections while `ps`/systemd still see
  a live process holding the singleton flock. Reproduced live (a spawned
  daemon crashed mid-synthesis on a pip PEP 668 error and lingered exactly
  like this). The daemon now logs the full traceback and hard-exits
  (`os._exit`), so the process dies for real and supervision restarts it.
- **Silent worker death**: asyncio swallows task exceptions by default — a
  worker task killed by anything its per-line guard doesn't catch left the
  listener happily queueing lines that would never be spoken. A
  done-callback on the worker task now converts that into a clean daemon
  exit (and therefore a supervised restart).
- **Orphan daemon leak on client auto-spawn**: `ensure_daemon()` now
  prefers `systemctl --user restart voxcaster.service` when a user unit
  exists — the respawned daemon stays tracked (journal, `Restart=` policy,
  watchdog) and runs under the unit's own interpreter/env, instead of
  leaking an untracked orphan bound to whatever python the client happened
  to be running under. The raw detached spawn remains as the last resort
  when no unit exists or the unit serves a different socket path.

## [0.2.1] - 2026-07-07

### Fixed
- **Chronic post-spawn death**: the daemon had no handler for `SIGHUP` and
  its default disposition (immediate termination) killed it silently —
  no exception, no log line — any time its launching shell/session hung up
  (e.g. a bare `daemon &` spawn from a script or ephemeral terminal). This
  produced weeks of unexplained `FALLBACK reason=daemon down after spawn
  attempt` entries in client logs with nothing in `daemon.log` to explain
  them. `SIGHUP` is now ignored (with a log line), matching standard
  daemon/nohup semantics — reproduced live via `kill -HUP <pid>` before and
  after the fix.
- Any other uncaught exception during daemon startup/run is now logged
  with a full traceback before the process exits (previously relied on
  whatever redirected the process's stderr, which was not always captured).
- Completed the `speakd` → `voxcaster` rename at the git level: the
  actually-installed/running package (`src/voxcaster/`) had never been
  committed, while the superseded `src/speakd/` copy was still tracked —
  a fresh clone of this repo would not have produced a working daemon.

### Added
- `packaging/systemd/voxcaster.service.example`: an optional `systemd --user`
  unit with `Restart=on-failure` for supervised auto-restart — a second line
  of defense against any death the daemon itself cannot prevent (OOM, CUDA
  driver faults, `kill -9`). Verified surviving 3 consecutive `SIGKILL`s in
  a row, restarting within ~2s each time.

## [0.2.0] - 2026-06-22

### Changed
- Renamed project from `speakd` to `voxcaster`. PyPI: `pip install voxcaster`.

### Added
- Markdown preprocessing: `strip_markdown()` removes code fences, URLs, headers, and formatting before synthesis. On by default (`markdown_preprocess = true`).
- TTS_SUMMARY extraction: embed `<!-- TTS_SUMMARY concise version TTS_SUMMARY -->` in text and only the summary is spoken — ideal for Claude Code hook integration.
- Per-request voice override: `speak("text", voice="bf_emma")` and `speak --voice bf_emma` override the daemon's configured voice for a single line.
- `ping()` / `speak --ping`: health-check the daemon — returns `ready`, `starting`, or `down` without triggering synthesis.
- `always_interrupt` config flag: every incoming request preempts in-flight playback (useful for Claude Code hooks where only the latest narration matters).

## [0.1.0] - 2026-06-11

### Added
- Initial public release.
- `voxcaster` daemon: Kokoro TTS behind a Unix domain socket with a FIFO
  narration queue, flock singleton, and stale-socket cleanup.
- Dynamic GPU offload: the model rides the GPU during narration bursts and
  releases its VRAM after a configurable idle keepalive.
- Wire protocol: fire-and-forget speak, interrupt (drain queue + cut
  playback), and live volume control.
- `speak` client CLI and a stdlib-only Python API (`speak`, `set_volume`,
  `ensure_daemon`) with daemon auto-spawn and graceful espeak fallback.
- TOML configuration with `VOXCASTER_*` environment overrides.
