# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

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
