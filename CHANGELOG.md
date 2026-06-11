# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-06-11

### Added
- Initial public release.
- `speakd` daemon: Kokoro TTS behind a Unix domain socket with a FIFO
  narration queue, flock singleton, and stale-socket cleanup.
- Dynamic GPU offload: the model rides the GPU during narration bursts and
  releases its VRAM after a configurable idle keepalive.
- Wire protocol: fire-and-forget speak, interrupt (drain queue + cut
  playback), and live volume control.
- `speak` client CLI and a stdlib-only Python API (`speak`, `set_volume`,
  `ensure_daemon`) with daemon auto-spawn and graceful espeak fallback.
- TOML configuration with `SPEAKD_*` environment overrides.
