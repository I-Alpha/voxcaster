"""Configuration for speakd.

Precedence (lowest to highest):

    1. Built-in defaults (work out of the box on CPU)
    2. TOML config file
    3. ``SPEAKD_*`` environment variables
    4. CLI flags (applied by the entry points)

The config file is looked up in this order:

    1. ``$SPEAKD_CONFIG``
    2. ``$XDG_CONFIG_HOME/speakd/config.toml``
       (default: ``~/.config/speakd/config.toml``)

Missing files are fine — every key has a sane default.
"""
from __future__ import annotations

import os
import tempfile
from dataclasses import dataclass, field, fields

try:
    import tomllib  # Python 3.11+
except ModuleNotFoundError:  # pragma: no cover - Python 3.10
    import tomli as tomllib  # type: ignore[no-redef]

VALID_DEVICE_POLICIES = ("auto", "cpu", "gpu")


def default_socket_path() -> str:
    """Per-user socket path: ``$XDG_RUNTIME_DIR/speakd.sock`` when available,
    otherwise a uid-suffixed path under the system temp dir."""
    runtime_dir = os.environ.get("XDG_RUNTIME_DIR")
    if runtime_dir and os.path.isdir(runtime_dir):
        return os.path.join(runtime_dir, "speakd.sock")
    return os.path.join(tempfile.gettempdir(), f"speakd-{os.getuid()}.sock")


def default_state_dir() -> str:
    """``$XDG_STATE_HOME/speakd`` (default: ``~/.local/state/speakd``)."""
    state_home = os.environ.get(
        "XDG_STATE_HOME", os.path.join(os.path.expanduser("~"), ".local", "state")
    )
    return os.path.join(state_home, "speakd")


def default_config_file() -> str:
    config_home = os.environ.get(
        "XDG_CONFIG_HOME", os.path.join(os.path.expanduser("~"), ".config")
    )
    return os.path.join(config_home, "speakd", "config.toml")


@dataclass
class Config:
    """Effective speakd configuration. See ``config.example.toml`` for docs."""

    # [tts]
    voice: str = "af_heart"  # Kokoro voice id (af_heart, bf_emma, am_adam, ...)
    speed: float = 1.0       # speech-rate multiplier
    lang_code: str = "a"     # Kokoro language code ("a" = American English)

    # [device]
    device: str = "auto"            # "cpu" | "gpu" | "auto" (dynamic offload)
    keepalive_seconds: int = 180    # idle seconds before GPU -> CPU offload

    # [daemon]
    socket_path: str = field(default_factory=default_socket_path)
    socket_mode: int = 0o600        # permissions applied to the socket file
    log_file: str = field(
        default_factory=lambda: os.path.join(default_state_dir(), "daemon.log")
    )

    # [audio]
    volume: int = 100                    # playback volume, 0-130 (mpv scale)
    max_playback_seconds: int = 120      # kill the player after this long
    player: list[str] = field(
        default_factory=lambda: ["mpv", "--no-terminal", "--volume={volume}", "{file}"]
    )

    # [fallback] - argv template used when TTS fails; [] disables the fallback
    fallback: list[str] = field(
        default_factory=lambda: ["espeak", "-s", "160", "-v", "en-us", "{text}"]
    )

    # [client]
    connect_timeout: float = 0.5    # seconds to connect/send on the socket
    ack_timeout: float = 300.0      # seconds to wait for the ack in blocking mode
    spawn_wait: float = 4.0         # seconds to wait for an auto-spawned daemon

    # Path of the TOML file this config was loaded from ("" if defaults only).
    source_file: str = ""

    @property
    def lock_path(self) -> str:
        """Singleton flock file, always derived from the socket path."""
        return self.socket_path + ".lock"

    @property
    def fallback_log(self) -> str:
        """Client-side log of fallback events, next to the daemon log."""
        return os.path.join(os.path.dirname(self.log_file), "fallback.log")

    def describe(self) -> str:
        """Human-readable dump of the effective configuration."""
        lines = [f"# effective speakd config (source: {self.source_file or 'defaults'})"]
        for f in fields(self):
            if f.name == "source_file":
                continue
            value = getattr(self, f.name)
            if f.name == "socket_mode":
                value = oct(value)
            lines.append(f"{f.name} = {value!r}")
        lines.append(f"lock_path = {self.lock_path!r}")
        lines.append(f"fallback_log = {self.fallback_log!r}")
        return "\n".join(lines)


# (section, key, attribute, caster) - the full TOML surface.
_FILE_KEYS = [
    ("tts", "voice", "voice", str),
    ("tts", "speed", "speed", float),
    ("tts", "lang_code", "lang_code", str),
    ("device", "policy", "device", str),
    ("device", "keepalive_seconds", "keepalive_seconds", int),
    ("daemon", "socket_path", "socket_path", str),
    ("daemon", "socket_mode", "socket_mode", lambda v: int(str(v), 8)),
    ("daemon", "log_file", "log_file", str),
    ("audio", "volume", "volume", int),
    ("audio", "max_playback_seconds", "max_playback_seconds", int),
    ("audio", "player", "player", lambda v: [str(a) for a in v]),
    ("fallback", "command", "fallback", lambda v: [str(a) for a in v]),
    ("client", "connect_timeout", "connect_timeout", float),
    ("client", "ack_timeout", "ack_timeout", float),
    ("client", "spawn_wait", "spawn_wait", float),
]

# Environment overrides for the headline knobs.
_ENV_KEYS = [
    ("SPEAKD_VOICE", "voice", str),
    ("SPEAKD_SPEED", "speed", float),
    ("SPEAKD_LANG", "lang_code", str),
    ("SPEAKD_DEVICE", "device", str),
    ("SPEAKD_KEEPALIVE", "keepalive_seconds", int),
    ("SPEAKD_SOCKET", "socket_path", str),
    ("SPEAKD_VOLUME", "volume", int),
    ("SPEAKD_LOG_FILE", "log_file", str),
]


def load_config(path: str | None = None) -> Config:
    """Build the effective config: defaults -> TOML file -> environment.

    ``path`` (or ``$SPEAKD_CONFIG``) names an explicit TOML file; an explicit
    path that does not exist raises ``FileNotFoundError``. The default
    XDG-location file is optional and silently skipped when absent.
    """
    cfg = Config()

    explicit = path or os.environ.get("SPEAKD_CONFIG")
    file = explicit or default_config_file()
    if explicit and not os.path.exists(explicit):
        raise FileNotFoundError(f"config file not found: {explicit}")
    if os.path.exists(file):
        with open(file, "rb") as fh:
            data = tomllib.load(fh)
        for section, key, attr, cast in _FILE_KEYS:
            if section in data and key in data[section]:
                try:
                    setattr(cfg, attr, cast(data[section][key]))
                except (TypeError, ValueError) as e:
                    raise ValueError(f"bad value for [{section}] {key} in {file}: {e}") from e
        cfg.source_file = file

    for env, attr, cast in _ENV_KEYS:
        raw = os.environ.get(env)
        if raw is not None and raw != "":
            try:
                setattr(cfg, attr, cast(raw))
            except ValueError as e:
                raise ValueError(f"bad value for ${env}={raw!r}: {e}") from e

    if cfg.device not in VALID_DEVICE_POLICIES:
        raise ValueError(
            f"device policy must be one of {VALID_DEVICE_POLICIES}, got {cfg.device!r}"
        )
    return cfg
