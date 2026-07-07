"""voxcaster — fire-and-forget local TTS narration over a Unix socket.

A small daemon that turns text lines into speech with `Kokoro
<https://github.com/hexgrad/kokoro>`_, plus a zero-dependency client.
Designed for narrating long-running work (training runs, builds, pipelines)
without ever blocking or crashing the thing doing the work.

Quickstart::

    from voxcaster import speak
    speak("experiment finished")                 # fire-and-forget
    speak("loss is NaN — stopping", interrupt=True)
"""
from typing import TYPE_CHECKING

__version__ = "0.2.1"

if TYPE_CHECKING:  # real imports for type checkers / IDEs
    from .client import ensure_daemon, ping, set_volume, speak
    from .config import Config, load_config
    from .markdown import extract_tts_summary, preprocess_for_speech, strip_markdown

__all__ = [
    "speak", "ping", "set_volume", "ensure_daemon",
    "Config", "load_config",
    "strip_markdown", "extract_tts_summary", "preprocess_for_speech",
    "__version__",
]

_CLIENT_ATTRS = ("speak", "ping", "set_volume", "ensure_daemon")
_CONFIG_ATTRS = ("Config", "load_config")
_MARKDOWN_ATTRS = ("strip_markdown", "extract_tts_summary", "preprocess_for_speech")


def __getattr__(name: str):
    """Lazy re-exports (PEP 562): keep ``import voxcaster`` instant and avoid
    eagerly importing submodules that ``python -m voxcaster.<mod>`` re-executes."""
    if name in _CLIENT_ATTRS:
        from . import client
        return getattr(client, name)
    if name in _CONFIG_ATTRS:
        from . import config
        return getattr(config, name)
    if name in _MARKDOWN_ATTRS:
        from . import markdown
        return getattr(markdown, name)
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
