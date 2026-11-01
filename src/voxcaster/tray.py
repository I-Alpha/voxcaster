"""System-tray control for the voxcaster daemon.

Needs PyGObject with Gtk 3 and AyatanaAppIndicator3, which conda environments
usually lack; ``voxcaster-tray`` re-runs itself under ``/usr/bin/python3`` when
they are missing. The daemon stays headless: the tray is a separate process
that speaks the wire protocol and edits the config file.
"""
from __future__ import annotations

import argparse
import fcntl
import os
import sys

from . import protocol
from .config import _runtime_path, load_config
from .traylogic import (
    DEFAULT_CONFIG,
    DEPTHS,
    HOLDS_MS,
    PRESETS,
    SocketWire,
    TrayController,
    db_to_volume,
    line_label,
    volume_to_db,
)

try:
    import gi

    gi.require_version("Gtk", "3.0")
    gi.require_version("AyatanaAppIndicator3", "0.1")
    from gi.repository import AyatanaAppIndicator3 as AppIndicator
    from gi.repository import GLib, Gtk
except (ImportError, ValueError):
    gi = None

REFRESH_SECONDS = 3
VOLUME_DEBOUNCE_MS = 350
REFRESH_AFTER_ACTION_MS = 250
SYSTEM_PYTHON = "/usr/bin/python3"


def _db_text(volume: int) -> str:
    return "silent" if volume <= 0 else f"{volume_to_db(volume):+.1f} dB"


class VolumeWindow:
    def __init__(self, controller: TrayController) -> None:
        self._controller = controller
        self._pending = 0
        self.window = Gtk.Window(title="Voxcaster voice volume")
        self.window.set_default_size(360, 120)
        self.window.set_border_width(12)
        self.window.connect("delete-event", lambda w, e: w.hide() or True)
        box = Gtk.Box(orientation=Gtk.Orientation.VERTICAL, spacing=8)
        self.label = Gtk.Label()
        self.scale = Gtk.Scale.new_with_range(
            Gtk.Orientation.HORIZONTAL, protocol.VOLUME_MIN, protocol.VOLUME_MAX, 1)
        self.scale.set_digits(0)
        self.scale.set_draw_value(False)
        self.scale.add_mark(100, Gtk.PositionType.BOTTOM, "100")
        self.scale.connect("value-changed", self._changed)
        test = Gtk.Button(label="Test voice")
        test.connect("clicked", lambda b: controller.test_voice())
        box.pack_start(self.label, False, False, 0)
        box.pack_start(self.scale, False, False, 0)
        box.pack_start(test, False, False, 0)
        self.window.add(box)
        self._set_label(int(self.scale.get_value()))

    def show(self, volume: int | None) -> None:
        if volume is not None:
            self.scale.set_value(volume)
        self.window.show_all()
        self.window.present()

    def _set_label(self, volume: int) -> None:
        self.label.set_text(f"Volume {volume}  ({_db_text(volume)} on the player scale)")

    def _changed(self, scale) -> None:
        self._set_label(int(scale.get_value()))
        if self._pending:
            GLib.source_remove(self._pending)
        self._pending = GLib.timeout_add(VOLUME_DEBOUNCE_MS, self._apply)

    def _apply(self) -> bool:
        self._pending = 0
        try:
            self._controller.set_volume(int(self.scale.get_value()))
        except Exception as e:
            self.label.set_text(f"Could not apply: {e}")
        return False


class TrayApp:
    def __init__(self, controller: TrayController) -> None:
        self.controller = controller
        self._syncing = False
        self._volume_window: VolumeWindow | None = None
        self._shown_queue: list[tuple[int, str]] | None = None
        self.indicator = AppIndicator.Indicator.new(
            "voxcaster", "audio-volume-high",
            AppIndicator.IndicatorCategory.APPLICATION_STATUS)
        self.indicator.set_title("Voxcaster")
        self.indicator.set_status(AppIndicator.IndicatorStatus.ACTIVE)
        self.items: dict[str, object] = {}
        self.indicator.set_menu(self._build_menu())
        self.refresh()

    # ── menu ──────────────────────────────────────────────────────────────

    def _item(self, name: str, label: str, handler=None, menu=None):
        item = Gtk.MenuItem(label=label)
        if handler:
            item.connect("activate", lambda _i: self._guarded(handler))
        self.items[name] = item
        (menu or self.menu).append(item)
        return item

    def _build_menu(self):
        self.menu = Gtk.Menu()
        status = self._item("status", "Voxcaster: ...")
        status.set_sensitive(False)
        self.menu.append(Gtk.SeparatorMenuItem())

        self._item("volume", "Voice volume...", self.open_volume_window)
        presets = Gtk.Menu()
        for label, db in PRESETS:
            volume = db_to_volume(db)
            self._item(f"preset:{label}", f"{label}  ({db:+.1f} dB, {volume})",
                       lambda v=volume: self.controller.set_volume(v), presets)
        self.menu.append(self._submenu("Volume presets", presets))
        mute = Gtk.CheckMenuItem(label="Mute voice")
        mute.connect("toggled", lambda _i: self._on_mute())
        self.items["mute"] = mute
        self.menu.append(mute)
        self.menu.append(Gtk.SeparatorMenuItem())

        self.menu.append(self._submenu("Ducking", self._ducking_menu()))
        self.menu.append(Gtk.SeparatorMenuItem())

        self._item("now", "Now: nothing playing").set_sensitive(False)
        self._item("skip", "Skip current line", self.controller.skip)
        self._item("pause", "Pause", self.controller.toggle_pause)
        self._item("clear", "Clear queue", self.controller.clear_queue)
        self._queue_menu = Gtk.Menu()
        self.items["queue"] = self._submenu("Queue", self._queue_menu)
        self.menu.append(self.items["queue"])
        self._item("test", "Test voice", self.controller.test_voice)
        self._item("stop", "Stop everything", self.controller.stop_speech)
        voices = Gtk.Menu()
        for name, voice in self.controller.agent_voices():
            entry = Gtk.MenuItem(label=f"{name}: {voice}")
            entry.set_sensitive(False)
            voices.append(entry)
        self.menu.append(self._submenu("Agent voices (read only)", voices))
        self.menu.append(Gtk.SeparatorMenuItem())

        self._item("restart", "Restart daemon", self.controller.restart_daemon)
        self._item("config", "Open config", self.controller.open_config)
        self._item("quit", "Quit", self.quit)
        self.menu.connect("show", lambda _m: self.refresh())
        self.menu.show_all()
        return self.menu

    def _submenu(self, label: str, submenu):
        parent = Gtk.MenuItem(label=label)
        parent.set_submenu(submenu)
        return parent

    def _ducking_menu(self):
        menu = Gtk.Menu()
        enabled = Gtk.CheckMenuItem(label="Duck other audio while speaking")
        enabled.connect("toggled", lambda i: self._on_duck_toggle(i))
        self.items["duck_enabled"] = enabled
        menu.append(enabled)
        menu.append(Gtk.SeparatorMenuItem())
        group = None
        for label, factor in DEPTHS:
            radio = Gtk.RadioMenuItem.new_with_label_from_widget(group, label)
            group = group or radio
            radio.connect("toggled", lambda i, f=factor: self._on_depth(i, f))
            self.items[f"depth:{factor}"] = radio
            menu.append(radio)
        menu.append(Gtk.SeparatorMenuItem())
        group = None
        for ms in HOLDS_MS:
            radio = Gtk.RadioMenuItem.new_with_label_from_widget(
                group, f"Hold {ms} ms after a line" if ms else "No hold")
            group = group or radio
            radio.connect("toggled", lambda i, m=ms: self._on_hold(i, m))
            self.items[f"hold:{ms}"] = radio
            menu.append(radio)
        return menu

    # ── actions ───────────────────────────────────────────────────────────

    def _guarded(self, handler) -> None:
        try:
            handler()
        except Exception as e:
            self.items["status"].set_label(f"Voxcaster: {e}")
        else:
            GLib.timeout_add(REFRESH_AFTER_ACTION_MS, lambda: self.refresh() and False)

    def _on_mute(self) -> None:
        if not self._syncing:
            self._guarded(self.controller.toggle_mute)
            self.refresh()

    def _on_duck_toggle(self, item) -> None:
        if not self._syncing:
            self._guarded(lambda: self.controller.set_ducking(enabled=item.get_active()))

    def _on_depth(self, item, factor: float) -> None:
        if item.get_active() and not self._syncing:
            self._guarded(lambda: self.controller.set_ducking(factor=factor))

    def _on_hold(self, item, ms: int) -> None:
        if item.get_active() and not self._syncing:
            self._guarded(lambda: self.controller.set_ducking(hold_ms=ms))

    def open_volume_window(self) -> None:
        if self._volume_window is None:
            self._volume_window = VolumeWindow(self.controller)
        status = self.controller.status()
        self._volume_window.show(status["volume"] if status else None)

    def quit(self) -> None:
        Gtk.main_quit()

    # ── state ─────────────────────────────────────────────────────────────

    def _show_transport(self, s: dict) -> None:
        voices = self.controller.agent_voices()
        current, queue = s.get("current"), s.get("queue", [])
        self.items["now"].set_label(
            "Now: " + (line_label(current, voices) if current else "nothing playing"))
        self.items["pause"].set_label("Resume" if s.get("paused") else "Pause")
        self.items["skip"].set_sensitive(current is not None)
        self.items["clear"].set_sensitive(bool(queue))
        self.items["queue"].set_label(f"Queue ({len(queue)})")
        self.items["queue"].set_sensitive(bool(queue))
        shown = [(line["id"], line_label(line, voices)) for line in queue]
        if shown == self._shown_queue:
            return
        self._shown_queue = shown
        for child in self._queue_menu.get_children():
            self._queue_menu.remove(child)
        for line_id, label in shown:
            item = Gtk.MenuItem.new_with_label("Remove: " + label)
            item.connect("activate", lambda _i, n=line_id: self._guarded(
                lambda: self._drop(n)))
            self._queue_menu.append(item)
        self._queue_menu.show_all()

    def _drop(self, line_id: int) -> None:
        if not self.controller.drop_line(line_id):
            raise RuntimeError("that line is no longer queued")

    def refresh(self) -> bool:
        s = self.controller.status()
        self._syncing = True
        try:
            if s is None:
                self.items["status"].set_label("Voxcaster: daemon not running")
                self.indicator.set_icon_full("dialog-error", "daemon down")
                return True
            muted = s["volume"] <= 0
            state = "ready" if s["ready"] else "starting"
            doing = ("speaking" if s["speaking"] else "working" if s["busy"] else "idle")
            self.items["status"].set_label(
                f"Voxcaster: {state}, {doing}, queue {s['queued']}, "
                f"volume {s['volume']} ({_db_text(s['volume'])})")
            self.indicator.set_icon_full(
                "audio-volume-muted" if muted else "audio-volume-high",
                "muted" if muted else "voice on")
            self.items["mute"].set_active(muted)
            self._show_transport(s)
            duck = s["ducking"]
            self.items["duck_enabled"].set_active(duck["enabled"])
            for factor in (f for _, f in DEPTHS):
                if abs(factor - duck["factor"]) < 1e-9:
                    self.items[f"depth:{factor}"].set_active(True)
            if duck["hold_ms"] in HOLDS_MS:
                self.items[f"hold:{duck['hold_ms']}"].set_active(True)
        finally:
            self._syncing = False
        return True


def _single_instance():
    lock = open(_runtime_path("voxcaster-tray", ".lock"), "w")
    try:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
    except OSError:
        print("voxcaster-tray is already running", file=sys.stderr)
        raise SystemExit(0)
    return lock


def _rerun_under_system_python(argv: list[str]) -> None:
    if os.environ.get("VOXCASTER_TRAY_REEXEC") or not os.path.exists(SYSTEM_PYTHON):
        raise SystemExit(
            "voxcaster-tray needs PyGObject with Gtk 3 and AyatanaAppIndicator3 "
            "(apt: python3-gi gir1.2-ayatanaappindicator3-0.1)")
    package_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    env = dict(os.environ, VOXCASTER_TRAY_REEXEC="1")
    env["PYTHONPATH"] = package_root + os.pathsep + env.get("PYTHONPATH", "")
    os.execve(SYSTEM_PYTHON, [SYSTEM_PYTHON, "-m", "voxcaster.tray", *argv], env)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    parser = argparse.ArgumentParser(prog="voxcaster-tray",
                                     description="Tray control for the voxcaster daemon.")
    parser.add_argument("--socket", help="daemon socket (default: from config/env)")
    parser.add_argument("--config", default=DEFAULT_CONFIG, help="config.toml to edit")
    args = parser.parse_args(argv)
    if gi is None:
        _rerun_under_system_python(argv)
    lock = _single_instance()
    socket_path = args.socket or load_config(args.config if os.path.exists(args.config)
                                             else None).socket_path
    controller = TrayController(SocketWire(socket_path), config_path=args.config)
    app = TrayApp(controller)
    GLib.timeout_add_seconds(REFRESH_SECONDS, app.refresh)
    try:
        Gtk.main()
    finally:
        lock.close()


if __name__ == "__main__":
    main()
