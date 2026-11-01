"""The daemon's FIFO of lines waiting to be spoken, with stable ids so a single
line can be listed and removed."""
from __future__ import annotations

import asyncio
import itertools
from collections import deque
from dataclasses import dataclass


@dataclass(frozen=True)
class Line:
    id: int
    text: str
    voice: str | None
    render_path: str | None
    writer: asyncio.StreamWriter


class LineQueue:
    def __init__(self) -> None:
        self._lines: deque[Line] = deque()
        self._arrived = asyncio.Event()
        self._ids = itertools.count(1)

    def put(self, text: str, voice: str | None, render_path: str | None,
            writer: asyncio.StreamWriter) -> Line:
        line = Line(next(self._ids), text, voice, render_path, writer)
        self._lines.append(line)
        self._arrived.set()
        return line

    async def get(self) -> Line:
        while not self._lines:
            self._arrived.clear()
            await self._arrived.wait()
        return self._lines.popleft()

    def snapshot(self) -> list[Line]:
        return list(self._lines)

    def remove(self, line_id: int) -> Line | None:
        for line in self._lines:
            if line.id == line_id:
                self._lines.remove(line)
                return line
        return None

    def drain(self) -> list[Line]:
        lines = list(self._lines)
        self._lines.clear()
        return lines

    def __len__(self) -> int:
        return len(self._lines)
