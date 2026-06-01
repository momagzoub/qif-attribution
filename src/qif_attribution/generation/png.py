"""Tiny PNG writer used for deterministic placeholder generation."""

from __future__ import annotations

import struct
import zlib
from pathlib import Path

PNG_SIGNATURE = b"\x89PNG\r\n\x1a\n"


def write_solid_rgb_png(path: Path, *, width: int, height: int, rgb: tuple[int, int, int]) -> None:
    """Write a valid solid-color RGB PNG without image-library dependencies."""

    if width <= 0 or height <= 0:
        raise ValueError("width and height must be positive")
    if any(channel < 0 or channel > 255 for channel in rgb):
        raise ValueError("rgb channels must be between 0 and 255")

    path.parent.mkdir(parents=True, exist_ok=True)
    raw_row = b"\x00" + bytes(rgb) * width
    raw_data = raw_row * height
    with path.open("wb") as handle:
        handle.write(PNG_SIGNATURE)
        handle.write(_chunk(b"IHDR", struct.pack(">IIBBBBB", width, height, 8, 2, 0, 0, 0)))
        handle.write(_chunk(b"IDAT", zlib.compress(raw_data, level=9)))
        handle.write(_chunk(b"IEND", b""))


def _chunk(chunk_type: bytes, data: bytes) -> bytes:
    crc = zlib.crc32(chunk_type)
    crc = zlib.crc32(data, crc)
    return struct.pack(">I", len(data)) + chunk_type + data + struct.pack(">I", crc & 0xFFFFFFFF)
