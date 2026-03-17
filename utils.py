from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime
from typing import Any, Optional


def log(message: str, *, level: str = "INFO") -> None:
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] [{level}] {message}")


def clamp(value: int, min_value: int, max_value: int) -> int:
    return max(min_value, min(value, max_value))


@dataclass(frozen=True)
class FrameInfo:
    """
    Minimal metadata for a processed frame.

    You can extend this later (e.g., add lane ROIs, per-lane counts, etc.).
    """

    vehicle_count: int
    source: Optional[str] = None
    extra: Optional[dict[str, Any]] = None

