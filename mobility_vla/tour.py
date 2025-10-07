"""Utilities and data abstractions for demonstration tours."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, List, Optional

from .types import Pose


@dataclass
class TourFrame:
    """Single frame collected during the demonstration tour."""

    frame_id: str
    pose: Pose
    image_path: Optional[str] = None
    narrative: Optional[str] = None
    global_descriptor: Optional[List[float]] = None
    metadata: Dict[str, str] = field(default_factory=dict)

    def summary(self) -> str:
        """Returns a human-readable summary used in prompts and logs."""

        parts: List[str] = [f"Frame {self.frame_id}"]
        if self.narrative:
            parts.append(self.narrative.strip())
        if self.metadata:
            details = ", ".join(f"{k}={v}" for k, v in sorted(self.metadata.items()))
            parts.append(f"({details})")
        return " ".join(parts)


@dataclass
class DemonstrationTour:
    """Container for an ordered demonstration tour."""

    frames: List[TourFrame]

    def __post_init__(self) -> None:
        if not self.frames:
            raise ValueError("DemonstrationTour requires at least one frame.")

    def __iter__(self) -> Iterable[TourFrame]:
        return iter(self.frames)

    def __len__(self) -> int:
        return len(self.frames)

    def frame_by_id(self, frame_id: str) -> TourFrame:
        """Random access lookup with a friendly error message."""

        for frame in self.frames:
            if frame.frame_id == frame_id:
                return frame
        raise KeyError(f"Unknown frame_id '{frame_id}' requested.")
