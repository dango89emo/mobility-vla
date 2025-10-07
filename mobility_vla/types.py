"""Common data types used throughout the MobilityVLA implementation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Dict, Iterable, Optional, Sequence


@dataclass(frozen=True)
class Pose:
    """Simple pose container (x, y, z, yaw).

    The reference frame is assumed to be robot-centric with yaw expressed in
    radians. The z component captures height when available. Consumers that only
    reason about planar motion can set ``z=0.0``.
    """

    x: float
    y: float
    z: float = 0.0
    yaw: float = 0.0


@dataclass(frozen=True)
class Instruction:
    """Runtime multimodal instruction.

    Attributes:
        text: Natural language instruction provided by the user.
        image_path: Optional reference image path that accompanies the text
            instruction (e.g., a user pointing at an object).
        meta: Arbitrary metadata that downstream components may require.
    """

    text: str
    image_path: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)


@dataclass(frozen=True)
class Observation:
    """Robot observation at a single timestep.

    Attributes:
        image_path: Optional path to the captured RGB frame.
        features: Optional embedding or descriptor vector represented as a
            sequence of floats; consumers decide how to interpret the values.
        hint: Optional coarse textual hint about the current location (useful in
            simulation or for offline evaluation scripts).
        meta: Arbitrary metadata.
    """

    image_path: Optional[str] = None
    features: Optional[Sequence[float]] = None
    hint: Optional[str] = None
    meta: Dict[str, str] = field(default_factory=dict)


def normalize_tokens(text: str) -> Iterable[str]:
    """Utility tokenizer used by several heuristics.

    The function lowercases the text and splits on non-alphabetic characters to
    produce a simple bag-of-words representation. Tokens shorter than two
    characters are discarded.
    """

    token = []
    for char in text.lower():
        if char.isalpha():
            token.append(char)
            continue
        if token:
            assembled = "".join(token)
            if len(assembled) > 1:
                yield assembled
            token = []
    if token:
        assembled = "".join(token)
        if len(assembled) > 1:
            yield assembled

