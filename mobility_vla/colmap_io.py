"""Utilities to construct MobilityVLA tours from COLMAP outputs."""

from __future__ import annotations

import math
import re
from pathlib import Path
from typing import Iterable, List, Sequence, Tuple

import numpy as np

from .tour import DemonstrationTour, TourFrame
from .types import Pose

_POSE_PATTERN = re.compile(
    r"""
    ^\s*
    (?P<image>\S+)
    \s+q=\[(?P<quat>[^\]]+)\]
    \s+t=\[(?P<trans>[^\]]+)\]
    \s*$
    """,
    re.VERBOSE,
)


def _parse_pose_line(line: str) -> Tuple[str, Sequence[float], Sequence[float]]:
    match = _POSE_PATTERN.match(line)
    if not match:
        raise ValueError(f"Unable to parse COLMAP pose line: {line.strip()!r}")

    image_name = match.group("image")
    quat = [float(v) for v in match.group("quat").split()]
    trans = [float(v) for v in match.group("trans").split()]

    if len(quat) != 4:
        raise ValueError(f"Expected a quaternion of length 4, got {quat}")
    if len(trans) != 3:
        raise ValueError(f"Expected a translation vector of length 3, got {trans}")
    return image_name, quat, trans


def _rotation_matrix_from_quaternion(q: Sequence[float]) -> np.ndarray:
    qw, qx, qy, qz = q
    norm = math.sqrt(qw * qw + qx * qx + qy * qy + qz * qz)
    if norm == 0.0:
        raise ValueError("Quaternion has zero magnitude.")
    qw, qx, qy, qz = [v / norm for v in (qw, qx, qy, qz)]

    # Follows the standard Hamilton convention.
    return np.array(
        [
            [
                1 - 2 * (qy * qy + qz * qz),
                2 * (qx * qy - qz * qw),
                2 * (qx * qz + qy * qw),
            ],
            [
                2 * (qx * qy + qz * qw),
                1 - 2 * (qx * qx + qz * qz),
                2 * (qy * qz - qx * qw),
            ],
            [
                2 * (qx * qz - qy * qw),
                2 * (qy * qz + qx * qw),
                1 - 2 * (qx * qx + qy * qy),
            ],
        ],
        dtype=float,
    )


def _camera_center(rotation_cw: np.ndarray, translation_cw: Sequence[float]) -> np.ndarray:
    t = np.asarray(translation_cw, dtype=float).reshape(3, 1)
    return (-rotation_cw.T) @ t


def _camera_yaw(rotation_cw: np.ndarray) -> float:
    """Returns yaw (around +Z) from the world-from-camera rotation."""

    rotation_wc = rotation_cw.T
    return math.atan2(rotation_wc[1, 0], rotation_wc[0, 0])


def _metadata_from_name(image_name: str) -> str:
    stem = Path(image_name).stem.replace("_", " ")
    return stem.lower()


def load_tour_from_colmap(
    poses_path: Path | str,
    images_root: Path | str,
) -> DemonstrationTour:
    """Constructs a DemonstrationTour from COLMAP pose exports.

    Args:
        poses_path: Path to the ``poses.txt`` file produced by
            ``scripts/run_pycolmap_sfm.py``.
        images_root: Directory containing the original input images referenced
            by ``poses.txt``. The loader does not check for existence to remain
            flexible when the images were moved or renamed; callers may validate
            the paths separately.
    """

    poses_path = Path(poses_path)
    images_root = Path(images_root)
    if not poses_path.is_file():
        raise FileNotFoundError(f"poses.txt not found: {poses_path}")

    frames: List[TourFrame] = []
    with poses_path.open("r", encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            image_name, quat, trans = _parse_pose_line(line)
            rotation_cw = _rotation_matrix_from_quaternion(quat)
            center = _camera_center(rotation_cw, trans).ravel()

            pose = Pose(
                x=float(center[0]),
                y=float(center[1]),
                z=float(center[2]),
                yaw=_camera_yaw(rotation_cw),
            )

            metadata = {"keywords": _metadata_from_name(image_name)}
            frame = TourFrame(
                frame_id=Path(image_name).stem,
                pose=pose,
                image_path=str(images_root / image_name),
                narrative=None,
                metadata=metadata,
            )
            frames.append(frame)

    if not frames:
        raise ValueError(f"No frames were parsed from {poses_path}")

    return DemonstrationTour(frames=frames)

