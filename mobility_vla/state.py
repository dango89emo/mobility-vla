"""Lightweight pose tracking utilities used across the pipeline."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from .localization import LocalizationResult
from .navigation import WaypointAction
from .types import Pose


def _normalize_angle(angle: float) -> float:
    """Wraps an angle to the range [-pi, pi]."""

    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


@dataclass
class PoseEstimate:
    """Current best estimate of the robot pose anchored to a tour frame."""

    frame_id: str
    pose: Pose


class PoseTracker:
    """Maintains a running pose estimate via predict/correct updates."""

    def __init__(self) -> None:
        self._estimate: Optional[PoseEstimate] = None

    def reset(self) -> None:
        self._estimate = None

    def correct(self, localization: LocalizationResult) -> PoseEstimate:
        """Applies a measurement update (e.g., via PnP localization)."""

        estimate = PoseEstimate(frame_id=localization.frame_id, pose=localization.pose)
        self._estimate = estimate
        return estimate

    def predict(self, action: WaypointAction, next_frame_id: Optional[str] = None) -> PoseEstimate:
        """Propagates the pose using the commanded action in the robot frame."""

        if self._estimate is None:
            raise RuntimeError("PoseTracker.predict() requires an existing estimate.")

        current_pose = self._estimate.pose
        cos_yaw = math.cos(current_pose.yaw)
        sin_yaw = math.sin(current_pose.yaw)

        dx_world = cos_yaw * action.dx - sin_yaw * action.dy
        dy_world = sin_yaw * action.dx + cos_yaw * action.dy

        predicted_pose = Pose(
            x=current_pose.x + dx_world,
            y=current_pose.y + dy_world,
            z=current_pose.z,
            yaw=_normalize_angle(current_pose.yaw + action.dtheta),
        )

        frame_id = next_frame_id or self._estimate.frame_id
        estimate = PoseEstimate(frame_id=frame_id, pose=predicted_pose)
        self._estimate = estimate
        return estimate

    @property
    def estimate(self) -> Optional[PoseEstimate]:
        return self._estimate
