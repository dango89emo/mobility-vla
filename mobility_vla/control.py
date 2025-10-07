"""Adapters that convert abstract waypoint actions into robot motion commands."""

from __future__ import annotations

import math
from dataclasses import dataclass

from .navigation import WaypointAction


@dataclass(frozen=True)
class MotionCommand:
    """Simple time-parameterized velocity command.

    The command mirrors the common ROS Twist parameterization and captures the
    duration for which the command should be executed.
    """

    linear_x: float
    linear_y: float
    angular_z: float
    duration: float


class ActionCommandAdapter:
    """Base interface for translating waypoint actions into motion commands."""

    def to_command(self, action: WaypointAction) -> MotionCommand:
        raise NotImplementedError


class SimpleCommandAdapter(ActionCommandAdapter):
    """Heuristic adapter that converts relative displacements into velocities.

    The adapter assumes a holonomic base and generates linear velocities in the
    robot frame accompanied by an angular velocity around the vertical axis. The
    command duration is chosen such that both the translational and rotational
    components finish simultaneously.
    """

    def __init__(
        self,
        translation_speed: float = 0.25,
        rotation_speed: float = 0.75,
        minimum_duration: float = 1e-3,
    ) -> None:
        if translation_speed <= 0.0:
            raise ValueError("translation_speed must be positive.")
        if rotation_speed <= 0.0:
            raise ValueError("rotation_speed must be positive.")
        if minimum_duration < 0.0:
            raise ValueError("minimum_duration cannot be negative.")

        self._translation_speed = translation_speed
        self._rotation_speed = rotation_speed
        self._minimum_duration = minimum_duration

    def to_command(self, action: WaypointAction) -> MotionCommand:
        distance = math.hypot(action.dx, action.dy)
        duration_translation = distance / self._translation_speed if distance else 0.0
        duration_rotation = abs(action.dtheta) / self._rotation_speed if action.dtheta else 0.0

        if duration_translation == 0.0 and duration_rotation == 0.0:
            return MotionCommand(linear_x=0.0, linear_y=0.0, angular_z=0.0, duration=0.0)

        duration = max(duration_translation, duration_rotation, self._minimum_duration)
        linear_x = action.dx / duration
        linear_y = action.dy / duration
        angular_z = action.dtheta / duration
        return MotionCommand(
            linear_x=linear_x,
            linear_y=linear_y,
            angular_z=angular_z,
            duration=duration,
        )
