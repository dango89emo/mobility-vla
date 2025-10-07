"""Low-level navigation utilities."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import List

from .graph import TopologicalGraph, _normalize_angle
from .types import Pose


@dataclass
class WaypointAction:
    dx: float
    dy: float
    dtheta: float


class LowLevelNavigator:
    """Consumes localization results and computes waypoint actions."""

    def __init__(self, graph: TopologicalGraph):
        self._graph = graph

    def plan_path(self, start_id: str, goal_id: str) -> List[str]:
        return self._graph.shortest_path(start_id, goal_id)

    @staticmethod
    def compute_action(current_pose: Pose, target_pose: Pose) -> WaypointAction:
        dx_world = target_pose.x - current_pose.x
        dy_world = target_pose.y - current_pose.y

        cos_yaw = math.cos(current_pose.yaw)
        sin_yaw = math.sin(current_pose.yaw)

        dx_robot = cos_yaw * dx_world + sin_yaw * dy_world
        dy_robot = -sin_yaw * dx_world + cos_yaw * dy_world

        dtheta = _normalize_angle(target_pose.yaw - current_pose.yaw)
        return WaypointAction(dx=dx_robot, dy=dy_robot, dtheta=dtheta)

