"""Topological graph representation built from demonstration tours."""

from __future__ import annotations

import heapq
import math
from dataclasses import dataclass
from typing import Dict, Iterable, List, MutableMapping, Tuple

from .tour import DemonstrationTour, TourFrame
from .types import Pose


def _euclidean_distance(a: Pose, b: Pose) -> float:
    return math.sqrt((a.x - b.x) ** 2 + (a.y - b.y) ** 2 + (a.z - b.z) ** 2)


def _normalize_angle(angle: float) -> float:
    """Wraps angle to the range [-pi, pi]."""

    while angle > math.pi:
        angle -= 2.0 * math.pi
    while angle < -math.pi:
        angle += 2.0 * math.pi
    return angle


def _bearing(from_pose: Pose, to_pose: Pose) -> float:
    return math.atan2(to_pose.y - from_pose.y, to_pose.x - from_pose.x)


@dataclass
class TopologicalGraph:
    """Directed topological graph built from the demonstration tour."""

    frames: Dict[str, TourFrame]
    edges: Dict[str, Dict[str, float]]

    def neighbors(self, frame_id: str) -> Dict[str, float]:
        """Returns the neighbor dictionary for ``frame_id``."""

        try:
            return self.edges[frame_id]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Frame '{frame_id}' has no outgoing edges.") from exc

    def frame(self, frame_id: str) -> TourFrame:
        try:
            return self.frames[frame_id]
        except KeyError as exc:  # pragma: no cover - defensive programming
            raise KeyError(f"Unknown frame '{frame_id}'.") from exc

    def shortest_path(self, start_id: str, goal_id: str) -> List[str]:
        """Shortest path using Dijkstra's algorithm."""

        if start_id == goal_id:
            return [start_id]

        frontier: List[Tuple[float, str]] = [(0.0, start_id)]
        best_cost: Dict[str, float] = {start_id: 0.0}
        predecessor: Dict[str, str] = {}

        while frontier:
            cost, node = heapq.heappop(frontier)
            if node == goal_id:
                break

            for neighbor, weight in self.edges.get(node, {}).items():
                new_cost = cost + weight
                if new_cost < best_cost.get(neighbor, math.inf):
                    best_cost[neighbor] = new_cost
                    predecessor[neighbor] = node
                    heapq.heappush(frontier, (new_cost, neighbor))

        if goal_id not in best_cost:
            raise ValueError(
                f"No path exists between '{start_id}' and '{goal_id}' in the graph."
            )

        path: List[str] = [goal_id]
        while path[-1] != start_id:
            path.append(predecessor[path[-1]])
        path.reverse()
        return path


class TopologicalGraphBuilder:
    """Constructs the topological graph according to MobilityVLA's heuristics."""

    def __init__(self, distance_threshold: float = 2.0, yaw_threshold_deg: float = 90.0):
        self._distance_threshold = distance_threshold
        self._yaw_threshold = math.radians(yaw_threshold_deg)

    def build(self, tour: DemonstrationTour) -> TopologicalGraph:
        frames = {frame.frame_id: frame for frame in tour}
        edges: MutableMapping[str, Dict[str, float]] = {
            frame.frame_id: {} for frame in tour
        }

        ordered_frames = list(tour)
        for idx, src in enumerate(ordered_frames):
            for jdx, dst in enumerate(ordered_frames):
                if idx == jdx:
                    continue
                distance = _euclidean_distance(src.pose, dst.pose)
                if distance == 0.0 or distance > self._distance_threshold:
                    continue

                bearing = _bearing(src.pose, dst.pose)
                yaw_delta = _normalize_angle(bearing - src.pose.yaw)
                if abs(yaw_delta) > self._yaw_threshold:
                    continue

                edges[src.frame_id][dst.frame_id] = distance

            # ensure sequential connectivity for robustness
            if idx + 1 < len(ordered_frames):
                nxt = ordered_frames[idx + 1]
                dist = _euclidean_distance(src.pose, nxt.pose)
                edges[src.frame_id][nxt.frame_id] = dist

        return TopologicalGraph(frames=frames, edges=dict(edges))

