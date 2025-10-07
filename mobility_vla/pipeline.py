"""High-level orchestrator for the MobilityVLA navigation stack."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional, Sequence

from .control import ActionCommandAdapter, MotionCommand, SimpleCommandAdapter
from .graph import TopologicalGraph, TopologicalGraphBuilder
from .high_level import HighLevelGoalSelector, KeywordHighLevelSelector
from .localization import LocalizationResult, Localizer, NearestNeighborLocalizer
from .navigation import LowLevelNavigator, WaypointAction
from .state import PoseTracker
from .tour import DemonstrationTour
from .types import Instruction, Observation, Pose


@dataclass
class MobilityVLAConfig:
    graph_builder: TopologicalGraphBuilder = field(
        default_factory=TopologicalGraphBuilder
    )
    high_level_selector: HighLevelGoalSelector = field(
        default_factory=KeywordHighLevelSelector
    )
    localizer: Localizer = field(default_factory=NearestNeighborLocalizer)
    control_adapter: Optional[ActionCommandAdapter] = field(
        default_factory=SimpleCommandAdapter
    )


class MobilityVLA:
    """Turns instructions and observations into waypoint actions."""

    def __init__(self, tour: DemonstrationTour, config: Optional[MobilityVLAConfig] = None):
        self._tour = tour
        self._config = config or MobilityVLAConfig()
        self._graph: TopologicalGraph = self._config.graph_builder.build(tour)
        self._high_level_selector = self._config.high_level_selector
        self._localizer = self._config.localizer
        self._navigator = LowLevelNavigator(self._graph)
        self._pose_tracker = PoseTracker()
        self._control_adapter = self._config.control_adapter

        self._goal_frame_id: Optional[str] = None

    @property
    def goal_frame_id(self) -> Optional[str]:
        return self._goal_frame_id

    @property
    def current_pose(self) -> Optional[Pose]:
        estimate = self._pose_tracker.estimate
        return estimate.pose if estimate else None

    @property
    def current_frame_id(self) -> Optional[str]:
        estimate = self._pose_tracker.estimate
        return estimate.frame_id if estimate else None

    def set_instruction(self, instruction: Instruction) -> str:
        """Selects a goal frame and returns its identifier."""

        goal_frame_id = self._high_level_selector.select_goal(self._tour, instruction)
        self._goal_frame_id = goal_frame_id
        return goal_frame_id

    def update_localization(self, observation: Observation) -> LocalizationResult:
        """Updates the internal pose estimate using a fresh observation."""

        localization = self._localizer.localize(observation, self._graph)
        self._pose_tracker.correct(localization)
        return localization

    def step(self, observation: Observation) -> "MobilityVLA.StepResult":
        if self._goal_frame_id is None:
            raise RuntimeError("set_instruction must be called before step().")

        localization = self.update_localization(observation)
        path = self._navigator.plan_path(localization.frame_id, self._goal_frame_id)
        next_frame_id = path[1] if len(path) > 1 else path[0]
        next_pose = self._graph.frame(next_frame_id).pose
        action = self._navigator.compute_action(localization.pose, next_pose)
        predicted_estimate = self._pose_tracker.predict(action, next_frame_id=next_frame_id)
        command = (
            self._control_adapter.to_command(action) if self._control_adapter else None
        )
        return MobilityVLA.StepResult(
            action=action,
            path=path,
            localization=localization,
            predicted_pose=predicted_estimate.pose,
            predicted_frame_id=predicted_estimate.frame_id,
            command=command,
        )

    @dataclass
    class StepResult:
        action: WaypointAction
        path: Sequence[str]
        localization: LocalizationResult
        predicted_pose: Pose
        predicted_frame_id: str
        command: Optional[MotionCommand] = None
