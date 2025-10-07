from __future__ import annotations

import pytest

from mobility_vla import (
    DemonstrationTour,
    Instruction,
    MobilityVLA,
    Observation,
    Pose,
    SimpleCommandAdapter,
    TourFrame,
    WaypointAction,
)


def build_tour() -> DemonstrationTour:
    frames = [
        TourFrame(
            frame_id="F1",
            pose=Pose(x=0.0, y=0.0, yaw=0.0),
            narrative="Entrance lobby near reception desk.",
            metadata={"keywords": "entrance lobby reception"},
        ),
        TourFrame(
            frame_id="F2",
            pose=Pose(x=2.0, y=0.0, yaw=0.0),
            narrative="Hallway leading to offices and conference rooms.",
            metadata={"keywords": "hallway offices conference"},
        ),
        TourFrame(
            frame_id="F3",
            pose=Pose(x=4.0, y=0.0, yaw=0.0),
            narrative="Mail room with the package return box.",
            metadata={"keywords": "mail room return box"},
        ),
    ]
    return DemonstrationTour(frames=frames)


def test_high_level_goal_selection() -> None:
    tour = build_tour()
    mobility_vla = MobilityVLA(tour)
    instruction = Instruction(text="Where should I return this package?")
    goal = mobility_vla.set_instruction(instruction)
    assert goal == "F3"


def test_navigation_sequence() -> None:
    tour = build_tour()
    mobility_vla = MobilityVLA(tour)
    mobility_vla.set_instruction(Instruction(text="Where should I return this package?"))

    observations = [
        Observation(hint="entrance lobby"),
        Observation(hint="hallway"),
        Observation(hint="mail room"),
    ]

    result1 = mobility_vla.step(observations[0])
    assert result1.path == ["F1", "F2", "F3"]
    assert result1.action.dx == pytest.approx(2.0)
    assert result1.action.dy == pytest.approx(0.0)
    assert result1.predicted_frame_id == "F2"
    assert result1.predicted_pose.x == pytest.approx(2.0)
    assert result1.predicted_pose.y == pytest.approx(0.0)
    assert mobility_vla.current_frame_id == "F2"
    assert result1.command is not None
    assert result1.command.duration > 0.0

    result2 = mobility_vla.step(observations[1])
    assert result2.path == ["F2", "F3"]
    assert result2.action.dx == pytest.approx(2.0)
    assert result2.action.dy == pytest.approx(0.0)
    assert result2.predicted_frame_id == "F3"
    assert mobility_vla.current_frame_id == "F3"
    assert result2.command is not None
    assert result2.command.linear_x > 0.0

    result3 = mobility_vla.step(observations[2])
    assert result3.path == ["F3"]
    assert result3.action.dx == pytest.approx(0.0)
    assert result3.action.dy == pytest.approx(0.0)
    assert result3.command is not None
    assert result3.command.duration == pytest.approx(0.0)
    assert mobility_vla.current_frame_id == "F3"


def test_update_localization_refreshes_pose() -> None:
    tour = build_tour()
    mobility_vla = MobilityVLA(tour)
    mobility_vla.set_instruction(Instruction(text="Where should I return this package?"))

    localization = mobility_vla.update_localization(Observation(hint="hallway"))
    assert localization.frame_id == "F2"
    assert mobility_vla.current_frame_id == "F2"
    assert mobility_vla.current_pose is not None


def test_simple_command_adapter_converts_action() -> None:
    adapter = SimpleCommandAdapter(translation_speed=1.0, rotation_speed=1.0, minimum_duration=0.1)
    command = adapter.to_command(WaypointAction(dx=1.5, dy=0.0, dtheta=0.0))
    assert command.duration == pytest.approx(1.5, abs=1e-6)
    assert command.linear_x == pytest.approx(1.0, abs=1e-6)
    assert command.linear_y == pytest.approx(0.0, abs=1e-6)
    assert command.angular_z == pytest.approx(0.0, abs=1e-6)

    zero_command = adapter.to_command(WaypointAction(dx=0.0, dy=0.0, dtheta=0.0))
    assert zero_command.duration == pytest.approx(0.0)
    assert zero_command.linear_x == pytest.approx(0.0)
