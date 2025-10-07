"""Minimal MobilityVLA demo using mock data."""

from __future__ import annotations

import sys
from pathlib import Path

# Ensure the repository root is in PYTHONPATH when executed via `python scripts/demo.py`.
sys.path.append(str(Path(__file__).resolve().parents[1]))

from mobility_vla import Instruction, MobilityVLA, Observation, Pose, TourFrame, DemonstrationTour


def build_mock_tour() -> DemonstrationTour:
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


def main() -> None:
    tour = build_mock_tour()
    mobility_vla = MobilityVLA(tour)

    instruction = Instruction(text="Where should I return this package?")
    goal_frame = mobility_vla.set_instruction(instruction)
    print(f"High-level goal frame: {goal_frame}")

    observations = [
        Observation(hint="entrance lobby"),
        Observation(hint="hallway"),
        Observation(hint="mail room"),
    ]

    for step_idx, observation in enumerate(observations, start=1):
        result = mobility_vla.step(observation)
        action = result.action
        print(
            f"Step {step_idx}: path={result.path} "
            f"action=(dx={action.dx:.2f}, dy={action.dy:.2f}, dtheta={action.dtheta:.2f}) "
            f"(localized to {result.localization.frame_id})"
        )


if __name__ == "__main__":
    main()
