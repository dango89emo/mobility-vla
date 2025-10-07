from __future__ import annotations

from pathlib import Path

import numpy as np

from mobility_vla.colmap_io import load_tour_from_colmap


def _pose_line(image: str, quat: tuple[float, float, float, float], trans: tuple[float, float, float]) -> str:
    q_str = " ".join(f"{v:.6f}" for v in quat)
    t_str = " ".join(f"{v:.6f}" for v in trans)
    return f"{image} q=[{q_str}] t=[{t_str}]\n"


def test_load_tour_from_colmap(tmp_path: Path) -> None:
    poses_path = tmp_path / "poses.txt"
    images_dir = tmp_path / "images"
    images_dir.mkdir()

    lines = [
        _pose_line("frame0.jpg", (1.0, 0.0, 0.0, 0.0), (0.0, 0.0, 0.0)),
        _pose_line("frame1.jpg", (1.0, 0.0, 0.0, 0.0), (-1.0, 0.0, 0.0)),
    ]
    poses_path.write_text("".join(lines), encoding="utf-8")

    tour = load_tour_from_colmap(poses_path, images_dir)
    assert len(tour.frames) == 2

    frame0, frame1 = tour.frames
    assert frame0.frame_id == "frame0"
    assert frame0.pose.x == 0.0
    assert frame0.pose.y == 0.0
    assert frame0.pose.z == 0.0
    assert frame0.pose.yaw == 0.0
    assert frame0.metadata["keywords"] == "frame0"

    assert frame1.frame_id == "frame1"
    np.testing.assert_allclose(frame1.pose.x, 1.0, atol=1e-6)
    np.testing.assert_allclose(frame1.pose.y, 0.0, atol=1e-6)
    np.testing.assert_allclose(frame1.pose.z, 0.0, atol=1e-6)
    assert frame1.pose.yaw == 0.0
