"""Command-line utility to run Structure-from-Motion with PyCOLMAP.

Given a directory of images (extracted from a tour video), the script performs:
1. Image import into a COLMAP database.
2. SIFT feature extraction.
3. Exhaustive feature matching.
4. Incremental SfM reconstruction.
5. Export of camera poses (quaternions + translation) to ``poses.txt``.

Example:
    python scripts/run_pycolmap_sfm.py \\
        --image-dir data/tour_frames \\
        --output-dir data/tour_reconstruction
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Dict, Optional, Sequence, Tuple, Union

import pycolmap
import shutil


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--image-dir",
        required=True,
        type=Path,
        help="Directory containing RGB tour frames (JPG/PNG).",
    )
    parser.add_argument(
        "--output-dir",
        required=True,
        type=Path,
        help="Directory to store COLMAP database, model, and pose outputs.",
    )
    parser.add_argument(
        "--camera-model",
        default="SIMPLE_RADIAL",
        help="COLMAP camera model to use during import (default: SIMPLE_RADIAL).",
    )
    parser.add_argument(
        "--max-image-size",
        type=int,
        default=1600,
        help="Max image size for SIFT extraction; larger images are downscaled.",
    )
    parser.add_argument(
        "--use-gpu",
        action="store_true",
        help="Use GPU for SIFT if available.",
    )
    parser.add_argument(
        "--overwrite",
        action="store_true",
        help="Overwrite existing reconstruction output directory.",
    )
    return parser.parse_args()


def _reset_directory(path: Path, overwrite: bool) -> None:
    if path.exists():
        if not overwrite:
            raise FileExistsError(
                f"Output directory '{path}' already exists. Use --overwrite to replace it."
            )
        # Only remove files that we control; avoid deleting arbitrary user data.
        for entry in path.iterdir():
            if entry.is_dir():
                shutil.rmtree(entry)
            else:
                entry.unlink()
    path.mkdir(parents=True, exist_ok=True)


def _sift_device(use_gpu: bool) -> pycolmap.Device:
    return pycolmap.Device.cuda if use_gpu else pycolmap.Device.cpu


def _update_camera_params(
    database_path: Path,
    params: Sequence[float],
) -> None:
    database = pycolmap.Database(str(database_path))
    try:
        cameras = database.read_cameras()
        if not cameras:
            raise RuntimeError(
                f"No cameras were imported into database '{database_path}'."
            )
        for camera_id, camera in cameras.items():
            camera.params = list(params)
            database.update_camera(camera)
        database.commit()
    finally:
        database.close()


def _run_pipeline(
    image_dir: Path,
    output_dir: Path,
    camera_model: str,
    max_image_size: int,
    use_gpu: bool,
    overwrite: bool,
    *,
    initial_camera_params: Optional[Sequence[float]] = None,
    refine_intrinsics: bool = True,
    return_reconstruction: bool = False,
) -> Tuple[Dict[str, object], Optional[pycolmap.Reconstruction]]:
    if not image_dir.is_dir():
        raise FileNotFoundError(f"Image directory '{image_dir}' does not exist.")

    _reset_directory(output_dir, overwrite=overwrite)

    database_path = output_dir / "database.db"
    model_root = output_dir / "model"
    model_root.mkdir(parents=True, exist_ok=True)

    if database_path.exists():
        database_path.unlink()
    pycolmap.Database(str(database_path)).close()

    reader_opts = pycolmap.ImageReaderOptions()
    reader_opts.camera_model = camera_model

    sift_opts = pycolmap.SiftExtractionOptions()
    sift_opts.max_image_size = max_image_size
    sift_opts.use_gpu = use_gpu
    sift_opts.num_threads = 4

    pycolmap.import_images(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        options=reader_opts,
    )

    if initial_camera_params is not None:
        _update_camera_params(database_path, initial_camera_params)

    pycolmap.extract_features(
        database_path=str(database_path),
        image_path=str(image_dir),
        camera_mode=pycolmap.CameraMode.SINGLE,
        camera_model=camera_model,
        reader_options=reader_opts,
        sift_options=sift_opts,
        device=_sift_device(use_gpu),
    )

    match_opts = pycolmap.SiftMatchingOptions()
    match_opts.use_gpu = use_gpu

    pycolmap.match_exhaustive(
        database_path=str(database_path),
        sift_options=match_opts,
        device=_sift_device(use_gpu),
    )

    pipeline_opts = pycolmap.IncrementalPipelineOptions()
    mapper_opts = pipeline_opts.mapper
    mapper_opts.init_min_num_inliers = 30
    mapper_opts.init_min_tri_angle = 3.0
    mapper_opts.abs_pose_min_num_inliers = 15
    mapper_opts.abs_pose_min_inlier_ratio = 0.05
    mapper_opts.filter_min_tri_angle = 0.5
    pipeline_opts.min_model_size = 5
    pipeline_opts.init_num_trials = 500
    mapper_opts.ba_refine_focal_length = refine_intrinsics
    mapper_opts.ba_refine_principal_point = refine_intrinsics
    mapper_opts.ba_refine_extra_params = refine_intrinsics

    reconstructions = pycolmap.incremental_mapping(
        database_path=str(database_path),
        image_path=str(image_dir),
        output_path=str(model_root),
        options=pipeline_opts,
    )

    if not reconstructions:
        raise RuntimeError("PyCOLMAP did not produce a reconstruction.")

    reconstruction = max(
        reconstructions.values(),
        key=lambda rec: len(rec.images),
    )

    poses_path = output_dir / "poses.txt"
    with poses_path.open("w", encoding="utf-8") as f:
        for image_id, image in sorted(reconstruction.images.items()):
            pose = image.cam_from_world()
            quat = pose.rotation.quat
            tvec = pose.translation
            quat_str = " ".join(f"{v:.6f}" for v in quat)
            tvec_str = " ".join(f"{v:.6f}" for v in tvec)
            f.write(f"{image.name} q=[{quat_str}] t=[{tvec_str}]\n")

    summary: Dict[str, object] = {
        "cameras": len(reconstruction.cameras),
        "images": len(reconstruction.images),
        "points3D": len(reconstruction.points3D),
        "poses_file": str(poses_path),
        "model_dir": str(model_root),
    }

    with (output_dir / "summary.json").open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)

    return (summary, reconstruction if return_reconstruction else None)


def run_pycolmap_pipeline(
    image_dir: Path,
    output_dir: Path,
    *,
    camera_model: str = "SIMPLE_RADIAL",
    max_image_size: int = 1600,
    use_gpu: bool = False,
    overwrite: bool = False,
    initial_camera_params: Optional[Sequence[float]] = None,
    refine_intrinsics: bool = True,
    return_reconstruction: bool = False,
) -> Union[Dict[str, object], Tuple[Dict[str, object], pycolmap.Reconstruction]]:
    """Public helper that wraps the internal COLMAP pipeline.

    This is intended for programmatic use from other scripts so they do not
    have to invoke the CLI entry point defined in ``main``.
    """
    summary, reconstruction = _run_pipeline(
        image_dir=image_dir,
        output_dir=output_dir,
        camera_model=camera_model,
        max_image_size=max_image_size,
        use_gpu=use_gpu,
        overwrite=overwrite,
        initial_camera_params=initial_camera_params,
        refine_intrinsics=refine_intrinsics,
        return_reconstruction=return_reconstruction,
    )
    if return_reconstruction:
        return summary, reconstruction
    return summary


def main() -> None:
    args = _parse_args()
    summary, _ = _run_pipeline(
        image_dir=args.image_dir,
        output_dir=args.output_dir,
        camera_model=args.camera_model,
        max_image_size=args.max_image_size,
        use_gpu=args.use_gpu,
        overwrite=args.overwrite,
        initial_camera_params=None,
        refine_intrinsics=True,
        return_reconstruction=False,
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
